// radon.zig — DICOM -> (recon, sinogram) tensors, computed on CPU, written as .npy
//
// Replaces the skimage half of data_pipeline.py so the cache can be rebuilt on the
// Spark without waking the 4090 box. Pipeline per slice:
//
//     DICOM -> HU -> clamp to air -> +offset (air=0) -> resize -> radon(NANG angles)
//
// NOT data_pipeline.normalize_hu(). That did clip(round(hu+1024), 0, 4095) before
// the transform; the CLIP was the defect. It is nonlinear, so it does not commute
// with Radon, and it flattened every scanner's FOV padding sentinel into a single
// 0 — fabricating a hard edge that was never in the data. The +1024 offset it also
// carried was sound, and is kept here (as +1000) for a physical reason: Radon
// integrates ATTENUATION, and air has mu = 0. See toHU().
//
// The rim outside the reconstruction circle is left alone. skimage's radon() only
// *warns* about it, it never zeroes it (data_pipeline.py's comment claims
// otherwise and is wrong), so leaving it is the faithful behavior.
//
// Geometry convention matches twin_common.py: output sinogram axis0 = DETECTOR
// position, axis1 = projection ANGLE.
//
// Validated against skimage (raw HU, floor/offset disabled): Pearson r = 1.000000
// on both recon and sinogram, residuals at float32 precision. See validate_zig.py.
//
// Usage:
//   radon --list FILE --outdir DIR [--threads N] [--size 256] [--angles 256]
//         [--hu-floor -1000] [--hu-offset 1000]
//   FILE is newline-separated DICOM paths. Outputs <outdir>/{recon,sinogram}/<id>.npy

const std = @import("std");

const DEF_SIZE: usize = 256;
const DEF_NANG: usize = 256;

// ---------------------------------------------------------------- DICOM parse

const Dicom = struct {
    rows: usize = 0,
    cols: usize = 0,
    bits_alloc: u16 = 16,
    pix_rep: u16 = 0,
    slope: f64 = 1.0,
    intercept: f64 = 0.0,
    px_off: usize = 0,
    px_len: usize = 0,
};

fn isLongVR(vr: [2]u8) bool {
    const two = [_][2]u8{
        .{ 'O', 'B' }, .{ 'O', 'W' }, .{ 'O', 'F' },
        .{ 'S', 'Q' }, .{ 'U', 'T' }, .{ 'U', 'N' },
    };
    for (two) |t| if (vr[0] == t[0] and vr[1] == t[1]) return true;
    return false;
}

fn rdU16(b: []const u8, p: usize) u16 {
    return std.mem.readInt(u16, b[p..][0..2], .little);
}
fn rdU32(b: []const u8, p: usize) u32 {
    return std.mem.readInt(u32, b[p..][0..4], .little);
}

fn parseDS(s: []const u8) f64 {
    const t = std.mem.trim(u8, s, &[_]u8{ 0, ' ', '\t', '\n', '\r' });
    return std.fmt.parseFloat(f64, t) catch 0.0;
}

fn parseDicom(buf: []const u8) !Dicom {
    var d = Dicom{};
    var p: usize = 0;
    var explicit = true;

    // File meta group (0002,*) is ALWAYS Explicit VR LE, regardless of the
    // transfer syntax it announces for the dataset that follows.
    if (buf.len > 132 and std.mem.eql(u8, buf[128..132], "DICM")) {
        p = 132;
        while (p + 8 <= buf.len) {
            const save = p;
            const g = rdU16(buf, p);
            const e = rdU16(buf, p + 2);
            if (g != 0x0002) {
                p = save;
                break;
            }
            p += 4;
            const vr = [2]u8{ buf[p], buf[p + 1] };
            p += 2;
            var len: usize = undefined;
            if (isLongVR(vr)) {
                p += 2;
                len = rdU32(buf, p);
                p += 4;
            } else {
                len = rdU16(buf, p);
                p += 2;
            }
            if (p + len > buf.len) return error.Truncated;
            if (e == 0x0010) { // TransferSyntaxUID
                const ts = std.mem.trim(u8, buf[p .. p + len], &[_]u8{ 0, ' ' });
                explicit = !std.mem.eql(u8, ts, "1.2.840.10008.1.2");
            }
            p += len;
        }
    } else {
        // No preamble (pydicom force=True territory): assume implicit VR LE.
        p = 0;
        explicit = false;
    }

    // Dataset. Scan sequentially; PixelData terminates.
    while (p + 8 <= buf.len) {
        const g = rdU16(buf, p);
        const e = rdU16(buf, p + 2);
        p += 4;
        var len: usize = undefined;
        if (explicit) {
            if (p + 2 > buf.len) break;
            const vr = [2]u8{ buf[p], buf[p + 1] };
            p += 2;
            if (isLongVR(vr)) {
                if (p + 6 > buf.len) break;
                p += 2;
                len = rdU32(buf, p);
                p += 4;
            } else {
                if (p + 2 > buf.len) break;
                len = rdU16(buf, p);
                p += 2;
            }
        } else {
            if (p + 4 > buf.len) break;
            len = rdU32(buf, p);
            p += 4;
        }

        if (g == 0x7FE0 and e == 0x0010) {
            d.px_off = p;
            d.px_len = if (len == 0xFFFFFFFF) buf.len - p else len;
            break;
        }
        if (len == 0xFFFFFFFF) return error.UndefinedLength; // SQ; not expected pre-PixelData
        if (p + len > buf.len) break;

        switch (g) {
            0x0028 => switch (e) {
                0x0010 => d.rows = rdU16(buf, p),
                0x0011 => d.cols = rdU16(buf, p),
                0x0100 => d.bits_alloc = rdU16(buf, p),
                0x0103 => d.pix_rep = rdU16(buf, p),
                0x1052 => d.intercept = parseDS(buf[p .. p + len]),
                0x1053 => d.slope = parseDS(buf[p .. p + len]),
                else => {},
            },
            else => {},
        }
        p += len;
    }

    if (d.rows == 0 or d.cols == 0 or d.px_len == 0) return error.MissingPixelData;
    if (d.bits_alloc != 16) return error.UnsupportedBits;
    if (d.px_len < d.rows * d.cols * 2) return error.ShortPixelData;
    return d;
}

/// Raw stored values -> HU. Matches pydicom: dtype follows PixelRepresentation
/// (u16 vs i16) using BitsAllocated; BitsStored is NOT masked/sign-extended.
///
/// The rescale is applied unconditionally, and that is correct for this dataset.
/// It was worth checking: the archive does contain slices whose stored pixels are
/// ALREADY Hounsfield, where applying a -1024 intercept again would land brain at
/// ~-996 (air density). But every such slice carries intercept=0, so the affine
/// map is a no-op on exactly the files that must not be shifted. Verified against
/// the air (~-1000) and brain (~+30) anchors over a 2500-slice random sample:
/// zero files were double-applied.
///
/// `floor` removes FOV padding sentinels. Scanners mark "outside the reconstructed
/// field" with raw -1024/-2000/-3024, which survive the intercept as HU -2048 /
/// -3024 / -4048. Those are non-physical (below air) and are a fiction of the
/// recon, not attenuation — yet Radon integrates straight through them, injecting
/// a large scanner-dependent negative mass into every ray that clips the rim.
/// Clamping to air makes "outside the patient" mean the same thing on every
/// scanner. Pass a very negative floor to disable.
///
/// `offset` is added AFTER the clamp, to put air at 0. Radon integrates
/// ATTENUATION, and air has mu = 0 — but air is -1000 HU, so projecting raw HU
/// makes every ray carry a large negative term proportional to its path length
/// through the field. That path-length pattern, not the anatomy, then dominates
/// the sinogram's dynamic range (a full air ray reached -256000 against a signal
/// max of ~45k). With floor=-1000 and offset=+1000, air maps to exactly 0, water
/// to 1000, brain to ~1030, and every line integral is a non-negative sum of
/// attenuations — the physical quantity.
///
/// This is NOT data_pipeline.normalize_hu(). That did clip(round(hu+1024), 0,
/// 4095): the same offset, but the CLIP was the defect — nonlinear, and it
/// flattened every scanner's padding sentinel into a fabricated hard edge. A pure
/// offset is linear and commutes with Radon; a clip does not.
fn toHU(d: Dicom, buf: []const u8, out: []f32, floor: f32, offset: f32) void {
    const n = d.rows * d.cols;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const raw = rdU16(buf, d.px_off + i * 2);
        const v: f64 = if (d.pix_rep == 1)
            @floatFromInt(@as(i16, @bitCast(raw)))
        else
            @floatFromInt(raw);
        const hu: f32 = @floatCast(v * d.slope + d.intercept);
        out[i] = (if (hu < floor) floor else hu) + offset;
    }
}

// ---------------------------------------------------------------- resize

fn mirrorIdx(i: isize, n: isize) usize {
    if (n == 1) return 0;
    const period = 2 * (n - 1);
    var x = @mod(i, period);
    if (x < 0) x += period;
    if (x >= n) x = period - x;
    return @intCast(x);
}

/// Separable Gaussian, 'mirror' boundary — matches skimage.resize(anti_aliasing=True),
/// which maps its default mode='reflect' onto ndimage's 'mirror'.
fn gaussian(src: []const f32, w: usize, h: usize, sigma: f64, tmp: []f32, dst: []f32) void {
    if (sigma <= 0) {
        @memcpy(dst, src);
        return;
    }
    const radius: usize = @intFromFloat(@floor(4.0 * sigma + 0.5)); // scipy truncate=4.0
    const klen = 2 * radius + 1;
    var kern: [65]f64 = undefined;
    var sum: f64 = 0;
    var k: usize = 0;
    while (k < klen) : (k += 1) {
        const x: f64 = @floatFromInt(@as(isize, @intCast(k)) - @as(isize, @intCast(radius)));
        kern[k] = @exp(-(x * x) / (2.0 * sigma * sigma));
        sum += kern[k];
    }
    k = 0;
    while (k < klen) : (k += 1) kern[k] /= sum;

    // horizontal
    var y: usize = 0;
    while (y < h) : (y += 1) {
        var x: usize = 0;
        while (x < w) : (x += 1) {
            var acc: f64 = 0;
            var j: usize = 0;
            while (j < klen) : (j += 1) {
                const sx = mirrorIdx(@as(isize, @intCast(x)) + @as(isize, @intCast(j)) - @as(isize, @intCast(radius)), @intCast(w));
                acc += kern[j] * src[y * w + sx];
            }
            tmp[y * w + x] = @floatCast(acc);
        }
    }
    // vertical
    y = 0;
    while (y < h) : (y += 1) {
        var x: usize = 0;
        while (x < w) : (x += 1) {
            var acc: f64 = 0;
            var j: usize = 0;
            while (j < klen) : (j += 1) {
                const sy = mirrorIdx(@as(isize, @intCast(y)) + @as(isize, @intCast(j)) - @as(isize, @intCast(radius)), @intCast(h));
                acc += kern[j] * tmp[sy * w + x];
            }
            dst[y * w + x] = @floatCast(acc);
        }
    }
}

fn bilinear(img: []const f32, w: usize, h: usize, x: f64, y: f64) f64 {
    const x0f = @floor(x);
    const y0f = @floor(y);
    const x0: isize = @intFromFloat(x0f);
    const y0: isize = @intFromFloat(y0f);
    const wx = x - x0f;
    const wy = y - y0f;
    var acc: f64 = 0;
    var dy: isize = 0;
    while (dy < 2) : (dy += 1) {
        var dx: isize = 0;
        while (dx < 2) : (dx += 1) {
            const xi = x0 + dx;
            const yi = y0 + dy;
            // mode='constant', cval=0 (skimage warp default in radon)
            if (xi < 0 or yi < 0 or xi >= @as(isize, @intCast(w)) or yi >= @as(isize, @intCast(h))) continue;
            const wgt = (if (dx == 0) 1.0 - wx else wx) * (if (dy == 0) 1.0 - wy else wy);
            acc += wgt * img[@as(usize, @intCast(yi)) * w + @as(usize, @intCast(xi))];
        }
    }
    return acc;
}

/// skimage.resize coordinate convention: in = (out + 0.5) * scale - 0.5
fn resizeTo(src: []const f32, sw: usize, sh: usize, dst: []f32, dw: usize, dh: usize, scratch: []f32, blurred: []f32) void {
    const fx = @as(f64, @floatFromInt(sw)) / @as(f64, @floatFromInt(dw));
    const fy = @as(f64, @floatFromInt(sh)) / @as(f64, @floatFromInt(dh));
    const sigma = @max(0.0, (@max(fx, fy) - 1.0) / 2.0);

    const base = if (sigma > 0) blk: {
        gaussian(src, sw, sh, sigma, scratch, blurred);
        break :blk blurred;
    } else src;

    var y: usize = 0;
    while (y < dh) : (y += 1) {
        const sy = (@as(f64, @floatFromInt(y)) + 0.5) * fy - 0.5;
        var x: usize = 0;
        while (x < dw) : (x += 1) {
            const sx = (@as(f64, @floatFromInt(x)) + 0.5) * fx - 0.5;
            dst[y * dw + x] = @floatCast(bilinear(base, sw, sh, sx, sy));
        }
    }
}

// ---------------------------------------------------------------- radon

/// Forward Radon, circle=True, theta = linspace(0,180,nang,endpoint=False).
/// Mirrors skimage: for each angle, rotate about center=(n//2) with bilinear
/// interpolation and zero fill, then sum down axis 0.
///   out[j, i] = sum_y image[ rot(j, y; theta_i) ]
/// j = detector position (axis0), i = angle (axis1).
fn radon(img: []const f32, n: usize, nang: usize, out: []f32) void {
    const c: f64 = @floatFromInt(n / 2);
    var i: usize = 0;
    while (i < nang) : (i += 1) {
        const deg = 180.0 * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(nang));
        const t = deg * std.math.pi / 180.0;
        const ct = @cos(t);
        const st = @sin(t);
        var j: usize = 0;
        while (j < n) : (j += 1) {
            const dx = @as(f64, @floatFromInt(j)) - c;
            var acc: f64 = 0;
            var y: usize = 0;
            while (y < n) : (y += 1) {
                const dy = @as(f64, @floatFromInt(y)) - c;
                const xi = c + ct * dx + st * dy;
                const yi = c - st * dx + ct * dy;
                acc += bilinear(img, n, n, xi, yi);
            }
            out[j * nang + i] = @floatCast(acc);
        }
    }
}

// ---------------------------------------------------------------- npy

fn writeNpy(path: []const u8, data: []const f32, rows: usize, cols: usize) !void {
    var hdr_buf: [128]u8 = undefined;
    const dict = try std.fmt.bufPrint(&hdr_buf, "{{'descr': '<f4', 'fortran_order': False, 'shape': ({d}, {d}), }}", .{ rows, cols });
    // total header (10 + len) padded to a multiple of 64, terminated by '\n'
    var total: usize = 10 + dict.len + 1;
    const pad = (64 - (total % 64)) % 64;
    total += pad;
    const dict_len = total - 10;

    var out = std.ArrayList(u8).init(std.heap.page_allocator);
    defer out.deinit();
    try out.appendSlice(&[_]u8{ 0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0 });
    try out.append(@intCast(dict_len & 0xFF));
    try out.append(@intCast((dict_len >> 8) & 0xFF));
    try out.appendSlice(dict);
    var i: usize = 0;
    while (i < pad) : (i += 1) try out.append(' ');
    try out.append('\n');
    try out.appendSlice(std.mem.sliceAsBytes(data));

    const f = try std.fs.cwd().createFile(path, .{ .truncate = true });
    defer f.close();
    try f.writeAll(out.items);
}

// ---------------------------------------------------------------- driver

const Job = struct {
    paths: [][]const u8,
    recon_dir: []const u8,
    sino_dir: []const u8,
    size: usize,
    nang: usize,
    hu_floor: f32,
    hu_offset: f32,
    next: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
    ok: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
    fail: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
};

fn stemOf(path: []const u8) []const u8 {
    const base = std.fs.path.basename(path);
    if (std.mem.lastIndexOfScalar(u8, base, '.')) |d| return base[0..d];
    return base;
}

fn worker(job: *Job) void {
    const alloc = std.heap.page_allocator;
    const n = job.size;
    const small = alloc.alloc(f32, n * n) catch return;
    defer alloc.free(small);
    const sino = alloc.alloc(f32, n * job.nang) catch return;
    defer alloc.free(sino);

    while (true) {
        const idx = job.next.fetchAdd(1, .monotonic);
        if (idx >= job.paths.len) break;
        const path = job.paths[idx];

        const buf = std.fs.cwd().readFileAlloc(alloc, path, 64 << 20) catch {
            _ = job.fail.fetchAdd(1, .monotonic);
            continue;
        };
        defer alloc.free(buf);

        const d = parseDicom(buf) catch {
            _ = job.fail.fetchAdd(1, .monotonic);
            continue;
        };

        const hu = alloc.alloc(f32, d.rows * d.cols) catch {
            _ = job.fail.fetchAdd(1, .monotonic);
            continue;
        };
        defer alloc.free(hu);
        const scratch = alloc.alloc(f32, d.rows * d.cols) catch {
            _ = job.fail.fetchAdd(1, .monotonic);
            continue;
        };
        defer alloc.free(scratch);
        const blurred = alloc.alloc(f32, d.rows * d.cols) catch {
            _ = job.fail.fetchAdd(1, .monotonic);
            continue;
        };
        defer alloc.free(blurred);

        toHU(d, buf, hu, job.hu_floor, job.hu_offset);
        resizeTo(hu, d.cols, d.rows, small, n, n, scratch, blurred);
        radon(small, n, job.nang, sino);

        const stem = stemOf(path);
        var pb: [512]u8 = undefined;
        const rp = std.fmt.bufPrint(&pb, "{s}/{s}.npy", .{ job.recon_dir, stem }) catch {
            _ = job.fail.fetchAdd(1, .monotonic);
            continue;
        };
        writeNpy(rp, small, n, n) catch {
            _ = job.fail.fetchAdd(1, .monotonic);
            continue;
        };
        var pb2: [512]u8 = undefined;
        const sp = std.fmt.bufPrint(&pb2, "{s}/{s}.npy", .{ job.sino_dir, stem }) catch {
            _ = job.fail.fetchAdd(1, .monotonic);
            continue;
        };
        writeNpy(sp, sino, n, job.nang) catch {
            _ = job.fail.fetchAdd(1, .monotonic);
            continue;
        };

        const done = job.ok.fetchAdd(1, .monotonic) + 1;
        if (done % 500 == 0) {
            std.debug.print("[radon] {d}/{d} ok={d} fail={d}\n", .{ done + job.fail.load(.monotonic), job.paths.len, done, job.fail.load(.monotonic) });
        }
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = gpa.allocator();

    var list_path: ?[]const u8 = null;
    var outdir: []const u8 = "out";
    var threads: usize = 0;
    var size: usize = DEF_SIZE;
    var nang: usize = DEF_NANG;
    var hu_floor: f32 = -1000.0; // air; disable with --hu-floor -1e9
    var hu_offset: f32 = 1000.0; // shift air to 0 so Radon sums attenuation, not HU

    const argv = try std.process.argsAlloc(alloc);
    defer std.process.argsFree(alloc, argv);
    var i: usize = 1;
    while (i < argv.len) : (i += 1) {
        const a = argv[i];
        if (std.mem.eql(u8, a, "--list") and i + 1 < argv.len) {
            i += 1;
            list_path = argv[i];
        } else if (std.mem.eql(u8, a, "--outdir") and i + 1 < argv.len) {
            i += 1;
            outdir = argv[i];
        } else if (std.mem.eql(u8, a, "--threads") and i + 1 < argv.len) {
            i += 1;
            threads = try std.fmt.parseInt(usize, argv[i], 10);
        } else if (std.mem.eql(u8, a, "--size") and i + 1 < argv.len) {
            i += 1;
            size = try std.fmt.parseInt(usize, argv[i], 10);
        } else if (std.mem.eql(u8, a, "--angles") and i + 1 < argv.len) {
            i += 1;
            nang = try std.fmt.parseInt(usize, argv[i], 10);
        } else if (std.mem.eql(u8, a, "--hu-floor") and i + 1 < argv.len) {
            i += 1;
            hu_floor = try std.fmt.parseFloat(f32, argv[i]);
        } else if (std.mem.eql(u8, a, "--hu-offset") and i + 1 < argv.len) {
            i += 1;
            hu_offset = try std.fmt.parseFloat(f32, argv[i]);
        }
    }
    if (list_path == null) {
        std.debug.print("usage: radon --list FILE --outdir DIR [--threads N] [--size 256] [--angles 256]\n", .{});
        return error.MissingArgs;
    }
    if (threads == 0) threads = std.Thread.getCpuCount() catch 4;

    const text = try std.fs.cwd().readFileAlloc(alloc, list_path.?, 1 << 30);
    defer alloc.free(text);
    var paths = std.ArrayList([]const u8).init(alloc);
    defer paths.deinit();
    var it = std.mem.tokenizeAny(u8, text, "\r\n");
    while (it.next()) |line| {
        const t = std.mem.trim(u8, line, " \t");
        if (t.len > 0) try paths.append(t);
    }

    var rb: [512]u8 = undefined;
    var sb: [512]u8 = undefined;
    const recon_dir = try std.fmt.bufPrint(&rb, "{s}/recon", .{outdir});
    const sino_dir = try std.fmt.bufPrint(&sb, "{s}/sinogram", .{outdir});
    try std.fs.cwd().makePath(recon_dir);
    try std.fs.cwd().makePath(sino_dir);

    std.debug.print("[radon] {d} slices -> {s}/{{recon,sinogram}}  size={d} angles={d} threads={d} hu_floor={d:.0} hu_offset={d:.0}\n", .{ paths.items.len, outdir, size, nang, threads, hu_floor, hu_offset });

    var job = Job{
        .paths = paths.items,
        .recon_dir = recon_dir,
        .sino_dir = sino_dir,
        .size = size,
        .nang = nang,
        .hu_floor = hu_floor,
        .hu_offset = hu_offset,
    };

    const t0 = std.time.milliTimestamp();
    const handles = try alloc.alloc(std.Thread, threads);
    defer alloc.free(handles);
    for (handles) |*h| h.* = try std.Thread.spawn(.{}, worker, .{&job});
    for (handles) |h| h.join();
    const dt = @as(f64, @floatFromInt(std.time.milliTimestamp() - t0)) / 1000.0;

    std.debug.print("[radon] DONE ok={d} fail={d} in {d:.1}s ({d:.1}/s)\n", .{
        job.ok.load(.monotonic), job.fail.load(.monotonic), dt,
        @as(f64, @floatFromInt(job.ok.load(.monotonic))) / dt,
    });
}
