#!/usr/bin/env python3
"""
validate_zig.py — check radon.zig against the skimage reference it replaces.

The Zig pipeline is NOT a port of data_pipeline.py's exact steps: normalize_hu()
(the +1024 offset and clip to 0..4095) is deliberately dropped, since Radon is
linear and the clip was nonlinearly flattening scanner FOV padding. So the
reference here is the intended math — raw HU -> resize -> radon — not the old
cache, which no longer exists to be compatible with anyway.

Usage: validate_zig.py --list FILE --zig-out DIR
"""
import argparse
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")

SIZE = 256
NANG = 256


def ref_pipeline(path):
    import pydicom
    from skimage.transform import radon, resize
    d = pydicom.dcmread(path, force=True)
    a = d.pixel_array.astype(np.float32)
    slope = float(getattr(d, "RescaleSlope", 1.0))
    inter = float(getattr(d, "RescaleIntercept", 0.0))
    hu = a * slope + inter
    img = resize(hu, (SIZE, SIZE), anti_aliasing=True, preserve_range=True)
    theta = np.linspace(0.0, 180.0, NANG, endpoint=False)
    sino = radon(img, theta=theta, circle=True)
    return img.astype(np.float64), sino.astype(np.float64)


def rel(a, b):
    """Scale-relative max error, robust to the tensor's dynamic range."""
    denom = max(np.abs(b).max(), 1e-6)
    return float(np.abs(a - b).max() / denom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--zig-out", required=True)
    a = ap.parse_args()

    paths = [l.strip() for l in open(a.list) if l.strip()]
    print(f"{'id':<16} {'recon_r':>9} {'recon_rel':>10} {'sino_r':>9} {'sino_rel':>10}")
    rr, sr = [], []
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        zr = np.load(os.path.join(a.zig_out, "recon", stem + ".npy")).astype(np.float64)
        zs = np.load(os.path.join(a.zig_out, "sinogram", stem + ".npy")).astype(np.float64)
        pr, ps = ref_pipeline(p)
        r_recon = np.corrcoef(zr.ravel(), pr.ravel())[0, 1]
        r_sino = np.corrcoef(zs.ravel(), ps.ravel())[0, 1]
        rr.append(r_recon); sr.append(r_sino)
        print(f"{stem:<16} {r_recon:>9.6f} {rel(zr,pr):>10.2e} {r_sino:>9.6f} {rel(zs,ps):>10.2e}")
    print(f"\nrecon  pearson r: min={min(rr):.6f} mean={np.mean(rr):.6f}")
    print(f"sino   pearson r: min={min(sr):.6f} mean={np.mean(sr):.6f}")


if __name__ == "__main__":
    main()
