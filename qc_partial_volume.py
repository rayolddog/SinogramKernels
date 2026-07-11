#!/usr/bin/env python3
"""
qc_partial_volume.py — quantify PARTIAL-VOLUME PSEUDO-ICH in the 50k recon cache.

Clinical premise (subfrontal/subtemporal SDH miss): where the skull is TANGENTIAL
to the axial plane (vertex, orbital roofs, floors of the anterior/middle fossae),
a voxel averages bone (~1000-1800 HU) with brain/CSF (~0-40 HU) through the slice
thickness, landing in the acute-ICH window (48-175 HU). This produces false ICH
signal that hugs the inner table exactly where thin subdurals hide.

Method — use NORMAL (any=0) slices as a label-free control:
  * any 48-175 HU voxel in a no-hemorrhage slice is artifact by construction.
  * split ICH-window voxels into PERIPHERAL (within a few px of bone) vs DEEP.
  * peripheral burden in normals == partial-volume pseudo-ICH rate.
  * bin normals by bone fraction (proxy for tangential-bone vertex/base/orbit
    slices); pseudo-ICH should RISE with bone content -> the hypothesis, measured.
  * compare against subdural-positive slices to show how co-located (and thus how
    confusable) the true target is with the artifact.

Recon tensors are uint16 LINEAR HU: stored value = HU + 1024  (recover HU = v-1024).
"""
import csv
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.ndimage import (binary_closing, binary_fill_holes, label,
                           distance_transform_edt)
from skimage.segmentation import clear_border

CACHE = "/mnt/mlcache/subset50k"
RECON = os.path.join(CACHE, "recon")
LABELS = os.path.join(CACHE, "labels.csv")
HU_OFFSET = 1024

ICH_LO, ICH_HI = 48, 175       # acute-ICH HU window
BONE_HU = 300                  # HU >= this = bone (cortical/cancellous)
TISSUE_HU = -400               # HU > this = head tissue (excludes air ~ -1000)

# distance-from-inner-table bins (pixels; ~0.9 mm/px at 256 from a ~230 mm FOV)
DIST_EDGES = np.array([0, 1.5, 3, 5, 8, 13, 40], dtype=float)
NEAR_PX = 2.0                  # "near wall" = within this many px of bone

N_PER_GROUP = 4000             # sampled slices per group
SEED = 0
LABEL_COLS = ["epidural", "intraparenchymal", "intraventricular",
              "subarachnoid", "subdural", "any"]


def load_labels():
    per = {}
    with open(LABELS) as f:
        for row in csv.DictReader(f):
            per[row["id"]] = [int(row[c]) for c in LABEL_COLS]
    return per


def intracranial_mask(hu):
    """Leak-proof intracranial cavity mask, or None if the bone ring is open.

    Close small suture/foramen gaps, fill the ring, drop bone. If the ring is
    incomplete the filled region leaks to the exterior and TOUCHES THE BORDER —
    clear_border then deletes it, so leaked (skull-base/orbit) slices self-reject
    instead of silently pulling in scalp/temporalis muscle (~40-60 HU = ICH band)."""
    bone = hu >= BONE_HU
    filled = binary_fill_holes(binary_closing(bone, iterations=2))
    interior = clear_border(filled & ~bone)          # leaked cavity touches border -> removed
    if interior.sum() < 2000:                         # no plausible cavity on this slice
        return None, bone
    lab, n = label(interior)
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    interior = lab == sizes.argmax()
    if not (2000 <= interior.sum() <= 55000):         # implausible -> reject
        return None, bone
    return interior, bone


def slice_metrics(path):
    """Per-slice ICH-window prevalence binned by distance to the inner table.
    Returns picklable arrays; valid=False if no clean intracranial mask."""
    hu = np.load(path).astype(np.float32) - HU_OFFSET
    interior, bone = intracranial_mask(hu)
    stem = os.path.splitext(os.path.basename(path))[0]
    head = hu > TISSUE_HU
    if interior is None:
        return {"id": stem, "valid": False,
                "bone_frac": float(bone.sum()) / max(head.sum(), 1)}

    dist = distance_transform_edt(~bone)                  # px distance to nearest bone
    d_in = dist[interior]
    ich_in = ((hu >= ICH_LO) & (hu <= ICH_HI))[interior]

    binidx = np.digitize(d_in, DIST_EDGES) - 1
    nbin = len(DIST_EDGES) - 1
    tissue_by_d = np.array([(binidx == b).sum() for b in range(nbin)], dtype=np.int64)
    ich_by_d = np.array([((binidx == b) & ich_in).sum() for b in range(nbin)],
                        dtype=np.int64)
    return {
        "id": stem, "valid": True,
        "bone_frac": float(bone.sum()) / max(head.sum(), 1),
        "interior": int(interior.sum()),
        "tissue_by_d": tissue_by_d, "ich_by_d": ich_by_d,
        "near_ich": int((ich_in & (d_in <= NEAR_PX)).sum()),
        "near_tissue": int((d_in <= NEAR_PX).sum()),
        "deep_ich": int((ich_in & (d_in > NEAR_PX)).sum()),
        "deep_tissue": int((d_in > NEAR_PX).sum()),
    }


def run_group(ids):
    paths = [os.path.join(RECON, i + ".npy") for i in ids]
    out = []
    with ProcessPoolExecutor(max_workers=16) as ex:
        for m in ex.map(slice_metrics, paths, chunksize=32):
            out.append(m)
    return out


def summarize(rows, name):
    """Pool over valid slices: prevalence = sum(ICH voxels)/sum(tissue voxels)
    per distance bin (weights by voxel count, not by slice)."""
    valid = [r for r in rows if r.get("valid")]
    tiss = np.sum([r["tissue_by_d"] for r in valid], axis=0)
    ich = np.sum([r["ich_by_d"] for r in valid], axis=0)
    prev = ich / np.maximum(tiss, 1)
    near = sum(r["near_ich"] for r in valid) / max(sum(r["near_tissue"] for r in valid), 1)
    deep = sum(r["deep_ich"] for r in valid) / max(sum(r["deep_tissue"] for r in valid), 1)
    print(f"\n[{name}]")
    print(f"  valid slices (clean intracranial mask): {len(valid)}/{len(rows)}")
    print(f"  ICH-window prevalence  near wall (<={NEAR_PX:.0f}px): {100*near:.2f}%"
          f"   deep (>{NEAR_PX:.0f}px): {100*deep:.2f}%"
          f"   near:deep = {near/max(deep,1e-9):.1f} : 1")
    print("  prevalence by distance-to-bone bin:")
    for b in range(len(DIST_EDGES) - 1):
        print(f"    {DIST_EDGES[b]:4.1f}-{DIST_EDGES[b+1]:4.1f}px : "
              f"{100*prev[b]:6.2f}%   (n_tissue={tiss[b]:,})")
    return dict(valid=valid, prev=prev, tiss=tiss, near=near, deep=deep)


def bone_frac_trend(valid):
    """NORMALS: near-wall pseudo-ICH prevalence vs bone fraction of slice."""
    bf = np.array([r["bone_frac"] for r in valid])
    near_i = np.array([r["near_ich"] for r in valid], float)
    near_t = np.array([r["near_tissue"] for r in valid], float)
    edges = np.quantile(bf, [0, .2, .4, .6, .8, 1.0])
    centers, prev = [], []
    print("\n[NORMAL] near-wall pseudo-ICH vs bone fraction of slice "
          "(thin/tangential bone -> low frac; thick/dense skull-base -> high frac):")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (bf >= lo) & (bf <= hi)
        if m.sum() == 0:
            continue
        p = near_i[m].sum() / max(near_t[m].sum(), 1)
        centers.append(bf[m].mean()); prev.append(p)
        print(f"  bone {100*lo:4.1f}-{100*hi:4.1f}% (n={m.sum():4d}): "
              f"near-wall pseudo-ICH {100*p:.2f}%")
    return np.array(centers), np.array(prev)


def main():
    rng = np.random.default_rng(SEED)
    labels = load_labels()
    present = {os.path.splitext(x)[0] for x in os.listdir(RECON)
               if x.endswith(".npy")}
    ids = sorted(present & set(labels))

    normal = [i for i in ids if labels[i][5] == 0]
    subdural = [i for i in ids if labels[i][4] == 1]
    print(f"[qc] {len(ids)} slices  |  normal(any=0)={len(normal)}  "
          f"subdural=+{len(subdural)}")

    def sample(pool):
        pool = list(pool)
        if len(pool) <= N_PER_GROUP:
            return pool
        idx = rng.choice(len(pool), size=N_PER_GROUP, replace=False)
        return [pool[j] for j in idx]

    norm_ids, sdh_ids = sample(normal), sample(subdural)
    print(f"[qc] sampling {len(norm_ids)} normal, {len(sdh_ids)} subdural ...",
          flush=True)

    N = summarize(run_group(norm_ids),
                  "NORMAL (any=0) — all ICH-window signal is artifact")
    S = summarize(run_group(sdh_ids),
                  "SUBDURAL-positive — artifact + true peripheral bleed")

    bf_c, bf_prev = bone_frac_trend(N["valid"])
    make_figure(N, S, bf_c, bf_prev)


def _overlay_axes(ax, path, title):
    """recon grayscale (brain window) + red = near-wall ICH-window pixels."""
    from scipy.ndimage import binary_dilation
    hu = np.load(path).astype(np.float32) - HU_OFFSET
    interior, bone = intracranial_mask(hu)
    ax.imshow(hu, cmap="gray", vmin=0, vmax=80)          # brain window
    if interior is not None:
        dist = distance_transform_edt(~bone)
        near_ich = (hu >= ICH_LO) & (hu <= ICH_HI) & interior & (dist <= NEAR_PX)
        near_ich = binary_dilation(near_ich, iterations=1)   # thicken for display only
        rgba = np.zeros((*hu.shape, 4))
        rgba[near_ich] = [1, 0.15, 0.15, 1.0]
        ax.imshow(rgba)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _illustrative(rows):
    """Pick the slice with the most near-wall ICH voxels among mid-sized interiors
    (avoids messy skull-base/vertex slices) — the clearest crescent to show."""
    v = [r for r in rows if r.get("valid") and 12000 <= r["interior"] <= 45000]
    if not v:
        v = [r for r in rows if r.get("valid")]
    v.sort(key=lambda r: -r["near_ich"])
    return v


def make_figure(N, S, bf_c, bf_prev):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centers = 0.5 * (DIST_EDGES[:-1] + DIST_EDGES[1:])
    fig = plt.figure(figsize=(13.5, 9.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05],
                          hspace=0.30, wspace=0.28)

    # A: prevalence vs distance-to-inner-table (THE money figure)
    axA = fig.add_subplot(gs[0, 0])
    axA.plot(centers, 100*N["prev"], "o-", color="#7f8c8d",
             label="normal (pure artifact)")
    axA.plot(centers, 100*S["prev"], "s-", color="#c0392b",
             label="subdural + (artifact+bleed)")
    axA.axvspan(0, NEAR_PX, color="#f2c94c", alpha=0.25)
    axA.set_xlabel("distance from inner table (px, ~0.9 mm/px)")
    axA.set_ylabel("P(voxel ∈ 48–175 HU)  %")
    axA.set_title("A. Pseudo-ICH spikes AT the bone\nand decays inward",
                  fontsize=10)
    axA.legend(fontsize=8); axA.grid(alpha=0.3)

    # B: THE hypothesis — near-wall pseudo-ICH rises with bone content (normals)
    axB = fig.add_subplot(gs[0, 1])
    axB.plot(100*bf_c, 100*bf_prev, "o-", color="#c0392b")
    axB.set_ylim(0, 70)
    axB.set_xlabel("bone fraction of slice  (thin→thick / dense skull)")
    axB.set_ylabel("near-wall pseudo-ICH prevalence %")
    axB.set_title("B. Pervasive (38–63%) across slices;\nworst where skull is THIN/tangential",
                  fontsize=10)
    axB.grid(alpha=0.3)

    # C: near vs deep prevalence, normal vs subdural
    axC = fig.add_subplot(gs[0, 2])
    x = np.arange(2); w = 0.35
    axC.bar(x - w/2, [100*N["near"], 100*N["deep"]], w, color="#7f8c8d",
            label="normal")
    axC.bar(x + w/2, [100*S["near"], 100*S["deep"]], w, color="#c0392b",
            label="subdural +")
    axC.set_xticks(x); axC.set_xticklabels([f"near wall\n(≤{NEAR_PX:.0f}px)", "deep"])
    axC.set_ylabel("ICH-window prevalence %")
    axC.set_title("C. True SDH lifts the near-wall shelf\nabove the artifact floor",
                  fontsize=10)
    axC.legend(fontsize=8)

    # D,E,F: example overlays — two clearest normals + a subdural
    nv = _illustrative(N["valid"])
    ex_n1, ex_n2 = nv[0]["id"], nv[1]["id"]
    ex_s = _illustrative(S["valid"])[0]["id"]

    _overlay_axes(fig.add_subplot(gs[1, 0]), os.path.join(RECON, ex_n1 + ".npy"),
                  f"D. NORMAL {ex_n1}\nred = pseudo-ICH on inner table (no bleed)")
    _overlay_axes(fig.add_subplot(gs[1, 1]), os.path.join(RECON, ex_n2 + ".npy"),
                  f"E. NORMAL {ex_n2}\nfalse ICH hugs the skull, no bleed")
    _overlay_axes(fig.add_subplot(gs[1, 2]), os.path.join(RECON, ex_s + ".npy"),
                  f"F. SUBDURAL+ {ex_s}\ntrue bleed sits in the same band")

    fig.suptitle("Partial-volume pseudo-ICH QC on the 50k recon cache  "
                 f"(ICH window {ICH_LO}–{ICH_HI} HU, bone ≥{BONE_HU} HU)",
                 fontsize=12, weight="bold")
    fig.savefig("qc_partial_volume_50k.png", dpi=150, bbox_inches="tight")
    print("\n[qc] wrote qc_partial_volume_50k.png")


if __name__ == "__main__":
    main()
