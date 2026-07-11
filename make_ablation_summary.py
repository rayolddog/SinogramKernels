#!/usr/bin/env python3
"""make_ablation_summary.py — attribute aniso2's gain to individual modifications.
Parses each abl_* run for best AUC and median epoch time, computes params, and
plots AUC + efficiency. Baseline = abl_base (200k); reference = aniso2_400k."""
import re
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import twin_common as tc

BASE_AUC = None
ANISO2_400K = 0.9142   # all four changes + 400k data (from earlier run)

# (dir, label, model-build spec) — spec None => iso arch
CFG = [
    ("abl_base",   "baseline (15×60, s2, 3×3)", ("h",  False, 3, False)),
    ("abl_tall",   "+ taller 30×60 stem",       ("tall", False, 3, False)),
    ("abl_fulres", "+ less downsampling",        ("h",  True,  3, False)),
    ("abl_dil",    "+ dilated body (1,2,4)",     ("h",  False, 3, True)),
    ("abl_5x5",    "+ 5×5 body",                 ("h",  False, 5, False)),
    ("abl_all",    "all four (=aniso2 @200k)",   ("tall", True, 5, True)),
    ("abl_iso",    "isotropic stem",             None),
]


def params(spec):
    if spec is None:
        m = tc.AnisoStemNet(6, stem_spec=tc.ISO_STEM)
    else:
        stem, ful, rk, rd = spec
        m = tc.AnisoStemNet(6, stem_spec=tc.ANISO2_STEM if stem == "tall" else tc.ANISO_STEM,
                            stem_strides=(2, 2, 2, 1) if ful else (2, 2, 2, 2),
                            res_k=rk, res_dilations=(1, 2, 4) if rd else (1, 1, 1))
    return sum(p.numel() for p in m.parameters()) / 1e6


def parse(d):
    log = open(f"{d}/stdout.log").read()
    auc = float(re.search(r"best_val_macroAUC=([0-9.]+)", log).group(1))
    times = [int(t) for t in re.findall(r"\((\d+)s\)", log)]
    return auc, (float(np.median(times)) if times else 0.0)


def main():
    rows = []
    for d, lab, spec in CFG:
        auc, ep = parse(d)
        rows.append((d, lab, auc, ep, params(spec)))
    base = next(r[2] for r in rows if r[0] == "abl_base")

    print(f"{'config':13s} {'AUC':>6s} {'Δbase':>7s} {'params':>7s} {'s/epoch':>8s}")
    for d, lab, auc, ep, pm in rows:
        print(f"{d:13s} {auc:.4f} {auc-base:+.4f}  {pm:5.1f}M {ep:7.0f}")
    print(f"\naniso2 @400k: {ANISO2_400K:.4f}  (data effect: all@200k {rows[5][2]:.4f} "
          f"-> {ANISO2_400K:.4f} = {ANISO2_400K-rows[5][2]:+.4f})")

    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5.5))
    labs = [r[1] for r in rows]
    aucs = np.array([r[2] for r in rows])
    y = np.arange(len(rows))[::-1]
    cols = ["#7f8c8d" if r[0] == "abl_base" else
            ("#27ae60" if r[2] >= base else "#c0392b") for r in rows]
    a.barh(y, aucs, color=cols)
    a.axvline(base, color="#333", ls="--", lw=1, label=f"baseline {base:.3f}")
    a.axvline(ANISO2_400K, color="#2c7fb8", ls="--", lw=1.5,
              label=f"aniso2 @400k {ANISO2_400K:.3f}")
    a.set_yticks(y); a.set_yticklabels(labs, fontsize=9)
    a.set_xlim(0.86, 0.92); a.set_xlabel("val macro-AUC")
    a.set_title("Each change at fixed 200k data\n(green≥baseline, red<baseline)",
                fontsize=11)
    a.legend(fontsize=8, loc="lower right")
    for yi, r in zip(y, rows):
        a.text(r[2] + 0.0005, yi, f"{r[2]-base:+.3f}", va="center", fontsize=7.5)

    eps = np.array([r[3] for r in rows])
    b.scatter(eps, aucs, s=60, c=cols, zorder=3)
    for r in rows:
        b.annotate(r[1].replace("+ ", "").split(" (")[0], (r[3], r[2]),
                   fontsize=7.5, xytext=(4, 4), textcoords="offset points")
    b.axhline(base, color="#333", ls="--", lw=1)
    b.set_xlabel("seconds / epoch  (efficiency cost →)")
    b.set_ylabel("val macro-AUC")
    b.set_title("Cost vs. benefit\n(dilation is free; tall/less-downsample are costly)",
                fontsize=11)
    b.grid(alpha=0.3)

    fig.suptitle("aniso2 ablation — which modification helped, at what cost "
                 "(sinogram, 200k, 20 epochs)", fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig("ablation_summary.png", dpi=150)
    print("\nwrote ablation_summary.png")


if __name__ == "__main__":
    main()
