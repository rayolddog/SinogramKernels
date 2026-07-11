#!/usr/bin/env python3
"""
make_scale_summary.py — the actual result of the kernel study: how the recon-vs-
sinogram layer-1 difference depends on first-layer SCALE. Effect size (Cohen's d,
sinogram−recon) for each kernel descriptor at 9x9 vs 31x31.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analyze_kernels as ak

KEYS = ["aniso", "neg", "osc", "cent", "dc", "det_ratio"]
LABELS = ["anisotropy", "neg-lobe", "oscillation", "spec.centroid",
          "DC frac", "detector:angle"]


def effects(out_dir):
    ak.OUT = out_dir
    rec = ak.describe(ak.load_bank("recon"))
    sino = ak.describe(ak.load_bank("sinogram"))
    d = {}
    for k in KEYS:
        a = np.array([r[k] for r in rec])
        b = np.array([r[k] for r in sino])
        d[k] = ak.cohens_d(b, a)
    return d


def main():
    small = effects("kernel_exp")        # 9x9, 48 filters
    big = effects("kernel_exp_big")      # 31x31, 16 filters

    x = np.arange(len(KEYS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(x - w/2, [small[k] for k in KEYS], w, color="#8fb8d8",
           label="9×9 first layer (48 filters)")
    ax.bar(x + w/2, [big[k] for k in KEYS], w, color="#c0392b",
           label="31×31 first layer (16 filters)")
    ax.axhspan(-0.2, 0.2, color="0.85", alpha=0.6, zorder=0)
    ax.axhline(0, color="k", lw=0.8)
    ax.text(len(KEYS)-0.5, 0.0, "  negligible\n  (|d|<0.2)", va="center",
            fontsize=8, color="0.4")
    ax.set_xticks(x); ax.set_xticklabels(LABELS, rotation=20, ha="right")
    ax.set_ylabel("effect size  Cohen's d  (sinogram − recon)")
    ax.set_title("Recon vs. sinogram layer-1 kernels: the difference is SCALE-DEPENDENT\n"
                 "indistinguishable at 9×9; directional signal only at 31×31 "
                 "(detector-axis energy ↑, oscillation ↓ for sinogram)",
                 fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig("kernel_scale_summary.png", dpi=150)
    print("wrote kernel_scale_summary.png")
    print("\n           9x9      31x31")
    for k, lab in zip(KEYS, LABELS):
        print(f"  {lab:16s} {small[k]:+.2f}    {big[k]:+.2f}")


if __name__ == "__main__":
    main()
