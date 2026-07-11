#!/usr/bin/env python3
"""make_datasize_summary.py — the overnight result: how the sinogram-vs-recon
layer-1 kernel signature and classification accuracy scale with training-set size
(whitened inputs, 31x31/48-filter conv1, val-optimal best.pt)."""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analyze_kernels as ak

RUNS = [("k200_N50000", 50), ("k200_N100000", 100), ("k200_N0", 180)]
KEYS = [("det_ratio", "detector:angle  (orientation)", "#c0392b"),
        ("cent", "spectral centroid  (−=lower freq)", "#2c7fb8"),
        ("aniso", "anisotropy  (elongation)", "#27ae60")]


def load(run, dom):
    ck = torch.load(f"{run}/{dom}/best.pt", map_location="cpu", weights_only=False)
    return ck["conv1"][:, 0], ck["val_auc"]


def main():
    Ns = [n for _, n in RUNS]
    eff = {k: [] for k, _, _ in KEYS}
    auc_r, auc_s = [], []
    for run, _ in RUNS:
        rb, ra = load(run, "recon")
        sb, sa = load(run, "sinogram")
        auc_r.append(ra); auc_s.append(sa)
        rd, sd = ak.describe(rb), ak.describe(sb)
        for k, _, _ in KEYS:
            a = np.array([r[k] for r in rd]); b = np.array([r[k] for r in sd])
            eff[k].append(ak.cohens_d(b, a))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for k, lab, col in KEYS:
        ax1.plot(Ns, eff[k], "o-", color=col, lw=2, label=lab)
    ax1.axhspan(-0.2, 0.2, color="0.85", alpha=0.6, zorder=0)
    ax1.axhline(0, color="k", lw=0.8)
    ax1.set_xlabel("training slices (thousands)")
    ax1.set_ylabel("kernel effect size  Cohen's d  (sinogram − recon)")
    ax1.set_title("Domain signature STRENGTHENS with data\n"
                  "(more cases → higher-order trace structure appears in kernels)",
                  fontsize=11)
    ax1.legend(fontsize=8, loc="center left")
    ax1.grid(alpha=0.3)
    for k, _, col in KEYS:
        for x, y in zip(Ns, eff[k]):
            ax1.annotate(f"{y:+.1f}", (x, y), fontsize=7, color=col,
                         xytext=(0, 5), textcoords="offset points", ha="center")

    ax2.plot(Ns, auc_r, "s-", color="#2c7fb8", lw=2, label="recon CNN")
    ax2.plot(Ns, auc_s, "o-", color="#c0392b", lw=2, label="sinogram CNN")
    ax2.set_xlabel("training slices (thousands)")
    ax2.set_ylabel("val macro-AUC (best epoch)")
    ax2.set_title("Whitened sinogram CNN matches/edges recon\n"
                  "both improve with data (feasibility)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    for x, y in zip(Ns, auc_s):
        ax2.annotate(f"{y:.3f}", (x, y), fontsize=7, color="#c0392b",
                     xytext=(0, 6), textcoords="offset points", ha="center")
    for x, y in zip(Ns, auc_r):
        ax2.annotate(f"{y:.3f}", (x, y), fontsize=7, color="#2c7fb8",
                     xytext=(0, -12), textcoords="offset points", ha="center")

    fig.suptitle("Sinogram vs. recon layer-1 kernels — scaling with training data "
                 "(whitened, 31×31)", fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("kernel_datasize_summary.png", dpi=150)
    print("wrote kernel_datasize_summary.png")
    print("\n  N(k)   det_ratio   cent    aniso   | recon_AUC  sino_AUC")
    for i, (_, n) in enumerate(RUNS):
        print(f"  {n:4d}   {eff['det_ratio'][i]:+.2f}      {eff['cent'][i]:+.2f}"
              f"   {eff['aniso'][i]:+.2f}   |  {auc_r[i]:.4f}    {auc_s[i]:.4f}")


if __name__ == "__main__":
    main()
