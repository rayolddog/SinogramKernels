#!/usr/bin/env python3
"""compare_aniso_kernels.py DIR [DIR ...] — level-1 stem kernels of several aniso
runs side by side, to judge speckle/noise. Pass run dirs as args; each must have
sinogram/best.pt. Kernels are L2-normalized per-kernel for display."""
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def mosaic(bank, cols, gap=2):
    C, H, W = bank.shape
    rows = int(np.ceil(C / cols))
    canvas = np.full((rows * (H + gap) + gap, cols * (W + gap) + gap), np.nan)
    for i, k in enumerate(bank):
        k = k / (np.abs(k).max() + 1e-9)
        r, c = divmod(i, cols)
        y0, x0 = r * (H + gap) + gap, c * (W + gap) + gap
        canvas[y0:y0 + H, x0:x0 + W] = k
    return canvas


def main():
    runs = sys.argv[1:] or ["aniso_200k", "aniso_h_200k"]
    fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 6))
    if len(runs) == 1:
        axes = [axes]
    for ax, r in zip(axes, runs):
        ck = torch.load(f"{r}/sinogram/best.pt", map_location="cpu",
                        weights_only=False)
        L1 = ck["model"]["stem.0.weight"].numpy()[:, 0]
        h, w = L1.shape[1], L1.shape[2]
        cols = 6 if w > h else 12
        ax.imshow(mosaic(L1, cols), cmap="RdBu_r", vmin=-1, vmax=1,
                  interpolation="nearest")
        ax.set_title(f"{r}\nL1 {h}×{w}, AUC {ck['val_auc']:.3f}, ep{ck['epoch']}",
                     fontsize=11)
        ax.axis("off")
    fig.suptitle("Level-1 stem kernels — speckle comparison", fontsize=13,
                 weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig("aniso_kernel_comparison.png", dpi=150)
    print("wrote aniso_kernel_comparison.png  (runs: " + ", ".join(runs) + ")")


if __name__ == "__main__":
    main()
