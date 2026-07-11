#!/usr/bin/env python3
"""visualize_stem.py — first-level (stem) conv kernels of the trained se_200k
SE-ResNeXt50, recon vs sinogram. SE-ResNeXt's stem is a single 7x7 conv (64
filters, 1 input channel) — directly visualizable. Its second level is grouped
3x3 / 1x1 bottleneck convs, which are NOT image-visualizable — one motivation for
the anisotropic-stem architecture, which has several visualizable levels."""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = "se_200k"


def stem_kernels(dom):
    ck = torch.load(f"{EXP}/{dom}/best.pt", map_location="cpu", weights_only=False)
    return ck["model"]["conv1.weight"].numpy()[:, 0]      # (64, 7, 7)


def grid(ax, bank, title):
    C, K, _ = bank.shape
    cols = 8
    rows = int(np.ceil(C / cols))
    canvas = np.full((rows * (K + 1) + 1, cols * (K + 1) + 1), np.nan)
    for i, k in enumerate(bank):
        k = k / (np.abs(k).max() + 1e-9)
        r, c = divmod(i, cols)
        y0, x0 = r * (K + 1) + 1, c * (K + 1) + 1
        canvas[y0:y0 + K, x0:x0 + K] = k
    ax.imshow(canvas, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def main():
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.6))
    grid(ax[0], stem_kernels("recon"), "RECON stem  (SE-ResNeXt50 conv1, 7×7 ×64)")
    grid(ax[1], stem_kernels("sinogram"), "SINOGRAM stem  (conv1, 7×7 ×64)")
    fig.suptitle("Trained first-level (stem) kernels — se_200k  (red +, blue −)",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("stem_kernels_se200k.png", dpi=150)
    print("wrote stem_kernels_se200k.png")


if __name__ == "__main__":
    main()
