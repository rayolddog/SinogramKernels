#!/usr/bin/env python3
"""visualize_aniso.py — top-2-level kernels of the trained AnisoStemNet (sinogram).
Level 1: 48 kernels of 60×15 (detector-tall, angle-narrow), single input channel,
directly visualizable. Level 2: 96 kernels of 30×8 with 48 input channels —
summarized by averaging over input channels to show the dominant spatial pattern.
Convention: rows = detector axis, cols = angle axis."""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = os.environ.get("ANISO_EXP", "aniso_200k")
OUTPNG = os.environ.get("ANISO_OUT", "aniso_kernels.png")


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
    ck = torch.load(f"{EXP}/sinogram/best.pt", map_location="cpu",
                    weights_only=False)
    L1 = ck["model"]["stem.0.weight"].numpy()[:, 0]           # (48, 60, 15)
    L2 = ck["model"]["stem.3.weight"].numpy().mean(1)         # (96, 30, 8) input-avg

    def orient(bank):
        h, w = bank.shape[1], bank.shape[2]
        cols = 6 if w > h else 12          # fewer columns for wide (angle-wide) kernels
        tag = ("detector-short ↕, angle-wide ↔" if w > h
               else "detector-tall ↕, angle-narrow ↔")
        return cols, f"{h}×{w}", tag

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    c1, s1, t1 = orient(L1)
    ax[0].imshow(mosaic(L1, cols=c1), cmap="RdBu_r", vmin=-1, vmax=1,
                 interpolation="nearest")
    ax[0].set_title(f"Level 1 — {s1} kernels (×{L1.shape[0]})\n{t1}; red +, blue −",
                    fontsize=11)
    ax[0].axis("off")
    c2, s2, _ = orient(L2)
    ax[1].imshow(mosaic(L2, cols=c2), cmap="RdBu_r", vmin=-1, vmax=1,
                 interpolation="nearest")
    ax[1].set_title(f"Level 2 — {s2} kernels (×{L2.shape[0]}, input-averaged)",
                    fontsize=11)
    ax[1].axis("off")

    fig.suptitle(f"AnisoStemNet top-2-level kernels — {EXP} "
                 f"(val AUC {ck['val_auc']:.3f})", fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUTPNG, dpi=150)
    print(f"wrote {OUTPNG}  (level-1 kernel shape {L1.shape[1]}×{L1.shape[2]})")


if __name__ == "__main__":
    main()
