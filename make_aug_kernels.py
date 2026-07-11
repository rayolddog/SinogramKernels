#!/usr/bin/env python3
"""make_aug_kernels.py — what does the angular wedge augmentation do to the
FIRST-LAYER (stem.0) kernels? Controlled pair, identical aniso_cfg / seed / data,
differ ONLY by --aug-angular:
    aug-OFF = flip_noaug     aug-ON = abl_base
Level-1 stem kernels are 15 (detector rows) × 60 (angle cols), single input
channel — directly visualizable. Convention: rows = detector axis (↕),
cols = projection angle (↔). The augmentation zeroes random contiguous ANGLE
columns during training, so the question is whether the aug-ON kernels rely less
on specific contiguous angle structure (flatter energy along the angle axis).
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAIR = [("flip_noaug", "aug-OFF"), ("abl_base", "aug-ON")]


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


def load_stem(exp):
    ck = torch.load(f"{exp}/sinogram/best.pt", map_location="cpu",
                    weights_only=False)
    return ck["model"]["stem.0.weight"].numpy()[:, 0], ck.get("val_auc", float("nan"))


def main():
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.7])
    profiles = {}
    for j, (exp, tag) in enumerate(PAIR):
        bank, auc = load_stem(exp)                      # (48, 15, 60)
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(mosaic(bank, cols=6), cmap="RdBu_r", vmin=-1, vmax=1,
                  interpolation="nearest")
        ax.set_title(f"{tag}  ({exp}, val AUC {auc:.3f})\n"
                     f"stem.0: 48 × 15(detector↕)×60(angle↔)", fontsize=11)
        ax.axis("off")
        # energy profile along the ANGLE axis (mean |w| over kernels & detector rows)
        profiles[tag] = np.abs(bank).mean(axis=(0, 1))   # (60,)

    axp = fig.add_subplot(gs[0, 2])
    for tag, p in profiles.items():
        axp.plot(np.linspace(0, 1, len(p)), p / p.mean(), lw=2, label=tag)
    axp.set_xlabel("position along angle axis (0→1)")
    axp.set_ylabel("mean |weight| (normalized)")
    axp.set_title("Angle-axis energy profile\nflatter = less reliant on\nspecific angle columns",
                  fontsize=10)
    axp.legend(fontsize=9); axp.grid(alpha=0.3)

    fig.suptitle("First-layer stem kernels: effect of angular wedge augmentation "
                 "(aniso_cfg, sinogram, seed 0)", fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = "aug_kernels_comparison.png"
    fig.savefig(out, dpi=150)
    # quantify: std of the angle profile (lower = flatter = more wedge-agnostic)
    for tag, p in profiles.items():
        pn = p / p.mean()
        print(f"{tag}: angle-profile CoV={pn.std():.3f}  "
              f"edge/center ratio={ (pn[:6].mean()+pn[-6:].mean())/2 / pn[27:33].mean():.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
