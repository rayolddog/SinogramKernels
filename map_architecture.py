#!/usr/bin/env python3
"""map_architecture.py — side-by-side architecture map of the base net vs the
dilated net: the 4 anisotropic stem conv layers (kernel size, stride, channels,
feature-map size) then the 3x3 ResNet blocks (kernel, dilation, effective RF).
Only difference: the ResBlock dilations (1,1,1) vs (1,2,4)."""
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import twin_common as tc


def build(dils):
    return tc.AnisoStemNet(6, stem_spec=tc.ANISO_STEM, stem_strides=(2, 2, 2, 2),
                           res_k=3, res_dilations=dils)


def trace(model):
    x = torch.randn(1, 1, 256, 256)
    rows = [("input", "1 channel", "1×256×256", "io")]
    h = x
    si = 0
    for layer in model.stem:
        h = layer(h)
        if isinstance(layer, nn.Conv2d):
            si += 1
            kh, kw = layer.kernel_size
            rows.append((f"stem conv {si}",
                         f"{kh}×{kw} kernel · stride {layer.stride[0]} · "
                         f"{layer.in_channels}→{layer.out_channels}",
                         f"{h.shape[1]}×{h.shape[2]}×{h.shape[3]}", "stem"))
    for i, blk in enumerate(model.res):
        h = blk(h)
        d = blk.c1.dilation[0]
        rf = 4 * d + 1               # two stacked 3x3 at dilation d
        rows.append((f"resblock {i+1}",
                     f"3×3 ×2 · dilation {d} · eff. RF {rf}×{rf} · 256 ch",
                     f"{h.shape[1]}×{h.shape[2]}×{h.shape[3]}", "res"))
    rows.append(("head", "global avg-pool → FC", "6 logits", "io"))
    return rows


COL = {"io": "#bdc3c7", "stem": "#e67e22", "res": "#2980b9"}


def draw(ax, rows, title):
    n = len(rows)
    ax.set_xlim(0, 10); ax.set_ylim(0, n)
    ax.set_title(title, fontsize=12, weight="bold")
    ax.axis("off")
    for i, (name, detail, shape, kind) in enumerate(rows):
        y = n - 1 - i
        box = FancyBboxPatch((0.5, y + 0.12), 9, 0.76,
                             boxstyle="round,pad=0.02,rounding_size=0.1",
                             fc=COL[kind], ec="#333", alpha=0.85, lw=1)
        ax.add_patch(box)
        ax.text(0.8, y + 0.5, name, fontsize=9.5, weight="bold", va="center")
        ax.text(3.3, y + 0.5, detail, fontsize=8.5, va="center")
        ax.text(9.3, y + 0.5, shape, fontsize=8, va="center", ha="right",
                style="italic", color="#222")
        if i < n - 1:
            ax.annotate("", xy=(5, y + 0.12), xytext=(5, y + 0.0 + 0.12 + 0.12),
                        arrowprops=dict(arrowstyle="-", color="#999", lw=0.8))


def main():
    base = trace(build((1, 1, 1)))
    dil = trace(build((1, 2, 4)))
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 7.5))
    draw(a, base, "BASE net  (abl_base, AUC 0.877)")
    draw(b, dil, "DILATED net  (abl_dil, AUC 0.878)")
    # legend
    for ax in (a, b):
        for lab, k in [("stem (anisotropic)", "stem"), ("ResNet body", "res")]:
            ax.scatter([], [], marker="s", s=80, c=COL[k], label=lab)
        ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle("Architecture map — 4 anisotropic stem layers + 3×3 ResNet "
                 "(only difference: body dilation 1,1,1 → 1,2,4)",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("architecture_map.png", dpi=150)
    print("wrote architecture_map.png")
    for tag, rows in [("BASE", base), ("DILATED", dil)]:
        print(f"\n{tag}:")
        for name, detail, shape, _ in rows:
            print(f"  {name:12s} {detail:48s} -> {shape}")


if __name__ == "__main__":
    main()
