#!/usr/bin/env python3
"""
analyze_kernels.py — compare the layer-1 conv kernels of the recon-trained and
sinogram-trained twin CNNs, testing the postulate that SINOGRAM kernels are more:
  (1) anisotropic          — matched to 1-D trace tangents (not 2-D blobs)
  (2) oscillatory/high-pass — rediscovering the FBP ramp the RAW sinogram lacks
                              (positive ridge with negative perpendicular sidelobes)
  (3) high spatial frequency and detector-axis oriented (ramp acts along detector)

All descriptors are computed on L2-normalized kernels, so they are invariant to
the arbitrary conv1 scale that BatchNorm absorbs. Convention: kernel axis kH =
detector, kW = angle.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "kernel_exp"


def load_bank(domain):
    w = np.load(os.path.join(OUT, domain, "conv1.npy"))     # (C1,1,K,K)
    return w[:, 0]                                           # (C1,K,K)


# ---------------- per-kernel descriptors -----------------------------------
def norml2(k):
    return k / (np.linalg.norm(k) + 1e-12)


def dc_fraction(k):
    """|mean| relative to L1 energy: ~0 => zero-sum (DC-killing, high-pass)."""
    return abs(k.sum()) / (np.abs(k).sum() + 1e-12)


def power_spectrum(k):
    K = np.fft.fftshift(np.abs(np.fft.fft2(k)) ** 2)
    return K


def freq_grid(n):
    f = np.fft.fftshift(np.fft.fftfreq(n)) * 2          # [-1,1] (Nyquist=1)
    fx, fy = np.meshgrid(f, f)                          # fy=rows(detector), fx=cols(angle)
    return fx, fy


def spectral_centroid(k):
    """Mean radial frequency weighted by power (0..~1), and detector/angle split."""
    P = power_spectrum(k)
    n = k.shape[0]
    fx, fy = freq_grid(n)
    Ps = P.sum() + 1e-12
    r = np.sqrt(fx ** 2 + fy ** 2)
    cent = float((P * r).sum() / Ps)
    det_hf = float((P * np.abs(fy)).sum() / Ps)         # energy along DETECTOR axis
    ang_hf = float((P * np.abs(fx)).sum() / Ps)         # energy along ANGLE axis
    return cent, det_hf, ang_hf


def anisotropy(k):
    """Orientation coherence from the power-spectrum inertia tensor: 0=isotropic,
    1=perfectly oriented (energy on a line)."""
    P = power_spectrum(k)
    n = k.shape[0]
    fx, fy = freq_grid(n)
    w = P / (P.sum() + 1e-12)
    Jxx = float((w * fx * fx).sum())
    Jyy = float((w * fy * fy).sum())
    Jxy = float((w * fx * fy).sum())
    tr = Jxx + Jyy
    det = Jxx * Jyy - Jxy * Jxy
    disc = max(tr * tr - 4 * det, 0.0)
    l1 = (tr + np.sqrt(disc)) / 2
    l2 = (tr - np.sqrt(disc)) / 2
    return float((l1 - l2) / (l1 + l2 + 1e-12))


def oscillation(k):
    """Sign changes along the dominant spatial axis (proxy for negative sidelobes /
    ramp-like oscillation). Collapse to the axis of larger variance, count crossings."""
    var_r = k.sum(1).var()      # profile along detector (collapse angle)
    var_c = k.sum(0).var()      # profile along angle
    prof = k.sum(1) if var_r >= var_c else k.sum(0)
    prof = prof - prof.mean()
    s = np.sign(prof)
    s[s == 0] = 1
    return int((np.diff(s) != 0).sum())


def negative_sidelobe(k):
    """Energy fraction carried by negative weights (a pure blob≈0; a ramp/edge≈~0.5)."""
    neg = np.abs(k[k < 0]).sum()
    return float(neg / (np.abs(k).sum() + 1e-12))


def describe(bank):
    rows = []
    for k in bank:
        k = norml2(k)
        cent, det_hf, ang_hf = spectral_centroid(k)
        rows.append(dict(dc=dc_fraction(k), aniso=anisotropy(k),
                         osc=oscillation(k), neg=negative_sidelobe(k),
                         cent=cent, det_hf=det_hf, ang_hf=ang_hf,
                         det_ratio=det_hf / (ang_hf + 1e-12)))
    return rows


def summarize(name, rows):
    keys = ["aniso", "neg", "osc", "cent", "dc", "det_ratio"]
    print(f"\n[{name}]  ({len(rows)} kernels)")
    out = {}
    for kk in keys:
        v = np.array([r[kk] for r in rows])
        out[kk] = v
        print(f"  {kk:10s} mean={v.mean():.3f}  median={np.median(v):.3f}")
    return out


def cohens_d(a, b):
    va, vb = a.var(ddof=1), b.var(ddof=1)
    sp = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / (sp + 1e-12)


# ---------------- figure ---------------------------------------------------
def kernel_grid(ax, bank, title):
    C, K, _ = bank.shape
    cols = 8
    rows = int(np.ceil(C / cols))
    canvas = np.full((rows * (K + 1) + 1, cols * (K + 1) + 1), np.nan)
    for i, k in enumerate(bank):
        k = norml2(k)
        m = np.abs(k).max() + 1e-12
        r, c = divmod(i, cols)
        y0, x0 = r * (K + 1) + 1, c * (K + 1) + 1
        canvas[y0:y0 + K, x0:x0 + K] = k / m
    ax.imshow(canvas, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def hist_panel(ax, rec, sino, key, title, xlabel):
    a, b = rec[key], sino[key]
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    bins = np.linspace(lo, hi, 20)
    ax.hist(a, bins=bins, alpha=0.6, color="#2c7fb8", density=True, label="recon")
    ax.hist(b, bins=bins, alpha=0.6, color="#c0392b", density=True, label="sinogram")
    ax.axvline(np.median(a), color="#2c7fb8", ls="--", lw=1)
    ax.axvline(np.median(b), color="#c0392b", ls="--", lw=1)
    d = cohens_d(b, a)
    ax.set_title(f"{title}\n(Cohen's d = {d:+.2f}, sino−recon)", fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.legend(fontsize=7)


def main():
    rec_bank, sino_bank = load_bank("recon"), load_bank("sinogram")
    rec, sino = describe(rec_bank), describe(sino_bank)
    R = summarize("RECON", rec)
    S = summarize("SINOGRAM", sino)

    print("\n[effect sizes  Cohen's d  (sinogram − recon)]")
    for kk in ["aniso", "neg", "osc", "cent", "dc", "det_ratio"]:
        print(f"  {kk:10s} d = {cohens_d(S[kk], R[kk]):+.2f}")

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1.0], hspace=0.32, wspace=0.28)
    kernel_grid(fig.add_subplot(gs[0, :2]), rec_bank,
                "RECON-domain layer-1 kernels (red +, blue −)")
    kernel_grid(fig.add_subplot(gs[0, 2:]), sino_bank,
                "SINOGRAM-domain layer-1 kernels")

    hist_panel(fig.add_subplot(gs[1, 0]), R, S, "aniso",
               "Anisotropy", "0 isotropic → 1 oriented")
    hist_panel(fig.add_subplot(gs[1, 1]), R, S, "neg",
               "Negative-lobe energy", "fraction from negative weights")
    hist_panel(fig.add_subplot(gs[1, 2]), R, S, "cent",
               "Spectral centroid", "mean radial freq (Nyquist=1)")
    hist_panel(fig.add_subplot(gs[1, 3]), R, S, "det_ratio",
               "Detector : angle energy", "ramp acts along detector →")

    fig.suptitle("Layer-1 kernel comparison: reconstructed image vs. raw sinogram "
                 "(identical CNN, identical init, paired data)",
                 fontsize=13, weight="bold")
    fig.savefig("kernel_comparison.png", dpi=150, bbox_inches="tight")
    print("\n[analyze] wrote kernel_comparison.png")


if __name__ == "__main__":
    main()
