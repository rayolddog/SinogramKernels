#!/usr/bin/env python3
"""whitening_explain.py — show exactly what the spectral whitening does, and check
whether the anisotropic filter (not the network) could be responsible for the
detector-axis kernel result.

Panels: original vs whitened example slice (recon + sinogram); radial power
spectrum before/after (whitening flattens it); the filters themselves. Prints the
detector:angle energy ratio of the FILTER vs of the learned KERNELS — if the
kernels are more detector-oriented than the filter, that orientation is LEARNED,
not imposed by whitening.
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import twin_common as tc
import analyze_kernels as ak

EXP = "k200_N0"
CACHE = "/mnt/mlcache/subset200k"


def load_filter(dom):
    return np.load(f"{EXP}/{dom}/whiten.npy")           # unshifted freq response (real)


def whiten_img(x, Wf):
    xw = np.real(np.fft.ifft2(np.fft.fft2(x) * Wf))
    return (xw - xw.mean()) / (xw.std() + 1e-6)


def radial_profile(P2d):
    n = P2d.shape[0]; c = n // 2
    y, x = np.indices(P2d.shape)
    r = np.hypot(x - c, y - c).astype(int)
    tot = np.bincount(r.ravel(), P2d.ravel())
    cnt = np.bincount(r.ravel())
    return tot / np.maximum(cnt, 1)


def avg_power(dom, ids, mean, std, Wf=None, n=150):
    acc = None
    for s in ids[:n]:
        arr = np.load(f"{CACHE}/{dom}/{s}.npy").astype(np.float32)
        x = (arr - mean) / std
        if Wf is not None:
            x = whiten_img(x, Wf)
        p = np.abs(np.fft.fft2(x)) ** 2
        acc = p if acc is None else acc + p
    return np.fft.fftshift(acc / min(n, len(ids)))


def det_angle_ratio_of_filter(Wf):
    """Energy of the filter's frequency response weighted by detector-freq (|fy|)
    vs angle-freq (|fx|). >1 = amplifies detector axis more; ~1 = isotropic."""
    n = Wf.shape[0]
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.fftfreq(n)[None, :]
    P = Wf ** 2
    return float((P * np.abs(fy)).sum() / (P * np.abs(fx)).sum())


def kernel_det_ratio(dom):
    ck = torch.load(f"{EXP}/{dom}/best.pt", map_location="cpu", weights_only=False)
    rows = ak.describe(ck["conv1"][:, 0])
    return float(np.mean([r["det_ratio"] for r in rows]))


def main():
    tc.CACHE = CACHE                     # read labels/stats from the 200k cache
    labels = tc.load_labels()
    ids = sorted(labels)
    ex = ids[0]

    fig, ax = plt.subplots(3, 3, figsize=(13, 11))
    print("detector:angle energy ratio  (>1 = detector-axis oriented)")
    print(f"{'domain':10s} {'WHITEN FILTER':>14s} {'LEARNED KERNELS':>16s}")

    for row, dom in enumerate(["recon", "sinogram"]):
        mean, std = tc.domain_stats(dom, ids, n=300)
        Wf = load_filter(dom)
        arr = np.load(f"{CACHE}/{dom}/{ex}.npy").astype(np.float32)
        x = (arr - mean) / std
        xw = whiten_img(x, Wf)

        vlim = np.percentile(np.abs(xw), 99)
        ax[row, 0].imshow(x, cmap="gray"); ax[row, 0].set_title(
            f"{dom}: original (standardized)", fontsize=10); ax[row, 0].axis("off")
        ax[row, 1].imshow(xw, cmap="gray", vmin=-vlim, vmax=vlim)
        ax[row, 1].set_title(f"{dom}: WHITENED", fontsize=10); ax[row, 1].axis("off")

        Pbefore = avg_power(dom, ids, mean, std, None)
        Pafter = avg_power(dom, ids, mean, std, Wf)
        rb, ra = radial_profile(Pbefore), radial_profile(Pafter)
        axp = ax[row, 2]
        axp.semilogy(rb / rb.max(), color="#2c7fb8", label="before (∝1/f²)")
        axp.semilogy(ra / ra.max(), color="#c0392b", label="after (flat + rolloff)")
        axp.set_title(f"{dom}: avg radial power spectrum", fontsize=10)
        axp.set_xlabel("radial spatial frequency"); axp.legend(fontsize=8)
        axp.grid(alpha=0.3)

        fr = det_angle_ratio_of_filter(Wf)
        kr = kernel_det_ratio(dom)
        print(f"{dom:10s} {fr:14.3f} {kr:16.3f}")

    # bottom row: the two filters (fftshifted) + summary text
    for col, dom in enumerate(["recon", "sinogram"]):
        Wf = np.fft.fftshift(load_filter(dom))
        im = ax[2, col].imshow(Wf, cmap="magma")
        ax[2, col].set_title(f"{dom}: whitening filter W(f)\n(gain vs spatial freq)",
                             fontsize=10)
        ax[2, col].axis("off")
        plt.colorbar(im, ax=ax[2, col], fraction=0.046)
    ax[2, 2].axis("off")
    ax[2, 2].text(0.0, 0.5,
                  "WHITENING = one fixed linear filter per domain,\n"
                  "estimated from the average image power spectrum\n"
                  "ONLY (no labels, no metadata, no per-image fit).\n\n"
                  "It flattens the 1/f² spectrum → removes generic\n"
                  "second-order (pairwise) structure so the network\n"
                  "can't coast on it and reaches lesion structure.\n\n"
                  "It is NOT covariate adjustment — it knows nothing\n"
                  "about scanner, age, or slice position.",
                  fontsize=9, va="center")

    fig.suptitle("What spectral whitening does (and the detector-axis confound check)",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("whitening_explain.png", dpi=150)
    print("\nwrote whitening_explain.png")


if __name__ == "__main__":
    main()
