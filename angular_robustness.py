#!/usr/bin/env python3
"""angular_robustness.py — Track 1(a): does detection survive lost angular data
better in the sinogram than in the reconstructed image?

Same corruption, native to each domain: remove a contiguous angular wedge of
width `frac` of 180°.
  * sinogram model  <- sinogram with those angle columns zeroed
  * recon model     <- iradon() of that same wedged sinogram (limited-angle FBP,
                       i.e. streak artifact — the image-space face of lost angles)
Report macro-AUC (and subdural, the peripheral case the optimal-angle argument
favors) vs wedge width, plus RETENTION = AUC(frac)/AUC(0) which is robust to any
baseline offset from the FBP round-trip.
"""
import json
import os
import numpy as np
import torch
from skimage.transform import iradon
from sklearn.metrics import roc_auc_score

import twin_common as tc
from train_twin import apply_whiten

EXP = os.environ.get("ROBUST_EXP", "k200_N0")
CACHE = "/mnt/mlcache/subset200k"
N_TEST = 2000
FRACS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
THETA = np.linspace(0.0, 180.0, 256, endpoint=False)
BATCH = 128
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(dom):
    ck = torch.load(f"{EXP}/{dom}/best.pt", map_location="cpu", weights_only=False)
    if ck.get("arch", "smallcnn") == "seresnext50":
        import timm
        m = timm.create_model("seresnext50_32x4d", pretrained=False,
                              in_chans=1, num_classes=6).to(DEV)
    else:
        m = tc.SmallCNN(6, k1=ck["K1"], c1=ck["C1"], s1=ck["S1"]).to(DEV)
    m.load_state_dict(ck["model"]); m.eval()
    wp = f"{EXP}/{dom}/whiten.npy"
    Wf = (torch.from_numpy(np.load(wp))[None, None].to(DEV)
          if os.path.exists(wp) else None)
    return m, ck["mean"], ck["std"], Wf


def zero_wedge(sino, frac):
    k = int(round(frac * sino.shape[1]))
    if k == 0:
        return sino
    s = sino.copy()
    a = (sino.shape[1] - k) // 2
    s[:, a:a + k] = 0.0
    return s


def infer(model, imgs, mean, std, Wf):
    x = (np.stack(imgs).astype(np.float32) - mean) / std
    x = torch.from_numpy(x)[:, None].to(DEV)
    x = apply_whiten(x, Wf)
    with torch.no_grad(), torch.amp.autocast("cuda"):
        return torch.sigmoid(model(x)).float().cpu().numpy()


def macro_auc(Y, P):
    aucs = [roc_auc_score(Y[:, c], P[:, c]) for c in range(Y.shape[1])
            if Y[:, c].min() != Y[:, c].max()]
    return float(np.mean(aucs))


def main():
    tc.CACHE = CACHE
    labels = tc.load_labels()
    ids = sorted(labels)
    _, val = tc.split_ids(ids)
    rng = np.random.default_rng(0)
    test = [val[j] for j in rng.choice(len(val), min(N_TEST, len(val)), replace=False)]
    Y = np.array([labels[i] for i in test], dtype=np.float32)
    sdx = tc.LABEL_COLS.index("subdural")
    print(f"[robust] {len(test)} test slices; fracs {FRACS}", flush=True)

    rm, rmean, rstd, rWf = load_model("recon")
    sm, smean, sstd, sWf = load_model("sinogram")

    res = {"frac": FRACS, "sino_macro": [], "recon_macro": [],
           "sino_sdh": [], "recon_sdh": []}
    for frac in FRACS:
        Ps, Pr = [], []
        for b in range(0, len(test), BATCH):
            bids = test[b:b + BATCH]
            sinos = [np.load(f"{CACHE}/sinogram/{i}.npy").astype(np.float32) for i in bids]
            wedged = [zero_wedge(s, frac) for s in sinos]
            recons = [iradon(w, theta=THETA, filter_name="ramp", circle=True)
                      for w in wedged]
            Ps.append(infer(sm, wedged, smean, sstd, sWf))
            Pr.append(infer(rm, recons, rmean, rstd, rWf))
        Ps, Pr = np.concatenate(Ps), np.concatenate(Pr)
        res["sino_macro"].append(macro_auc(Y, Ps))
        res["recon_macro"].append(macro_auc(Y, Pr))
        res["sino_sdh"].append(roc_auc_score(Y[:, sdx], Ps[:, sdx]))
        res["recon_sdh"].append(roc_auc_score(Y[:, sdx], Pr[:, sdx]))
        print(f"[robust] frac={frac:.2f}  macro sino={res['sino_macro'][-1]:.4f} "
              f"recon={res['recon_macro'][-1]:.4f} | SDH sino={res['sino_sdh'][-1]:.4f} "
              f"recon={res['recon_sdh'][-1]:.4f}", flush=True)

    json.dump(res, open("angular_robustness.json", "w"), indent=2)
    plot(res)
    print("[robust] done -> angular_robustness.png / .json")


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = np.array(res["frac"]) * 100
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    ax[0].plot(f, res["sino_macro"], "o-", color="#c0392b", lw=2, label="sinogram model")
    ax[0].plot(f, res["recon_macro"], "s-", color="#2c7fb8", lw=2, label="recon model")
    ax[0].set_xlabel("angular wedge removed (% of 180°)")
    ax[0].set_ylabel("macro-AUC")
    ax[0].set_title("Absolute detection vs lost angular data", fontsize=11)
    ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)

    sr = np.array(res["sino_macro"]) / res["sino_macro"][0]
    rr = np.array(res["recon_macro"]) / res["recon_macro"][0]
    ax[1].plot(f, sr, "o-", color="#c0392b", lw=2, label="sinogram")
    ax[1].plot(f, rr, "s-", color="#2c7fb8", lw=2, label="recon")
    ax[1].axhline(1.0, color="k", lw=0.6)
    ax[1].set_xlabel("angular wedge removed (% of 180°)")
    ax[1].set_ylabel("retention  AUC(frac) / AUC(0)")
    ax[1].set_title("Retention — higher = more robust to lost angles", fontsize=11)
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)

    fig.suptitle("Track 1(a): angular-dropout robustness — the 'information spread "
                 "throughout the sinogram' advantage", fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("angular_robustness.png", dpi=150)


if __name__ == "__main__":
    main()
