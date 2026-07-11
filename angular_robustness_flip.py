#!/usr/bin/env python3
"""angular_robustness_flip.py — Tier-1 test for the angular-robustness flip.

Isolates ONE factor: training-time angular augmentation. Two sinogram models,
identical arch (aniso_cfg default) / whiten=off / seed / 200k / 20ep, differ ONLY
by --aug-angular. We ask: does the sinogram model's wedge retention flip with
augmentation alone?  If aug-OFF collapses and aug-ON survives, the "graceful
sinogram degradation" seen in se_200k is TRAINED (H_aug), not intrinsic to the
projection domain (H_intrinsic).

Sinogram-only (the ablation runs have no recon branch); handles the aniso_cfg
architecture that the original angular_robustness.py did not. Same wedge protocol,
same 2000-slice seed-0 test set, so numbers are comparable across experiments.

Usage:
  SINO_CACHE=/mnt/mlcache/subset200k ROBUST_EXP=abl_base python angular_robustness_flip.py
Writes {EXP}_flip.json  with FRACS, sino_macro, sino_sdh, and retention.
"""
import json
import os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import twin_common as tc
from train_twin import apply_whiten   # canonical fft2-layout whitening (matches training)

EXP = os.environ.get("ROBUST_EXP", "abl_base")
DOMAIN = os.environ.get("ROBUST_DOMAIN", "sinogram")   # sinogram | recon_wedge
CACHE = os.environ.get("SINO_CACHE", "/mnt/mlcache/subset200k")
N_TEST = 2000
FRACS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
BATCH = 128
DEV = "cuda" if torch.cuda.is_available() else "cpu"
THETA = np.linspace(0.0, 180.0, 256, endpoint=False)   # for recon iradon
_IRADON = None
if DOMAIN == "recon_wedge":
    from skimage.transform import iradon as _IRADON


def build_from_ckpt(ck):
    """Reconstruct the model from the checkpoint's arch tag. aniso_cfg is rebuilt
    at its DEFAULT config (stem=h, strides 2,2,2,2, res_k=3, no dilation) — the
    Tier-1 runs use those defaults, so no extra knobs need to be stored."""
    arch = ck.get("arch", "smallcnn")
    n = len(tc.LABEL_COLS)
    if arch == "seresnext50":
        import timm
        return timm.create_model("seresnext50_32x4d", pretrained=False,
                                 in_chans=1, num_classes=n)
    if arch == "aniso_cfg":
        return tc.AnisoStemNet(n, stem_spec=tc.ANISO_STEM,
                               stem_strides=(2, 2, 2, 2), res_k=3,
                               res_dilations=(1, 1, 1))
    if arch == "aniso":
        return tc.AnisoStemNet(n)
    if arch == "iso":
        return tc.AnisoStemNet(n, stem_spec=tc.ISO_STEM)
    return tc.SmallCNN(n, k1=ck["K1"], c1=ck["C1"], s1=ck["S1"])


def load_model(exp, domain):
    ck = torch.load(f"{exp}/{domain}/best.pt", map_location="cpu",
                    weights_only=False)
    m = build_from_ckpt(ck).to(DEV)
    m.load_state_dict(ck["model"]); m.eval()
    wp = f"{exp}/{domain}/whiten.npy"
    Wf = (torch.from_numpy(np.load(wp))[None, None].to(DEV)
          if os.path.exists(wp) else None)
    return m, ck["mean"], ck["std"], Wf, ck.get("arch", "?")


def domain_input(wedged_sino):
    """Map a wedged sinogram to the model's input domain. sinogram models read the
    wedged sino directly; recon models read iradon(wedged sino) — the same
    iradon-domain the recon bank was built in, so frac=0 == iradon(full sino)."""
    if DOMAIN == "recon_wedge":
        return _IRADON(wedged_sino, theta=THETA, filter_name="ramp",
                       circle=True).astype(np.float32)
    return wedged_sino


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
    test = [val[j] for j in rng.choice(len(val), min(N_TEST, len(val)),
                                       replace=False)]
    Y = np.array([labels[i] for i in test], dtype=np.float32)
    sdx = tc.LABEL_COLS.index("subdural")

    m, mean, std, Wf, arch = load_model(EXP, DOMAIN)
    print(f"[flip] EXP={EXP} domain={DOMAIN} arch={arch} "
          f"whiten={'on' if Wf is not None else 'off'} "
          f"| {len(test)} test slices; fracs {FRACS}", flush=True)

    res = {"exp": EXP, "domain": DOMAIN, "arch": arch, "whiten": Wf is not None,
           "frac": FRACS, "macro": [], "sdh": []}
    for frac in FRACS:
        Ps = []
        for b in range(0, len(test), BATCH):
            bids = test[b:b + BATCH]
            sinos = [np.load(f"{CACHE}/sinogram/{i}.npy").astype(np.float32)
                     for i in bids]
            imgs = [domain_input(zero_wedge(s, frac)) for s in sinos]
            Ps.append(infer(m, imgs, mean, std, Wf))
        Ps = np.concatenate(Ps)
        res["macro"].append(macro_auc(Y, Ps))
        res["sdh"].append(roc_auc_score(Y[:, sdx], Ps[:, sdx]))
        print(f"[flip] frac={frac:.2f}  macro={res['macro'][-1]:.4f} "
              f"SDH={res['sdh'][-1]:.4f}", flush=True)

    base = res["macro"][0]
    res["retention"] = [a / base for a in res["macro"]]
    out = f"{EXP}_{DOMAIN}_flip.json"
    json.dump(res, open(out, "w"), indent=2)
    print(f"[flip] retention@0.3 = {res['retention'][3]:.3f}  -> {out}", flush=True)


if __name__ == "__main__":
    main()
