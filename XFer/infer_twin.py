#!/usr/bin/env python3
"""
infer_twin.py — run the REAL trained twin checkpoints (twin_common.AnisoStemNet
et al.) over the motion arms produced by motion_experiment.py, and write a
results.csv that motion_experiment.py's `stats` consumes unchanged.

Why this exists: motion_experiment.py's own `infer` was drafted against the local
model.py stub (SinogramResNet, input BatchNorm, raw-array input). The production
models are different, and — crucially — AnisoStemNet has NO input normalization
layer: standardization happens in the DATASET as (arr - mean)/std, plus an
optional spectral whitening filter applied in the eval loop. This adapter
reproduces train_twin.evaluate()'s exact preprocessing so the AUCs are faithful.

It reads everything it needs FROM the checkpoints, which train_twin.save() stamps
with mean/std/arch:
    best.pt = dict(model=state_dict, mean=..., std=..., arch=..., K1,C1,S1, ...)
Whitening, if the run used --whiten, is saved separately as `whiten.npy` in the
same run directory; this adapter auto-loads it if present (pass --no-whiten to
force it off, or --whiten-recon/--whiten-sino to point at a specific file).

Run (on the machine with the checkpoints + motion arms):
    python infer_twin.py \
        --arms ./motion_arms --cohort cohort.csv \
        --ckpt-recon ~/sino_work/.../recon/best.pt \
        --ckpt-sino  ~/sino_work/.../sinogram/best.pt \
        --severities 0,2,4,8 --out results.csv
    python motion_experiment.py stats --results results.csv --cohort cohort.csv

Contract with motion_experiment.py:
  * arms layout   : <arms>/sino_s{P}/ID.npy , <arms>/recon_s{P}/ID.npy
  * results.csv   : columns  id,domain,severity,p_<LABEL_COLS...>
                    domain in {"sino","recon"}  (what `stats` keys on)
  * the recon arm is iradon(...) in ~HU+intercept units, matching the scale the
    recon model trained on; the recon checkpoint's own mean/std standardize it.
    The CLEAN recon arm is iradon(clean sino) (the round-trip control), so the
    train-on-originals/test-on-reprojections domain shift is identical in both
    columns and cancels in the within-domain clean-minus-motion difference.

Deps: torch, numpy (+ timm only if a checkpoint's arch == 'seresnext50').
"""
import argparse
import csv
import os

import numpy as np
import torch

import twin_common as tc
from motion_experiment import read_cohort

LABEL_COLS = tc.LABEL_COLS


# ---- rebuild the exact architecture from the saved arch tag ----------------
def build_arch(arch, ck):
    """Reconstruct the model the checkpoint was trained with. The stem/dilation
    recipes mirror train_twin.py's --arch branches verbatim; keep them in sync if
    that file changes."""
    n = len(LABEL_COLS)
    if arch == "aniso":
        return tc.AnisoStemNet(n)
    if arch == "iso":
        return tc.AnisoStemNet(n, stem_spec=tc.ISO_STEM)
    if arch == "aniso2":
        return tc.AnisoStemNet(n, stem_spec=tc.ANISO2_STEM,
                               stem_strides=(2, 2, 2, 1), res_k=5,
                               res_dilations=(1, 2, 4))
    if arch == "aniso3":
        return tc.AnisoStemNet(n, stem_spec=tc.ANISO3_STEM,
                               stem_strides=(2, 2, 2, 2), res_k=3,
                               res_dilations=(2, 4, 8))
    if arch == "seresnext50":
        import timm
        return timm.create_model("seresnext50_32x4d", pretrained=False,
                                 in_chans=1, num_classes=n)
    if arch == "smallcnn":
        return tc.SmallCNN(n, k1=ck.get("K1", tc.K1), c1=ck.get("C1", tc.C1),
                           s1=ck.get("S1", 2))
    if arch == "aniso_cfg":
        # The ablation toggles (stem/fulres/resdil/resk) are NOT stored in the
        # checkpoint, so the exact variant can't be recovered from best.pt alone.
        # Default to the plain aniso stem and let load_state_dict verify by shape.
        print(f"[infer_twin] WARNING: arch='aniso_cfg' toggles not saved; "
              f"assuming ANISO_STEM. If load_state_dict raises a shape/key "
              f"mismatch, pass the exact stem/strides/dilations used for this run.")
        return tc.AnisoStemNet(n)
    raise ValueError(f"unknown arch tag in checkpoint: {arch!r}")


def load_twin(ckpt, device, whiten_path=None, use_whiten=True):
    """Load a production checkpoint. Returns (model, mean, std, Wf) where Wf is the
    whitening filter tensor (1,1,H,W) or None. mean/std/arch come from the
    checkpoint dict; whiten.npy is looked up next to best.pt unless overridden."""
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "model" not in ck:
        raise KeyError(f"{ckpt}: expected a train_twin checkpoint with a 'model' "
                       f"key; got keys {list(ck)[:8]}")
    arch = ck.get("arch", "aniso")
    mean, std = float(ck["mean"]), float(ck["std"])
    print(f"[infer_twin] {os.path.basename(os.path.dirname(ckpt))}/"
          f"{os.path.basename(ckpt)}: arch={arch} mean={mean:.3f} std={std:.3f} "
          f"(best val_auc={ck.get('val_auc')})")

    model = build_arch(arch, ck)
    sd = {k.replace("module.", "", 1): v for k, v in ck["model"].items()}
    model.load_state_dict(sd)                     # raises loudly on any mismatch
    model = model.to(device).eval()

    Wf = None
    if use_whiten:
        wp = whiten_path or os.path.join(os.path.dirname(ckpt), "whiten.npy")
        if os.path.exists(wp):
            Wnp = np.load(wp)
            Wf = torch.from_numpy(Wnp)[None, None].to(device)
            print(f"[infer_twin]   whitening ON  <- {wp} "
                  f"(gain std {float(Wnp.std()):.3f})")
        elif whiten_path:
            raise FileNotFoundError(f"--whiten path not found: {wp}")
        else:
            print("[infer_twin]   whitening OFF (no whiten.npy beside checkpoint)")
    return model, mean, std, Wf


def apply_whiten(x, Wf):
    """Identical to train_twin.apply_whiten: spectral whiten then per-image
    re-standardize. No-op if Wf is None."""
    if Wf is None:
        return x
    xw = torch.fft.ifft2(torch.fft.fft2(x) * Wf).real
    m = xw.mean(dim=(2, 3), keepdim=True)
    s = xw.std(dim=(2, 3), keepdim=True) + 1e-6
    return (xw - m) / s


# ---- inference over the motion arms ----------------------------------------
def infer_arm(model, mean, std, Wf, arm_dir, ids, device, batch):
    """Return (N, 6) sigmoid-probability matrix, aligned to `ids`. Preprocessing
    matches DomainDS.__getitem__ (standardize) + evaluate (whiten)."""
    probs = np.zeros((len(ids), len(LABEL_COLS)), dtype=np.float32)
    with torch.no_grad():
        for b0 in range(0, len(ids), batch):
            chunk = ids[b0:b0 + batch]
            arrs = []
            for s in chunk:
                a = np.load(os.path.join(arm_dir, s + ".npy")).astype(np.float32)
                arrs.append((a - mean) / std)               # DomainDS standardize
            x = torch.from_numpy(np.stack(arrs))[:, None].to(device)  # (B,1,H,W)
            x = apply_whiten(x, Wf)                          # evaluate() whiten
            probs[b0:b0 + len(chunk)] = torch.sigmoid(model(x)).cpu().numpy()
    return probs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="./motion_arms")
    ap.add_argument("--cohort", default="cohort.csv")
    ap.add_argument("--ckpt-recon", required=True)
    ap.add_argument("--ckpt-sino", required=True)
    ap.add_argument("--severities", default="0,2,4,8")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--no-whiten", action="store_true",
                    help="force whitening OFF for both models")
    ap.add_argument("--whiten-recon", help="explicit whiten.npy for the recon model")
    ap.add_argument("--whiten-sino", help="explicit whiten.npy for the sino model")
    a = ap.parse_args()

    sev = [int(x) for x in a.severities.split(",")]
    device = a.device or ("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[infer_twin] device={device}")
    ids, _ = read_cohort(a.cohort)

    use_wh = not a.no_whiten
    m_recon = load_twin(a.ckpt_recon, device, a.whiten_recon, use_wh)
    m_sino = load_twin(a.ckpt_sino, device, a.whiten_sino, use_wh)

    # (results-domain-key, arm-subdir-template, loaded-model-tuple)
    plan = [("sino", "sino_s{p}", m_sino), ("recon", "recon_s{p}", m_recon)]

    rows = []
    for dkey, sub, (model, mean, std, Wf) in plan:
        for p in sev:
            arm_dir = os.path.join(a.arms, sub.format(p=p))
            if not os.path.isdir(arm_dir):
                raise FileNotFoundError(f"missing arm dir {arm_dir} — run "
                                        f"`motion_experiment.py inject` first")
            probs = infer_arm(model, mean, std, Wf, arm_dir, ids, device, a.batch)
            for i, s in enumerate(ids):
                rows.append([s, dkey, p] + probs[i].tolist())
            print(f"[infer_twin] {dkey} severity {p}: {len(ids)} images done")

    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "domain", "severity"] + [f"p_{c}" for c in LABEL_COLS])
        w.writerows(rows)
    print(f"[infer_twin] wrote {len(rows)} rows -> {a.out}\n"
          f"[infer_twin] next: python motion_experiment.py stats "
          f"--results {a.out} --cohort {a.cohort}")


if __name__ == "__main__":
    main()
