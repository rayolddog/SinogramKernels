#!/usr/bin/env python3
"""
train_twin.py — train the SAME CNN on recon and on sinogram (sequentially), with
identical init/hyperparameters, and save each conv1 kernel bank for comparison.

  train_twin.py [--epochs 12] [--bs 128] [--out kernel_exp] [--domains recon,sinogram]

Both domains share the deterministic id split and identical weight init (seed 0),
so any layer-1 difference is attributable to the DOMAIN, not to initialization.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import twin_common as tc

try:
    from sklearn.metrics import roc_auc_score
    HAVE_SK = True
except Exception:
    HAVE_SK = False


def pos_weight(labels, ids, cap=50.0):
    Y = np.array([labels[i] for i in ids], dtype=np.float32)
    p = Y.mean(0)
    return torch.tensor(np.minimum((1 - p) / np.maximum(p, 1e-6), cap),
                        dtype=torch.float32)


def apply_whiten(x, Wf):
    """Batched spectral whitening on GPU, then per-image re-standardization.
    Wf is a real (1,1,H,W) filter in unshifted fft2 layout; no-op if None."""
    if Wf is None:
        return x
    xw = torch.fft.ifft2(torch.fft.fft2(x) * Wf).real
    m = xw.mean(dim=(2, 3), keepdim=True)
    s = xw.std(dim=(2, 3), keepdim=True) + 1e-6
    return (xw - m) / s


def evaluate(model, loader, dev, Wf=None):
    model.eval()
    P, T = [], []
    with torch.no_grad():
        for x, y in loader:
            x = apply_whiten(x.to(dev, non_blocking=True), Wf)
            logit = model(x)
            P.append(torch.sigmoid(logit).cpu().numpy())
            T.append(y.numpy())
    P, T = np.concatenate(P), np.concatenate(T)
    if not HAVE_SK:
        return float("nan"), P, T
    aucs = []
    for c in range(T.shape[1]):
        if T[:, c].min() != T[:, c].max():
            aucs.append(roc_auc_score(T[:, c], P[:, c]))
    return float(np.mean(aucs)) if aucs else float("nan"), P, T


def train_domain(domain, ids_tr, ids_va, labels, args, dev, log):
    mean, std = tc.domain_stats(domain, ids_tr)
    log(f"[{domain}] stats mean={mean:.3f} std={std:.3f}")

    outdir = os.path.join(args.out, domain)
    os.makedirs(outdir, exist_ok=True)
    Wf = None
    if args.whiten:
        Wnp = tc.whitening_filter(domain, ids_tr, mean, std)
        np.save(os.path.join(outdir, "whiten.npy"), Wnp)
        Wf = torch.from_numpy(Wnp)[None, None].to(dev)
        log(f"[{domain}] spectral whitening ON (gain std {Wnp.std():.3f})")

    aug = None
    if args.aug_angular and domain == "sinogram":
        aug = tc.angular_dropout
        log(f"[{domain}] angular-dropout augmentation ON (p=0.5, wedge≤40%)")
    if args.aug_wedge_recon and domain == "recon_wedge":
        aug = tc.recon_wedge_pick
        log(f"[{domain}] recon-wedge augmentation ON (p=0.5, iradon bank)")
    dl_tr = DataLoader(tc.DomainDS(domain, ids_tr, labels, mean, std, augment=aug),
                       batch_size=args.bs, shuffle=True, num_workers=8,
                       pin_memory=True, drop_last=True, persistent_workers=True)
    dl_va = DataLoader(tc.DomainDS(domain, ids_va, labels, mean, std),
                       batch_size=args.bs, shuffle=False, num_workers=8,
                       pin_memory=True, persistent_workers=True)

    torch.manual_seed(args.seed)              # varied for multi-seed noise-floor test
    np.random.seed(args.seed)
    if args.arch == "seresnext50":
        import timm
        model = timm.create_model("seresnext50_32x4d", pretrained=False,
                                  in_chans=1, num_classes=len(tc.LABEL_COLS)).to(dev)
    elif args.arch == "aniso":
        model = tc.AnisoStemNet(len(tc.LABEL_COLS)).to(dev)
    elif args.arch == "iso":
        model = tc.AnisoStemNet(len(tc.LABEL_COLS), stem_spec=tc.ISO_STEM).to(dev)
    elif args.arch == "aniso2":
        # taller-curve stem + less downsampling (last stride 1) + dilated 5x5 body
        model = tc.AnisoStemNet(len(tc.LABEL_COLS), stem_spec=tc.ANISO2_STEM,
                                stem_strides=(2, 2, 2, 1), res_k=5,
                                res_dilations=(1, 2, 4)).to(dev)
    elif args.arch == "aniso3":
        # bigger stem (more horizontal trace) + wider-RF dilated body (RF 9,17,33)
        model = tc.AnisoStemNet(len(tc.LABEL_COLS), stem_spec=tc.ANISO3_STEM,
                                stem_strides=(2, 2, 2, 2), res_k=3,
                                res_dilations=(2, 4, 8)).to(dev)
    elif args.arch == "aniso_cfg":
        # each modification toggled independently, for the ablation
        stem = tc.ANISO2_STEM if args.stem == "tall" else tc.ANISO_STEM
        strides = (2, 2, 2, 1) if args.fulres else (2, 2, 2, 2)
        dils = (1, 2, 4) if args.resdil else (1, 1, 1)
        model = tc.AnisoStemNet(len(tc.LABEL_COLS), stem_spec=stem,
                                stem_strides=strides, res_k=args.resk,
                                res_dilations=dils).to(dev)
    else:
        model = tc.SmallCNN(len(tc.LABEL_COLS), k1=args.k1, c1=args.c1, s1=args.s1,
                            dropout=args.dropout).to(dev)

    pw = pos_weight(labels, ids_tr).to(dev)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    # --- interrupt/resume: last.pt carries full training state ---
    start_ep, best = 0, -1.0
    lastp = os.path.join(outdir, "last.pt")
    if args.resume and os.path.exists(lastp):
        ck = torch.load(lastp, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start_ep, best = ck["epoch"], ck.get("best", -1.0)
        if start_ep >= args.epochs:
            log(f"[{domain}] already complete at epoch {start_ep}; skipping")
            return
        log(f"[{domain}] RESUMED from epoch {start_ep} (best={best:.4f})")

    def save(path, ep, auc):
        st = dict(model=model.state_dict(), opt=opt.state_dict(),
                  sched=sched.state_dict(), scaler=scaler.state_dict(),
                  epoch=ep, best=best, val_auc=auc, mean=mean, std=std,
                  arch=args.arch, K1=args.k1, C1=args.c1, S1=args.s1)
        if args.arch == "smallcnn":
            st["conv1"] = model.conv1.weight.detach().cpu().numpy()
            st["conv1_bias"] = model.conv1.bias.detach().cpu().numpy()
        torch.save(st, path)

    for ep in range(start_ep, args.epochs):
        model.train()
        t0 = time.time()
        run = 0.0
        for bi, (x, y) in enumerate(dl_tr):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            x = apply_whiten(x, Wf)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.item()
        sched.step()
        auc, _, _ = evaluate(model, dl_va, dev, Wf)
        log(f"[{domain}] epoch {ep+1}/{args.epochs} loss={run/(bi+1):.4f} "
            f"val_macroAUC={auc:.4f} ({time.time()-t0:.0f}s)")
        newbest = (not np.isnan(auc)) and auc > best
        if newbest:
            best = auc
        save(lastp, ep + 1, auc)              # resume point after this epoch
        if newbest:
            save(os.path.join(outdir, "best.pt"), ep + 1, auc)
    if args.arch == "smallcnn":
        np.save(os.path.join(outdir, "conv1.npy"),
                model.conv1.weight.detach().cpu().numpy())
    log(f"[{domain}] DONE best_val_macroAUC={best:.4f}  -> {outdir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--out", default="kernel_exp")
    ap.add_argument("--domains", default="recon,sinogram")
    ap.add_argument("--k1", type=int, default=tc.K1)
    ap.add_argument("--c1", type=int, default=tc.C1)
    ap.add_argument("--s1", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--whiten", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--max-train", type=int, default=0)   # 0 = use all
    ap.add_argument("--arch", default="smallcnn",
                    choices=["smallcnn", "seresnext50", "aniso", "iso", "aniso2",
                             "aniso_cfg", "aniso3"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--aug-angular", action="store_true")   # sinogram wedge dropout
    ap.add_argument("--aug-wedge-recon", action="store_true")  # recon iradon-wedge bank
    # aniso_cfg ablation knobs
    ap.add_argument("--stem", default="h", choices=["h", "tall"])
    ap.add_argument("--fulres", action="store_true")        # last stem stride 1
    ap.add_argument("--resk", type=int, default=3)          # body kernel size
    ap.add_argument("--resdil", action="store_true")        # body dilations 1,2,4
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    logpath = os.path.join(args.out, "train.log")

    def log(msg):
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        with open(logpath, "a") as f:
            f.write(line + "\n")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True     # fixed 256x256 input -> autotune
    labels = tc.load_labels()
    ids = sorted(labels)
    ids_tr, ids_va = tc.split_ids(ids)
    if args.max_train and len(ids_tr) > args.max_train:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(ids_tr), args.max_train, replace=False)
        ids_tr = sorted(ids_tr[j] for j in idx)
    log(f"[setup] {len(ids)} slices  train={len(ids_tr)} val={len(ids_va)}  "
        f"dev={dev}  arch={args.arch} resume={args.resume} lr={args.lr} "
        f"wd={args.wd} whiten={args.whiten} epochs={args.epochs} sklearn={HAVE_SK}")

    for domain in args.domains.split(","):
        train_domain(domain.strip(), ids_tr, ids_va, labels, args, dev, log)
    log("[all] complete")


if __name__ == "__main__":
    main()
