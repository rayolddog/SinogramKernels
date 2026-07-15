#!/usr/bin/env python3
"""
train_cached_features.py — train the transformer side on PRECOMPUTED stem
features (cache_stem_features.py), skipping the stem forward entirely. This is
the "learn the front-end once, iterate back-ends cheaply" workflow: with the
5M-param stem out of the loop, epochs are faster, cooler, and lighter on
memory — small enough to iterate architectures on the 4090 or even a laptop.

Equivalence to arm 1: the trainable module is arm 1's exact transformer side
(1x1 adapter -> MaxViT stages 2-3 -> norm -> head), and by default it loads
the SAME tf_init_s<seed>.pt the arms trained from (--init-dir points at the
arms' out dir), so a cached run with matching seeds should reproduce arm 1's
learning curve up to fp16 rounding. Two known, harmless numeric gaps:
(a) cached features carry training-time autocast (fp16) numerics, while arm
1's evaluate() ran the stem in fp32 — val AUC may differ in the last digits;
(b) AMP loss-scale dynamics may diverge slightly. The curves should overlay;
bit-identity is not expected.

Recipe, logging, checkpoints, resume, and thermal policy mirror
train_stem_transfer.py (bs 128, lr 3e-4 + 2-epoch warmup + cosine, wd 1e-4,
grad-clip 1.0, per-epoch history.json with secs/cuda_peak_gb/temps). Off the
Spark, soc_temp() is nan and the cooldown self-disables — on the 4090 there
is no forced 120 s pause.

Usage (Spark, after cache_stem_features.py):
  train_cached_features.py                                # cached arm-1 twin
  train_cached_features.py --n-train 10000 --seed 1       # sweep point
Compare: plot_convergence.py --run arm1=.../arm1_s0 --run cached=.../cached_s0
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import twin_common as tc
from train_twin import soc_temp, pos_weight, evaluate
from train_stem_transfer import SinoMaxViT, tf_init_path, STEM_OUT_CH, STEM_MAP


class FeatureDS(torch.utils.data.Dataset):
    """Cached stem features from disk: (256,32,32) fp16 .npy -> fp32 tensor.
    Features are already standardized+stem-transformed; no further transform."""

    def __init__(self, featdir, ids, labels):
        self.dir = featdir
        self.ids = ids
        self.labels = labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        sid = self.ids[i]
        f = np.load(os.path.join(self.dir, sid + ".npy")).astype(np.float32)
        y = np.asarray(self.labels[sid], dtype=np.float32)
        return torch.from_numpy(f), torch.from_numpy(y)


class FeatureHead(nn.Module):
    """Arm 1's transformer side, detached from the stem: takes the cached
    stem feature map where SinoMaxViT.forward would have produced it."""

    def __init__(self, full):
        super().__init__()
        self.adapter = full.adapter
        self.stages = full.stages
        self.norm = full.norm
        self.head = full.head

    def forward(self, f):
        return self.head(self.norm(self.stages(self.adapter(f))))


def build_head(args, dev, log):
    """Transformer side bit-identical to the arms: construct a full SinoMaxViT
    under the same manual_seed (same RNG consumption order as build_model in
    train_stem_transfer.py), then load/save tf_init_s<seed>.pt exactly as the
    arms do and keep only the four transformer-side modules (the unused stem
    is garbage-collected)."""
    initp = tf_init_path(args.init_dir, args.tf_seed)
    torch.manual_seed(args.tf_seed)
    full = SinoMaxViT(len(tc.LABEL_COLS)).to(dev)
    if os.path.exists(initp):
        saved = torch.load(initp, map_location=dev, weights_only=False)
        for name in ("adapter", "stages", "norm", "head"):
            getattr(full, name).load_state_dict(saved[name])
        log(f"[init] transformer side loaded from {initp} "
            f"(same init as the arms -> comparable curves)")
    else:
        torch.save(dict(adapter=full.adapter.state_dict(),
                        stages=full.stages.state_dict(),
                        norm=full.norm.state_dict(),
                        head=full.head.state_dict()), initp)
        log(f"[init] WARNING: no existing tf_init at {initp}; seeded a fresh "
            f"one (tf_seed={args.tf_seed}). Fine for a new comparison, but "
            f"NOT comparable to arms trained from a different init file.")
    return FeatureHead(full)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features",
                    default=os.path.expanduser("~/sino_work/stem_features"))
    ap.add_argument("--init-dir",
                    default=os.path.expanduser("~/sino_work/stem_transfer"),
                    help="where tf_init_s<seed>.pt lives; default = the arms' "
                         "out dir so cached runs share the arms' init")
    ap.add_argument("--out",
                    default=os.path.expanduser("~/sino_work/stem_transfer"))
    ap.add_argument("--tag", default="cached",
                    help="run-dir prefix; use a distinct tag per back-end "
                         "architecture variant")
    ap.add_argument("--domains", default="sinogram")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0,
                    help="data order / dropout; match arm 1's for the "
                         "reproduce-arm1 check")
    ap.add_argument("--tf-seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=0,
                    help="data-size sweep: subsample train ids to N (0 = all); "
                         "same per-seed subset as train_stem_transfer.py")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--gpu-mem-frac", type=float, default=0.67)
    ap.add_argument("--cooldown", type=int, default=120,
                    help="seconds between epochs (thermal); auto-skipped off "
                         "the Spark (no readable thermal zones)")
    ap.add_argument("--cool-target", type=float, default=70.0)
    ap.add_argument("--split-file",
                    default=os.path.expanduser("~/sino_work/subset_dc/split.json"))
    args = ap.parse_args()

    tag = f"{args.tag}_s{args.seed}"
    if args.n_train:
        tag += f"_n{args.n_train}"
    outdir = os.path.join(args.out, tag)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(args.init_dir, exist_ok=True)
    logpath = os.path.join(outdir, "train.log")

    def log(msg):
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        with open(logpath, "a") as f:
            f.write(line + "\n")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    if dev == "cuda" and args.gpu_mem_frac > 0:
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_frac)

    domain = args.domains.strip()
    featdir = os.path.join(args.features, domain)
    metap = os.path.join(args.features, "meta.json")
    if not os.path.isdir(featdir) or not os.path.exists(metap):
        sys.exit(f"no feature cache at {featdir} — run cache_stem_features.py")
    with open(metap) as f:
        meta = json.load(f)
    if meta.get("shape") != [STEM_OUT_CH, STEM_MAP, STEM_MAP]:
        sys.exit(f"cache shape {meta.get('shape')} != expected "
                 f"{[STEM_OUT_CH, STEM_MAP, STEM_MAP]} — stale cache?")
    log(f"[setup] features={featdir} from stem={meta.get('stem_ckpt')} "
        f"(source val_auc={meta.get('stem_ckpt_val_auc')}) "
        f"mean={meta.get('mean'):.3f} std={meta.get('std'):.3f}")

    labels = tc.load_labels()
    with open(args.split_file) as f:
        sp = json.load(f)
    have = set(labels)
    # subset BEFORE checking the cache, on the same base list with the same
    # rng, so an --n-train run trains on the identical ids as the uncached one
    ids_tr = sorted(set(sp["train"]) & have)
    ids_va = sorted(set(sp["val"]) & have)
    if args.n_train and len(ids_tr) > args.n_train:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(ids_tr), args.n_train, replace=False)
        ids_tr = sorted(ids_tr[j] for j in idx)
    missing = [s for s in ids_tr + ids_va
               if not os.path.exists(os.path.join(featdir, s + ".npy"))]
    if missing:
        sys.exit(f"{len(missing)} split ids have no cached features "
                 f"(e.g. {missing[0]}) — rerun cache_stem_features.py")
    log(f"[setup] train={len(ids_tr)} val={len(ids_va)} bs={args.bs} "
        f"lr={args.lr} wd={args.wd} epochs={args.epochs} "
        f"warmup={args.warmup_epochs} tf_seed={args.tf_seed} seed={args.seed}")

    dl_tr = DataLoader(FeatureDS(featdir, ids_tr, labels),
                       batch_size=args.bs, shuffle=True, num_workers=args.workers,
                       drop_last=True, persistent_workers=args.workers > 0)
    dl_va = DataLoader(FeatureDS(featdir, ids_va, labels),
                       batch_size=args.bs, shuffle=False, num_workers=args.workers,
                       persistent_workers=args.workers > 0)

    model = build_head(args, dev, log)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    n_par = sum(p.numel() for p in model.parameters())
    log(f"[model] trainable transformer-side={n_par/1e6:.2f}M (stem: cached)")

    pw = pos_weight(labels, ids_tr).to(dev)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd)

    def lr_lambda(ep):     # linear warmup, then cosine to 0 (same as the arms)
        if ep < args.warmup_epochs:
            return (ep + 1) / args.warmup_epochs
        t = (ep - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * t))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))

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
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                        sched=sched.state_dict(), scaler=scaler.state_dict(),
                        epoch=ep, best=best, val_auc=auc, cached=True,
                        features=featdir, tf_seed=args.tf_seed,
                        seed=args.seed), path)

    histpath = os.path.join(outdir, "history.json")
    hist = []
    if args.resume and os.path.exists(histpath):
        try:
            with open(histpath) as f:
                hist = json.load(f)
            hist = [h for h in hist if h["epoch"] <= start_ep]
        except Exception:
            hist = []

    for ep in range(start_ep, args.epochs):
        model.train()
        t0 = time.time()
        run = 0.0
        for bi, (x, y) in enumerate(dl_tr):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(dev == "cuda")):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            run += loss.item()
        sched.step()
        auc, _, _ = evaluate(model, dl_va, dev)
        peak = (torch.cuda.max_memory_allocated() / 2**30) if dev == "cuda" else 0.0
        log(f"[{domain}] {tag} epoch {ep+1}/{args.epochs} "
            f"loss={run/(bi+1):.4f} val_macroAUC={auc:.4f} "
            f"({time.time()-t0:.0f}s) lr={opt.param_groups[0]['lr']:.2e} "
            f"cuda_peak={peak:.2f}GB")
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        newbest = (not np.isnan(auc)) and auc > best
        if newbest:
            best = auc
        hist.append({"epoch": ep + 1, "loss": run / (bi + 1),
                     "val_macro_auc": auc, "best_so_far": best,
                     "secs": time.time() - t0, "cached": True,
                     "lr": opt.param_groups[0]["lr"], "cuda_peak_gb": peak})
        with open(histpath, "w") as f:
            json.dump(hist, f, indent=1)
        save(lastp, ep + 1, auc)
        if newbest:
            save(os.path.join(outdir, "best.pt"), ep + 1, auc)
        # DGX Spark thermal cooldown — same policy as the arms; skipped when
        # there are no readable thermal zones (nan => 4090 / Mac).
        if args.cooldown > 0 and not math.isnan(soc_temp()):
            t_hot = soc_temp()
            log(f"[{domain}] thermal cooldown: {t_hot:.1f}C, "
                f"sleeping {args.cooldown}s")
            time.sleep(args.cooldown)
            t_cool = soc_temp()
            extra = 0
            while t_cool > args.cool_target and extra < 10:
                extra += 1
                log(f"[{domain}] still {t_cool:.1f}C "
                    f"(target {args.cool_target:.0f}C); extra minute {extra}/10")
                time.sleep(60)
                t_cool = soc_temp()
            log(f"[{domain}] cooldown done: {t_hot:.1f}C -> {t_cool:.1f}C"
                + (f" (+{extra} extra min)" if extra else ""))
            hist[-1]["temp_pre_cooldown_c"] = t_hot
            hist[-1]["temp_post_cooldown_c"] = t_cool
            hist[-1]["cooldown_extra_min"] = extra
            with open(histpath, "w") as f:
                json.dump(hist, f, indent=1)
            if t_cool > args.cool_target:
                log(f"[{domain}] TERMINATING: still {t_cool:.1f}C after "
                    f"{args.cooldown}s + {extra} extra minutes. Epoch {ep+1} "
                    f"checkpoint is saved; resume with --resume")
                sys.exit(1)
    log(f"[{domain}] DONE {tag} best_val_macroAUC={best:.4f} -> {outdir}")


if __name__ == "__main__":
    main()
