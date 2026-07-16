#!/usr/bin/env python3
"""
train_stem_transfer.py — stem-transferability experiment (EXPERIMENT_stem_transferability.md).

Does the anisotropic sinogram stem trained inside a ResNet (arm0, train_twin.py
--arch aniso_cfg --stem tall --resdil --resk 3) transfer as the front-end of a
TRANSFORMER? Three arms, identical except for the stem's initialization:

  --arm 1   pretrained stem, FROZEN (BN in eval mode; excluded from optimizer)
  --arm 2   pretrained stem, frozen for --unfreeze-epoch epochs, then fine-tuned
            at lr * --stem-lr-mult
  --arm 3   random-init stem (same architecture), trained from scratch (control)

Grafted architecture (SinoMaxViT):

  sinogram [1,256,256]
    -> arm0 stem (ANISO2 kernels, widths 48/96/160/256) at strides (2,2,2,1)
       -> [256, 32, 32]          # the codebase's existing --fulres variant;
                                 # conv weights carry no stride, so the arm0
                                 # checkpoint (trained at 2,2,2,2) loads as-is
    -> 1x1 conv adapter 256->128  # learnable; SAME fixed init in all arms
    -> timm maxvit_tiny stages 2+3 (img_size=256) -> head (6 logits)

Why this graft: at 256 input, MaxViT stage 2 expects a stride-8, 128-ch map
(32x32) — exactly what the stem+adapter produce — so the pretrained stem
replaces MaxViT's conv stem and stages 0-1, keeping 7 of 11 attention blocks.
MaxViT attention uses relative position biases, so no explicit positional
encoding is added at the seam.

The transformer side (adapter + stages + head) is initialized ONCE per
--tf-seed, saved to tf_init_s<seed>.pt, and loaded verbatim by every arm: the
stem is the only variable. The transfer signal is the SHAPE of the val-AUC
curve (early epochs / small N), recorded per-epoch in history.json.

Usage:
  train_stem_transfer.py --arm 1 --stem-ckpt .../arm0/sinogram/best.pt \
      --split-file .../split.json --out ~/sino_work/stem_transfer
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


def macro_auc(P, T):
    """Same per-class macro AUC as train_twin.evaluate, for train-side preds."""
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return float("nan")
    aucs = [roc_auc_score(T[:, c], P[:, c]) for c in range(T.shape[1])
            if T[:, c].min() != T[:, c].max()]
    return float(np.mean(aucs)) if aucs else float("nan")

# arm0's exact architecture flags (run_arm0.sh); the stem checkpoint is loaded
# through a full AnisoStemNet with strict=True so a config drift fails loudly
# instead of silently training a different model.
ARM0_STEM_SPEC = tc.ANISO2_STEM
ARM0_RES_K = 3
ARM0_RES_DIL = (1, 2, 4)
STEM_STRIDES = (2, 2, 2, 1)      # deploy at --fulres so the output is ~32x32
STEM_OUT_CH = 256
STEM_MAP = 32                    # transformer-side grid (must divide by window 8)
ADAPTER_OUT_CH = 128             # maxvit_tiny stage-2 input channels


class SinoMaxViT(nn.Module):
    """arm0 stem -> 1x1 adapter -> MaxViT stages 2-3 -> head."""

    def __init__(self, n_out=6, img_size=256, tf_variant="maxvit_tiny_tf_224"):
        super().__init__()
        import timm
        full = tc.AnisoStemNet(n_out, stem_spec=ARM0_STEM_SPEC,
                               stem_strides=STEM_STRIDES, res_k=ARM0_RES_K,
                               res_dilations=ARM0_RES_DIL)
        self.stem = full.stem
        self.adapter = nn.Conv2d(STEM_OUT_CH, ADAPTER_OUT_CH, 1, bias=False)
        mv = timm.create_model(tf_variant, pretrained=False, in_chans=1,
                               num_classes=n_out, img_size=img_size)
        self.stages = nn.Sequential(mv.stages[2], mv.stages[3])
        self.norm = mv.norm
        self.head = mv.head

    def forward(self, x):
        f = self.stem(x)
        # even-kernel padding makes the stem map 34x34; center-crop to 32x32 so
        # MaxViT's window partitioning divides evenly. The trimmed border ring
        # is padding-dominated, and the crop is identical across arms.
        h, w = f.shape[-2:]
        f = f[..., (h - STEM_MAP) // 2:(h - STEM_MAP) // 2 + STEM_MAP,
              (w - STEM_MAP) // 2:(w - STEM_MAP) // 2 + STEM_MAP]
        f = self.adapter(f)
        return self.head(self.norm(self.stages(f)))

    def load_stem_from(self, ckpt_path, dev):
        """Load arm0 stem weights via a full AnisoStemNet (strict) and return
        the checkpoint's (mean, std, val_auc) for provenance."""
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        probe = tc.AnisoStemNet(6, stem_spec=ARM0_STEM_SPEC,
                                stem_strides=STEM_STRIDES, res_k=ARM0_RES_K,
                                res_dilations=ARM0_RES_DIL)
        probe.load_state_dict(ck["model"], strict=True)
        self.stem.load_state_dict(probe.stem.state_dict())
        self.stem.to(dev)
        return ck.get("mean"), ck.get("std"), ck.get("val_auc")


def tf_init_path(outroot, tf_seed):
    return os.path.join(outroot, f"tf_init_s{tf_seed}.pt")


def build_model(args, dev, log):
    """Build SinoMaxViT with the transformer side bit-identical across arms."""
    initp = tf_init_path(args.out, args.tf_seed)
    torch.manual_seed(args.tf_seed)
    model = SinoMaxViT(len(tc.LABEL_COLS)).to(dev)
    tf_state = dict(adapter=model.adapter.state_dict(),
                    stages=model.stages.state_dict(),
                    norm=model.norm.state_dict(),
                    head=model.head.state_dict())
    if os.path.exists(initp):
        saved = torch.load(initp, map_location=dev, weights_only=False)
        for name in ("adapter", "stages", "norm", "head"):
            getattr(model, name).load_state_dict(saved[name])
        log(f"[init] transformer side loaded from {initp}")
    else:
        torch.save(tf_state, initp)
        log(f"[init] transformer side seeded (tf_seed={args.tf_seed}) "
            f"and saved to {initp}")

    stem_meta = {}
    if args.arm in (1, 2):
        m, s, auc = model.load_stem_from(args.stem_ckpt, dev)
        stem_meta = dict(stem_ckpt=args.stem_ckpt, stem_ckpt_val_auc=auc)
        log(f"[init] arm {args.arm}: pretrained stem from {args.stem_ckpt} "
            f"(source val_auc={auc:.4f})")
    else:
        # arm 3 control: same architecture, random weights, own seed
        torch.manual_seed(args.tf_seed + 10_000 + args.seed)
        for mod in model.stem.modules():
            if isinstance(mod, nn.Conv2d):
                nn.init.kaiming_normal_(mod.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(mod, nn.BatchNorm2d):
                nn.init.ones_(mod.weight)
                nn.init.zeros_(mod.bias)
        log(f"[init] arm 3: random stem (control), same architecture")
    return model, stem_meta


def set_stem_frozen(model, frozen):
    for p in model.stem.parameters():
        p.requires_grad_(not frozen)


def stem_is_frozen(args, ep):
    if args.arm == 1:
        return True
    if args.arm == 2:
        return ep < args.unfreeze_epoch
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--stem-ckpt",
                    default=os.path.expanduser(
                        "~/sino_work/arm0_resnet_sino/sinogram/best.pt"))
    ap.add_argument("--out",
                    default=os.path.expanduser("~/sino_work/stem_transfer"))
    ap.add_argument("--domains", default="sinogram")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=128)   # must match arm0 (see spec)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=int, default=2,
                    help="linear LR warmup (from-scratch attention stability); "
                         "identical across arms")
    ap.add_argument("--unfreeze-epoch", type=int, default=3,
                    help="arm 2: epoch at which the stem unfreezes")
    ap.add_argument("--stem-lr-mult", type=float, default=0.1,
                    help="arm 2: stem LR = lr * this after unfreezing")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0,
                    help="data order / dropout / arm-3 stem init")
    ap.add_argument("--tf-seed", type=int, default=0,
                    help="transformer-side init; all arms of one comparison "
                         "MUST share it")
    ap.add_argument("--n-train", type=int, default=0,
                    help="data-size sweep: subsample train ids to N (0 = all)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--workers", type=int, default=2)          # DGX Spark thermal
    ap.add_argument("--gpu-mem-frac", type=float, default=0.67)
    ap.add_argument("--cooldown", type=int, default=120,
                    help="seconds to sleep between epochs (thermal); 0 disables")
    ap.add_argument("--cool-target", type=float, default=70.0)
    ap.add_argument("--split-file",
                    default=os.path.expanduser("~/sino_work/subset_dc/split.json"))
    args = ap.parse_args()

    tag = f"arm{args.arm}_s{args.seed}"
    if args.n_train:
        tag += f"_n{args.n_train}"
    outdir = os.path.join(args.out, tag)
    os.makedirs(outdir, exist_ok=True)
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
        total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
        log(f"[setup] CUDA memory capped at {args.gpu_mem_frac:.2f} of "
            f"{total_gb:.0f} GB (over-allocation raises OOM, not a freeze)")

    labels = tc.load_labels()
    with open(args.split_file) as f:
        sp = json.load(f)
    have = set(labels)
    ids_tr = sorted(set(sp["train"]) & have)
    ids_va = sorted(set(sp["val"]) & have)
    if args.n_train and len(ids_tr) > args.n_train:
        rng = np.random.default_rng(args.seed)   # deterministic per-seed subset
        idx = rng.choice(len(ids_tr), args.n_train, replace=False)
        ids_tr = sorted(ids_tr[j] for j in idx)
    log(f"[setup] arm={args.arm} train={len(ids_tr)} val={len(ids_va)} "
        f"bs={args.bs} lr={args.lr} wd={args.wd} epochs={args.epochs} "
        f"warmup={args.warmup_epochs} tf_seed={args.tf_seed} seed={args.seed}")

    domain = args.domains.strip()
    # domain_stats is deterministic (seed 0 over the same ids), so all arms —
    # and arm0 itself — standardize identically.
    mean, std = tc.domain_stats(domain, ids_tr)
    log(f"[{domain}] stats mean={mean:.3f} std={std:.3f}")

    dl_tr = DataLoader(tc.DomainDS(domain, ids_tr, labels, mean, std),
                       batch_size=args.bs, shuffle=True, num_workers=args.workers,
                       drop_last=True, persistent_workers=args.workers > 0)
    dl_va = DataLoader(tc.DomainDS(domain, ids_va, labels, mean, std),
                       batch_size=args.bs, shuffle=False, num_workers=args.workers,
                       persistent_workers=args.workers > 0)

    model, stem_meta = build_model(args, dev, log)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    n_stem = sum(p.numel() for p in model.stem.parameters())
    n_tf = sum(p.numel() for p in model.parameters()) - n_stem
    log(f"[model] stem={n_stem/1e6:.2f}M  transformer-side={n_tf/1e6:.2f}M")

    pw = pos_weight(labels, ids_tr).to(dev)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)

    # Two param groups so arm 2 can fine-tune the stem at a reduced LR. Arm 1's
    # stem is permanently frozen and never enters the optimizer.
    tf_params = [p for n, p in model.named_parameters()
                 if not n.startswith("stem.")]
    groups = [dict(params=tf_params, lr=args.lr)]
    if args.arm == 2:
        groups.append(dict(params=list(model.stem.parameters()),
                           lr=args.lr * args.stem_lr_mult))
    elif args.arm == 3:
        groups.append(dict(params=list(model.stem.parameters()), lr=args.lr))
    opt = torch.optim.AdamW(groups, weight_decay=args.wd)

    def lr_lambda(ep):     # linear warmup, then cosine to 0
        if ep < args.warmup_epochs:
            return (ep + 1) / args.warmup_epochs
        t = (ep - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * t))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda")

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
                        epoch=ep, best=best, val_auc=auc, mean=mean, std=std,
                        arm=args.arm, tf_seed=args.tf_seed, seed=args.seed,
                        **stem_meta), path)

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
        frozen = stem_is_frozen(args, ep)
        set_stem_frozen(model, frozen)
        if args.arm == 2 and ep == args.unfreeze_epoch:
            log(f"[{domain}] arm 2: stem UNFROZEN at epoch {ep} "
                f"(stem lr = {args.lr * args.stem_lr_mult:g} * cosine)")
        model.train()
        if frozen:
            model.stem.eval()      # freeze BN running stats too, not just weights
        t0 = time.time()
        run = 0.0
        trP, trT = [], []   # train-side preds/targets for the memorization gap
        for bi, (x, y) in enumerate(dl_tr):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logit = model(x)
                loss = crit(logit, y)
            trP.append(torch.sigmoid(logit.detach()).float().cpu().numpy())
            trT.append(y.cpu().numpy())
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            run += loss.item()
        sched.step()
        # Train macro-AUC from the predictions the loop already made. Caveat:
        # the model evolves within the epoch, so this slightly UNDERSTATES the
        # end-of-epoch train fit — fine for locating the train/val divergence.
        tr_auc = macro_auc(np.concatenate(trP), np.concatenate(trT))
        auc, P, T = evaluate(model, dl_va, dev)
        # Val loss under the same pos-weighted BCE as training (from post-sigmoid
        # probs; identical to BCEWithLogitsLoss(pos_weight=pw) mean reduction).
        Pc = np.clip(P, 1e-7, 1.0 - 1e-7)
        pw_np = pw.cpu().numpy()
        val_loss = float((-(pw_np * T * np.log(Pc)
                            + (1.0 - T) * np.log(1.0 - Pc))).mean())
        peak = (torch.cuda.max_memory_allocated() / 2**30) if dev == "cuda" else 0.0
        reserved = (torch.cuda.max_memory_reserved() / 2**30) if dev == "cuda" else 0.0
        log(f"[{domain}] arm{args.arm} epoch {ep+1}/{args.epochs} "
            f"loss={run/(bi+1):.4f} val_loss={val_loss:.4f} "
            f"trainAUC={tr_auc:.4f} val_macroAUC={auc:.4f} "
            f"({time.time()-t0:.0f}s) stem_frozen={frozen} "
            f"lr={opt.param_groups[0]['lr']:.2e} "
            f"cuda_peak={peak:.2f}GB reserved={reserved:.2f}GB")
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        newbest = (not np.isnan(auc)) and auc > best
        if newbest:
            best = auc
        hist.append({"epoch": ep + 1, "loss": run / (bi + 1),
                     "val_loss": val_loss, "train_macro_auc": tr_auc,
                     "val_macro_auc": auc, "best_so_far": best,
                     "secs": time.time() - t0, "stem_frozen": frozen,
                     "lr": opt.param_groups[0]["lr"], "cuda_peak_gb": peak,
                     "cuda_reserved_gb": reserved})
        with open(histpath, "w") as f:
            json.dump(hist, f, indent=1)
        save(lastp, ep + 1, auc)
        if newbest:
            save(os.path.join(outdir, "best.pt"), ep + 1, auc)
        # DGX Spark thermal cooldown — same policy as train_twin.py.
        if args.cooldown > 0:
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
                    f"checkpoint is saved; retry when room temperature drops "
                    f"below 85F (resume with --resume)")
                sys.exit(1)
    log(f"[{domain}] DONE arm{args.arm} best_val_macroAUC={best:.4f} -> {outdir}")


if __name__ == "__main__":
    main()
