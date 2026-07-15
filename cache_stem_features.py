#!/usr/bin/env python3
"""
cache_stem_features.py — precompute the frozen arm0 stem's features once, so
back-end architectures can train on cached features (no stem forward per epoch:
less compute, less heat, lower peak memory — the "learn the front-end once,
iterate back-ends cheaply" workflow from EXPERIMENT_stem_transferability.md).

Faithfulness: the arm-1 recipe applies NO augmentation and freezes the stem
with BN in eval mode, so each slice has exactly ONE deterministic feature
tensor. This script replicates the training-time input path bit-for-bit:
same DomainDS loading, same domain_stats(ids_tr) standardization (full train
split — NOT the checkpoint's stats), same autocast forward, same 34->32
center-crop. Output: one fp16 .npy of shape (256, 32, 32) per slice id
(~512 KB), mirroring the input-cache layout, plus meta.json for provenance.

The stem is loaded through a full AnisoStemNet with strict=True (same guard
as train_stem_transfer.py) so config drift fails loudly.

Idempotent/resumable: existing feature files are skipped; writes are atomic
(tmp + rename), so an interrupted run leaves no partial files — just rerun.
Thermal policy mirrors training: periodic SoC-temp checks pause the sweep and
terminate cleanly if the box won't cool (soc_temp() is nan off the Spark, so
the checks self-disable on the 4090/Mac).

NOTE for the data-size sweep: cached features embed FULL-train-split stats.
Cached n-train-subset runs are internally consistent (all arms see identical
features) but not bit-identical to uncached runs that recompute stats on the
subset. Record which regime a run used.

Usage (Spark, after the arm chain):
  cache_stem_features.py                      # cache train+val, then verify 16
  cache_stem_features.py --verify-only 32     # re-check an existing cache
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import twin_common as tc
from train_twin import soc_temp
from train_stem_transfer import (ARM0_STEM_SPEC, ARM0_RES_K, ARM0_RES_DIL,
                                 STEM_STRIDES, STEM_MAP, STEM_OUT_CH)


def build_stem(ckpt_path, dev):
    """arm0 stem via a full AnisoStemNet, strict load (loud on config drift)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    probe = tc.AnisoStemNet(len(tc.LABEL_COLS), stem_spec=ARM0_STEM_SPEC,
                            stem_strides=STEM_STRIDES, res_k=ARM0_RES_K,
                            res_dilations=ARM0_RES_DIL)
    probe.load_state_dict(ck["model"], strict=True)
    stem = probe.stem.to(dev).eval()
    for p in stem.parameters():
        p.requires_grad_(False)
    return stem, ck.get("val_auc")


def center_crop(f):
    """Same 34->32 crop as SinoMaxViT.forward (identical across arms)."""
    h, w = f.shape[-2:]
    return f[..., (h - STEM_MAP) // 2:(h - STEM_MAP) // 2 + STEM_MAP,
             (w - STEM_MAP) // 2:(w - STEM_MAP) // 2 + STEM_MAP]


@torch.no_grad()
def stem_forward(stem, x, dev):
    with torch.amp.autocast("cuda", enabled=(dev == "cuda")):
        f = center_crop(stem(x.to(dev, non_blocking=True)))
    return f.half().cpu().numpy()


def cool_pause(log, hot_c, cool_c):
    """If the SoC is hot, pause the sweep; terminate cleanly if it won't cool
    (rerun to resume — completed slices are skipped). nan (non-Spark) => no-op."""
    t = soc_temp()
    if not (t > hot_c):                          # False for nan
        return
    log(f"[thermal] SoC {t:.1f}C > {hot_c:.0f}C, pausing")
    for extra in range(1, 11):
        time.sleep(60)
        t = soc_temp()
        if t <= cool_c:
            log(f"[thermal] cooled to {t:.1f}C after {extra} min")
            return
    log(f"[thermal] TERMINATING: still {t:.1f}C after 10 min; cache is "
        f"resumable — rerun when the room cools")
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem-ckpt",
                    default=os.path.expanduser(
                        "~/sino_work/arm0_resnet_sino/sinogram/best.pt"))
    ap.add_argument("--out",
                    default=os.path.expanduser("~/sino_work/stem_features"))
    ap.add_argument("--domains", default="sinogram")
    ap.add_argument("--split-file",
                    default=os.path.expanduser("~/sino_work/subset_dc/split.json"))
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)   # DGX Spark thermal
    ap.add_argument("--gpu-mem-frac", type=float, default=0.67)
    ap.add_argument("--hot-threshold", type=float, default=85.0)
    ap.add_argument("--cool-target", type=float, default=70.0)
    ap.add_argument("--check-every", type=int, default=10,
                    help="thermal check every N batches")
    ap.add_argument("--verify", type=int, default=16,
                    help="after caching, recompute N random slices and compare")
    ap.add_argument("--verify-only", type=int, default=0,
                    help="skip generation; verify N random cached slices")
    args = ap.parse_args()

    domain = args.domains.strip()
    outdir = os.path.join(args.out, domain)
    os.makedirs(outdir, exist_ok=True)
    logpath = os.path.join(args.out, "cache.log")

    def log(msg):
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        with open(logpath, "a") as f:
            f.write(line + "\n")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    if dev == "cuda" and args.gpu_mem_frac > 0:
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_frac)

    labels = tc.load_labels()
    with open(args.split_file) as f:
        sp = json.load(f)
    have = set(labels)
    ids_tr = sorted(set(sp["train"]) & have)
    ids_va = sorted(set(sp["val"]) & have)
    ids_all = ids_tr + ids_va

    # standardization must match train_stem_transfer.py exactly: stats over the
    # FULL train split (deterministic, seed 0) — not the checkpoint's stats.
    mean, std = tc.domain_stats(domain, ids_tr)
    stem, src_auc = build_stem(args.stem_ckpt, dev)
    log(f"[setup] dev={dev} train={len(ids_tr)} val={len(ids_va)} "
        f"stats mean={mean:.3f} std={std:.3f} "
        f"stem={args.stem_ckpt} (source val_auc={src_auc:.4f})")

    def featpath(sid):
        return os.path.join(outdir, sid + ".npy")

    if not args.verify_only:
        todo = [s for s in ids_all if not os.path.exists(featpath(s))]
        log(f"[cache] {len(ids_all) - len(todo)} already cached, "
            f"{len(todo)} to compute")
        if todo:
            # shuffle=False + drop_last=False => batch b covers
            # todo[b*bs:(b+1)*bs]; DomainDS is the training input path verbatim.
            ds = tc.DomainDS(domain, todo, labels, mean, std)
            dl = DataLoader(ds, batch_size=args.bs, shuffle=False,
                            num_workers=args.workers)
            t0, done = time.time(), 0
            for b, (x, _) in enumerate(dl):
                if b % args.check_every == 0:
                    cool_pause(log, args.hot_threshold, args.cool_target)
                feats = stem_forward(stem, x, dev)
                for j in range(feats.shape[0]):
                    sid = todo[b * args.bs + j]
                    tmp = featpath(sid) + ".tmp.npy"
                    np.save(tmp, feats[j])
                    os.replace(tmp, featpath(sid))
                done += feats.shape[0]
                if b % 50 == 0:
                    log(f"[cache] {done}/{len(todo)} "
                        f"({done / (time.time() - t0):.0f} slices/s)")
            peak = (torch.cuda.max_memory_allocated() / 2**30
                    if dev == "cuda" else 0.0)
            log(f"[cache] done: {done} slices in {time.time() - t0:.0f}s, "
                f"cuda_peak={peak:.2f}GB")
        meta = dict(domain=domain, stem_ckpt=args.stem_ckpt,
                    stem_ckpt_val_auc=src_auc, mean=mean, std=std,
                    split_file=args.split_file, n_train=len(ids_tr),
                    n_val=len(ids_va), shape=[STEM_OUT_CH, STEM_MAP, STEM_MAP],
                    dtype="float16", stem_strides=list(STEM_STRIDES),
                    stem_spec=[list(k) for k in ARM0_STEM_SPEC],
                    res_k=ARM0_RES_K, res_dilations=list(ARM0_RES_DIL))
        with open(os.path.join(args.out, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)

    # verification: recompute a random sample fresh and compare to disk. Same
    # hardware + same code path should agree to fp16 rounding; a large diff
    # means the cache is stale (wrong ckpt/stats) — regenerate it.
    n_ver = args.verify_only or args.verify
    if n_ver:
        cached = [s for s in ids_all if os.path.exists(featpath(s))]
        rng = np.random.default_rng(0)
        pick = [cached[j] for j in
                rng.choice(len(cached), min(n_ver, len(cached)), replace=False)]
        ds = tc.DomainDS(domain, pick, labels, mean, std)
        dl = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=0)
        worst = 0.0
        for b, (x, _) in enumerate(dl):
            fresh = stem_forward(stem, x, dev)
            for j in range(fresh.shape[0]):
                on_disk = np.load(featpath(pick[b * args.bs + j]))
                worst = max(worst, float(np.abs(fresh[j].astype(np.float32)
                                                - on_disk.astype(np.float32)).max()))
        verdict = "OK" if worst < 1e-2 else "MISMATCH — regenerate the cache"
        log(f"[verify] {len(pick)} slices, max |fresh - cached| = {worst:.2e} "
            f"-> {verdict}")
        if worst >= 1e-2:
            sys.exit(2)


if __name__ == "__main__":
    main()
