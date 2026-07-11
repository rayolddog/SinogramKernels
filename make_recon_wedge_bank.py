#!/usr/bin/env python3
"""make_recon_wedge_bank.py — precompute the recon-side wedge-augmentation bank
for the Tier-2 factorial.

Parity requires the recon branch to see the SAME physical corruption as the
sinogram branch. The recon test (angular_robustness) feeds iradon(wedged sino),
with frac=0 == iradon(full sino) — i.e. the recon test domain is the IRADON
domain, not the original uint16 recon cache. So the recon training domain must
also be iradon-domain for consistency.

Per train image we store a bank of (1 + K) variants, float16, shape (1+K,256,256):
  variant 0    = iradon(full sino)              (clean, used when aug not applied)
  variant 1..K = iradon(wedged sino)            (random frac/phase, like angular_dropout)
The wedge-aug recon loader picks variant 0 with prob (1-p), else a random wedged
variant — mirroring angular_dropout(p=0.5). Resumable: existing files are skipped.

Writes to  $SINO_CACHE/recon_wedge/{id}.npy  (train ids only; the eval reprojects
the 2000-slice val test set on the fly).
"""
import os
import sys
import numpy as np
from multiprocessing import Pool
from skimage.transform import iradon

import twin_common as tc

CACHE = os.environ.get("SINO_CACHE", "/mnt/mlcache/subset200k")
OUT = os.path.join(CACHE, "recon_wedge")
K = int(os.environ.get("WEDGE_K", "3"))            # wedged variants per image
NWORK = int(os.environ.get("WEDGE_WORKERS", "14"))
THETA = np.linspace(0.0, 180.0, 256, endpoint=False)
SEED = 12345


def wedge_cols(sino, frac, start):
    """Zero a contiguous angular wedge (columns) of width frac, phase `start`,
    with wrap-around — matches tc.angular_dropout geometry."""
    A = sino.shape[1]
    w = int(frac * A)
    if w <= 0:
        return sino
    s = sino.copy()
    end = start + w
    if end <= A:
        s[:, start:end] = 0.0
    else:
        s[:, start:] = 0.0
        s[:, :end - A] = 0.0
    return s


def build_one(args):
    stem, seed = args
    outp = os.path.join(OUT, stem + ".npy")
    if os.path.exists(outp):
        return 0
    try:
        sino = np.load(os.path.join(CACHE, "sinogram", stem + ".npy")).astype(np.float32)
    except Exception:
        return -1
    rng = np.random.default_rng(seed)
    variants = np.empty((1 + K, 256, 256), dtype=np.float16)
    variants[0] = iradon(sino, theta=THETA, filter_name="ramp",
                         circle=True).astype(np.float16)
    for k in range(K):
        frac = float(rng.uniform(0.05, 0.40))
        start = int(rng.integers(0, 256))
        w = wedge_cols(sino, frac, start)
        variants[1 + k] = iradon(w, theta=THETA, filter_name="ramp",
                                 circle=True).astype(np.float16)
    tmp = outp + ".tmp"
    with open(tmp, "wb") as f:          # file object -> np.save won't append .npy
        np.save(f, variants)
    os.replace(tmp, outp)
    return 1


def main():
    os.makedirs(OUT, exist_ok=True)
    tc.CACHE = CACHE
    labels = tc.load_labels()
    ids = sorted(labels)
    train, val = tc.split_ids(ids)
    split = os.environ.get("WEDGE_SPLIT", "all")   # train | val | all
    sel = {"train": train, "val": val, "all": train + val}[split]
    jobs = [(stem, SEED + i) for i, stem in enumerate(sel)]
    print(f"[bank] split={split}", flush=True)
    print(f"[bank] {len(jobs)} train imgs, K={K} wedged variants, {NWORK} workers "
          f"-> {OUT}", flush=True)
    done = skip = err = 0
    with Pool(NWORK) as pool:
        for i, r in enumerate(pool.imap_unordered(build_one, jobs, chunksize=32)):
            if r == 1:
                done += 1
            elif r == 0:
                skip += 1
            else:
                err += 1
            if (i + 1) % 5000 == 0:
                print(f"[bank] {i+1}/{len(jobs)}  built={done} skip={skip} err={err}",
                      flush=True)
    print(f"[bank] DONE built={done} skip={skip} err={err}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
