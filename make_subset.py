#!/usr/bin/env python3
"""
make_subset.py — pick a STRATIFIED subset of RSNA-ICH slices and stage them as
symlinks so data_pipeline.precompute can run against a smaller, class-balanced-
in-proportion sample.

Stratum = the full 6-tuple ICH label vector (epidural, intraparenchymal,
intraventricular, subarachnoid, subdural, any). Proportional (largest-remainder)
allocation preserves both the ~14% positive prevalence AND the subtype
co-occurrence structure of the full set. Deterministic via --seed.

Outputs under <out>/:
  dicom/ID_*.dcm   symlinks -> the selected source DICOMs
  labels.csv       id + 6-vector, ONLY the selected ids (matches precompute's format)
  subset_ids.txt   the chosen ids, one per line

Usage:
  make_subset.py --dicom-dir /mnt/mlcache/dicom_staging \
                 --labels /home/johnb/Documents/stage_2_train.csv \
                 --out /mnt/mlcache/subset50k --n 50000 --seed 0
"""
import argparse
import os

import numpy as np

import data_pipeline as dp


def largest_remainder(strata, n_total, n_target, rng):
    """Proportional allocation of n_target picks across strata (dict key->list of
    ids), rounded by the largest-remainder method so the picks sum EXACTLY to
    min(n_target, n_total). Returns the flat list of chosen ids."""
    keys = list(strata)
    raw = {k: len(strata[k]) / n_total * n_target for k in keys}
    floor = {k: int(np.floor(raw[k])) for k in keys}
    # cap each stratum at its available size
    floor = {k: min(floor[k], len(strata[k])) for k in keys}
    allocated = sum(floor.values())
    remainder = sorted(keys, key=lambda k: raw[k] - floor[k], reverse=True)
    i = 0
    while allocated < min(n_target, n_total) and i < len(remainder) * 4:
        k = remainder[i % len(remainder)]
        if floor[k] < len(strata[k]):
            floor[k] += 1
            allocated += 1
        i += 1
    chosen = []
    for k in keys:
        ids = strata[k]
        take = floor[k]
        if take >= len(ids):
            chosen.extend(ids)
        else:
            idx = rng.choice(len(ids), size=take, replace=False)
            chosen.extend([ids[j] for j in idx])
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom-dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)

    print("[subset] loading labels ...", flush=True)
    labels = dp.load_labels(a.labels)          # {ID: [6-vector]}

    print("[subset] scanning DICOM dir ...", flush=True)
    present = {dp.dicom_id(x) for x in os.listdir(a.dicom_dir)
               if x.endswith(".dcm")}
    ids = sorted(present & set(labels))
    print(f"[subset] {len(present)} dicoms, {len(ids)} with labels", flush=True)

    # group by label tuple
    strata = {}
    for i in ids:
        strata.setdefault(tuple(labels[i]), []).append(i)
    pos = sum(len(v) for k, v in strata.items() if k[-1] == 1)
    print(f"[subset] {len(strata)} distinct label combos; "
          f"positives(any=1): {pos} ({100*pos/len(ids):.1f}%)", flush=True)

    chosen = largest_remainder(strata, len(ids), a.n, rng)
    chosen.sort()
    cpos = sum(labels[i][-1] for i in chosen)
    print(f"[subset] selected {len(chosen)}  positives: {cpos} "
          f"({100*cpos/len(chosen):.1f}%)", flush=True)

    dicom_out = os.path.join(a.out, "dicom")
    os.makedirs(dicom_out, exist_ok=True)
    made = 0
    for i in chosen:
        src = os.path.join(a.dicom_dir, i + ".dcm")
        dst = os.path.join(dicom_out, i + ".dcm")
        if not os.path.islink(dst) and not os.path.exists(dst):
            os.symlink(src, dst)
            made += 1
    print(f"[subset] symlinked {made} new (of {len(chosen)}) -> {dicom_out}",
          flush=True)

    # subset label manifest + id list
    import csv
    with open(os.path.join(a.out, "labels.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id"] + dp.LABEL_COLS)
        for i in chosen:
            w.writerow([i] + labels[i])
    with open(os.path.join(a.out, "subset_ids.txt"), "w") as f:
        f.write("\n".join(chosen) + "\n")
    print(f"[subset] wrote labels.csv + subset_ids.txt -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
