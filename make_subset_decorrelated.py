#!/usr/bin/env python3
"""
make_subset_decorrelated.py — pick a subset of RSNA-ICH slices that are actually
independent, and split train/val BY PATIENT.

Why this exists (make_subset.py does neither of these things):

  1. DECORRELATION. make_subset.py stratifies by the 6-tuple label vector but is
     blind to patients, because stage_2_train.csv is slice-level and has no patient
     column. A head CT contributes ~38 near-duplicate axial slices; drawing 50k
     slices at random therefore yields far fewer than 50k independent samples. The
     effective sample size is the PATIENT count, not the row count. We cap slices
     per patient (--per-patient) and prefer slices spaced apart in z.

  2. LEAKAGE. twin_common.split_ids() splits on md5(slice_id), so slices of one
     patient land on BOTH sides. Val AUC then partly measures memorization of that
     patient's anatomy. EXPERIMENT_recon_objective.md notes this and explicitly
     defers it ("Fix the split for the *performance* claims, not this one") — which
     is fine for the ramp-emergence question (it inspects weights, not
     generalization) but NOT for stem transferability, whose entire output is AUC
     and convergence curves. We split by patient hash instead.

Reads the header scan from build_patient_index.py (patient/study/series/z per slice)
joined to the RSNA label CSV.

Outputs under <out>/:
  labels.csv       id + 6-vector for the selected ids (twin_common-compatible)
  subset_ids.txt   selected ids
  split.json       {"train": [...], "val": [...]} — patient-disjoint
  dicom_list.txt   absolute DICOM paths, for radon.zig --list
  manifest.json    provenance + the counts that matter

Usage:
  make_subset_decorrelated.py --index patient_index.csv --labels stage_2_train.csv \
      --dicom-dir DIR --out subset_dc --per-patient 2 --val-frac 0.1 --seed 0
"""
import argparse
import csv
import hashlib
import json
import os
import random
from collections import defaultdict

import data_pipeline as dp

LABEL_COLS = dp.LABEL_COLS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--dicom-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-patient", type=int, default=2,
                    help="max slices kept per patient (1 = fully independent)")
    ap.add_argument("--min-z-gap", type=float, default=10.0,
                    help="mm; keep slices at least this far apart within a patient")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--max-total", type=int, default=0, help="0 = no cap")
    ap.add_argument("--prefer-positive", action="store_true", default=True,
                    help="when capping a patient, keep hemorrhage-positive slices first")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)

    print("[subset] loading labels ...", flush=True)
    labels = dp.load_labels(a.labels)

    print("[subset] loading patient index ...", flush=True)
    by_patient = defaultdict(list)
    n_idx = n_nolabel = n_blank = 0
    with open(a.index) as f:
        for r in csv.DictReader(f):
            n_idx += 1
            if not r["patient"]:
                n_blank += 1
                continue
            if r["id"] not in labels:
                n_nolabel += 1
                continue
            try:
                z = float(r["z"])
            except ValueError:
                z = float("nan")
            by_patient[r["patient"]].append((r["id"], z))
    print(f"[subset] index rows={n_idx} blank={n_blank} unlabeled={n_nolabel} "
          f"patients={len(by_patient)}", flush=True)
    if n_blank:
        print(f"[subset] WARNING: {n_blank} index rows have no patient — "
              f"re-run build_patient_index.py; those slices are dropped.", flush=True)

    # ---- pick slices within each patient -----------------------------------
    chosen = []
    for pid, slices in by_patient.items():
        # z-sort so "spread apart" is meaningful; NaN z -> arbitrary but stable
        slices.sort(key=lambda t: (t[1] if t[1] == t[1] else 0.0, t[0]))
        if a.prefer_positive:
            # anchor on a positive slice when the patient has one: dropping to 1-2
            # slices/patient would otherwise discard most of the hemorrhages, and
            # positives are the minority class we can least afford to thin.
            pos = [s for s in slices if labels[s[0]][-1] == 1]
            pool = pos if pos else slices
        else:
            pool = slices
        keep, last_z = [], None
        for sid, z in pool:
            if len(keep) >= a.per_patient:
                break
            if last_z is not None and z == z and last_z == last_z and abs(z - last_z) < a.min_z_gap:
                continue
            keep.append(sid)
            last_z = z
        chosen.extend(keep)

    chosen.sort()
    if a.max_total and len(chosen) > a.max_total:
        rng.shuffle(chosen)
        chosen = sorted(chosen[:a.max_total])

    pos_n = sum(labels[i][-1] for i in chosen)
    print(f"[subset] selected {len(chosen)} slices from {len(by_patient)} patients "
          f"({len(chosen)/max(len(by_patient),1):.2f}/patient); "
          f"positives={pos_n} ({100*pos_n/max(len(chosen),1):.1f}%)", flush=True)

    # ---- patient-disjoint split --------------------------------------------
    sid2pat = {}
    for pid, slices in by_patient.items():
        for sid, _ in slices:
            sid2pat[sid] = pid

    def pat_bucket(pid):
        return int(hashlib.md5((pid + f"|seed{a.seed}").encode()).hexdigest(), 16) % 1000

    thresh = int(a.val_frac * 1000)
    train = [i for i in chosen if pat_bucket(sid2pat[i]) >= thresh]
    val = [i for i in chosen if pat_bucket(sid2pat[i]) < thresh]
    tp = {sid2pat[i] for i in train}
    vp = {sid2pat[i] for i in val}
    overlap = tp & vp
    print(f"[subset] split train={len(train)} ({len(tp)} patients)  "
          f"val={len(val)} ({len(vp)} patients)  patient_overlap={len(overlap)}", flush=True)
    assert not overlap, "patient leaked across split"

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "labels.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id"] + LABEL_COLS)
        for i in chosen:
            w.writerow([i] + labels[i])
    with open(os.path.join(a.out, "subset_ids.txt"), "w") as f:
        f.write("\n".join(chosen) + "\n")
    with open(os.path.join(a.out, "split.json"), "w") as f:
        json.dump({"train": train, "val": val}, f)
    with open(os.path.join(a.out, "dicom_list.txt"), "w") as f:
        for i in chosen:
            f.write(os.path.join(os.path.abspath(a.dicom_dir), i + ".dcm") + "\n")
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump({
            "n_slices": len(chosen), "n_patients": len(by_patient),
            "per_patient": a.per_patient, "min_z_gap": a.min_z_gap,
            "positives": pos_n, "pos_frac": pos_n / max(len(chosen), 1),
            "train": len(train), "val": len(val),
            "train_patients": len(tp), "val_patients": len(vp),
            "val_frac": a.val_frac, "seed": a.seed,
            "split": "patient-level (md5 of PatientID)",
        }, f, indent=2)
    print(f"[subset] wrote labels.csv, subset_ids.txt, split.json, dicom_list.txt, "
          f"manifest.json -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
