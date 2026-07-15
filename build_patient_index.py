#!/usr/bin/env python3
"""
build_patient_index.py — scan DICOM headers and record the grouping keys the
RSNA label CSV does not carry.

stage_2_train.csv is SLICE-level only: it has no patient or study column. Any
subset or train/val split built from it alone therefore cannot know that ~38
slices share a patient. Two consequences, both fatal to a performance claim:

  1. Correlation — adjacent slices of one head CT are near-duplicates. A subset
     drawn by slice is effectively a much smaller sample of independent cases
     than its row count suggests.
  2. Leakage — twin_common.split_ids() splits on md5(slice_id), so slices of the
     SAME patient land on BOTH sides of the split. Val AUC is then partly a
     memorization score.

This writes the patient/study/series/z-position for every DICOM present, so a
downstream selector can pick at most-N-per-patient and split BY PATIENT.

Output: a CSV with one row per DICOM found.

Usage:
  build_patient_index.py --dicom-dir DIR --out patient_index.csv [--workers N]
"""
import argparse
import csv
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings("ignore")

FIELDS = ["id", "patient", "study", "series", "z", "rows", "cols", "slope", "intercept"]


def read_one(path):
    import pydicom
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        d = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        ipp = getattr(d, "ImagePositionPatient", None)
        z = float(ipp[2]) if ipp is not None and len(ipp) == 3 else float("nan")
        return [stem,
                str(getattr(d, "PatientID", "")),
                str(getattr(d, "StudyInstanceUID", "")),
                str(getattr(d, "SeriesInstanceUID", "")),
                z,
                int(getattr(d, "Rows", 0)),
                int(getattr(d, "Columns", 0)),
                float(getattr(d, "RescaleSlope", 1.0)),
                float(getattr(d, "RescaleIntercept", 0.0))]
    except Exception as e:
        return [stem, "", "", "", float("nan"), 0, 0, 1.0, 0.0, f"fail:{type(e).__name__}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    files = [os.path.join(a.dicom_dir, x) for x in os.listdir(a.dicom_dir)
             if x.endswith(".dcm")]
    files.sort()
    print(f"[index] {len(files)} DICOMs -> {a.out} ({a.workers} workers)", flush=True)

    n_fail = 0
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for i, row in enumerate(ex.map(read_one, files, chunksize=256)):
                if len(row) > len(FIELDS):
                    n_fail += 1
                    row = row[:len(FIELDS)]
                w.writerow(row)
                if (i + 1) % 25000 == 0:
                    print(f"[index] {i+1}/{len(files)} fail={n_fail}", flush=True)
    print(f"[index] DONE rows={len(files)} fail={n_fail} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
