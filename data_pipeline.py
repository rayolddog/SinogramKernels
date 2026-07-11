#!/usr/bin/env python3
"""
data_pipeline.py — precompute paired (reconstructed, sinogram) tensors from CT
DICOMs for the kernel-shape experiment.

For every input DICOM it produces TWO 256x256 tensors and stores them in SEPARATE
folders, named from the DICOM's root ID so they correlate with the RSNA
classification CSV:

    <dicom_dir>/ID_abc123.dcm
        -> <out>/recon/ID_abc123.npy      (reconstructed axial, uint16 linear HU)
        -> <out>/sinogram/ID_abc123.npy   (Radon transform, float32)

Precomputing once (rather than Radon-transforming on the fly) is the speed-up for
large datasets; a torch Dataset then just memory-maps the .npy files.

Both branches derive from the SAME linear-HU float image, so they carry identical
information and differ only in domain (reconstructed vs projection). The recon
branch is 12-bit linear HU (value = HU + 1024, range 0..4095) stored as uint16;
normalize by /RECON_MAX (4095) at train time. The sinogram branch is float32 raw
line integrals (peaks ~2e5); give it its own scale — see DomainTensorDataset.

CLI:
  data_pipeline.py precompute --dicom-dir DIR --out OUT [--labels CSV] [--workers N]
  data_pipeline.py labels --labels CSV --out OUT     # just build the label manifest
"""
import argparse
import csv
import glob
import os

import numpy as np

IMG_SIZE = 256
N_ANGLES = 256                         # -> sinogram is 256 x 256 (square, matches model)
# LINEAR Hounsfield representation (no narrow window — information-preserving).
# value = HU + HU_INTERCEPT, so 1 integer step = 1 HU. 12-bit holds HU -1024..+3071
# (the full standard CT range); use 16-bit if you need dense metal (>+3071 HU) for
# the metal-artifact experiments without clipping.
RECON_BITS = 12
HU_INTERCEPT = 1024
RECON_MAX = 2 ** RECON_BITS - 1        # 4095 (12-bit) ; 65535 if RECON_BITS=16
LABEL_COLS = ["epidural", "intraparenchymal", "intraventricular",
              "subarachnoid", "subdural", "any"]


def read_dicom_hu(path):
    """DICOM -> Hounsfield-unit float array (rescale slope/intercept applied)."""
    import pydicom
    d = pydicom.dcmread(path, force=True)
    a = d.pixel_array.astype(np.float32)
    slope = float(getattr(d, "RescaleSlope", 1.0))
    inter = float(getattr(d, "RescaleIntercept", 0.0))
    return a * slope + inter


def normalize_hu(hu):
    """LINEAR Hounsfield mapping: value = HU + intercept, clipped to the target
    bit depth (1 integer step = 1 HU). No windowing — preserves the full linear
    HU relationship. *** Swap this body for your own calibrated normalization if
    you prefer; just return a non-negative float array. ***"""
    return np.clip(np.round(hu + HU_INTERCEPT), 0, RECON_MAX).astype(np.float32)


def prepare_image(hu):
    """Resize to 256x256 and apply the (linear-HU) normalization. Both branches
    derive from THIS single prepared image."""
    from skimage.transform import resize
    img = resize(hu, (IMG_SIZE, IMG_SIZE), anti_aliasing=True, preserve_range=True)
    return normalize_hu(img)


def recon_tensor(lin):
    """Reconstructed-axial branch: the linear-HU image, stored as uint16
    (holds 12-bit 0..4095, or 16-bit 0..65535)."""
    return lin.astype(np.uint16)


def sinogram_tensor(lin):
    """Sinogram branch: Radon transform of the linear-HU image. Stored as
    FLOAT32 — line integrals of a 12-bit image over ~256 px reach ~1e6, far past
    any integer range, and real raw sinograms are float anyway. circle=True ->
    detector dim = image side, so output is 256 x 256."""
    import warnings
    from skimage.transform import radon
    # air maps to +1024 (not 0), so the corner rim outside the reconstruction
    # circle is nonzero; radon zeroes it and warns once per call. Expected — mute.
    warnings.filterwarnings("ignore", message="Radon transform: image must be zero")
    theta = np.linspace(0.0, 180.0, N_ANGLES, endpoint=False)
    sino = radon(lin, theta=theta, circle=True)             # (256, 256) float64
    return sino.astype(np.float32)


def dicom_id(path):
    """Root ID from the DICOM filename (RSNA: 'ID_abc123.dcm' -> 'ID_abc123')."""
    return os.path.splitext(os.path.basename(path))[0]


def process_one(dcm_path, recon_dir, sino_dir, overwrite=False):
    stem = dicom_id(dcm_path)
    rp = os.path.join(recon_dir, stem + ".npy")
    sp = os.path.join(sino_dir, stem + ".npy")
    if not overwrite and os.path.exists(rp) and os.path.exists(sp):
        return stem, "skip"
    try:
        lin = prepare_image(read_dicom_hu(dcm_path))        # shared source image
        np.save(rp, recon_tensor(lin))                      # uint16 linear HU
        np.save(sp, sinogram_tensor(lin))                   # float32 raw sinogram
        return stem, "ok"
    except Exception as e:                                    # corrupt/odd DICOMs
        return stem, f"fail:{type(e).__name__}"


# ---- labels ---------------------------------------------------------------
def load_labels(csv_path):
    """RSNA long-format CSV (rows 'ID_abc123_subtype,Label') -> {ID: [6-vector]}."""
    per = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            full = row["ID"]
            base, subtype = full.rsplit("_", 1)
            per.setdefault(base, {})[subtype] = int(float(row["Label"]))
    return {b: [d.get(c, 0) for c in LABEL_COLS] for b, d in per.items()}


def write_label_manifest(labels, out_dir):
    path = os.path.join(out_dir, "labels.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id"] + LABEL_COLS)
        for base, vec in sorted(labels.items()):
            w.writerow([base] + vec)
    print(f"[labels] wrote {len(labels)} rows -> {path}")
    return path


# ---- batch precompute -----------------------------------------------------
def precompute(dicom_dir, out_dir, labels_csv=None, workers=1, overwrite=False):
    recon_dir = os.path.join(out_dir, "recon")
    sino_dir = os.path.join(out_dir, "sinogram")
    os.makedirs(recon_dir, exist_ok=True)
    os.makedirs(sino_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(dicom_dir, "**", "*.dcm"), recursive=True))
    print(f"[precompute] {len(files)} DICOMs -> {out_dir}/{{recon,sinogram}}/  "
          f"({workers} worker(s))")

    counts = {"ok": 0, "skip": 0}
    fails = []

    def handle(res):
        stem, status = res
        if status.startswith("fail"):
            fails.append((stem, status))
        else:
            counts[status] += 1
        done = counts["ok"] + counts["skip"] + len(fails)
        if done % 500 == 0:
            print(f"[precompute] {done}/{len(files)}  "
                  f"ok={counts['ok']} skip={counts['skip']} fail={len(fails)}",
                  flush=True)

    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(process_one, f, recon_dir, sino_dir, overwrite)
                    for f in files]
            for fut in futs:
                handle(fut.result())
    else:
        for f in files:
            handle(process_one(f, recon_dir, sino_dir, overwrite))

    print(f"[precompute] DONE ok={counts['ok']} skip={counts['skip']} "
          f"fail={len(fails)}")
    if fails:
        print(f"[precompute] first failures: {fails[:5]}")
    if labels_csv:
        write_label_manifest(load_labels(labels_csv), out_dir)


# ---- training-time Dataset ------------------------------------------------
class DomainTensorDataset:
    """torch Dataset over the precomputed tensors of ONE domain ('recon' or
    'sinogram'), joined to labels by ID. Returns (int16 tensor (1,256,256),
    float label vector). Import torch lazily so precompute doesn't need it."""

    def __init__(self, out_dir, domain, ids=None):
        import torch
        self.torch = torch
        self.dir = os.path.join(out_dir, domain)
        # labels
        self.labels = {}
        lp = os.path.join(out_dir, "labels.csv")
        if os.path.exists(lp):
            with open(lp) as f:
                for row in csv.DictReader(f):
                    self.labels[row["id"]] = [int(row[c]) for c in LABEL_COLS]
        # ids present in this domain folder (optionally restricted to a split)
        have = {os.path.splitext(x)[0] for x in os.listdir(self.dir)
                if x.endswith(".npy")}
        keep = have & set(self.labels) if self.labels else have
        self.ids = sorted(keep if ids is None else (set(ids) & keep))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        stem = self.ids[i]
        arr = np.load(os.path.join(self.dir, stem + ".npy"))      # uint16 or float32
        x = self.torch.from_numpy(arr.astype(np.float32))[None]   # (1,256,256) float

        y = self.torch.tensor(self.labels.get(stem, [0] * len(LABEL_COLS)),
                              dtype=self.torch.float32)
        return x, y


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("precompute")
    p.add_argument("--dicom-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--labels")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    q = sub.add_parser("labels")
    q.add_argument("--labels", required=True)
    q.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "precompute":
        precompute(a.dicom_dir, a.out, a.labels, a.workers, a.overwrite)
    elif a.cmd == "labels":
        os.makedirs(a.out, exist_ok=True)
        write_label_manifest(load_labels(a.labels), a.out)


if __name__ == "__main__":
    main()
