#!/usr/bin/env python3
"""
motion_experiment.py — SIIM-CAIMI 2026 quick experiment: does the sinogram-domain
model degrade more gracefully under simulated patient motion than the
reconstructed-domain model?

Design (2x2 x severity, all arms derive from the SAME measurements):

                     clean measurements        motion measurements
  sinogram model     clean sinogram            motion sinogram
  recon model        iradon(clean sino)        iradon(motion sino)

The recon model's CLEAN arm is iradon(clean sinogram) — NOT the original image —
so the radon/iradon round-trip loss is controlled and the only variable across
columns is motion. Primary endpoint: bootstrap CI on the difference-of-differences
ddAUC = [AUC_sino(clean) - AUC_sino(motion)] - [AUC_recon(clean) - AUC_recon(motion)]
for the 'any' label, per severity.

Motion model: rigid in-plane translation (dx, dy) beginning at a random onset
angle. A projection at angle theta shifts along the DETECTOR axis by
    s(theta) = dx*cos(theta) + dy*sin(theta)   [pixels]
('step' = patient moves once and stays; 'drift' = displacement ramps linearly
from onset to the last view, the shear-like variant). Vacated detector cells are
zero-filled — line integrals through air are ~0, and dimensions never change.

skimage radon axis convention (as in data_pipeline.py): sinogram[detector, angle].

Workflow (each step resumable / re-runnable):
  python motion_experiment.py cohort --tensors ./tensors --n 1000 [--dicom-dir D]
  python motion_experiment.py inject --tensors ./tensors --cohort cohort.csv \
      --out ./motion_arms --severities 0,2,4,8 [--mode step]
  python motion_experiment.py qc     --arms ./motion_arms --tensors ./tensors
  python motion_experiment.py infer  --arms ./motion_arms --tensors ./tensors \
      --cohort cohort.csv --ckpt-recon R.pt --ckpt-sino S.pt --out results.csv
  python motion_experiment.py stats  --results results.csv --out stats.md

Deps: numpy scipy scikit-image torch (infer only) matplotlib (qc only) pydicom
(cohort --dicom-dir only).
"""
import argparse
import csv
import glob
import hashlib
import json
import os

import numpy as np

from data_pipeline import LABEL_COLS, IMG_SIZE, N_ANGLES

THETA_DEG = np.linspace(0.0, 180.0, N_ANGLES, endpoint=False)
THETA_RAD = np.deg2rad(THETA_DEG)
ANY_COL = LABEL_COLS.index("any")


# ---- cohort ----------------------------------------------------------------
def build_cohort(tensors, n, labels_csv=None, dicom_dir=None, seed=20260722,
                 out="cohort.csv"):
    """Sample a balanced (any-positive vs negative) cohort from the precomputed
    tensors. If --dicom-dir is given, PatientID is read from the DICOM headers
    and the cohort is deduplicated to ONE slice per patient (the 'uncorrelated'
    requirement); otherwise slices are sampled directly and a warning is
    printed — verify patient-level uniqueness before submission."""
    lp = labels_csv or os.path.join(tensors, "labels.csv")
    labels = {}
    with open(lp) as f:
        for row in csv.DictReader(f):
            labels[row["id"]] = [int(row[c]) for c in LABEL_COLS]
    have = {os.path.splitext(x)[0]
            for x in os.listdir(os.path.join(tensors, "sinogram"))
            if x.endswith(".npy")}
    ids = sorted(have & set(labels))
    print(f"[cohort] {len(ids)} labeled tensors available")

    patient = {}
    if dicom_dir:
        import pydicom
        files = {os.path.splitext(os.path.basename(p))[0]: p
                 for p in glob.glob(os.path.join(dicom_dir, "**", "*.dcm"),
                                    recursive=True)}
        for i in ids:
            if i in files:
                d = pydicom.dcmread(files[i], stop_before_pixels=True, force=True)
                patient[i] = str(getattr(d, "PatientID", i))
        print(f"[cohort] PatientID resolved for {len(patient)}/{len(ids)}")
    else:
        print("[cohort] WARNING: no --dicom-dir; sampling at slice level. "
              "Verify one-slice-per-patient before submission.")

    rng = np.random.default_rng(seed)
    pos = [i for i in ids if labels[i][ANY_COL] == 1]
    neg = [i for i in ids if labels[i][ANY_COL] == 0]
    rng.shuffle(pos), rng.shuffle(neg)

    picked, seen_pat = [], set()
    for pool in (pos, neg):
        k = 0
        for i in pool:
            if k >= n // 2:
                break
            p = patient.get(i, i)
            if p in seen_pat:
                continue
            seen_pat.add(p)
            picked.append(i)
            k += 1
        if k < n // 2:
            print(f"[cohort] WARNING: only {k} unique-patient samples in one arm")

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "patient"] + LABEL_COLS)
        for i in sorted(picked):
            w.writerow([i, patient.get(i, "")] + labels[i])
    npos = sum(labels[i][ANY_COL] for i in picked)
    print(f"[cohort] wrote {len(picked)} rows ({npos} pos / "
          f"{len(picked) - npos} neg) -> {out}")


def read_cohort(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    ids = [r["id"] for r in rows]
    y = np.array([[int(r[c]) for c in LABEL_COLS] for r in rows], dtype=np.float32)
    return ids, y


# ---- motion injection -------------------------------------------------------
def _per_image_rng(stem, seed):
    h = int(hashlib.sha1(f"{seed}:{stem}".encode()).hexdigest()[:16], 16)
    return np.random.default_rng(h)


def inject_motion(sino, peak, rng, mode="step"):
    """Rigid-translation motion in the sinogram. peak = |(dx,dy)| in pixels,
    random direction, random onset in the middle 60% of views (motion at the
    very first/last views is nearly invisible). Returns (motion sinogram,
    params). Zero-fill (air) at vacated detector cells; shape is unchanged."""
    from scipy.ndimage import shift as ndshift
    if peak == 0:
        return sino.copy(), {"peak": 0.0}
    psi = rng.uniform(0.0, 2.0 * np.pi)
    dx, dy = peak * np.cos(psi), peak * np.sin(psi)
    onset = int(rng.integers(int(0.2 * N_ANGLES), int(0.8 * N_ANGLES)))
    out = sino.copy()
    for j in range(onset, N_ANGLES):
        s = dx * np.cos(THETA_RAD[j]) + dy * np.sin(THETA_RAD[j])
        if mode == "drift":                      # ramps 0 -> full by the last view
            s *= (j - onset) / max(1, N_ANGLES - 1 - onset)
        out[:, j] = ndshift(sino[:, j], s, order=1, mode="constant", cval=0.0)
    return out, {"peak": float(peak), "dx": float(dx), "dy": float(dy),
                 "onset": onset, "mode": mode}


def inject(tensors, cohort, out_dir, severities, mode="step", seed=20260722):
    """Precompute all arms to disk:
        <out>/sino_s{P}/ID.npy      motion sinogram at peak P (P=0 -> clean copy)
        <out>/recon_s{P}/ID.npy     iradon of that sinogram (float32)
    Resumable: existing files are skipped. Params logged to <out>/motion_log.csv."""
    from skimage.transform import iradon
    ids, _ = read_cohort(cohort)
    os.makedirs(out_dir, exist_ok=True)
    logp = os.path.join(out_dir, "motion_log.csv")
    log = open(logp, "a", newline="")
    logw = csv.writer(log)
    if os.path.getsize(logp) == 0:
        logw.writerow(["id", "severity", "dx", "dy", "onset", "mode"])

    for p in severities:
        os.makedirs(os.path.join(out_dir, f"sino_s{p}"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, f"recon_s{p}"), exist_ok=True)

    for k, stem in enumerate(ids):
        sino = np.load(os.path.join(tensors, "sinogram", stem + ".npy"))
        for p in severities:
            sp = os.path.join(out_dir, f"sino_s{p}", stem + ".npy")
            rp = os.path.join(out_dir, f"recon_s{p}", stem + ".npy")
            if os.path.exists(sp) and os.path.exists(rp):
                continue
            # fresh rng from the same (id, seed) each call -> identical
            # (direction, onset) across severities; only the peak differs
            m, prm = inject_motion(sino, p, _per_image_rng(stem, seed), mode)
            np.save(sp, m.astype(np.float32))
            rec = iradon(m.astype(np.float64), theta=THETA_DEG, circle=True,
                         filter_name="ramp", output_size=IMG_SIZE)
            np.save(rp, np.clip(rec, 0, None).astype(np.float32))
            if p > 0:
                logw.writerow([stem, p, prm.get("dx"), prm.get("dy"),
                               prm.get("onset"), prm.get("mode")])
        if (k + 1) % 50 == 0:
            print(f"[inject] {k + 1}/{len(ids)}", flush=True)
    log.close()
    print(f"[inject] DONE -> {out_dir} (params in motion_log.csv)")


# ---- inference --------------------------------------------------------------
def _load_model(ckpt, device):
    import torch
    from model import build
    net = build(num_classes=len(LABEL_COLS))
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    net.load_state_dict(sd)
    return net.to(device).eval()


def infer(arms, tensors, cohort, ckpt_recon, ckpt_sino, out, severities,
          batch=32, device=None):
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else
                        "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[infer] device={device}")
    ids, _ = read_cohort(cohort)
    models = {"recon": _load_model(ckpt_recon, device),
              "sino": _load_model(ckpt_sino, device)}

    rows = []
    for domain, sub in (("sino", "sino_s{p}"), ("recon", "recon_s{p}")):
        for p in severities:
            d = os.path.join(arms, sub.format(p=p))
            probs = np.zeros((len(ids), len(LABEL_COLS)), dtype=np.float32)
            with torch.no_grad():
                for b0 in range(0, len(ids), batch):
                    chunk = ids[b0:b0 + batch]
                    x = np.stack([np.load(os.path.join(d, s + ".npy"))
                                  for s in chunk]).astype(np.float32)
                    x = torch.from_numpy(x)[:, None].to(device)
                    probs[b0:b0 + len(chunk)] = torch.sigmoid(
                        models[domain](x)).cpu().numpy()
            for i, s in enumerate(ids):
                rows.append([s, domain, p] + probs[i].tolist())
            print(f"[infer] {domain} severity {p}: done ({len(ids)} images)")

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "domain", "severity"] + [f"p_{c}" for c in LABEL_COLS])
        w.writerows(rows)
    print(f"[infer] wrote {len(rows)} rows -> {out}")


# ---- stats ------------------------------------------------------------------
def _auc(y, p):
    """Rank-based AUC (Mann-Whitney), no sklearn dependency."""
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def stats(results, cohort, out, n_boot=2000, seed=20260722):
    ids, y = read_cohort(cohort)
    idx = {s: i for i, s in enumerate(ids)}
    yany = y[:, ANY_COL]

    # probs[domain][severity] -> aligned (N,) any-ICH prob vector
    probs = {}
    sevs = set()
    with open(results) as f:
        for row in csv.DictReader(f):
            d, p = row["domain"], int(float(row["severity"]))
            sevs.add(p)
            probs.setdefault(d, {}).setdefault(p, np.zeros(len(ids)))[
                idx[row["id"]]] = float(row["p_any"])
    sevs = sorted(sevs)
    base = sevs[0]                                   # severity 0 = clean column

    rng = np.random.default_rng(seed)
    boots = [rng.integers(0, len(ids), len(ids)) for _ in range(n_boot)]

    lines = ["| severity | AUC sino | AUC recon | dAUC sino | dAUC recon | "
             "ddAUC [95% CI] |", "|---|---|---|---|---|---|"]
    summary = {}
    for p in sevs:
        a_s, a_r = _auc(yany, probs["sino"][p]), _auc(yany, probs["recon"][p])
        d_s = _auc(yany, probs["sino"][base]) - a_s
        d_r = _auc(yany, probs["recon"][base]) - a_r
        dd = d_r - d_s          # >0 => recon degrades MORE (sinogram more robust)
        bb = []
        for b in boots:
            yb = yany[b]
            ds = _auc(yb, probs["sino"][base][b]) - _auc(yb, probs["sino"][p][b])
            dr = _auc(yb, probs["recon"][base][b]) - _auc(yb, probs["recon"][p][b])
            bb.append(dr - ds)
        lo, hi = np.nanpercentile(bb, [2.5, 97.5])
        lines.append(f"| {p} px | {a_s:.4f} | {a_r:.4f} | {d_s:+.4f} | "
                     f"{d_r:+.4f} | {dd:+.4f} [{lo:+.4f}, {hi:+.4f}] |")
        summary[p] = {"auc_sino": a_s, "auc_recon": a_r, "ddauc": dd,
                      "ci": [float(lo), float(hi)]}

    md = ("# Motion-robustness results (any-ICH AUC)\n\n" + "\n".join(lines) +
          "\n\nddAUC = dAUC(recon) - dAUC(sino); positive = sinogram-domain "
          "model is more motion-robust. Paired bootstrap over images, "
          f"{n_boot} resamples.\n")
    with open(out, "w") as f:
        f.write(md)
    with open(os.path.splitext(out)[0] + ".json", "w") as f:
        json.dump(summary, f, indent=2)
    print(md)


# ---- QC figure --------------------------------------------------------------
def qc(arms, tensors, cohort, severity, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ids, y = read_cohort(cohort)
    stem = next(s for s, v in zip(ids, y) if v[ANY_COL] == 1)   # first positive
    panels = [
        (np.load(os.path.join(tensors, "sinogram", stem + ".npy")), "clean sinogram"),
        (np.load(os.path.join(arms, f"sino_s{severity}", stem + ".npy")),
         f"motion sinogram ({severity} px)"),
        (np.load(os.path.join(arms, "recon_s0", stem + ".npy")), "iradon(clean)"),
        (np.load(os.path.join(arms, f"recon_s{severity}", stem + ".npy")),
         f"iradon(motion {severity} px)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    for ax, (img, title) in zip(axes.ravel(), panels):
        ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"{stem} — motion arms QC", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[qc] wrote {out}")


# ---- CLI --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("cohort")
    p.add_argument("--tensors", required=True)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--labels")
    p.add_argument("--dicom-dir")
    p.add_argument("--seed", type=int, default=20260722)
    p.add_argument("--out", default="cohort.csv")

    p = sub.add_parser("inject")
    p.add_argument("--tensors", required=True)
    p.add_argument("--cohort", default="cohort.csv")
    p.add_argument("--out", default="./motion_arms")
    p.add_argument("--severities", default="0,2,4,8")
    p.add_argument("--mode", choices=["step", "drift"], default="step")
    p.add_argument("--seed", type=int, default=20260722)

    p = sub.add_parser("infer")
    p.add_argument("--arms", default="./motion_arms")
    p.add_argument("--tensors", required=True)
    p.add_argument("--cohort", default="cohort.csv")
    p.add_argument("--ckpt-recon", required=True)
    p.add_argument("--ckpt-sino", required=True)
    p.add_argument("--severities", default="0,2,4,8")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device")
    p.add_argument("--out", default="results.csv")

    p = sub.add_parser("stats")
    p.add_argument("--results", default="results.csv")
    p.add_argument("--cohort", default="cohort.csv")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--out", default="stats.md")

    p = sub.add_parser("qc")
    p.add_argument("--arms", default="./motion_arms")
    p.add_argument("--tensors", required=True)
    p.add_argument("--cohort", default="cohort.csv")
    p.add_argument("--severity", type=int, default=8)
    p.add_argument("--out", default="qc_motion.png")

    a = ap.parse_args()
    if a.cmd == "cohort":
        build_cohort(a.tensors, a.n, a.labels, a.dicom_dir, a.seed, a.out)
    elif a.cmd == "inject":
        sev = [int(x) for x in a.severities.split(",")]
        inject(a.tensors, a.cohort, a.out, sev, a.mode, a.seed)
    elif a.cmd == "infer":
        sev = [int(x) for x in a.severities.split(",")]
        infer(a.arms, a.tensors, a.cohort, a.ckpt_recon, a.ckpt_sino,
              a.out, sev, a.batch, a.device)
    elif a.cmd == "stats":
        stats(a.results, a.cohort, a.out, a.n_boot)
    elif a.cmd == "qc":
        qc(a.arms, a.tensors, a.cohort, a.severity, a.out)


if __name__ == "__main__":
    main()
