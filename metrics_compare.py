#!/usr/bin/env python3
"""metrics_compare.py — beyond-AUC comparison of the baseline aniso net vs the
go-for-broke (aniso2 architecture) net, both trained at 200k. Uses the existing
checkpoints (inference only). Measures AUC hides:
  - per-subtype AUC (esp. subarachnoid, the hardest / most label-ambiguous)
  - calibration (reliability curve, ECE, Brier) — does it know when it's unsure
  - clinical net benefit (decision-curve analysis) for the 'any' decision
  - convergence speed (val-AUC learning curves from the training logs)
Evaluated on the shared val split (the basis best.pt was selected on) — consistent
for both models, so the comparison is fair even if absolute values are optimistic.
"""
import re
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, brier_score_loss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import twin_common as tc

tc.CACHE = "/mnt/mlcache/subset200k"
DEV = "cuda"
ANY = tc.LABEL_COLS.index("any")

MODELS = {
    "baseline": dict(dir="abl_base", color="#7f8c8d",
                     kw=dict(stem_spec=tc.ANISO_STEM, stem_strides=(2, 2, 2, 2),
                             res_k=3, res_dilations=(1, 1, 1))),
    "go-for-broke": dict(dir="abl_all", color="#c0392b",
                         kw=dict(stem_spec=tc.ANISO2_STEM, stem_strides=(2, 2, 2, 1),
                                 res_k=5, res_dilations=(1, 2, 4))),
}


def infer(cfg, ids, labels):
    ck = torch.load(f"{cfg['dir']}/sinogram/best.pt", map_location=DEV,
                    weights_only=False)
    m = tc.AnisoStemNet(6, **cfg["kw"]).to(DEV)
    m.load_state_dict(ck["model"]); m.eval()
    dl = DataLoader(tc.DomainDS("sinogram", ids, labels, ck["mean"], ck["std"]),
                    batch_size=256, num_workers=8)
    P, T = [], []
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for x, y in dl:
            P.append(torch.sigmoid(m(x.to(DEV))).float().cpu().numpy())
            T.append(y.numpy())
    return np.concatenate(P), np.concatenate(T)


def ece(p, t, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum():
            e += m.mean() * abs(p[m].mean() - t[m].mean())
    return e


def reliability(p, t, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() > 5:
            xs.append(p[m].mean()); ys.append(t[m].mean())
    return np.array(xs), np.array(ys)


def net_benefit(p, t, pts):
    N = len(t); prev = t.mean()
    nb, nb_all = [], []
    for pt in pts:
        pred = p >= pt
        tp = (pred & (t == 1)).sum(); fp = (pred & (t == 0)).sum()
        w = pt / (1 - pt)
        nb.append(tp / N - fp / N * w)
        nb_all.append(prev - (1 - prev) * w)
    return np.array(nb), np.array(nb_all)


def curve(d):
    txt = open(f"{d}/stdout.log").read()
    return [(int(a), float(b)) for a, b in
            re.findall(r"epoch (\d+)/\d+ .*?val_macroAUC=([0-9.]+)", txt)]


def main():
    labels = tc.load_labels()
    _, val = tc.split_ids(sorted(labels))
    res = {}
    for name, cfg in MODELS.items():
        P, T = infer(cfg, val, labels)
        aucs = {c: roc_auc_score(T[:, i], P[:, i]) for i, c in enumerate(tc.LABEL_COLS)
                if T[:, i].min() != T[:, i].max()}
        res[name] = dict(P=P, T=T, aucs=aucs,
                         brier=brier_score_loss(T[:, ANY], P[:, ANY]),
                         ece=ece(P[:, ANY], T[:, ANY]), cfg=cfg)

    print(f"{'metric':22s} {'baseline':>10s} {'go-for-broke':>13s}")
    print(f"{'macro AUC':22s} {np.mean(list(res['baseline']['aucs'].values())):10.4f} "
          f"{np.mean(list(res['go-for-broke']['aucs'].values())):13.4f}")
    for c in tc.LABEL_COLS:
        print(f"{'AUC '+c:22s} {res['baseline']['aucs'].get(c,float('nan')):10.4f} "
              f"{res['go-for-broke']['aucs'].get(c,float('nan')):13.4f}")
    print(f"{'Brier (any)':22s} {res['baseline']['brier']:10.4f} "
          f"{res['go-for-broke']['brier']:13.4f}   (lower=better)")
    print(f"{'ECE (any)':22s} {res['baseline']['ece']:10.4f} "
          f"{res['go-for-broke']['ece']:13.4f}   (lower=better)")

    fig, ax = plt.subplots(2, 2, figsize=(13.5, 10))
    # A: learning curves
    for name, cfg in MODELS.items():
        c = curve(cfg["dir"])
        ax[0, 0].plot([e for e, _ in c], [a for _, a in c], "o-",
                      color=cfg["color"], label=name, ms=3)
    ax[0, 0].set_title("A. Convergence (val macro-AUC vs epoch)", fontsize=11)
    ax[0, 0].set_xlabel("epoch"); ax[0, 0].set_ylabel("val macro-AUC")
    ax[0, 0].legend(fontsize=9); ax[0, 0].grid(alpha=0.3)
    # B: per-subtype AUC
    x = np.arange(len(tc.LABEL_COLS)); w = 0.38
    for k, name in enumerate(MODELS):
        vals = [res[name]["aucs"].get(c, np.nan) for c in tc.LABEL_COLS]
        ax[0, 1].bar(x + (k - 0.5) * w, vals, w, color=MODELS[name]["color"], label=name)
    ax[0, 1].set_xticks(x); ax[0, 1].set_xticklabels(
        [c[:8] for c in tc.LABEL_COLS], rotation=30, ha="right", fontsize=8)
    ax[0, 1].set_ylim(0.80, 1.0)
    ax[0, 1].set_title("B. Per-subtype AUC (subarachnoid = hardest)", fontsize=11)
    ax[0, 1].legend(fontsize=8)
    # C: reliability
    ax[1, 0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    for name, cfg in MODELS.items():
        xs, ys = reliability(res[name]["P"][:, ANY], res[name]["T"][:, ANY])
        ax[1, 0].plot(xs, ys, "o-", color=cfg["color"],
                      label=f"{name} (ECE {res[name]['ece']:.3f})")
    ax[1, 0].set_title("C. Calibration — 'any' (diagonal = perfect)", fontsize=11)
    ax[1, 0].set_xlabel("predicted probability"); ax[1, 0].set_ylabel("observed frequency")
    ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=0.3)
    # D: net benefit
    pts = np.linspace(0.01, 0.6, 60)
    for name, cfg in MODELS.items():
        nb, nb_all = net_benefit(res[name]["P"][:, ANY], res[name]["T"][:, ANY], pts)
        ax[1, 1].plot(pts, nb, "-", color=cfg["color"], lw=2, label=name)
    ax[1, 1].plot(pts, nb_all, "--", color="#999", lw=1, label="treat all")
    ax[1, 1].axhline(0, color="k", lw=1, label="treat none")
    ax[1, 1].set_ylim(-0.02, res["baseline"]["T"][:, ANY].mean() + 0.02)
    ax[1, 1].set_title("D. Clinical net benefit — 'any' (higher=better)", fontsize=11)
    ax[1, 1].set_xlabel("threshold probability"); ax[1, 1].set_ylabel("net benefit")
    ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=0.3)

    fig.suptitle("Beyond AUC: baseline aniso vs go-for-broke (aniso2 arch), 200k",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("metrics_compare.png", dpi=150)
    print("\nwrote metrics_compare.png")


if __name__ == "__main__":
    main()
