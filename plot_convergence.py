#!/usr/bin/env python3
"""
plot_convergence.py — AUC-vs-epoch curves across the stem-transferability arms.

The point of these plots is the SHAPE, not the endpoint. Transfer shows up as
faster convergence and better data-efficiency, and it often vanishes at long
training — so comparing only final AUC can hide a transfer that saved most of the
data (EXPERIMENT_stem_transferability.md, "What to measure"). Each arm is drawn
with its running best and annotated with epochs-to-90/95/99% of its OWN best, which
is the rate question stated directly.

Usage:
  plot_convergence.py --run NAME=DIR [--run NAME=DIR ...] --out fig.png
  (DIR is a train_twin.py --out dir containing <domain>/history.json)
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(d):
    hp = os.path.join(d, "history.json")
    if not os.path.exists(hp):
        hits = glob.glob(os.path.join(d, "*", "history.json"))
        if not hits:
            return None
        hp = hits[0]
    with open(hp) as f:
        return json.load(f)


def epochs_to(hist, frac):
    """First epoch reaching `frac` of the run's own final best AUC."""
    best = max(h["val_macro_auc"] for h in hist)
    for h in hist:
        if h["val_macro_auc"] >= frac * best:
            return h["epoch"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help="NAME=DIR; repeatable")
    ap.add_argument("--out", default="convergence.png")
    ap.add_argument("--title", default="Stem transferability: convergence to best val macro-AUC")
    a = ap.parse_args()

    runs = []
    for spec in a.run:
        name, _, d = spec.partition("=")
        h = load(d)
        if not h:
            print(f"[plot] SKIP {name}: no history.json under {d}")
            continue
        runs.append((name, h))
    if not runs:
        raise SystemExit("[plot] no runs with history")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for name, h in runs:
        ep = [x["epoch"] for x in h]
        auc = [x["val_macro_auc"] for x in h]
        run_best = [x["best_so_far"] for x in h]
        line, = ax1.plot(ep, auc, marker="o", ms=3, lw=1.2, alpha=0.75, label=name)
        ax1.plot(ep, run_best, ls="--", lw=1.0, alpha=0.5, color=line.get_color())
        ax2.plot(ep, [b / max(run_best) for b in run_best], lw=1.6, label=name,
                 color=line.get_color())

    ax1.set_xlabel("epoch"); ax1.set_ylabel("val macro-AUC")
    ax1.set_title("val AUC per epoch (dashed = running best)")
    ax1.grid(alpha=0.3); ax1.legend(fontsize=8)

    ax2.axhline(0.99, color="k", lw=0.6, ls=":", alpha=0.6)
    ax2.axhline(0.95, color="k", lw=0.6, ls=":", alpha=0.6)
    ax2.set_xlabel("epoch"); ax2.set_ylabel("fraction of that run's OWN best AUC")
    ax2.set_title("convergence rate (normalized to each run's own best)")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=8)

    fig.suptitle(a.title)
    fig.tight_layout()
    fig.savefig(a.out, dpi=140)
    print(f"[plot] wrote {a.out}")

    print(f"\n{'run':<28} {'best_auc':>9} {'@ep':>5} {'ep>=90%':>8} {'ep>=95%':>8} {'ep>=99%':>8}")
    summary = {}
    for name, h in runs:
        best = max(x["val_macro_auc"] for x in h)
        at = next(x["epoch"] for x in h if x["val_macro_auc"] == best)
        e90, e95, e99 = epochs_to(h, .90), epochs_to(h, .95), epochs_to(h, .99)
        print(f"{name:<28} {best:>9.4f} {at:>5} {str(e90):>8} {str(e95):>8} {str(e99):>8}")
        summary[name] = {"best_auc": best, "best_epoch": at,
                         "ep_to_90pct": e90, "ep_to_95pct": e95, "ep_to_99pct": e99,
                         "n_epochs": len(h)}
    sp = os.path.splitext(a.out)[0] + "_summary.json"
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[plot] wrote {sp}")


if __name__ == "__main__":
    main()
