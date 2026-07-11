#!/usr/bin/env python3
"""multiseed_pooled.py — pool the noise-floor test and ask whether the
go-for-broke (gfb) architecture's edge over baseline separates from seed variance.

Configs (best_val_macroAUC per seed):
  base : abl_base (seed0) + seed_base_s1/2/3
  gfb  : abl_all  (seed0) + seed_gfb_s1/2/3
Reports per-config mean±sd, the paired per-seed gap, a paired t-test, and
Cohen's d. Reads best_val_macroAUC from each run's stdout.log (or train.log).
"""
import glob
import os
import re
import statistics as st

RUNS = {
    "base": {0: "abl_base", 1: "seed_base_s1", 2: "seed_base_s2", 3: "seed_base_s3",
             4: "seed_base_s4", 5: "seed_base_s5"},
    "gfb":  {0: "abl_all",  1: "seed_gfb_s1",  2: "seed_gfb_s2",  3: "seed_gfb_s3",
             4: "seed_gfb_s4",  5: "seed_gfb_s5"},
}


def read_auc(run):
    for name in ("stdout.log", "train.log"):
        p = os.path.join(run, name)
        if os.path.exists(p):
            vals = re.findall(r"best_val_macroAUC=([0-9.]+)", open(p).read())
            if vals:
                return float(vals[-1])
    # fall back to any log matching the run dir name
    for p in glob.glob(f"{run}*.log"):
        vals = re.findall(r"best_val_macroAUC=([0-9.]+)", open(p).read())
        if vals:
            return float(vals[-1])
    return None


def ttest_paired(d):
    n = len(d)
    if n < 2:
        return float("nan"), float("nan")
    m = st.mean(d)
    sd = st.stdev(d)
    if sd == 0:
        return float("inf"), 0.0
    t = m / (sd / (n ** 0.5))
    return t, m / sd  # t, Cohen's d (paired)


def main():
    seeds = sorted(set(RUNS["base"]) & set(RUNS["gfb"]))
    rows, paired = {}, []
    print(f"{'seed':>4}  {'base':>8}  {'gfb':>8}  {'gap':>8}")
    for cfg in ("base", "gfb"):
        rows[cfg] = {s: read_auc(RUNS[cfg][s]) for s in RUNS[cfg]}
    have = []
    for s in seeds:
        b, g = rows["base"].get(s), rows["gfb"].get(s)
        if b is None or g is None:
            print(f"{s:>4}  {str(b):>8}  {str(g):>8}  {'--':>8}  (incomplete)")
            continue
        gap = g - b
        paired.append(gap)
        have.append(s)
        print(f"{s:>4}  {b:>8.4f}  {g:>8.4f}  {gap:>+8.4f}")

    bvals = [rows["base"][s] for s in have]
    gvals = [rows["gfb"][s] for s in have]
    print("\n--- pooled ---")
    for cfg, vals in (("base", bvals), ("gfb", gvals)):
        if vals:
            m = st.mean(vals)
            sd = st.stdev(vals) if len(vals) > 1 else 0.0
            print(f"{cfg:>5}: n={len(vals)} mean={m:.4f} sd={sd:.4f} "
                  f"[{min(vals):.4f}, {max(vals):.4f}]")
    if len(paired) >= 2:
        t, d = ttest_paired(paired)
        print(f"\npaired gap (gfb-base): mean={st.mean(paired):+.4f} "
              f"sd={st.stdev(paired):.4f} n={len(paired)}")
        print(f"paired t={t:.3f}  Cohen's d={d:.3f}")
        # rough two-sided significance flag at n=4 (t_crit(df=3,.05)=3.182)
        crit = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}.get(len(paired) - 1)
        if crit is not None:
            verdict = "SEPARATES from seed noise" if abs(t) > crit else \
                      "WITHIN seed noise (gap not distinguishable)"
            print(f"vs t_crit(df={len(paired)-1},.05)={crit}: {verdict}")
    else:
        print("\nnot enough paired seeds yet.")


if __name__ == "__main__":
    main()
