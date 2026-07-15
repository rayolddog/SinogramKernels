---
name: cached-features-platform
description: "Cached-stem-features platform SHIPPED 2026-07-15 (commits a152651, be4047b): cache_stem_features.py + train_cached_features.py. Tonight's chain command, equivalence guarantees, validation expectations, and known numeric caveats inside."
metadata:
  node_type: memory
  type: project
  originSessionId: 5e0c7dfc-4383-4dcc-a334-d2852770daad
---

Written by the desktop/Mac-side Claude session on 2026-07-15 (not on the
Spark). Context: John reframed the project's goal — he "wasn't searching for
the best AI model, but for the most efficient way to test different
architectures." Arm 1 + cached stem features is therefore no longer just a
control: it is the product. Train the projection-domain front-end ONCE, cache
its features, then iterate back-end architectures cheaply on modest hardware
(4090, even a laptop). This methods framing feeds the grant narrative.

**Two scripts shipped to the repo root, both smoke-tested end-to-end on CPU
with synthetic fixtures:**

1. `cache_stem_features.py` (commit a152651) — frozen arm0 stem -> one fp16
   `(256,32,32)` .npy per slice (~512 KB, ~18 GB full domain) in
   `~/sino_work/stem_features/<domain>/`, plus `meta.json` provenance.
   Replicates the arm-1 input path exactly: DomainDS loading, domain_stats
   over the FULL train split (the trainer's stats, NOT the checkpoint's),
   autocast forward, 34->32 center-crop; stem loaded through a full
   AnisoStemNet strict=True. Idempotent (skips existing, atomic tmp+rename
   writes), SoC-temp pauses (self-disable off-Spark: soc_temp() is nan),
   built-in verify pass recomputes a sample and exits 2 on mismatch.

2. `train_cached_features.py` (commit be4047b) — FeatureDS + FeatureHead
   (arm 1's exact transformer side: adapter -> MaxViT stages 2-3 -> norm ->
   head). Loads the arms' `tf_init_s<seed>.pt` by default (`--init-dir` =
   `~/sino_work/stem_transfer`); VERIFIED bit-identical init across scripts
   under the same tf_seed (max |diff| = 0.0). Same recipe / history.json
   schema / resume / thermal policy as the arms; cooldown self-disables
   off-Spark so 4090 runs don't sleep 120 s per epoch. `--n-train` draws the
   identical per-seed subset as train_stem_transfer.py; missing cached ids
   are a hard error; `--tag` names back-end variants.

**Tonight's sequence on the Spark (each step gated on the previous):**

    cd ~/SinogramKernels && git pull
    bash ~/sino_work/run_arms_chain.sh && \
      ~/miniconda3/envs/ich_train/bin/python ~/SinogramKernels/cache_stem_features.py && \
      ~/miniconda3/envs/ich_train/bin/python ~/SinogramKernels/train_cached_features.py

Finishes arm2 (was thermally stopped at epoch 20/30 the morning of Jul 15 —
clean stop, resumable), builds + verifies the cache, then runs the cached
arm-1 twin into `~/sino_work/stem_transfer/cached_s0/`.

**Validation expectation:** `cached_s0` should reproduce `arm1_s0`'s curve
(overlay via plot_convergence.py). fp16-rounding tolerance, NOT bit-exact —
two known, harmless gaps: (a) the cache carries training-time autocast
numerics while arm 1's evaluate() ran the stem in fp32; (b) AMP loss-scale
dynamics may diverge slightly. The `secs` and `cuda_peak_gb` columns of
cached-vs-arm1 history.json are the measured efficiency claim (epoch time,
peak memory).

**Caveat for the data-size sweep:** cached features embed FULL-train-split
stats; uncached --n-train runs recompute stats on the subset. Cached sweep
runs are internally consistent across arms but not bit-identical to uncached
subset runs — record which regime each run used.

**Interpretation guardrails for the arm results (from the Jul 15 review):**
the transfer signal is the early-epoch / small-N learning-curve gap, not
final AUC — a small final arm1-vs-arm3 gap is the ANTICIPATED pattern, and
frozen-arm1 matching trainable-arm3 while converging faster is the
supportive outcome. Don't compare arm2's epoch-20 AUC to 30-epoch arms;
single-seed gaps may sit inside seed spread. Wall-clock claims are
confounded by thermal clock-throttling (clock-based DVFS — GB10 has no watt
cap; "thermally induced clock throttling" is the defensible phrase), but
epoch-based convergence claims are untouched: throttling changes how fast
the arithmetic runs, not what the optimizer computes. Cooldowns are already
excluded from `secs` (timer starts after the previous cooldown).
