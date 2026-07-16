# Experiment: Extending training beyond memorization onset

**Status:** DRAFT outline (2026-07-15) — design not yet frozen, no code written.
**Machine:** DGX Spark (screens); target hardware for the workflow claim: RTX 4090.
**Authors:** J. Bramble, MD + Claude (Anthropic Fable 5).

## Question

Once a network starts remembering individual training cases, generalizable learning
stops — val AUC peaks and declines while train loss keeps falling. **Which strategies
move that onset later (or remove it), letting longer training keep buying
generalization instead of memorization?**

This is now cleanly testable for the first time in this project: the arms train on the
patient-decorrelated split (`subset_dc`, 18,938 patients, 1.94 slices/patient,
patient_overlap = 0 — see the `sinogram-patient-leakage` memory), so the observed val
decline is true case memorization, not patient leakage. Under the old slice-level split
this experiment would have been meaningless.

## Baseline (already collected, free)

Stem-transfer arms 1–3 (`~/sino_work/stem_transfer/arm{1,2,3}_s0/history.json`),
30/14/30 epochs, nearly regularization-free (wd 1e-4, no augmentation, no mixing):

| arm | best smoothed AUC | onset (peak) epoch | ep30 AUC |
|---|---|---|---|
| 1 (frozen pretrained stem) | 0.8618 | ~12 | 0.8149 |
| 2 (fine-tuned stem) | 0.8599 | ~12 (14 ep so far) | — |
| 3 (random stem) | 0.8631 | ~19 | 0.8283 |

Train loss falls monotonically (0.58 → 0.48 over ep9–14 in arm 2) while val declines —
the memorization signature. Noise yardstick from these runs: epoch-to-epoch val std is
~0.004 (pretrained arms) / ~0.020 (random arm); differences inside that band are noise.

## Metrics — define onset before testing levers

All from `history.json` (per-epoch `val_macro_auc`, train `loss`); smoothing kills the
single-epoch dips (arm 3 drops to 0.786 at ep12 then recovers):

- **Onset epoch** = argmax of the 3-epoch moving average of val macro-AUC.
- **Peak** = that smoothed maximum.
- **Retention** = mean smoothed AUC over the final 5 epochs ÷ peak (how much a run
  keeps after its best; baseline arm 1 ≈ 0.95, arm 3 ≈ 0.96).
- **Generalization-gap slope** after onset: d(val loss − train loss)/d(epoch).
  *Code needed:* log per-epoch **val loss** and (optionally) **train AUC** in the
  trainers — currently only train loss + val AUC are recorded.
- Optional memorization probe (tier 3): snapshot the per-slice train-loss histogram
  every 5 epochs; memorization = a growing near-zero mode.

**Success criterion:** a lever "extends training" if onset shifts later by more than
the seed spread AND peak does not drop; retention-only gains (flatter tail, same peak)
are reported as a separate, weaker outcome.

## Design — tiered single-lever screens, then combine

Fixed base recipe (comparability with arms 1–3): MaxViT-tiny graft, frozen arm0 stem,
bs 128, lr 3e-4, `tf_init_s0.pt` init, `subset_dc` split, 30 epochs unless stated.
Screens run **frozen-stem via the cached-features path**
(`cache_stem_features.py` → `train_cached_features.py`): no stem forward, transformer
side only — this is exactly the cheap-screening workflow the stem-transfer study
validated, and what makes a multi-lever sweep affordable (and 4090-portable).

### Tier 0 — schedule control (mandatory, cheap, do first)

The cosine LR schedule is tied to `--epochs`, so "training longer" silently changes
the schedule, and late-epoch decline under a *decaying* LR is already suggestive but
confounded. Two controls at baseline settings:

- **C1:** 60-epoch cosine (same everything else). Does onset scale with schedule
  length (→ schedule artifact) or stay at ~ep12 in absolute epochs (→ memorization)?
- **C2:** constant LR after warmup, 30 epochs. Establishes onset with no schedule
  confound. All later arms are judged against the matching control.

### Tier 1 — single levers, 1 seed each

| arm | lever | implementation notes |
|---|---|---|
| L1 | angular wedge augmentation | the project's one *proven* generalization lever (EXPERIMENTS.md §10–13). *Code needed:* port `--aug-angular` (train_twin.py) into the stem-transfer/cached path — for cached features this means caching K augmented variants or augmenting pre-stem (which forfeits the cache; decide at implementation). |
| L2 | weight decay sweep | wd ∈ {1e-3, 1e-2, 5e-2} vs baseline 1e-4. The ich-maxvit recipe (`project_maxvit_training` memory) used 0.05 successfully. Zero new code. |
| L3 | label smoothing 0.05 | from the ich-maxvit recipe. Trivial code. |
| L4 | feature-space mixup | CutMix on raw sinograms is physically dubious (a spliced sinogram is not a valid projection set); mixup on the *cached stem features* (manifold-mixup style) is natural and free with the cached path. |
| L5 | data dose | rebuild subset with `make_subset_decorrelated.py` raising slices/patient 2 → 4 → 8 (keep z-gap, patient split unchanged; only 36,674 of 263,486 local slices currently used). Tests whether onset is simply data-starvation — the cheapest possible "strategy" and the null against which the others must win. |

### Tier 2 — combine Tier-1 winners

Stack levers that individually moved onset; 3 seeds (`--seed`/`--tf-seed`) on the
winner(s) and on baseline to put error bars on the headline.

### Tier 3 — expensive / exploratory

- **Multi-task reconstruction decoder** — arm iii of `EXPERIMENT_recon_objective.md`,
  already flagged there as "may regularize": the auxiliary physics task anchors shared
  features to geometry rather than case identity. Requires the decoder build-out and
  gives up the feature cache; run only if Tiers 0–2 leave headroom.
- Per-slice train-loss histograms (the direct memorization probe above).
- Unfrozen-stem replication of the winning recipe (does the conclusion survive
  end-to-end fine-tuning? — arm-2-style run).

## Controls & rigor notes

- **Patient-decorrelated split is mandatory** for every run — under slice-level splits
  val AUC rewards memorization, inverting this experiment's sign. Never reuse the old
  `make_subset.py` splits here.
- `subset_dc` prevalence is 35.8% (patient-level, `--prefer-positive`), not RSNA's
  ~14% slice-level: fine for AUC, but revisit `pos_weight` if loss weighting changes,
  and no calibration/PPV claims.
- One seed per Tier-1 cell is a *screen*, not a result — anything within the arm-3
  noise band (±0.02) or the pretrained band (±0.004) needs the Tier-2 multi-seed pass
  before being believed.
- Keep bs 128 and the pinned `tf_init_s0.pt` unless a lever explicitly requires
  otherwise; if L5 changes N, remember cosine-per-epoch changes too (report onset in
  both epochs and optimizer steps).
- Don't conflate "later onset because learning is slower" with "more generalization":
  always report peak alongside onset. A lever that merely flattens everything (huge wd)
  will shift onset later and help nothing.

## Compute

- Cached-features path: transformer-only epochs, no stem forward — the screening
  workhorse; also the 4090-portable configuration (this experiment doubles as the
  demonstration for the lesser-hardware training claim).
- Full-pipeline epochs (L1 pre-stem variant, Tier 3 decoder) cost ~4 min + 2 min
  thermal cooldown on the Spark (see `dgx-spark-thermal-freezes` memory for run
  logistics — `--workers 2`, cooldown defaults, and the 2026-07-15 power-transient
  caveat: swap cable/outlet before multi-night chains).
- Rough Tier 0+1 budget at 30 epochs: ~8 runs; with cached features well under one
  night; Tier 2 adds ~6 runs (3 seeds × baseline + winner).
