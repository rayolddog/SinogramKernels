# XFer — cross-machine transfer for the SIIM-CAIMI 2026 motion experiment

Scratch drop-box for moving **memories, code, and the plan** between the laptop
(where this was drafted) and the apartment training machines (4090 / DGX Spark,
where the trained checkpoints and RSNA tensors live). Not part of the research
program proper — safe to delete after the experiment ships.

**Experiment (one line):** does an ICH classifier trained on **sinograms**
degrade more gracefully under simulated **patient motion** than one trained on
**reconstructed 2D axials**, when both see the *same* motion-corrupted
measurements? Target: SIIM-CAIMI 2026 abstract, **deadline 2026-07-24**.

## Contents
- `PLAN.md` — the ordered work plan, design decisions, and the **compatibility
  findings you must act on** (the checkpoints are `AnisoStemNet`, not the
  `model.py` stub `motion_experiment.py` was drafted against). **Read this first.**
- `motion_experiment.py` — cohort / motion-injection / QC / inference / stats.
  `cohort`, `inject`, `qc`, `stats` are model-agnostic and ready. `infer` targets
  the wrong model class and **must be swapped** per PLAN.md §"Checkpoint reality".
- `memory_sinogram_kernel_experiment.md` — snapshot of the project memory
  (the SIIM push + the degraded-image-detector spinoff are at the bottom).

## Use it
On the machine that has the checkpoints and tensors:
```
git pull
# follow PLAN.md; work OUTSIDE XFer/ (copy files up to the repo root as needed)
```
Push back anything worth keeping (the cohort CSV, stats.md, the QC figure) so the
laptop can pick it up for the abstract text.
