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
- `motion_experiment.py` — cohort / motion-injection / QC / stats (all ready).
  Its own `infer` targets the local `model.py` stub; **use `infer_twin.py`
  instead** for the real checkpoints.
- `infer_twin.py` — the production adapter (AnisoStemNet etc.). Self-configuring
  from each `best.pt` (reads mean/std/arch, auto-loads `whiten.npy`); writes the
  `results.csv` that `motion_experiment.py stats` consumes. Round-trip self-tested.
- `memory_sinogram_kernel_experiment.md` — snapshot of the project memory
  (the SIIM push + the degraded-image-detector spinoff are at the bottom).

## Use it
On the machine that has the checkpoints and tensors:
```
git pull
cp XFer/motion_experiment.py XFer/infer_twin.py .   # up to repo root, next to
                                                    # twin_common.py / data_pipeline.py
python infer_twin.py -h                              # sanity: no ImportError
# then follow PLAN.md
```
Push back anything worth keeping (the cohort CSV, stats.md, the QC figure) so the
laptop can pick it up for the abstract text.
