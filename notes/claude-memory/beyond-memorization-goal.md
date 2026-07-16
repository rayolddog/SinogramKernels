---
name: beyond-memorization-goal
description: "User's eventual goal: test strategies to extend training past the point where the net memorizes individual cases (which terminates generalizable learning). Cross-project pointers to the relevant docs/memories inside."
metadata: 
  node_type: memory
  type: project
  originSessionId: 17c5a547-9932-424a-a2ed-5480aecebd3d
---

Stated by the user 2026-07-15: after the stem-transfer study (see
[[stem-transfer-next-steps]] — whose headline is convergence RATE, not AUC), the
eventual goal is to test **strategies that extend useful training beyond
memorization onset** — the point where the network starts remembering individual
cases and generalizable learning stops.

**Why the current setup is a clean testbed:** `subset_dc` is patient-decorrelated
(memory `sinogram-patient-leakage` in the NewICH project dir:
`~/.claude/projects/-home-justinolddog-NewICH/memory/`) — 18,938 patients, 1.94
slices/patient, patient_overlap=0. So the val-AUC decline seen at ~ep12-14 in all
stem-transfer arms is TRUE case memorization, not leakage. Before that fix, every
val AUC in EXPERIMENTS.md was partly a memorization score.

**Outline drafted 2026-07-15: `~/SinogramKernels/EXPERIMENT_beyond_memorization.md`**
(tier 0 schedule controls → tier 1 single-lever screens via cached-features path →
tier 2 multi-seed winners → tier 3 recon-decoder). Design not frozen; not committed.

**Source material it consolidates:**
- `~/SinogramKernels/EXPERIMENTS.md` — wedge AUGMENTATION is the proven
  generalization lever in this project (§10-13); "Queued/open" lists a
  self-supervised/recon objective and a 750K full-data plan (leakage-prone unless
  patient-split).
- `~/SinogramKernels/EXPERIMENT_recon_objective.md` — arm iii (shared encoder +
  reconstruction decoder, multi-task) is explicitly flagged "may regularize" —
  a candidate anti-memorization strategy (auxiliary physics task anchors features
  to geometry, not case identity).
- `~/.claude/projects/-home-justinolddog/memory/project_maxvit_training.md` — the
  ich-maxvit recipe (focal + label smoothing 0.05 + CutMix p=0.5 + wd 0.05) is the
  user's existing heavy-regularization stack; current stem-transfer arms run
  nearly regularization-free (wd 1e-4, no aug) — large untested headroom.
- Data lever: subset_dc uses only 36,674 of 263,486 local slices; slices/patient
  cap can be raised (with z-gap) without adding patients.

**Target-hardware constraints (user, 2026-07-15):** ich-maxvit (maxvit_base,
118.7M) fit only bs=1 on the RTX 4090 vs bs≈26 on the Spark — fixed CUDA
overhead (context + optimizer moments + workspaces) eats a 24 GB card
disproportionately; expect less headroom than parameter-count math suggests.
Measured for the current 34M graft at bs 128: 8.9 GB allocated / 12.0 GB
reserved (fragmentation creeps; try expandable_segments). 4090 testing is
blocked in summer by apartment heat (upper 80s F); `nvidia-smi -pl 250` works
on the 4090 (unlike GB10) and roughly halves heat for ~10-15% speed.

**How to apply:** when the user returns to this, offer to draft
`EXPERIMENT_beyond_memorization.md` consolidating these levers into a controlled
design (the arms' history.json files give per-epoch curves to define
memorization-onset metrics). Related: [[dgx-spark-thermal-freezes]] for run
logistics.
