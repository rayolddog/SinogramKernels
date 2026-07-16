---
name: stem-transfer-next-steps
description: "Stem-transferability experiment COMPLETE 2026-07-15: all arms 30/30. Headline (convergence rate): pretrained stem hits 0.855+ AUC in 1 epoch vs control's 4-6, 5x lower variance; unfreezing adds nothing. Best AUCs 0.8618/0.8599/0.8631. VRAM 8.9/12.0 GB @ bs128. Results and design inside."
metadata: 
  node_type: memory
  type: project
  originSessionId: 482f2046-d70e-4842-83ab-a178524c22cf
---

**Why this experiment exists (user, 2026-07-15): the headline metric is RATE OF
CONVERGENCE, not best AUC.** This tests a strategy for making transformer
training cheap on lesser hardware (e.g. the user's RTX 4090): if a pretrained
stem lets a candidate input architecture be judged in ~3-5 epochs instead of 30,
input-architecture ideas can be screened rapidly. Do not frame results around
peak AUC.

**COMPLETE 2026-07-15 23:25 — final results (all arms 30/30, seed 0):**
arm1 (frozen pretrained) best 0.8618 @ep12; arm2 (unfrozen @ep4) 0.8599 @ep12;
arm3 (random control) 0.8631 @ep19. **Convergence (the headline): pretrained
arms open at 0.8561 in epoch 1 — the control needs 4-6 epochs to get there —
with std 0.004 vs 0.020.** Unfreezing = no effect vs frozen (differences
within noise): frozen-stem screening (and hence the feature cache) is the
right mode. Val peak comes EARLIER with pretrained stem (ep12 vs ep19):
divergence onset tracks proximity to the ~0.86 data-limited ceiling, not stem
type. arm2 ep15-30 carry train_macro_auc/val_loss (added 2026-07-15): gap
0.078→0.185 monotone, val loss 1.34→2.64 while val AUC drifts 0.852→0.813.
VRAM @bs128: 8.91 GB alloc / 11.99 GB reserved (arm2); arm1 frozen 7.80 GB.
Artifacts: ~/sino_work/stem_transfer/{arm*_s0, convergence.png,
convergence_summary.json}. Open follow-ups: multi-seed + data-size sweep
(designed below), instrumented re-runs of arms 1/3 for cross-arm gap timing,
then [[beyond-memorization-goal]].

Original design notes follow (kept for the sweep follow-ups).

Next phase after arm0 completed (see [[arm0-training-recovery]]): test whether the
pretrained sinogram stem transfers to a transformer. Spec:
`~/SinogramKernels/EXPERIMENT_stem_transferability.md`. Code:
`~/SinogramKernels/train_stem_transfer.py` (smoke-tested end-to-end and
committed 2026-07-14, commit e3d5c98).

**Training plan (user's, set late 2026-07-14):** start in the EARLY AM of
2026-07-15 (possibly ~1 am if awake) while the apartment is cool; ambient is
expected to stay cool enough until ~9:30 am. The in-script temp checks are the
backstop — they stop a run cleanly if it gets too hot (see
[[dgx-spark-thermal-freezes]]).

**One command chains everything:** `bash ~/sino_work/run_arms_chain.sh`
(~7.5-8 h total; each arm ~2.6 h = 30 x (~3 min epoch + 2 min cooldown)).
Order arm1 -> arm3 -> arm2, deliberate: the critical comparison is
arm1-vs-arm3, so if heat ends the session early the decisive arms finish
first. A thermal termination stops the whole chain (don't start the next arm
hot). Rerun the same command when cool to resume; completed arms skip.
Individual scripts: run_arm{1,2,3}.sh. All idempotent (--resume). Outputs land in `~/sino_work/stem_transfer/arm{N}_s0/`
(train.log, history.json, last.pt, best.pt). `tf_init_s0.pt` in that dir pins the
transformer-side init shared by all arms — do NOT delete it between arms.

**Design decisions already made (don't re-litigate):** MaxViT-tiny (timm,
from-scratch, img_size=256) as the transformer, reusing the user's ich-maxvit
choice; arm0 stem deployed at fulres strides (2,2,2,1), output center-cropped
34->32 to match stage-2's grid; 1x1 conv adapter 256->128 (same fixed init all
arms); stages 2-3 + head kept (7/11 blocks); no explicit pos-enc (MaxViT rel-pos
handles it); recipe matches arm0 (bs 128 MANDATORY for comparability, lr 3e-4,
wd 1e-4, 30 epochs) plus 2-epoch LR warmup and grad-clip 1.0 (identical across
arms). Frozen stem = requires_grad False AND stem.eval() (BN stats frozen).

**Analysis command (verified working 2026-07-14):** per-epoch val macro-AUC is
in each arm's history.json; plot convergence with the BASE conda python
(ich_train lacks matplotlib):
`~/miniconda3/bin/python ~/SinogramKernels/plot_convergence.py
--run arm0=~/sino_work/arm0_resnet_sino/sinogram
--run arm1_frozen=~/sino_work/stem_transfer/arm1_s0
--run arm2_finetune=~/sino_work/stem_transfer/arm2_s0
--run arm3_control=~/sino_work/stem_transfer/arm3_s0 --out convergence.png`
It also prints/saves epochs-to-90/95/99%-of-own-best (the rate question).

**Encouraging smoke signal (not a result):** arm 1 with only 2,560 train slices
reached val macro-AUC 0.8322 in ONE epoch (arm0 needed ~3 epochs on 33k slices
to pass that). Analysis when real runs finish: compare val-AUC-vs-epoch curves
across arms (the transfer signal is curve SHAPE / early-epoch gap, not final
AUC); then the data-size sweep via `--n-train` (10k/25k/50k) and multi-seed via
`--seed`/`--tf-seed`.
