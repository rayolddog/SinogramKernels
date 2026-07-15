---
name: arm0-training-recovery
description: "How to recover/resume the arm0 stem-transferability training run after a crash — exact paths, commands, and post-crash checklist."
metadata: 
  node_type: memory
  type: project
  originSessionId: f1bd8d71-05c3-415a-955c-fc9343c35534
---

**STATUS: COMPLETE 2026-07-14 23:06** — all 30/30 epochs done, exit 0, no freeze. Best val_macroAUC **0.8714 at epoch 22** (`~/sino_work/arm0_resnet_sino/sinogram/best.pt`). Ran with the 120 s epoch cooldowns (see [[dgx-spark-thermal-freezes]]); peaks stayed 87.5–92.7 °C with full recovery to idle every epoch, zero extra minutes needed. Do not resume — next steps are the arm1/arm2 transformer arms (stem transplant, must keep bs=128). The recovery procedure below is kept for future runs.

Recovery procedure for the arm0 training run (ResNet aniso_cfg on sinograms, the stem-transferability experiment), current as of 2026-07-14.

**The one command:** `bash ~/sino_work/run_arm0.sh` — idempotent; `--resume` picks up from `last.pt` (or starts at epoch 0 if absent). That script is the SINGLE SOURCE OF TRUTH for the config: `--arch aniso_cfg --stem tall --resdil --resk 3 --domains sinogram --epochs 30 --bs 128 --lr 3e-4 --wd 1e-4`. Do not reconstruct flags from memory — strides/dilations are not in the state_dict, so wrong flags load the checkpoint silently and train a different model (this already happened once; the flags in the script were forensically recovered by matching val_auc 0.8450466).

**All the moving parts:**
- Run script: `~/sino_work/run_arm0.sh` (sets `SINO_CACHE`, cds into the repo, execs python)
- Python: `~/miniconda3/envs/ich_train/bin/python` (conda env `ich_train`)
- Code: `~/SinogramKernels/train_twin.py` + `twin_common.py` (all changes committed as of 2026-07-14 late evening — cooldown + stability knobs are in commit c7beb8b)
- Data cache: `SINO_CACHE=~/sino_work/cache_dc` (internal NVMe, 19 GB, 36,674 .npy per domain, rebuildable with `~/SinogramKernels/zig/radon --list <dicom_list> --outdir <dir>` from `~/sino_work/subset_dc/dicom_list.txt`)
- Split: `~/sino_work/subset_dc/split.json` (patient-disjoint, from make_subset_decorrelated.py — mandatory, the default slice-hash split leaks patients)
- Output/checkpoints: `~/sino_work/arm0_resnet_sino/sinogram/` — `last.pt` (full resume state), `best.pt`, `history.json`, `train.log`

**Post-crash checklist (in order):**
1. `tail ~/memwatch.log` — the synced forensics log shows available memory + top consumers right before the freeze (logger auto-restarts via cron `@reboot`; see [[dgx-spark-thermal-freezes]]).
2. Confirm vLLM is NOT running (`lsof -ti :8000`); if it must run, its cap is 0.70 in `~/start_nemotron.sh`.
3. `free -h` — want ≥100 GB available before resuming.
4. `bash ~/sino_work/run_arm0.sh` and confirm the log line says `RESUMED from epoch N` with the expected best (was best=0.8450 at epoch 3 on 2026-07-14).

**Why:** the 2026-07-14 crashes destroyed the shell history holding the launch command; only the checkpoint forensics recovered it. This memory exists so recovery never depends on shell history again.
