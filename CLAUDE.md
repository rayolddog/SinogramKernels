# SinogramKernels — project instructions for Claude Code

Sinogram-domain (projection-space) intracranial-hemorrhage detection on RSNA head
CTs. Research program behind the Banner grant proposals in this repo. User:
J. Bramble, MD (radiologist; JBB in docs).

## Reorient first (the user has lost sessions to machine crashes)

At the start of a session, before proposing work, reconstruct where things stand:

1. The auto-memory index (loaded with this file) is the curated state — trust it.
   Snapshots live in `notes/claude-memory/` (live copies:
   `~/.claude/projects/-home-justinolddog-SinogramKernels/memory/`). Sync + commit
   the snapshot when memory changes materially.
2. `git log --oneline -5` — recent commits are the work log.
3. Is anything running? `pgrep -af 'train_|run_arm'`; newest logs:
   `ls -t ~/sino_work/*.log | head` and each run dir's `train.log`.
4. If the machine just crashed: `journalctl --list-boots` (abrupt journal end =
   hard crash), `~/sino_work/boot_monitor/` watcher logs, and the
   `dgx-spark-thermal-freezes` memory before re-diagnosing anything.
5. Then state, in a few lines: what finished, what's partial, the single next step.

The user occasionally shifts gears mid-session to other projects (other folders,
e.g. ~/NewICH); some context lives in those folders' own memory dirs —
relevant here: `sinogram-patient-leakage` in the NewICH memory dir.

## Current state (update this section when it changes; dated 2026-07-15)

- arm0 (ResNet on sinograms): complete, best val macro-AUC 0.8714 @ ep22.
- Stem-transfer study (EXPERIMENT_stem_transferability.md): arm1 30/30 (best
  0.8618 @ ep12), arm3 30/30 (0.8631 @ ep19), **arm2 at 14/30 (0.8599 @ ep12) —
  resumable**: `bash ~/sino_work/run_arms_chain.sh` (idempotent, skips complete
  arms). Convergence verdict so far: pretrained stem = epoch-1 AUC 0.856 vs
  random 0.830, and 5x lower variance — the rate/stability win is the result.
- Cached-features platform shipped (cache_stem_features.py,
  train_cached_features.py); validation = cached_s0 should overlay arm1_s0.
- Next experiment drafted, not started: EXPERIMENT_beyond_memorization.md.
- Machine issue OPEN: 3 hard crashes 2026-07-15, two at GPU-load onset with cool
  temps — power-transient suspect; user should swap power cable/outlet before
  long chains. `startup-mem-watch.service` logs the first 5 min of every boot.

## Non-negotiable methodology (do not re-litigate; details in memory files)

- **Headline metric is RATE OF CONVERGENCE, not peak AUC.** The program goal is
  cheap screening of input architectures feeding a transformer on lesser
  hardware (user's RTX 4090). Never frame results around best-AUC alone.
- **Patient-decorrelated splits are mandatory** for any performance/convergence
  claim: use `~/sino_work/subset_dc` (18,938 patients, patient_overlap=0, built
  by make_subset_decorrelated.py). Slice-level splits (make_subset.py) leak
  patients — their val AUC partly measures memorization. subset_dc prevalence is
  35.8% (patient-level), not RSNA's ~14%: fine for AUC, not for calibration/PPV.
- **Comparability across arms:** bs 128, lr 3e-4, `tf_init_s0.pt` pinned (never
  delete/regenerate between arms), same split file.
- Terminology: "stem" = input/encoding layers (front-end), never "top layers".

## DGX Spark (spark-8d9d) logistics

- Thermal: sustained load hits 95-97C SoC and hard-freezes. Keep `--workers 2`,
  `--gpu-mem-frac 0.67`, and the default epoch cooldown (`--cooldown 120`,
  `--cool-target 70`) — decided with data, do not trim.
- `nvidia-smi` memory fields read [N/A] on GB10 (unified memory); temps via
  /sys/class/thermal. Memory is never the crash cause (peak CUDA 1.4 GB/121 GB).
- No passwordless sudo and no TTY for it: use `timeout 60 pkexec <cmd>` (GUI
  polkit dialog). For interactive logins have the user run `! <command>`.
- Live data cache: `~/sino_work/cache_dc` via `SINO_CACHE`.
- Long runs: everything must be `--resume`-safe; chain scripts already are.

## Layout

- Repo (code + experiment docs + EXPERIMENTS.md log): `~/SinogramKernels/`.
  EQUATIONS.md = math notation; grant context in Banner_*.md.
  FUTURE_DIRECTIONS.md = idea parking lot (allowed to be wrong); park the
  user's speculation there, graduate matured ideas to EXPERIMENT_*.md docs.
- Run scripts + outputs: `~/sino_work/` (run_arm{0..3}.sh, run_arms_chain.sh;
  outputs `stem_transfer/arm{N}_s0/`: train.log, history.json, best/last.pt).
- Convergence analysis: `~/miniconda3/bin/python plot_convergence.py --run
  name=dir ...` (base env; ich_train lacks matplotlib). Training env: conda
  `ich_train`.

## Working style

- Commit messages: work-log style (see git log); include memory-snapshot syncs.
- Update the "Current state" section above and the memory files at natural
  checkpoints (run finished, diagnosis changed, experiment drafted) — the user
  relies on them to recover from crashes.
