# PLAN — motion-robustness experiment (SIIM-CAIMI 2026, due 2026-07-24)

## The claim
Simulated rigid patient motion is a **localized, structured** perturbation in the
sinogram (a band of detector-shifted views past an onset angle) but becomes a
**delocalized streak/blur** artifact once back-projected. Hypothesis: the
sinogram-domain classifier's learned features survive motion better than the
recon-domain classifier's. Both models were trained on CLEAN data, so both face
out-of-distribution inputs — the only variable is which *representation* carries
the corruption. Either direction is publishable (honest equipoise); we frame it
as a **trust-calibration** question: how silently does each model fail under
motion it never saw?

## The 2×2 (× severity) — all arms from the SAME measurements
|                | clean measurements    | motion measurements    |
|----------------|-----------------------|------------------------|
| sinogram model | clean sinogram        | motion sinogram        |
| recon model    | **iradon(clean sino)**| iradon(motion sino)    |

**Round-trip control (non-negotiable):** the recon model's clean arm is
`iradon(clean sinogram)`, NOT the original DICOM image — otherwise FBP round-trip
loss + domain shift confound the motion effect. Both columns then differ only in
motion.

**Primary endpoint:** difference-of-differences on any-ICH AUC,
`ddAUC = [AUC_recon(clean) − AUC_recon(motion)] − [AUC_sino(clean) − AUC_sino(motion)]`,
with a paired bootstrap CI over patients. `ddAUC > 0` ⇒ sinogram more robust.
Run a **severity sweep** (0/2/4/8 px peak) for a degradation *curve*, not a single
before/after — monotonic dose-response preempts "is this just noise?".

## Motion model (in `motion_experiment.py`, verified)
Rigid in-plane translation `(dx,dy)`, random direction, onset in the middle 60% of
views. Projection at angle θ shifts along the **detector axis** by
`s(θ) = dx·cosθ + dy·sinθ`; vacated cells zero-filled (air). `step` = move once and
hold; `drift` = ramp from onset (the shear-like variant). Same direction/onset
across severities per image (seeded per-id), only the peak differs. ≤8 px because
`circle=True` clips anatomy that slides past the inscribed circle.

## ⚠ Checkpoint reality — the one thing to fix before `infer`
`motion_experiment.py` was drafted against the local `model.py` stub
(`SinogramResNet` with an input BatchNorm). **The actual trained models are
different.** From `twin_common.py` / `train_twin.py` in this repo:

- **Model class:** `AnisoStemNet(n_out=6, stem_spec=...)`. Stem variants:
  `ANISO_STEM=[(15,60),(8,30),(6,15),(4,8)]` (default), `ISO_STEM`, `ANISO2_STEM`,
  `ANISO3_STEM`. **Confirm which variant each production checkpoint used** (it's in
  each run's `train.log` header / the run script). The stem-transfer study's arms
  differ here.
- **Checkpoint format:** `best.pt` = `dict(model=state_dict, opt=…, sched=…,
  scaler=…, …)`. Load `ck["model"]`, NOT `ck` and NOT `ck["state_dict"]`.
  Location: run dirs under `~/sino_work/…/{recon,sinogram}/best.pt`.
- **Normalization lives in the dataset, not the model.** `AnisoStemNet` has **no
  input BatchNorm**; `DomainDS` applies `(arr − mean)/std` with **per-domain
  train-set** mean/std (`domain_stats`), and some arms also apply a **spectral
  whitening filter**. `motion_experiment.infer` feeds RAW arrays → wrong. The
  motion arms must be normalized with the *same* mean/std (and whitening, if that
  checkpoint used it) the model trained under. Pull those values from the run's
  `train.log` line `[domain] stats mean=… std=…` (pin them; do not recompute on a
  different cohort).
- **Tensor source/cache:** `CACHE = $SINO_CACHE` (e.g. `~/sino_work/cache_dc` or
  `/mnt/mlcache/subset50k`), layout `CACHE/{recon,sinogram}/ID.npy`. The `inject`
  step reads clean **sinogram** tensors from there.

**FIXED — use `XFer/infer_twin.py` (written + self-tested 2026-07-22).** It does
all of the above and is *self-configuring*: `train_twin.save()` stamps each
`best.pt` with `mean`, `std`, and `arch`, so the adapter rebuilds the right
architecture (aniso / iso / aniso2 / aniso3 / smallcnn / seresnext50), loads
`ck["model"]`, standardizes with the checkpoint's own mean/std, and auto-loads
`whiten.npy` if it sits beside the checkpoint (matching `train_twin.evaluate()`
exactly). You do NOT have to hunt logs for mean/std. It writes the same
`results.csv` schema `motion_experiment.py stats` consumes. Keep
`cohort`/`inject`/`qc`/`stats` from `motion_experiment.py` as-is.

Self-test coverage (round-tripped on this laptop, torch 2.11): every arch builds
and loads its own `state_dict` cleanly; a wrong-arch load raises (so a bad
`stem_spec` guess fails loudly, not silently); whitening applies; `infer_arm`
returns valid probabilities. The ONE thing untested here is a *real production
checkpoint* — if any arm used `arch='aniso_cfg'` (ablation toggles not saved),
`load_state_dict` will raise a shape mismatch and you'll pass the exact
stem/strides/dilations; plain `aniso` twins load with no fuss.

## Cohort / the "test group"
- **One slice per patient** (the "uncorrelated set"): `cohort` dedups by PatientID
  when given `--dicom-dir`; without it, it warns and samples at slice level.
- Balanced any-ICH pos/neg, N≈1000 (inference is cheap; bigger N tightens the
  ddAUC CI).
- **OPEN — resolve on the machine:** is there a genuinely held-out **test group**
  with labels? RSNA `stage_2_test` labels were never in the standard Kaggle
  download. Check the patient-decorrelated split (`subset_dc`, 18,938 patients,
  `patient_overlap=0`) for a partition neither model trained/validated on. If no
  labeled test group exists, fall back to the unused labeled portion of
  `stage_2_train` — same validity for this purpose. **Whatever you pick must be
  disjoint from BOTH models' train+val ids** or the abstract is dead on arrival.

## Order of operations (on the apartment machine)
1. Locate both `best.pt` checkpoints; note each one's `stem_spec` + mean/std
   (+ whitening flag) from its `train.log`.
2. Confirm the held-out cohort (see above) — the one real showstopper.
3. `cohort` → `inject 0,2,4,8` → `qc` (eyeball the 4-panel: motion visible in the
   sinogram band, streaks in the recon).
4. `infer_twin.py --ckpt-recon .../recon/best.pt --ckpt-sino .../sinogram/best.pt`
   → `motion_experiment.py stats`. (Copy `motion_experiment.py` + `infer_twin.py`
   from `XFer/` up to the repo root first, so they import `twin_common` and
   `data_pipeline` — sanity: `python infer_twin.py -h` should print without an
   ImportError.)
5. Read the ddAUC table; write the abstract. Figures: the QC 4-panel + the
   per-domain degradation curve.

## Thermal note
Inference-only = light load vs the training crashes. Optional insurance:
`nvidia-smi -pl <~70% default W>` before starting; throughput barely changes,
thermal ceiling drops. `thermal_governor.sh` / `thermal_shutdown.sh` in the repo
already guard the machine.

## After it ships (spinoff, not for this abstract)
The `motion_arms/` output is a **labeled degraded-image training set** (severity,
direction, onset ground truth, both domains, free from simulation). Natural next
project: train a **degraded-image detector** / trust gate — hypothesis is motion
is easier to *detect* in the sinogram (crisp broken-sinusoid discontinuity at the
onset angle) than in the streaky image. Squarely in the human-AI-collaboration
lane (AI telling the human when NOT to trust the read).
