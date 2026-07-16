# Future directions — speculation parking lot

Ideas worth keeping that are NOT yet designed experiments. When one matures,
it graduates to its own `EXPERIMENT_*.md` (or to EXPERIMENTS.md "Queued/open"
if it's a small concrete run). Rule of thumb: this file is allowed to be
wrong; experiment docs are not.

**Why this file exists:** the screening platform (pretrained/frozen stem +
cached features, validated by the stem-transfer study) makes each candidate
input architecture a few-epoch test instead of a 30-epoch commitment — so the
idea backlog, not compute, is becoming the bottleneck. Park ideas here before
they evaporate.

## Near — runnable with existing tooling

- **Fixed-basis stem screen (compression bases).** Three-way screen of fixed
  front-ends vs the learned arm0 stem as reference, all frozen, all via the
  cached-features path (~few epochs each):
  1. **DCT** (blockwise, JPEG-style);
  2. **CDF 9/7** — the JPEG 2000 lossy wavelet (Cohen-Daubechies-Feauveau;
     the DWT pyramid maps naturally onto the stem's 2,2,2,1 stride pyramid);
  3. **Gabor filter bank** (JBB's favorite from his compression background —
     and arm0's learned kernels already converged to Gabor-like anisotropic
     atoms, EQUATIONS.md §7, so this tests "did learning just rediscover the
     hand basis?").
  This subsumes the grant's queued "fixed-Gabor vs learned front-end"
  ablation (Aims 2-3). Judge by convergence rate + stability, per the
  program's headline metric.
- **Beyond-memorization levers** — designed, see
  `EXPERIMENT_beyond_memorization.md` (tiered: schedule controls →
  wd/label-smoothing/feature-mixup/aug/data-dose screens → multi-seed).
- **Instrumented re-runs of arms 1/3** with the new train-AUC/val-loss
  logging (added 2026-07-15), to answer "does the train/val split open later
  with a pretrained stem?" — cheap via cached features.
- **4090 portability demonstration.** Measured need at bs 128 is ~12 GB
  reserved (24 GB card); cached-features path needs less. Blocked seasonally
  by apartment heat; `nvidia-smi -pl 250` roughly halves the 4090's heat for
  ~10-15% speed. Doubles as the "train a transformer on lesser hardware"
  deliverable.

## Medium — needs new code or data work

- **Compressed-domain input (direct-read).** If source data is stored
  JPEG 2000 (DICOM transfer syntaxes 1.2.840.10008.1.2.4.90/.91), read the
  DWT coefficients from the codestream instead of computing the transform —
  precedent: Gueguen et al. 2018, *Faster Neural Networks Straight from
  JPEG* (DCT-from-file). **Caveat measured 2026-07-15:** local RSNA DICOMs
  are Explicit VR Little Endian (uncompressed), so for our training data this
  is a *deployment/efficiency* story (clinical PACS archives are often J2K),
  not a training-data shortcut.
- **Metal-artifact second-reader study.** The clinical thesis in its
  sharpest form: image-domain models lose whole neighborhoods to metal
  streaks; in projection space metal is a localized trace corruption that can
  be masked/inpainted (JBB has prior hands-on work here). Design would
  simulate metal traces, compare domain degradation curves — the §13 wedge
  methodology transplanted to metal. Positioning: **complementary second
  reader whose disagreements flag cases**, never a replacement claim.
- **Trace-tokenization ViT front end** (from EXPERIMENTS.md queued list) —
  Hough-style token enumeration; now cheaper to evaluate with the platform.

## Speculative — research bets

- **Raw-acquisition-space AI as artifact-free reader.** The program's long
  thesis: AI on raw projection data won't beat image-domain AI on average,
  but wins the *occasional case* by never seeing reconstruction artifacts
  (kernel choice, imperfect acquisitions, streaks). Every near-term result
  above (fixed bases, wedge/metal robustness, second-reader framing) is a
  stepping stone. Endgame requires actual scanner raw data (the grant ask).
- **Curvature measurement of learned kernels** — before ever claiming
  "curvelet," measure it (EXPERIMENT_recon_objective.md rigor note).
- **Compression-as-regularization** — if a fixed compression basis screens
  well, does its information bottleneck also delay memorization onset?
  Bridge between the fixed-basis screen and EXPERIMENT_beyond_memorization.

---
*Log of graduations:* (move lines here when an idea becomes an experiment doc)
