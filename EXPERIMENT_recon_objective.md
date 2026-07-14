# Experiment: Does the FBP ramp emerge only with a reconstruction objective?

**Status:** planned. **Machine:** DGX Spark. **Authors:** J. Bramble, MD + Claude (Anthropic Fable 5).

## Question

Is the sinogram-trained network learning the **transformed shape** of the abnormality
(reading it directly in projection space) rather than learning the **inverse transform**
(reconstruction)? Test by asking whether the filtered-back-projection **ramp** (|ω| high-pass:
oscillation + negative sidelobes) appears in the learned filters *only* when a reconstruction
objective is present.

## Design — three matched networks, same sinograms

| Arm | Objective | Structure |
|-----|-----------|-----------|
| i | detection only | encoder + classification head |
| ii | reconstruction only | encoder + decoder, **reconstruction loss** vs ground-truth image (learns sinogram→image inversion) |
| iii | both | **shared encoder** + detection head + reconstruction decoder (multi-task) |

## Prediction

The **ramp / inversion machinery** appears in the learned filters of **(ii) and (iii)** but
**NOT (i)**. Confirms a pure detector reads the projected abnormality *directly* (no
inversion), while asking for reconstruction forces the inversion structure to emerge.

## Why it matters

A **positive, controlled** demonstration on **non-whitened** inputs — sidesteps the whitening
confound in the current "no ramp in the classifier" evidence (whitening already does the
ramp's high-pass job, so its absence on whitened input is partly *expected*). Stronger and
more publishable than the current claim.

## What to measure / inspect

- Visualize the encoder's early-layer filters for each arm; look for ramp structure
  (oscillatory, negative sidelobes). Quantify — e.g. the radial frequency response of the
  filters (does it rise ~|ω|?).
- Reconstruction quality for arms ii/iii (PSNR/SSIM) as a sanity check that the decoder
  actually inverts.
- Detection AUC for arms i/iii.

## Rigor notes

- The decoder learns inversion **only with a reconstruction loss** — a decoder trained on
  classification won't.
- Arm iii (shared encoder) has a real **detection-vs-reconstruction representation tension**:
  reconstruction needs a near-sufficient bottleneck, detection wants aggressive compression →
  may regularize (help) or distract (hurt). Interesting either way.
- Do **NOT** claim the stem kernels are "curvelet" (curved) without *measuring* curvature —
  observed kernels are anisotropic / oriented (Gabor family); curvelet is a theoretical
  generalization under real geometry, not a demonstrated observation.
- **Leakage does NOT block this experiment:** it concerns what filters the network learns
  (weights), not generalization AUC, so the current slice-level split does not invalidate the
  ramp-emergence conclusion. (Fix the split for the *performance* claims, not this one.)

## Compute

- DGX Spark. Arms ii/iii add a decoder → larger, harder-to-estimate memory footprint
  (upsampling activations + image targets held alongside sinogram inputs). Budget
  conservatively; find batch size empirically (see the Spark unified-memory gotcha in
  `EXPERIMENT_stem_transferability.md`).
