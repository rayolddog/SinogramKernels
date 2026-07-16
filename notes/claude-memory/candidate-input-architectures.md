---
name: candidate-input-architectures
description: "User's pipeline of candidate input layers to screen with the cheap-screening platform: image-compression transforms (user's background) and radon/raw-acquisition-space front-ends; clinical rationale = reconstruction-artifact-free detection as a complementary (not superior) reader."
metadata: 
  node_type: memory
  type: project
  originSessionId: 17c5a547-9932-424a-a2ed-5480aecebd3d
---

Stated by the user 2026-07-15. Background: before radiology, the user worked in
**image data compression**, and long ago **programmed the radon transform** to
simulate CT reconstruction and metal-artifact reduction; they tested a radon
front-end on MNIST (easy task, but it seeded the raw-acquisition-space idea).

**Candidate input layers they want to screen** (with the cached-features /
pretrained-stem platform validated by the stem-transfer study — see
[[stem-transfer-next-steps]]):
- **Compression-derived transforms** as stems (DCT/wavelet/learned-basis
  front-ends). Natural fit: compression and feature extraction both seek
  compact information-preserving representations. Hooks into the roadmap item
  already queued in EXPERIMENTS.md: "fixed-Gabor vs learned front-end ablation"
  (proposal Aims 2-3).
- **Radon / raw-acquisition-space** processing (the program's core thesis).

**Specific bases the user wants to try (2026-07-15):** (a) DCT encoding;
(b) the JPEG 2000 wavelet — that's **CDF 9/7 (Cohen-Daubechies-Feauveau,
lossy) / CDF 5/3 (lossless)**, not plain Daubechies; (c) Gabor wavelets
(user's personal favorite from their compression background — and arm0's
learned kernels already converged to Gabor-like anisotropic atoms, see
EQUATIONS.md §7). Key idea: if source data is stored JPEG 2000 (DICOM
transfer syntaxes 1.2.840.10008.1.2.4.90/.91), the DWT coefficients could be
**read from the codestream instead of computed** — compressed-domain
learning. Direct precedent: Gueguen et al. 2018, "Faster Neural Networks
Straight from JPEG" (DCT coefficients from the file). Caveat: check what the
local RSNA DICOMs actually store before banking on the free transform.

**Clinical framing the user holds (use this, don't overclaim):** AI on raw
data may not exceed or even match image-domain AI on average, but the
**occasional case** wins because raw data lacks reconstruction artifacts
(algorithm-dependent kernels, imperfect acquisitions, metal streaks — in image
domain metal destroys whole neighborhoods; in projection space it is a
localized trace corruption). Position raw-domain AI as a **complementary
second reader** (disagreement flags cases), not a replacement. Consistent with
Banner proposal §3 (upstream of the kernel/preference axis) and EXPERIMENTS.md
§13 (sinogram wedge robustness).

Also noted: gradient accumulation caused memory problems for the user in the
past (details unknown) — don't assume it is a free lunch on the 4090; see
4090 constraints in [[beyond-memorization-goal]].
