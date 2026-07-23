---
name: project-sinogram-kernel-experiment
description: "JB's shareable research proposal (for his daughter, UW MD-PhD) — autocorrelation-derived convolution-kernel shapes for sinogram vs reconstructed-CT deep learning; Radon-simulation convergence comparison"
metadata: 
  node_type: memory
  type: project
  originSessionId: 28c170dd-e280-4c97-903d-5d25e1712c6c
  modified: 2026-07-22T22:37:06.963Z
---

**Recorded 2026-07-08 at JB's request, explicitly to share with his daughter**
(MD-PhD student at UW-Madison, AI-on-medical-images; see [[user-origin-and-refocus]]).
A self-contained, thesis-scale research proposal extending the sinogram/projection-
domain-AI direction ([[project-ich-detection-approaches]] sinogram thread).

## Core hypothesis
The optimal convolution-kernel SHAPE for deep learning is set by the data's
autocorrelation (classical Wiener / matched-filter theory: the optimal linear
kernel matches the data's second-order statistics). Therefore the optimal kernel
shape DIFFERS between domains:
- **Reconstructed CT** — noise is RED (correlated, colored by the ramp filter +
  back-projection: CT noise texture, streaks). Optimal kernel is complex; problem
  is worse-conditioned. Standard square kernels are a mismatch.
- **Sinogram (Radon) data** — noise is ~WHITE (Poisson per detector element,
  uncorrelated before reconstruction), and the signal has clean sinusoidal
  structure. JB predicts the optimal kernel is **ELLIPTICAL, or CROSS-SHAPED with
  elliptical arms** (the cross = the two principal axes, projection-angle and
  detector-position; the +/- lobes = positive and negative correlations; elliptical
  because correlation length differs along vs across the sinusoidal ridges).

## Proposed experiment
1. Take axial CT sections from a public dataset; **Radon-transform** them to
   simulate sinograms (e.g. scikit-image `radon`/`iradon`).
2. Train small CNNs and **compare rate of convergence**: a standard **square**
   convolution kernel vs the **shaped** (elliptical / cross-with-elliptical-arms)
   kernel on the sinogram data.
3. **Run the Fourier autocorrelation** (Wiener-Khinchin: power spectrum = FT of
   autocorrelation) of the sinogram data to CONFIRM the predicted elliptical /
   cross shape independently of training.
Success = the empirically fastest-converging kernel shape MATCHES the shape the
autocorrelation predicts, AND the shaped kernel beats the square kernel in
convergence rate / parameter efficiency on sinogram data.

## Refinements to fold in (from the 2026-07-08 discussion)
- Two independent confirmations (autocorrelation-predicted shape + training-
  convergence-preferred shape) that AGREE is the strong result — design for both.
- Subtlety: a square CNN kernel already *learns* arbitrary shapes within its
  window (weights can go to zero). So the real comparison is **shaped-prior vs
  unconstrained square**, judged on convergence SPEED and parameter efficiency,
  not just final accuracy.
- WHERE the autocorrelation is computed matters: raw projections are mostly
  positively correlated (blurry); the NEGATIVE (cross) lobes appear after
  ramp-filtering / with edge structure. Compute both to see the +/- structure.
- The sinogram autocorrelation may reveal the sinusoidal ridge structure more
  than a clean cross — let the empirical 2D autocorrelation adjudicate.

## KEY REFINEMENT (JB's prior work, recalled 2026-07-08) — curved sinusoidal trajectories
The kernel shape is not a static cross; it follows the sinusoidal PATH a point
traces through the sinogram. Radon geometry: a point at image polar coords
(r, φ) maps to s(θ) = r·cos(θ − φ). So:
- **dead-center point (r=0): a STRAIGHT horizontal line** through the sinogram;
- **off-center points: sinusoids** of amplitude = radius r, phase = angular
  position φ (bigger radius → taller sinusoid).
Therefore the optimal detection kernel is a **curved (sinusoidal-arc) matched
filter**: a strong curved ridge (positive correlation) along the trajectory,
flanked by a curved GRADIENT lobe (negative correlation) from the ramp/high-pass
content. Two consequences:
1. **The sinogram is NOT Cartesian shift-invariant.** A point's signature
   curvature depends on WHERE it is (radius→amplitude, phase→shift), so a fixed
   square kernel is fundamentally mismatched, and the right kernel is curved AND
   position-dependent. Architectural responses: deformable/steerable convolutions,
   a curvature filter bank, or a coordinate transform that straightens sinusoids
   (cf. the sinogram-unwrap/Möbius thread).
2. **The sinusoidal-arc matched filter IS back-projection.** Reconstructing a
   point = summing the sinogram along its sinusoid; the ramp pre-filter supplies
   the negative flanking lobes. So a CNN learning "optimal" sinogram kernels is
   essentially re-learning filtered back-projection — which suggests embedding
   analytic back-projection as a known-operator layer and learning only the
   residual (Maier et al. "known operator learning"). The experiment's learned
   kernels should look like sinusoidal arcs; if they do, that is the confirmation.

## Gabor-emergence: the deepest confirmation route (JB, 2026-07-08)
Empirical fact: CNNs trained on natural images reliably learn **Gabor-like
filters in the first conv layer** (AlexNet; and V1 simple cells are Gabor —
Hubel-Wiesel / Daugman). WHY: efficient-coding theory (Olshausen & Field 1996) —
the optimal sparse code for natural-image STATISTICS is Gabor. That is exactly
this proposal's thesis (data statistics fix the optimal kernel) already
confirmed for one dataset. So the sharp, testable PREDICTION: because sinograms
have DIFFERENT statistics (sinusoidal-arc trajectories + white Poisson noise, vs
natural images' ~1/f oriented-edge structure), a CNN trained on SINOGRAMS should
NOT learn Gabor first-layer filters — it should learn **sinusoidal-arc /
back-projection-like** filters. A third confirmation route for the experiment:
compare the EMERGENT first-layer filters on reconstructed-CT (expect Gabor-like)
vs sinogram data (expect sinusoidal-arc). If they differ as predicted, strong
result. (Note the through-line: Gabor invented holography — JB's origin — AND
the wavelet AND the optimal space-frequency uncertainty limit = the DK
Heisenberg-as-bandwidth thread. See [[user-origin-and-refocus]], [[project-heisenberg-bandwidth-nyquist]].)

## FOCUS CORRECTION (JB, 2026-07-08): the experiment is about KERNEL SHAPE, not performance
JB clarified his primary interest is **the shape of the emergent convolution
kernels** (representation geometry), NOT beating reconstructed-image classification
accuracy. He does **not expect** the sinogram to outperform — the pathology
(hemorrhage) is a compact LOCAL blob in the reconstructed image but is SMEARED
along a sinusoidal trajectory (delocalized across all angles) in the sinogram, so
classification is intrinsically harder there; and reconstruction is a nearly
information-preserving linear operator, so both domains hold the same information.
=> The deliverable is the FILTER-SHAPE comparison: do sinogram-trained first-layer
kernels take the **sinusoidal-arc / back-projection** form the Radon geometry
predicts, vs **Gabor** on reconstructed images? Make it rigorous by (a) visualizing
learned first-layer kernels, (b) fitting them to parametric templates (Gabor params
vs sinusoidal-arc params) for goodness-of-fit, (c) comparing to the data
autocorrelation (the two-independent-confirmations design). The noise/dose machinery
below is SECONDARY (it affects shape via efficient coding, but JB is not chasing a
performance win); keep it only insofar as noise realism affects the emergent shape.

## Why no clean-data win — and the ARTIFACT regime where sinogram SHOULD win (JB, 2026-07-08)
Clean-data reason (JB's actual point; corrects an earlier "delocalization" guess):
the true sinogram's real advantages — **freedom from reconstruction artifacts** and
**greater bit depth of raw detector measurements** — are ALREADY lost in RSNA's
reconstructed+quantized images, so a forward-Radon simulation can't exhibit them.
Hence parity expected on clean data (test kernel SHAPE there).

**The extension (JB's, and it's the strong result):** ADD artifacts *in the
sinogram domain, where they physically originate*, and the sinogram approach can
OUTPERFORM the reconstructed-image method:
- **Motion artifact = a SHEAR** in the sinogram (projections after motion are of a
  displaced object → a discontinuity/shift in the sinusoidal trajectories). In the
  image it becomes global blur/ghosting/streaks.
- **Metal artifact = a very dense point TRAJECTORY** (metal's high-attenuation
  sinusoid + photon starvation). In the image it becomes the classic star-burst
  streaks radiating everywhere.
Mechanism: **back-projection converts a LOCALIZED sinogram corruption into a GLOBAL
image artifact** (it smears the corrupted projections along every ray). So working
PRE-reconstruction keeps the corruption contained and detectable/correctable → the
sinogram network wins on artifact-corrupted cases. **This is literally JB's own
metal-artifact-reduction insight (Hendee-era; now standard MAR: segment the metal
trace in the sinogram, inpaint, reconstruct) recast as deep learning** — full
circle to [[user-origin-and-refocus]]. Two-regime experimental story: CLEAN data →
kernel-shape test, parity expected; ARTIFACT data (add shear / dense trajectory) →
robustness test, sinogram expected to win. Physically clean to simulate: inject the
shear or dense sinusoid in the sinogram, then feed both branches.

## MAR-by-artifact-modeling idea (JB, 2026-07-08) + the beam-hardening wall
JB's alternative MAR: reconstruct/segment the high-density METAL from the original
sinogram → forward-model its **artifact spread** in the image (back-projection of
the metal's projections) → **subtract that spread** from the reconstructed image.
Keeps real anatomy + metal, removes only the streak spread (vs inpainting, which
discards the metal data). **Why it works for the LINEAR part but not beam
hardening:** the geometric spread is a linear operation (exactly modelable/
subtractable if the system were monochromatic), but **beam hardening is NON-LINEAR
and path/energy-dependent** — a polychromatic beam is preferentially stripped of
low-energy photons through metal, so the effective attenuation drops along the
path, coupling non-linearly with every other ray sharing that path. It is not an
additive artifact you can isolate and subtract → "hard to deal with." Two right
responses: (1) **known-operator / hybrid deep learning** — analytic model for the
linear geometric spread (JB's subtraction), LEARNED correction for the non-linear
beam-hardening residual (exactly what nets are good at, what analytic subtraction
can't reach); (2) **spectral measurement** — dual-energy / photon-counting CT
resolves the energy bins and corrects beam hardening at the source (ties to the
photon-counting raw-data thread in [[project-ich-detection-approaches]]).

## Concrete design (JB, 2026-07-08) + one load-bearing refinement
JB's plan: **ResNet** (he has good ResNet training experience), first-layer conv
**14×14**, then **7×7**, then **5×5** deeper — larger kernels justified because a
sinogram point's signature (its sinusoid) is NON-local, spanning the angular
axis, so small kernels can't see the trajectory. **Same architecture on both
reconstructed axials and Radon-transformed axials** (the controlled comparison —
vary only the input domain). **Kaggle/RSNA ICH dataset** (per-slice 2D, large,
multi-label — good fit; each axial → its 2D sinogram via `radon`).

**LOAD-BEARING REFINEMENT (must do or the core hypothesis isn't tested):** RSNA
images are already RECONSTRUCTED. A naive forward-Radon of them gives a clean
re-projection that INHERITS the reconstructed (red) noise — so the sinogram
branch would NOT get the white-Poisson-noise advantage the whole hypothesis
rests on. Correct pipeline: `noisy_sino = Radon(clean_img) + Poisson_noise`
(simulate photon counting at a chosen dose); then **image branch = FBP(noisy_sino)**
[red, ramp-colored noise] vs **sinogram branch = noisy_sino** [white Poisson].
Both derive from the SAME noisy sinogram → identical information, differing only
in domain/noise-correlation. Sweep DOSE: the hypothesis predicts the sinogram
advantage GROWS as dose drops (lower dose = more noise = noise-conditioning
matters more). Other notes: sinogram axes (angle vs detector) are NOT equivalent
like image axes — square kernels treat them the same, which is part of what's
tested; use adaptive pooling (image vs sinogram tensors differ in shape); ResNet's
aggressive early stride/maxpool may destroy fine sinogram structure — keep it
identical across branches but expect it may disadvantage the sinogram branch (a
finding, not a bug). Metrics: **convergence rate** (headline) + final accuracy +
**visualize emergent first-layer filters** (Gabor-like on reconstructed vs
sinusoidal-arc on sinogram — the deepest confirmation).

## Kernel-SIZE adequacy (JB asked 2026-07-08: is 14×14 enough?)
Quantitative answer: a single kernel can NEVER hold the full sinusoid (spans the
whole angular axis); the net builds it through depth. At 14×14 the LOCAL curvature
is subtle — sagitta ≈ |s|·(Δθ)²/8 ≈ ~2 px for a peripheral point (r~256) on a
180-angle sinogram — so **layer-1 kernels may still come out Gabor-ISH (oriented
segments)**, distinguished from natural-image Gabors mainly by a DIFFERENT
ORIENTATION DISTRIBUTION (biased to steep peripheral-trajectory slopes), not by
obvious curvature. Clearly-curved (sinusoidal-arc / back-projection-like) kernels
emerge with a LARGER first kernel (~20-30 px to get ~5+ px sagitta, scales with
angular sampling) or in DEEPER layers. => Better design: **sweep first-layer
kernel size (7/14/21/28)** and find the SCALE at which learned kernels transition
Gabor→arc (that transition scale is itself a result, and should track angular
sampling per the sagitta formula). Also: try ANISOTROPIC kernels (θ vs s axes
differ), DILATED convs (capture curvature over wide angular span cheaply), and
ANALYZE DEEPER LAYERS (arc-integration shows up there even if layer 1 stays
Gabor-ish). Distinguishing metric even at layer 1: orientation-distribution of
kernels (sinogram = structured/slope-biased vs natural-image = ~uniform).

## Implementation on JB's 4090 (2026-07-08)
Large-kernel cost is QUADRATIC in K (K² per output px per channel-pair), NOT
exponential — and it's TRIVIAL in a FIRST layer (1 input channel): a 30×30 stride-1
first conv on a 512² single-channel sinogram is ~30 GFLOP/image, sub-ms on a 4090
(batch 32 ~10 ms). The historical pain was large kernels used DEEP (in_ch×out_ch
multiplies K²) — which is exactly why JB's taper (30→7→5) is right: big kernels
early (cheap), small deep (where channels make big kernels expensive). **KEEP
STRIDE 1** (JB's preference, correct here): stride 4 saves 16× but you don't need
it in layer 1, and it destroys the fine sinogram structure that matters (steep
peripheral trajectories, motion EDGES). Contingencies: bottleneck will be the deep
layers (economize there); if MEMORY-bound not compute-bound, use smaller batch or
FP16/mixed-precision (4090 tensor cores) BEFORE stride. Note: stride doesn't change
learned kernel SHAPE (still a 30×30 to visualize), only downstream resolution.
30×30 is also a better size than 14×14 for seeing curvature in layer 1 (per the
sagitta calc above). Motion detection: edge-emphasizing kernels ANISOTROPIC across
θ detect the motion shear (edge detection in sinogram = motion detection).

## Hardware + kernel-taper + non-square implementation (JB, 2026-07-08)
**Hardware:** 4090 (24GB, fast) for iteration; **DGX Spark / GB10 (128GB unified,
slower bandwidth)** as the MEMORY fallback when large-kernel/stride-1/big-image
activations exceed 24GB — capacity not speed (conv is often bandwidth-bound).
**Kernel step-down:** what matters is receptive field in INPUT coords; each 2×
downsample doubles a fixed kernel's input coverage → rule: **halve kernel per 2×
downsample** (30→15→7-8→5→3) to hold input-space window ~constant. Big kernels
early (cheap: few channels; where the raw trajectory lives) → taper deep (where
channels multiply K² and where downsampling already grows RF to span the full
sinusoid). The deep taper is compute/capacity optimization; the FIRST-layer size
is the scientific knob. Compare the halve-per-downsample taper vs the simple
"big 30×30 stem + standard 3×3 ResNet blocks" baseline.
**Non-square kernels — NO CUDA needed:** rectangular/anisotropic = native
`Conv2d(kernel_size=(H,W))` + tuple stride/dilation; arbitrary SHAPES (elliptical/
cross/sinusoidal-arc) = **weight masking** (multiply weights by a fixed or
learnable 0/1 mask, pure PyTorch, differentiable); curved/adaptive = **deformable
conv** (`torchvision.ops.deform_conv2d`, prebuilt); wide RF cheap = dilated conv.
Only a truly novel op needs custom code — and then **Triton**, not raw CUDA.
Whole kernel-shape space is explorable in pure PyTorch on the 4090.

## Multi-task extension: sinogram MAR + detection in ONE net (JB, 2026-07-08)
Key asymmetry JB spotted: the sinogram domain has a TASK-DEPENDENT difficulty
profile — **detection is HARDER** there (hemorrhage delocalized along its
sinusoid) but **metal-artifact reduction is EASY** (metal = a localized, extreme-
value sinusoidal trajectory, trivially thresholdable — vs global streaks in the
image). So do BOTH as MULTI-TASK: shared sinogram encoder → (a) classification
head (ICH, BCE) + (b) MAR decoder head (U-Net-style: corrupted sinogram → clean
sinogram, or metal mask). MAR as an auxiliary task can REGULARIZE/improve the
detection encoder, hedging the sinogram's detection disadvantage. Honesty:
LOCALIZATION + basic inpainting are easy; the NON-LINEAR beam-hardening/photon-
starvation residual is where the LEARNED net earns its keep (classical MAR
can't) — and metal occludes anatomy (photon starvation) that must be INPAINTED,
not recovered (fundamental MAR limit). Dynamic range: metal's extreme attenuation
is exactly why the float32 sinogram matters (12-bit clips it) — and the extreme
values make metal EASY TO FIND (a feature, not just a problem). **Training data
comes free from the artifact-simulation setup:** clean image → clean sinogram
(MAR target) + inject metal → corrupted sinogram (input); reconstructed branch =
iradon(corrupted). So the same simulation that runs the artifact experiment also
supplies paired MAR ground truth. Payoff: one sinogram-domain net that cleans
metal AND detects hemorrhage — a demonstrable advantage the image domain can't
match (image-domain MAR fights global streaks). Ties to JB's MAR legacy.

## ACTIVE PUSH (2026-07-22): SIIM-CAIMI 2026 abstract, deadline 2026-07-24
Raw-CT proposal got no takers → JB pivoted to completing the MOTION-robustness
experiment (the artifact-regime win predicted above) for SIIM-CAIMI 2026
(Philadelphia, Oct 26-27; abstract due Jul 24). Both domain models are ALREADY
TRAINED (checkpoints on the apartment machines — 4090/DGX; JB checking; prior
thermal-throttling issues, but this run is inference-only so light load).
**`motion_experiment.py` written+smoke-tested 2026-07-22** in ~/Desktop/
SinogramKernels (NOT a git repo yet — fix before submission): subcommands
cohort/inject/qc/infer/stats. Key design decisions (Claude, JB agreed):
- **Round-trip control:** recon model's CLEAN arm = iradon(clean sino), NOT the
  original image — isolates motion from FBP round-trip loss + domain shift.
- **Physical motion model, not pure shear:** rigid translation (dx,dy) at random
  onset angle; detector-axis row shift s(θ)=dx·cosθ+dy·sinθ; zero-fill = air
  (JB's air-gap instinct); 'drift' mode = the shear variant. Same direction/onset
  across severities per image (seeded per-id rng), only peak differs.
- **Severity sweep** 0/2/4/8 px (≤8 px: circle=True clips anatomy beyond that);
  degradation CURVES, not single before/after.
- **Endpoint:** ddAUC = dAUC(recon)−dAUC(sino) on any-ICH, paired bootstrap CI.
- **Cohort:** one slice per patient (needs --dicom-dir for PatientID; else warns),
  balanced any-ICH pos/neg, N~1000. OPEN: verify labels exist for the unused
  "test group" (RSNA stage_2_test labels were never in the standard download);
  fallback = unused labeled portion of stage_2_train.
- **Abstract framing:** trust calibration — "how silently do these models fail
  under motion"; equipoise honest (neither model trained on corrupted data;
  question is which REPRESENTATION contains corruption better).

## FUTURE SPINOFF (JB, 2026-07-22): degraded-image-recognition AI
JB wants to train an AI to RECOGNIZE degraded images (trust-gating: "when to
trust AI output is a premier task"). The motion_arms output is ALREADY the
labeled training set (severity/direction/onset ground truth, both domains, free
from simulation). Sharp hypothesis: motion is easier to DETECT in the sinogram
(crisp broken-sinusoid discontinuity at onset angle) than in the image (diffuse
streaks) → sinogram-domain QA gate before reconstruction. Squarely in JB's
human-AI-collaboration lane ([[project-ai-radiologist-paradigm]]): AI's job
includes telling the human when NOT to trust the diagnosis.

## Why it matters
Grounds the whole projection-domain-AI direction quantitatively: if sinogram
learning is better-conditioned (white/Poisson noise → simpler matched kernels),
that is a principled, testable argument for working pre-reconstruction. Ties to
JB's broader framings: autocorrelation/Fourier as the closed-form alternative to
SGD for linear/shift-invariant structure (Wiener filter, Yule-Walker/Levinson-
Durbin; Fourier Neural Operators for the nonlinear extension); attention as
error-correcting vectors on a data manifold. Same red-vs-white-noise distinction
as the T5 LIGO whitening work.
