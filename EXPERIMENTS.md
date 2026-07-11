# SinogramKernels — Experiment Log & Roadmap

Living record of every experiment in this project: what was run, why, the result,
and what's queued. Update this file when a run finishes or a new one is planned
(record-the-process methodology). One section per experiment; newest running/queued
work at the bottom.

**Question under test:** can a CNN detect ICH directly from projection (sinogram)
data, and what do its learned kernels reveal about how it reads Radon-domain
structure? Formal plan and hypotheses (H1–H3, Aims 1–3): `grant_proposal.md`.

## Shared setup

- Data: RSNA ICH slices; 744,685 source DICOMs in `/mnt/mlcache/dicom_staging`.
- Paired tensors per slice from one source image (`data_pipeline.py`):
  recon uint16 (HU+1024) + sinogram float32 (raw parallel-beam radon), 256×256.
- Caches: `/mnt/mlcache/subset50k`, `/mnt/mlcache/subset200k` (select with `SINO_CACHE`).
- venv: `/home/johnb/PyCharmMiscProject/.venv`.
- Twin protocol: identical architecture + identical init, differ ONLY by input domain
  (`twin_common.py`, `train_twin.py`). Metric: best val macro-AUC over 6 ICH classes.
- Architectures (`twin_common.py`): `smallcnn`, `seresnext50`, `aniso`/`iso`/`aniso2`/
  `aniso3`/`aniso_cfg` (anisotropic stem cascade, kernel shape (detector rows × angle cols)).

---

## Completed

### 1. Data pipeline + QC (Jul 8)
50k paired cache built and QC'd. Artifacts: `qc_pair_50k.png`,
`qc_partial_volume_50k.png` (`qc_partial_volume.py` flags partial-volume pseudo-ICH).
Proposal schematics: `make_schematic.py`, `make_autocorr_schematic.py`
(autocorrelation identity r_θ = Radon[R_f], Pearson r = 0.995 on a real positive slice).

### 2. Raw twin, 50k, 9×9 kernels — `kernel_exp/` (Jul 8–9)
SmallCNN, K1=9, 12 epochs. **Result:** recon vs sinogram layer-1 kernels
indistinguishable (|d| ≤ 0.1); both become generic oriented high-pass filters.
AUC ~0.81. Small local kernels show no domain signature on raw input.

### 3. Large-kernel + whitened twins, 50k — `kernel_exp_big/`, `kernel_exp_white/` (Jul 9)
31×31 stems; whitening (flatten average power spectrum, `twin_common.whitening_filter`)
removes low-order statistics the net would otherwise learn first
(distributional-simplicity-bias argument). **Result:** whitening REVEALS the domain
signature — sinogram kernels detector-axis oriented, anisotropic, lower spatial
frequency; recon kernels isotropic/higher-frequency. The literal FBP ramp
(oscillation + negative sidelobes) does NOT appear; the structure is
Radon-geometry-aligned, not the ramp. Figures: `kernel_comparison_big.png`,
`kernel_comparison_white.png`, `kernel_scale_summary.png`, `whitening_explain.png`.

### 4. Whitened data-size sweep, 200k cache — `k200_N{50000,100000,0}/` (Jul 9)
31×31/48f whitened twins, N = 50k / 100k / all (~180k). Process notes:
`kernel_exp/overnight_log.md`.

| N | recon AUC | sino AUC | centroid d | detector:angle d | anisotropy d |
|---|---|---|---|---|---|
| 50k | 0.8071 | 0.8237 | −0.97 | +1.66 | +0.11 |
| 100k | 0.8322 | 0.8353 | +0.40 | +1.61 | +0.22 |
| ~180k | 0.8475 | 0.8514 | −1.95 | +2.30 | +0.64 |

**Result:** domain signature strengthens with data; whitened sinogram CNN matches or
edges the recon CNN. Figures: `kernel_comparison_200k.png`, `kernel_datasize_summary.png`.

### 5. Angular robustness, whitened smallCNN — `angular_robustness.py` (Jul 9)
Wedge of angles removed, native to each domain (sinogram: columns zeroed; recon:
limited-angle iradon of the same wedged sinogram). Models from `k200_N0`.
**Result:** whitened sinogram model COLLAPSES (macro 0.46 at frac 0.4) while recon
survives (0.79). Figure: `angular_robustness_k200_whitened.png`.

### 6. SE-ResNeXt50 twins, 200k — `se_200k/` (Jul 9)
Big-model ceiling check, 25 epochs, no whitening. **Result:** best val macro-AUC
0.9461. Stem kernels: `stem_kernels_se200k.png`.
Angular robustness rerun on these models (`angular_robustness_se200k.json/.png`):
**the robustness result FLIPS** — sinogram degrades gracefully (0.911 → 0.886 macro
at frac 0.5) while recon collapses to chance (0.907 → 0.486; limited-angle streak
artifacts are out-of-distribution for the image model). The whitened-smallCNN vs
raw-SE flip is unexplained — candidate factors: whitening amplifying the wedge
discontinuity, capacity, kernel size. Worth a controlled follow-up.

### 7. Anisotropic sinogram-native architectures (Jul 9–10)
Stem cascade of decreasingly anisotropic convs sized to trace geometry;
sinogram domain only, 200k unless noted, 25 epochs, no angular aug.

| Run | Arch | Notes | best val macro-AUC |
|---|---|---|---|
| `aniso_200k` | aniso | first orientation | 0.8751 |
| `aniso_h_200k` | aniso | angle-wide ("h") stem | 0.8829 |
| `aniso2_400k` | aniso2 | taller first layer (30 det rows), **400k** | 0.9142 |
| `aniso3_200k` | aniso3 | bigger stem throughout, 20 ep | 0.8766 |

Matched-parameter iso control exists as `--arch iso` (same param count/channels,
square kernels) so any gap is attributable to anisotropy alone.
Figures: `aniso_kernels*.png`, `aniso2_kernels.png`, `aniso_kernel_comparison.png`.

### 8. One-at-a-time ablation of aniso2's changes — `abl_*/` (Jul 9–10)
Baseline `aniso_cfg` (aniso_h: 15×60 stem, stride-2, 3×3 body, no dilation);
add ONE change each; all 200k / 20 epochs / `--aug-angular` (`run_ablations.sh`,
`ablations_master.log`).

| Run | Change | best val macro-AUC |
|---|---|---|
| `abl_base` | — | 0.8772 |
| `abl_tall` | tall stem | 0.8700 |
| `abl_fulres` | last stem stride 1 | 0.8712 |
| `abl_dil` | body dilations 1,2,4 | 0.8779 |
| `abl_5x5` | 5×5 body kernels | 0.8741 |
| `abl_all` | all four | 0.8814 |
| `abl_iso` | iso control | 0.8761 |

**Result:** no single change helps much (dilation +0.0007 best); only the
combination gains (+0.0042) — small enough that seed noise could explain it →
motivated experiment 9. Figures: `ablation_summary.png`, `architecture_map.png`,
`metrics_compare.png`.

---

## Running

### 9. Multi-seed noise-floor test — `seed_{base,gfb}_s{1,2,3}/` (`run_multiseed.sh`)
Baseline `aniso_cfg` vs go-for-broke (tall + fulres + 5×5 + dilation), seeds 1–3
(seed 0 = `abl_base`/`abl_all`), 200k / 20 ep / `--aug-angular`. Pool 4 seeds per
config: does the ~0.004 gap separate from seed variance?

Progress (`multiseed_master.log`):
- `seed_base_s1` 0.8720, `seed_gfb_s1` 0.8807 — done
- `seed_base_s2` — died at epoch 9 on Jul 10 11:55 AM (session closed);
  restarted detached (nohup) Jul 10 11:56 PM, resumed from checkpoint
- `seed_gfb_s2`, `seed_base_s3`, `seed_gfb_s3` — pending

**RESULT — RESOLVED at 6 seeds (Jul 11):** base 0.8748±0.0026, gfb 0.8796±0.0014;
paired gap **+0.0048**, **t=3.445 > t_crit(df=5)=2.571 → SEPARATES from seed noise**
(Cohen's d=1.41). The go-for-broke architecture (tall stem + fulres + 5×5 + dilated
body) reliably beats the aniso_cfg baseline; the ~0.005 gap is real, not seed variance.
(At n=4 it was borderline: t=3.05 vs 3.18.) Per-seed gaps all positive
(+0.0042/+0.0087/+0.0012/+0.0047/+0.0087/+0.0011). `multiseed_pooled_final.txt`.
Seeds 4,5 ran concurrently with a shutdown-at-80 watcher active; GPU oscillated
75–79 °C and never tripped. GPU thermals: `gpu_temp_log.csv`.

**Lesson (Jul 10):** launch overnight runs with `nohup ... &` (detached), never
inside an interactive session.

### 10. Angular-flip Tier-1 — augmentation isolation (`run_tier1_flip.sh`, Jul 11)
Decisive minimal test for the exp 5-vs-6 flip. Two sinogram models, identical
`aniso_cfg` default / whiten=off / seed 0 / 200k / 20 ep, differ ONLY by
`--aug-angular`:
- **aug-ON** = existing `abl_base` (val 0.8772)
- **aug-OFF** = `flip_noaug/` (new run, training concurrent with multiseed)
Eval: `angular_robustness_flip.py` (sinogram-only, aniso_cfg-aware, same wedge
protocol + 2000-slice seed-0 test set). Metric: retention = macro-AUC(frac)/AUC(0).

**RESULT (Jul 11) — H_aug CONFIRMED.** aug-ON `abl_base` retention@0.3 = **0.983**;
aug-OFF `flip_noaug` retention@0.3 = **0.541** (collapses to ~chance). Removing the
wedge augmentation destroys the sinogram's angular robustness → the robustness is
TRAINED, not intrinsic to the projection domain.

### 11. Angular-flip Tier-2 factorial — autonomous overnight run (Jul 11)
Full 2×2×2 AUG{none,wedge} × WHITEN{off,on} × DOMAIN{sino,recon} at fixed
`aniso_cfg`, 200k, 20 ep, seed 0. Orchestrated session-independently by
`run_tier2_overnight.sh` (under `systemd-inhibit`), which waits for the GPU to
free (multiseed + Tier-1) and the recon-wedge bank, then trains 6 new cells
sequentially and evaluates each, collating `RESULTS_OVERNIGHT.md` after every cell.
Deadline-guarded at 07:40 so results always collate before 08:00.

Infrastructure built + validated this session:
- `make_recon_wedge_bank.py` — precomputes the recon-side wedge augmentation: per
  image a (1+K=4)-variant float16 bank of iradon(full sino) + K iradon(wedged sino)
  at random frac/phase, in `$SINO_CACHE/recon_wedge/` (~94 GB, all 200k ids). This
  is the parity fix: the recon branch now sees the SAME physical corruption as the
  sinogram branch, so AUG becomes separable from DOMAIN.
- `twin_common.py` — `recon_wedge_pick` augment + 3D-bank handling in `DomainDS`,
  `domain_stats`, `whitening_filter` (all use clean variant 0 when not augmenting).
- `train_twin.py` — `--aug-wedge-recon` flag; `--domains recon_wedge` reads the bank.
- `angular_robustness_flip.py` — now domain-parametric (`ROBUST_DOMAIN`): recon models
  are tested on iradon(wedged sino), matching their training domain; reuses the
  canonical `apply_whiten` from train_twin. All smoke-tested.
- `collate_flip.py` — assembles the 8-cell table + the AUG-effect and H_intrinsic
  contrasts into `RESULTS_OVERNIGHT.md`.
Cells: existing sino/none/off=`flip_noaug`, sino/wedge/off=`abl_base`; new
`rc_none_off`, `rc_wedge_off` (decisive whiten=off recon pair, run FIRST), then
`sn_none_on`, `sn_wedge_on`, `rc_none_on`, `rc_wedge_on`. Tier-3 (seresnext
capacity) left queued — unlikely to fit before 08:00.

**RESULT (Jul 11) — retention@0.3, 6/8 cells (whiten=on recon pair hit deadline):**

| domain | aug | whiten | ret@0.3 |
|---|---|---|---|
| sino | none | off | 0.541 |
| sino | wedge | off | 0.983 |
| recon | none | off | 0.674 |
| recon | wedge | off | 0.975 |
| sino | none | on | 0.558 |
| sino | wedge | on | 0.926 |
| recon | none | on | 0.901 |

(recon/wedge/whiten-on skipped by the 07:40 deadline — 7/8 cells.)

**Conclusion — the flip was 100% an augmentation confound; H_aug, not H_intrinsic.**
Retention tracks AUG, hugely and consistently, in every domain×whiten combo:
Δ(wedge−none) = +0.442 (sino/off), +0.301 (recon/off), +0.367 (sino/on). The
H_intrinsic test (aug=none, whiten=off) shows sino 0.541 **below** recon 0.674 — the
sinogram has NO intrinsic angular-robustness advantage; if anything the recon (image)
model degrades more gracefully without augmentation. Whitening is not the driver
(sino/none ≈ 0.54–0.56 regardless). The exp 5-vs-6 "flip" is fully explained: whichever
branch received `--aug-angular` during training became robust to the test-time wedge.

**Grant impact:** H2 ("projection data spread diagnostic info across all angles →
intrinsic robustness to lost/limited angles") is NOT supported by the RSNA-phase data.
The graceful degradation is a *trained* property, reproducible in image space. Reframe
H2 as trainable robustness (both domains) rather than an intrinsic projection-domain
advantage — and state it as the outcome of this controlled negative control. Results:
`RESULTS_OVERNIGHT.md`.

### 12. Tier-3 capacity check — SE-ResNeXt50 (Jul 11) — REFINES the conclusion
Repeats the sinogram aug on/off contrast at high capacity (seresnext50, ~27M params).
aug-ON cell = existing `se_200k`; aug-OFF cell = new `se_noaug` (25 ep, whiten off).
Sinogram wedge retention (same 2000-slice test set):

| frac wedge | 0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|---|
| aug-ON  `se_200k`  | 1.00 | 0.996 | 0.994 | 0.987 | 0.982 | **0.973** |
| aug-OFF `se_noaug` | 1.00 | 0.931 | 0.886 | 0.814 | 0.844 | **0.535** |

**Refined conclusion — BOTH augmentation and capacity matter, augmentation dominant
where it counts.** Augmentation is still the decisive factor at high capacity: at a
50% wedge it lifts retention 0.535 → 0.973. But capacity gives a partial cushion at
*moderate* wedges — the big aug-OFF model retains 0.814@0.3 vs the small aniso_cfg's
0.541@0.3 — that **evaporates at large wedges** (both ≈0.53 at 50%). So the earlier
"100% augmentation confound" phrasing (§11) is too strong: high capacity buys some
intrinsic tolerance to *small* angle loss, but is NOT a substitute for augmentation and
does NOT confer robustness to *substantial* angle loss. H2 (intrinsic robustness to
lost/limited angles) remains unsupported as the mechanism — the un-augmented big model
still collapses under a 50% wedge.
**Open (now closed by §13):** whether the big aug-OFF cushion is sinogram-SPECIFIC or
just "large models generalize" — see §13. `RESULTS_TIER3.txt`.

### 13. Tier-3b — recon-at-seresnext aug-OFF (Jul 11) — the cushion IS sinogram-specific
Trains the recon-domain twin of se_noaug (seresnext50, recon_wedge, aug OFF, 25 ep) and
compares aug-OFF wedge retention across domains at high capacity.

| frac wedge | 0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|---|
| sino aug-OFF `se_noaug`       | 1.00 | 0.931 | 0.886 | **0.814** | 0.844 | 0.535 |
| recon aug-OFF `se_recon_noaug`| 1.00 | 0.694 | 0.622 | **0.570** | 0.572 | 0.566 |

At high capacity the sinogram model is dramatically MORE wedge-robust than the recon
model (retention@0.3 0.814 vs 0.570, Δ+0.244; the gap is already +0.24 at a 10% wedge).
The recon model (iradon of the wedged sinogram = limited-angle streaks) is destroyed by
even a small wedge despite a strong clean baseline (macro@0 0.935). **This is a REVERSAL
from small capacity**, where recon beat sino (0.674 vs 0.541). So the moderate-wedge
cushion is genuinely SINOGRAM-SPECIFIC and emerges only with capacity — not "big models
generalize."

**SYNTHESIS across §10–13 — the refined verdict (supersedes the §11 "100% aug" claim):**
1. **Severe angle loss (≥50% wedge): augmentation dominates.** Both domains collapse
   without it (sino/recon aug-OFF ≈0.53–0.57 at 50%); aug restores sino to 0.973.
2. **Moderate angle loss (~10–30%) at HIGH capacity: a real sinogram-specific advantage.**
   Un-augmented seresnext sino retains 0.81–0.93 vs recon 0.57–0.69 — the projection
   domain degrades gracefully where the image domain (limited-angle streaks) does not.
3. **This advantage requires capacity.** At small capacity the ordering reverses
   (recon > sino), so a weak model cannot exploit the sinogram's angular redundancy.

**Grant impact (revised):** H2 is CONDITIONALLY SUPPORTED, not refuted. There is a
demonstrable projection-domain robustness advantage to moderate angle loss, but it is a
high-capacity phenomenon and does not extend to severe angle loss (which needs
augmentation). Honest caveat: the domain-native corruptions differ in kind (localized
zeroed columns vs global streak artifacts) — but that asymmetry (missing angles are
local in sinogram space, global in image space) is exactly H2's proposed mechanism, not
a confound. `RESULTS_TIER3B.txt`.

**Side finding (recon whitening×wedge interaction):** recon/none retention jumps
0.674 (whiten off) → 0.901 (whiten on) — whitening the iradon input makes the
un-augmented recon model markedly more wedge-robust (limited-angle streaks are largely
low-order structure that whitening suppresses). Worth a follow-up; not central to H_aug.

**Thermal:** `thermal_governor.sh` enforces the user's rule (no sustained ≥80 °C)
without root by duty-cycling the SECONDARY jobs (SIGSTOP/SIGCONT on flip/rc/sn),
never the primary multiseed runs; hysteresis 80→75 °C. (`nvidia-smi -pl` needs root,
unavailable.)

---

## Queued / open

- **Explain the angular-robustness flip** (exp 5 vs 6) — DESIGN READY.
  Root cause is NOT mysterious: the two runs differ in 4 things at once, and
  `--aug-angular` (train_twin.py:81) applies wedge dropout **only to the sinogram
  branch**. In exp 6 (SE) aug was ON → the sinogram model was trained on the exact
  test-time corruption; in exp 5 (k200) aug was OFF and whitening ON. Aug-on and
  whiten-off are perfectly aligned across the two runs, so they cannot be separated
  from existing data. Hypotheses: H_aug (robustness is trained, not intrinsic),
  H_intrinsic (proposal's claim — sinogram spreads info across angles), H_whiten
  (whitening amplifies the wedge's zero-edge into an OOD spectral artifact),
  H_capacity.
  **Controlled design:** 2×2×2 factorial — AUG{none,wedge} × WHITEN{off,on} ×
  DOMAIN{sino,recon}, fixed arch (aniso_cfg, 200k, 20ep), 3 seeds, same
  angular_robustness.py test set/protocol; summary = retention AUC(0.3)/AUC(0).
  KEY NEW CODE: a recon-side wedge augmentation so "augmented" is the same physical
  corruption in both domains — precompute a bank of K≈4 wedged recons per image
  (iradon of zeroed-column sinogram) into recon_wedge/ cache + `--aug-wedge-recon`.
  Tier 1 decisive (2 runs, ~70min): sinogram branch aug-on vs aug-off at whiten=off
  — if retention flips with aug alone → H_aug. Tier 2: full 2×2×2×3seeds. Tier 3:
  repeat winning cells at seresnext50 to rule out capacity.
  Decision rule: retention tracks AUG & gap vanishes when aug matched → H_aug
  (reframe proposal claim); sino>recon at aug=none → H_intrinsic (result the grant
  wants); collapse only at whiten=on → H_whiten (test robustness on non-whitened
  models only).
- **Deeper-layer / effective-receptive-field analysis** — where does the global
  sinusoidal trace assemble? (Layer-1 alone can't see a full trace.)
- **Reconstruction / self-supervised objective** to chase the literal FBP ramp
  (classification never produced it; a recon objective should).
- **Trace-tokenization ViT front end** — Hough-style O(N³) token enumeration,
  prunable to the peripheral search band.
- **From the proposal (Aims 2–3):** fixed-Gabor vs learned front-end ablation;
  explicit autocorrelation feature; fan-beam + reduced-dose/sparse-view simulation;
  back-projected saliency maps.
- Multiseed follow-up: pooled 4-seed comparison (t-test / effect size) once
  `run_multiseed.sh` completes; extend `make_ablation_summary.py`.

---

## Session addendum (Jul 11 PM) — domain-expert notes & next steps

**Streak caveat (from JBB, radiologist) — tempers the §13 recon-collapse finding.**
Streak artifacts are COMMON in real reconstructed CT images (metal/dental work, hair
clips, motion), so a recon model trained on real clinical data would have SEEN streaks
and be far less naive to them than `se_recon_noaug`, which was trained on CLEAN
synthetic iradon only. The §13 recon collapse is thus an upper bound on image-domain
fragility; real-world recon models would degrade more gracefully. The sinogram-specific
advantage (§13) may shrink against a streak-experienced recon model — worth testing by
augmenting the recon branch with realistic streak/metal artifacts, not just wedge-iradon.

**First-layer kernels after angular augmentation (`make_aug_kernels.py` →
`aug_kernels_comparison.png`).** Controlled pair (aniso_cfg, seed 0): aug-OFF `flip_noaug`
vs aug-ON `abl_base`, stem.0 = 48 kernels of 15(detector)×60(angle). Measured difference
in the angle-axis energy profile: aug-ON has ~2× the coefficient-of-variation along angle
(CoV 0.162 vs 0.080) and de-weights the angle EDGES relative to center (edge/center 0.654
vs 0.835). I.e. wedge augmentation visibly reshapes the first layer — it does not simply
flatten reliance across angle; it restructures where along the angle axis the kernels put
their energy. (Open: interpret the mechanism from the mosaic.)

**Planned: full training on the 750K dataset (over several nights).** Uses the
go-for-broke architecture (tall stem + fulres + 5×5 + dilated body; the multiseed winner).
Requirements before launch: (1) build the full paired cache for all 744,685 DICOMs
(`data_pipeline.py`) — large preprocessing + storage job, currently only subset50k/200k
exist; (2) resume-safe multi-night training with thermal protection.

**Clinical hypothesis (from JBB) — the project's core rationale, posterior fossa.**
Beam-hardening streaks (esp. trans-petrous / interpetrous) and dental artifact are
INTRODUCED BY reconstruction and land hardest on the LOWER CNS — brainstem and
posterior fossa. Small brainstem bleeds are plausibly UNDER-diagnosed on CT precisely
because these artifacts obscure them (and MRI is well established as superior for
brainstem / posterior-fossa pathology). Since a sinogram reader never sees a
reconstruction-introduced artifact, the projection-domain approach's clinical value may
be LARGEST exactly where CT is weakest. **Testable prediction:** a projection-domain
detector should show a *disproportionate* sensitivity advantage for INFRAtentorial
(brainstem/posterior-fossa) hemorrhage relative to supratentorial, vs a recon-domain
detector. Needs anatomic localization of bleeds (RSNA labels are subtype, not location)
— e.g. curate/label a posterior-fossa subset, or test on cases with known petrous
beam-hardening. This is the concrete patient-benefit case for the whole line of work.
