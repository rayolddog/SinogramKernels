# Detecting Intracranial Hemorrhage in the CT Projection Domain: A Sinogram-Native Deep-Learning Approach

*Submitted to the University of Arizona–Banner Health **Big Idea Challenge: Joint Translational Research Accelerator** (pilot award up to $250,000; 2-year).*

**Principal Investigator (academic lead) — sought:** a University of Arizona tenure / tenure-track faculty member (Medical Imaging / Medical Physics / imaging AI), to hold the PI role, budget, and lead authorship (§7).
**Originating / clinical co-investigator:** John Bramble, MD — Banner Medical Group (Radiology); John.Bramble@bannerhealth.com; contributes the concept and preliminary data, participating remotely.
**Priority areas addressed:** Artificial Intelligence in Health; Neurovascular Diseases and Treatments.
**Keywords:** intracranial hemorrhage; CT sinogram; fan-beam / helical geometry; DICOM-CT-PD raw projections; projection-domain deep learning; autocorrelation; Gabor wavelets; scanner-specific calibration.

---

## Project Summary

Deep neural networks can reconstruct a CT image directly from projection data (Li, Chen et al., *IEEE TMI* 2019); such a network has already learned, inside its encoder, to read diagnostic structure from projections. We propose to extend that capability from **reconstruction** to **diagnosis** — a network that identifies acute intracranial hemorrhage (ICH) from the projection domain, before an image is formed. The scientific motivation is precise: reconstruction is a lossy, regularized, vendor-specific inverse, and by the data-processing inequality it cannot add diagnostic information that the raw projections do not already carry. The mathematical structure the detector must exploit is second-order and oriented — the projection data encode the object's **autocorrelation** angularly (via the Fourier-slice and Wiener–Khinchin theorems), and are naturally analyzed with orientation- and scale-selective **Gabor** atoms, which are also what a convolutional network learns in its first layers. This proposal is deliberately honest about geometry and about what each dataset can prove: real CT is **fan-beam, acquired helically**, not the parallel-beam Radon idealization. The program is therefore staged across three data regimes: public RSNA reconstructions re-projected to *synthetic* sinograms (representation and architecture — **largely complete**, §5); the public Mayo Clinic / AAPM collection of **real measured fan-beam helical projections** (a validated real-geometry bridge, Aim 2); and a Banner IRB thin-slice head-CT cohort at **clinical scale** (Aim 3). The raw-data **fidelity and artifact advantages** are reserved for the on-scanner translational phase (Aim 4). Keeping these regimes explicitly separate is the central rigor of the program.

**Program fit.** The project addresses two of the Challenge's priority areas at once — *Artificial Intelligence in Health* and *Neurovascular Diseases and Treatments* — and is translational by construction: it pairs a Banner clinician-investigator (clinical data, IRB pathway, radiology domain expertise) with University of Arizona academic strength (medical-physics / imaging-AI depth, graduate students, high-performance compute), and is scoped to generate the preliminary data and validated methods that position the team for subsequent external (NIH R21/R01) and industry funding — the Challenge's stated purpose. The originating investigator (Bramble) brings the concept and feasibility data; a University of Arizona faculty co-investigator is sought to lead the academic execution (§7).

---

## 1. Historical Background

The mathematics predates the machine by half a century. **Johann Radon** (1917) proved that a function on the plane is determined by its integrals over all lines, and gave the inversion — the theoretical core of tomography. **Ronald Bracewell** (1956) independently used projection–reconstruction (“strip integration”) to map the radio sky, and articulated the **Fourier-slice relationship**. **Dennis Gabor** (1946) introduced the jointly space/frequency-localized “elementary signals” now called Gabor atoms in his *Theory of Communication*; Gabor also invented **holography** (Nobel Prize, 1971).

The clinical machine followed from **Allan Cormack** (1963–64), who worked out the radiological line-integral inversion, and **Godfrey Hounsfield** (1971–73), who built the first scanner at EMI; the two shared the 1979 Nobel Prize in Physiology or Medicine. The canonical mathematical treatment — parallel- and fan-beam filtered back-projection, algebraic reconstruction — is Kak & Slaney (1988). **Helical (spiral) CT** (Kalender et al., 1990) made the source trace a helix relative to the patient; exact inversion for the helical cone beam came only later (**Katsevich**, 2002). The autocorrelation–power-spectrum identity underlying our feature argument is the **Wiener–Khinchin** theorem (Wiener 1930; Khinchin 1934).

The deep-learning era reopened the inverse problem: **AUTOMAP** (Zhu et al., *Nature* 2018) learned image formation as a manifold transform; **Li, Chen et al.** (*IEEE TMI* 2019) learned the sinogram→image map directly on CT; **Xie, Shan & Wang** (DEAR, 2019) pursued few-view reconstruction. Each learns to make a *better image*. The present proposal reuses the same projection-domain modeling capability for a *different endpoint* — the diagnosis — and never forms the image.

---

## 2. Mathematical Framework

### 2.1 The real acquisition geometry: fan-beam and helical

A clinical scanner is **not** a parallel-beam Radon device. A point source illuminates a detector arc, so the measured rays are **divergent (fan-beam)**: a projection is g(β, γ), with β the source angle and γ the fan angle of a ray. Fan data relate to parallel data by a rebinning,

@@ θ = β + γ,     s = D sin γ,

with D the source-to-isocenter distance; fan-beam FBP additionally carries a distance weighting. Acquisition is **helical**: the table translates while the gantry rotates, so the source describes a helix relative to the patient,

@@ z(β) = (h / 2π) β     (pitch h per rotation).

Reconstructing an axial plane then requires longitudinal interpolation (single-slice helical), cone-beam methods (FDK), or exact helical inversion (Katsevich 2002), and in practice increasingly **model-based iterative** or **deep-learning** reconstruction (e.g. GE TrueFidelity/DLIR). The operational point: **raw data are divergent-beam, helically sampled, and reconstruction is a regularized, vendor-specific, non-injective-in-practice inverse.** That non-injectivity is exactly where kernel choice, apodization, and iterative priors inject the image-domain variability that projection-domain detection avoids.

### 2.2 The parallel-beam Radon transform (an idealization we use to reason)

The parallel-beam transform is the tractable model that exhibits projection structure. For attenuation field f,

@@ p_θ(s) = ∫ f(s cosθ − t sinθ,  s sinθ + t cosθ) dt.

A point mass at polar position (r₀, φ₀) maps to s = r₀ cos(θ − φ₀) — a sinusoid, hence *sino*gram. The **Fourier-slice theorem** connects it to the image spectrum F:

@@ (1-D FT of p_θ)(ω) = F(ω cosθ,  ω sinθ),

so projections tile the 2-D spectrum radially and f is recoverable with full angular coverage. Inversion is filtered back-projection,

@@ f(x,y) = ∫₀^π (p_θ ∗ H)(x cosθ + y sinθ) dθ,     H(ω) = |ω|,

whose ramp filter |ω| amplifies high-frequency noise — the ill-conditioning that forces the regularization choices of §2.1. **We treat this parallel-beam model as an illustrative, computable stand-in** for the projection structure; it is not the acquisition physics, and our synthetic data inherit its idealization (§2.5).

### 2.3 The sinogram encodes the image autocorrelation

The feature the detector must read is second-order. Define the image autocorrelation

@@ R_f(τ) = ∫ f(x) f(x + τ) dx,     (2-D FT of R_f) = |F|²     (Wiener–Khinchin).

The 1-D autocorrelation of one projection,

@@ r_θ(u) = ∫ p_θ(s) p_θ(s + u) ds,     (1-D FT of r_θ)(ω) = |F(ω cosθ, ω sinθ)|²,

is, by Fourier-slice applied to |·|², a **radial slice of the 2-D power spectrum**. Sweeping θ fills |F|², whose inverse transform is R_f. In words: **the projection autocorrelations, across angle, reconstruct the image autocorrelation** — each projection is an angular sample of the object’s correlation/texture structure. This is the exact sense in which the Radon/sinogram *demonstrates the expected shape of the autocorrelation*. A hemorrhage — a focal, hyperattenuating, texturally distinct region — perturbs R_f in an orientation-dependent way that is present, angularly encoded, in the sinogram.

### 2.4 Gabor wavelets: the natural feature basis

A Gabor atom is a Gaussian-windowed oriented sinusoid,

@@ ψ(x;  f₀, θ, σ, φ) = exp(−|x|² / 2σ²) · cos(2π f₀ (x · n_θ) + φ),

jointly localized in space and frequency (attaining the uncertainty bound; Gabor 1946) and selective in orientation θ and scale ∝ 1/f₀. Two facts make it the right basis here. First, an oriented image structure maps to a localized (θ, s) signature in the sinogram, so a Gabor bank over the sinogram reads projected orientation and scale directly — Gabor energy is a **localized power spectrum**, complementing the global autocorrelation of §2.3. Second, the first convolutional layers of trained CNNs are empirically Gabor-like; a CNN on the sinogram is, in effect, learning a **data-adaptive Gabor/autocorrelation front end**. The proposed detector therefore formalizes what its early layers would learn regardless — which we exploit both as initialization (fixed Gabor front end) and as an interpretability handle.

### 2.5 Information ordering — and an honest caveat

For a real scan the causal chain is

@@ Y (diagnosis)  →  f (attenuation field)  →  p_raw (fan-beam helical data)  →  x (reconstructed image),

with x computed from p_raw alone, so (Y, p_raw, x) form a Markov chain. The **data-processing inequality** gives

@@ I(Y; x) ≤ I(Y; p_raw).

Reconstruction cannot increase the diagnostic information the data carry; equality requires reconstruction to be a *sufficient statistic* for Y, which finite kernels, discretization, 12-bit quantization, apodization, and display windowing all break. **This is the rigorous form of the fidelity claim.**

But the public RSNA data provide x only. Our experimental sinograms are **synthesized by forward projection**, p_syn = R[x], so the chain is Y → x → p_syn and

@@ I(Y; p_syn) ≤ I(Y; x) ≤ I(Y; p_raw).

The synthetic sinogram is a re-encoding of the already-reconstructed image — it cannot exceed it — and it is parallel-beam, not fan-beam helical. **Consequently the RSNA phase can rigorously establish representation and architecture claims (can a network detect ICH from a projection-domain encoding? how efficiently?) and simulate geometry/dose robustness, but it cannot by itself demonstrate the raw-domain fidelity or artifact advantages.** Those require true p_raw and are the object of the translational phase. Keeping these two regimes separate is the central rigor of the program.

![**Figure 1.** The image↔projection loop and the motion argument. **(A)** An ideal image f (Shepp–Logan head phantom with a focal hyperdensity, red circle). **(B)** Its sinogram R[f]; the lesion is a clean, connected sinusoid (red track). **(C)** A sudden mid-scan patient motion, modeled as a step offset of the sinogram past the jump angle (yellow line) — the lesion trace offsets abruptly but each segment remains a coherent sinusoid. **(D)** Inverse Radon of the motion-corrupted sinogram, shown in a wide, low-centered lung window: the step discontinuity back-projects into sharp radiating streaks that bury the lesion. The abnormality's signature is better preserved in projection space (B, C) than in the reconstructed image (D), which motivates detection before reconstruction. All panels are real Radon/inverse-Radon transforms; the lesion track is the argmax ridge of the lesion-only sinogram under the same step offset.](schematic_sinogram_motion.png){0.75}

---

## 3. Hypotheses and Goals (falsifiable)

- **H1 — Representation.** A CNN reading synthetic sinograms detects ICH at AUC within a pre-specified margin δ of a matched image-domain model on identical splits. *Falsified* if projection-domain AUC is materially and significantly lower (paired DeLong test). *Preliminary data (§5) support H1: the projection- and image-domain twins are at parity (best validation macro-AUC 0.946, projection domain).*
- **H2 — Bound.** On synthetic data, projection-domain AUC ≤ image-domain AUC, per the Bayes bound of §2.5; the measured shortfall quantifies representation efficiency. *Falsified / flagged* if projection-domain significantly **exceeds** image-domain — which would indicate label leakage or a reprojection artifact to be investigated, not a physical gain. *Preliminary data (§5) are consistent with the bound: the domains sit at parity, the projection-domain model edging the image-domain model only by margins within seed noise (≤0.02 AUC), with no material or significant exceedance to date.*
- **H3 — Raw-domain advantage (translational).** With true fan-beam raw sinograms from a fixed scanner, projection-domain detection avoids reconstruction artifacts, retains higher bit depth and detector sampling, and is at least as accurate as, and more stable within-scanner than, image-domain detection. *Testable only with vendor raw data* (§6).

**Aim 1 — Representation, on synthetic sinograms (months 1–6; largely complete, §5).** Build the paired image/synthetic-sinogram dataset at full scale and train matched projection- and image-domain classifiers under identical protocols to test H1/H2; ablate a fixed-Gabor versus learned front end and an explicit autocorrelation feature. *Feasibility already demonstrated (§5): projection-domain detection at macro-AUC 0.946, at parity with the image domain.*

**Aim 2 — Real projection geometry, on public raw data (months 4–14).** Move from synthetic parallel-beam sinograms to **real fan-beam helical projection data** using the public Mayo Clinic / AAPM *LDCT-and-Projection-data* collection (TCIA), which provides measured **DICOM-CT-PD** projections and paired reconstructions for 99 non-contrast head, 100 chest, and 100 abdomen exams on Siemens and GE scanners. Two deliverables: **(i)** a **fan-beam helical acquisition-and-reconstruction pipeline** — helical→fan rebinning and a differentiable reconstruction layer feeding the detector head; and **(ii)** **validation of a forward-projection simulator against ground truth** — forward-project the collection's reconstructed images through the documented scanner geometry and compare to the *measured* projections (sim-to-real fidelity metrics). This is the bridge the synthetic phase cannot cross alone, and the natural home for the UA medical-physics contribution.

**Aim 3 — Scale, on a Banner clinical cohort (months 8–20).** Under a **Banner IRB** protocol, assemble a retrospective **thin-slice (≤0.5 mm) head-CT cohort enriched for ICH** — the data regime essentially absent from public archives — and apply the Aim-2-validated simulator to generate a training-scale projection-domain dataset with expert clinical labels. Retest matched projection- vs image-domain detection at scale; probe fan-beam / reduced-dose / sparse-view robustness (extending the **conditional** angular-robustness result of §5); and test the infratentorial-sensitivity prediction of §4.

**Aim 4 — Translation to on-scanner raw data (months 14–24).** Specify and initiate the raw-data acquisition/retention protocol needed to test H3 on a production Banner scanner (§6), approached from the strength of Aims 1–3.

---

## 4. Significance — the four advantages (properties of RAW data)

Each advantage is a property of true raw projections (the motivation); the RSNA phase can only partially probe them by simulation.

1. **Reconstruction artifacts avoided.** Beam-hardening, streak, cone-beam and iterative-reconstruction textures are introduced *by* reconstruction; a sinogram reader never sees them. *(Raw-data claim; approachable by simulation.)*
2. **Higher intrinsic fidelity.** Raw projections carry greater bit depth and finer detector sampling than the windowed, resampled image retains — formalized by the data-processing inequality of §2.5. *(Raw-data claim; not demonstrable on synthetic sinograms.)*
3. **Freedom from cross-algorithm inaccuracy.** An image-domain model inherits kernel/slice/vendor dependence — a leading cause of generalization failure across reconstructions. A projection-domain detector is instead **calibrated to one specific scanner’s raw output**; the reconstruction-algorithm variable is removed, not averaged over. In practice these kernels are set to individual radiologist *preference* at protocol setup — radiologists in one group routinely disagree over reconstruction algorithms, or simply reject studies as "too noisy" — and a coordinated health-system effort to standardize (as in Banner's own multi-vendor dose-reduction and protocol-standardization initiative) is limited by the incompatibility of kernels across manufacturers. A projection-domain detector sits upstream of that subjective, site-specific choice — removing the **kernel/preference axis** of variation (not the scanner-geometry axis) — which makes it directly relevant to a large standardization-seeking, multi-vendor system like Banner. *(Testable by simulated-geometry robustness now — preliminary limited-angle experiments (§5) show the projection domain's robustness advantage is real but **conditional** on model capacity and degradation severity, not an unconditional intrinsic property; confirmable on raw data.)*
4. **Speed of notification.** Detection precedes or runs concurrently with reconstruction, enabling the earliest physically possible flag for a finding in which minutes change neurological outcome. *(Latency measurable now; end-to-end only on-scanner.)*
5. **Two semi-independent readers.** Because the projection- and image-domain models read *different representations* of the same acquisition, they can be deployed together: agreement raises confidence, disagreement flags a case for human review, and a single adversarial perturbation is far harder to craft so that it survives into both a raw sinogram and its reconstruction. *(On synthetic data the two are not truly independent — the sinogram is projected from the image — so this benefit cashes out fully only on real raw data.)*

**The concrete patient-benefit case (neurovascular priority area).** Reconstruction-introduced artifacts are not uniform across the brain: beam-hardening streaks (trans-petrous / interpetrous) and dental artifact fall hardest on the **posterior fossa and brainstem** — precisely where small hemorrhages are most easily missed on CT, and where MRI is well established as superior. Because a projection-domain reader never sees a reconstruction-introduced artifact, the approach's clinical value may be *largest exactly where CT is weakest.* This yields a **falsifiable clinical prediction** the Banner cohort (Aim 3) is positioned to test: a projection-domain detector should show a disproportionate sensitivity advantage for **infratentorial** (brainstem / posterior-fossa) hemorrhage relative to supratentorial, versus an image-domain detector. A measurable sensitivity gain for under-diagnosed brainstem bleeds is a direct patient-care benefit and the clinical rationale that anchors the whole program.

---

## 5. Preliminary Data

A working pipeline exists and has produced analyzable cohorts **and a first set of quantitative results.** **Source:** RSNA 2019 Intracranial Hemorrhage Detection (>870,000 expertly labeled head-CT slices; six labels per slice — epidural, intraparenchymal, intraventricular, subarachnoid, subdural, *any*), the PI a registered participant. **Encoding:** each slice on a calibrated **linear-Hounsfield** scale (value = HU + 1024, preserving the physical attenuation relationship rather than a display window), forked into a reconstructed branch (256×256, 12-bit) and a **synthetic parallel-beam** 256-angle sinogram (256×256, float32) — both from one shared source, the controlled comparison H1/H2 require. **Cohorts:** stratified **50,000- and 200,000-slice** subsets, built with **zero processing failures**, preserving the full-dataset hemorrhage prevalence exactly (14.5% positive) and all 32 observed label-combinations proportionally; QC pairs confirm anatomically correct images and physically correct sinograms.

**Preliminary results** (synthetic parallel-beam sinograms; a *twin* protocol trains an identical architecture and initialization on each domain, differing only by input):

1. **Detection — H1 supported.** A projection-domain network detects ICH at best validation **macro-AUC 0.946** across the six classes (SE-ResNeXt50, 200k), statistically on par with the matched image-domain twin (held-out per-domain macro-AUC ≈ 0.91 for both). A whitened data-size sweep (50k→~180k) shows the projection-domain model matching or marginally edging the image-domain model (0.851 vs 0.848 at ~180k) — consistent with H1's near-parity and with the H2 bound (no material exceedance).
2. **Learned representation — a domain signature.** With the input's average power spectrum whitened (removing the low-order statistics a network fits first), the first-layer kernels differ by domain: sinogram-native kernels are **anisotropic, detector-axis-oriented, and lower-frequency**; image-domain kernels are isotropic and higher-frequency. The literal filtered-back-projection ramp does **not** emerge — the projection-domain network learns Radon-geometry-aligned structure directly, an interpretability handle for Aim 2. Architecture experiments over sinogram-native (anisotropic-stem) designs then identified a configuration with a small but **statistically robust** gain (+0.005 macro-AUC; paired *t* = 3.4 over six seeds, Cohen's *d* = 1.4).
3. **Angular robustness — a controlled result (informs §4, Aim 3).** Under simulated angular under-sampling (a removed wedge of projection angles, applied natively in each domain), a factorial over augmentation × spectral-whitening × domain × model-capacity establishes that the projection domain's graceful degradation is **not an unconditional intrinsic property.** It is in large part a *trained* property — reproducible in the image domain once the image branch is shown the matched physical corruption — and the residual projection-domain advantage is **conditional:** clearly present at moderate angle loss and high model capacity (retention 0.81 vs 0.57 for the image-domain model at a 30% wedge), but absent at severe angle loss, where both domains collapse without augmentation. This controlled result (a self-imposed negative control, then refinement) is the honest basis for the reframed robustness language in §4 and Aim 3.

**Compute:** local RTX 4090 / NVMe; full-cohort scaling validated. Per §2.5 these sinograms are synthetic and cannot show the raw-domain advantages of §4 — they establish representation and architecture feasibility (H1/H2) and simulate geometry robustness only.

---

## 6. Translation Path, Vendor Access, and the Path to External Funding

Raw projection data are proprietary and are not exposed at the DICOM/PACS layer, so testing H3 and any scanner-integrated deployment ultimately requires manufacturer cooperation — a defining feature of the problem, not an incidental obstacle. The program is deliberately staged so that **no vendor agreement is needed to produce its pilot deliverables:** Aim 1 runs on public RSNA data, Aim 2 on the public Mayo/AAPM *measured*-projection collection, and Aim 3 on a Banner IRB cohort processed through the Aim-2-validated simulator. A positive result across these aims is precisely the evidence that justifies — and de-risks — a manufacturer raw-data research agreement.

Crucially, a CT manufacturer signs a raw-data research agreement with an **institution, not an individual.** This is where the Banner–University of Arizona partnership is the enabling instrument: the award supplies the institutional IRB of record, the data-governance framework, and the institutional signature a vendor requires, and Banner's own multi-vendor CT fleet is the natural site for on-scanner raw-data retention. The team will approach the vendor(s) dominant in Banner's fleet from a position of strength — *"we already run on your raw-data format (DICOM-CT-PD), our forward-projection simulator is validated against measured data from your scanner class, and we hold an IRB-approved ICH cohort; what we require is raw-data retention on a production scanner and, eventually, on-scanner integration."*

**Positioning for external funding.** The pilot is scoped to yield exactly what a subsequent NIH R21/R01 or industry application needs: demonstrated projection-domain detection (Aim 1, in hand), a validated real-geometry pipeline (Aim 2), a scaled clinical cohort with an IRB record (Aim 3), and a concrete vendor-engagement plan (Aim 4) — the "position teams for future external funding" outcome the Challenge is designed to seed.

---

## 7. The University of Arizona–Banner Collaboration

The Challenge funds *joint* UA–Banner teams and weights the strength of that collaboration at 20% of the evaluation. This project is joint by construction rather than convenience — neither half can execute it alone.

- **University of Arizona — Principal Investigator / academic lead (sought).** Target: tenured / tenure-track faculty in the Department of Medical Imaging, Medical Physics, or an imaging-AI group. Holds the PI role, budget authority, trainee supervision, and lead authorship. Contributes the CT-physics depth for the fan-beam helical simulator and DICOM-CT-PD handling (Aim 2), high-performance compute and graduate-student effort for full-cohort training (the preliminary work to date has run on a single workstation), and the academic-research infrastructure and standing that position the work for external funding and manufacturer engagement. Eligibility as project lead is satisfied by the UA faculty appointment. To be finalized at or after the (virtual) July 17 open house.
- **Banner Health — John Bramble, MD (originating / clinical co-investigator).** Radiology domain expertise and the clinical framing; the Banner-**system** IRB pathway to the thin-slice head-CT data and expert ICH labels of Aim 3; and the institutional relationship to Banner's production CT fleet that Aim 4 requires. The concept and the preliminary feasibility data (§5) originate here. The originating investigator participates **remotely** and seeks a contributing role — not the PI position, budget authority, or first authorship. A Banner–Arizona clinical co-investigator is welcome to support on-site data work (the Challenge encourages Banner co-investigators).
- **Shared execution.** Aims 1–2 (representation, real-geometry pipeline, simulator validation) are jointly executed and are where the UA medical-physics contribution is most load-bearing; Aim 3 (IRB cohort) is Banner-led with UA analytical support; Aim 4 (vendor path) is jointly institutional. This division — clinical / IRB / scanner access from Banner, physics / compute / academic infrastructure from UA — is the substance of the collaboration, not a formality.

---

## 8. Budget and Resources (up to $250,000; 2 years)

Consistent with the Challenge's pilot intent, funds are requested to generate preliminary data and validate methods. Indicative categories, to be finalized with the UA co-investigator and the respective institutional research offices:

- **Personnel:** a graduate research assistant / postdoctoral contribution on the UA side (simulator development, real-geometry pipeline, training at scale); percent-effort for the PI/co-investigators; data-management and IRB-coordination effort on the Banner side.
- **Computing:** UA high-performance / GPU computing for full-cohort training beyond the originating investigator's existing single-GPU workstation, plus storage for the large paired projection/image caches (full RSNA cache and the Aim-3 cohort).
- **Data and regulatory:** Banner IRB submission and data-extraction support; TCIA data-use registration (no cost); de-identification tooling.
- **Dissemination:** publication and conference costs; preliminary-data packaging for the follow-on external application.

The originating investigator's stated binding constraint is **access to data and compute, not funding**; the budget should be sized by the UA academic lead to the space, equipment, and student effort an academic execution actually requires.

---

## 9. Timeline and Deliverables

- **Aim 1 (months 1–6):** full-scale paired dataset; matched twin classifiers; H1/H2 tested — results largely in hand (§5).
- **Aim 2 (months 4–14):** fan-beam helical pipeline on TCIA DICOM-CT-PD; forward-projection simulator validated against measured projections.
- **Aim 3 (months 8–20):** Banner IRB approval; thin-slice ICH cohort; scaled projection-domain detection; infratentorial-sensitivity test.
- **Aim 4 (months 14–24):** raw-data retention protocol specified; vendor engagement initiated; external-funding (R21/R01) application assembled.

---

## References

*Classical citations below are standard references given from the canonical literature; page-level details should be verified against the originals. Deep-learning citations were confirmed against publisher/PubMed records.*

1. Radon J. "Über die Bestimmung von Funktionen durch ihre Integralwerte längs gewisser Mannigfaltigkeiten." *Ber. Sächs. Akad. Wiss. Leipzig, Math.-Phys. Kl.* 1917;69:262–277.
2. Bracewell RN. "Strip integration in radio astronomy." *Aust. J. Phys.* 1956;9:198–217.
3. Gabor D. "Theory of communication." *J. Inst. Electr. Eng.* 1946;93(26):429–457.
4. Cormack AM. "Representation of a function by its line integrals, with some radiological applications." *J. Appl. Phys.* 1963;34:2722–2727.
5. Hounsfield GN. "Computerized transverse axial scanning (tomography): Part 1. Description of system." *Br. J. Radiol.* 1973;46:1016–1022.
6. Kak AC, Slaney M. *Principles of Computerized Tomographic Imaging.* IEEE Press, 1988.
7. Kalender WA, Seissler W, Klotz E, Vock P. "Spiral volumetric CT with single-breath-hold technique, continuous transport, and continuous scanner rotation." *Radiology* 1990;176(1):181–183.
8. Katsevich A. "Theoretically exact filtered backprojection-type inversion algorithm for spiral CT." *SIAM J. Appl. Math.* 2002;62(6):2012–2026.
9. Zhu B, Liu JZ, Cauley SF, Rosen BR, Rosen MS. "Image reconstruction by domain-transform manifold learning (AUTOMAP)." *Nature* 2018;555:487–492. doi:10.1038/nature25988
10. Li Y, Li K, Zhang C, Montoya J, Chen G-H. "Learning to Reconstruct CT Images Directly from Sinogram Data under a Variety of Data Acquisition Conditions." *IEEE Trans. Med. Imaging* 2019;38(10):2469–2481. doi:10.1109/TMI.2019.2910760
11. Zhang C, Li Y, Chen G-H. "Accurate and robust sparse-view angle CT image reconstruction using deep learning and prior image constrained compressed sensing (DL-PICCS)." *Med. Phys.* 2021;48(10):5765–5781. doi:10.1002/mp.15183
12. Xie H, Shan H, Wang G. "Deep Encoder-Decoder Adversarial Reconstruction (DEAR) Network for 3D CT from Few-View Data." *Bioengineering* 2019;6(4):111. doi:10.3390/bioengineering6040111
13. Flanders AE, Prevedello LM, Shih G, et al. "Construction of a Machine Learning Dataset through Collaboration: The RSNA 2019 Brain CT Hemorrhage Challenge." *Radiol. Artif. Intell.* 2020;2(3):e190211. doi:10.1148/ryai.2020190211
14. Moen TR, Chen B, Holmes DR III, Duan X, Yu Z, Yu L, Leng S, Fletcher JG, McCollough CH. "Low-dose CT image and projection dataset." *Med. Phys.* 2021;48(2):902–911. doi:10.1002/mp.14594 *(Mayo Clinic / AAPM LDCT-and-Projection-data; real DICOM-CT-PD helical projection data + paired reconstructions, The Cancer Imaging Archive.)*
