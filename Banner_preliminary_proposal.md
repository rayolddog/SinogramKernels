# Preliminary Proposal — Detecting Intracranial Hemorrhage in the CT Projection Domain

*University of Arizona–Banner Health **Big Idea Challenge: Joint Translational Research Accelerator** (pilot award up to $250,000; 2-year). Priority areas: **Artificial Intelligence in Health** and **Neurovascular Diseases and Treatments**.*

**Originating investigator:** John Bramble, MD — Banner Medical Group (Radiology).
**Sought:** a **University of Arizona faculty co-investigator to serve as academic lead** (Department of Medical Imaging, Medical Physics, or an imaging-AI group).

---

## Purpose of this preliminary proposal

This is a short pre-proposal to (1) establish the concept and its demonstrated feasibility, and (2) invite a University of Arizona faculty member to take the academic lead — as Principal Investigator — in developing the full proposal. The originating investigator is a practicing Banner radiologist who has produced the idea, a working pipeline, and preliminary results; what the project needs to reach full scale is an academic home — graduate students, high-performance compute, and CT-physics depth — which a UA lead supplies. The collaboration is scored at 20% of the evaluation and is central, not incidental.

**The originating investigator seeks to contribute, not to lead.** He is not positioned to manage a budget or supervise trainees, and does not seek the PI role or first authorship — those belong with the academic lead, who also gains the most from them. His aim is simply that a worthwhile idea, already shown to be feasible, gets developed by a group with the standing to carry it into practice, including the manufacturer engagement that individual investigators cannot achieve alone. The project is offered essentially ready to run.

## The idea

Deep networks can already reconstruct a CT image directly from projection (sinogram) data; such a network has learned, inside its encoder, to read diagnostic structure from projections. We propose to reuse that capability for a different endpoint — **detecting acute intracranial hemorrhage (ICH) directly from the projection domain, before an image is reconstructed.** The motivation is information-theoretic (by the data-processing inequality, reconstruction cannot add diagnostic information the raw projections do not already carry) and clinical (a hemorrhage flag raised before or during reconstruction is the earliest physically possible notification, in a setting where minutes change neurological outcome).

A concrete clinical thesis anchors the work: reconstruction-introduced artifacts — beam-hardening and dental streak — fall hardest on the **posterior fossa and brainstem**, exactly where small hemorrhages are most easily missed on CT. Because a projection-domain reader never sees a reconstruction artifact, the approach's value may be largest where CT is weakest — a falsifiable prediction of disproportionate sensitivity for **infratentorial** hemorrhage.

## Preliminary results — feasibility of the architecture is established

All results below are on synthetic (parallel-beam) sinograms forward-projected from the RSNA 2019 Brain CT Hemorrhage dataset, using a *twin* protocol: an identical network architecture and initialization trained on each domain, differing only by input. Work was performed on a single local GPU workstation.

- **Detection works, and matches the image domain.** A projection-domain network detects ICH at best validation **macro-AUC 0.946** across the six hemorrhage classes (SE-ResNeXt50, 200,000-slice cohort), statistically on par with the matched image-domain model. A data-size sweep shows the projection-domain model matching or marginally edging the image-domain model as data grows.
- **The network reads projection geometry, interpretably.** With the input's spectrum whitened, the learned first-layer kernels carry a domain signature — sinogram-native kernels are anisotropic and detector-axis-oriented; image-domain kernels are isotropic. This motivated a **sinogram-native architecture** with a small but statistically robust gain (paired *t* = 3.4 over six seeds).
- **Robustness, honestly characterized.** A controlled factorial (augmentation × whitening × domain × capacity) shows the projection domain's tolerance to lost/limited projection angles is a **real but conditional** advantage — present at moderate angle loss and high model capacity — after ruling out a training-augmentation confound. The team publishes its own negative controls.

**Bottom line:** the representation-and-architecture question — *can a network detect ICH from a projection-domain encoding, and how efficiently?* — is answered affirmatively. This is the feasibility foundation on which the full proposal builds.

## The plan, in brief (staged across three data regimes)

1. **Representation (largely complete)** — synthetic sinograms; the results above.
2. **Real projection geometry (public data)** — the Mayo Clinic / AAPM *LDCT-and-Projection-data* collection provides **real measured fan-beam helical projections** (DICOM-CT-PD) with paired reconstructions for ~99 head exams on Siemens and GE scanners. Deliverables: a fan-beam helical reconstruction pipeline feeding the detector, and a **forward-projection simulator validated against these measured projections.** No vendor agreement required.
3. **Clinical scale (Banner IRB)** — a retrospective thin-slice (≤0.5 mm) head-CT cohort enriched for ICH — the data regime essentially absent from public archives — processed through the validated simulator to a training-scale projection dataset; tests the infratentorial-sensitivity prediction.

A later translational phase pursues on-scanner raw data through a manufacturer research agreement — which an institutional award, IRB record, and validated pipeline make feasible.

## The team and the ask

- **University of Arizona — Principal Investigator / academic lead (sought).** Holds the PI role, the budget, trainee supervision, and lead authorship. Contributes CT-physics depth (fan-beam helical simulation, DICOM-CT-PD handling), graduate-student effort, and high-performance compute for full-cohort training, plus the academic standing to carry the work to publication and to a manufacturer research agreement.
- **Banner Health — originating / clinical co-investigator (Bramble).** The concept and feasibility data; radiology domain expertise and clinical grounding; and the Banner-system IRB pathway to the thin-slice ICH cohort and to Banner's CT fleet. Participates **remotely** as a contributor; a Banner–Arizona clinical co-investigator is welcome to strengthen on-site execution (the Challenge encourages Banner co-investigators).

**Next step:** identify the UA lead at the **virtual** July 17 open house and co-develop the full proposal by the August 10 deadline. Contact: John Bramble, MD — John.Bramble@bannerhealth.com.
