# Literature-to-code traceability and critical evidence audit

## 1. Why this document exists

The package combines algorithms that originated in different papers and
software libraries. A citation beside a branch does not validate the whole
branch. This document identifies exactly which implementation decision each
source supports, which claims it does not support, and which published evidence
argues against overinterpreting the current output.

Literature was rechecked on 30 July 2026. Primary articles and official project
documentation are preferred over secondary descriptions.

## 2. Evidence map

| Source | What it directly supports here | What it does **not** support here | Code location |
|---|---|---|---|
| Pan & Tompkins, 1985 | A historically established real-time QRS detection method. | That NeuroKit2's `pantompkins1985` is bit-identical to the original; that Pan-Tompkins is best on noisy peri-ictal ECG; exact R-wave fiducial accuracy for HRV. | `peaks.detect_r_peaks` through NeuroKit2. |
| Makowski et al., 2021 | NeuroKit2 as an open, reproducible neurophysiological processing toolbox with ECG routines. | Automatic validity on a new dataset or equivalence among its multiple detector methods. | Both cleaning/detection calls; package pin. |
| Kristof et al., 2024 | Comparative testing of 18 open QRS detectors; NeuroKit and UNSW performed best overall; ±75 ms same-QRS evaluation; quality-dependent degradation. | RR-interval accuracy, HRV suitability, seizure ECG, long peri-ictal EDFs, or proof that detector agreement equals truth. The authors explicitly state RR/HRV suitability was not evaluated. | Fixed 75 ms comparison and rationale for manual target-data validation. |
| Jeppesen et al., 2017 | Importance of refining QRS-region detections to stable R-wave locations for short-term HRV in epilepsy; high reported detector sensitivity/PPV in the paper's held-out patients. | A currently integrated open implementation, generalization to these EDF channels, or proof that either current NeuroKit2 path reproduces their fiducial refinement. | Motivates the validation/refinement gap; not implemented. |
| Jeppesen et al., 2019 | Phase-2 comparison of 26 HRV algorithms; combination of filtered modified CSI×slope and CSI×slope; 100-RR family; responder limitation. | High performance in all epilepsy patients, prospective frozen performance, or transfer to this patient without autonomic eligibility. | Choice of `J1` and `J2` continuous measurements. |
| Jeppesen et al., 2020 | Prospective validation of HRV seizure detection in the authors' device/program. | Direct validation of this Python feature implementation. | Context only. |
| Jeppesen et al., 2023/2024 | Patient-adaptive logistic-regression variants using the two continuous HRV parameters and seizure candidates; illustrates alternatives to a fixed threshold. | That this branch should output a probability now, or that candidate generation/model fitting can be copied without a patient-specific chronological study. | Not implemented; future fusion/model evidence. |
| Jeppesen et al., 2025 phase 3 | Prospective, blinded, multicentre real-time device study; 100-RR maximum-overlap calculations, seven-RR median filtering, elapsed-time least-squares absolute HR slope, products with CSI/modified CSI, personalized baseline, noise handling, refractory behavior. | General epilepsy-wide sensitivity, validity on artifact the noise detector would suppress, reproduction by offline EDF processing, or an exact unambiguous threshold implementation from the main-text wording alone. | `rr_hrv.py` continuous features; threshold intentionally absent. |

## 3. R-peak evidence: support and contradiction

### Evidence supporting the current exploratory pair

Pan-Tompkins is an established QRS detector, and the Jeppesen clinical system
used a Pan-Tompkins-related detection path. NeuroKit2 provides executable,
versionable implementations. Kristof et al. evaluated the NeuroKit detector
against manual annotations across six datasets and found NeuroKit and UNSW to
be the best overall among the 18 algorithms they tested. On high-quality
telehealth data, the two leading methods achieved F1 at or above 0.98; the paper
also reported fast execution for NeuroKit.

This supports using NeuroKit as a serious candidate and Pan-Tompkins as a
clinically relevant comparator. It does not support simply declaring either one
correct on the installed data.

### Evidence contradicting a blanket “NeuroKit is enough” claim

Kristof et al. found every detector degraded on low-quality unsupervised ECG;
even the leading methods were at or below 0.84 F1 in that low-quality subset.
They explicitly did not evaluate exact RR-interval accuracy or HRV suitability
and noted that additional R-wave localization can be necessary after QRS
detection.

Jeppesen et al. 2017 made the same distinction operationally: a QRS detector can
find the complex yet introduce Q/S-versus-R timing jitter into the tachogram.
Their epilepsy-specific work modified an existing detector to refine the R
fiducial and reported 99.979% sensitivity and 99.976% positive predictive value
on the last seven patients (467.6 hours). Those numbers are encouraging but
come from their device, split, labels, and implementation.

The current real-data finding—nearly equal candidate counts but poor 75 ms
agreement—is exactly why QRS count accuracy cannot be substituted for stable RR
timing.

### Better alternatives to audit

| Alternative | Published reason to test it | Barrier/limitation |
|---|---|---|
| NeuroKit (`nk`) as primary | Best overall with UNSW in the 2024 open benchmark; fast; Python-accessible. | Still degraded on low-quality ECG; current patient-specific manual score absent. |
| UNSW/Khamis detector | Co-best overall in Kristof; designed for telehealth ECG. | Benchmark implementation was MATLAB; not currently integrated; target-data validity still required. |
| Jeppesen 2017 R-fiducial refinement | Specifically designed to reduce tachogram jitter in epilepsy monitoring. | Reproducible source implementation has not been established in this package; device/domain differences. |
| gqrs | Earlier broad benchmark cited by Kristof found it strongest across 12 databases. | Performance can change sharply by telehealth domain; Kristof reported poor SAFER performance despite supervised-dataset strength. |
| Signal-quality-conditioned availability | Kristof directly shows quality-dependent accuracy and recommends quality handling. | A quality algorithm cannot identify the correct peak by itself and must also be validated. |

The next action is not to add all detectors. It is to annotate representative
patient ECG, benchmark a small evidence-ranked set under the same protocol, and
select only after target-domain results.

## 4. RR/HRV feature evidence

### Phase-2 selection

Jeppesen et al. 2019 evaluated 26 HRV-based algorithms in 100 recruited
patients; 43 patients contributed 126 seizures. The best combination used
`ModCSI100-filtered × Slope` together with `CSI100 × Slope`. Among the patients
classified as responders, sensitivity was 93.1%, with a reported false-alarm
rate of 1.0 per 24 hours and median latency of 30 seconds.

The critical counter-result is that only 53.5% of seizure patients were
responders. Marked ictal autonomic change, particularly a heart-rate change over
50 bpm, characterized responsiveness. Therefore this method is not evidence
for a universal ECG seizure signature.

### Phase-3 specification and performance

The 2025 phase-3 study was prospective, blinded, multicentre, and used a fixed
real-time system with a personalized threshold from the first 24 hours. Its
method description supports:

- maximum-overlap sliding windows of 100 RR intervals;
- CSI and modified CSI from Lorenz/Poincare axes;
- seven-interval median prefiltering for `ModCSI100-filtered`;
- absolute least-squares slope of prefiltered heart rate against time in the
  same 100-RR window;
- the two products `CSI100 × Slope` and `ModCSI100-filtered × Slope`;
- an alarm when either personalized parameter threshold is crossed;
- a minimum signal timeframe and a refractory period in the complete device;
- explicit device noise handling.

The trial selected eligible patients using a first-seizure ictal HR increase
greater than 50 bpm. Among 42 seizures in eligible patients, overall sensitivity
was 90.5%; focal seizures not evolving to bilateral tonic-clonic seizures had
82.6% sensitivity and all 19 bilateral tonic-clonic seizures were detected. The
mean false-alarm rate in eligible patients was 2.5 per 24 hours, the median was
1.1 per 24 hours, and median latency was 28 seconds. Only 47.2% of patients
showed the required autonomic change. In 19 ineligible patients, only 16.4% of
73 seizures were detected.

These ineligible-patient results directly contradict using phase-3 sensitivity
as if it applied to an unselected patient.

### Noise handling differs from this project

The phase-3 ASSURE application included a 2-second ECG RMS noise detector. Its
reported 20 mV threshold set both HRV parameters to zero during excessive
noise. The current architecture does **not** reproduce that behavior:

- it preserves the measurements;
- it carries detector disagreement as context;
- it has no calibrated RMS threshold;
- it does not convert artifact windows to zero, because zero would be confused
  with a physiological feature value in later fusion.

This is a deliberate architecture deviation. It is more suitable for the
planned context-aware fusion concept, but it means the implementation is not a
replica of the complete ASSURE alarm system. A future signal-quality branch can
include published RMS and other measurements, with an explicit availability
mask rather than overwriting measurements.

## 5. Threshold ambiguity and why no alarm is coded

Related Jeppesen reports describe an individualized threshold as 105% of the
highest seizure-excluded baseline value from the first 24 hours. The phase-3
main text also says a detection occurs when the personalized threshold of
either of the two parameters is exceeded. The implementation available to this
project does not establish with sufficient precision whether the two differently
scaled parameters each receive separate maxima/thresholds in every version of
the algorithm, how startup/invalid windows enter the maximum, and which exact
noise/refractory rules must be applied during calibration.

Because `J1` and `J2` have different units, inventing one shared scalar
threshold would be mathematically suspect. The package therefore exports both
continuous measurements and records `threshold_calibration_applied=false`.

This avoids claiming device reproduction. If thresholded detection is later
implemented, it must cite an exact algorithm specification or validated author
code and use a chronologically earlier, seizure-excluded baseline.

## 6. Statistical interpretation of branch outputs

The published seizure detector eventually makes a binary alarm, but the
intermediate quantities are statistical/engineered continuous values. The
current branch correctly exports measurements rather than a probability:

| Output | Nature | Not equivalent to |
|---|---|---|
| `RR` | Measured time difference between detected fiducials | True sinus NN interval unless peaks/rhythm are validated. |
| `SD1`, `SD2` | Sample dispersion of rotated successive-RR coordinates | Direct sympathetic/parasympathetic nerve measurement. |
| `CSI` | Dimensionless geometric ratio | Probability of seizure. |
| `ModCSI` | Scale-sensitive geometric index in ms | Calibrated risk. |
| `Slope` | Absolute linear HR trend in bpm/s | Direction of HR change. |
| `J1`, `J2` | Products used by the published algorithm | Independent evidence sources or final decisions. |
| Disagreement fields | Cross-detector context | Artifact truth or peak correctness. |

Binary labels become necessary at a later supervised evaluation stage because
the reference events are seizure/non-seizure intervals. That does not make the
branch itself a binary model.

## 7. Generalization limits

| Literature result | Limiting fact for this project |
|---|---|
| Strong QRS benchmark results | Benchmark ECGs were not long peri-ictal seizure recordings and exact HRV timing was not assessed. |
| Strong Jeppesen responder sensitivity | Only a subset with large autonomic ictal changes was eligible/responding. |
| Personalized threshold | Requires an adequate seizure-excluded baseline and an exact locked calibration protocol. |
| Multicentre phase-3 result | Validates the full device/app workflow, including noise handling and refractory behavior—not an extracted pair of Python formulas. |
| Many hours from one patient | Provides within-patient data volume but not many independent seizures or population generalizability. |
| Maximum-overlap 100-RR outputs | Creates highly correlated rows; it does not create a proportionate number of independent training examples. |

## 8. Literature-backed next experiments

1. **Manual peak benchmark in the target ECG.** Required because Kristof did not
   assess this domain and Jeppesen 2017 demonstrates the importance of stable R
   fiducials for HRV.
2. **Signal-quality branch before interpreting seizure peaks.** Supported by
   Kristof's quality-dependent degradation and the explicit phase-3 noise
   detector. It should output continuous measurements plus availability, not a
   silently destructive zeroing rule.
3. **Patient eligibility audit.** Measure whether seizures show the large,
   reproducible autonomic HR changes that defined Jeppesen eligibility. This is
   descriptive analysis on held-out events, not threshold tuning.
4. **Independent numeric reproduction.** Compare all continuous intermediates
   against fixed reference windows or author implementation before copying any
   threshold.
5. **Chronological episode-held-out evaluation.** Required because overlapping
   100-RR rows are not independent and because the intended model is
   patient-specific.

## 9. Primary references

1. Pan J, Tompkins WJ. *A Real-Time QRS Detection Algorithm*. IEEE Transactions
   on Biomedical Engineering. 1985. <https://doi.org/10.1109/TBME.1985.325532>
2. Makowski D et al. *NeuroKit2: A Python toolbox for neurophysiological signal
   processing*. Behavior Research Methods. 2021.
   <https://doi.org/10.3758/s13428-020-01516-y>
3. Jeppesen J et al. *Modified automatic R-peak detection algorithm for
   patients with epilepsy using a portable electrocardiogram recorder*. EMBC.
   2017. <https://doi.org/10.1109/EMBC.2017.8037753>
4. Jeppesen J et al. *Seizure detection based on heart rate variability using a
   wearable electrocardiography device*. Epilepsia. 2019.
   <https://doi.org/10.1111/epi.16343>
5. Jeppesen J et al. *Seizure detection using heart rate variability: A
   prospective validation study*. Epilepsia. 2020.
   <https://doi.org/10.1111/epi.16511>
6. Jeppesen J et al. *Personalized seizure detection using logistic regression
   machine learning based on wearable ECG-monitoring device*. Seizure. 2023.
   <https://doi.org/10.1016/j.seizure.2023.04.012>
7. Jeppesen J et al. *Detection of seizures with ictal tachycardia, using heart
   rate variability and patient adaptive logistic regression machine learning
   methods*. Epileptic Disorders. 2024.
   <https://doi.org/10.1002/epd2.20196>
8. Kristof F et al. *QRS detection in single-lead, telehealth
   electrocardiogram signals: Benchmarking open-source algorithms*. PLOS
   Digital Health. 2024. <https://doi.org/10.1371/journal.pdig.0000538>
9. Jeppesen J et al. *Seizure detection using wearable electrocardiogram
   connected to a smartphone: a phase 3 clinical validation study*.
   EBioMedicine. 2025. <https://doi.org/10.1016/j.ebiom.2025.105952>

## 10. Source-access notes

- Open phase-3 full text: <https://pmc.ncbi.nlm.nih.gov/articles/PMC12516532/>
- Open Kristof benchmark: <https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000538>
- Benchmark code: <https://github.com/floriankri/ecg_detector_assessment>
- NeuroKit ECG documentation:
  <https://neuropsychology.github.io/NeuroKit/functions/ecg.html>

