# Complete RR-HRV architecture: implementation and audit manual

## 1. Purpose and boundary

This document describes the code implemented in package version 0.2.0. The
branch converts one ECG channel into:

1. detector event timestamps;
2. refined primary R-fiducial timestamps;
3. consecutive RR intervals and instantaneous heart rate;
4. a per-QRS detector-support measurement;
5. a per-RR reliability measurement;
6. the continuous 100-RR Jeppesen HRV measurements;
7. polarity, detector-delay, tolerance, and Pan-Tompkins ablation context.

The branch does not output seizure probability, artifact probability, or a
clinical diagnosis. A zero reliability flag means the specified detectors did
not satisfy the specified agreement rule. It does not identify why they
disagreed.

## 2. Evidence hierarchy

| Component | Implemented source | Evidence status |
|---|---|---|
| UNSW QRS detection | `neurokit2` method `khamis2016` | Khamis et al. 2016; author Python translation reported by Ho et al. 2025 |
| NeuroKit secondary | Reproduction of NeuroKit2 0.2.13 gradient algorithm | NeuroKit software and Makowski et al. 2021 |
| UNSW + NeuroKit pairing | Primary UNSW, secondary NeuroKit | Recommended high-coverage pair in Ho et al. 2025 preprint |
| Close-detection exclusion | Keep one event within 150 ms | Ho analysis code and current method description |
| Same-QRS support | Exactly one secondary event within +/-50 ms | Current Ho method; 50 ms selected through tolerance analysis |
| RR support | Both boundaries supported and exactly two secondary events in expanded RR | Current Ho method |
| R-fiducial refinement | Highest oriented amplitude within +/-50 ms | Current Ho method |
| 100-RR features | SD1, SD2, CSI, filtered ModCSI, slope, products | Jeppesen phase-3 method |
| Seven-RR filter | Median of current and previous six RRs | Jeppesen phase-3 method |
| NeuroKit 250 ms | Inclusive minimum separation | Project experiment, not validated in the cited detector studies |
| NeuroKit 300 ms | Strict minimum separation | Installed software baseline |
| +/-150-ms support | Parallel reliability result | Historical report assumption retained only as an ablation |
| Dual polarity | Run original and inverted signals | Diagnostic extension motivated by record 207 and polarity-CNN precedent |
| Automatic polarity switching | Not implemented | Insufficient evidence for within-record routing |
| Pan-Tompkins | 300/250-ms context outputs | Diagnostic ablation; never a hard validator |

The Ho 2025 evidence remains a preprint and was developed on 30-second TELE and
SAFER telehealth ECGs, mainly in an atrial-fibrillation screening context. It
does not establish performance during seizures, on long peri-ictal recordings,
or on the eventual patient's morphology.

## 3. End-to-end data flow

```text
raw ECG and sampling rate
  |
  +-- original orientation -------------------------------------+
  |    |                                                        |
  |    +-- UNSW primary QRS events                              |
  |    +-- NeuroKit secondary: 250-ms inclusive experiment     |
  |    +-- NeuroKit secondary: 300-ms strict baseline          |
  |    +-- Pan context: 250 and 300 ms                          |
  |                                                             |
  +-- inverted orientation -------------------------------------+
       same detector set; context only unless explicitly selected

selected configured orientation
  |
  +-- collapse <=150-ms detections separately in each detector
  +-- compare primary with NeuroKit 250 at +/-50 ms
  +-- compare primary with NeuroKit 300 at +/-50 ms
  +-- compare primary with NeuroKit 250 at +/-150 ms (ablation)
  +-- refine every primary event within +/-50 ms
  +-- construct consecutive RR intervals
  +-- preserve every RR and attach all reliability columns
  +-- causal seven-RR median filter
  +-- trailing 100-RR HRV window advanced by one RR
  +-- return structured measurements and provenance
```

No path from a comparator detector deletes a primary timestamp. The primary
event list remains inspectable even when every RR reliability value is zero.

## 4. Configuration contract

The immutable `RRHRVConfig` dataclass is in `config.py`.

### Detector roles

- `primary_detector="unsw"`: fixed because the current architecture assigns
  timestamp ownership to the recommended primary.
- `secondary_detector="neurokit"`: fixed because the current literature-backed
  pair is UNSW plus NeuroKit.
- `processing_orientation="original"`: the orientation whose UNSW events own
  the output. It can be set to `"inverted"` only as an explicit experiment.
- `compute_inverted_context=True`: runs the other orientation and reports its
  measured coverage, but never changes the selected orientation.

### Timing parameters

- `neurokit_experimental_min_delay_ms=250`
- `neurokit_baseline_min_delay_ms=300`
- `min_delay_inclusive=True`
- `close_detection_exclusion_ms=150`
- `support_tolerance_ms=50`
- `legacy_support_tolerance_ms=150`
- `r_fiducial_refinement_ms=50`
- `audit_match_tolerance_ms=75`

At 360 Hz, 250 ms is 90 samples. With the default inclusive comparison, a
candidate exactly 90 samples after the last accepted event is eligible. The
unmodified NeuroKit implementation uses a strict comparison; therefore the
300-ms baseline is always run with `>`.

These rules change candidate eligibility, not ECG sampling. Every sample is
still used in the gradient and moving-window calculations.

### HRV parameters

- `hrv_window_rr_intervals=100`
- `causal_rr_median_width=7`
- `reliable_hrv_coverage=1.0`

The last value does not delete feature values. It controls only the separate
`feature_reliable` boolean. A threshold of one is conservative because no
paper-backed lower acceptable coverage was found for applying the Jeppesen
features to automatically detected RR intervals.

## 5. Detector implementation

### 5.1 UNSW primary

`detect_unsw()` calls NeuroKit2 0.2.13 `ecg_findpeaks` with method
`khamis2016`. The code is the Python translation written by authors of the Ho
study from the CC0 UNSW MATLAB implementation.

At a high level the detector:

1. detrends the ECG;
2. estimates and removes baseline variation;
3. applies high- and low-pass filtering;
4. differentiates the cleaned ECG;
5. combines slope and a local amplitude envelope;
6. estimates dominant heart-rate frequency;
7. smooths the QRS-energy feature;
8. applies adaptive turning-point thresholds;
9. performs lower-threshold search-back for possible misses;
10. re-evaluates implausibly short intervals at a higher threshold.

The detector therefore returns QRS event locations, not guaranteed raw-signal
R maxima. The separate refinement stage is necessary if sample-level R
fiducials are required.

### 5.2 NeuroKit secondary

`detect_neurokit_gradient()` first calls `ecg_clean(method="neurokit")`, then
reproduces the NeuroKit gradient detector so the minimum-delay comparison is
explicit.

The detector:

1. calculates the change between neighboring cleaned ECG samples;
2. takes the absolute gradient;
3. smooths it over approximately 100 ms;
4. estimates a slower approximately 750-ms local gradient baseline;
5. marks regions whose smoothed gradient exceeds 1.5 times that baseline;
6. ignores regions shorter than 0.4 times the mean candidate-region length;
7. finds the most prominent positive local maximum in each retained region;
8. accepts it only if the configured delay comparison is satisfied.

The 250-ms experiment can admit candidates which the 300-ms baseline rejects.
This can recover rapid beats, but it can also admit more T waves and secondary
maxima. Both outputs are therefore saved.

### 5.3 Pan-Tompkins context

`detect_pan_tompkins_context()` follows the tested NeuroKit port:

1. Pan-Tompkins-specific cleaning;
2. first difference;
3. squaring;
4. 120-ms moving-window integration, updated at each sample;
5. adaptive signal/noise energy thresholds;
6. configurable 300- or 250-ms minimum accepted-event delay;
7. search-back using the previous eight intervals.

The returned marker describes the energy-envelope event and can be displaced
from the raw R maximum. The code compares it with UNSW at the audit's 75-ms
tolerance only to produce context statistics. It cannot set `rr_supported`.

## 6. Polarity context

The same complete detector set is run on `x` and `-x`. Each orientation has
its own positive-amplitude definition. This matters because both the NeuroKit
positive-maximum localization and the Ho highest-amplitude refinement can fail
when the QRS becomes predominantly negative.

`polarity_context.csv` reports:

- primary and secondary counts;
- QRS support fraction at 50 ms;
- RR coverage for NeuroKit 250 and 300;
- RR coverage under the 150-ms support ablation;
- which orientation has higher measured coverage.

`higher_coverage_orientation` is descriptive, not a label. Two algorithms can
agree on the same wrong deflection. The code deliberately has no
`auto_orientation` mode.

Record 207 demonstrates both sides of this decision. Inversion greatly
increases UNSW-NeuroKit agreement and improves reference results in the
negative-QRS portion, but the recording contains changing morphologies. A
whole-record decision is not evidence that instantaneous routing is safe.

## 7. Per-QRS and per-RR reliability

### 7.1 Close detections

For each detector independently, detections separated by at most 150 ms are
collapsed. The candidate with the higher amplitude in the selected oriented
signal is retained.

This rule is not a universal physiological refractory period. It is a
method-specific duplicate-removal step from the current Ho analysis.

### 7.2 Per-QRS support

For primary event `p_i`, define the set of secondary events:

```text
M_i = {s_j : |s_j - p_i| <= tolerance}
```

The current support rule is:

```text
q_QRS(i) = 1 if and only if |M_i| = 1
```

The implementation records the count, nearest secondary sample, and signed
secondary-minus-primary timing offset. Zero matches and multiple matches are
different failure states even though both produce `q_QRS=0`.

### 7.3 Per-RR support

For the interval bounded by consecutive primary events `p_i` and `p_(i+1)`,
the current rule requires:

1. `q_QRS(i)=1`;
2. `q_QRS(i+1)=1`;
3. exactly two secondary events from `p_i-tolerance` through
   `p_(i+1)+tolerance`.

The third condition rejects a secondary detector's extra event inside the RR
interval. The output records all failed conditions in `reliability_reason`.

### 7.4 R-fiducial refinement

Every collapsed primary event is moved to the highest oriented ECG sample in
the author code's half-open `[-50,+50)`-ms NumPy search slice. Edge events are
left unchanged if the complete slice is unavailable. Consecutive refined
samples define RR:

```text
RR_i_ms = 1000 * (R_(i+1) - R_i) / sampling_rate
HR_i_bpm = 60000 / RR_i_ms
```

The initial UNSW event and refined R sample are both retained. This distinction
proved essential on MIT-BIH record 207: positive-maximum refinement on the
original orientation substantially worsened timing, while the inverted
orientation largely corrected it.

## 8. The 100-RR HRV measurements

### 8.1 Window

The first feature row is produced only after 100 consecutive primary RR
intervals. The window then advances by one RR interval.

The elapsed duration is measured, not assumed:

```text
window_elapsed_s = sum(RR_i_ms for the 100 intervals) / 1000
```

At 50 bpm the window covers about 120 s; at 120 bpm it covers about 50 s; at
240 bpm it covers about 25 s.

### 8.2 Poincare axes

For successive RRs `I_k` and `I_(k+1)`:

```text
SD1 = std((I_(k+1) - I_k) / sqrt(2))
SD2 = std((I_(k+1) + I_k) / sqrt(2))
```

Sample standard deviation (`ddof=1`) is used.

### 8.3 CSI and modified CSI

With `T=4*SD1` and `L=4*SD2`:

```text
CSI = L/T = SD2/SD1
ModCSI = L^2/T = (4*SD2)^2/(4*SD1)
```

`CSI100` uses raw RR intervals. `ModCSI100_filtered` uses the causal
seven-RR median-filtered sequence.

### 8.4 Causal median filter

For each new RR, the filter returns the median of that RR and up to the six
preceding intervals. It never uses a future interval. The first six rows use
the available shorter history.

### 8.5 Tachogram slope

Filtered heart rate is:

```text
HR_filtered = 60000 / RR_filtered_ms
```

The absolute ordinary-least-squares slope of filtered heart rate against real
elapsed time is calculated in bpm/s. Beat index is not used as the time axis.

### 8.6 Final continuous measurements

```text
J1 = CSI100 * slope100
J2 = ModCSI100_filtered * slope100
```

No threshold is applied. Jeppesen's individualized alarm threshold was trained
from an interictal baseline and belongs to a later patient-specific seizure
model, not to feature extraction.

### 8.7 Reliability accompanying the features

The feature values are calculated from the 100 consecutive primary RRs even
when one is unsupported. This prevents silent deletion and preserves the exact
physiological time sequence. The following context accompanies every row:

- `window_supported_rr_count`;
- `rr_reliability_coverage`;
- `window_contains_unsupported_rr`;
- `feature_defined`;
- `feature_reliable`.

`feature_defined` means the mathematics returned finite J1 and J2 values.
`feature_reliable` additionally requires the configured reliability coverage.
Neither means seizure.

## 9. Module map

| Module | Responsibility |
|---|---|
| `config.py` | Immutable parameters and evidence-status metadata. |
| `peaks.py` | UNSW, configurable NeuroKit, configurable Pan context, polarity, and general sequence comparison. |
| `reliability.py` | 150-ms collapse, +/-50-ms refinement, per-QRS support, and per-RR reliability. |
| `rr_hrv.py` | Causal median filter, Poincare axes, slope, CSI, ModCSI, J1, J2, and 100-RR coverage. |
| `edf.py` | EDF header inspection, explicit channel choice, and segment loading. |
| `wfdb_io.py` | WFDB signal and expert beat-annotation loading. |
| `pipeline.py` | Both orientations, 250/300 and 50/150 ablations, selected output, and provenance. |
| `outputs.py` | Stable CSV and JSON files. |
| `plotting.py` | Four-panel detector/reliability/HRV diagnostic. |
| `validation.py` | Expert-label detector and RR error metrics. |
| `cli.py` | Windows commands for EDF, WFDB, and MIT-BIH benchmarking. |

## 10. Output schemas

### `primary_qrs_events.csv`

Important fields:

- `primary_candidate_sample`: initial collapsed UNSW event;
- `r_fiducial_sample`: refined primary R sample;
- `secondary_match_count`: number of current secondary candidates in +/-50 ms;
- `nearest_secondary_sample`;
- `secondary_offset_ms`;
- `qrs_supported`;
- `qrs_supported_nk300_baseline`;
- `qrs_supported_support150_ablation`.

### `rr_hrv_features.csv`

Important groups:

1. timing: start/end candidate and refined samples, start/end seconds;
2. physiology: `rr_ms`, `heart_rate_bpm`;
3. current reliability: bounding support, secondary count, reason, and
   `rr_supported`;
4. ablations: `rr_supported_nk300_baseline` and
   `rr_supported_support150_ablation`;
5. 100-RR measurements and elapsed duration;
6. coverage, definition, and reliability flags.

### `detector_events.csv`

This long-format table is the most direct way to overlay all detector outputs.
Every row includes orientation, detector role, method, minimum delay, sample,
and time.

### `metadata.json`

The metadata contains all configuration values, package versions, source path,
selected orientation, detector counts, compact agreement summaries, and
explicit false statements for seizure classification, artifact classification,
threshold calibration, and automatic polarity routing.

## 11. Real-data findings before patient testing

The following first audit used the first MIT-BIH channel, all expert beat
symbols, and +/-75-ms one-to-one evaluation. These results validate software
execution and expose failure modes; they do not validate seizure performance.

| Record | Detector | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|---:|
| 108 | UNSW refined | 1757 | 7 | 6 | 0.9963 |
| 113 | UNSW refined | 1794 | 1 | 1 | 0.9994 |
| 207 | UNSW refined, original | 1850 | 145 | 10 | 0.9598 |
| 222 | UNSW refined | 2475 | 3 | 8 | 0.9978 |
| 231 | UNSW refined | 1571 | 11 | 0 | 0.9965 |

The requested NeuroKit 250-ms experiment did not dominate 300 ms:

- record 113: 250 ms produced 1,162 false positives versus 1,039 at 300 ms;
- record 231: 250 ms produced 415 false positives versus 410 at 300 ms;
- records 207 and 222 were unchanged;
- record 108 gained a few true detections but retained 181 false positives.

Current 50-ms RR coverage and the retained 150-ms ablation on the original
orientation were:

| Record | 250/50 coverage | 250/50 supported MAE (ms) | 300/50 coverage | 300/50 supported MAE (ms) | 250/150 coverage | 250/150 supported MAE (ms) |
|---|---:|---:|---:|---:|---:|---:|
| 108 | 0.7941 | 12.33 | 0.7935 | 12.29 | 0.8457 | 14.28 |
| 113 | 0.3534 | 1.36 | 0.4214 | 1.38 | 0.3534 | 1.36 |
| 207 | 0.0928 | 13.76 | 0.0928 | 13.76 | 0.5496 | 13.84 |
| 222 | 0.6936 | 1.87 | 0.6936 | 1.87 | 0.9956 | 1.70 |
| 231 | 0.7369 | 0.93 | 0.7381 | 0.93 | 0.7369 | 0.93 |

These five records do not establish that 50 or 150 ms is universally better.
The wider tolerance increases coverage dramatically on records 207 and 222,
but increases supported-RR MAE on record 108. The tolerance must therefore
remain an explicit validation parameter, not an invisible constant.

This is a counterexample to treating detector agreement as universal signal
quality. UNSW itself was nearly perfect on record 113, but NeuroKit's T-wave
double detections caused most RRs to fail the two-detector rule.

For record 207, inverted-orientation context changed 50-ms coverage from
0.0928 to 0.8572. When inversion was explicitly selected, the refined UNSW F1
was 0.9598 and supported RR MAE was about 4.76 ms. This supports polarity as an
important context variable but does not prove a safe switching algorithm.

### 11.1 Completed 48-record MIT-BIH benchmark

The same +/-75-ms one-to-one audit was subsequently completed on all 48
MIT-BIH Arrhythmia Database records (109,494 expert beats). No record failed
to execute. Pooled counts must be interpreted together with record-level and
morphology-stratified results; they are not seizure-validation results.

| Detector output | TP | FP | FN | Sensitivity | PPV | Pooled F1 | Median record F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| UNSW initial event | 109,244 | 350 | 250 | 0.99772 | 0.99681 | 0.99726 | 0.99967 |
| UNSW refined R fiducial | 109,207 | 387 | 287 | 0.99738 | 0.99647 | 0.99692 | 0.99936 |
| NeuroKit, 250-ms inclusive | 107,713 | 2,780 | 1,781 | 0.98373 | 0.97484 | 0.97927 | 0.99817 |
| NeuroKit, 300-ms baseline | 107,677 | 2,503 | 1,817 | 0.98341 | 0.97728 | 0.98033 | 0.99867 |
| Pan context, 250 ms | 108,587 | 1,422 | 907 | 0.99172 | 0.98707 | 0.98939 | 0.99836 |
| Pan context, 300 ms | 107,853 | 1,984 | 1,641 | 0.98501 | 0.98194 | 0.98347 | 0.99833 |

Three architecture decisions follow from this audit.

1. The R-fiducial refinement cannot be assumed to improve the primary
   detector. It slightly reduced pooled F1 from 0.99726 to 0.99692. Both the
   initial event and refined fiducial therefore remain in the outputs for
   patient-specific ablation.
2. The requested 250-ms NeuroKit variant increased sensitivity slightly but
   also increased false positives, giving lower pooled F1 than the unmodified
   300-ms baseline. Across records, its 50-ms RR coverage was higher in 10,
   equal in 31, and lower in 7. It is an experiment, not the new default truth.
3. Expanding support from 50 to 150 ms raised mean RR coverage from 0.9295 to
   0.9587 and median coverage from 0.9952 to 0.9985, but mean supported-RR MAE
   worsened from 3.48 to 3.70 ms. The wider window can label more intervals as
   supported while masking timing disagreement.

The complete machine-readable results are in
`output/rr_hrv_architecture/mitdb_all48_v1` at the repository root.

## 12. Tests

The automated suite covers:

- monotone peak matching and undefined empty-sequence metrics;
- exact 250-ms inclusive versus strict behavior;
- higher-amplitude selection during 150-ms collapse;
- the exactly-two-secondary-events RR rule;
- causal median filtering;
- Poincare axes and elapsed-time slope;
- the first valid 100-RR feature index;
- separation of finite feature values from reliability failure;
- complete UNSW/NeuroKit/ablation pipeline execution;
- existing EDF channel and viewer behavior.

Run:

```powershell
.venv\Scripts\python -m pytest -q
```

Passing tests do not establish beat accuracy. Use `benchmark-mitdb` for expert
annotations and create independent manual beat annotations for representative
patient peri-ictal ECG.

## 13. Remaining gates before training a seizure model

The all-record MIT-BIH benchmark, the 250-inclusive versus 300-strict
comparison, and the 50- versus 150-ms coverage/error comparison are complete.
The following work is still required:

1. Report performance by morphology and heart-rate strata, not pooled F1 only.
2. Add the 250-strict NeuroKit ablation; the completed comparison used the
   requested 250-inclusive experiment and the unmodified 300-strict baseline.
3. Manually annotate target-patient segments containing clean baseline,
   seizures, tachycardia, bradycardia, morphology transitions, and artifact.
4. Determine whether positive-maximum refinement should be orientation-specific
   or replaced by a validated polarity-robust fiducial method.
5. Test polarity on chronological blocks; do not tune and evaluate a routing
   rule on the same record sections.
6. Confirm that J1/J2 distributions are stable under small annotation jitter
   and individual missed/extra beats.
7. Only then decide how the final model consumes unreliable feature rows.

## 14. References

- Khamis H, et al. QRS detection algorithm for telehealth ECG recordings.
  IEEE Transactions on Biomedical Engineering. 2016.
  DOI: `10.1109/TBME.2016.2549060`.
- Kristof F, et al. QRS detection in single-lead telehealth ECG signals:
  benchmarking open-source algorithms. PLOS Digital Health. 2024.
  DOI: `10.1371/journal.pdig.0000538`.
- Ho SYS, et al. Accurate RR-interval extraction from single-lead telehealth
  ECG signals. medRxiv preprint, current 2025 version.
  DOI: `10.1101/2025.03.10.25323655`.
- Jeppesen J, et al. Phase-3 wearable HRV seizure-detection study. EBioMedicine.
  2025. DOI: `10.1016/j.ebiom.2025.105952`.
- Makowski D, et al. NeuroKit2: a Python toolbox for neurophysiological signal
  processing. Behavior Research Methods. 2021.
  DOI: `10.3758/s13428-020-01516-y`.
