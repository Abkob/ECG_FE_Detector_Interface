# Personalized morphology branch v1

## Decision

The morphology study now keeps four separately auditable lanes.  They are not
silently fused and none produces a seizure prediction.

1. The published Varon-2015 symmetric 120-ms, five-QRS Gram eigenspectrum is
   frozen as the replication control.
2. A patient-average-width Varon experiment changes only the symmetric capture
   width to the arithmetic mean total QRS duration measured in earlier manual
   calibration beats.
3. A patient/lead-specific whole-cycle template bank measures full P-QRS-ST-T
   shape with correlation, raw and normalized residuals, beat-to-beat change,
   and constrained derivative-DTW.
4. The Emrich-2024 prominence delineator supplies interpretable P/QRS/T
   landmarks, intervals, amplitudes, area, polarity, missingness, and ordering
   flags.

The RR/HRV and morphology branches share R anchors but remain distinct: RR/HRV
measures beat timing; these lanes measure waveform geometry.

## Patient-average-width Varon experiment

### Exact rule

For each record and calibration size `k`:

1. Select the first `k` manual QRS onset/R/offset triplets.
2. Calculate the arithmetic mean total QRS duration `X`.
3. Extract `X/2` before and `X/2` after every later reference R anchor.
4. Compare complete-manual-QRS capture with the unchanged 60+60-ms control.

No onset-to-R, R-to-offset, waveform-template, safety-margin, population, or
classifier parameter is introduced into this experiment.

### Held-out manual-boundary result

| dataset | calibration beats | records | later beats | median personalized width | fixed 120-ms capture | patient-average capture |
|---|---:|---:|---:|---:|---:|---:|
| LUDB II | 3 | 199 | 1,220 | 93.3 ms | 73.6% | 18.4% |
| LUDB II | 5 | 196 | 823 | 94.0 ms | 74.1% | 18.8% |
| LUDB II | 10 | 44 | 105 | 91.2 ms | 74.3% | 21.9% |
| QTDB MLII | 3 | 12 | 426 | 88.7 ms | 62.2% | 16.4% |
| QTDB MLII | 5 | 12 | 402 | 90.4 ms | 61.4% | 17.4% |
| QTDB MLII | 10 | 12 | 342 | 92.6 ms | 62.0% | 19.3% |
| QTDB MLII | 20 | 12 | 222 | 94.9 ms | 61.3% | 27.5% |

The proposed total-width personalization is therefore a valid negative result:
it makes the crop shorter and does not account for R being off-center within
the QRS.  It must not replace the 120-ms control.  The code is retained so the
result remains reproducible and future asymmetric or margin ablations can be
compared against the exact original proposal.

Outputs:

- `outputs/patient_average_varon_capture_v1/held_out_beat_capture.csv`
- `outputs/patient_average_varon_capture_v1/record_calibrations.csv`
- `outputs/patient_average_varon_capture_v1/capture_summary.csv`
- `outputs/patient_average_varon_capture_v1/benchmark_scope.json`

### Corrected one-number symmetric calibration

The follow-up keeps the user's intended one-width-per-patient design but
calibrates the quantity that a symmetric R-centered crop actually requires.
For calibration beat `i`:

```text
required_width_i = 2 * max(R_i - QRS_onset_i, QRS_offset_i - R_i)
```

The patient still receives one number `X`; no separate onset-to-R or
R-to-offset feature is added.  Three summaries were tested: mean, p95, and
maximum calibration required width.

| dataset | calibration beats | later beats | fixed 120-ms capture | mean required capture | p95 required capture | maximum required capture |
|---|---:|---:|---:|---:|---:|---:|
| LUDB II | 3 | 1,220 | 73.6% | 53.7% | 72.2% | 81.1% |
| LUDB II | 5 | 823 | 74.1% | 57.4% | 83.1% | 87.4% |
| LUDB II | 10 | 105 | 74.3% | 51.4% | 89.5% | 94.3% |
| QTDB MLII | 3 | 426 | 62.2% | 51.9% | 72.8% | 75.1% |
| QTDB MLII | 5 | 402 | 61.4% | 53.5% | 79.1% | 83.3% |
| QTDB MLII | 10 | 342 | 62.0% | 50.6% | 82.5% | 90.9% |
| QTDB MLII | 20 | 222 | 61.3% | 51.8% | 88.7% | 91.9% |

The p95 variant is the preferred personalized-width challenger.  With five
calibration beats its median windows were 116 ms on LUDB and 133.6 ms on QTDB,
while complete-QRS capture rose to 83.1% and 79.1%.  Three LUDB calibration
beats were insufficient: p95 capture was 72.2%, slightly below fixed 120 ms.

The maximum variant obtained the highest capture, but its record-level window
distribution had very large tails.  Depending on dataset and calibration size,
the 95th percentile of patient maximum windows reached roughly 184--232 ms.
Those crops can include substantial non-QRS signal and can be controlled by one
unusual calibration beat.  Maximum is therefore retained as a sensitivity
analysis rather than the default.

The LUDB 10-beat row contains only 44 records with enough later beats and is
selection-limited.  All results concern manual-boundary capture, not seizure
discrimination or Varon-feature superiority.

Corrected outputs:

- `outputs/patient_symmetric_varon_capture_v2/held_out_beat_capture.csv`
- `outputs/patient_symmetric_varon_capture_v2/record_calibrations.csv`
- `outputs/patient_symmetric_varon_capture_v2/capture_summary.csv`
- `outputs/patient_symmetric_varon_capture_v2/benchmark_scope.json`

### Separately calibrated asymmetric p95 window

Status: implemented and tested as an experimental crop. It has not replaced
the fixed Varon control and has not been evaluated for seizure discrimination.

For the first `k` complete same-record, same-lead manual QRS triplets:

```text
pre_ms  = p95(R - QRS_onset)
post_ms = p95(QRS_offset - R)
```

The implementation applies NumPy's default linear quantile separately to the
two calibration distributions. Each side is rounded outward with `ceil`, and
the R sample is included once:

```text
pre_samples  = ceil(pre_ms  * fs / 1000)
post_samples = ceil(post_ms * fs / 1000)
crop = ecg[R - pre_samples : R + post_samples + 1]
```

Fewer than five calibration beats trigger an explicit fixed-window fallback.
This safeguard prevents the measured three-beat degradation, but five is not
claimed to be sufficient for deployment.

A real wrapper check on LUDB `data/1`, lead II, calibrated from its first five
complete triplets. Separate p95 values of 53.2 ms before and 65.6 ms after R
became 27 and 33 samples at 500 Hz, plus R once (61 samples). Six anchor rows
and both possible five-beat Gram-feature rows were defined. This is an execution
check, not evidence of seizure relevance or feature improvement.

The earlier `extract_varon_morphology` code used the total requested duration
and then split its sample count symmetrically. It therefore could not honor an
asymmetric request. The new experimental path uses independent sample counts.
The unchanged fixed control still uses its original rule. Consequently its
actual boundary reach is 60/58 ms in 500-Hz LUDB and 60/56 ms in 250-Hz QTDB,
although the historical nominal audit remains 60/60 ms. Both controls are
reported so earlier results remain reproducible.

#### Ground truth and split

- LUDB: lead-II cardiologist QRS onset/R/offset annotations. Record `data/104`
  had no complete triplet and was skipped; 199 records contributed at `k=3`.
- QTDB: the 12 records whose annotated channel 0 is MLII. Selected manual
  `q1c` boundaries were associated with the record's expert `atr` R anchors.
- Calibration is always the first `k` complete annotated triplets; evaluation
  uses only later triplets from the same record.
- MIT-BIH Arrhythmia was not used. Its beat annotations are valid R/beat
  references, but they are not beatwise manual QRS-onset/QRS-offset truth.

#### Held-out boundary-capture result

The table compares the actual sampled fixed extractor, the earlier corrected
symmetric p95 crop, and the proposed separate p95 crop. `Extra` is the mean
amount of sampled signal before manual onset plus after manual offset; it is a
geometric quantity and does not prove feature contamination.

| dataset | k | records | later beats | method | median boundary reach | complete capture | mean extra |
|---|---:|---:|---:|---|---:|---:|---:|
| LUDB II | 3 | 199 | 1,220 | sampled fixed | 118 ms | 72.5% | 26.4 ms |
| LUDB II | 3 | 199 | 1,220 | separate p95 | 102 ms | 65.5% | 12.5 ms |
| LUDB II | 5 | 196 | 823 | sampled fixed | 118 ms | 73.4% | 26.4 ms |
| LUDB II | 5 | 196 | 823 | symmetric p95 | 114 ms | 81.7% | 29.7 ms |
| LUDB II | 5 | 196 | 823 | separate p95 | 106 ms | 75.5% | 16.7 ms |
| LUDB II | 10 | 44 | 105 | sampled fixed | 118 ms | 73.3% | 25.6 ms |
| LUDB II | 10 | 44 | 105 | symmetric p95 | 118 ms | 88.6% | 35.0 ms |
| LUDB II | 10 | 44 | 105 | separate p95 | 106 ms | 86.7% | 23.1 ms |
| QTDB MLII | 3 | 12 | 426 | sampled fixed | 116 ms | 60.3% | 29.4 ms |
| QTDB MLII | 3 | 12 | 426 | separate p95 | 112 ms | 57.5% | 23.8 ms |
| QTDB MLII | 5 | 12 | 402 | sampled fixed | 116 ms | 59.7% | 29.5 ms |
| QTDB MLII | 5 | 12 | 402 | symmetric p95 | 130 ms | 79.1% | 50.2 ms |
| QTDB MLII | 5 | 12 | 402 | separate p95 | 116 ms | 65.9% | 26.8 ms |
| QTDB MLII | 10 | 12 | 342 | sampled fixed | 116 ms | 59.9% | 29.6 ms |
| QTDB MLII | 10 | 12 | 342 | symmetric p95 | 140 ms | 81.3% | 59.4 ms |
| QTDB MLII | 10 | 12 | 342 | separate p95 | 136 ms | 77.8% | 36.9 ms |
| QTDB MLII | 20 | 12 | 222 | sampled fixed | 116 ms | 59.9% | 30.9 ms |
| QTDB MLII | 20 | 12 | 222 | symmetric p95 | 140 ms | 88.3% | 71.0 ms |
| QTDB MLII | 20 | 12 | 222 | separate p95 | 142 ms | 88.7% | 47.1 ms |

The proposal is therefore **useful but not yet the default**. It is generally
more crop-efficient than the symmetric p95 rule: for example, LUDB `k=10`
lost only 1.9 percentage points of capture while reducing median reach by 12 ms
and mean extra signal by 11.8 ms. QTDB required more calibration; at `k=20`,
separate p95 slightly exceeded symmetric capture while including 23.9 ms less
extra signal on average. With only three beats it was worse than fixed on both
datasets, and with five beats its capture remained well below symmetric p95.

The maximum of each separate calibration side was retained as a sensitivity
analysis, not selected as the method. Manual P-offset or T-onset overlap was
also measured where available. Overlap rates were low in this test, but QTDB
had only 11--28 held-out T-onset annotations depending on `k`, so that result
cannot establish general T-wave safety.

Outputs:

- `outputs/patient_asymmetric_varon_capture_v3/held_out_beat_windows.csv`
- `outputs/patient_asymmetric_varon_capture_v3/record_calibrations.csv`
- `outputs/patient_asymmetric_varon_capture_v3/capture_summary.csv`
- `outputs/patient_asymmetric_varon_capture_v3/skipped_records.csv`
- `outputs/patient_asymmetric_varon_capture_v3/benchmark_scope.json`
- `outputs/patient_asymmetric_varon_capture_v3/SUMMARY.md`

## Whole-cycle patient template bank

### Beat construction

For an R anchor `R_i`, the complete cycle begins at the midpoint between
`R_(i-1)` and `R_i` and ends at the midpoint between `R_i` and `R_(i+1)`.
The pre-R and post-R portions are resampled separately to 64 and 128 points.
This keeps R at a fixed output column while the original RR timing remains in
the RR/HRV branch.

Polarity is preserved.  The median of the first and last five percent of the
resampled cycle supplies a local baseline.  Both baseline-centered raw
waveforms and unit-L2-normalized waveforms are retained.

### Calibration and templates

Only the first requested complete cycles form the fixed patient/lead baseline.
A deterministic chronological template bank retains up to three recurring
calibration shapes:

- compare each new calibration beat with existing normalized median templates;
- assign it to the highest-correlation template when correlation is at least
  0.90;
- otherwise create a new template if fewer than three exist;
- after calibration, freeze every template for later evaluation beats.

The 0.90 threshold and three-template limit are implementation decisions and
must receive sensitivity analyses before clinical use.  A singleton template
is retained and reported rather than silently deleted.

### Per-beat measurements

- matched template index and its calibration member count;
- signed template correlation;
- normalized-shape RMSE;
- raw residual RMS and energy;
- constrained derivative-DTW cost and warp fraction;
- previous-beat normalized RMSE and correlation;
- cycle, pre-R, and post-R durations;
- baseline, signed R amplitude, peak-to-peak amplitude, and RMS;
- calibration/evaluation flags and explicit no-classifier flags.

Low correlation remains a measurement.  It is not automatically called noise,
artifact, arrhythmia, or seizure.

### Real-data execution checks

- MIT-BIH record 100, channel-0 MLII, first 60 seconds: 74 reference anchors,
  72 complete midpoint cycles, 20 calibration cycles, 52 later evaluation
  cycles, and one 20-member template.  Median later-beat correlation was about
  0.989.
- MIT-BIH record 207, channel-0 MLII, first 120 seconds: 110 reference anchors,
  108 complete cycles, 30 calibration cycles, 78 later evaluation cycles, and
  three templates with 15, 14, and 1 members.  Later morphology absent from
  the calibration section remained low-correlation rather than being dropped.

These checks establish execution and expected baseline/novelty behavior.  They
do not establish seizure relevance or optimal template thresholds.

## Prominence P-QRS-T lane

The NeuroKit2 0.2.13 implementation of the Emrich-2024 prominence method was
run using the same lead-controlled manual gate as the earlier DWT/CWT study:
all 200 LUDB lead-II records and the 12 QTDB records whose manually annotated
channel 0 is MLII.  Reference R anchors were supplied to isolate delineation.

### QRS result

| dataset | landmark | reference | sensitivity | PPV | median absolute error | P95 absolute error |
|---|---|---:|---:|---:|---:|---:|
| LUDB II | QRS onset | 1,817 | 100% | 100% | 8 ms | 24 ms |
| LUDB II | QRS offset | 1,830 | 100% | 100% | 14 ms | 44 ms |
| QTDB MLII | QRS onset | 462 | 100% | 100% | 12 ms | 48 ms |
| QTDB MLII | QRS offset | 462 | 100% | 100% | 16 ms | 72 ms |

This is a large QRS-onset improvement over the tested NeuroKit DWT/CWT paths.
QTDB QRS-offset tail error remains material and must be reported.

### P/T limitations

P coverage was high, but QTDB P-onset median absolute error was 20 ms and P95
was 68.2 ms.  T offset reached 96.9% sensitivity on LUDB but only 86.4% on the
matched QTDB subset.  QTDB has only 31 manual T-onset references in this scope;
prominence matched them but had a 96-ms median absolute error.  Therefore P/T
features carry availability and physiological-order flags and are not treated
as universally reliable.

The runtime extractor returns durations, PR/QRS/ST/QT intervals, signed P/R/T
amplitudes, QRS peak-to-peak amplitude, signed QRS area, polarity, landmark
availability, completeness, and ordering.  Missing landmarks remain missing.

## Code and commands

Implemented modules:

- `src/ecg_cascade/morphology.py`: fixed, patient-average, corrected symmetric,
  and patient/lead-specific asymmetric Varon capture paths.
- `src/ecg_cascade/patient_template_morphology.py`: whole-cycle templates and
  derivative-DTW measurements.
- `src/ecg_cascade/prominence_morphology.py`: prominence delineation and PQRST
  feature table.

Patient-average capture audit:

```powershell
.venv\Scripts\python scripts\benchmark_patient_average_varon_capture.py `
  --output outputs\patient_symmetric_varon_capture_v2
```

Asymmetric capture audit:

```powershell
.venv\Scripts\python scripts\benchmark_patient_asymmetric_varon_capture.py `
  --output outputs\patient_asymmetric_varon_capture_v3
```

Prominence gate:

```powershell
.venv\Scripts\python scripts\benchmark_delineation_gate.py `
  --methods prominence `
  --output outputs\delineation_gate_prominence_matched_lead_v1
```

Whole-cycle WFDB run:

```powershell
.venv\Scripts\python scripts\run_patient_template_morphology.py `
  --record "C:\path\mit-bih-arrhythmia-database-1.0.0\100" `
  --channel 0 `
  --duration-s 60 `
  --calibration-beats 20 `
  --patient-id mitdb_100 `
  --output outputs\patient_template_mitdb_100_60s_v1
```

## Current boundary

The feature extractors are separately callable and software-tested.  They have
not yet been joined to the main RR/HRV runtime output, benchmarked across all
48 MIT-BIH records, or evaluated on chronologically held-out seizure periods.
The next comparison must keep every lane separate: RR/HRV-only, fixed Varon,
patient-average Varon, prominence features, whole-cycle templates, and their
explicit combinations.
