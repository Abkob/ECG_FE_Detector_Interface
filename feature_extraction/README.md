# Dual-track RR-HRV and QRS-morphology feature-extraction branch

This package implements auditable RR-HRV measurement architectures. It does
not train a seizure classifier and does not label artifact.

It also contains a separately callable conduction/repolarization information
branch. That branch preserves paper-aligned PQRST peak dynamics and produces
five-beat descriptive summaries, but it is not called by the main pipeline and
is not eligible for the final seizure feature matrix. See
[`docs/21_CONDUCTION_INFORMATION_BRANCH_V2.md`](docs/21_CONDUCTION_INFORMATION_BRANCH_V2.md).

## New NeuroKit--Zhai architecture

```text
single-lead ECG
  |-- unmodified NeuroKit gradient detector timestamps
  |      |-- NeuroKit RR -> NeuroKit 100-RR HRV
  |      `-- NeuroKit-aligned Varon five-QRS eigenvalues
  |
  `-- full Zhai 2023 paper reimplementation
         |-- zero-phase 5--35 Hz filter
         |-- squared/5-Hz-low-pass QRS envelope
         |-- dynamic QRS windows
         |-- automatic 120-ms template
         `-- maximum absolute normalized cross-correlation timestamps
                |-- Zhai RR -> Zhai 100-RR HRV
                `-- Zhai-aligned Varon five-QRS eigenvalues

NeuroKit--Zhai matching -> support/context only
Each track -> same-track HRV + morphology fusion-ready measurements
No raw-positive-amplitude refinement
No averaging, majority vote, hard veto, or automatic timestamp owner
No seizure probability or artifact label
```

The complete Zhai method is an independent track. The package does not attach
only its correlation step to NeuroKit candidates, because that would be an
unpublished hybrid. The timestamp owner remains explicitly unselected pending
expert-annotation validation.

The full 48-record MIT--BIH audit gave pooled 75-ms F1 values of 0.98033 for
NeuroKit and 0.98564 for this Zhai reimplementation. Zhai wins only 18 of 48
individual records and performs worse at a 25-ms criterion, so neither track
is silently declared final. Details are in
[`docs/10_NEUROKIT_ZHAI_TWO_TRACK_ARCHITECTURE.md`](docs/10_NEUROKIT_ZHAI_TWO_TRACK_ARCHITECTURE.md).
The implemented morphology and fusion-ready extension is documented in
[`docs/11_DUAL_TRACK_RR_HRV_VARON_MORPHOLOGY.md`](docs/11_DUAL_TRACK_RR_HRV_VARON_MORPHOLOGY.md).

## Earlier UNSW-primary control architecture

```text
single-lead ECG
  -> original and inverted polarity contexts (no automatic routing)
  -> UNSW/Khamis primary QRS detector
  -> NeuroKit secondary detector
       -> requested 250-ms, inclusive experiment
       -> unmodified 300-ms, strict baseline
  -> collapse detections separated by <=150 ms
  -> current same-QRS support test at +/-50 ms
       -> +/-150-ms Li/Ho paper replication is separately named in the
          signal-quality branch; the older pipeline field name is retained
          only for output compatibility
  -> primary R-fiducial refinement within +/-50 ms
  -> per-QRS support and per-RR reliability mask
  -> 100-consecutive-RR Jeppesen measurements
       -> SD1, SD2, CSI100
       -> causal seven-RR median filter
       -> filtered ModCSI100
       -> heart-rate slope
       -> J1 = CSI100 x slope
       -> J2 = filtered ModCSI100 x slope
  -> measurements + reliability + missingness + elapsed coverage
```

Pan-Tompkins is also run at 300 and 250 ms, but only as diagnostic context.
It cannot delete primary events and is not called an artifact detector.

## Signal-quality v0 measurements

`ecg_cascade.signal_quality` now exposes one-window and causal trailing-window
feature extraction.  It keeps Li bSQI (Jaccard, 150 ms), Zhao qSQI (Dice, named
with its tolerance), and Ho RR endpoint support (150 ms) separate.  ADC-rail,
QRS-onset, mains, and morphology-group features are returned as unavailable
when their prerequisites are missing; they are never filled with zero.  The
branch produces continuous measurements only—no trained artifact classifier or
calibrated probability.  See
[`docs/21_SIGNAL_QUALITY_BRANCH_V0.md`](docs/21_SIGNAL_QUALITY_BRANCH_V0.md).

## Evidence and intentional deviations

- The primary detector is the Python UNSW implementation by Ho and colleagues,
  translated from Khamis et al. and distributed in NeuroKit2 0.2.13.
- The current Ho RR method uses a 150-ms close-detection exclusion, a
  +/-50-ms same-QRS tolerance, no additional secondary QRS within the RR
  interval, and +/-50-ms primary fiducial refinement.
- The requested NeuroKit 250-ms rule is experimental. Every run also calculates
  the unmodified 300-ms baseline. The code records whether the comparator is
  `>=` or `>`; the default 250-ms experiment is inclusive.
- The earlier report's +/-150-ms same-QRS tolerance is not silently discarded.
  It is calculated in parallel as `support150_ablation`.
- The published polarity CNN was a recording-level classifier. No validated
  model exists here for switching polarity inside a pathological recording.
  Both orientations are therefore measured, but the configured orientation
  remains the timestamp owner.
- Jeppesen's features were developed using verified RR intervals. This package
  never drops unsupported RR intervals before calculating the continuous
  measurements. It returns a separate reliability coverage and marks a feature
  reliable only when the configured coverage criterion is met.

## Install on Windows

From `C:\Users\Salam\Documents\ECGdetector\feature_extraction`:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[test]"
```

The existing environment is already installed in editable mode.

## Run the NeuroKit--Zhai architecture

WFDB record paths omit `.hea`:

```powershell
.venv\Scripts\ecg-features run-neurokit-zhai-wfdb `
  --record "C:\path\mitdb\113" `
  --channel 0 `
  --output-dir "C:\path\results\113_neurokit_zhai"
```

EDF:

```powershell
.venv\Scripts\ecg-features run-neurokit-zhai-edf `
  --edf "C:\path\patient.edf" `
  --channel ECG `
  --start-s 0 `
  --duration-s 600 `
  --output-dir "C:\path\results\patient_segment"
```

The output directory contains separate NeuroKit and Zhai RR/HRV tables,
five-beat Varon morphology tables, the actual aligned QRS waveform arrays,
same-track fusion-ready measurement tables, detector associations, Zhai QRS
windows/template, a JSON audit summary, and diagnostic plots. No file is
presented as a fused or selected timestamp result.

Python:

```python
from ecg_cascade import NeuroKitZhaiConfig, run_neurokit_zhai_array

result = run_neurokit_zhai_array(
    ecg,
    sampling_rate_hz,
    config=NeuroKitZhaiConfig(),
)
neurokit_features = result.selected.neurokit.features
zhai_features = result.selected.zhai.features
agreement_context = result.selected.audit_agreement
neurokit_morphology = result.selected.neurokit.morphology.features
zhai_morphology = result.selected.zhai.morphology.features
neurokit_fusion_ready = result.selected.neurokit.fusion_ready
```

## Run the earlier control on a WFDB record

The record path omits `.hea`:

```powershell
.venv\Scripts\ecg-features run-rr-hrv-wfdb `
  --record "C:\path\mitdb\113" `
  --channel 0 `
  --output-dir "C:\path\results\113"
```

Use `--orientation inverted` only as an explicit experiment. By default the
program also computes the other orientation as context, but does not route to
it automatically.

## Run on EDF

Inspect labels first:

```powershell
.venv\Scripts\ecg-features inspect-edf --edf "C:\path\record.edf"
```

Then run a selected segment:

```powershell
.venv\Scripts\ecg-features run-rr-hrv `
  --edf "C:\path\record.edf" `
  --channel ECG `
  --start-s 0 `
  --duration-s 600 `
  --output-dir "C:\path\results"
```

## Benchmark the problematic MIT-BIH records

```powershell
.venv\Scripts\ecg-features benchmark-mitdb `
  --database-dir "C:\path\mit-bih-arrhythmia-database-1.0.0" `
  --records 108 113 207 222 231 `
  --channel 0 `
  --save-record-plots `
  --output-dir "C:\path\mitdb_rr_hrv_audit"
```

This writes detector sensitivity, PPV, F1, timing error, RR coverage, supported
RR mean absolute error, and beat-symbol sensitivity. A high coverage value is
not ground truth; all metrics must be interpreted against the expert labels.

## Output files

| File | Meaning |
|---|---|
| `detector_events.csv` | Long table containing every detector, polarity, delay, sample, and time. |
| `primary_qrs_events.csv` | UNSW candidates, refined R fiducials, secondary counts, offsets, and support flags. |
| `rr_intervals.csv` | Current 50-ms per-RR reliability result. |
| `rr_hrv_features.csv` | RR values, 250/300 and 50/150 ablations, 100-RR measurements, coverage, and reliability flags. |
| `polarity_context.csv` | Original/inverted detector counts and measured coverage; never an automatic route. |
| `metadata.json` | Full configuration, versions, roles, limitations, and compact summaries. |
| `rr_hrv_complete_diagnostic.png` | ECG markers, RR support, ablation coverage, and J1/J2 plots. |

## Python API

```python
from ecg_cascade import RRHRVConfig, run_rr_hrv_array

config = RRHRVConfig(
    processing_orientation="original",
    neurokit_experimental_min_delay_ms=250,
    min_delay_inclusive=True,
    support_tolerance_ms=50,
)
result = run_rr_hrv_array(ecg, sampling_rate_hz, config=config)

features = result.selected.features
events = result.selected.primary_events
polarity = result.polarity_context
```

## Validation status

The code is software-tested, not clinically validated for seizure ECG. A
complete 48-record MIT-BIH expert-annotation audit finished without execution
failures and confirms that the components can fail differently:

- The new Zhai reproduction has pooled 75-ms F1 0.98564 versus NeuroKit's
  0.98033, largely because it repairs NeuroKit's catastrophic records 108,
  113, 207, 222, and 231. It nevertheless loses to NeuroKit on 27 individual
  records and has lower 25-ms F1 (0.91084 versus 0.94053).
- The Zhai reproduction does not reproduce the paper's reported 99.78%
  sensitivity/PPV. It is explicitly labelled a paper reimplementation because
  no official author source code was located and several operational details
  are under-specified.
- UNSW initial-event pooled F1 was 0.99726; the R-fiducial refinement was
  slightly worse at 0.99692, so both timestamp forms are retained.
- NeuroKit 250 ms worsens the record-113 T-wave double-detection problem in
  comparison with 300 ms; over all records, its pooled F1 was also slightly
  lower (0.97927 versus 0.98033).
- Whole-record inversion greatly increases detector agreement on record 207,
  but agreement alone cannot authorize automatic polarity switching.
- The +/-50-ms two-detector rule can have low coverage even when UNSW is
  accurate, so the quality mask must remain context until it is validated on
  the target patient's ECG.
- The 150-ms support ablation increased coverage but also increased mean
  supported-RR timing error, demonstrating why support tolerance cannot be
  chosen from coverage alone.

Full benchmark tables and the exact configuration are in
[`../output/rr_hrv_architecture/mitdb_all48_v1`](../output/rr_hrv_architecture/mitdb_all48_v1).

Run all tests with:

```powershell
.venv\Scripts\python -m pytest -q
```

Detailed implementation and evidence notes are in
[`docs/09_COMPLETE_RR_HRV_ARCHITECTURE.md`](docs/09_COMPLETE_RR_HRV_ARCHITECTURE.md).

## Review R peaks, labels, RR, and morphology interactively

Double-click `Launch_ECG_Cascade_Reviewer.bat` (the older
`Launch_RPeak_Reviewer.bat` remains as a compatible alias), or run:

```powershell
.venv\Scripts\ecg-rpeak-viewer `
  "C:\path\mit-bih-arrhythmia-database-1.0.0\207.hea"
```

The reviewer can choose records recursively from a dataset folder; toggle
NeuroKit, UNSW, Zhai, and expert markers; select the track evaluated against
expert `.atr` beats; highlight a particular expert beat symbol; and plot the
Varon-2015 120-ms five-beat morphology measurements. Hover near a marker for
its timing, label, cross-detector offsets, and nearest eigenvalue vector.
Right-drag selects a region and summarizes its expert labels and detector
counts. Use `Z`/`X` to zoom in/out and the arrow keys to move.

The UI shows detector TP/FP/FN against expert beats. It does not display a
seizure prediction because the morphology branch is still a feature extractor
and MIT--BIH has no seizure labels.

## Present every extraction branch in a web browser

Double-click `Launch_Clinical_Feature_Demo.bat`, or run:

```powershell
.venv\Scripts\ecg-clinical-demo
```

The local browser app accepts EDF, WFDB (`.hea` plus its `.dat` file), and
numeric CSV/TXT signals. It previews the selected ECG, runs the existing
NeuroKit--Zhai branch implementation, and replays the deterministic extraction
as an animated walkthrough. The default one-screen overview fits the signal,
all four branch summaries, slow-motion guide, and combined matrix into a
1280-by-720 viewport without page scrolling; **Detailed view** restores the
complete explanatory workbenches. The morphology workbench keeps three lanes visibly
separate: patient/lead-specific asymmetric-P95 five-QRS Varon; the frozen
whole-cycle template/residual/D-DTW lane; and automatic prominence P--QRS--T
measurements. The app does not display or export the legacy fixed-window Varon
control. Calibration defaults to the first 20 chronological eligible beats.
Its animation shows the running pre-R and post-R P95s, the exact freeze point,
the patient mask overlaid on raw QRS traces, and the later-beat-only evaluation
state. Insufficient calibration leaves patient Varon unavailable instead of
substituting another crop. The autonomic workbench independently shows the
causal 7-RR median and 100-RR Jeppesen-family measurements plus the reproduced
Varon-2015 PRSA/BPRSA lane. That lane uses an 80-beat window, 4-Hz RR and
R-amplitude resampling, acceleration-period anchors, and the published
short- and long-term PRSA/BPRSA equations. It requires no patient calibration.
The signal-quality workbench adds causal trailing 10-second integrity,
spectral, detector-agreement, RR-support, and beat-template measurements. It
shows each feature's computability and missing prerequisite independently and
does not apply a clean/artifact threshold. The conduction/repolarization
workbench shows automatic beatwise PR/QRS/QT/JT/Tpeak-to-end timing, three
named QT corrections, five-beat local summaries, nine-beat QT/RR context, and
30-beat variability. This branch is explicitly information-only because its
automatic landmarks have not completed interval-level manual validation.

The matrix can be filtered by feature family and exports 52 deterministic
columns: 25 core morphology/RR--HRV measurements, six PRSA/BPRSA context
values, eight signal-quality context values, and 13 conduction information
values. Separate flags record computability, reliability, artifact-label
absence, conduction eligibility, and whether a context family affects the
core gate. Neither quality nor conduction decides whether the core
morphology--RR/HRV row is valid.
This keeps potentially informative autonomic/cardiorespiratory behavior
visible without silently treating fragile BPRSA values as clean inputs. The
matrix does not implement the paper's kernel-spectral-clustering classifier or
report a seizure probability.

For a no-setup presentation, use **Prepared example** in the app. It loads the
bundled MIT--BIH record 100 and defaults to a 120-second segment, which is long
enough to demonstrate the complete 100-RR HRV window. The timestamp-track
selector changes which independent detector track is presented; it does not
declare a validated timestamp owner. “Combined” means a causal same-track
measurement join only. The app does not produce seizure probabilities,
artifact labels, or clinical decisions. The browser demo always runs automatic
prominence delineation on the selected signal. MIT--BIH expert annotations are
beat labels, not manual QRS onset/offset ground truth, so the prepared example
labels calibration boundaries as automatic estimates rather than truth.

## Reproduce the PRSA/BPRSA audit

Run the deterministic Varon-2015 PRSA/BPRSA extractor over the available
MIT--BIH records and local CHB04_28 recording:

```powershell
.venv\Scripts\python scripts\benchmark_prsa_bprsa_available_data.py `
  --output outputs\prsa_bprsa_available_data_v1
```

The MIT--BIH portion tests feature definition, numerical quality, support, and
lead/polarity sensitivity against expert R-peak anchors. It cannot measure
seizure accuracy because MIT--BIH has no seizure labels. The CHB portion is an
exploratory stress test only: the automatic detector loses strict support
during the labeled seizures, so its unfiltered separation scores are rejected
as clinical evidence. Exact equations, deviations, results, output fields, and
remaining validation gates are recorded in
[`docs/latex_morphology_branch/morphology_branch_complete_technical_record.tex`](docs/latex_morphology_branch/morphology_branch_complete_technical_record.tex).
The focused context-matrix, controlled-noise, arrhythmia-confounding, and fresh
CHB seizure-interval audits are summarized in
[`docs/22_PRSA_BPRSA_CONTEXT_VALIDATION.md`](docs/22_PRSA_BPRSA_CONTEXT_VALIDATION.md).

## Reproduce the morphology fidelity audit

Five difficult records:

```powershell
.venv\Scripts\python scripts\benchmark_morphology_fidelity.py `
  --dataset "C:\path\mit-bih-arrhythmia-database-1.0.0" `
  --records 108 113 207 222 231 `
  --output outputs\morphology_fidelity_five_records_v1
```

Omit `--records` to run every numeric WFDB record. The output tests detector
accuracy and fidelity of the five eigenvalues to expert-timestamp features;
it does not measure seizure accuracy. See
[`docs/17_MORPHOLOGY_FIDELITY_AND_REVIEWER.md`](docs/17_MORPHOLOGY_FIDELITY_AND_REVIEWER.md).

## Run the personalized morphology challengers

The published fixed 120-ms Varon branch remains unchanged.  Three independent
research challengers are now separately callable:

- a patient-average symmetric Varon width, retained as a measured negative
  result, a corrected symmetric p95 width, and a separately calibrated
  pre/post-R p95 challenger with a fixed-window fallback below five beats;
- a patient/lead whole-cycle template bank with correlation, residual, and
  constrained derivative-DTW features;
- prominence P/QRS/T delineation with explicit missingness and ordering flags.

Reproduce the patient-average capture audit:

```powershell
.venv\Scripts\python scripts\benchmark_patient_average_varon_capture.py `
  --output outputs\patient_symmetric_varon_capture_v2
```

Reproduce the chronological asymmetric-window audit:

```powershell
.venv\Scripts\python scripts\benchmark_patient_asymmetric_varon_capture.py `
  --output outputs\patient_asymmetric_varon_capture_v3
```

The asymmetric audit uses manual QRS boundaries from LUDB lead II and the
matched QTDB MLII subset. MIT-BIH Arrhythmia is intentionally excluded from
this boundary test because it has beat annotations but no beatwise manual QRS
onset/offset ground truth.

Run the prominence manual-landmark gate:

```powershell
.venv\Scripts\python scripts\benchmark_delineation_gate.py `
  --methods prominence `
  --output outputs\delineation_gate_prominence_matched_lead_v1
```

Run a causal initial-template experiment on one WFDB lead:

```powershell
.venv\Scripts\python scripts\run_patient_template_morphology.py `
  --record "C:\path\mit-bih-arrhythmia-database-1.0.0\100" `
  --channel 0 `
  --duration-s 60 `
  --calibration-beats 20 `
  --patient-id mitdb_100 `
  --output outputs\patient_template_mitdb_100_60s_v1
```

The implementations, exact results, feature meanings, and validation limits
are documented in
[`docs/19_PERSONALIZED_MORPHOLOGY_BRANCH_V1.md`](docs/19_PERSONALIZED_MORPHOLOGY_BRANCH_V1.md).

## Conduction timing branch prototype

The standalone `conduction` branch consumes aligned P/QRS/T landmarks and
exports raw PR, QRS, QT, and JT intervals, named QT-correction alternatives,
first differences, and causal trailing beat-to-beat summaries. It remains
outside `pipeline.py`: current P-onset and T-offset error tails can exceed the
small physiological variation that QT-variability methods attempt to measure.
Run a clearly labeled diagnostic with reference R anchors and automatic
prominence boundaries:

```powershell
.venv\Scripts\python scripts\run_conduction_timing.py `
  --record "C:\path\mit-bih-arrhythmia-database-1.0.0\100" `
  --channel 0 `
  --duration-s 60 `
  --patient-id mitdb_100 `
  --output outputs\conduction_mitdb_100_60s_v1
```

See
[`docs/20_CONDUCTION_TIMING_BRANCH_V1.md`](docs/20_CONDUCTION_TIMING_BRANCH_V1.md).

## Executed standalone branch notebooks

Four independent, detailed Jupyter notebooks are available in
[`notebooks/standalone_branches`](notebooks/standalone_branches):

- `01_RR_HRV_Standalone_Detailed.ipynb`
- `02_Morphology_Standalone_Detailed.ipynb`
- `03_Signal_Quality_Artifact_Context_Standalone_Detailed.ipynb`
- `04_Conduction_Standalone_Detailed.ipynb`

Each notebook loads its own ECG input, freezes its own configuration, runs the
tested package implementation, displays intermediate tables and diagnostic
plots, exports CSV/JSON artifacts under `outputs/standalone_notebooks`, runs
explicit numerical and safety assertions, and embeds the complete relevant
source modules with SHA-256 hashes. The checked-in notebooks were executed on
MIT--BIH record 100/MLII and contain their outputs.

Regenerate and execute the suite from the repository root:

```powershell
feature_extraction\.venv\Scripts\python.exe tools\build_standalone_branch_notebooks.py
feature_extraction\.venv\Scripts\python.exe tools\run_standalone_branch_notebooks.py
```

The signal-quality notebook demonstrates controlled degradation but does not
claim to be an artifact classifier. The conduction notebook remains
information-only, and none of the notebooks produces a seizure or clinical
decision.

## Primary references

- Zhai et al. (2023), DOI `10.3934/mbe.2023848`.
- Khamis et al. (2016), DOI `10.1109/TBME.2016.2549060`.
- Kristof et al. (2024), DOI `10.1371/journal.pdig.0000538`.
- Ho et al. (2025), DOI `10.1101/2025.03.10.25323655` (preprint).
- Jeppesen et al. (2025), DOI `10.1016/j.ebiom.2025.105952`.
- Varon et al. (2015), DOI `10.1016/j.jelectrocard.2015.08.020`.
- Emrich et al. (2024), EUSIPCO paper `0001402`.
