# Rigorous classification audit of the comprehensive ECG branch matrix

## Verdict boundary

This audit tests whether the 52 extracted measurements contain information
about five explicit binary labels. It does not validate a clinical diagnostic
device. Every reported prediction is out-of-group: the complete
`lineage_group_id` that owns a row is absent while its preprocessing and model
are fitted. There is still no untouched external cohort. A configuration
selected from these same cross-validation results is therefore an inspection
winner, not an unbiased final-test estimate.

The older model page used one deterministic holdout and then chose its best
cell from that holdout. That protocol is optimistic and is retained only as a
historical comparison. `rigorous_audit_v2` replaces it for engineering review.

## Completed rerun result

The September 4, 2026 rerun produced 375 complete target/model/branch cells
and 3,452,850 durable out-of-fold predictions. The inspection winner for each
target is shown below. “Group BA” is the selection metric; the interval is the
row-pooled balanced accuracy after 1,000 resamples of complete lineage groups.

| Target | Winner | Rows / groups | TN / FP / FN / TP | Group BA | Row BA (95% group-bootstrap CI) | Errors |
|---|---|---:|---:|---:|---:|---:|
| Artifact | logistic regression, B2+B4 | 11,317 / 17 | 4,737 / 318 / 759 / 5,503 | 0.903 | 0.908 (0.871–0.949) | 1,077 |
| Seizure | linear SVM, B1 | 11,346 / 3 | 7,230 / 2,960 / 428 / 728 | 0.671 | 0.670 (0.652–0.688) | 3,388 |
| Abnormal beat | histogram gradient boosting, B1+B2+B3 | 15,651 / 48 | 11,277 / 1,222 / 949 / 2,203 | 0.800 | 0.801 (0.723–0.870) | 2,171 |
| NSTDB noise active | linear SVM, B2+B4 | 2,847 / 2 | 798 / 36 / 359 / 1,654 | 0.889 | 0.889 (0.886–0.893) | 395 |
| Strict unusable | linear SVM, B2 | 4,877 / 15 | 4,138 / 83 / 48 / 608 | 0.970 | 0.954 (0.894–0.996) | 131 |

These values are not all correct and are not final-test estimates. The seizure
winner is particularly weak: its group-macro BA is 0.671 versus 0.654 from a
missingness-only control. Noise has only two independent source patients.
Strict unusable is high, but feature missingness alone already reaches 0.769
group-macro BA, so its 0.970 result needs an external source before it can be
treated as generalizable.

All 20 training-label permutations per selected cell returned near-chance
means: abnormal 0.485, artifact 0.482, noise 0.542, seizure 0.503 and unusable
0.491. This is a necessary negative control, not proof of clinical validity.

The raw inventory byte-read and SHA-256-hashed 4,856 files (8,150,494,018
bytes) with zero read errors. Six isolated raw-to-matrix reconstructions then
reproduced 39,664/39,664 row IDs with zero missing or unexpected rows, zero
differences across the 52 feature columns, zero semantic differences across
the 56 label columns and zero failed segments. Ten separate label-semantic
checks also found zero mismatches. These checks establish reproducibility of
the curated matrix, not representativeness of an external clinical population.

## What was read

The audit has two distinct scopes that must not be confused:

1. The raw integrity pass byte-reads every file under `Datasets/` plus every
   configured seizure EDF and stores file size and SHA-256. This proves which
   local bytes were present and readable.
2. The model audit uses the completed label-rich matrix. A raw waveform is
   eligible for a supervised target only where its source database supplies
   that target. Unknown target values stay null and are excluded; they are
   never turned into class 0.

The exhaustive waveform plan contains 557.81 signal-hours, 392 records and
11,337 gap-free windows. Much of that signal has no applicable label. Running
a supervised classifier on those unlabelled samples would not add ground
truth; it would only add invented negatives. The exhaustive plan is therefore
kept separate from the label-rich supervised evidence.

The completed matrix is not falsely described as a full-waveform extraction.
Its frozen validation profile uses one label-diverse 180-second MIT-BIH window
per record, the first NSTDB electrode-motion interval with causal context, all
ten-second LUDB records, twelve QTDB MLII/q1c records, the longest available
BUT-QDB interval for each represented quality class, and every configured
seizure with five-minute context plus an inter-event control. Consequently the
out-of-group rerun audits every row and patient *in that matrix*, but it does
not establish that the selected windows represent all 557.81 raw hours. The
raw inventory and exhaustive plan make that gap measurable instead of hiding
it behind the word “comprehensive.”

One row is a pipeline-produced beat anchor, not an arbitrary waveform sample.
For abnormal-beat classification, expert truth is attached only when that
anchor is within 75 ms of an expert beat. Expert beats that the detector misses
do not become classifier false negatives because no matrix row exists for
them. The abnormal task therefore evaluates the classification of matched,
detected beats; it does **not** evaluate end-to-end abnormal-beat detection.
Likewise, quality and seizure tasks describe produced anchors inside their
labelled time regions. Detector failure must be audited separately and cannot
be repaired by a good classifier score.

## The four branches

### B1 — peak detection, RR, HRV and PRSA/BPRSA context (15 columns)

B1 contains instantaneous RR and heart rate; 100-RR SD1, SD2, CSI, filtered
ModCSI, slope, J1 and J2; and 80-beat PRSA/BPRSA context. These are causal
history measurements. Early rows legitimately lack long-window values.

### B2 — morphology (16 columns)

B2 contains five calibrated patient-specific Varon eigenvalues, frozen
whole-cycle template correlation/residual/D-DTW measurements, and automatic
P–QRS–T duration, amplitude, area and landmark-availability measurements.
Patient-template calibration freezes from the chronological calibration
prefix; label values do not choose the template.

### B3 — conduction and repolarization information (13 columns)

B3 contains JT and Tpeak-to-end timing, three QT corrections, changes in PR/QT,
P-to-R and T-to-R timing, five-beat QT summaries and a nine-beat mean
Fridericia QTc. These automatic landmarks remain information-only and are not
clinically validated reference intervals.

### B4 — signal-quality context (8 columns)

B4 contains finite and flat fractions, baseline- and QRS-band spectral ratios,
two detector-agreement scores, RR support and beat-template correlation. It
contains continuous measurements only. It does not contain the BUT-QDB class,
the NSTDB schedule or a pipeline-produced artifact decision.

All 15 non-empty branch sets are evaluated: four single branches, six pairs,
four triples and B1+B2+B3+B4. The model receives only the exact numeric columns
listed for that combination in `branch_combinations.csv`.

## Dataset combination rules

There is no universal label shared by all six databases. Pooling is
target-specific:

| Dataset | Information contributed | Binary tasks supported |
|---|---|---|
| MIT-BIH Arrhythmia | Expert `.atr` beat symbols | abnormal beat |
| MIT-BIH Noise Stress Test | Reused 118/119 beat truth plus official electrode-motion schedule and SNR | abnormal beat, noise active, composite artifact |
| QT Database | Expert beats and manual delineation | abnormal beat where expert beat truth exists; manual intervals remain label-side references |
| BUT-QDB | Three annotators and official consensus quality classes | strict unusable, composite artifact |
| LUDB | Diagnoses and manual lead landmarks | descriptive/multilabel and interval references; no current binary target |
| CHB-MIT/Siena EDFs | Configured seizure onset/offset intervals | ictal seizure interval |

The composite artifact target intentionally asks a broader question than
either source alone: BUT-QDB class 1 maps to 0 and classes 2/3 map to 1, while
NSTDB inactive maps to 0 and active maps to 1. Results must also be examined by
source because a model can exploit corpus differences. The dataset-only
baseline measures this risk.

## Exact binary labels

- `target_artifact_binary`: 0 for a pure trailing-ten-second BUT-QDB class-1
  window or inactive NSTDB schedule; 1 for a pure class-2/3 window or active
  NSTDB schedule.
- `target_signal_unusable_binary`: BUT-QDB class 1 = 0 and class 3 = 1;
  class 2 is excluded.
- `target_noise_active_binary`: official NSTDB inactive = 0 and active = 1.
- `target_abnormal_beat_binary`: AAMI N = 0; AAMI S/V/F/Q = 1. The original
  WFDB symbol remains beside this derived view.
- `target_seizure_binary`: 1 only inside an original configured seizure
  interval and 0 elsewhere in a configured seizure recording.

## Dependency grouping

Rows are never split randomly. A lineage can include more than one file:

- every NSTDB SNR variant derived from MIT-BIH 118 belongs to `mitdb:118`, and
  likewise for 119;
- QTDB excerpts whose header identifies a MIT-BIH source share that MIT-BIH
  lineage;
- all recordings from one Siena patient share one patient lineage;
- all windows from one BUT-QDB patient prefix remain together;
- repeated segments and both timestamp tracks, if present, remain together.

Thus no fold may train on one noisy copy or excerpt and test on its source.

## How a fold is fitted

For a target, rows with known truth are collected and complete lineage groups
are assigned to deterministic stratified folds. Five folds are used when at
least five groups exist; the seizure task has three patient folds and NSTDB
noise has two source-patient folds. For every model and branch combination:

1. The current fold's lineage groups are removed in full.
2. Median imputation, missing indicators and scaling—when required by the
   model—are fitted only on remaining training rows.
3. The classifier is fitted with its frozen configuration and balanced class
   weights.
4. The excluded groups are predicted once.
5. Predictions from all folds are concatenated so every labelled row has one
   out-of-group result.

The five frozen classifier families are regularized logistic regression,
linear SVM, random forest, extremely randomized trees and histogram gradient
boosting. They are a complementary comparison panel, not a claim that five
papers identify universally best ECG classifiers.

## Calculations

For confusion counts TN, FP, FN and TP:

```text
sensitivity        = TP / (TP + FN)
specificity        = TN / (TN + FP)
precision          = TP / (TP + FP)
negative predictive value = TN / (TN + FN)
accuracy           = (TP + TN) / (TN + FP + FN + TP)
balanced accuracy  = (sensitivity + specificity) / 2
```

Accuracy is never treated as sufficient under imbalance. ROC AUC measures
ranking, not the percentage of correct predictions. Average precision is
reported beside prevalence. Undefined per-record rates remain blank.

Row-pooled metrics allow long recordings to contribute more beats. The primary
audit ranking therefore uses group-macro sensitivity and specificity: each
eligible lineage contributes one rate before the two macro rates are averaged.
The 95% uncertainty interval resamples complete lineage groups 1,000 times;
individual beats are never bootstrapped as if independent.

## Leakage and shortcut challenges

The audit checks all of the following:

- uniqueness of `row_id` and source record/time coordinates;
- absence of label, target, patient, record, subject or fold tokens from the
  52-column feature contract;
- exact duplicated feature vectors and whether any cross lineage groups;
- invariant lineage-to-fold assignment;
- train/test lineage overlap for every fold;
- train-only preprocessing;
- an out-of-group model using only feature-missingness masks;
- an out-of-group model using only dataset identity;
- 20 repeats with training labels permuted for the selected inspection cell;
- per-feature univariate target separation and class-dependent missingness;
- every fold, record, patient, false positive and false negative.

A near-chance permutation result is necessary but not sufficient evidence
against leakage. A high dataset-only or missingness-only result is a warning
that source or availability shortcuts exist. Strong physiological features can
also legitimately separate a target, so high univariate AUC is a review flag,
not automatic proof of leakage.

## Evidence outputs

`outputs/comprehensive_branch_matrix_v1/rigorous_audit_v2/` contains:

- `raw_file_inventory.csv` and `raw_duplicate_files.csv`;
- `dataset_inventory.csv`, `lineage_group_inventory.csv` and missingness maps;
- `group_oof_results.csv` for all 375 experiments;
- `fold_metrics.csv` and `split_manifest.csv`;
- `all_group_oof_predictions.csv.gz` for every experiment and row;
- `selected_case_predictions.csv.gz` with all 52 values for each displayed
  inspection winner;
- `record_metrics.csv`, `patient_metrics.csv` and `weak_records_ranked.csv`;
- `weak_cases_all_errors.csv.gz` for every false positive and false negative;
- `weak_case_explanations.csv.gz`, a non-causal comparison of each error with
  the true-class and predicted-class feature medians plus its missing columns;
- missingness, dataset-identity and naive baselines;
- permuted-label checks and univariate feature/target association tables;
- `audit_results.json` and `AUDIT_REPORT.md`.

The web page displays a bounded weak-case preview for responsiveness; the GZIP
files remain the authoritative exhaustive case ledger.

## Primary external references checked for this audit

- PhysioNet, *MIT-BIH Noise Stress Test Database v1.0.0*:
  https://physionet.org/content/nstdb/1.0.0/ — confirms that the stress records
  are generated from only MIT-BIH 118 and 119, use copied beat annotations and
  alternate two-minute noisy/clean intervals after minute five.
- PhysioNet, *Brno University of Technology ECG Quality Database v1.0.0*:
  https://physionet.org/content/butqdb/1.0.0/ — confirms 18 recordings from 15
  subjects, the three-expert consensus and the exact class 1/2/3 meanings.
- Collins et al., *TRIPOD+AI statement* (BMJ 2024):
  https://www.bmj.com/content/385/bmj-2023-078378 — motivates explicit data
  sources, model-building steps, internal-validation reporting and attention
  to clustered data.
- Banerjee et al., *Shortcut learning in medical AI hinders generalization*
  (npj Digital Medicine 2024):
  https://www.nature.com/articles/s41746-024-01118-4 — motivates the
  source-identity and missingness shortcut challenges.
- Ribeiro et al., *CODE-II* (npj Digital Medicine 2026):
  https://www.nature.com/articles/s41746-026-02704-4 — provides a contemporary
  example of patient-exclusive development splits followed by a patient-unique
  independent test benchmark. This project does not yet possess the latter.
