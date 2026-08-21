# PRSA/BPRSA context-matrix validation

Date: 2026-08-19

## Bottom line

The deterministic PRSA/BPRSA implementation and its causal attachment to the
matrix are validated as software. PRSA is robust to ECG amplitude corruption
when R timestamps are fixed. BPRSA is not: its R-amplitude driver changes with
lead, polarity, and electrode-motion noise. Neither lane is validated as a
seizure detector.

The current decision is therefore:

- keep the six values as autonomic/cardiorespiratory **context**;
- retain `defined`, `reliable`, numerical-quality, lead, polarity, and age
  metadata;
- do not let PRSA/BPRSA change the core morphology--RR/HRV matrix gate;
- interpret `computationally_usable` only as mathematical plus
  detector/numerical readiness; the matrix explicitly exports
  `signal_quality_attached = false` and `model_eligible = false` at this stage;
- do not make BPRSA model-eligible without the separate signal-quality context
  and a stable lead/polarity state;
- retain arrhythmia/beat-type context because genuine ectopy can change PRSA
  without being artifact or seizure-specific;
- make no seizure-accuracy claim from the present datasets.

## What “accuracy” means here

There is no manual clinical annotation called “the true PRSA value.” Validation
must therefore be separated into four endpoints:

1. **Equation accuracy:** do the program's anchors, phase averages, SDNN, and
   slopes equal an independent direct calculation of the published equations?
2. **Matrix accuracy:** is the latest permissible 80-beat row attached without
   future leakage, value changes, or modification of the core gates?
3. **Measurement fidelity:** do the values remain stable under lead, polarity,
   artifact, arrhythmia, and R-peak perturbations?
4. **Seizure usefulness:** do the values improve event-level seizure detection
   on unseen patients and long continuous recordings at an acceptable false
   alarm rate?

Only endpoints 1--3 have partial or complete evidence. Endpoint 4 remains
unevaluable with the current reliable seizure windows.

## Audits run

### 1. Equations, warm-up, causality, and input handling

The unit tests independently construct decreasing-driver anchors and phase
averages, then compare the complete curves and six scalar outputs. They also
test:

- population SDNN with denominator 80;
- the local and long-range slope denominators;
- 79 explicit warm-up rows;
- unchanged earlier outputs after future beats are appended;
- PRSA invariance and BPRSA sensitivity under global polarity inversion;
- mathematical definition separated from reliability;
- exact ECG R-amplitude sampling;
- invalid input rejection.

The full repository suite passed: **103 tests passed**.

### 2. Causal matrix audit

The fresh all-record expert-anchor table was probed at every exact context time,
every between-context midpoint, and once before the first eligible window.

| Quantity | Result |
|---|---:|
| MIT-BIH records | 48 |
| Source PRSA/BPRSA rows | 105,654 |
| Matrix probe rows | 211,308 |
| Attached-time mismatches | 0 |
| Six-value mismatches | 0 |
| Future leaks | 0 |
| Negative context ages | 0 |
| Availability/usable-flag mismatches | 0 / 0 |
| Core definition-gate changes | 0 |
| Core context-gate changes | 0 |
| Context rows marked as core | 0 |
| Context rows marked signal-quality-attached/model-eligible | 0 / 0 |

This validates the backward as-of matrix join. It does not validate seizure
discrimination.

### 3. All-48 MIT-BIH expert-anchor rerun

Expert `atr` beat timestamps bypassed automatic detector error so feature
arithmetic, availability, numerical behavior, and lead sensitivity could be
isolated.

| Quantity | Fresh result |
|---|---:|
| Expert beats | 109,494 |
| Eligible 80-beat windows | 105,654 |
| Mathematically defined | 105,654 (100.000%) |
| Positive-resampled-RR gate | 104,708 (99.1046%) |
| Windows with nonpositive cubic RR | 946 (0.8954%) |
| Median / P05 PRSA anchors | 100 / 67 |
| Median / P05 BPRSA anchors | 102 / 67 |

The fresh feature, record-summary, lead/polarity, and pooled-summary files were
exactly equal to the earlier audit after parsing. This establishes deterministic
reproducibility on the installed environment.

MIT-BIH Arrhythmia has no seizure labels; these percentages are not sensitivity,
PPV, specificity, or false-alarm rates.

### 4. Lead and polarity

Using identical expert RR timestamps on records 108, 113, 207, 222, and 231:

- global polarity inversion: median PRSA correlation 1.0000; median BPRSA
  correlation -0.8638;
- different lead: median PRSA correlation 1.0000; median BPRSA correlation
  0.0235.

PRSA depends only on RR and is therefore amplitude-sign invariant. BPRSA uses
R amplitude to select respiratory-phase anchors, so its values are not
universal across leads or polarity states.

### 5. Controlled electrode-motion noise

Clean MIT-BIH records 118/119 were compared with their calibrated MIT-BIH Noise
Stress Test Database copies. The same expert R timestamps were supplied to
clean and noisy signals, so detector error was removed and only waveform-noise
effects remained.

RR-only PRSA exact-match fraction was 1.000000 and its maximum median absolute
error was zero. BPRSA degraded as follows (median across the two records and
two BPRSA slopes):

| Electrode-motion SNR | BPRSA correlation | BPRSA sign agreement |
|---:|---:|---:|
| +24 dB | 0.8100 | 0.8836 |
| +18 dB | 0.7672 | 0.7999 |
| +12 dB | 0.6714 | 0.7383 |
| +6 dB | 0.3903 | 0.7214 |
| 0 dB | 0.1343 | 0.6380 |
| -6 dB | 0.0315 | 0.5678 |

The current numerical-quality gate checks whether cubic-resampled RR remains
positive. With fixed R timestamps, it is identical for clean and noisy ECG and
cannot detect amplitude corruption. BPRSA therefore needs the independent
signal-quality context; `feature_reliable` alone must not be interpreted as
“artifact-free.”

### 6. Arrhythmia confounding

Each expert-anchor feature row was aligned back to its 81 annotated beats (80
RR intervals).

| Quantity | Result |
|---|---:|
| Expert-anchor windows | 105,654 |
| Windows containing at least one non-`N` beat | 77,696 (73.54%) |
| Cubic-RR numerical failures | 946 |
| Failures containing a non-`N` beat | 946 (100.00%) |
| Numerical pass, all-`N` windows | 100.000% |
| Numerical pass, non-normal windows | 98.782% |

Arrhythmia is not automatically artifact. It can be real seizure-associated or
non-seizure physiology. The beat type/rhythm state must therefore be retained
as confounding context and used for stratified evaluation rather than silently
discarded or treated as proof of seizure.

### 7. CHB04_28 seizure-interval rerun

The current code was rerun on one patient with two labeled seizure intervals.
The new CSV files were exactly equal to the earlier audit after parsing.

| Event | Strongest unfiltered descriptive feature | Direction-free overlapping-window AUC | Strict reliable ictal windows |
|---:|---|---:|---:|
| 1 | `sdnn80_ms` | 0.991 | 0 / 118 |
| 2 | `mean_rr80_ms` | 0.886 | 0 / 146 |

These AUCs cannot be interpreted as seizure performance. They use highly
overlapping windows from one patient, and no ictal window passed the strict
cross-detector support gate. The correct result is **clinically unevaluable**,
not positive and not negative.

## Required seizure validation

Before these context columns are fed to a seizure model:

1. Use long, continuous, same-lead ECG with video-EEG seizure onset/offset and
   sufficient non-seizure recording for false-alarm measurement.
2. Manually audit or correct ECG R peaks around baseline, preictal, ictal,
   postictal, arrhythmia, and artifact periods.
3. Attach the separate signal-quality measurements causally. Prespecify which
   quality fields are covariates and which conditions make BPRSA unevaluable;
   do not learn this rule on the test patients.
4. Preserve beat-type/rhythm context and report clean sinus, arrhythmia,
   clean-seizure, and seizure-plus-artifact strata separately.
5. Use patient-disjoint chronological evaluation. Purge at least the shared
   80-beat history at train/test and phase boundaries; ordinary random row
   splitting is invalid because adjacent windows overlap by 79 beats.
6. Compare prespecified ablations: core RR/HRV+morphology; core+PRSA;
   core+BPRSA; core+PRSA+BPRSA; and the same comparisons after quality/rhythm
   stratification.
7. Report event sensitivity, false alarms/hour, PPV, detection latency,
   time-in-warning, analyzable coverage, and failure reasons. Window AUROC alone
   is insufficient.
8. Repeat on more than one dataset/device/lead configuration and freeze all
   extraction and quality rules before the held-out test.

## Reproducible outputs

- `outputs/prsa_bprsa_context_audit_20260819/`: fresh all-48 MIT-BIH rerun.
- `outputs/prsa_bprsa_context_matrix_audit_20260819/`: causal matrix audit.
- `outputs/prsa_bprsa_noise_fidelity_20260819/`: controlled electrode-motion
  noise fidelity.
- `outputs/prsa_bprsa_arrhythmia_context_20260819/`: expert beat-type confound
  audit.
- `outputs/prsa_bprsa_chb04_28_rerun_20260819/`: fresh two-seizure reliability
  rerun.

Reproduction scripts:

- `scripts/benchmark_prsa_bprsa_available_data.py`
- `scripts/audit_prsa_bprsa_context_matrix.py`
- `scripts/benchmark_prsa_bprsa_noise_fidelity.py`
- `scripts/audit_prsa_bprsa_arrhythmia_context.py`
