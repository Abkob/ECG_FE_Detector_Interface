# Testing and validation manual

> Version note: this chapter contains the original validation framework. The
> active v0.2 MIT-BIH commands, detector roles, and first results are documented
> in [09_COMPLETE_RR_HRV_ARCHITECTURE.md](09_COMPLETE_RR_HRV_ARCHITECTURE.md).

## 1. The four validation levels

This project must keep four questions separate:

| Level | Question | Current status |
|---|---|---|
| Unit correctness | Does an isolated function implement its stated equation or rule? | Six tests pass. |
| Integration correctness | Do the modules read real EDFs and produce internally consistent artifacts? | Demonstrated on selected 256 and 512 Hz segments. |
| Measurement validity | Are detected peaks and derived intervals correct against expert annotations? | Not yet established. |
| Clinical/model validity | Do the measurements predict/detect seizures prospectively at an acceptable false-alarm rate? | Not implemented. |

Passing a lower level does not imply the next level. A perfectly coded equation
can produce invalid physiology when fed incorrect peaks.

## 2. Running the automated suite

From `feature_extraction`:

```powershell
.venv\Scripts\python -m pytest
.venv\Scripts\python -m pip check
```

The current suite contains six tests. `pip check` verifies declared dependency
compatibility, not scientific correctness.

## 3. What each test proves

### `test_peak_agreement_matches_within_75_ms_at_256_hz`

The fixture provides four candidates from each detector. Three lie within the
75 ms sample tolerance and one remains unmatched on each side. It verifies:

- expected one-to-one matched and unmatched arrays;
- agreement score `0.75`;
- both unmatched fractions `0.25`.

It does not test NeuroKit2 detection, noisy ECG, ambiguous competing matches,
or manual R-peak truth.

### `test_empty_peak_sequences_are_explicitly_undefined`

It verifies that two empty sequences return undefined (`NaN`) agreement and
timing instead of a misleading perfect score or zero. JSON serialization later
turns these values into `null`.

It does not test the one-empty/one-nonempty cases or the CLI's behavior when too
few primary peaks prevent RR extraction.

### `test_causal_median_filter_never_uses_future_values`

The short fixture `[1, 100, 2, 3]` with width three has expected output
`[1, 50.5, 2, 3]`. It verifies the expanding cold start and that index 1 does
not see the future value at index 2.

It does not prove that width seven is clinically optimal, that the filter
removes ectopy, or that chunk boundaries preserve history.

### `test_poincare_constant_rr_has_zero_axes`

One hundred identical intervals must have no geometric spread. The test
verifies `SD1` and `SD2` are numerically zero within tolerance.

It does not check a hand-calculated nonconstant case, a reference library, or
the downstream division-by-zero path beyond the extraction test.

### `test_absolute_slope_uses_elapsed_time_not_beat_number`

Heart rate rising exactly 2 bpm each second must yield absolute slope 2 bpm/s.
The test locks the intended independent variable to elapsed time.

It does not test irregular time spacing, descending HR, non-finite input, or
the difference between signed and absolute slope.

### `test_jeppesen_features_start_only_after_100_rr_intervals`

A deterministic variable-RR sequence creates 130 intervals. The test verifies:

- 130 output rows;
- `J1` and `J2` are missing through row 98;
- full features start at row 99;
- finite rows are marked `feature_defined`;
- identical detector sequences produce no disagreement flag.

It does not compare numerical `J1`/`J2` values against the original authors'
implementation, test artifact, or validate a seizure threshold.

## 4. Current automated-test gaps

The next software tests should include:

1. EDF read boundaries, end clipping, physical units, and explicit channel
   ambiguity.
2. One-empty peak comparison, duplicate candidates, equality at the tolerance
   boundary, and competing candidates near one reference beat.
3. Hand-calculated nonconstant SD1/SD2, modified CSI, and both final products.
4. A descending HR sequence proving the slope is absolute.
5. Window disagreement counts exactly at both closed boundaries.
6. CLI integration using a tiny redistributable EDF fixture.
7. Output-schema regression tests for all JSON and CSV fields.
8. Chunked-versus-continuous equivalence after a stateful implementation is
   added.
9. Explicit tests for zero `SD1`, non-finite input, and too-few peaks.

These tests improve software assurance. They still cannot replace clinical
annotation.

## 5. Real-data execution checks already completed

The package successfully produced all output types on selected 600-second
segments at two sampling rates. Results are in `output/feature_tests`.

| Segment | Sampling rate | Main outcome |
|---|---:|---|
| `chb04_28`, 0–600 s | 256 Hz | Both detectors found 812 candidates, but 75 ms agreement was 0.373. |
| `chb04_28`, 1500–2100 s | 256 Hz | Pan found 748, NeuroKit 659; agreement 0.381 amid severe seizure/postictal contamination. |
| `PN06-1`, 0–600 s | 512 Hz | Pan found 741, NeuroKit 740; agreement 0.115 due largely to localization difference. |

Successful execution proves compatibility with those files. It does not make
the resulting RR/HRV values acceptable biomarkers.

## 6. Manual R-peak validation study

### 6.1 Fixed purpose

The purpose is to determine which candidate detector, if either, produces RR
timing accurate enough for this patient's HRV branch. It is not to maximize
agreement between algorithms.

### 6.2 Sampling strata

Build an annotation set before viewing final detector-level scores. It should
contain intervals sampled from:

- clean interictal periods from every acquisition system/lead configuration;
- representative sleep/wake or activity states if these exist in the data;
- preictal intervals at predefined offsets;
- early ictal ECG before gross contamination;
- contaminated ictal and postictal intervals;
- morphology or polarity changes;
- rhythm changes and ectopic beats;
- detector-agreement intervals;
- Pan-only and NeuroKit-only candidates;
- high-timing-displacement matches.

Sampling only obvious disagreements would overestimate the normal error rate;
sampling only quiet ECG would underestimate failure during the target state.
The final report should preserve strata rather than pool everything into one
score.

### 6.3 Annotation procedure

1. Freeze detector code, software versions, and matching tolerance.
2. Export ECG windows with adequate context and a stable time coordinate.
3. Have a qualified reviewer mark one fiducial convention consistently, such
   as the dominant R deflection within each QRS complex.
4. Mark intervals as unscorable when the ECG cannot support a defensible QRS
   annotation; do not force a reference peak through artifact.
5. If possible, use a second reviewer for an overlap subset and report
   reviewer disagreement.
6. Lock the reference set before computing detector performance.

The exact fiducial convention matters because the current detectors sometimes
identify different points within the same complex. HRV needs temporal
consistency even when both locations appear visually QRS-related.

### 6.4 Metrics against manual truth

Using the fixed 75 ms same-QRS tolerance:

\[
Sensitivity=\frac{TP}{TP+FN},
\]

\[
PPV=\frac{TP}{TP+FP},
\]

\[
F1=\frac{2TP}{2TP+FP+FN}.
\]

Also report candidate timing error in milliseconds using median, interquartile
range, 95th percentile, and a distribution plot. Report every metric by signal
stratum and recording source, with counts and confidence intervals where
appropriate.

Do not use detector-versus-detector `agreement_f1` as any of these metrics.

### 6.5 Detector decision rule

Select a primary detector only after defining acceptable performance for the
use case. The benchmark literature can supply comparison context, but it cannot
choose a safety threshold for this patient's peri-ictal data. If neither
detector is acceptable in a stratum, preserve an unavailable/low-validity
state; do not average two wrong peak sequences.

## 7. RR/HRV numerical validation

Once R peaks are accepted, validate the feature equations independently:

1. Export a small set of fixed RR windows.
2. Calculate RR, HR, filtered RR, SD1, SD2, CSI, modified CSI, slope, `J1`, and
   `J2` in a separate reference notebook or vetted implementation.
3. Compare every intermediate value with explicit absolute and relative
   tolerances.
4. Include regular rhythm, gradual tachycardia, abrupt outlier, ectopy, and
   constant-RR edge cases.
5. Compare uninterrupted processing with any future chunked implementation.

This stage validates reimplementation fidelity. It does not test seizure
discrimination.

## 8. Single-patient chronological model validation

The available target design has many hours from one patient. That supports
patient-specific development, but the effective independent sample size is not
the number of overlapping windows. Adjacent 100-RR rows share 99% of their RR
content and are strongly dependent.

### Required split structure

- Split by whole chronological recording blocks or seizure episodes, never by
  randomly shuffled feature rows.
- Keep every overlapping window from one episode/file on only one side of a
  split.
- Fit normalization, feature selection, threshold calibration, and model
  hyperparameters using training data only.
- Preserve a final later-time test set untouched until choices are frozen.
- If evaluating seizure prediction, exclude the declared preictal/ictal/post-
  ictal periods from interictal baseline according to a predeclared protocol.

Random row splitting would place almost identical 100-RR windows in training
and testing and can severely inflate performance.

### Evaluation unit

For seizure work, window accuracy alone is insufficient. Report at least:

- event sensitivity: seizures with a qualifying warning/detection divided by
  evaluable seizures;
- false alarms per 24 hours of eligible monitoring;
- time in warning, if prediction is the goal;
- detection or prediction latency distribution;
- proportion of time/episodes unavailable because ECG was unscorable;
- performance separately by artifact/quality state.

The number of seizures, not the number of overlapping rows, primarily limits
event-level confidence.

## 9. Leakage audit for every later model

Before accepting a result, verify:

| Leakage route | Audit question |
|---|---|
| Overlapping windows | Can two rows sharing RR intervals appear in different splits? |
| File identity | Can file-specific acquisition signatures reveal the label? |
| Event-adjacent normalization | Were future or seizure-containing values used to normalize an earlier prediction? |
| Feature selection | Were held-out episodes used to choose the retained features? |
| Threshold search | Was the final test set used to select an alarm threshold? |
| Artifact correlation | Is the model detecting recording disruption that happens to co-occur with seizure annotation? |
| Duplicate seizures | Can one prolonged event appear as multiple training/test examples? |

## 10. Branch acceptance gates

| Gate | Required evidence | Current state |
|---|---|---|
| G1: I/O | Correct channel, units, samples, timestamps. | Partly software-verified; sample-grid caveat remains. |
| G2: Peak measurement | Manual-annotation sensitivity, PPV, F1, timing by stratum. | Not done. |
| G3: RR/HRV fidelity | Independent numerical comparison of all intermediate values. | Partly unit-tested; full reference comparison not done. |
| G4: Signal validity | Artifact/rhythm context and unavailable-state rules. | Not implemented. |
| G5: Patient-specific discrimination | Chronological, episode-held-out performance. | Not implemented. |
| G6: Prospective lock | Frozen preprocessing/model/threshold on later data. | Not implemented. |

No later gate should retroactively be described as passed because a model AUC
or F1 looks high.

## 11. Interpretation of negative results

A failed detector or feature is informative. It can indicate:

- insufficient ECG quality for the measurement;
- fiducial inconsistency despite correct beat counts;
- a feature dominated by ectopy or rhythm rather than autonomic change;
- insufficient seizure events for stable patient-specific estimation;
- acquisition-domain shift;
- a true absence of the proposed cardiac seizure signature.

The response should be to locate the failing gate and compare a published
alternative under the same locked evaluation—not to add ad hoc smoothing or a
new threshold after seeing the test labels.
