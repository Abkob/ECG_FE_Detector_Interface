# All-48-record distance from expert R-peak annotations

## Scope

The current NeuroKit 300-ms strict detector and the current Zhai 2023
reimplementation were rerun on channel 0 of all 48 MIT--BIH Arrhythmia
Database records. The reference contains 109,494 expert beat annotations.

For the primary report, detected and expert events were matched monotonically
and one-to-one within +/-75 ms. For every true-positive pair:

\[
e_i = 1000\frac{\hat r_i-r_i^{expert}}{f_s}.
\]

The reported timing distance is \(|e_i|\). Positive signed error means the
detector timestamp occurred after the expert annotation; negative error means
it occurred before it.

Timing statistics are conditional on matched true positives. An extra
detection and a missed expert beat do not have a validated correspondence and
therefore do not receive a true timing error. FP, FN, sensitivity, PPV, and F1
must be read beside the timing values.

## Pooled results at +/-75 ms

| Measurement | NeuroKit | Zhai reimplementation |
|---|---:|---:|
| Expert annotations | 109,494 | 109,494 |
| Detected events | 110,180 | 108,282 |
| True positives | 107,677 | 107,324 |
| False positives | 2,503 | 958 |
| False negatives | 1,817 | 2,170 |
| Sensitivity | 0.98341 | 0.98018 |
| PPV | 0.97728 | 0.99115 |
| F1 | 0.98033 | 0.98564 |
| Mean absolute timing distance | 3.57 ms | 5.23 ms |
| Median absolute timing distance | 0.00 ms | 2.78 ms |
| 95th-percentile absolute distance | 19.44 ms | 36.11 ms |
| 99th-percentile absolute distance | 61.11 ms | 44.44 ms |
| Mean signed timing error | -1.86 ms | -0.38 ms |
| Signed-error standard deviation | 10.68 ms | 11.92 ms |
| Consecutive matched-RR MAE | 3.31 ms | 3.83 ms |
| Consecutive matched-RR P95 error | 13.89 ms | 36.11 ms |

Zhai has the higher F1 because it produces far fewer false positives. NeuroKit
has the smaller pooled mean, median, and 95th-percentile timing distance among
the beats it successfully matches. Neither result establishes a universal
timestamp owner.

Across the 48 records:

- NeuroKit has the smaller mean absolute matched-beat distance in 34 records;
- Zhai has the smaller mean distance in 14 records;
- NeuroKit has the higher F1 in 27 records;
- Zhai has the higher F1 in 18 records;
- three records tie in F1.

The mean-distance and F1 comparisons answer different questions. Mean timing
distance asks how accurately matched timestamps are localized. F1 asks how
often a usable one-to-one detection exists.

## Five previously difficult records

| Record | NK mean / median / P95 distance | NK FP / FN / F1 | Zhai mean / median / P95 distance | Zhai FP / FN / F1 |
|---|---:|---:|---:|---:|
| 108 | 20.55 / 2.78 / 58.33 ms | 181 / 199 / 0.8917 | 12.82 / 2.78 / 44.44 ms | 11 / 14 / 0.9929 |
| 113 | 0.14 / 0.00 / 0.00 ms | 1,039 / 1 / 0.7753 | 1.32 / 0.00 / 2.78 ms | 1 / 1 / 0.9994 |
| 207 | 54.54 / 66.67 / 69.44 ms | 60 / 532 / 0.8177 | 2.58 / 2.78 / 2.78 ms | 163 / 25 / 0.9513 |
| 222 | 0.54 / 0.00 / 2.78 ms | 539 / 537 / 0.7834 | 1.48 / 2.78 / 2.78 ms | 3 / 73 / 0.9845 |
| 231 | 1.18 / 0.00 / 2.78 ms | 410 / 0 / 0.8846 | 1.00 / 0.00 / 2.78 ms | 0 / 0 / 1.0000 |

Record 113 demonstrates why timing distance alone is insufficient. NeuroKit's
matched true beats are essentially perfectly localized, but it adds 1,039
false T-wave detections. Record 207 shows a different failure: NeuroKit has a
large displacement among its matched events and also misses 532 expert beats.

## Largest record-level 95th-percentile distances

NeuroKit's largest P95 distances occur in records 207 (69.44 ms), 108
(58.33 ms), 200 (50.00 ms), 233 (44.44 ms), and 203 (41.67 ms).

Zhai's largest P95 distances occur in records 119 (75.00 ms), 208 (52.78 ms),
108 (44.44 ms), 233 (41.67 ms), and 102 (38.89 ms).

Because matching is limited to 75 ms, the upper tail is censored: an event
farther away normally becomes an FP/FN rather than appearing as a timing error.

## Reproducible outputs

The complete output directory is
`outputs/neurokit_zhai_all48_true_rpeak_distance`.

| File | Content |
|---|---|
| `record_rpeak_distance_75ms.csv` | One row per record with both detectors' timing and detection metrics. |
| `matched_rpeak_timing_errors_75ms.csv` | Every matched event, expert symbol, signed error, and absolute error. |
| `unmatched_detected_events_75ms.csv` | Every false-positive event plus nearest-expert context explicitly marked as non-correspondence. |
| `missed_expert_beats_75ms.csv` | Every missed expert beat plus nearest-detection context explicitly marked as non-correspondence. |
| `record_rpeak_distance_to_expert_75ms.png` | Median/P95 timing distance and F1 for all records. |
| `worst_record_rpeak_timing_ranking_75ms.csv` | Detector-record ranking by P95 timing distance. |
| `record_symbol_sensitivity.csv` | Sensitivity stratified by expert beat symbol. |
| `pooled_metrics.csv` | Pooled results at 75 ms, 25 ms, and one sample. |

## Decision boundary

These results do not justify globally selecting one detector. The next audit
must stratify the timing and detection errors by beat symbol, signal lead,
polarity, wide-QRS morphology, paced rhythm, ectopy, noise, and local rhythm.
It must then quantify how each failure changes RR intervals and the Varon
morphology eigenvalues.
