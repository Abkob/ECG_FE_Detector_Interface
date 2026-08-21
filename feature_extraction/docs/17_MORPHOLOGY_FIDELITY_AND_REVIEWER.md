# Morphology fidelity audit and multi-branch ECG reviewer

## What was implemented

The morphology audit compares the published five-beat Varon 2015 measurements
calculated from three automatic timestamp tracks with the same measurements
calculated from expert MIT--BIH beat timestamps:

```text
expert .atr timestamps -----------------> expert five-beat reference

UNSW timestamps -----> five-beat branch --+
NeuroKit timestamps -> five-beat branch --+--> feature-fidelity audit
Zhai timestamps -----> five-beat branch --+
```

This is not seizure classification. MIT--BIH supplies expert beat events and
beat symbols but no seizure onset labels.

The implementation is in:

- `src/ecg_cascade/morphology.py`;
- `src/ecg_cascade/morphology_validation.py`;
- `scripts/benchmark_morphology_fidelity.py`;
- `src/ecg_cascade/rpeak_viewer.py`.

## Strict comparability rule

A candidate five-beat window is compared with an expert five-beat window only
when:

1. all five expert beats have one-to-one candidate matches within 75 ms; and
2. the five matched candidates are consecutive in their detector's stream.

This matters because the branch slides by one beat. A false extra event or a
miss changes the membership of several overlapping windows. Comparing only
the nearest five detections would conceal that failure.

## Metrics

Beat detection uses

\[
\mathrm{Sensitivity}=\frac{TP}{TP+FN},\qquad
\mathrm{PPV}=\frac{TP}{TP+FP},
\]

and

\[
F_1=\frac{2TP}{2TP+FP+FN}.
\]

For comparable five-beat windows, the audit reports:

- raw eigenvalue relative L1 error;
- exploratory normalized eigen-spectrum L1 error;
- Lin concordance correlation coefficient for each raw eigenvalue;
- symmetric mean absolute percentage error for each raw eigenvalue;
- strict five-beat window coverage.

Raw eigenvalues are the paper-backed output. Normalized-spectrum error is an
exploratory diagnostic that reduces overall gain sensitivity; it is not a
replacement feature.

## Full 48-record MIT--BIH result

All records used channel 0, their native 360-Hz sampling rate, expert `.atr`
beat annotations, and a 75-ms one-to-one event-matching tolerance.

| Timestamp track | Beat sensitivity | Beat PPV | Beat F1 | Median absolute timing error | Strict five-beat coverage | Median raw eigenvalue relative L1 error | Median exploratory normalized-spectrum L1 error |
|---|---:|---:|---:|---:|---:|---:|---:|
| UNSW | 0.99772 | 0.99681 | 0.99726 | 5.56 ms | 0.98708 | 0.02277 | 0.01131 |
| NeuroKit | 0.98341 | 0.97728 | 0.98033 | 0.00 ms | 0.93424 | 0.00707 | 0.00516 |
| Zhai reimplementation | 0.98018 | 0.99115 | 0.98564 | 2.78 ms | 0.92258 | 0.00825 | 0.00515 |

![All-48 morphology fidelity summary](../outputs/morphology_fidelity_all48_v1/morphology_fidelity_summary.png)

Interpretation:

- UNSW supplies the most continuous five-beat sequence, but its timestamps
  cause larger morphology-feature changes relative to expert alignment.
- NeuroKit is very accurately aligned for the beats it matches, but its
  misses and extras reduce usable five-beat coverage.
- Zhai has similarly low conditional feature error, higher PPV than NeuroKit,
  and lower coverage than both alternatives.
- The result is a coverage-versus-fidelity tradeoff, not evidence that three
  detectors should be averaged or majority-voted.

The median NeuroKit timing error of 0 ms does not mean perfect detection. It
means that at least half of its *matched* candidates fall on the expert sample;
its 2,503 false positives and 1,817 misses still affect branch continuity.

## Counterexamples hidden by pooled results

- Record 113: NeuroKit beat F1 is 0.775 and strict five-beat coverage is only
  0.107 because of repeated extra detections, even though the median timing
  error of matched beats is 0 ms.
- Record 207: NeuroKit median timing error reaches 66.7 ms and five-beat
  coverage is 0.372; this is the morphology-changing, polarity-difficult case.
- Record 203: Zhai five-beat coverage is 0.386 and UNSW coverage is 0.860.
- Record 102: UNSW detects every expert beat, yet the median normalized
  eigen-spectrum error is 0.540. Correct event existence alone therefore does
  not guarantee faithful waveform alignment.
- Records 208 and 233 are high-error Zhai counterexamples despite acceptable
  pooled performance.

These cases are why the UI exposes all marker tracks and expert-aligned
morphology rather than reporting only one average score.

## Reviewer behavior

Launch with `Launch_ECG_Cascade_Reviewer.bat` (the previous R-peak launcher is
kept as an alias) or:

```powershell
.venv\Scripts\ecg-rpeak-viewer "C:\path\to\record.hea"
```

The reviewer supports:

- recursive dataset discovery and a record selector;
- individual EDF, WFDB `.hea`, or `.dat` selection;
- visible NeuroKit, UNSW, Zhai, and expert marker toggles;
- an active track controlling RR and TP/FP/FN evaluation;
- expert-label filtering/highlighting;
- five-beat morphology eigenvalue-energy traces;
- the expert-timestamp morphology trace when `.atr` annotations exist;
- marker hover showing time, sample, expert symbol, detector offsets, and the
  nearest lambda vector;
- right-drag region selection summarizing expert symbols and detector counts;
- `Z` to zoom in, `X` to zoom out, and arrow keys to move through the record.

![Record 207 in the multi-branch reviewer](../outputs/ui_validation/record_207_branch_reviewer_full.png)

When an expert annotation exists, the ECG panel distinguishes true-positive,
false-positive, and missed beats for the active detector. The morphology panel
does not show "predicted seizure versus true seizure" because neither output
exists in this dataset or branch.

## Saved evidence

The all-record output directory contains:

| File | Meaning |
|---|---|
| `record_summary.csv` | Per-record detector and morphology-fidelity metrics. |
| `window_fidelity.csv` | Every five-beat candidate/reference comparison and availability reason. |
| `matched_beat_timing.csv` | Every one-to-one candidate-minus-expert timing error. |
| `pooled_summary.csv` | Pooled detector and feature-fidelity summary. |
| `morphology_fidelity_summary.png` | Four-panel visual comparison. |
| `audit_scope.json` | Dataset, tolerance, method, and explicitly unmeasured claims. |

## Decision and next gate

Keep all three tracks visible during development. Use expert timestamps as the
reference lane, not as an input available during deployment. Do not substitute
or average timestamps merely to improve one pooled metric.

Before seizure modeling, manually annotate representative patient ECG across
clean baseline, movement, preictal, ictal, postictal, tachycardic, bradycardic,
and morphology-changing intervals. Then repeat both event and feature-fidelity
audits on that patient. Only after that should one timestamp policy be frozen
for the RR/HRV and morphology branches.
