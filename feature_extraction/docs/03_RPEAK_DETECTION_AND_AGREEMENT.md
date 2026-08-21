# R-peak detection and detector agreement

> Version note: this chapter documents the original v0.1 Pan-Tompkins/
> NeuroKit comparison. The implemented v0.2 UNSW-primary architecture and its
> current 50-ms RR-support rule are documented in
> [09_COMPLETE_RR_HRV_ARCHITECTURE.md](09_COMPLETE_RR_HRV_ARCHITECTURE.md).

## 1. Purpose of this stage

The R-peak stage converts a one-dimensional ECG segment into candidate beat
locations. Those locations are the time base for every quantity in the current
RR/HRV branch. If a peak is missed, added, or assigned to the wrong waveform,
the error propagates into RR intervals, heart rate, Poincare geometry, CSI,
modified CSI, slope, `J1`, and `J2`.

This implementation therefore runs two published detector pipelines and records
their agreement before computing RR/HRV. Detector agreement is a **development
warning**, not ground truth. Two detectors can agree and both be wrong; they can
also identify the same QRS complex at different fiducial positions.

Implementation: `src/ecg_cascade/peaks.py`.

## 2. Returned objects

### `PeakDetection`

| Field | Type | Meaning |
|---|---|---|
| `method` | string | Exact NeuroKit2 method name used for both cleaning and detection. |
| `peak_samples` | integer array | R-peak candidate indices relative to the loaded segment. |
| `cleaned_ecg` | float array | The method-specific cleaned signal passed to the detector. |

The peak indices are **not EDF-global sample numbers**. A segment peak at index
`p` occurs at

\[
t_{EDF}=t_{start}+\frac{p}{f_s},
\]

where `t_start` is the requested segment start in seconds and `f_s` is the EDF
channel sampling rate.

### `PeakAgreement`

| Field | Meaning |
|---|---|
| `matched_primary_samples` | Primary-detector samples paired with comparator samples. |
| `matched_comparator_samples` | Comparator samples paired with primary samples. |
| `unmatched_primary_samples` | Primary candidates with no comparator candidate inside the tolerance. |
| `unmatched_comparator_samples` | Comparator candidates with no primary candidate inside the tolerance. |
| `agreement_f1` | Symmetric overlap score between the two candidate sets. |
| `median_timing_difference_ms` | Median absolute displacement among matched pairs only. |
| `unmatched_primary_fraction` | Unmatched primary candidates divided by all primary candidates. |
| `unmatched_comparator_fraction` | Unmatched comparator candidates divided by all comparator candidates. |

The word *primary* means the detector selected to create RR intervals. It does
not mean that the detector has been proven correct.

## 3. `detect_r_peaks`: exact processing sequence

For each method, the code performs:

1. Convert the ECG to a one-dimensional `float64` NumPy array.
2. Reject a non-one-dimensional signal, a segment shorter than three seconds,
   or any non-finite sample.
3. Call `neurokit2.ecg_clean(signal, sampling_rate=fs, method=method)`.
4. Pass that method-specific cleaned signal to
   `neurokit2.ecg_peaks(..., method=method, correct_artifacts=False)`.
5. Read `info["ECG_R_Peaks"]` and store the indices as signed 64-bit integers.

The two methods used by `pipeline.py` are:

| Pipeline name | Cleaner | Detector | Role in this package |
|---|---|---|---|
| `pantompkins1985` | NeuroKit2's Pan-Tompkins-specific cleaning path | NeuroKit2's Pan-Tompkins implementation | Default source of RR intervals. |
| `neurokit` | NeuroKit2's default ECG cleaning path | NeuroKit2's default gradient-based detector | Comparator, or optional primary via the CLI. |

This mirrors the detector construction used for the corresponding NeuroKit2
entries in Kristof et al.'s 2024 open detector benchmark. It does **not** prove
that this Python Pan-Tompkins implementation is numerically identical to the
original 1985 code or to the Android implementation in the Jeppesen device.

`correct_artifacts=False` is deliberate. NeuroKit2's optional post-processing
could insert, remove, or shift peaks. Leaving it off makes the two raw detector
outputs auditable and avoids silently converting disagreement into agreement.
It also means that the current peak arrays receive no ectopic-beat correction or
post-detection physiological editing.

## 4. The 75 ms pairing rule

Kristof et al. treated detections within 75 ms of a reference annotation as
belonging to the same QRS complex. They selected this tolerance to reduce the
chance that a P or T wave is counted as the correct QRS. The current code uses
that published tolerance to compare the two detectors:

\[
\tau_{samples}=\operatorname{round}\left(0.075 f_s\right).
\]

At 256 Hz, this is 19 samples after rounding. At 512 Hz, it is 38 samples.

The code does not search every possible assignment. It performs a monotone,
one-to-one walk through two sorted unique sequences:

```text
i = first primary peak; j = first comparator peak
while both sequences have remaining peaks:
    if abs(primary[i] - comparator[j]) <= tolerance:
        pair them and advance both
    else if primary[i] is earlier:
        mark primary[i] unmatched and advance primary
    else:
        mark comparator[j] unmatched and advance comparator
append all remaining candidates as unmatched
```

This is appropriate for chronologically ordered beat candidates when a local
match is interpreted as the same QRS. The use of `np.unique` means duplicate
sample indices from a detector are collapsed before comparison.

### Boundary behavior

`<= tolerance` is accepted. Therefore a displacement exactly equal to the
rounded sample tolerance is a match. If the requested tolerance corresponds to
less than one sample, the function raises an error rather than pretending that
sub-sample matching was performed.

## 5. Agreement equations

Let:

- \(N_P\) be the number of primary candidates;
- \(N_C\) be the number of comparator candidates;
- \(M\) be the number of one-to-one matched pairs.

The implemented agreement score is

\[
A=\frac{2M}{N_P+N_C}.
\]

This has the same algebraic form as an F1/Dice score. It is called
`agreement_f1` in the output, but neither detector is labeled as truth. It is
therefore **not clinical sensitivity**, **not positive predictive value**, and
**not a seizure-classification F1 score**.

For matched samples \(p_k\) and \(c_k\), the timing statistic is

\[
d_{50}=\operatorname{median}_k
\left(\frac{1000|p_k-c_k|}{f_s}\right)\text{ ms}.
\]

The unmatched fractions are

\[
u_P=\frac{N_P-M}{N_P},\qquad
u_C=\frac{N_C-M}{N_C}.
\]

If both sequences are empty, the agreement and timing values are undefined and
serialized as JSON `null`. If no pair matches, timing is also undefined.

### A crucial interpretation trap

`median_timing_difference_ms` is conditional on a match. A small median can
coexist with very poor overall agreement because all nonmatching candidates are
excluded from the median. Always read it together with `agreement_f1` and both
unmatched fractions.

## 6. How disagreement is attached to RR/HRV windows

For each completed 100-RR window, `rr_hrv.py` counts unmatched candidates inside
the closed interval from the first peak supporting that window to its ending
peak. It stores:

- `unmatched_primary_count`;
- `unmatched_comparator_count`;
- `window_has_detector_disagreement`.

These values are quality/context measurements. The code does **not** discard the
window and does not replace its physiological measurements. This preserves the
project rule that quality is context rather than an automatic veto. It also
means downstream code must not ignore these fields.

## 7. Real-data observations from the installed EDFs

The initial audit used 600-second blocks:

| Recording/block | Pan candidates | NeuroKit candidates | Agreement at 75 ms | Interpretation |
|---|---:|---:|---:|---|
| `chb04_28`, 0–600 s | 812 | 812 | 0.373 | Counts match, but fiducial placement often differs by more than 75 ms. |
| `chb04_28`, 1500–2100 s | 748 | 659 | 0.381 | The seizure-containing block has both placement disagreement and marked contamination. |
| `PN06-1`, 0–600 s | 741 | 740 | 0.115 | Similar counts conceal major localization differences. |

The tolerance audit was:

| Block | 75 ms | 100 ms | 125 ms | 150 ms |
|---|---:|---:|---:|---:|
| `chb04_28`, quiet block | 0.373 | 0.983 | 0.990 | 0.990 |
| `chb04_28`, seizure-containing block | 0.381 | 0.706 | 0.733 | 0.741 |
| `PN06-1`, quiet block | 0.115 | 0.506 | 0.994 | 0.999 |

The quiet-block jump at a wider tolerance is compatible with systematic
within-complex localization differences. The seizure-block result contains
additional missed/extra candidates. This is an inference from detector
behavior and visual inspection, not a manual-annotation result.

Increasing the official tolerance would hide the documented problem and would
not repair RR timing. A 120 ms displacement can still materially change two
adjacent RR intervals even when the two detections refer to the same beat.

## 8. What this stage establishes—and what it does not

### Established in software

- Both named NeuroKit2 pipelines execute on the installed 256 and 512 Hz EDFs.
- The returned indices are valid locations in the loaded segment in the tested
  runs.
- Matching is sorted, unique, one-to-one, and tested at a 75 ms tolerance.
- Agreement failures are carried into each 100-RR feature window.

### Not established

- Which detector is correct at a disputed beat.
- Sensitivity or positive predictive value against expert R-peak annotations.
- Robustness during the patient's preictal, ictal, or postictal contamination.
- Suitability of either sequence for HRV.
- Generalization to another lead, recording system, or patient.
- Equivalence between a NeuroKit2 implementation and any similarly named
  implementation in a clinical paper.

## 9. Required acceptance experiment

Create manual R-peak annotations for representative intervals covering clean
interictal ECG, preictal/early-ictal ECG, contaminated ictal/postictal ECG,
polarity or morphology changes, and intervals where the two detectors agree
and disagree. Against the fixed annotations, report for each detector:

\[
\text{sensitivity}=\frac{TP}{TP+FN},\qquad
\text{PPV}=\frac{TP}{TP+FP},\qquad
F1=\frac{2TP}{2TP+FP+FN},
\]

plus the full timing-error distribution, not only its median. Do not select a
tolerance or detector on the same samples used for the final report.

Only this experiment can promote one candidate sequence from *configured
primary* to *validated primary for this patient's data*.

## 10. Primary literature

- Pan J, Tompkins WJ. *A Real-Time QRS Detection Algorithm*. IEEE Transactions
  on Biomedical Engineering. 1985. <https://doi.org/10.1109/TBME.1985.325532>
- Makowski D et al. *NeuroKit2: A Python toolbox for neurophysiological signal
  processing*. Behavior Research Methods. 2021.
  <https://doi.org/10.3758/s13428-020-01516-y>
- Kristof F et al. *Assessment of algorithmic methods to detect the R-peak in
  ECGs with varying quality*. PLOS Digital Health. 2024.
  <https://doi.org/10.1371/journal.pdig.0000538>
