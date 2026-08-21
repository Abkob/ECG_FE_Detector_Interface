# RR/HRV mathematics and implementation

> Version note: the feature equations remain relevant, but the current v0.2
> input is the UNSW-primary RR table with explicit detector-reliability masks.
> See [09_COMPLETE_RR_HRV_ARCHITECTURE.md](09_COMPLETE_RR_HRV_ARCHITECTURE.md)
> for the active pipeline and column schema.

## 1. Scope

`src/ecg_cascade/rr_hrv.py` turns one selected R-peak sequence into continuous
RR/HRV measurements. It implements the 100-RR CSI/modified-CSI/slope feature
construction described by Jeppesen and colleagues. It does not implement an
alarm threshold or output a seizure probability.

The output is one table row per RR interval. The first 99 rows contain basic RR
and heart-rate measurements but cannot contain a complete 100-RR feature
window. Starting at row index 99, a new overlapping 100-RR window is computed
at every new interval.

## 2. From R peaks to RR intervals

Let the selected peak locations relative to the segment be

\[
p_0,p_1,\ldots,p_K
\]

in samples, with sampling rate \(f_s\) samples/second. The interval ending at
peak \(p_i\), for \(i\ge 1\), is

\[
RR_i=1000\frac{p_i-p_{i-1}}{f_s}\quad\text{milliseconds}.
\]

This is the meaning of expressions such as
`1000 * (r_i - r_{i-1}) / fs`: subtract consecutive peak sample numbers,
convert samples to seconds by dividing by the sampling rate, and multiply by
1000 to convert seconds to milliseconds.

The row timestamp is assigned to the ending peak:

\[
t_i=t_{segment\ start}+\frac{p_i}{f_s}.
\]

Instantaneous heart rate is the reciprocal of the RR interval:

\[
HR_i=\frac{60000}{RR_i}\quad\text{beats/minute}.
\]

For example, \(RR=800\) ms gives \(HR=75\) bpm. This is an interval-derived
instantaneous value, not a one-minute beat count.

The code sorts and deduplicates peak indices with `np.unique`. It rejects fewer
than two peaks and any nonpositive RR interval.

## 3. Causal seven-RR median filter

`causal_median_filter(values, width=7)` replaces each RR interval with the
median of the current and up to six previous intervals:

\[
\widetilde{RR}_i=
\operatorname{median}(RR_{\max(0,i-6)},\ldots,RR_i).
\]

It is causal: no future interval is read. This matters for later real-time or
prospective evaluation.

The first six values have a shorter history:

| Index | Values available to median |
|---:|---|
| 0 | `RR[0]` |
| 1 | `RR[0:2]` |
| 2 | `RR[0:3]` |
| … | … |
| 6 and later | Current RR plus six preceding RR intervals |

This cold-start behavior is an implementation decision. It does not affect a
complete 100-RR window after adequate history, but it must be preserved if the
software is later changed to process adjoining chunks.

Filtered instantaneous heart rate is

\[
\widetilde{HR}_i=\frac{60000}{\widetilde{RR}_i}.
\]

No additional ectopic-beat removal, interpolation, physiological RR bounds, or
signal-quality rejection is applied. That omission is intentional for audit:
detector errors remain visible instead of being silently repaired.

## 4. The 100-RR sliding window

For an output row ending at interval \(i\), the full window is

\[
W_i=[RR_{i-99},\ldots,RR_i].
\]

It contains 100 intervals and is updated after every new interval. Adjacent
windows overlap by 99 intervals. A 100-RR window is **beat-count based**, not a
fixed-duration window:

- at 60 bpm it lasts about 100 seconds;
- at 100 bpm it lasts about 60 seconds;
- at 120 bpm it lasts about 50 seconds.

This differs from the QRS-morphology concept using five beats. The 100-RR
window measures slower autonomic dynamics; the five-beat window measures rapid
shape consistency. Different durations are not inherently a defect, but their
timestamps and available-history flags must be aligned explicitly during later
fusion.

The `window_rr_count` column rises from 1 to 100 and then remains 100. It is an
availability indicator, not a quality score.

## 5. Poincare construction

For a 100-value RR window, make 99 successive pairs:

\[
(RR_1,RR_2),(RR_2,RR_3),\ldots,(RR_{99},RR_{100}).
\]

Each pair describes how one interval changes into the next. Rotate the usual
Poincare coordinates by 45 degrees:

\[
u_j=\frac{RR_{j+1}-RR_j}{\sqrt{2}},\qquad
v_j=\frac{RR_{j+1}+RR_j}{\sqrt{2}}.
\]

Then

\[
SD1=\operatorname{sample\ SD}(u_j),\qquad
SD2=\operatorname{sample\ SD}(v_j).
\]

The code uses NumPy `std(..., ddof=1)`, so these are sample standard
deviations. Both are measured in milliseconds.

- `SD1` represents short-term point-to-point RR variability: large alternating
  changes increase the difference coordinate.
- `SD2` represents longer-axis spread: slower displacement of the RR cloud
  increases the sum coordinate.

`poincare_axes` requires at least three finite RR values. A constant RR window
returns both axes as zero. If `SD1` is zero, ratios that divide by `SD1` are left
undefined rather than forced to infinity.

### Raw and filtered axes are separate

The code computes:

- `sd1_raw_ms`, `sd2_raw_ms` from the original 100 RR intervals;
- `sd1_filtered_ms`, `sd2_filtered_ms` from the causal median-filtered
  intervals.

They are not duplicate widgets. The raw axes support `CSI`; the filtered axes
support modified CSI in the published feature construction used here.

## 6. CSI

The cardiac sympathetic index used in this branch is

\[
CSI_{100}=\frac{SD2_{raw}}{SD1_{raw}}.
\]

Both numerator and denominator are milliseconds, so CSI is dimensionless. A
large value means that long-axis spread is large relative to beat-to-beat
spread. It is a geometric ratio, not a direct measurement of sympathetic nerve
activity and not a seizure probability.

The code reports `NaN` if raw `SD1` is non-finite or not greater than zero.

## 7. Modified CSI

Using the median-filtered axes, the implemented modified CSI is

\[
ModCSI_{100}^{(7)}=
\frac{(4SD2_{filtered})^2}{4SD1_{filtered}}.
\]

Algebraically this equals \(4SD2^2/SD1\). Because a squared millisecond value
is divided by milliseconds, modified CSI has units of milliseconds. It is on a
different numerical scale from dimensionless CSI and should not be compared to
CSI by magnitude alone.

The superscript `(7)` describes the seven-RR median filter; it is not a power.
The subscript `100` denotes the 100-RR window.

The code reports `NaN` when filtered `SD1` is non-finite or zero.

## 8. Absolute heart-rate slope

Within the same 100-RR window, the code fits a least-squares straight line to
filtered heart rate as a function of **actual elapsed time**:

\[
\widetilde{HR}_k=a+bt_k+\epsilon_k.
\]

The signed slope is

\[
b=\frac{\sum_k(t_k-\bar t)
(\widetilde{HR}_k-\overline{\widetilde{HR}})}
{\sum_k(t_k-\bar t)^2}.
\]

The exported value is

\[
Slope_{100}=|b|\quad\text{bpm/second}.
\]

Taking the absolute value makes acceleration and deceleration both positive.
Consequently this output cannot by itself say whether heart rate rose or fell.
The code uses seconds, not beat number, because nonuniform RR intervals make
the beat index a distorted time axis.

## 9. Final continuous indices

The two branch outputs modeled after the Jeppesen construction are

\[
J_1(t)=CSI_{100}(t)\,Slope_{100}(t),
\]

\[
J_2(t)=ModCSI_{100}^{(7)}(t)\,Slope_{100}(t).
\]

Their meanings differ:

- `J1` increases when the **relative shape** of the raw Poincare cloud and the
  absolute HR trend are both large. Its units are bpm/second.
- `J2` increases when the **scale-sensitive filtered modified CSI** and the
  absolute HR trend are both large. Its units are ms·bpm/second.

They are engineered continuous indices, not independent physiological sensors.
Both reuse the same RR sequence and the same slope, so they are correlated.
Passing both downstream adds information only to the extent that raw CSI and
filtered modified CSI behave differently. More numbers do not automatically
reduce overfitting; correlated features can increase effective model
complexity and must be evaluated using held-out chronological data.

## 10. `feature_defined` is not a validity verdict

`feature_defined=True` means only that both `J1` and `J2` are finite. It does
not mean:

- the R peaks are correct;
- the window is artifact-free;
- the rhythm is sinus;
- the values are physiologically plausible;
- a seizure occurred;
- the measurements are acceptable for final fusion.

Detector disagreement is reported separately. Later branches should preserve
the distinction between *mathematically computable* and *clinically credible*.

## 11. Complete CSV dictionary

| Column | Unit/type | Exact meaning |
|---|---|---|
| `time_s` | seconds | EDF-relative time of the R peak ending this RR interval. |
| `r_peak_sample_in_file` | sample index | Rounded `time_s * fs`; intended EDF-channel sample coordinate. |
| `rr_ms` | ms | Time from the previous selected R peak to the current selected R peak. |
| `heart_rate_bpm` | bpm | `60000 / rr_ms`. |
| `rr_median7_ms` | ms | Causal median of the current and at most six previous RR intervals. |
| `window_rr_count` | integer | Number of RR intervals currently available, capped at 100. |
| `sd1_raw_ms` | ms | Poincare minor-axis SD from the raw 100-RR window. |
| `sd2_raw_ms` | ms | Poincare major-axis SD from the raw 100-RR window. |
| `csi100` | dimensionless | `sd2_raw_ms / sd1_raw_ms`. |
| `sd1_filtered_ms` | ms | SD1 after the causal seven-RR median filter. |
| `sd2_filtered_ms` | ms | SD2 after the causal seven-RR median filter. |
| `modcsi100_filtered_ms` | ms | `(4*SD2_filtered)^2 / (4*SD1_filtered)`. |
| `slope100_bpm_per_s` | bpm/s | Absolute elapsed-time least-squares slope of filtered HR. |
| `j1_csi_x_slope` | bpm/s | `csi100 * slope100_bpm_per_s`. |
| `j2_modcsi_filtered_x_slope` | ms·bpm/s | `modcsi100_filtered_ms * slope100_bpm_per_s`. |
| `feature_defined` | Boolean | Both final indices are finite. |
| `window_has_detector_disagreement` | Boolean | At least one unmatched candidate from either detector falls inside the supporting peak interval. |
| `unmatched_primary_count` | integer | Unmatched primary candidates in the window's closed peak interval. |
| `unmatched_comparator_count` | integer | Unmatched comparator candidates in the same interval. |

### Coordinate caveat

`r_peak_sample_in_file` is reconstructed from the reported time. If a segment
start is not exactly on the channel's sample grid, it can differ by about one
sample from the precise EDF start-sample plus segment-relative peak. This does
not affect RR differences within a segment, but it should be corrected before
sample-exact cross-file annotation import.

## 12. Row availability example

With 131 detected peaks there are 130 RR intervals and therefore 130 rows:

| Row indices | Basic RR/HR | 100-RR features |
|---|---|---|
| 0–98 | Available | `NaN`; only 1–99 intervals have accumulated. |
| 99 | Available | First complete window, RR rows 0–99. |
| 100 | Available | Second window, RR rows 1–100. |
| … | Available | Window advances by one RR each row. |

This is verified by `test_jeppesen_features_start_only_after_100_rr_intervals`.

## 13. Missing operations and their consequence

| Not implemented | Why that matters |
|---|---|
| Published alarm/calibration threshold | `J1` and `J2` cannot be converted into an alarm or seizure decision. |
| Ectopic-beat identification | Ectopy and detector errors can dominate Poincare features. |
| RR interpolation/resampling | No frequency-domain HRV is currently computed. |
| Minimum signal-quality requirement | Artifact-contaminated values remain present with disagreement metadata. |
| Rhythm stratification | Atrial or ventricular rhythm changes could produce large indices unrelated to seizure. |
| Chunk-to-chunk state | Running separate files/segments resets the filter and 100-RR history. |
| Directional slope | Only the magnitude, not increase versus decrease, is retained. |

Frequency-domain HRV requires an additional, explicitly specified uneven-to-
even sampling procedure and window-length validation. It should not be inferred
from the current code.

## 14. Evidence boundary

Jeppesen's phase-3 paper supports the use of 100 RR intervals, a seven-beat
median-filtered path, Poincare-derived features, and continuous products with
heart-rate slope in their evaluated device system. It does not establish that
this independent Python reimplementation will reproduce their device alarms,
that these values remain valid under the installed EDF artifacts, or that their
published group performance transfers to this single patient.

The phase-3 threshold wording is not sufficiently unambiguous for this package
to claim an exact reproduced calibration rule. The threshold has therefore not
been invented or implemented. That is a scientific guardrail, not an omitted
software convenience.

## 15. Primary literature

- Jeppesen J et al. Phase-3 validation of ECG-derived HRV seizure detection.
  EBioMedicine. 2025. <https://doi.org/10.1016/j.ebiom.2025.105952>
- Pan J, Tompkins WJ. *A Real-Time QRS Detection Algorithm*. 1985.
  <https://doi.org/10.1109/TBME.1985.325532>
- Kristof F et al. R-peak detector assessment across ECG quality. 2024.
  <https://doi.org/10.1371/journal.pdig.0000538>
