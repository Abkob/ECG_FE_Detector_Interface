# NeuroKit--Zhai two-track RR--HRV architecture

## Status and scientific boundary

This is the new experimental architecture. It runs two complete signal-
processing detectors on the same single-lead ECG and calculates an independent
RR/HRV series from each. It does **not** average their timestamps, take a
majority vote, splice their beats, or select a winner without expert
annotations.

The earlier UNSW-primary architecture remains in the package as a documented
legacy/control implementation. Its positive-amplitude fiducial refinement is
not used anywhere in this new two-track path.

```text
raw ECG
  |-- NeuroKit cleaning + gradient detector
  |      `-- unmodified NeuroKit timestamps -> NeuroKit RR -> NeuroKit HRV
  |
  `-- full Zhai 2023 paper reimplementation
         |-- 5--35 Hz zero-phase band-pass
         |-- square + 5 Hz zero-phase low-pass
         |-- dynamic QRS windows
         |-- automatic 120-ms template
         `-- maximum absolute normalized cross-correlation
                `-- Zhai timestamps -> Zhai RR -> Zhai HRV

NeuroKit <-> Zhai temporal matching
  `-- support/context only; no hard veto and no timestamp fusion
```

## Code map

| File | Responsibility |
|---|---|
| `src/ecg_cascade/zhai.py` | Literal Zhai preprocessing, windowing, template selection, cross-correlation localization, and exposed intermediate signals. |
| `src/ecg_cascade/neurokit_zhai.py` | Runs both independent tracks, constructs non-destructive support tables, calculates both RR/HRV outputs, and saves artifacts. |
| `src/ecg_cascade/config.py` | `NeuroKitZhaiConfig`; explicitly records that the timestamp owner is unselected. |
| `src/ecg_cascade/plotting.py` | Four-panel detector, QRS-envelope, correlation, and separate-RR diagnostic. |
| `scripts/benchmark_neurokit_zhai.py` | Reproducible annotation audit at 75 ms, 25 ms, and one sample. |
| `tests/test_zhai.py` | Synthetic localization, whole-record inversion, and no-fusion/no-refinement contracts. |

## NeuroKit track

The NeuroKit track uses NeuroKit2 0.2.13:

1. `ecg_clean(method="neurokit")`.
2. Absolute ECG gradient to identify steep QRS regions.
3. A 100-ms smoothed-gradient window.
4. A 750-ms local threshold context.
5. Positive-peak prominence within each accepted region.
6. The unmodified strict separation rule of more than 300 ms.

The timestamps returned by this detector are used directly. The code does not
apply the old local `argmax(ECG)` refinement afterward. NeuroKit's own internal
positive-peak selection remains a known polarity/morphology limitation; it is
not concealed by the new architecture.

## Zhai track: published operations

The track follows Zhai et al., *Precise detection and localization of R-peaks
from ECG signals*, 2023, DOI `10.3934/mbe.2023848`.

### Preprocessing

The raw oriented ECG is passed through a second-order Butterworth 5--35 Hz
band-pass in both forward and reverse directions. This is zero-phase offline
filtering. If the resulting signal is (Y[i]), the enhancement signal is

\[
S[i]=Y[i]^2.
\]

`S` is passed through a second-order, forward--backward 5-Hz low-pass filter to
form the QRS envelope `L`.

### Dynamic window calculation

The threshold is updated in 400-ms blocks. For block (n):

- (M(n)) is the maximum envelope value in the current 400-ms block;
- (D(n)) is the running mean of the previous/current block maxima;
- (A(n)) is the maximum in a forward 2-s amplitude context.

The paper's threshold is implemented as

\[
\theta(n)=\max\left(0.3M(n)+0.1D(n),\;0.05A(n)\right).
\]

Samples satisfying (L[i]>\theta(n)) become initial QRS-window samples.
Windows shorter than one quarter of the mean initial width are removed. When
two window centers are separated by less than 400 ms, the narrower window is
removed. Remaining windows are widened symmetrically to at least 200 ms.

The 400-ms duplicate-window rule can be problematic for genuine rates above
150 beats/min. It is retained because it is part of the published method, not
because it has been validated for peri-ictal tachycardia.

### Template selection

The first five accepted QRS windows are inspected. Inside each, one preliminary
absolute extremum is used to measure candidate amplitude. The window with
amplitude closest to the median is selected, and an odd-sample window nearest
120 ms is extracted around that center.

This one-time absolute-extremum step seeds the template. It is not the old
per-beat highest-positive-sample refinement.

At 360 Hz the published template contains 43 samples. At other sampling rates,
this implementation rounds 120 ms to the nearest sample and increases an even
result by one so the template has a unique center. That generalization is an
implementation decision because the paper evaluates 360-Hz MIT--BIH data.

### Cross-correlation localization

For every possible template alignment, the implementation calculates Pearson
normalized cross-correlation:

\[
C_i(\tau)=\operatorname{corr}\left(T,Y_i[\tau-j:\tau+j]\right).
\]

The final timestamp in each QRS window is

\[
\hat r_i=\arg\max_{\tau}|C_i(\tau)|.
\]

This is an `argmax` of **whole-waveform similarity**, not an `argmax` of raw
positive ECG amplitude. The signed correlation and its absolute value are
saved beside every Zhai event.

If an RR interval is shorter than 0.4 times the current mean RR, the paper
removes the member of the pair with the lower absolute correlation peak. The
paper does not specify whether this is iterated; this reimplementation applies
the rule iteratively until no violating pair remains.

## Paper ambiguities and deviations

No official author implementation was located. Therefore this code is labelled
`paper_reimplementation_no_author_code_located`, not author code. The following
choices are explicit:

1. The 2-s value (A(n)) uses a forward, clipped context beginning at the
   current 400-ms block.
2. Equal-width windows inside 400 ms retain the earlier window.
3. The short-RR/lower-correlation rule is iterated.
4. Sampling rates other than 360 Hz use the odd-sample rule described above.
5. No record-wise group-delay correction is applied to timestamps used for RR.

These choices may partly explain differences from the published results and
must not be retroactively attributed to Zhai et al.

## Detector support without fusion

For each primary timestamp (p_i), the other detector's timestamps inside

\[
[p_i-50\text{ ms},p_i+50\text{ ms}]
\]

are counted. A timestamp receives `qrs_supported=True` only when exactly one
comparator event is present. An RR receives `rr_supported=True` only when both
bounding timestamps are supported and exactly two comparator events lie in
the tolerance-expanded interval.

Unsupported timestamps and RR intervals are retained. Agreement is neither an
artifact label nor ground truth. Each direction receives its own table:

- NeuroKit primary, Zhai comparator;
- Zhai primary, NeuroKit comparator.

## RR and HRV outputs

Each detector's timestamp sequence independently produces

\[
RR_i=1000\frac{r_i-r_{i-1}}{f_s},
\qquad
HR_i=\frac{60000}{RR_i}.
\]

Each RR table then enters the same 100-consecutive-RR Jeppesen feature code:
raw SD1/SD2, CSI, causal seven-RR median path, filtered modified CSI, heart-rate
slope, `J1`, and `J2`. Feature reliability is stored separately from the
continuous value.

## Full MIT--BIH audit

The new benchmark was run on channel 0 of all 48 MIT--BIH Arrhythmia Database
records, using the same 109,494 expert beat annotations used in the Zhai paper.
These are results of **this reimplementation**, not copied paper results.

### Pooled results

| Tolerance | Detector | TP | FP | FN | Sensitivity | PPV | F1 | Median absolute timing | P95 absolute timing |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 75 ms | NeuroKit | 107,677 | 2,503 | 1,817 | 0.98341 | 0.97728 | 0.98033 | 0.00 ms | 19.44 ms |
| 75 ms | Zhai reproduction | 107,324 | 958 | 2,170 | 0.98018 | 0.99115 | 0.98564 | 2.78 ms | 36.11 ms |
| 25 ms | NeuroKit | 103,305 | 6,875 | 6,189 | 0.94348 | 0.93760 | 0.94053 | 0.00 ms | 8.33 ms |
| 25 ms | Zhai reproduction | 99,180 | 9,102 | 10,314 | 0.90580 | 0.91594 | 0.91084 | 2.78 ms | 5.56 ms |
| one sample | NeuroKit | 95,800 | 14,380 | 13,694 | 0.87493 | 0.86949 | 0.87220 | 0.00 ms | 2.78 ms |
| one sample | Zhai reproduction | 89,065 | 19,217 | 20,429 | 0.81342 | 0.82253 | 0.81795 | 2.78 ms | 2.78 ms |

At 75 ms, Zhai has the better pooled F1 because it greatly reduces false
positives in several catastrophic NeuroKit records. It does not dominate
record by record: Zhai wins 18 records, ties 3, and loses 27. The 25-ms and
one-sample results reject the claim that our current reproduction is already a
universally superior exact-timing detector.

For intervals where both detected endpoints matched two consecutive expert
beats at 75 ms, NeuroKit had RR mean absolute error 3.31 ms and 95th-percentile
absolute error 13.89 ms; Zhai had RR mean absolute error 3.83 ms and
95th-percentile absolute error 36.11 ms. Both had median absolute RR error 0
ms, illustrating that a consistent per-record timestamp displacement can
cancel when successive timestamps are subtracted. These RR statistics exclude
intervals broken by an extra or missed detection and therefore cannot replace
the event-level FP/FN analysis. At the 25-ms matched subset Zhai's RR MAE was
1.22 ms versus NeuroKit's 1.40 ms, but the subset itself differs because Zhai
matched fewer events at that tolerance.

### Five previously difficult records at 75 ms

| Record | NeuroKit F1 | Zhai F1 | Main observation |
|---|---:|---:|---|
| 108 | 0.8917 | 0.9929 | Zhai recovers most mixed errors. |
| 113 | 0.7753 | 0.9994 | Zhai avoids NeuroKit's repeated T-wave detections. |
| 207 | 0.8177 | 0.9513 | Zhai handles changing polarity/morphology much better, but still adds 163 false events. |
| 222 | 0.7834 | 0.9845 | Zhai strongly reduces the mixed false/missed detections. |
| 231 | 0.8846 | 1.0000 | Zhai avoids the 410 extra NeuroKit detections. |

Zhai's worst record is 203 (`F1=0.8884`). It also deteriorates on records 119,
217, 233, and 208. This agrees with the paper's own warning that its single
normal-width template can degrade on PVCs, paced beats, fusion beats, and
atypical morphologies. Multiple templates were proposed by the authors as
future work, not implemented or validated in their paper.

The published Zhai result was sensitivity/PPV 99.78% and ADE 8.35 ms. Our code
must not claim to reproduce those numbers. Likely contributors include
under-specified windowing details, the absence of author code, morphology mix,
and the paper's record-wise group-delay compensation during timing evaluation.

Raw tables are in `outputs/neurokit_zhai_mitdb_all48`.

## Commands

WFDB:

```powershell
ecg-features run-neurokit-zhai-wfdb `
  --record "C:\path\mitdb\113" `
  --channel 0 `
  --output-dir "C:\path\results\113_neurokit_zhai"
```

EDF:

```powershell
ecg-features run-neurokit-zhai-edf `
  --edf "C:\path\patient.edf" `
  --channel ECG `
  --start-s 0 `
  --duration-s 600 `
  --output-dir "C:\path\results\patient_segment"
```

Full MIT--BIH detector audit:

```powershell
$env:PYTHONPATH="src"
python scripts\benchmark_neurokit_zhai.py `
  --database-dir "C:\path\mit-bih-arrhythmia-database-1.0.0" `
  --output-dir "outputs\neurokit_zhai_mitdb_all48" `
  --channel 0
```

## Current decision

The code establishes Zhai as a highly valuable independent comparator and a
possible future timestamp owner. It does not yet justify automatically choosing
Zhai, NeuroKit, or a beat-wise mixture. The next gate is manual annotation of
representative patient ECG stratified by clean/noisy periods, rate, polarity,
PVCs, bundle-branch morphology, and peri-ictal state, followed by RR-error and
downstream HRV-error comparison.
