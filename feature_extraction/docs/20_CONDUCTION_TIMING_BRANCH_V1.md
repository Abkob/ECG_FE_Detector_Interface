# Conduction and repolarization timing branch v1

## Decision

The short branch name is **conduction**.  Its scientifically complete scope is
conduction **and repolarization timing**.  It is a standalone measurement
prototype and is not enabled in the main seizure pipeline.

“Between-beat timing” alone means RR variability and remains owned by the
RR/HRV branch.  The new branch measures within-beat electrical timing and then
asks how those measurements change across consecutive beats:

```text
P onset -> P offset          atrial depolarization duration
P onset -> QRS onset        PR interval; atrial-to-ventricular conduction
P offset -> QRS onset       PR segment
QRS onset -> QRS offset     ventricular depolarization/conduction duration
QRS onset -> T offset       QT; depolarization plus repolarization
QRS offset -> T offset      JT; repolarization separated from QRS duration
```

The branch emits measurements and availability evidence.  It does not diagnose
heart block, long-QT syndrome, bundle-branch block, SUDEP risk, or seizures.

## Why this lane is evidence-based

The seizure literature provides support, contradiction, and important limits:

- Nei et al. measured PR and QTc in preictal, ictal, and postictal periods in
  51 seizures.  Rhythm/conduction abnormalities occurred, but mean PR and QTc
  did not change clinically significantly.  This contradicts treating every
  interval movement as a universal seizure marker.  DOI:
  <https://doi.org/10.1111/j.1528-1157.2000.tb00207.x>.
- Brotherstone et al. paired consecutive RR and QT intervals and compared four
  QT-correction equations across 156 seizures.  They found significant QTc
  changes in some seizures, using nine-beat averages.  DOI:
  <https://doi.org/10.1111/j.1528-1167.2009.02281.x>.
- Pensel et al. measured single-lead PR and RR intervals in 56 mesial temporal
  seizures.  Average PR shortened during seizures, and PR/heart-rate regression
  differed by seizure laterality; clinical relevance remained uncertain.  DOI:
  <https://doi.org/10.3389/fneur.2021.661391>.
- Berger et al. introduced the QT variability index using the log ratio of
  normalized QT variance to normalized heart-rate variance.  DOI:
  <https://doi.org/10.1161/01.CIR.96.5.1557>.
- Hinterseer et al. used 30 consecutive lead-II beats and defined
  `STVQT = sum(abs(QT[n+1]-QT[n])) / (30*sqrt(2))`.  DOI:
  <https://doi.org/10.1093/eurheartj/ehm586>.
- The EHRA/ESC QT-variability position statement warns that normal beatwise QT
  fluctuations are small, usually with SD below 5 ms at stable heart rates,
  and that T-end measurement requires high-quality, dedicated methods.  DOI:
  <https://doi.org/10.1093/europace/euv405>.
- A large 2022 analysis found short-term QT variability was influenced more by
  recording quality/morphological instability than by RR variability.  DOI:
  <https://doi.org/10.3389/fphys.2022.863873>.

These papers justify measuring the intervals.  They do **not** establish that
this exact feature set detects seizures in the target recordings.

## Implemented contract

Input: one patient, one lead, and one time-ordered table of aligned P/QRS/T
landmarks, such as the output of `build_pqrst_feature_table`.

Beatwise output:

- preceding RR and instantaneous heart rate as context, not duplicated HRV;
- P duration, PR interval, PR segment, QRS duration, QT, JT, and T-peak-to-end;
- QRS-onset-to-R and R-to-QRS-offset timing;
- separately named Bazett, Fridericia, and Framingham QTc values;
- first differences and absolute first differences for PR, QRS, QT, and JT;
- separate `*_defined` flags and feature-specific support-end samples.

Trailing 30-beat output:

- mean, median, SD, RMSSD, valid count, and availability fraction for PR, QRS,
  QT, and JT;
- published QT STV30 only when all 30 consecutive QT values exist;
- exploratory QTVI using both the original heart-rate denominator and the
  commonly used RR-denominator adaptation;
- descriptive PR-versus-heart-rate slope and correlation.

The 30-beat QTVI columns are explicitly prefixed `exploratory_`: much of the
QTVI literature uses longer stationary recordings.  The PR/heart-rate rolling
regression is an implementation translation of published epoch-level analysis,
not a reproduction of the Pensel study.

## Exact QT corrections

With QT in milliseconds and RR in seconds:

```text
QTc_Bazett      = QT / RR^(1/2)
QTc_Fridericia  = QT / RR^(1/3)
QTc_Framingham  = QT + 154 * (1 - RR)
```

No formula is silently designated “correct.”  Fixed corrections can remain
biased during rapid heart-rate change, which is especially relevant around a
seizure.  Each QT is paired with the preceding RR ending at that beat, and the
choice is recorded in metadata.

## Causality and missingness

Every rolling row uses only the current and preceding beats.  A QT measurement
is not available until its T offset occurs; the per-feature support-end sample
records that delay.  Missing P onset or T offset remains missing and never
becomes zero.

The aggregation is causal **given finalized landmarks**.  NeuroKit cleaning and
delineation are not asserted to be streaming-causal, so this module cannot yet
support a real-time latency claim.

## Current measurement gate: not passed for variability

The matched-lead prominence benchmark reported:

| dataset | landmark | median absolute error | P95 absolute error |
|---|---|---:|---:|
| LUDB II | P onset | 8 ms | 34 ms |
| LUDB II | QRS onset | 8 ms | 24 ms |
| LUDB II | QRS offset | 14 ms | 44 ms |
| LUDB II | T offset | 18 ms | 52 ms |
| QTDB MLII | P onset | 20 ms | 68.2 ms |
| QTDB MLII | QRS onset | 12 ms | 48 ms |
| QTDB MLII | QRS offset | 16 ms | 72 ms |
| QTDB MLII | T offset | 16 ms | 108 ms |

Those endpoint errors can be much larger than true beat-to-beat QT variability.
Therefore the code is suitable for equation tests and interval-level validation,
but its variability values must not enter a seizure model yet.

## Required next validation

1. Compute paired manual-versus-automatic errors for PR, QRS, QT, and JT—not
   only separate landmark errors.
2. Measure how much observed SD/RMSSD/STV remains after accounting for boundary
   error, sampling rate, signal quality, ectopy, and morphology changes.
3. Test lead and patient dependence; one single-lead boundary is not a global
   12-lead clinical interval.
4. Compare fixed nine-beat averages, causal 30-beat variability, and longer
   stationary QTV windows without selecting the final test set.
5. On video-EEG seizure data, use patient/episode-held-out evaluation and report
   false alarms per 24 hours, unavailable monitoring time, and confidence
   intervals.
6. Add the branch to `pipeline.py` only after the interval-level measurement
   gate and a streaming-causality audit pass.

## Real-data smoke diagnostic

The standalone runner was exercised on the first 60 seconds of MIT--BIH record
100, channel MLII, using expert `.atr` R anchors and automatic prominence
boundaries.  This is a software/availability diagnostic, not a manual interval
validation and not a seizure experiment.

| item | result |
|---|---:|
| beats | 74 |
| PR available | 73 |
| QRS available | 74 |
| QT available | 74 |
| PR range | 158.3--208.3 ms |
| QRS range | 58.3--86.1 ms |
| QT range | 286.1--502.8 ms |
| final trailing QT SD30 | 43.0 ms |
| final QT STV30 | 28.0 ms |

The QT spread is far larger than the sub-5-ms SD described for stable,
high-quality normal recordings in the QT-variability consensus literature.
Without beat-level manual QT references, it cannot be separated into true
physiology, morphology differences, and boundary/noise error.  This result
supports retaining the measurement gate rather than treating the generated
QTV features as validated.

Outputs:

- `outputs/conduction_mitdb_100_60s_v1/conduction_beat_features.csv`
- `outputs/conduction_mitdb_100_60s_v1/conduction_rolling_features.csv`
- `outputs/conduction_mitdb_100_60s_v1/pqrst_landmarks_and_morphology.csv`
- `outputs/conduction_mitdb_100_60s_v1/run_scope.json`

## Code and tests

- `src/ecg_cascade/conduction_timing.py`
- `tests/test_conduction_timing.py`
- `scripts/run_conduction_timing.py`

The tests cover hand-computed intervals, QT formulas, missing endpoints,
minimum history, zero-variance QTVI, schema rejection, and a future-perturbation
causality check.  They prove mathematical behavior, not clinical accuracy or
seizure discrimination.
