# Conduction-repolarization information branch v2

## Decision

This is an **observational information branch**, not a seizure-model branch.
It is not called by `pipeline.py`, is not joined to a fusion-ready table, and
sets the following fields to false on every output row:

- `final_feature_matrix_eligible`;
- `seizure_probability_produced`;
- `clinical_interpretation_produced`;
- `abnormality_label_produced`.

The branch answers two limited questions:

1. What conduction/repolarization timings were measured on each beat?
2. Over the current five beats, what is their typical value, spread, local
   trend, and change relative to the preceding non-overlapping five beats?

It does not answer whether a change is caused by a seizure or whether it is
clinically dangerous.

## Paper audit that determined the scope

### Direct PQRST seizure paper: keep the raw peak dynamics

[Diab et al. 2025](https://doi.org/10.1016/j.neucli.2025.103098) did not use
five-beat averages of PR, QRS duration, QT, or JT.  It created nine timing
features per beat:

```text
within the current beat:
  deltaP = Ppeak(i) - R(i)
  deltaQ = Qpeak(i) - R(i)
  deltaS = Speak(i) - R(i)
  deltaT = Tpeak(i) - R(i)

between successive beats:
  DeltaP = Ppeak(i) - Ppeak(i-1)
  DeltaQ = Qpeak(i) - Qpeak(i-1)
  DeltaR = R(i)     - R(i-1)
  DeltaS = Speak(i) - Speak(i-1)
  DeltaT = Tpeak(i) - Tpeak(i-1)
```

Absolute recording time was removed.  The ordered raw values from 3-60
successive beats were supplied to the classifier.  The authors explicitly
state that they did not aggregate, average, or calculate the variance of these
features.  Therefore this implementation preserves the nine raw per-beat
features and does not claim that the five-beat summaries reproduce Diab.

The paper contains a numerical reporting inconsistency: its explicit equation
gives `9 * N` features (540 for 60 beats), while the discussion later reports
600 for 60 beats and 30 for 3 beats.  This code follows the explicit four
within-beat plus five interbeat definitions, not the inconsistent later count.

### QT seizure physiology: retain the published nine-beat context

[Brotherstone et al. 2010](https://doi.org/10.1111/j.1528-1167.2009.02281.x)
averaged QT and the associated RR over nine consecutive beats and then applied
Bazett, Hodges, Fridericia, and Framingham correction formulas.  The v2 output
therefore includes a separate nine-beat QT/RR context table.  It is physiology
context, not a reproduced seizure detector.

### Why changepoint detection is not yet applied to these intervals

[Cheung et al. 2021](https://doi.org/10.1109/EMBC46164.2021.9629760) is a useful
continuous-ECG architecture precedent: one-minute adaptive QRS+T templates,
cross-correlation artifact rejection, and mean changepoint detection over long
recordings.  The methods mention QRS length and P-R, R-Q, R-T, and S-T
intervals.  However, the reported seizure-specific analysis focused on only
four measures: RR, heart rate, atrial approximate entropy, and atrial fractal
dimension.  It therefore does not validate changepoint detection of PR, QRS,
QT, or JT as seizure information.  Changepoints remain a later challenger
after direct interval validation and external artifact eligibility exist.

### Why five-beat QT variability is excluded

The [EHRA/ESC QT-variability position statement](https://doi.org/10.1093/europace/euv405)
warns that QT variability is easily contaminated by T-end error, baseline
wander, respiratory axis movement, ectopy, and heart-rate change.  It
recommends dedicated high-quality measurement, steady heart rate, and exclusion
of ectopic and subsequent beats for regular-conduction QTV analysis.  The
published STVQT equation in the v1 control also requires exactly 30 beats.

Consequently, v2 does not calculate five-beat QTVI or relabel five-beat QT
spread as repolarization instability.

## Implemented outputs

### 1. `conduction_information_beats.csv`

One row per beat containing:

- P duration, PR interval, PR segment, QRS duration, QT, JT, and
  Tpeak-to-Tend;
- separately named Bazett, Fridericia, and Framingham QTc;
- preceding RR and instantaneous heart rate as interpretation context;
- the nine Diab peak-dynamics features listed above;
- availability, support-end timing, and information-only flags.

Q and S peak positions are retained from the prominence delineator instead of
being discarded.  Their presence and physiological order are separately
reported; they have not been manually validated on the seizure recordings.

### 2. `conduction_information_windows5_long.csv`

This is a long table: one row per measurement per five-beat window endpoint.
The summarized measurements are:

- P duration;
- PR interval;
- PR segment;
- QRS duration;
- QT;
- JT;
- Tpeak-to-Tend;
- Bazett, Fridericia, and Framingham QTc.

For each measurement it reports:

- valid-beat count, availability fraction, and support status;
- mean and median;
- sample SD, raw MAD, IQR, minimum, maximum, and range;
- RMSSD, median absolute successive difference, and maximum absolute
  successive difference;
- Theil-Sen robust slope in milliseconds per beat;
- median of the last two minus median of the first two beats;
- current value minus the median of the preceding beats;
- median shift versus the immediately preceding non-overlapping five beats;
- the absolute shift and a MAD/resolution-scaled descriptive effect size.

The median and MAD are included because one bad landmark can strongly distort a
five-value mean and SD.  The mean remains present because it is familiar and
auditable.  None of these summaries is called a seizure biomarker.

### 3. `conduction_information_qt_context9.csv`

For nine consecutive eligible beats it reports mean QT, mean preceding RR, and
QTc calculated from those two means.  All nine paired values are required.

### 4. `pqrst_landmarks.csv` and `run_scope.json`

These preserve the upstream landmarks, lead/patient provenance, parameters,
evidence mapping, and explicit non-integration status.

## External artifact branch contract

The function accepts an optional boolean `eligibility_column`.  If supplied,
that decision is consumed from the separate quality/artifact branch and
ineligible beats are excluded from summaries while remaining visible in the
beat table.  This branch does not invent, reinterpret, or replace the general
artifact score.

Without an eligibility column, finite landmark measurements are summarized
and metadata explicitly states that no artifact gate was applied.

## Why there is no `bad_change` flag

A five-beat change can be caused by:

- true conduction or repolarization physiology;
- ordinary heart-rate adaptation;
- ectopy or the post-ectopic beat;
- movement or lead-vector change;
- baseline/EMG contamination;
- P, Q, S, or T landmark error.

The current matched-lead benchmark showed endpoint-error tails that can exceed
the expected physiological beat-to-beat variation.  Therefore a universal
millisecond threshold would mainly convert measurement error into a confident
label.

A future `measurable_shift_candidate` flag may be enabled only when all of the
following exist:

1. external domain-specific quality eligibility;
2. homogeneous sinus-beat chains;
3. patient/lead/rate-aware baseline;
4. direct interval-level manual error estimates;
5. a shift larger than the prespecified measurement-error floor;
6. persistence in at least two quality-qualified windows.

Even that flag would mean “change worth reviewing,” not “seizure” or
“dangerous ECG.”

## Software and smoke-test status

Implemented files:

- `src/ecg_cascade/conduction_information.py`;
- `scripts/run_conduction_information.py`;
- `tests/test_conduction_information.py`;
- Q/S peak retention in `src/ecg_cascade/prominence_morphology.py`.

The tests cover the nine published peak-dynamics equations, five-beat robust
statistics, non-overlapping-window shift, nine-beat QT averaging, external
eligibility, causal future perturbation, and explicit non-prediction status.

The branch was smoke-tested on the first 60 seconds of MIT-BIH record 100,
channel MLII, with expert `.atr` R anchors:

| item | result |
|---|---:|
| beats | 74 |
| complete five-beat endpoints | 70 |
| beats with all nine Diab dynamics | 73 |
| classifier or threshold | none |
| final-matrix integration | none |

MIT-BIH `.atr` supplies beat/R references, not manual P/Q/S/T boundaries.
This execution verifies software availability only; it is not accuracy or
seizure validation.

Example command:

```powershell
.venv\Scripts\python scripts\run_conduction_information.py `
  --record "C:\path\to\mitdb\100" `
  --channel 0 `
  --start-s 0 `
  --duration-s 60 `
  --patient-id mitdb_100 `
  --lead-name MLII `
  --output outputs\conduction_information_mitdb_100_60s_v2
```
