# Interval-consistent RR fusion and pure-window HRV audit

## Decision

**Keep unchanged UNSW timestamps as the conservative RR/HRV baseline.**

The proposed same-source-per-RR rule is a useful experimental challenger, but
it is not accepted as the automatic timestamp owner. It improved pooled RR
error and overall J1/J2 concordance, but it did not dominate the baseline on
all prespecified downstream tests:

- combined NeuroKit--Zhai--UNSW RR MAE improved from **3.971 ms to 3.182 ms**;
- successive-RR-change MAE improved from **7.325 ms to 6.166 ms**;
- J1 CCC improved from **0.990377 to 0.990778**;
- J2 CCC improved from **0.993876 to 0.994390**;
- 95th-percentile J1 threshold F1 improved from **0.8602 to 0.8621**;
- 95th-percentile J2 threshold F1 deteriorated from **0.8596 to 0.8460**;
- at adjacent-RR source transitions, successive-change MAE remained
  **23.155 ms**, versus **5.868 ms** when adjacent RRs kept the same source.

Thus, the rule removes mixed detector endpoints *inside* an RR interval, but
it does not create a single coherent timestamp sequence. Residual error moves
to locations where one RR uses one detector and the next RR uses another.

This is an extraction-fidelity result on MIT--BIH arrhythmia ECG. It is not a
seizure-detection result and does not validate an abnormality threshold.

## Evidence status

| Component | Status |
|---|---|
| UNSW primary detector with a second detector used for RR reliability | Published structure in Ho et al. |
| qSQI-style detector agreement as signal-quality context | Published SQI concept in Zhao and Zhang |
| Direct comparison of detector-derived versus annotated HRV using Lin CCC | Published validation strategy in Weinstein et al. |
| 100-RR CSI, filtered ModCSI, and HR-slope seizure features | Published Jeppesen method |
| NeuroKit then Zhai then UNSW same-source-per-RR substitution | **Project ablation; not a published fusion algorithm** |
| Pure NeuroKit/Zhai 100-RR challenger lanes conditioned on unique UNSW associations | **Project ablation; not a standalone detector benchmark** |
| Per-record 90th/95th/99th-percentile thresholds | **Measurement stress test; not the Jeppesen seizure threshold** |

Primary literature:

- Ho B et al. *Automated RR Interval Detection and Quality Assessment in
  Telehealth*. Computing in Cardiology. 2024.
  https://cinc.org/archives/2024/pdf/CinC2024-084.pdf
- Zhao Z, Zhang Y. *SQI Quality Evaluation Mechanism of Single-Lead ECG Signal
  Based on Simple Heuristic Fusion and Fuzzy Comprehensive Evaluation*.
  Frontiers in Physiology. 2018. https://pmc.ncbi.nlm.nih.gov/articles/PMC6011094/
- Weinstein A, Rodino J, Otero M. *Effects of electrocardiogram QRS detection
  algorithms in heart rate variability metrics*. Scientific Reports. 2026.
  https://doi.org/10.1038/s41598-026-49215-6
- Jeppesen J et al. *Seizure detection using wearable electrocardiogram
  connected to a smartphone: a phase 3 clinical validation study*.
  EBioMedicine. 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC12516532/

Ho et al. used a primary detector to calculate RR and a secondary detector to
assess whether both bounding QRS events were supported. That paper does **not**
establish our timestamp-substitution rule. Zhao and Zhang support detector
agreement as an SQI; detector agreement is not a calibrated probability that
a beat is correct. Weinstein et al. support measuring the actual downstream
HRV error instead of inferring HRV validity from beat F1 alone.

## Exact implemented rules

### Conservative baseline

Every interval uses unchanged UNSW samples:

\[
RR_i^{U}=1000\frac{U_{i+1}-U_i}{f_s}.
\]

NeuroKit and Zhai candidates are stored as context and do not replace either
endpoint.

### Old per-beat hierarchy

The previous experiment chose one timestamp per beat, then differenced the
selected sequence. This allowed intervals such as

\[
N_{i+1}-U_i,
\]

which combine two timestamp conventions. In the current common evaluation
grid, 2,843 RR intervals had mixed detector endpoints.

### New same-source-per-RR hierarchy

For every UNSW-indexed interval:

\[
RR_i=
\begin{cases}
1000(N_{i+1}-N_i)/f_s,&N_i,N_{i+1}\text{ both uniquely associated},\\
1000(Z_{i+1}-Z_i)/f_s,&\text{otherwise, if }Z_i,Z_{i+1}\text{ both associated},\\
1000(U_{i+1}-U_i)/f_s,&\text{otherwise}.
\end{cases}
\]

This guarantees zero mixed-endpoint intervals. It does not guarantee that
adjacent intervals share one timestamp at their common physiological beat:

```text
RR i     = NeuroKit(i+1) - NeuroKit(i)
RR i+1   = UNSW(i+2)     - UNSW(i+1)
                            ^
             same heartbeat, different timestamp conventions
```

Consequently, the sequence of RR values is mathematically valid interval by
interval but is not derived from one global timestamp line.

### Pure candidate lanes

The pure NeuroKit and pure Zhai lanes are stricter. A candidate RR exists only
when both UNSW-indexed boundaries have one unique corresponding candidate from
that detector. A complete 100-RR HRV challenger requires **107 consecutive
candidate events**:

- 101 events define 100 RR intervals;
- six earlier RR intervals supply full history for the causal seven-RR median
  filter used by filtered ModCSI and slope.

These are association-conditioned lanes. They exclude ambiguous or missing
candidate associations and therefore are not identical to unconstrained
standalone NeuroKit or Zhai tachograms.

## Dataset and evaluation

- Database: all 48 MIT--BIH Arrhythmia Database records.
- Lead: first database channel, normally MLII but not universally the same
  physical lead in every record.
- Sampling frequency: 360 Hz; one sample is 2.778 ms.
- Reference: expert `atr` beat annotations.
- Beat matching tolerance: +/-75 ms.
- Primary fusion setting declared before this audit: unmodified NeuroKit
  300-ms strict configuration with 50-ms association tolerance.
- Sensitivity settings: NeuroKit 250-ms experimental variant and association
  tolerances 50, 75, 100, and 150 ms.

RR error was evaluated only where two consecutive UNSW events matched two
consecutive expert events. There were 109,546 algorithmic RR intervals and
108,874 evaluable expert-paired intervals. HRV comparison used 104,450 paired
100-RR windows.

Adjacent 100-RR outputs overlap by 99 intervals. They are not independent
observations. Pooled window counts must not be treated as a clinical sample
size.

## Complete primary-setting comparison

`q95 F1` asks whether the algorithmic feature crossed the same per-record
expert-derived 95th-percentile threshold. It does not classify seizures.

| Method | RR availability | RR MAE (ms) | Successive-change MAE (ms) | HRV window coverage | J1 CCC | J2 CCC | q95 F1 J1 / J2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Unchanged UNSW baseline | 100.0% | 3.971 | 7.325 | 100.0% | 0.990377 | 0.993876 | 0.8602 / 0.8596 |
| Old per-beat fusion, common grid | 100.0% | 3.416 | 6.627 | 100.0% | 0.990785 | 0.994326 | 0.8633 / 0.8457 |
| Interval NeuroKit then UNSW | 100.0% | 3.185 | 6.163 | 100.0% | 0.990786 | 0.994337 | 0.8627 / 0.8490 |
| Interval Zhai then UNSW | 100.0% | 3.523 | 6.776 | 100.0% | 0.990387 | 0.994169 | 0.8527 / 0.8570 |
| Interval NeuroKit then Zhai then UNSW | 100.0% | 3.182 | 6.166 | 100.0% | 0.990778 | 0.994390 | 0.8621 / 0.8460 |
| Pure NeuroKit candidate lane | 95.3% | 2.967 | 5.675 | 71.0% | 0.999633 | 0.996874 | 0.9421 / 0.9386* |
| Pure Zhai candidate lane | 93.2% | 2.706 | 5.161 | 57.3% | 0.999340 | 0.998084 | 0.9481 / 0.9421* |
| Whole-window priority NeuroKit, Zhai, UNSW | -- | -- | -- | 100.0% | 0.990682 | 0.993677 | 0.8630 / 0.8595 |

`*` Pure-lane F1 is conditional on available windows and is therefore not
comparable without its coverage. If missing windows are treated as missed
expert-high events, q95 hard-veto F1 falls to:

| Lane | J1 hard-veto F1 | J2 hard-veto F1 | Expert-high-window coverage J1 / J2 |
|---|---:|---:|---:|
| Pure NeuroKit | 0.718 | 0.693 | 61.4% / 58.6% |
| Pure Zhai | 0.586 | 0.539 | 44.7% / 40.1% |

Therefore, the very high pure-lane CCC and conditional F1 partly reflect
selection of easier, fully associated windows. They are promising precision
lanes, not complete replacements.

## Comparison with the previous standalone audit

These earlier numbers use the prior standalone/fusion evaluation convention
and are retained for context:

| Prior method | Beat F1 | Mean absolute timing error (ms) | RR MAE (ms) |
|---|---:|---:|---:|
| UNSW | 0.997261 | 8.705 | 3.971 |
| NeuroKit 300-ms strict | 0.980334 | 3.574 | 3.307 |
| Zhai reimplementation | 0.985637 | 5.232 | 3.832 |
| Old per-beat fusion | 0.997070 | 3.009 | 3.382 |

The old fusion RR MAE is 3.382 ms in the previous standalone evaluation and
3.416 ms on the current UNSW-indexed common interval grid. They answer slightly
different questions and must not be presented as an unexplained discrepancy.

The pure association-conditioned lanes do not have their own beat F1. Their
parent events are UNSW events, and their reported availability states how often
a unique candidate could be associated. The standalone NeuroKit and Zhai beat
F1 values above remain the relevant independent beat-existence results.

## Source-transition audit

| Method | Mixed-endpoint RRs | RR-source transitions | Same-source successive-change MAE | Transition successive-change MAE |
|---|---:|---:|---:|---:|
| Old per-beat fusion | 2,843 | 4,157 | 5.617 ms | 36.491 ms |
| Interval NeuroKit then UNSW | 0 | 2,051 | 5.902 ms | 21.892 ms |
| Interval Zhai then UNSW | 0 | 5,274 | 6.159 ms | 19.809 ms |
| Interval NeuroKit then Zhai then UNSW | 0 | 2,197 | 5.868 ms | 23.155 ms |

The new rule therefore materially improves the old transition failure, but
does not solve it. In particular, adding the Zhai fallback to the NeuroKit--UNSW
interval hierarchy changes RR MAE by only **0.0029 ms** (3.1845 to 3.1816),
while transition error rises and J2 q95 F1 falls. The evidence does not justify
the added interval-level complexity.

## Record-level heterogeneity

For the combined interval hierarchy, RR MAE improved in 33 records and worsened
in 15. The mean paired record delta was -0.836 ms and the median was -0.573 ms.
An exploratory record-level Wilcoxon signed-rank test produced p=0.00037, but a
20,000-resample record bootstrap 95% interval for the mean delta was
[-1.80, +0.24] ms because a few records deteriorated severely. These are
development-set descriptions, not confirmatory inference.

| Record | UNSW RR MAE | Combined interval RR MAE | Change | Interpretation supported by this audit |
|---|---:|---:|---:|---|
| 108 | 8.379 | 9.220 | +0.840 | Worse; mixed arrhythmia/noise case remains difficult. |
| 113 | 0.919 | 0.251 | -0.668 | Improved; NeuroKit candidate lane is stable in this record. |
| 200 | 6.757 | 22.223 | +15.467 | Catastrophic deterioration. Header documents multiform PVCs and noise/artifact. |
| 207 | 3.635 | 2.520 | -1.115 | Improved overall despite the known changing-morphology difficulty. |
| 222 | 5.433 | 1.165 | -4.268 | Large improvement. |
| 231 | 1.525 | 1.235 | -0.290 | Modest improvement. |
| 233 | 8.291 | 15.499 | +7.208 | Major deterioration. Header documents multiform PVCs. |

The official local headers establish that records 200 and 233 contain
multiform PVCs, and record 200 also contains noise. This association is
consistent with morphology-dependent timing failure, but does not prove which
mechanism caused every erroneous interval. A waveform-level error audit is
required before correction.

## Tolerance sensitivity

The predeclared 50-ms setting was best for the NeuroKit-containing interval
rules. Widening the association window did not rescue the transition problem:

| NeuroKit 300-ms strict tolerance | NK--UNSW RR MAE / successive-change MAE | NK--Zhai--UNSW RR MAE / successive-change MAE |
|---:|---:|---:|
| 50 ms | 3.185 / 6.163 | 3.182 / 6.166 |
| 75 ms | 3.367 / 6.526 | 3.364 / 6.528 |
| 100 ms | 3.384 / 6.560 | 3.395 / 6.585 |
| 150 ms | 3.850 / 7.486 | 3.892 / 7.568 |

At 150 ms, combined successive-change error exceeded the UNSW baseline. The
250-ms NeuroKit experiment produced nearly the same pattern and did not beat
the unmodified 300-ms configuration at the predeclared 50-ms setting.

## Long-run threshold interpretation

The q90/q95/q99 simulations answer:

> If an expert-derived HRV feature exceeds a high within-record threshold,
> does the feature calculated from algorithmic timestamps cross the same
> numerical threshold?

They do **not** answer whether the window is a seizure. MIT--BIH has arrhythmia
labels, not seizure-onset labels. The phase-3 Jeppesen system used a
patient-specific baseline procedure and an alarm when either published feature
crossed its defined threshold. Our per-record percentile thresholds are only
stress tests of measurement preservation and must not be substituted for that
clinical rule.

At q95:

- UNSW baseline: J1/J2 F1 = 0.860/0.860;
- interval NeuroKit--UNSW: 0.863/0.849;
- combined interval hierarchy: 0.862/0.846;
- whole-window priority: 0.863/0.860.

The combined interval hierarchy therefore does not preserve both feature
thresholds better than UNSW.

## Recommended implemented architecture

```text
Raw ECG
  |
  +-- UNSW events ------------------------> always-available RR/HRV baseline
  |
  +-- NeuroKit candidates -- unique association/context
  |       +-- fully contiguous 107-event run --> pure NK HRV challenger
  |
  +-- Zhai candidates ------ unique association/context
          +-- fully contiguous 107-event run --> pure Zhai HRV challenger

Per interval/window output
  +-- UNSW RR, J1, J2                         [decision baseline]
  +-- pure NeuroKit RR/J1/J2 if available    [experimental challenger]
  +-- pure Zhai RR/J1/J2 if available        [experimental challenger]
  +-- source, coverage, offsets, ambiguity, qSQI context
  +-- no automatic timestamp substitution
```

The interval-consistent NeuroKit--UNSW result should remain an offline ablation.
If only one interval challenger is retained, use the simpler NeuroKit--UNSW
variant: the Zhai fallback gave negligible pooled RR benefit and worsened key
transition and J2-threshold results.

## Patient-specific validation gate

Before changing the timestamp owner for the seizure patient:

1. Manually annotate representative ECG from separate clean, artifact,
   interictal, preictal, ictal, and postictal periods, retaining beat morphology.
2. Split chronologically into development, threshold-calibration, and untouched
   test periods; overlapping 100-RR windows must not cross splits.
3. Lock detector settings and the patient threshold using development and
   calibration data only.
4. On the untouched test period, report beat F1/timing, RR MAE, successive-RR
   error, J1/J2 CCC, threshold-crossing sensitivity/PPV/F1, availability, and
   expert-high-window coverage.
5. Report seizure sensitivity, false alarms per hour, and latency only after
   seizure annotations exist. Detector disagreement must remain context, not an
   artifact label or hard veto.

## Reproducible artifacts

- Implementation:
  `feature_extraction/src/ecg_cascade/interval_consistent_fusion.py`
- Vectorized HRV calculations:
  `feature_extraction/src/ecg_cascade/rr_hrv.py`
- Benchmark:
  `feature_extraction/scripts/benchmark_interval_consistent_fusion.py`
- Complete method table:
  `feature_extraction/outputs/interval_consistent_fusion_all48_v1/complete_method_comparison.csv`
- Record RR metrics:
  `feature_extraction/outputs/interval_consistent_fusion_all48_v1/record_rr_metrics.csv`
- Pooled RR/HRV/threshold tables:
  `feature_extraction/outputs/interval_consistent_fusion_all48_v1/pooled_rr_metrics.csv`,
  `pooled_hrv_metrics.csv`, and `pooled_threshold_metrics.csv`
- Decision record:
  `feature_extraction/outputs/interval_consistent_fusion_all48_v1/decision_summary.json`
- Plots: RR/successive error, transition error, HRV CCC, threshold preservation,
  record deterioration, and tolerance sensitivity in the same output directory.

## Software verification

- The user example with true 800/800-ms intervals is covered by an automated
  test: the old mixed fusion creates 800/815 ms, whereas interval-consistent
  fallback returns 800/800 ms.
- Zero mixed-endpoint intervals are asserted for the new rule.
- Vectorized J1/J2 equations reproduce the existing audited implementation.
- Vectorized pure-window construction was compared against the prior
  implementation and produced zero numerical difference in counts, CCC, MAE,
  and sMAPE.

