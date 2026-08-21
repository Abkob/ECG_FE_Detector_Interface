# Expert-beat versus unchanged-UNSW long-run HRV fidelity

## Decision

The current UNSW-owned RR stream is **promising enough to continue into
patient-specific validation**, but it is **not yet accurate enough to claim a
validated abnormality or seizure alarm**.

Across 104,450 paired 100-RR windows, the two intended branch outputs closely
tracked the corresponding expert-annotation outputs:

| Output | Pooled Lin CCC | Median record CCC | Records with CCC >= 0.8 |
|---|---:|---:|---:|
| J1 = CSI100 x slope100 | 0.9904 | 0.9937 | 44/48 |
| J2 = filtered ModCSI100 x slope100 | 0.9939 | 0.9921 | 42/48 |

However, agreement deteriorated for the most extreme windows and in several
individual records. At an expert-derived, patient-specific 95th-percentile
threshold, the same absolute threshold produced F1 0.860 for J1 and 0.860 for
J2. At the 99th percentile, F1 fell to 0.707 and 0.750. These are measurement-
preservation results, not seizure-detection results.

## Why this benchmark was necessary

The 2026 Scientific Reports benchmark by Weinstein, Rodino, and Otero compared
HRV calculated from manual R annotations with HRV calculated from eight QRS
detectors. It found that detector ranking depended strongly on the HRV feature
and acquisition condition and that beat-detection accuracy did not reliably
predict downstream HRV agreement. That paper used Lin's concordance
correlation coefficient (CCC) because CCC penalizes correlation, scale error,
and systematic bias relative to the identity line.

This audit therefore does not infer HRV validity from UNSW's beat F1. It
directly recalculates the branch features from expert and algorithmic beat
sequences.

Primary sources:

- Weinstein A, Rodino J, Otero M. *Effects of electrocardiogram QRS detection
  algorithms in heart rate variability metrics*. Scientific Reports. 2026.
  https://doi.org/10.1038/s41598-026-49215-6
- Jeppesen J et al. *Phase-3 validation of ECG-derived HRV seizure detection*.
  EBioMedicine. 2025. https://doi.org/10.1016/j.ebiom.2025.105952
- Jeppesen J et al. *Detection of epileptic seizures with a modified heart
  rate variability algorithm based on Lorenz plot*. Seizure. 2015.
  https://doi.org/10.1016/j.seizure.2014.11.004

## Exact comparison

For every MIT--BIH record:

1. Reconstruct the unchanged UNSW sequence used by the current architecture.
2. Load the expert `atr` beat annotations.
3. Match UNSW and expert events monotonically within 75 ms.
4. At each common endpoint with sufficient history, calculate two independent
   trailing 100-RR windows:

```text
same matched ending heartbeat
             |
expert beats |<------- preceding 100 expert RR intervals -------|
UNSW beats   |<------- preceding 100 UNSW RR intervals ---------|
```

The two windows are not forced to have the same internal beats. A missed or
extra detection therefore changes the true input that the deployed branch
would have seen. Aligning windows by row number instead would hide these
errors.

Both streams use the exact implemented seven-RR causal median filter,
Poincare equations, elapsed-time slope, J1, and J2 calculations. No detector
timestamp is substituted and no RR interval is corrected.

Adjacent outputs overlap by 99 RR intervals. Thus, 104,450 paired windows do
not represent 104,450 independent biological observations. Results are
reported descriptively without treating each window as an independent sample.

## Continuous feature fidelity

| Feature | Pooled CCC | Median record CCC | Fraction of records >=0.8 |
|---|---:|---:|---:|
| Raw SD1 | 0.261 | 0.978 | 81.3% |
| Raw SD2 | 0.257 | 0.992 | 79.2% |
| CSI100 | 0.973 | 0.991 | 83.3% |
| Filtered SD1 | 0.988 | 0.986 | 85.4% |
| Filtered SD2 | 0.989 | 0.998 | 87.5% |
| Filtered ModCSI100 | 0.988 | 0.995 | 85.4% |
| HR slope100 | 0.975 | 0.995 | 89.6% |
| J1 | **0.990** | **0.994** | **91.7%** |
| J2 | **0.994** | **0.992** | **87.5%** |

The raw SD1/SD2 pooled CCC values are a warning, not a typographical error.
They are dominated by extreme all-record windows, particularly record 207.
The median record values are high, showing that a pooled statistic and a
typical-record statistic answer different questions.

Record 207 contains ventricular flutter and is officially described as
extremely difficult. Ventricular-flutter waves are not ordinary beat
fiducials; the project's beat loader intentionally excludes their non-beat
annotation symbol. Windows spanning these episodes can therefore contain a
long expert RR gap while a detector continues producing QRS-like events. HRV
during ventricular flutter is not standard normal-to-normal HRV. This stress
test is still retained because it demonstrates how an unstratified stream can
fail.

When all 101 expert beats supporting a window were labelled `N`, concordance
was stronger:

| Output | Normal-only pooled CCC | Normal-only median record CCC |
|---|---:|---:|
| J1 | 0.9988 | 0.9985 |
| J2 | 0.9984 | 0.9974 |

This supports accurate feature reproduction in stable normal morphology, but
does not solve arrhythmia, artifact, or seizure generalization.

![Feature concordance](../outputs/expert_unsw_hrv_fidelity_all48_v1/hrv_feature_concordance.png)

## Simulated threshold crossings

No clinical seizure threshold was invented. MIT--BIH has arrhythmia labels,
not seizure labels, and the later Jeppesen device calibration cannot be
reconstructed from this database.

Instead, each record's expert-derived feature distribution defined a fixed
90th, 95th, or 99th-percentile value. Expert and UNSW features were then asked
whether they crossed the exact same numerical threshold. This tests the
question:

> If a future patient-specific threshold were placed at a high part of the
> expert-derived distribution, how often would detector-derived HRV make the
> same window-level decision?

| Threshold | J1 sensitivity / PPV / F1 | J2 sensitivity / PPV / F1 |
|---:|---:|---:|
| 90th percentile | 0.889 / 0.890 / **0.889** | 0.885 / 0.900 / **0.892** |
| 95th percentile | 0.865 / 0.855 / **0.860** | 0.855 / 0.864 / **0.860** |
| 99th percentile | 0.746 / 0.672 / **0.707** | 0.717 / 0.787 / **0.750** |

The 95th-percentile result means that detector-derived J1 missed approximately
13.5% of expert high windows and produced false high windows in approximately
14.5% of its alerts. J2 behaved similarly. The most extreme 1% of windows were
less reliable. Therefore, high pooled CCC does not guarantee identical alarm
crossings.

Individual records also failed. At the 95th-percentile threshold:

- record 107: F1 0.093 for J1 and 0.088 for J2;
- record 104: F1 0.320 for J1 and 0.500 for J2;
- record 114: F1 0.344 for J1 and 0.412 for J2;
- record 231: F1 0.544 for J1 and 0.559 for J2.

The overall average therefore cannot justify a universal threshold.

![J1/J2 threshold agreement](../outputs/expert_unsw_hrv_fidelity_all48_v1/j1_j2_threshold_agreement.png)

## Why NeuroKit support must remain context

Restricting evaluation to windows with high UNSW--NeuroKit agreement increases
conditional concordance:

| Accepted window requirement | Window coverage | J1 CCC | J2 CCC |
|---|---:|---:|---:|
| None | 100% | 0.9904 | 0.9939 |
| At least 95/100 RR supported | 81.44% | 0.9987 | 0.9966 |
| All 100 RR supported | 66.20% | 0.9995 | 0.9997 |

This does not justify a hard gate. At the 95th-percentile threshold, requiring
all 100 intervals to be supported retained only 54.9% of expert-high J1
windows and 52.4% of expert-high J2 windows. If every rejected window were
treated as a non-event, effective F1 would fall to 0.665 for J1 and 0.640 for
J2.

Therefore:

- agreement can accompany J1/J2 as reliability context;
- low agreement may trigger review, abstention, or a downstream missingness
  indicator;
- low agreement must not automatically set J1/J2 to zero;
- low agreement must not automatically mean artifact;
- unsupported high-HRV windows must not be silently erased.

## What “close enough” means here

### Supported conclusion

For long-run continuous J1/J2 feature extraction, unchanged UNSW timestamps
are sufficiently close to expert annotations to justify continuing the
architecture into patient-specific testing. The normal-only result is
particularly strong.

### Unsupported conclusion

This audit does not establish that the branch is close enough for a clinical
or seizure alarm because:

- no seizure labels are present;
- no validated patient threshold is present;
- extreme-threshold agreement is materially lower than continuous CCC;
- several records fail badly;
- MIT--BIH contains ectopy, paced rhythm, bundle-branch block, flutter, noise,
  and non-beat ventricular-flutter annotations;
- all-beat RRI is not the same as clinically edited normal-to-normal HRV;
- adjacent 100-RR windows are highly dependent.

## Required patient-specific gate

The next validation must use chronologically separated patient ECG with:

1. manually adjudicated QRS/R fiducials in representative clean, noisy,
   interictal, preictal, and ictal segments;
2. expert seizure onset/offset labels independent of ECG;
3. expert-derived and UNSW-derived J1/J2 calculated with frozen code;
4. threshold or model calibration on an earlier training period only;
5. threshold-crossing agreement, event sensitivity, false alarms per hour,
   latency, and abstention coverage on a later held-out period;
6. separate reporting for sinus rhythm, ectopy/conduction changes, artifact,
   and detector disagreement.

Only that experiment can answer whether the residual feature errors change
the patient's actual seizure decisions.

## Reproducible outputs

- Implementation: `scripts/benchmark_expert_unsw_hrv_fidelity.py`
- Agreement utilities: `src/ecg_cascade/hrv_fidelity.py`
- Paired windows: `outputs/expert_unsw_hrv_fidelity_all48_v1/paired_expert_unsw_hrv_windows.csv.gz`
- Pooled feature table: `outputs/expert_unsw_hrv_fidelity_all48_v1/pooled_feature_agreement.csv`
- Per-record feature table: `outputs/expert_unsw_hrv_fidelity_all48_v1/record_feature_agreement.csv`
- Threshold tables: `outputs/expert_unsw_hrv_fidelity_all48_v1/*threshold_agreement.csv`
- Plots: `outputs/expert_unsw_hrv_fidelity_all48_v1/*.png`
