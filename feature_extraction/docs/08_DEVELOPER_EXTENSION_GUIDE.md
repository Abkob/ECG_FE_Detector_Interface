# Developer extension guide for new ECG branches

## 1. Objective

This guide defines how to extend the package from the current RR/HRV branch to
signal quality, QRS morphology, and conduction/repolarization measurements
without turning the project into an untraceable collection of features.

The planned system is feasible as a staged research architecture only if every
branch has a stable contract, missingness is explicit, time alignment is
causal, and feature/model choices are evaluated chronologically. The presence
of several branches does not itself improve accuracy or reduce overfitting.

## 2. Current dependency tree

```text
EDF channel
    |
    +-- raw physical ECG ---------------------------> signal-quality branch (next)
    |
    +-- R-peak candidates + agreement
            |
            +-- RR/HRV continuous measurements ----> implemented
            |
            +-- beat segmentation ------------------> QRS morphology (future)
            |
            +-- P/QRS/T delineation ----------------> conduction/QT (future)
```

The branches are not fully independent. RR/HRV depends on peaks; morphology
depends on beat alignment; PR/QRS/QT depend on stricter wave delineation. The
quality branch should observe raw signal and intermediate measurement failures
but should not silently rewrite the other branches.

## 3. Mandatory branch output contract

Every branch should output a timestamped table containing four categories:

| Category | Purpose | Example |
|---|---|---|
| Measurements | Published or explicitly defined values. | RR, SD1, QRS similarity, QTc. |
| Measurement validity/context | Evidence about whether the value is credible. | Detector disagreement count, delineation confidence, clipping fraction. |
| Availability | Whether enough history/input exists. | 100 RR accumulated, T end found, channel present. |
| Provenance | Method, version, window, units, and source. | `method=neurokit`, `window_beats=100`. |

A branch should not return only `seizure/not seizure`. Its job is to measure one
lens of the ECG. A later temporal/fusion model may learn how those measurements
relate to annotated seizures.

### Recommended in-memory shape

```python
@dataclass(frozen=True)
class BranchResult:
    name: str
    features: pd.DataFrame
    metadata: dict[str, object]
    diagnostics: dict[str, object]
```

This is a proposed software interface, not a published clinical algorithm. It
formalizes the categories already returned by the current pipeline. Do not add
it until two branches make the shared abstraction useful.

## 4. Naming and units

Column names must encode enough information to prevent accidental mixing:

- append `_ms`, `_s`, `_hz`, `_bpm`, or `_bpm_per_s` to dimensional values;
- append window length or method when scientifically material, such as
  `csi100` or `rr_median7_ms`;
- use `_defined` for mathematical availability and `_valid` only for a stated
  validity rule;
- use `_score` only when the scale and direction are documented;
- never call an arbitrary continuous value `confidence` unless it is calibrated
  against an appropriate reference outcome.

Do not standardize or normalize inside extraction code. Scaling parameters must
be learned from the training period only and belong to the later model artifact.

## 5. Time alignment contract

Every row must specify what its timestamp represents:

- an RR row is timestamped at the ending R peak;
- a beat-shape row can be timestamped at its central R peak;
- a fixed signal-quality window should be timestamped at its end if causal;
- a QT row should be timestamped at the corresponding beat's R peak or T end,
  but the choice must be fixed and documented.

For later fusion at time \(t\), use only branch information whose supporting
samples end at or before \(t\). Do not center a window on \(t\) during model
training if future samples would be unavailable in deployment.

Different branch histories are acceptable. A 100-RR autonomic feature can be
joined with a five-beat morphology feature using the latest causally available
row from each branch, plus age/staleness and availability fields. The exact join
policy is an implementation decision that must be tested and frozen before
evaluation.

## 6. Missingness rules

Use missing values for unavailable measurements and separate flags to explain
why. Do not encode unavailable as zero.

Recommended reasons:

| Code | Meaning |
|---|---|
| `insufficient_history` | Window has not accumulated enough causal input. |
| `channel_unavailable` | Required ECG lead is absent. |
| `peak_invalid` | Beat time cannot be accepted. |
| `delineation_failed` | P/QRS/T landmark not found under the fixed method. |
| `signal_unscorable` | Published/validated quality rule says the signal cannot support the measurement. |
| `nonfinite_input` | Numeric corruption, not physiological missingness. |

The downstream model can receive a measurement, an availability Boolean, and
an age/quality value. It should never have to infer missingness from a magic
number.

## 7. Adding a module

For a new branch:

1. Create one focused module under `src/ecg_cascade`, for example
   `signal_quality.py`.
2. Implement small pure functions for published equations before writing the
   orchestrator.
3. Add typed/frozen result structures only where arrays and metadata must travel
   together.
4. Add the branch to `pipeline.py` only after unit tests pass.
5. Add explicit CLI options and output files; never alter an existing schema
   silently.
6. Record method names, constants, package versions, and threshold state in
   metadata.
7. Generate a diagnostic that exposes intermediate values and failures.
8. Update the literature traceability table with support, contradiction, and
   target-data gaps.

## 8. Evidence intake checklist

Before coding a paper's method, record:

| Question | Required answer |
|---|---|
| Population/domain | Patients, lead configuration, sampling rate, duration, inpatient/ambulatory, and seizure types. |
| Input definition | Raw ECG, cleaned ECG, R peaks, NN intervals, beat templates, or delineated waves. |
| Causality | Does any filter/window use future samples? |
| Exact equations | All units, constants, edge handling, and window alignment. |
| Labels/reference | Manual beats, cardiologist morphology, video-EEG seizure onset, artifact labels, etc. |
| Split | Patient-wise, episode-wise, chronological, or random windows. |
| Metrics | Measurement error versus classification metrics; confidence intervals and denominators. |
| Code availability | Exact author code, third-party reproduction, or only prose. |
| Failure cases | Noise, ectopy, rhythm, morphology, missing leads, device changes. |
| Applicability | Which step transfers to this patient and which requires revalidation. |

If an equation or threshold cannot be recovered, implement the continuous
upstream measurements only or mark the method blocked. Do not fill in missing
steps from intuition.

## 9. Signal-quality branch protocol

This should be the next branch because the current seizure block contains
artifact severe enough to corrupt peak timing and RR/HRV. Implementation should
proceed in layers:

1. Raw-signal measurements with published definitions, units, and window
   lengths (for example clipping/saturation, flatline, amplitude/RMS, baseline
   behavior, and high-frequency contamination where supported).
2. Beat-based measurements only after peak validation (template agreement,
   RR plausibility, detector agreement).
3. Output continuous values and availability rather than one unexamined binary
   artifact label.
4. If a binary artifact model is later trained, retain its calibrated
   probability and the underlying measurements.
5. Validate by artifact type and ECG measurability, not only overall accuracy.

The phase-3 Jeppesen system's 2-second RMS/20 mV noise rule is a published
candidate but is device-unit dependent and acted as a hard suppression rule.
It cannot be copied into microvolt EDF data without verifying units, electrode
system, and target behavior.

## 10. QRS-morphology branch protocol

Morphology must not be built on unvalidated beat locations. Once peaks pass the
measurement gate:

1. Extract causally defined beat windows with an explicit pre/post-R duration.
2. Decide whether amplitude normalization is appropriate; preserve raw-scale
   features separately if clinically meaningful.
3. Implement the published five-beat morphology/PCA construction as an
   independently testable measurement module.
4. Report eigenvalues or derived indices plus beat availability, clipping,
   alignment displacement, and peak validity.
5. Validate against morphology/rhythm strata and visually audited beats before
   evaluating seizure association.

The five-beat window and the 100-RR window do not need the same duration. They
must share an explicit causal timestamp and cannot be treated as independent
samples when they overlap heavily.

## 11. Conduction and repolarization branch protocol

PR, QRS duration, QT, QTc, and related values require P onset, QRS onset/offset,
and T end—not merely R peaks. Their acceptance gate is therefore stricter:

1. Select a delineator from comparative validation literature appropriate to
   the lead and sampling rate.
2. Validate every landmark against manual annotations on this patient's clean
   and difficult morphologies.
3. Export raw intervals first: PR ms, QRS duration ms, QT ms, RR ms.
4. Compute correction formulas (for example QTc variants) as separate named
   columns; do not hide the correction equation.
5. Carry heart-rate range and rhythm context because QT correction formula bias
   depends on rate and population.
6. Mark a beat unavailable when the T end or P onset is not defensible instead
   of forcing a value.

This branch must be implemented after peak and signal-quality validation,
because its endpoints are more fragile than RR intervals.

## 12. Testing standard for a new feature

Every new mathematical feature needs:

- a hand-computed fixture with a known answer;
- edge tests for minimum history, zero denominator, non-finite input, and
  missing landmarks;
- a causality test proving future samples are not read;
- a units test or invariant;
- a schema test;
- at least one real-data diagnostic;
- a comparison to manual/reference output on target data;
- a statement of what the test does not prove.

For learned components, additionally require chronological train/validation/
test separation and persist the training-only normalization/selection objects.

## 13. Adding a learned model

Feature extraction and seizure modeling should remain separate packages or
modules. A model artifact must contain:

- exact feature-column list and order;
- branch schema versions;
- scaling/imputation learned from training data;
- feature selection learned from training data;
- model weights and hyperparameters;
- patient identifier/scope;
- chronological training interval;
- label protocol and exclusion zones;
- calibration method;
- decision threshold and its source;
- software versions and random seeds.

For the one-patient design, hold out whole later seizures/recordings. Randomly
splitting maximum-overlap rows is unacceptable because nearly identical windows
would cross the split.

## 14. Fusion guardrails

Do not assume that concatenating more values reduces overfitting. A later fusion
model should begin with a documented baseline and add branches one at a time.
For each addition, report:

- change in episode-held-out performance and uncertainty;
- change in false alarms per 24 hours;
- change in unavailable monitoring time;
- feature correlation/redundancy;
- ablation performance without that branch;
- behavior under artifact and acquisition changes.

If two branches are calculated from the same R peaks, their errors are
correlated. Detector disagreement and signal quality should accompany them as
context; neither turns corrupted inputs into independent evidence.

## 15. Versioning and schema changes

Add a future `schema_version` to every branch's metadata before external model
training begins. Changes requiring a version increment include:

- renamed/removed columns;
- different units;
- different window alignment or causality;
- a changed detector/library version that changes outputs;
- a changed formula, filter, or missingness rule;
- altered channel selection.

Never train a model from outputs generated by mixed schemas unless an explicit
migration verifies equivalence.

## 16. Pre-merge checklist

- [ ] Every equation has a primary citation or is labeled an implementation
  decision.
- [ ] Supporting and contradictory evidence are recorded.
- [ ] Units appear in code/schema/documentation.
- [ ] Window support and timestamp semantics are explicit.
- [ ] Missing/unavailable is distinct from zero.
- [ ] The branch emits measurement context and provenance.
- [ ] Pure-equation unit tests pass.
- [ ] Real EDF diagnostics were visually reviewed.
- [ ] Manual/reference measurement validity is reported or marked pending.
- [ ] No seizure threshold was tuned on final evaluation events.
- [ ] No random split leaks overlapping rows across partitions.
- [ ] Documentation and literature traceability were updated.

