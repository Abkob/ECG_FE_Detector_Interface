# Matched-lead delineation gate v1

## Purpose

This gate tests whether a delineator can recover manual P/QRS/T landmarks
before delineated morphology enters the runtime feature branch. It does not
test R-peak detection, seizure accuracy, or incremental value beyond RR/HRV.

The published Varon five-beat implementation remains frozen as the replication
control. No delineator was promoted into that runtime branch by this study.

## Lead-controlled scope

- LUDB: all 200 records, manual lead-II annotations, 500 Hz.
- QTDB: 12 records whose manually annotated channel 0 is MLII, 250 Hz.
- QTDB reference: `q1c` second-pass manual landmarks on selected beats.
- QTDB delineator context: the full expert `atr` beat stream; only manually
  delineated beats are scored.
- R/QRS reference anchors are supplied to the delineator to isolate waveform
  delineation from beat-detection errors.
- One-to-one landmark matching tolerance: 150 ms. Timing-error distributions
  are reported inside this audit tolerance; the tolerance is not a clinical
  accuracy claim.

## Methods

- NeuroKit2 0.2.13 DWT.
- NeuroKit2 0.2.13 CWT.

The CWT method returns unequal-length landmark arrays. NeuroKit's optional
`check=True` path therefore raises `ValueError: All arrays must be of the same
length`. The audit runs CWT with `check=False`, preserves missing landmarks,
and associates variable-length outputs with reference R anchors by physiological
phase. This is an audited software limitation, not a successful consistency
check.

## Main results

### QRS boundaries

| Dataset | Method | Landmark | Sensitivity | PPV | Median absolute error | P95 absolute error |
|---|---|---|---:|---:|---:|---:|
| LUDB II | DWT | QRS onset | 94.3% | 98.3% | 34 ms | 124 ms |
| LUDB II | DWT | QRS offset | 90.2% | 95.8% | 10 ms | 66 ms |
| QTDB MLII | DWT | QRS onset | 87.7% | 98.3% | 60 ms | 128 ms |
| QTDB MLII | DWT | QRS offset | 73.6% | 100% | 8 ms | 44 ms |
| LUDB II | CWT | QRS onset | 65.8% | 100% | 18 ms | 62 ms |
| LUDB II | CWT | QRS offset | 34.0% | 100% | 8 ms | 32 ms |
| QTDB MLII | CWT | QRS onset | 24.9% | 100% | 28 ms | 64 ms |
| QTDB MLII | CWT | QRS offset | 34.6% | 100% | 8 ms | 32 ms |

CWT is often conditionally precise but omits too many boundaries. DWT has
better coverage, but QRS onset is systematically early and has large tail
errors. Neither method passes a complete-QRS acceptance gate.

### P and T landmarks

DWT P-wave sensitivity and PPV are approximately 99% on LUDB and 100% on the
matched QTDB subset. Median errors are small, though P-wave boundary tails
remain non-trivial on LUDB.

T-wave results are less reliable. DWT T-offset sensitivity is 86.4% on LUDB
and 69.5% on QTDB. QTDB contains only 31 manually marked T onsets in this
matched-lead subset, so it cannot support a stable T-onset conclusion by
itself.

## Polarity finding

LUDB lead II contains 15 negative- or mixed-dominant QRS records under the
manual-interval polarity audit. DWT QRS-onset performance changes materially:

| Polarity group | Sensitivity | PPV | Median absolute error | P95 absolute error |
|---|---:|---:|---:|---:|
| Positive-dominant | 95.6% | 99.0% | 28 ms | 122 ms |
| Negative/mixed-dominant | 75.8% | 87.9% | 72 ms | 133.5 ms |

This confirms that a lead-aware/polarity-preserving evaluation is required.
Signals must not be silently flipped to make R positive.

## Test of the current fixed Varon capture

Manual QRS boundaries permit a direct characterization of the fixed 60 ms
before/60 ms after R capture:

| Dataset | Complete manual QRS beats | Entire QRS inside fixed 120 ms |
|---|---:|---:|
| LUDB lead II | 1,817 | 73.7% |
| QTDB MLII | 462 | 62.8% |
| Combined | 2,279 | 71.5% |

The combined median manual QRS duration is 92 ms, but the P95 is 140 ms.
Consequently, the Varon window remains a faithful fixed-window replication
feature, not a universal complete-QRS representation.

## Decision

1. Keep Varon unchanged as the control lane.
2. Do not integrate NeuroKit CWT into the morphology branch.
3. Retain DWT only as a documented baseline; do not use it yet for QRS
   duration, QT, or the structured morphology beat object.
4. Benchmark ECGdeli and HiDC-QRS next, using the identical loader, lead scope,
   reference anchors, metrics, and polarity strata.
5. Require acceptable QRS onset and offset performance on both datasets before
   implementing variable-boundary QRS features.
6. Evaluate a separate full P/T semantic delineator if the QRS-only challenger
   does not supply those landmarks.

## Reproduction

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_delineation_gate.py `
  --methods dwt cwt `
  --output outputs\delineation_gate_matched_lead_v1

.\.venv\Scripts\python.exe scripts\analyze_varon_fixed_qrs_capture.py `
  --output outputs\delineation_gate_matched_lead_v1
```

Machine-readable outputs are stored in
`outputs/delineation_gate_matched_lead_v1/`.
