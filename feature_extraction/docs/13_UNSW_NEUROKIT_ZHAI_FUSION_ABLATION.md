# UNSW--NeuroKit--Zhai timestamp-fusion ablation

## Question

Can the detector strengths be combined as follows?

1. UNSW/Khamis determines that a heartbeat exists.
2. A uniquely associated NeuroKit detection supplies the timestamp.
3. If NeuroKit is unavailable or ambiguous, a uniquely associated Zhai
   template-correlation fiducial supplies the timestamp.
4. Otherwise the original UNSW QRS-event timestamp is retained and marked as
   uncertain.

The experiment does **not** use raw-amplitude `argmax`. It does not add or
remove UNSW events. The final event count is therefore identical to the UNSW
event count. What changes is the timestamp assigned to each event.

## Evidence boundary

Ho et al. published the use of UNSW as a primary detector and NeuroKit as a
secondary detector for estimating RR reliability. An RR interval was
considered reliable when both bounding primary detections received secondary
support. Ho et al. did not replace UNSW timestamps with NeuroKit timestamps.

- Ho et al., *Automated RR Interval Detection and Quality Assessment in
  Telehealth Electrocardiograms*, Computing in Cardiology 2024,
  DOI: [10.22489/CinC.2024.084](https://doi.org/10.22489/CinC.2024.084).

Zhai et al. published an independent detector/localizer that aligns the whole
QRS waveform with a representative template by maximum absolute normalized
cross-correlation. The current code is a paper reimplementation because no
official author implementation was located.

- Zhai et al., *Precise detection and localization of R-peaks from ECG
  signals*, 2023,
  DOI: [10.3934/mbe.2023848](https://doi.org/10.3934/mbe.2023848).

The exact NeuroKit-then-Zhai timestamp-substitution hierarchy is therefore an
**experimental project ablation**, not a published architecture.

## Implemented rule

All associations are unique in both directions:

- exactly one independent-detector event must lie within the tolerance of an
  UNSW event; and
- that independent event must lie within the tolerance of exactly one UNSW
  event.

This prevents one candidate from being reused for two rapid adjacent beats.
The experiment tested association tolerances of 50, 75, 100, and 150 ms and
both existing NeuroKit configurations:

- 250 ms minimum delay with an inclusive comparison;
- the unchanged 300 ms strict NeuroKit baseline.

No Zhai correlation cutoff was invented. Absolute correlation is exported as
context because the paper does not establish a universal deployment cutoff.

## Dataset and evaluation

- All 48 MIT--BIH Arrhythmia Database records.
- First channel of every record.
- Sampling rate: 360 Hz; one sample is 2.78 ms.
- Expert comparison: monotone one-to-one matching within 75 ms.
- Timing statistics are calculated only for matched true positives; FP and FN
  remain separate.
- RR error is evaluated only when both detected events and both expert events
  are consecutive.

## Pooled result

The best hierarchical configuration by RR MAE used NeuroKit 300-ms strict and
a 50-ms UNSW association tolerance.

| Method | TP | FP | FN | F1 | Mean timing error | P95 timing error | RR MAE | RR P95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| UNSW initial | 109,244 | 350 | 250 | 0.997261 | 8.705 ms | 27.778 ms | 3.971 ms | 19.444 ms |
| NeuroKit alone, 300 ms | 107,677 | 2,503 | 1,817 | 0.980334 | 3.574 ms | 19.444 ms | 3.307 ms | 13.889 ms |
| UNSW event + NeuroKit timestamp | 109,231 | 363 | 263 | 0.997143 | 3.048 ms | 16.667 ms | 3.382 ms | 16.667 ms |
| UNSW event + Zhai timestamp | 109,215 | 379 | 279 | 0.996997 | 5.234 ms | 36.111 ms | 3.864 ms | 27.778 ms |
| NeuroKit then Zhai hierarchy | 109,223 | 371 | 271 | 0.997070 | 3.009 ms | 16.667 ms | 3.382 ms | 16.667 ms |

### What improved

Relative to unchanged UNSW, the full hierarchy reduced:

- mean absolute timing error from 8.705 to 3.009 ms;
- P95 timing error from 27.778 to 16.667 ms;
- consecutive RR MAE from 3.971 to 3.382 ms.

### What deteriorated

The hierarchy lost a net 21 expert matches relative to UNSW:

- TP decreased from 109,244 to 109,223;
- FP increased from 350 to 371;
- FN increased from 250 to 271;
- F1 decreased from 0.997261 to 0.997070.

This does not mean that 21 physical heartbeats were deleted; UNSW still owns
exactly the same 109,594 events. It means that substituted timestamps moved 21
more events outside the 75-ms expert-matching range than they moved inside it.
Of the individual expert matches lost before considering gains, 21 were
NeuroKit substitutions and eight were Zhai substitutions. NeuroKit
substitutions gained eight different matches, producing the net loss of 21.

No tested hierarchy strictly dominated UNSW simultaneously in F1, RR MAE, and
P95 timing error.

## Detector-source transitions are the main HRV danger

For the best hierarchy:

- 105,758 timestamps came from NeuroKit;
- 2,850 came from Zhai;
- 986 remained UNSW timestamps;
- 2,843 adjacent intervals crossed from one timestamp source to another.

Among expert-evaluable intervals:

| RR subset | Evaluable RR | RR MAE |
|---|---:|---:|
| Same timestamp source at both boundaries | 106,375 | 2.916 ms |
| Source transition at an RR boundary | 2,457 | 23.535 ms |
| All intervals pooled | 108,832 | 3.382 ms |

Thus, the pooled improvement hides a nearly eight-fold increase in error at
source-transition boundaries. These intervals must not be passed into HRV as
ordinary measurements.

The NeuroKit-only substitution showed the same phenomenon:

- same-source RR MAE: 2.944 ms;
- source-transition RR MAE: 22.915 ms.

## Record dependence

The pooled result is not universal.

- NeuroKit-only substitution improved record-level RR MAE in 31 records and
  worsened it in 17.
- The complete hierarchy improved RR MAE in 33 records and worsened it in 15.
- Mean timing error improved in 43 records but worsened in five.
- F1 improved in only two hierarchy records, deteriorated in nine, and was
  unchanged in 37.

Examples:

| Record | Method | F1 | Mean timing error | P95 timing error | RR MAE |
|---|---|---:|---:|---:|---:|
| 108 | UNSW | 0.996314 | 13.059 ms | 25.000 ms | 8.379 ms |
| 108 | hierarchy | 0.996314 | 15.462 ms | 52.778 ms | 11.184 ms |
| 113 | UNSW | 0.999443 | 4.993 ms | 5.556 ms | 0.919 ms |
| 113 | hierarchy | 0.999443 | 0.136 ms | 0 ms | 0.251 ms |
| 207 | UNSW | 0.962387 | 6.047 ms | 11.111 ms | 3.635 ms |
| 207 | hierarchy | 0.962905 | 3.312 ms | 5.556 ms | 3.065 ms |
| 222 | UNSW | 0.998186 | 4.157 ms | 11.111 ms | 5.433 ms |
| 222 | hierarchy | 0.998186 | 0.809 ms | 2.778 ms | 1.240 ms |
| 231 | UNSW | 0.996511 | 9.474 ms | 13.889 ms | 1.525 ms |
| 231 | hierarchy | 0.996511 | 1.183 ms | 2.778 ms | 1.235 ms |

Record 108 is a direct counterexample to declaring NeuroKit the universal
timing owner. Records 117, 200, and 233 also became materially worse. In record
200, RR MAE increased from 6.757 ms with UNSW to 22.207 ms with the hierarchy.

## Decision

The full automatic hierarchy is **not accepted as the single RR timestamp
owner**.

The evidence supports the following narrower interpretation:

1. Retain unchanged UNSW timestamps as the full-coverage control stream.
2. Retain NeuroKit and Zhai timestamps as independent timing candidates.
3. Export unique-association and source-transition fields.
4. Treat RR intervals crossing timestamp sources as unreliable for HRV.
5. Do not claim that agreement means artifact-free or physiologically correct.
6. For the eventual patient, determine whether NeuroKit relocalization is
   beneficial on a manually annotated calibration subset containing that
   patient's actual lead, rhythms, and morphologies.
7. Validate the selected rule on held-out patient time blocks. Selecting the
   50-ms tolerance on MIT--BIH and reporting it on the same records is
   development-set selection, not independent validation.

Zhai was valuable in specific records, especially 207 and 222, but as an
automatic global fallback it added complexity without a meaningful pooled RR
advantage over NeuroKit-only substitution and further reduced F1.

## Reproduction

```powershell
cd C:\Users\Salam\Documents\ECGdetector\feature_extraction
python scripts\benchmark_unsw_neurokit_zhai_fusion.py `
  --database-dir ..\Datasets\mit-bih-arrhythmia-database-1.0\mit-bih-arrhythmia-database-1.0.0 `
  --output-dir outputs\unsw_neurokit_zhai_fusion_all48_v1
```

Primary artifacts:

- `pooled_metrics.csv`
- `pooled_metrics_with_transition_audit.csv`
- `record_metrics.csv`
- `five_difficult_records.csv`
- `fusion_event_decisions.csv.gz`
- `pooled_fusion_comparison.png`
- `five_difficult_fusion_errors.png`
- `fusion_timestamp_source_usage.png`
- `record_rr_mae_change_from_unsw.png`
- `source_transition_rr_error.png`
- `decision_summary.json`
