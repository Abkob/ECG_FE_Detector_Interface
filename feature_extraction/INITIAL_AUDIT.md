# Initial implementation audit: shared R peaks and RR/HRV

Date: 2026-07-30

## What has actually been validated

- The EDF loader reads physical samples from a named channel and preserves the
  EDF sampling rate and physical unit.
- Unit tests verify the 75 ms one-to-one matching calculation, causal seven-RR
  median filter, elapsed-time least-squares slope, Poincare axes, and the rule
  that Jeppesen features remain undefined until 100 RR intervals exist.
- The code executes on two installed datasets at 256 and 512 Hz and produces
  inspectable CSV, JSON, and PNG outputs.

These checks validate software behavior. They do **not** establish R-peak
accuracy or seizure-discrimination performance.

## Real-data results

| Recording and block | Pan peaks | NeuroKit peaks | 75 ms agreement F1 | Main observation |
|---|---:|---:|---:|---|
| `chb04_28`, 0-600 s | 812 | 812 | 0.373 | Counts agree, but reported fiducial locations frequently differ by more than 75 ms. |
| `chb04_28`, 1500-2100 s, seizure 1679-1781 s | 748 | 659 | 0.381 | The ictal and postictal ECG contains severe high-amplitude contamination; both RR sequences become implausible. |
| `PN06-1`, 0-600 s | 741 | 740 | 0.115 | Counts again agree closely, but the two implementations report different points within or near each complex. |

All fully formed 100-RR windows in these tests contain at least one 75 ms
detector disagreement. Therefore neither detector can currently be declared a
validated primary detector for this patient data.

The tolerance audit supports two separate phenomena:

| Block | F1 at 75 ms | F1 at 100 ms | F1 at 125 ms | F1 at 150 ms |
|---|---:|---:|---:|---:|
| `chb04_28`, 0-600 s | 0.373 | 0.983 | 0.990 | 0.990 |
| `chb04_28`, seizure-containing block | 0.381 | 0.706 | 0.733 | 0.741 |
| `PN06-1`, 0-600 s | 0.115 | 0.506 | 0.994 | 0.999 |

This does not justify increasing the official tolerance. Kristof et al. chose
75 ms to distinguish detections located on the same QRS complex from detections
on other waves. The improvement in quiet blocks at larger tolerances suggests
systematic localization differences; the persistent seizure-block failure
shows additional missed/false detections during contamination.

## Critical implications

1. Similar beat counts are not enough for HRV. A changing fiducial position
   inside a beat changes consecutive RR intervals and therefore changes SD1,
   SD2, CSI, ModCSI, and both final continuous indices.
2. The very large seizure-block `J2` values cannot presently be interpreted as
   autonomic seizure physiology. They co-occur with visually obvious signal
   contamination and unstable RR intervals.
3. The 2024 benchmark found NeuroKit and UNSW strongest overall, but also found
   substantial degradation on low-quality ECG. It did not validate either
   detector on these peri-ictal EDF recordings.
4. The Jeppesen phase-3 study used Pan-Tompkins, but naming the same historical
   algorithm does not prove that NeuroKit2's implementation reproduces the
   authors' Android implementation on these leads.

## Required validation gate before accepting RR/HRV values

Create a manually annotated R-peak set containing, at minimum:

- clean interictal ECG from each acquisition system;
- preictal and early ictal ECG before gross contamination;
- contaminated ictal/postictal ECG;
- morphology and polarity changes;
- representative detector-agreement and detector-disagreement intervals.

Against those annotations, report sensitivity, positive predictive value, F1,
and timing error at 75 ms for each detector. Only then select the primary peak
sequence or a published refinement method. Do not tune the matching tolerance
on the evaluation annotations.

## Recommended implementation order from this result

1. Add a small annotation/review workflow and establish R-peak ground truth.
2. Implement the non-learned signal-quality measurements, because the first
   seizure block already shows that artifact can dominate HRV values.
3. Re-run and accept/reject the RR/HRV branch using the validated peak source.
4. Implement QRS morphology only on beats passing the required measurement
   validity metadata.
5. Implement conduction/repolarization last because P onset and T end require
   stricter fiducial validation than RR intervals.

## Primary sources

- Jeppesen et al. phase 3: <https://doi.org/10.1016/j.ebiom.2025.105952>
- Kristof et al. detector benchmark: <https://doi.org/10.1371/journal.pdig.0000538>
- NeuroKit2: <https://doi.org/10.3758/s13428-020-01516-y>
- Pan and Tompkins: <https://doi.org/10.1109/TBME.1985.325532>

