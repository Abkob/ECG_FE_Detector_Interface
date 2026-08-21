# Command-line interface and output artifacts

> Version note: commands and artifacts in this chapter describe v0.1. The
> active v0.2 CLI and outputs are documented in the package README and
> [09_COMPLETE_RR_HRV_ARCHITECTURE.md](09_COMPLETE_RR_HRV_ARCHITECTURE.md).

## 1. Purpose

`src/ecg_cascade/cli.py` is the reproducible entry point for inspecting EDF
headers and running the current R-peak/RR-HRV branch. It translates command-line
arguments into a call to `pipeline.run_rr_hrv_segment`, saves every important
intermediate result, and prints a compact JSON summary.

The command line intentionally exposes a narrow set of options. Scientific
constants such as the 75 ms match tolerance, 100-RR window, and seven-RR median
width are recorded in metadata but are not casual tuning switches.

## 2. Installation

From `feature_extraction`:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[test]"
```

`-e` installs the local source in editable mode, so a code change is used by
the next command without rebuilding a wheel. The `[test]` extra installs
pytest.

The versions pinned by `pyproject.toml` and verified in the current virtual
environment are:

| Package | Version | Use |
|---|---:|---|
| Python | 3.12 or newer required | Runtime and type syntax. |
| NeuroKit2 | 0.2.13 | ECG cleaning and both R-peak implementations. |
| NumPy | 2.4.2 | Numeric arrays, matching, and mathematical operations. |
| pandas | 2.3.3 | Tabular feature construction and CSV output. |
| SciPy | 1.17.0 | NeuroKit2 numerical dependency. |
| pyEDFlib | 0.1.42 | EDF header and physical-signal reading. |
| Matplotlib | 3.10.8 | Noninteractive diagnostic PNG generation. |
| pytest | 9.1.1 | Automated tests. |

Exact pinning improves reproducibility, but it does not freeze operating-system
libraries or guarantee identical floating-point results on every platform.

## 3. Command: `inspect-edf`

```powershell
.venv\Scripts\ecg-features inspect-edf `
  --edf "C:\path\recording.edf"
```

### Argument

| Argument | Required | Meaning |
|---|---|---|
| `--edf` | Yes | Existing EDF/EDF+ file to inspect. |

### Result

The command prints a JSON object containing the resolved file path, file
duration, number of signals, and a record for every channel. Channel records
contain the channel index, exact EDF label, sampling rate, sample count,
physical dimension, and physical range.

Run this before extraction. Do not assume that channel number or label is
consistent across datasets.

## 4. Command: `run-rr-hrv`

```powershell
.venv\Scripts\ecg-features run-rr-hrv `
  --edf "C:\path\recording.edf" `
  --channel ECG `
  --start-s 1500 `
  --duration-s 600 `
  --primary-method pantompkins1985 `
  --event 1679:1781 `
  --output-dir "C:\path\results"
```

### Arguments

| Argument | Required/default | Meaning and restrictions |
|---|---|---|
| `--edf` | Required | Input EDF path. The source file is read but not modified. |
| `--channel` | Optional | Exact channel label, compared case-insensitively. If omitted, automatic selection succeeds only for one unambiguous ECG/EKG-labeled channel. |
| `--start-s` | Default `0.0` | Segment start relative to EDF start. Must be nonnegative and earlier than the selected channel's end. |
| `--duration-s` | Default `600.0` | Requested duration. Must be positive. Reading is clipped at channel end. |
| `--primary-method` | Default `pantompkins1985` | Which of the two candidate sequences creates RR intervals: `pantompkins1985` or `neurokit`. The other becomes the comparator. |
| `--event START:END` | Optional, repeatable | EDF-relative interval used only to shade the diagnostic plot. Must satisfy `0 <= START < END`. |
| `--output-dir` | Required | Directory created if absent. Files with standard names are overwritten in that directory. |

### Event isolation

`--event` does not label feature rows, crop the signal, tune a threshold, or
alter any calculation. `cli.py` adds the interval to metadata under
`events_for_plot_only_s` and sends it to `plotting.py` for shading.

## 5. Pipeline orchestration

`pipeline.run_rr_hrv_segment` executes in this order:

1. Load the named ECG channel and segment.
2. Run NeuroKit2's `pantompkins1985` cleaning/detection path.
3. Run NeuroKit2's `neurokit` cleaning/detection path.
4. Select primary and comparator according to `--primary-method`.
5. Compare candidates using the fixed 75 ms one-to-one rule.
6. Compute RR/HRV measurements from the primary sequence.
7. Return the signal, feature table, both peak arrays and cleaned signals,
   agreement object, and metadata.

The CLI then serializes the results and creates the plot. The cleaned signals
are retained in memory for diagnostics but are not currently written to disk.

## 6. Output directory contract

Every successful `run-rr-hrv` command writes six files:

| File | Format | Purpose |
|---|---|---|
| `metadata.json` | JSON | Provenance, channel header, processing constants, package versions, and row counts. |
| `rpeaks_pan_tompkins.csv` | CSV | Pan-Tompkins candidate locations. |
| `rpeaks_neurokit.csv` | CSV | NeuroKit candidate locations. |
| `rpeak_agreement.json` | JSON | Global agreement metrics and every unmatched candidate. |
| `rr_hrv_features.csv` | CSV | One row per RR interval, with continuous 100-RR outputs when available. |
| `rr_hrv_diagnostic.png` | PNG | Visual inspection of ECG/peaks, RR-derived HR, and `J1`/`J2`. |

The output directory is not transactional: an exception after some writes can
leave a partial result directory. Treat a run as complete only when all six
files exist and metadata can be parsed.

## 7. Peak CSV schema

Both peak CSVs have the same columns:

| Column | Meaning |
|---|---|
| `sample_in_segment` | Zero-based candidate index relative to the loaded array. |
| `time_s` | EDF-relative time: `segment_start_s + sample_in_segment / fs`. |

The amplitude is not included. To retrieve it, index the original loaded signal
using `sample_in_segment`. The candidates in these two files are independent;
row 10 in one file is not guaranteed to match row 10 in the other.

## 8. `rpeak_agreement.json` schema

| Field | Type | Meaning |
|---|---|---|
| `agreement_f1` | number or `null` | `2 * matched_count / (primary_count + comparator_count)`. |
| `median_timing_difference_ms` | number or `null` | Median displacement among matched pairs. |
| `unmatched_primary_fraction` | number or `null` | Unmatched primary count divided by primary count. |
| `unmatched_comparator_fraction` | number or `null` | Unmatched comparator count divided by comparator count. |
| `matched_count` | integer | One-to-one pairs inside 75 ms. |
| `primary_count` | integer | Total candidates from the selected primary. |
| `comparator_count` | integer | Total candidates from the comparator. |
| `unmatched_primary_samples` | integer array | Segment-relative samples. |
| `unmatched_comparator_samples` | integer array | Segment-relative samples. |
| `unmatched_primary_times_s` | number array | Same primary disagreements converted to EDF-relative seconds. |
| `unmatched_comparator_times_s` | number array | Same comparator disagreements converted to EDF-relative seconds. |

JSON has no standard NaN literal. `_finite_or_none` converts undefined numeric
metrics to `null`.

## 9. `metadata.json` schema

| Field | Meaning |
|---|---|
| `edf_path` | Resolved source path used for the run. |
| `channel` | Full `ChannelInfo`: index, label, sampling rate, sample count, unit, physical min/max. |
| `segment_start_s` | Requested segment start retained by the `SignalSegment`. |
| `segment_duration_s` | Duration represented by the returned sample count. |
| `primary_rpeak_method` | NeuroKit2 method whose candidates generated RR. |
| `comparator_rpeak_method` | Other NeuroKit2 method. |
| `rpeak_matching_tolerance_ms` | Fixed at 75 ms in the current pipeline. |
| `rr_hrv_window_rr_intervals` | Fixed at 100. |
| `causal_rr_median_width` | Fixed at 7. |
| `threshold_calibration_applied` | Always `false` in this implementation. |
| `package_versions` | Installed versions of the core numeric/data packages. |
| `events_for_plot_only_s` | User-supplied plot intervals. |
| `feature_rows` | Number of RR intervals exported. Normally number of primary peaks minus one. |
| `fully_defined_feature_rows` | Rows where both continuous final indices are finite. |

`fully_defined_feature_rows` is not the number of trustworthy rows. It is only
the count with `feature_defined=True`.

## 10. `rr_hrv_features.csv`

The complete mathematical dictionary is in
[04 RR/HRV mathematics and code](04_RR_HRV_MATHEMATICS_AND_CODE.md). Empty CSV
cells in the first 99 rows represent unavailable 100-RR values, not zeros.

When pandas reads this file, explicitly preserve Boolean and nullable fields if
they will be used for fusion. Do not fill all missing values with zero: zero is
a possible mathematical value and is not equivalent to unavailable history.

## 11. Diagnostic figure

`plotting.py` sets the Matplotlib backend to `Agg`, allowing PNG generation
without a display. It creates three panels:

1. **Twenty-second ECG audit view.** Raw EDF physical samples with both sets of
   candidate markers. The view centers on the first overlapping event; if no
   event is supplied, it centers on the first detector disagreement; otherwise
   it uses the segment midpoint.
2. **RR-derived instantaneous heart rate.** Computed from the chosen primary
   peak source.
3. **Continuous indices.** `J1` and `J2`, explicitly labeled as having no
   seizure threshold.

Event regions are shaded only on panels 2 and 3. The first panel remains a
twenty-second local audit view. The figure is saved at 170 dpi and then closed
to prevent accumulation during batch execution.

The two final indices have different units and numerical scales. Plotting them
on one axis is useful for timing inspection but not for direct magnitude
comparison.

## 12. Reproducible examples using the local manifest

The manifest `config/datasets.toml` records paths, exact ECG channel labels,
expected sampling rates, and known event intervals. It is a local experiment
manifest, not executable clinical truth.

CHB-MIT example:

```powershell
.venv\Scripts\ecg-features run-rr-hrv `
  --edf "C:\Users\Salam\Downloads\chb04_28\chb04_28.edf" `
  --channel ECG --start-s 1500 --duration-s 600 `
  --primary-method pantompkins1985 --event 1679:1781 `
  --output-dir "C:\Users\Salam\Documents\ECGdetector\output\feature_tests\chb04_28_seizure1_1500_2100"
```

Siena example:

```powershell
.venv\Scripts\ecg-features run-rr-hrv `
  --edf "C:\Users\Salam\Downloads\Seizures-list-PN06\PN06-1.edf" `
  --channel "EKG EKG" --start-s 0 --duration-s 600 `
  --primary-method pantompkins1985 `
  --output-dir "C:\Users\Salam\Documents\ECGdetector\output\feature_tests\pn06_1_interictal_0_600_pan"
```

## 13. Common failure messages

| Failure | Likely cause | Correct response |
|---|---|---|
| EDF does not exist | Incorrect path or moved dataset. | Resolve the source path; do not substitute another file silently. |
| More than one ECG/EKG candidate | Ambiguous header labels. | Run `inspect-edf` and pass `--channel` explicitly. |
| No ECG/EKG-labeled channel | Unusual label. | Inspect units and montage, then specify a verified channel. |
| Segment starts after channel end | Wrong seconds or wrong file. | Compare start/event times with EDF duration. |
| At least three seconds required | Segment too short for the detector wrapper. | Increase duration for this implementation. |
| At least two R peaks required | Detector returned insufficient candidates. | Inspect the signal and detector output; do not synthesize RR values. |

## 14. Batch-processing warning

Each command is independent. Running successive ten-minute chunks resets the
seven-RR median history and the 100-RR window, causing a cold start in every
chunk. Do not concatenate those outputs as if the boundaries were continuous.
A future streaming implementation must explicitly carry peak, RR, and median
state across chunks and test equivalence against a single uninterrupted run.
