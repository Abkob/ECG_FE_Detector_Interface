# 02. EDF and data layer

## 1. Source file

Implementation: `src/ecg_cascade/edf.py`.

The EDF layer is intentionally conservative. It does not assume that the last
channel is ECG, that every channel has the same sampling rate, or that a numeric
label such as `1` or `2` denotes an ECG lead. Those assumptions are unsafe in
mixed EEG/ECG monitoring files.

## 2. `ChannelInfo`

`ChannelInfo` is an immutable dataclass with seven fields.

| Field | Type | Meaning |
|---|---|---|
| `index` | `int` | Zero-based signal index used by pyEDFlib. |
| `label` | `str` | Whitespace-trimmed EDF signal label. |
| `sampling_rate_hz` | `float` | Samples per second for this channel. |
| `sample_count` | `int` | Total samples available for this channel. |
| `physical_dimension` | `str` | EDF physical unit, such as `uV`, if present. |
| `physical_min` | `float` | Minimum physical calibration value declared in the header. |
| `physical_max` | `float` | Maximum physical calibration value declared in the header. |

The physical minimum and maximum are calibration metadata; they are not the
observed minimum and maximum in the loaded interval.

The dataclass is frozen so downstream code cannot accidentally change the
sampling rate or channel identity after data are loaded.

## 3. `RecordingInfo`

`RecordingInfo` represents one EDF header:

- resolved path;
- file duration in seconds;
- EDF start date/time serialized as ISO 8601;
- tuple of `ChannelInfo` objects.

`to_dict()` converts nested dataclasses into JSON-compatible dictionaries.
This method is used by the `inspect-edf` command.

### Start-time warning

The installed `chb04_28` EDF reports a year of 2079. EDF date fields in public
or de-identified datasets may be shifted or synthetic. The code records the
header value but does not use it to align seizure events. Current event times
are seconds from the beginning of each EDF.

## 4. `SignalSegment`

`SignalSegment` is the in-memory signal passed to detector code.

| Field | Meaning |
|---|---|
| `path` | Resolved EDF path. |
| `channel` | Selected immutable `ChannelInfo`. |
| `start_s` | Requested start in seconds from EDF start. |
| `duration_s` | Actual returned duration, computed from sample count and channel sampling rate. |
| `samples` | One-dimensional NumPy `float64` physical samples. |

No copy of every other EEG or physiological channel is loaded. This keeps the
current branch ECG-only and prevents accidental use of EEG in a model intended
to be evaluated as ECG-only.

## 5. `inspect_edf(path)`

### Inputs

- a string or `Path`;
- relative paths are resolved to absolute paths;
- `~` is expanded by `Path.expanduser()`.

### Processing

1. Confirm that the path is a file.
2. Open it using `pyedflib.EdfReader`.
3. Read all signal labels and per-signal sample counts.
4. For each signal, read its own frequency, unit, and physical range.
5. Read file start time and duration.
6. Close the reader in a `finally` block even if metadata extraction fails.
7. Return `RecordingInfo`.

### Why the `finally` block matters

EDF files are large and file handles can prevent deletion, copying, or repeated
analysis on Windows. Closing in `finally` guarantees cleanup when an exception
is raised halfway through header processing.

### Exceptions

- `FileNotFoundError`: the resolved path is not an existing file.
- pyEDFlib errors: invalid or unsupported EDF content.

## 6. `choose_ecg_channel(recording, requested_label)`

### Explicit-label mode

When `requested_label` is supplied:

1. leading and trailing whitespace are removed from the request;
2. matching is case-insensitive using `casefold()`;
3. exactly one header label must match;
4. zero or multiple matches raise `ValueError` and list all available labels.

This strictness prevents a spelling error from silently selecting an EEG
channel.

### Automatic mode

When no label is supplied, a channel is considered a candidate only if its
uppercased label contains the distinct token `ECG` or `EKG`. The regular
expression uses nonalphabetic boundaries, preventing unrelated strings that
merely contain those letters from matching.

The function succeeds only when exactly one candidate exists.

- No candidates: stop and require human verification.
- Multiple candidates: stop and require an exact label.

### What the function does not do

It does not inspect waveform morphology to prove that the channel is ECG. A
mislabelled EDF can still pass. Channel selection must be visually checked at
least once per acquisition system.

## 7. `load_ecg_segment(...)`

### Input validation

- `start_s` must be at least zero.
- `duration_s` must be greater than zero.
- `start_s` must be before the EDF duration.
- channel selection must succeed.

### Seconds-to-samples conversion

For channel sampling rate `fs`:

```text
start_sample = round(start_s * fs)
requested_count = round(duration_s * fs)
```

If the requested interval runs beyond the channel end, the sample count is
clipped to the remaining samples. The returned `duration_s` therefore records
the actual interval, not blindly the requested duration.

### Signal read

`reader.readSignal(channel.index, start=start_sample, n=sample_count)` returns
physical values after EDF digital-to-physical calibration. The result is
converted to NumPy `float64`.

### Integrity checks

- Returned sample count must equal the expected count.
- Every sample must be finite.
- Nonfinite samples cause a hard error; the loader does not interpolate them.

This prevents hidden imputation from being mistaken for measured ECG.

## 8. Installed dataset manifest

`config/datasets.toml` records local proof-of-concept paths and event times.
The manifest is not an authoritative clinical annotation database. It is an
execution aid derived from the available dataset summary files.

### `chb04_28`

- path: `C:/Users/Salam/Downloads/chb04_28/chb04_28.edf`;
- one header-labelled `ECG` channel;
- 256 Hz;
- approximately four hours;
- seizures at 1679-1781 s and 3782-3898 s according to `chb04-summary.txt`.

`chb04_01` was not included because its EDF header has no ECG channel.

### PN06 and PN12

- header-labelled channel: `EKG EKG`;
- 512 Hz;
- the summary documents describe `EKG 1` and `EKG 2` using channel numbering
  that does not directly match the current EDF labels;
- numeric channels `1` and `2` are therefore not assumed to be ECG;
- current runs explicitly select `EKG EKG`, whose physical range and morphology
  are consistent with an ECG-like auxiliary channel, but this mapping still
  requires dataset documentation or manual confirmation.

## 9. Known data-layer limitations

1. Actual `start_sample` is not stored in `SignalSegment`.
2. No EDF discontinuity handling is implemented.
3. No EDF annotation channel is parsed.
4. Event times are supplied separately rather than read from a standardized
   annotation object.
5. No lead-name ontology maps `ECG`, `EKG EKG`, lead I, or lead II.
6. No signal polarity normalization is performed.
7. No chunked iterator exists for processing many hours without loading one
   requested segment at a time.

These are engineering tasks, not evidence that the current loader is wrong for
the tested segments. They become necessary before full-record production runs.

