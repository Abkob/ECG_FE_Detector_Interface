"""Manual ECG-delineation references and lead-controlled validation helpers.

This module is deliberately separate from the seizure feature pipeline.  It
answers whether P/QRS/T landmarks can be measured against manual references;
it does not produce seizure labels or silently promote a delineator into the
runtime morphology branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import neurokit2 as nk
import numpy as np
import wfdb

from .peaks import compare_peak_sequences
from .wfdb_io import load_wfdb_beat_annotations


LANDMARK_NAMES = (
    "p_onset",
    "p_peak",
    "p_offset",
    "qrs_onset",
    "r_peak",
    "qrs_offset",
    "t_onset",
    "t_peak",
    "t_offset",
)

_NEUROKIT_KEYS = {
    "p_onset": "ECG_P_Onsets",
    "p_peak": "ECG_P_Peaks",
    "p_offset": "ECG_P_Offsets",
    "qrs_onset": "ECG_R_Onsets",
    "qrs_offset": "ECG_R_Offsets",
    "t_onset": "ECG_T_Onsets",
    "t_peak": "ECG_T_Peaks",
    "t_offset": "ECG_T_Offsets",
}

_BEAT_SYMBOLS = {
    "N", "L", "R", "B", "A", "a", "J", "S", "V", "r", "F",
    "e", "j", "n", "E", "/", "f", "Q", "?",
}


@dataclass(frozen=True)
class ManualDelineationRecord:
    """One ECG lead with manual landmarks and explicit evaluation anchors."""

    dataset: str
    record_id: str
    record_path: str
    lead_name: str
    lead_index: int
    sampling_rate_hz: float
    samples: np.ndarray
    landmarks: Mapping[str, np.ndarray]
    anchor_samples: np.ndarray
    evaluation_anchor_indices: np.ndarray
    annotation_source: str

    @property
    def evaluation_anchor_samples(self) -> np.ndarray:
        return self.anchor_samples[self.evaluation_anchor_indices]


def load_ludb_delineation_record(
    dataset: str | Path,
    record_id: str,
    *,
    lead: str = "ii",
) -> ManualDelineationRecord:
    """Load one LUDB lead and its lead-specific cardiologist annotations."""

    dataset = Path(dataset).expanduser().resolve()
    record_path = dataset / str(record_id)
    header = wfdb.rdheader(str(record_path))
    lead_matches = [
        index
        for index, name in enumerate(header.sig_name)
        if str(name).casefold() == str(lead).casefold()
    ]
    if len(lead_matches) != 1:
        raise ValueError(
            f"{record_id}: lead {lead!r} matched {len(lead_matches)} signals; "
            f"available={header.sig_name}"
        )
    lead_index = lead_matches[0]
    record = wfdb.rdrecord(
        str(record_path), channels=[lead_index], physical=True
    )
    if record.p_signal is None:
        raise ValueError(f"{record_id}: no physical signal returned")
    samples = np.asarray(record.p_signal[:, 0], dtype=np.float64)
    annotation = wfdb.rdann(str(record_path), str(lead).casefold())
    landmarks = parse_ludb_annotations(annotation.sample, annotation.symbol)
    anchors = landmarks["r_peak"]
    return _build_record(
        dataset="ludb",
        record_id=str(record_id),
        record_path=record_path,
        lead_name=str(header.sig_name[lead_index]),
        lead_index=lead_index,
        sampling_rate_hz=float(header.fs),
        samples=samples,
        landmarks=landmarks,
        anchor_samples=anchors,
        evaluation_anchor_indices=np.arange(anchors.size, dtype=np.int64),
        annotation_source=f"{lead}.manual",
    )


def load_qtdb_delineation_record(
    dataset: str | Path,
    record_id: str,
    *,
    annotation_extension: str = "q1c",
    required_channel0: str | None = "MLII",
    anchor_tolerance_ms: float = 75.0,
) -> ManualDelineationRecord:
    """Load QTDB channel 0 and selected manual waveform annotations.

    QTDB manual annotations cover selected beats.  When a full ``atr`` stream
    exists, it supplies surrounding expert QRS anchors to the delineator; only
    anchors corresponding to manually delineated beats enter evaluation.
    """

    dataset = Path(dataset).expanduser().resolve()
    record_path = dataset / str(record_id)
    header = wfdb.rdheader(str(record_path))
    lead_name = str(header.sig_name[0])
    if (
        required_channel0 is not None
        and lead_name.casefold() != required_channel0.casefold()
    ):
        raise ValueError(
            f"{record_id}: channel 0 is {lead_name!r}, not "
            f"{required_channel0!r}"
        )
    record = wfdb.rdrecord(str(record_path), channels=[0], physical=True)
    if record.p_signal is None:
        raise ValueError(f"{record_id}: no physical signal returned")
    samples = np.asarray(record.p_signal[:, 0], dtype=np.float64)
    annotation = wfdb.rdann(str(record_path), annotation_extension)
    landmarks = parse_qtdb_annotations(
        annotation.sample,
        annotation.symbol,
        annotation.num,
    )
    manual_anchors = landmarks["r_peak"]
    atr_path = record_path.with_suffix(".atr")
    if atr_path.is_file():
        anchors, _ = load_wfdb_beat_annotations(record_path)
        evaluation_indices = match_anchor_indices(
            anchors,
            manual_anchors,
            sampling_rate_hz=float(header.fs),
            tolerance_ms=anchor_tolerance_ms,
        )
        annotation_source = f"{annotation_extension}.manual+atr_context"
    else:
        anchors = manual_anchors
        evaluation_indices = np.arange(anchors.size, dtype=np.int64)
        annotation_source = f"{annotation_extension}.manual_only"
    return _build_record(
        dataset="qtdb",
        record_id=str(record_id),
        record_path=record_path,
        lead_name=lead_name,
        lead_index=0,
        sampling_rate_hz=float(header.fs),
        samples=samples,
        landmarks=landmarks,
        anchor_samples=anchors,
        evaluation_anchor_indices=evaluation_indices,
        annotation_source=annotation_source,
    )


def parse_ludb_annotations(
    samples: np.ndarray | list[int],
    symbols: np.ndarray | list[str],
) -> dict[str, np.ndarray]:
    """Convert LUDB ``( peak )`` groups into a common landmark schema."""

    sample_array, symbol_array = _annotation_arrays(samples, symbols)
    output = _empty_landmarks()
    center_map = {
        "p": ("p_onset", "p_peak", "p_offset"),
        "N": ("qrs_onset", "r_peak", "qrs_offset"),
        "t": ("t_onset", "t_peak", "t_offset"),
    }
    for index, symbol in enumerate(symbol_array.tolist()):
        if symbol not in center_map:
            continue
        onset_name, peak_name, offset_name = center_map[symbol]
        output[peak_name].append(int(sample_array[index]))
        if index > 0 and symbol_array[index - 1] == "(":
            output[onset_name].append(int(sample_array[index - 1]))
        if index + 1 < symbol_array.size and symbol_array[index + 1] == ")":
            output[offset_name].append(int(sample_array[index + 1]))
    return _finalize_landmarks(output)


def parse_qtdb_annotations(
    samples: np.ndarray | list[int],
    symbols: np.ndarray | list[str],
    nums: np.ndarray | list[int],
) -> dict[str, np.ndarray]:
    """Convert QTDB manual annotations into the common landmark schema.

    QTDB uses ``num`` 0/1/2 for P/QRS/T boundary markers.  T-wave onsets are
    present only for a subset of manually delineated beats and remain missing
    rather than being imputed.
    """

    sample_array, symbol_array = _annotation_arrays(samples, symbols)
    num_array = np.asarray(nums, dtype=np.int64)
    if num_array.ndim != 1 or num_array.size != sample_array.size:
        raise ValueError("nums must be one-dimensional and match samples")
    output = _empty_landmarks()
    for index, symbol in enumerate(symbol_array.tolist()):
        if symbol == "p":
            boundary_num = 0
            names = ("p_onset", "p_peak", "p_offset")
        elif symbol == "t":
            boundary_num = 2
            names = ("t_onset", "t_peak", "t_offset")
        elif symbol in _BEAT_SYMBOLS:
            boundary_num = 1
            names = ("qrs_onset", "r_peak", "qrs_offset")
        else:
            continue
        onset_name, peak_name, offset_name = names
        output[peak_name].append(int(sample_array[index]))
        if (
            index > 0
            and symbol_array[index - 1] == "("
            and int(num_array[index - 1]) == boundary_num
        ):
            output[onset_name].append(int(sample_array[index - 1]))
        if (
            index + 1 < symbol_array.size
            and symbol_array[index + 1] == ")"
            and int(num_array[index + 1]) == boundary_num
        ):
            output[offset_name].append(int(sample_array[index + 1]))
    return _finalize_landmarks(output)


def match_anchor_indices(
    full_anchors: np.ndarray,
    evaluation_anchors: np.ndarray,
    *,
    sampling_rate_hz: float,
    tolerance_ms: float = 75.0,
) -> np.ndarray:
    """Map selected manual anchors bijectively into a complete anchor stream."""

    full = _sorted_unique_int(full_anchors, "full_anchors")
    selected = _sorted_unique_int(evaluation_anchors, "evaluation_anchors")
    agreement = compare_peak_sequences(
        full,
        selected,
        sampling_rate_hz=sampling_rate_hz,
        tolerance_ms=tolerance_ms,
    )
    if agreement.unmatched_comparator_samples.size:
        raise ValueError(
            "Manual evaluation anchors missing from complete reference stream: "
            f"{agreement.unmatched_comparator_samples.tolist()}"
        )
    index_by_sample = {int(value): index for index, value in enumerate(full)}
    return np.asarray(
        [index_by_sample[int(value)] for value in agreement.matched_primary_samples],
        dtype=np.int64,
    )


def run_neurokit_delineation(
    record: ManualDelineationRecord,
    *,
    method: str,
    cleaning_method: str = "neurokit",
) -> dict[str, np.ndarray]:
    """Run one NeuroKit delineator using reference QRS anchors.

    ``check=False`` is intentional.  NeuroKit2 0.2.13's CWT path returns
    unequal-length landmark lists, causing its optional consistency checker to
    raise before results can be audited.  Unequal lists are retained and
    associated with anchors by physiological phase below.
    """

    cleaned = np.asarray(
        nk.ecg_clean(
            record.samples,
            sampling_rate=record.sampling_rate_hz,
            method=cleaning_method,
        ),
        dtype=np.float64,
    )
    _, waves = nk.ecg_delineate(
        cleaned,
        rpeaks=record.anchor_samples,
        sampling_rate=record.sampling_rate_hz,
        method=method,
        check=False,
    )
    predictions = _empty_landmarks()
    for landmark, neurokit_key in _NEUROKIT_KEYS.items():
        reference_anchor_indices = associate_landmarks_with_anchors(
            record.landmarks[landmark],
            record.anchor_samples,
            landmark=landmark,
            sampling_rate_hz=record.sampling_rate_hz,
        )
        evaluation_set = {
            int(index)
            for index in reference_anchor_indices.tolist()
            if index >= 0
        }
        raw = list(waves.get(neurokit_key, []))
        if len(raw) == record.anchor_samples.size:
            selected = [
                raw[index]
                for index in sorted(evaluation_set)
                if _is_finite_scalar(raw[index])
            ]
        else:
            finite = _finite_sample_values(raw, record.samples.size)
            associated = associate_landmarks_with_anchors(
                finite,
                record.anchor_samples,
                landmark=landmark,
                sampling_rate_hz=record.sampling_rate_hz,
            )
            selected = [
                sample
                for sample, anchor_index in zip(finite.tolist(), associated.tolist())
                if anchor_index in evaluation_set
            ]
        predictions[landmark].extend(int(value) for value in selected)
    # R peaks are inputs, not delineator predictions.  They are retained only
    # for schema completeness and must not be reported as model performance.
    predictions["r_peak"].extend(record.evaluation_anchor_samples.tolist())
    return _finalize_landmarks(predictions)


def associate_landmarks_with_anchors(
    landmark_samples: np.ndarray,
    anchor_samples: np.ndarray,
    *,
    landmark: str,
    sampling_rate_hz: float,
) -> np.ndarray:
    """Associate variable-length delineator outputs with R-anchor indices."""

    values = _sorted_unique_int(landmark_samples, "landmark_samples")
    anchors = _sorted_unique_int(anchor_samples, "anchor_samples")
    if anchors.size == 0:
        return np.full(values.size, -1, dtype=np.int64)
    result = np.full(values.size, -1, dtype=np.int64)
    if landmark.startswith("p_"):
        maximum_ms = 600.0
        for index, value in enumerate(values):
            anchor_index = int(np.searchsorted(anchors, value, side="left"))
            if anchor_index < anchors.size:
                delay_ms = 1000.0 * (anchors[anchor_index] - value) / sampling_rate_hz
                if 0.0 <= delay_ms <= maximum_ms:
                    result[index] = anchor_index
    elif landmark == "qrs_onset":
        maximum_ms = 250.0
        for index, value in enumerate(values):
            insertion = int(np.searchsorted(anchors, value))
            choices = [candidate for candidate in (insertion - 1, insertion) if 0 <= candidate < anchors.size]
            if choices:
                anchor_index = min(choices, key=lambda candidate: abs(int(anchors[candidate]) - int(value)))
                delay_ms = 1000.0 * abs(int(anchors[anchor_index]) - int(value)) / sampling_rate_hz
                if delay_ms <= maximum_ms:
                    result[index] = anchor_index
    else:
        maximum_ms = 300.0 if landmark == "qrs_offset" else 800.0
        for index, value in enumerate(values):
            anchor_index = int(np.searchsorted(anchors, value, side="right")) - 1
            if anchor_index >= 0:
                delay_ms = 1000.0 * (value - anchors[anchor_index]) / sampling_rate_hz
                if 0.0 <= delay_ms <= maximum_ms:
                    result[index] = anchor_index
    return result


def evaluate_landmark_samples(
    predicted_samples: np.ndarray,
    reference_samples: np.ndarray,
    *,
    sampling_rate_hz: float,
    tolerance_ms: float = 150.0,
) -> tuple[dict[str, float | int], np.ndarray]:
    """Calculate one-to-one landmark coverage, precision, and timing errors."""

    predicted = _sorted_unique_int(predicted_samples, "predicted_samples")
    reference = _sorted_unique_int(reference_samples, "reference_samples")
    agreement = compare_peak_sequences(
        predicted,
        reference,
        sampling_rate_hz=sampling_rate_hz,
        tolerance_ms=tolerance_ms,
    )
    matched = int(agreement.matched_primary_samples.size)
    predicted_count = int(predicted.size)
    reference_count = int(reference.size)
    errors_ms = (
        1000.0
        * (
            agreement.matched_primary_samples
            - agreement.matched_comparator_samples
        )
        / sampling_rate_hz
    )
    absolute = np.abs(errors_ms)
    metrics: dict[str, float | int] = {
        "reference_count": reference_count,
        "predicted_count": predicted_count,
        "matched_count": matched,
        "sensitivity": _divide(matched, reference_count),
        "ppv": _divide(matched, predicted_count),
        "f1": _divide(2 * matched, reference_count + predicted_count),
        "mean_error_ms": _mean_or_nan(errors_ms),
        "median_error_ms": _median_or_nan(errors_ms),
        "median_absolute_error_ms": _median_or_nan(absolute),
        "p95_absolute_error_ms": _percentile_or_nan(absolute, 95.0),
        "within_20_ms": _mean_or_nan(absolute <= 20.0),
        "within_40_ms": _mean_or_nan(absolute <= 40.0),
        "within_80_ms": _mean_or_nan(absolute <= 80.0),
    }
    return metrics, np.asarray(errors_ms, dtype=np.float64)


def fixed_qrs_capture_rows(
    record: ManualDelineationRecord,
    *,
    pre_r_ms: float = 60.0,
    post_r_ms: float = 60.0,
) -> list[dict[str, object]]:
    """Measure whether a fixed R-centered window contains the manual QRS."""

    if pre_r_ms <= 0 or post_r_ms <= 0:
        raise ValueError("capture durations must be positive")
    onset_indices = associate_landmarks_with_anchors(
        record.landmarks["qrs_onset"],
        record.anchor_samples,
        landmark="qrs_onset",
        sampling_rate_hz=record.sampling_rate_hz,
    )
    offset_indices = associate_landmarks_with_anchors(
        record.landmarks["qrs_offset"],
        record.anchor_samples,
        landmark="qrs_offset",
        sampling_rate_hz=record.sampling_rate_hz,
    )
    onset_by_anchor = {
        int(index): int(sample)
        for sample, index in zip(record.landmarks["qrs_onset"], onset_indices)
        if index >= 0
    }
    offset_by_anchor = {
        int(index): int(sample)
        for sample, index in zip(record.landmarks["qrs_offset"], offset_indices)
        if index >= 0
    }
    output: list[dict[str, object]] = []
    for anchor_index in record.evaluation_anchor_indices.tolist():
        if anchor_index not in onset_by_anchor or anchor_index not in offset_by_anchor:
            continue
        r_peak = int(record.anchor_samples[anchor_index])
        onset = onset_by_anchor[anchor_index]
        offset = offset_by_anchor[anchor_index]
        onset_lead_ms = 1000.0 * (r_peak - onset) / record.sampling_rate_hz
        offset_lag_ms = 1000.0 * (offset - r_peak) / record.sampling_rate_hz
        output.append(
            {
                "dataset": record.dataset,
                "record_id": record.record_id,
                "lead_name": record.lead_name,
                "anchor_index": int(anchor_index),
                "qrs_onset_sample": onset,
                "r_peak_sample": r_peak,
                "qrs_offset_sample": offset,
                "onset_lead_ms": float(onset_lead_ms),
                "offset_lag_ms": float(offset_lag_ms),
                "qrs_duration_ms": float(onset_lead_ms + offset_lag_ms),
                "pre_r_ms": float(pre_r_ms),
                "post_r_ms": float(post_r_ms),
                "complete_qrs_captured": bool(
                    onset_lead_ms <= pre_r_ms and offset_lag_ms <= post_r_ms
                ),
            }
        )
    return output


def qrs_polarity_summary(
    record: ManualDelineationRecord,
    *,
    baseline_ms: float = 40.0,
) -> dict[str, object]:
    """Classify dominant QRS polarity from manual QRS intervals on one lead."""

    if baseline_ms <= 0:
        raise ValueError("baseline_ms must be positive")
    baseline_samples = max(
        1, int(np.rint(baseline_ms * record.sampling_rate_hz / 1000.0))
    )
    negative = 0
    measured = 0
    for row in fixed_qrs_capture_rows(record):
        onset = int(row["qrs_onset_sample"])
        offset = int(row["qrs_offset_sample"])
        start = max(0, onset - baseline_samples)
        baseline = (
            float(np.median(record.samples[start:onset])) if onset > start else 0.0
        )
        qrs = record.samples[onset : offset + 1] - baseline
        if qrs.size == 0:
            continue
        measured += 1
        positive_excursion = max(0.0, float(np.max(qrs)))
        negative_excursion = abs(min(0.0, float(np.min(qrs))))
        if negative_excursion > positive_excursion:
            negative += 1
    negative_fraction = float(negative / measured) if measured else np.nan
    return {
        "dataset": record.dataset,
        "record_id": record.record_id,
        "lead_name": record.lead_name,
        "qrs_beats_measured": measured,
        "negative_dominant_qrs_count": negative,
        "negative_dominant_qrs_fraction": negative_fraction,
        "record_polarity_group": (
            "negative_or_mixed"
            if measured and negative_fraction >= 0.5
            else "positive"
            if measured
            else "unavailable"
        ),
        "baseline_ms": float(baseline_ms),
    }


def _build_record(
    *,
    dataset: str,
    record_id: str,
    record_path: Path,
    lead_name: str,
    lead_index: int,
    sampling_rate_hz: float,
    samples: np.ndarray,
    landmarks: Mapping[str, np.ndarray],
    anchor_samples: np.ndarray,
    evaluation_anchor_indices: np.ndarray,
    annotation_source: str,
) -> ManualDelineationRecord:
    signal = np.asarray(samples, dtype=np.float64)
    if signal.ndim != 1 or signal.size == 0 or not np.all(np.isfinite(signal)):
        raise ValueError(f"{record_id}: ECG must be finite, non-empty, and one-dimensional")
    if sampling_rate_hz <= 0:
        raise ValueError(f"{record_id}: sampling rate must be positive")
    anchors = _sorted_unique_int(anchor_samples, "anchor_samples")
    indices = np.asarray(evaluation_anchor_indices, dtype=np.int64)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= anchors.size):
        raise ValueError(f"{record_id}: invalid evaluation anchor indices")
    if np.unique(indices).size != indices.size:
        raise ValueError(f"{record_id}: duplicate evaluation anchor indices")
    normalized_landmarks = {
        name: _sorted_unique_int(landmarks.get(name, []), name)
        for name in LANDMARK_NAMES
    }
    for name, values in normalized_landmarks.items():
        if values.size and (values[0] < 0 or values[-1] >= signal.size):
            raise ValueError(f"{record_id}: {name} lies outside the ECG")
    return ManualDelineationRecord(
        dataset=dataset,
        record_id=record_id,
        record_path=str(record_path.resolve()),
        lead_name=lead_name,
        lead_index=int(lead_index),
        sampling_rate_hz=float(sampling_rate_hz),
        samples=signal,
        landmarks=normalized_landmarks,
        anchor_samples=anchors,
        evaluation_anchor_indices=indices,
        annotation_source=annotation_source,
    )


def _annotation_arrays(
    samples: np.ndarray | list[int],
    symbols: np.ndarray | list[str],
) -> tuple[np.ndarray, np.ndarray]:
    sample_array = np.asarray(samples, dtype=np.int64)
    symbol_array = np.asarray(symbols, dtype=object)
    if sample_array.ndim != 1 or symbol_array.ndim != 1:
        raise ValueError("annotation arrays must be one-dimensional")
    if sample_array.size != symbol_array.size:
        raise ValueError("samples and symbols must have equal length")
    if sample_array.size and np.any(np.diff(sample_array) < 0):
        raise ValueError("annotation samples must be monotone")
    return sample_array, symbol_array


def _empty_landmarks() -> dict[str, list[int]]:
    return {name: [] for name in LANDMARK_NAMES}


def _finalize_landmarks(
    values: Mapping[str, list[int]],
) -> dict[str, np.ndarray]:
    return {
        name: np.unique(np.asarray(values.get(name, []), dtype=np.int64))
        for name in LANDMARK_NAMES
    }


def _finite_sample_values(values: list[object], signal_size: int) -> np.ndarray:
    output = [
        int(np.rint(float(value)))
        for value in values
        if _is_finite_scalar(value)
        and 0 <= int(np.rint(float(value))) < signal_size
    ]
    return np.unique(np.asarray(output, dtype=np.int64))


def _is_finite_scalar(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _sorted_unique_int(values: object, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return np.unique(array)


def _divide(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else np.nan


def _mean_or_nan(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else np.nan


def _median_or_nan(values: np.ndarray) -> float:
    return float(np.median(values)) if values.size else np.nan


def _percentile_or_nan(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values.size else np.nan
