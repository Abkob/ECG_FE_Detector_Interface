"""Interactive EDF/WFDB reviewer for the current ECG feature branches.

The viewer deliberately separates automated detections, deterministic
morphology measurements, expert beat annotations, and manual review.  An
unannotated marker is not called a true positive, detector disagreement is not
called artifact, and Varon eigenvalues are not called a seizure prediction.
Manual decisions are persisted to CSV.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Iterable

import numpy as np

from .edf import ChannelInfo, RecordingInfo, inspect_edf, load_ecg_segment
from .morphology import VaronMorphologyResult, extract_varon_morphology
from .morphology_validation import build_reference_context
from .peaks import (
    compare_peak_sequences,
    detect_neurokit_gradient,
    detect_r_peaks,
    detect_unsw,
)
from .zhai import detect_zhai_template


DETECTION_METHOD = "neurokit"
DETECTION_PIPELINE = (
    'NeuroKit2 ecg_clean(method="neurokit") + '
    'ecg_peaks(method="neurokit", correct_artifacts=False)'
)
REVIEW_LABELS = {
    "confirmed_r_peak",
    "false_positive",
    "uncertain",
    "manual_missed_r_peak",
}
MIT_BIH_BEAT_SYMBOLS = {
    "N", "L", "R", "B", "A", "a", "J", "S", "V", "r", "F",
    "e", "j", "n", "E", "/", "f", "Q", "?",
}
MIT_BIH_SYMBOL_NAMES = {
    "N": "normal beat",
    "L": "left bundle-branch block beat",
    "R": "right bundle-branch block beat",
    "B": "bundle-branch block beat",
    "A": "atrial premature beat",
    "a": "aberrated atrial premature beat",
    "J": "nodal/junctional premature beat",
    "S": "supraventricular premature beat",
    "V": "premature ventricular contraction",
    "r": "R-on-T premature ventricular contraction",
    "F": "fusion of ventricular and normal beat",
    "e": "atrial escape beat",
    "j": "nodal/junctional escape beat",
    "n": "supraventricular escape beat",
    "E": "ventricular escape beat",
    "/": "paced beat",
    "f": "fusion of paced and normal beat",
    "Q": "unclassifiable beat",
    "?": "beat not otherwise specified",
}
TRACK_DISPLAY_NAMES = {
    "neurokit": "NeuroKit",
    "unsw": "UNSW/Khamis",
    "zhai": "Zhai template",
    "expert": "Expert .atr",
}


@dataclass(frozen=True)
class TrackReferenceComparison:
    """Visible one-to-one detector-versus-expert comparison."""

    matched_detection_samples: np.ndarray
    matched_reference_samples: np.ndarray
    false_positive_samples: np.ndarray
    missed_reference_samples: np.ndarray

    def metrics(self) -> dict[str, float | int]:
        tp = int(self.matched_detection_samples.size)
        fp = int(self.false_positive_samples.size)
        fn = int(self.missed_reference_samples.size)
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
            "ppv": tp / (tp + fp) if tp + fp else float("nan"),
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan"),
        }


@dataclass(frozen=True)
class DetectionWindow:
    """One visible ECG window with detector, expert, and morphology tracks."""

    recording_path: str
    channel: ChannelInfo
    recording_duration_s: float
    start_sample: int
    end_sample: int
    raw_ecg: np.ndarray
    cleaned_ecg: np.ndarray
    neurokit_peak_samples: np.ndarray
    input_format: str
    reference_beat_samples: np.ndarray
    reference_symbols: tuple[str, ...]
    matched_detection_samples: np.ndarray
    matched_reference_samples: np.ndarray
    false_positive_samples: np.ndarray
    missed_reference_samples: np.ndarray
    detector_peak_samples: dict[str, np.ndarray]
    reference_comparisons: dict[str, TrackReferenceComparison]
    morphology_results: dict[str, VaronMorphologyResult]

    @property
    def sampling_rate_hz(self) -> float:
        return self.channel.sampling_rate_hz

    @property
    def start_s(self) -> float:
        return self.start_sample / self.sampling_rate_hz

    @property
    def end_s(self) -> float:
        return self.end_sample / self.sampling_rate_hz

    @property
    def times_s(self) -> np.ndarray:
        return np.arange(self.start_sample, self.end_sample) / self.sampling_rate_hz

    def peaks_for(self, track: str) -> np.ndarray:
        if track == "expert":
            return self.reference_beat_samples
        return self.detector_peak_samples.get(track, np.asarray([], dtype=np.int64))

    def comparison_for(self, track: str) -> TrackReferenceComparison | None:
        return self.reference_comparisons.get(track)


@dataclass(frozen=True)
class ReviewRecord:
    recording_path: str
    channel: str
    sampling_rate_hz: float
    sample_index: int
    time_s: float
    source: str
    label: str
    comment: str
    updated_utc: str


def discover_recordings(root: str | Path) -> list[Path]:
    """Return selectable EDF and WFDB headers below a dataset directory."""

    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        return []
    recordings = list(base.rglob("*.edf")) + list(base.rglob("*.hea"))
    return sorted(
        {path.resolve() for path in recordings if path.is_file()},
        key=lambda path: (path.stem.casefold(), str(path).casefold()),
    )


def _run_detector_tracks(
    signal: np.ndarray,
    sampling_rate_hz: float,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Run the current independent timestamp candidates on one padded signal."""

    neurokit = detect_r_peaks(signal, sampling_rate_hz, method=DETECTION_METHOD)
    unsw = detect_unsw(signal, sampling_rate_hz, orientation="original")
    zhai = detect_zhai_template(signal, sampling_rate_hz, orientation="original")
    return (
        {
            "neurokit": np.asarray(neurokit.peak_samples, dtype=np.int64),
            "unsw": np.asarray(unsw.peak_samples, dtype=np.int64),
            "zhai": np.asarray(zhai.peak_samples, dtype=np.int64),
        },
        np.asarray(neurokit.cleaned_ecg, dtype=np.float64),
    )


def _visible_track_samples(
    local_tracks: dict[str, np.ndarray],
    *,
    process_start_sample: int,
    visible_start_sample: int,
    visible_end_sample: int,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for name, local in local_tracks.items():
        global_samples = process_start_sample + np.asarray(local, dtype=np.int64)
        output[name] = global_samples[
            (global_samples >= visible_start_sample)
            & (global_samples < visible_end_sample)
        ]
    return output


def _visible_reference_comparison(
    detector_global_samples: np.ndarray,
    reference_global_samples: np.ndarray,
    *,
    sampling_rate_hz: float,
    tolerance_ms: float,
    visible_start_sample: int,
    visible_end_sample: int,
) -> TrackReferenceComparison:
    agreement = compare_peak_sequences(
        detector_global_samples,
        reference_global_samples,
        sampling_rate_hz=sampling_rate_hz,
        tolerance_ms=tolerance_ms,
    )
    detection_visible = (
        (agreement.matched_primary_samples >= visible_start_sample)
        & (agreement.matched_primary_samples < visible_end_sample)
    )
    return TrackReferenceComparison(
        matched_detection_samples=agreement.matched_primary_samples[detection_visible],
        matched_reference_samples=agreement.matched_comparator_samples[detection_visible],
        false_positive_samples=agreement.unmatched_primary_samples[
            (agreement.unmatched_primary_samples >= visible_start_sample)
            & (agreement.unmatched_primary_samples < visible_end_sample)
        ],
        missed_reference_samples=agreement.unmatched_comparator_samples[
            (agreement.unmatched_comparator_samples >= visible_start_sample)
            & (agreement.unmatched_comparator_samples < visible_end_sample)
        ],
    )


def _extract_morphology_tracks(
    signal: np.ndarray,
    local_tracks: dict[str, np.ndarray],
    sampling_rate_hz: float,
    *,
    process_start_sample: int,
    reference_local_samples: np.ndarray | None = None,
    reference_symbols: np.ndarray | None = None,
) -> dict[str, VaronMorphologyResult]:
    """Run the 2015 120-ms/five-beat core with explicit anchor provenance."""

    outputs: dict[str, VaronMorphologyResult] = {}
    references = (
        np.asarray(reference_local_samples, dtype=np.int64)
        if reference_local_samples is not None
        else np.asarray([], dtype=np.int64)
    )
    symbols = (
        np.asarray(reference_symbols, dtype=object)
        if reference_symbols is not None
        else np.asarray([], dtype=object)
    )
    for name, anchors in local_tracks.items():
        context = None
        if references.size:
            context, _ = build_reference_context(
                anchors,
                references,
                symbols,
                sampling_rate_hz=sampling_rate_hz,
            )
        outputs[name] = extract_varon_morphology(
            signal,
            anchors,
            sampling_rate_hz,
            anchor_track=name,
            segment_start_s=process_start_sample / sampling_rate_hz,
            event_context=context,
        )
    if references.size:
        expert_context = {
            "primary_timestamp_sample": references,
            "qrs_supported": np.ones(references.size, dtype=bool),
            "comparator_match_count": np.ones(references.size, dtype=int),
            "comparator_offset_ms": np.zeros(references.size, dtype=float),
            "reference_symbol": symbols,
            "reference_match_error_ms": np.zeros(references.size, dtype=float),
        }
        import pandas as pd

        outputs["expert"] = extract_varon_morphology(
            signal,
            references,
            sampling_rate_hz,
            anchor_track="expert_atr",
            segment_start_s=process_start_sample / sampling_rate_hz,
            event_context=pd.DataFrame(expert_context),
        )
    return outputs


def detect_neurokit_window(
    path: str | Path,
    *,
    channel_label: str,
    start_s: float,
    duration_s: float,
    padding_s: float = 3.0,
) -> DetectionWindow:
    """Detect visible peaks with extra signal on both sides of the window.

    Padding reduces filter/detector boundary effects.  Only detections whose
    sample indices fall inside the requested visible interval are returned.
    The full EDF is never loaded into memory.
    """

    if duration_s < 1.0:
        raise ValueError("The display window must be at least 1 second")
    if duration_s > 300.0:
        raise ValueError("The display window is limited to 300 seconds")
    if padding_s < 3.0:
        raise ValueError("At least 3 seconds of detector padding are required")

    recording = inspect_edf(path)
    channels = [c for c in recording.channels if c.label == channel_label]
    if len(channels) != 1:
        raise ValueError(f"Channel {channel_label!r} is not unique in the EDF")
    channel = channels[0]
    fs = channel.sampling_rate_hz

    maximum_start_s = max(0.0, recording.duration_s - duration_s)
    visible_start_s = min(max(0.0, float(start_s)), maximum_start_s)
    visible_end_s = min(recording.duration_s, visible_start_s + duration_s)
    visible_start_sample = int(round(visible_start_s * fs))
    visible_end_sample = min(channel.sample_count, int(round(visible_end_s * fs)))

    process_start_s = max(0.0, visible_start_s - padding_s)
    process_end_s = min(recording.duration_s, visible_end_s + padding_s)
    process_start_sample = int(round(process_start_s * fs))
    process_end_sample = min(channel.sample_count, int(round(process_end_s * fs)))
    process_duration_s = (process_end_sample - process_start_sample) / fs

    segment = load_ecg_segment(
        path,
        channel_label=channel_label,
        start_s=process_start_sample / fs,
        duration_s=process_duration_s,
    )
    local_tracks, neurokit_cleaned = _run_detector_tracks(segment.samples, fs)
    visible_tracks = _visible_track_samples(
        local_tracks,
        process_start_sample=process_start_sample,
        visible_start_sample=visible_start_sample,
        visible_end_sample=visible_end_sample,
    )
    morphology_results = _extract_morphology_tracks(
        segment.samples,
        local_tracks,
        fs,
        process_start_sample=process_start_sample,
    )

    first = visible_start_sample - process_start_sample
    last = first + (visible_end_sample - visible_start_sample)
    raw = np.asarray(segment.samples[first:last], dtype=np.float64)
    cleaned = np.asarray(neurokit_cleaned[first:last], dtype=np.float64)
    if raw.size != visible_end_sample - visible_start_sample:
        raise RuntimeError("EDF window length changed during reading")

    return DetectionWindow(
        recording_path=recording.path,
        channel=channel,
        recording_duration_s=recording.duration_s,
        start_sample=visible_start_sample,
        end_sample=visible_end_sample,
        raw_ecg=raw,
        cleaned_ecg=cleaned,
        neurokit_peak_samples=visible_tracks["neurokit"],
        input_format="EDF",
        reference_beat_samples=np.asarray([], dtype=np.int64),
        reference_symbols=(),
        matched_detection_samples=np.asarray([], dtype=np.int64),
        matched_reference_samples=np.asarray([], dtype=np.int64),
        false_positive_samples=np.asarray([], dtype=np.int64),
        missed_reference_samples=np.asarray([], dtype=np.int64),
        detector_peak_samples=visible_tracks,
        reference_comparisons={},
        morphology_results=morphology_results,
    )


def inspect_wfdb(path: str | Path) -> RecordingInfo:
    """Inspect a single-segment WFDB record from its local header."""

    import wfdb

    header_path = Path(path).expanduser().resolve()
    if header_path.suffix.lower() != ".hea":
        header_path = header_path.with_suffix(".hea")
    if not header_path.is_file():
        raise FileNotFoundError(header_path)
    record_base = str(header_path.with_suffix(""))
    header = wfdb.rdheader(record_base)
    if header.sig_len is None or header.fs is None or header.n_sig is None:
        raise ValueError("WFDB header does not define signal length, sampling rate, and channels")
    names = list(header.sig_name or [f"channel_{i}" for i in range(header.n_sig)])
    units = list(header.units or ["" for _ in range(header.n_sig)])
    channels = tuple(
        ChannelInfo(
            index=index,
            label=str(names[index]),
            sampling_rate_hz=float(header.fs),
            sample_count=int(header.sig_len),
            physical_dimension=str(units[index] or ""),
            physical_min=float("nan"),
            physical_max=float("nan"),
        )
        for index in range(int(header.n_sig))
    )
    return RecordingInfo(
        path=str(header_path),
        duration_s=float(header.sig_len / header.fs),
        start_datetime="",
        channels=channels,
    )


def inspect_recording(path: str | Path) -> RecordingInfo:
    """Dispatch inspection by recording container."""

    suffix = Path(path).suffix.lower()
    if suffix == ".edf":
        return inspect_edf(path)
    if suffix in {".hea", ".dat"}:
        return inspect_wfdb(path)
    raise ValueError("Choose an EDF file or a WFDB .hea/.dat record")


def detect_wfdb_window(
    path: str | Path,
    *,
    channel_label: str,
    start_s: float,
    duration_s: float,
    padding_s: float = 3.0,
    tolerance_ms: float = 75.0,
) -> DetectionWindow:
    """Run NeuroKit and compare it with local WFDB ``atr`` beat annotations."""

    import wfdb

    if duration_s < 1.0 or duration_s > 300.0:
        raise ValueError("The display window must be between 1 and 300 seconds")
    if padding_s < 3.0:
        raise ValueError("At least 3 seconds of detector padding are required")

    recording = inspect_wfdb(path)
    channels = [channel for channel in recording.channels if channel.label == channel_label]
    if len(channels) != 1:
        raise ValueError(f"Channel {channel_label!r} is not unique in the WFDB record")
    channel = channels[0]
    fs = channel.sampling_rate_hz
    maximum_start_s = max(0.0, recording.duration_s - duration_s)
    visible_start_s = min(max(0.0, float(start_s)), maximum_start_s)
    visible_end_s = min(recording.duration_s, visible_start_s + duration_s)
    visible_start_sample = int(round(visible_start_s * fs))
    visible_end_sample = min(channel.sample_count, int(round(visible_end_s * fs)))
    process_start_sample = max(0, int(round((visible_start_s - padding_s) * fs)))
    process_end_sample = min(
        channel.sample_count, int(round((visible_end_s + padding_s) * fs))
    )

    record_base = str(Path(recording.path).with_suffix(""))
    record = wfdb.rdrecord(
        record_base,
        sampfrom=process_start_sample,
        sampto=process_end_sample,
        channels=[channel.index],
        physical=True,
    )
    if record.p_signal is None:
        raise ValueError("WFDB reader did not return a physical signal")
    signal = np.asarray(record.p_signal[:, 0], dtype=np.float64)
    local_tracks, neurokit_cleaned = _run_detector_tracks(signal, fs)
    global_tracks = {
        name: process_start_sample + samples
        for name, samples in local_tracks.items()
    }

    atr_path = Path(recording.path).with_suffix(".atr")
    reference_samples = np.asarray([], dtype=np.int64)
    reference_symbols_array = np.asarray([], dtype=object)
    matched_detection = np.asarray([], dtype=np.int64)
    matched_reference = np.asarray([], dtype=np.int64)
    false_positive = np.asarray([], dtype=np.int64)
    missed_reference = np.asarray([], dtype=np.int64)
    padded_reference = np.asarray([], dtype=np.int64)
    padded_symbols = np.asarray([], dtype=object)
    comparisons: dict[str, TrackReferenceComparison] = {}
    if atr_path.is_file():
        annotation = wfdb.rdann(
            record_base,
            "atr",
            sampfrom=process_start_sample,
            sampto=process_end_sample,
            shift_samps=False,
        )
        annotation_samples = np.asarray(annotation.sample, dtype=np.int64)
        annotation_symbols = np.asarray(annotation.symbol, dtype=object)
        is_beat = np.asarray(
            [symbol in MIT_BIH_BEAT_SYMBOLS for symbol in annotation_symbols], dtype=bool
        )
        padded_reference = annotation_samples[is_beat]
        padded_symbols = annotation_symbols[is_beat]
        comparisons = {
            name: _visible_reference_comparison(
                samples,
                padded_reference,
                sampling_rate_hz=fs,
                tolerance_ms=tolerance_ms,
                visible_start_sample=visible_start_sample,
                visible_end_sample=visible_end_sample,
            )
            for name, samples in global_tracks.items()
        }
        neurokit_comparison = comparisons["neurokit"]
        matched_detection = neurokit_comparison.matched_detection_samples
        matched_reference = neurokit_comparison.matched_reference_samples
        false_positive = neurokit_comparison.false_positive_samples
        missed_reference = neurokit_comparison.missed_reference_samples
        reference_visible = (
            (annotation_samples >= visible_start_sample)
            & (annotation_samples < visible_end_sample)
            & is_beat
        )
        reference_samples = annotation_samples[reference_visible]
        reference_symbols_array = annotation_symbols[reference_visible]

    first = visible_start_sample - process_start_sample
    last = first + (visible_end_sample - visible_start_sample)
    visible_tracks = _visible_track_samples(
        local_tracks,
        process_start_sample=process_start_sample,
        visible_start_sample=visible_start_sample,
        visible_end_sample=visible_end_sample,
    )
    morphology_results = _extract_morphology_tracks(
        signal,
        local_tracks,
        fs,
        process_start_sample=process_start_sample,
        reference_local_samples=padded_reference - process_start_sample,
        reference_symbols=padded_symbols,
    )
    return DetectionWindow(
        recording_path=recording.path,
        channel=channel,
        recording_duration_s=recording.duration_s,
        start_sample=visible_start_sample,
        end_sample=visible_end_sample,
        raw_ecg=np.asarray(signal[first:last], dtype=np.float64),
        cleaned_ecg=np.asarray(neurokit_cleaned[first:last], dtype=np.float64),
        neurokit_peak_samples=visible_tracks["neurokit"],
        input_format="WFDB",
        reference_beat_samples=reference_samples,
        reference_symbols=tuple(str(value) for value in reference_symbols_array.tolist()),
        matched_detection_samples=matched_detection,
        matched_reference_samples=matched_reference,
        false_positive_samples=false_positive,
        missed_reference_samples=missed_reference,
        detector_peak_samples=visible_tracks,
        reference_comparisons=comparisons,
        morphology_results=morphology_results,
    )


def detect_recording_window(
    path: str | Path,
    *,
    channel_label: str,
    start_s: float,
    duration_s: float,
) -> DetectionWindow:
    suffix = Path(path).suffix.lower()
    if suffix == ".edf":
        return detect_neurokit_window(
            path,
            channel_label=channel_label,
            start_s=start_s,
            duration_s=duration_s,
        )
    if suffix in {".hea", ".dat"}:
        return detect_wfdb_window(
            path,
            channel_label=channel_label,
            start_s=start_s,
            duration_s=duration_s,
        )
    raise ValueError("Choose an EDF file or a WFDB .hea/.dat record")


class ReviewStore:
    """CSV-backed manual labels for one EDF channel."""

    FIELDNAMES = [
        "recording_path",
        "channel",
        "sampling_rate_hz",
        "sample_index",
        "time_s",
        "source",
        "label",
        "comment",
        "updated_utc",
    ]

    def __init__(
        self,
        *,
        recording_path: str | Path,
        channel: str,
        sampling_rate_hz: float,
        output_dir: str | Path,
    ) -> None:
        self.recording_path = str(Path(recording_path).resolve())
        self.channel = channel
        self.sampling_rate_hz = float(sampling_rate_hz)
        safe_stem = _safe_filename(Path(recording_path).stem)
        safe_channel = _safe_filename(channel)
        self.output_path = Path(output_dir).resolve() / (
            f"{safe_stem}__{safe_channel}__rpeak_review.csv"
        )
        self.records: dict[int, ReviewRecord] = {}
        self.load()

    def load(self) -> None:
        self.records.clear()
        if not self.output_path.is_file():
            return
        with self.output_path.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row.get("recording_path") != self.recording_path:
                    continue
                if row.get("channel") != self.channel:
                    continue
                record = ReviewRecord(
                    recording_path=row["recording_path"],
                    channel=row["channel"],
                    sampling_rate_hz=float(row["sampling_rate_hz"]),
                    sample_index=int(row["sample_index"]),
                    time_s=float(row["time_s"]),
                    source=row["source"],
                    label=row["label"],
                    comment=row.get("comment", ""),
                    updated_utc=row["updated_utc"],
                )
                self.records[record.sample_index] = record

    def set_label(
        self,
        sample_index: int,
        *,
        source: str,
        label: str,
        comment: str = "",
    ) -> ReviewRecord:
        if label not in REVIEW_LABELS:
            raise ValueError(f"Unknown review label {label!r}")
        if source not in {"neurokit", "manual"}:
            raise ValueError(f"Unknown review source {source!r}")
        sample_index = int(sample_index)
        record = ReviewRecord(
            recording_path=self.recording_path,
            channel=self.channel,
            sampling_rate_hz=self.sampling_rate_hz,
            sample_index=sample_index,
            time_s=sample_index / self.sampling_rate_hz,
            source=source,
            label=label,
            comment=comment.strip(),
            updated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self.records[sample_index] = record
        self.save()
        return record

    def clear_label(self, sample_index: int) -> bool:
        removed = self.records.pop(int(sample_index), None) is not None
        if removed:
            self.save()
        return removed

    def records_between(self, first_sample: int, last_sample: int) -> list[ReviewRecord]:
        return [
            record
            for sample, record in sorted(self.records.items())
            if first_sample <= sample < last_sample
        ]

    def save(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDNAMES)
            writer.writeheader()
            for record in sorted(self.records.values(), key=lambda item: item.sample_index):
                writer.writerow(
                    {
                        "recording_path": record.recording_path,
                        "channel": record.channel,
                        "sampling_rate_hz": f"{record.sampling_rate_hz:.10g}",
                        "sample_index": record.sample_index,
                        "time_s": f"{record.time_s:.9f}",
                        "source": record.source,
                        "label": record.label,
                        "comment": record.comment,
                        "updated_utc": record.updated_utc,
                    }
                )


def _safe_filename(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return result.strip("._") or "unnamed"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


class RPeakReviewerApp:
    """Tk/Matplotlib EDF/WFDB interface; imported lazily for headless tests."""

    def __init__(self, root, *, initial_recording: str | Path | None = None) -> None:
        import tkinter as tk
        from tkinter import ttk

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        from matplotlib.figure import Figure

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.root.title("ECG Cascade Branch Reviewer")
        self.root.geometry("1500x920")
        self.root.minsize(1100, 720)

        self.recording: RecordingInfo | None = None
        self.window: DetectionWindow | None = None
        self.review_store: ReviewStore | None = None
        self.selected_peak_sample: int | None = None
        self.cursor_sample: int | None = None
        self.selected_region_samples: tuple[int, int] | None = None
        self._region_drag_start_sample: int | None = None
        self.record_catalog: dict[str, Path] = {}

        self.file_var = tk.StringVar(value="No recording loaded")
        self.channel_var = tk.StringVar()
        self.duration_var = tk.StringVar(value="15")
        self.start_var = tk.DoubleVar(value=0.0)
        self.start_text_var = tk.StringVar(value="0.000")
        self.show_cleaned_var = tk.BooleanVar(value=False)
        self.record_choice_var = tk.StringVar()
        self.active_track_var = tk.StringVar(value="neurokit")
        self.show_track_vars = {
            "neurokit": tk.BooleanVar(value=True),
            "unsw": tk.BooleanVar(value=True),
            "zhai": tk.BooleanVar(value=True),
            "expert": tk.BooleanVar(value=True),
        }
        self.show_morphology_var = tk.BooleanVar(value=True)
        self.expert_label_filter_var = tk.StringVar(value="All labels")
        self.status_var = tk.StringVar(
            value="Choose a record. Expert comparison is available only when a WFDB .atr file exists."
        )
        self.selection_var = tk.StringVar(value="No peak selected")
        self.region_var = tk.StringVar(
            value="Right-drag over the ECG to inspect a region and its expert beat labels."
        )

        self._build_controls()

        # Keep the full three-panel figure visible on ordinary laptop screens.
        # Tk will expand the canvas on larger displays, while this smaller
        # requested height prevents the lower morphology/RR axes being clipped.
        self.figure = Figure(figsize=(12.5, 3.0), dpi=100, constrained_layout=True)
        self.ax_ecg, self.ax_morph, self.ax_rr = self.figure.subplots(
            3, 1, sharex=True, gridspec_kw={"height_ratios": [3.4, 1.5, 1.0]}
        )
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().configure(height=300)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar_frame = ttk.Frame(self.root)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side=tk.LEFT)
        self.canvas.mpl_connect("button_press_event", self._on_plot_press)
        self.canvas.mpl_connect("button_release_event", self._on_plot_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_plot_motion)
        self.hover_annotation = None

        status_frame = ttk.Frame(self.root, padding=(8, 3, 8, 7))
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(status_frame, textvariable=self.selection_var).pack(side=tk.LEFT)
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.RIGHT)
        ttk.Label(self.root, textvariable=self.region_var, padding=(8, 2, 8, 2)).pack(
            side=tk.BOTTOM, fill=tk.X
        )

        self.root.bind("<Left>", lambda _event: self._shift_window(-0.5))
        self.root.bind("<Right>", lambda _event: self._shift_window(0.5))
        self.root.bind("<Key-c>", lambda _event: self._label_selected("confirmed_r_peak"))
        self.root.bind("<Key-f>", lambda _event: self._label_selected("false_positive"))
        self.root.bind("<Key-u>", lambda _event: self._label_selected("uncertain"))
        self.root.bind("<Key-m>", lambda _event: self._add_missed_at_cursor())
        self.root.bind("<Delete>", lambda _event: self._clear_selected_label())
        self.root.bind("<Key-n>", lambda _event: self._select_adjacent_peak(1))
        self.root.bind("<Key-p>", lambda _event: self._select_adjacent_peak(-1))
        self.root.bind("<Key-z>", lambda _event: self._zoom(0.5))
        self.root.bind("<Key-x>", lambda _event: self._zoom(2.0))
        self.root.bind("<Escape>", lambda _event: self._clear_region())

        self._refresh_record_catalog(_project_root() / "Datasets")

        if initial_recording is not None:
            self.root.after(50, lambda: self.open_recording(initial_recording))

    def _build_controls(self) -> None:
        from tkinter import ttk

        outer = ttk.Frame(self.root, padding=8)
        outer.pack(side=self.tk.TOP, fill=self.tk.X)

        row0 = ttk.Frame(outer)
        row0.pack(fill=self.tk.X, pady=(0, 5))
        ttk.Label(row0, text="Dataset record:").pack(side=self.tk.LEFT)
        self.record_combo = ttk.Combobox(
            row0,
            textvariable=self.record_choice_var,
            state="readonly",
            width=34,
        )
        self.record_combo.pack(side=self.tk.LEFT, padx=(4, 6))
        self.record_combo.bind(
            "<<ComboboxSelected>>", lambda _event: self._open_selected_record()
        )
        ttk.Button(
            row0,
            text="Choose dataset folder",
            command=self._choose_dataset_folder,
        ).pack(side=self.tk.LEFT, padx=(0, 6))
        ttk.Button(row0, text="Open individual ECG", command=self._choose_recording).pack(
            side=self.tk.LEFT
        )
        row1 = ttk.Frame(outer)
        row1.pack(fill=self.tk.X, pady=(0, 5))
        ttk.Label(row1, text="Loaded:").pack(side=self.tk.LEFT)
        ttk.Label(row1, textvariable=self.file_var, width=20).pack(
            side=self.tk.LEFT, padx=(4, 14)
        )
        ttk.Label(row1, text="ECG channel:").pack(side=self.tk.LEFT)
        self.channel_combo = ttk.Combobox(
            row1, textvariable=self.channel_var, state="readonly", width=18
        )
        self.channel_combo.pack(side=self.tk.LEFT, padx=(4, 12))
        self.channel_combo.bind("<<ComboboxSelected>>", lambda _event: self.analyze())
        ttk.Label(row1, text="Window (s):").pack(side=self.tk.LEFT)
        self.duration_combo = ttk.Combobox(
            row1,
            textvariable=self.duration_var,
            values=("1", "2", "5", "10", "15", "30", "60", "120", "300"),
            width=6,
        )
        self.duration_combo.pack(side=self.tk.LEFT, padx=(4, 8))
        ttk.Button(row1, text="Analyze visible window", command=self.analyze).pack(
            side=self.tk.LEFT
        )
        ttk.Checkbutton(
            row1,
            text="Cleaned trace",
            variable=self.show_cleaned_var,
            command=self._draw,
        ).pack(side=self.tk.LEFT, padx=(12, 0))

        row2 = ttk.Frame(outer)
        row2.pack(fill=self.tk.X, pady=(0, 5))
        ttk.Button(row2, text="◀ window", command=lambda: self._shift_window(-1.0)).pack(
            side=self.tk.LEFT
        )
        ttk.Button(row2, text="◀ half", command=lambda: self._shift_window(-0.5)).pack(
            side=self.tk.LEFT, padx=(4, 8)
        )
        ttk.Label(row2, text="Start (s):").pack(side=self.tk.LEFT)
        self.start_entry = ttk.Entry(row2, textvariable=self.start_text_var, width=12)
        self.start_entry.pack(side=self.tk.LEFT, padx=(4, 6))
        self.start_entry.bind("<Return>", lambda _event: self._start_from_entry())
        self.time_scale = self.tk.Scale(
            row2,
            variable=self.start_var,
            from_=0.0,
            to=1.0,
            orient=self.tk.HORIZONTAL,
            resolution=0.1,
            showvalue=False,
            length=520,
        )
        self.time_scale.pack(side=self.tk.LEFT, fill=self.tk.X, expand=True)
        self.time_scale.bind("<ButtonRelease-1>", lambda _event: self._start_from_slider())
        ttk.Button(row2, text="half ▶", command=lambda: self._shift_window(0.5)).pack(
            side=self.tk.LEFT, padx=(8, 4)
        )
        ttk.Button(row2, text="window ▶", command=lambda: self._shift_window(1.0)).pack(
            side=self.tk.LEFT
        )

        row3 = ttk.Frame(outer)
        row3.pack(fill=self.tk.X)
        ttk.Button(row3, text="Previous (P)", command=lambda: self._select_adjacent_peak(-1)).pack(
            side=self.tk.LEFT
        )
        ttk.Button(row3, text="Next (N)", command=lambda: self._select_adjacent_peak(1)).pack(
            side=self.tk.LEFT, padx=(4, 12)
        )
        ttk.Button(
            row3, text="Confirm (C)", command=lambda: self._label_selected("confirmed_r_peak")
        ).pack(side=self.tk.LEFT)
        ttk.Button(
            row3, text="False (F)", command=lambda: self._label_selected("false_positive")
        ).pack(side=self.tk.LEFT, padx=4)
        ttk.Button(
            row3, text="Uncertain (U)", command=lambda: self._label_selected("uncertain")
        ).pack(side=self.tk.LEFT)
        ttk.Button(row3, text="Add missed (M)", command=self._add_missed_at_cursor).pack(
            side=self.tk.LEFT, padx=(12, 4)
        )
        ttk.Button(row3, text="Clear (Delete)", command=self._clear_selected_label).pack(
            side=self.tk.LEFT
        )
        ttk.Button(row3, text="Review CSV", command=self._open_review_csv).pack(
            side=self.tk.RIGHT
        )

        row4 = ttk.Frame(outer)
        row4.pack(fill=self.tk.X, pady=(5, 0))
        ttk.Label(row4, text="Evaluate/plot RR from:").pack(side=self.tk.LEFT)
        self.active_track_combo = ttk.Combobox(
            row4,
            textvariable=self.active_track_var,
            state="readonly",
            values=("neurokit", "unsw", "zhai", "expert"),
            width=12,
        )
        self.active_track_combo.pack(side=self.tk.LEFT, padx=(4, 10))
        self.active_track_combo.bind(
            "<<ComboboxSelected>>", self._active_track_changed
        )
        ttk.Label(row4, text="Visible markers:").pack(side=self.tk.LEFT)
        for name in ("neurokit", "unsw", "zhai", "expert"):
            ttk.Checkbutton(
                row4,
                text=TRACK_DISPLAY_NAMES[name],
                variable=self.show_track_vars[name],
                command=self._draw,
            ).pack(side=self.tk.LEFT, padx=(3, 0))

        row5 = ttk.Frame(outer)
        row5.pack(fill=self.tk.X, pady=(3, 0))
        ttk.Checkbutton(
            row5,
            text="Five-beat morphology",
            variable=self.show_morphology_var,
            command=self._draw,
        ).pack(side=self.tk.LEFT, padx=(0, 8))
        ttk.Label(row5, text="Highlight expert label:").pack(side=self.tk.LEFT)
        self.expert_label_combo = ttk.Combobox(
            row5,
            textvariable=self.expert_label_filter_var,
            state="readonly",
            values=("All labels",),
            width=18,
        )
        self.expert_label_combo.pack(side=self.tk.LEFT, padx=(4, 0))
        self.expert_label_combo.bind(
            "<<ComboboxSelected>>", lambda _event: self._draw()
        )
        ttk.Label(
            row5,
            text="Z/X zoom | arrows move | right-drag selects region | hover shows details",
        ).pack(side=self.tk.RIGHT)

    def _refresh_record_catalog(self, directory: str | Path) -> None:
        recordings = discover_recordings(directory)
        catalog: dict[str, Path] = {}
        for path in recordings:
            base = f"{path.stem} | {path.parent.name}"
            label = base
            suffix = 2
            while label in catalog:
                label = f"{base} ({suffix})"
                suffix += 1
            catalog[label] = path
        self.record_catalog = catalog
        values = list(catalog)
        if hasattr(self, "record_combo"):
            self.record_combo["values"] = values
        if values:
            self.record_choice_var.set(values[0])
            self.status_var.set(
                f"Found {len(values)} selectable records below {Path(directory).name}."
            )
        else:
            self.record_choice_var.set("")
            self.status_var.set(f"No EDF or WFDB records found below {directory}.")

    def _choose_dataset_folder(self) -> None:
        from tkinter import filedialog

        initial = _project_root() / "Datasets"
        selected = filedialog.askdirectory(
            title="Choose dataset folder",
            initialdir=str(initial if initial.is_dir() else _project_root()),
        )
        if selected:
            self._refresh_record_catalog(selected)

    def _open_selected_record(self) -> None:
        path = self.record_catalog.get(self.record_choice_var.get())
        if path is not None:
            self.open_recording(path)

    def _active_track_changed(self, _event=None) -> None:
        """Redraw and report the evidence status of the selected branch."""

        self._draw()
        if self.window is None:
            return
        track = self.active_track_var.get()
        comparison = self.window.comparison_for(track)
        if comparison is not None:
            metrics = comparison.metrics()
            self.status_var.set(
                f"{TRACK_DISPLAY_NAMES[track]} versus expert .atr: "
                f"TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}, "
                f"F1={metrics['f1']:.4f} at +/-75 ms."
            )
        elif track == "expert":
            self.status_var.set(
                "Expert timestamps selected: morphology is the reference-aligned "
                "feature trace; no detector TP/FP/FN is applicable."
            )
        else:
            self.status_var.set(
                f"{TRACK_DISPLAY_NAMES.get(track, track)} selected. This recording "
                "has no expert .atr annotations, so accuracy is not measurable here."
            )

    def _choose_recording(self) -> None:
        from tkinter import filedialog

        initial = _project_root() / "Datasets"
        selected = filedialog.askopenfilename(
            title="Open EDF or WFDB recording",
            initialdir=str(initial if initial.is_dir() else _project_root()),
            filetypes=[
                ("Supported ECG", "*.edf *.hea *.dat"),
                ("EDF recordings", "*.edf"),
                ("WFDB headers", "*.hea"),
                ("All files", "*.*"),
            ],
        )
        if selected:
            self.open_recording(selected)

    def open_recording(self, path: str | Path) -> None:
        from tkinter import messagebox

        try:
            recording = inspect_recording(path)
        except Exception as exc:
            messagebox.showerror("Could not open recording", str(exc))
            return

        self.recording = recording
        self.file_var.set(Path(recording.path).name)
        labels = [channel.label for channel in recording.channels]
        self.channel_combo["values"] = labels
        ecg_candidates = [label for label in labels if "ECG" in label.upper() or "EKG" in label.upper()]
        self.channel_var.set(ecg_candidates[0] if ecg_candidates else labels[0])
        self.start_var.set(0.0)
        self.start_text_var.set("0.000")
        self.selected_peak_sample = None
        self.cursor_sample = None
        self.selected_region_samples = None
        self.status_var.set(
            f"Loaded {len(labels)} channels, duration {recording.duration_s:.1f} s. "
            "Verify the selected channel visually."
        )
        self.analyze()

    def open_edf(self, path: str | Path) -> None:
        """Backward-compatible alias for earlier launch/test code."""

        self.open_recording(path)

    def _duration_s(self) -> float:
        duration = float(self.duration_var.get())
        if not 1.0 <= duration <= 300.0:
            raise ValueError("Window duration must be between 1 and 300 seconds")
        return duration

    def analyze(self) -> None:
        from tkinter import messagebox

        if self.recording is None:
            return
        self.status_var.set(
            "Running NeuroKit, UNSW/Khamis, Zhai, and five-beat morphology…"
        )
        self.root.update_idletasks()
        try:
            duration = self._duration_s()
            start = float(self.start_var.get())
            window = detect_recording_window(
                self.recording.path,
                channel_label=self.channel_var.get(),
                start_s=start,
                duration_s=duration,
            )
        except Exception as exc:
            messagebox.showerror("Detection failed", str(exc))
            self.status_var.set("Detection failed; no labels were changed.")
            return

        self.window = window
        maximum_start = max(0.0, window.recording_duration_s - duration)
        self.time_scale.configure(to=maximum_start)
        self.start_var.set(window.start_s)
        self.start_text_var.set(f"{window.start_s:.3f}")
        output_dir = _project_root() / "output" / "rpeak_review"
        self.review_store = ReviewStore(
            recording_path=window.recording_path,
            channel=window.channel.label,
            sampling_rate_hz=window.sampling_rate_hz,
            output_dir=output_dir,
        )
        present_symbols = sorted(set(window.reference_symbols))
        label_values = ["All labels"] + [
            f"{symbol}: {MIT_BIH_SYMBOL_NAMES.get(symbol, 'expert beat')}"
            for symbol in present_symbols
        ]
        self.expert_label_combo["values"] = label_values
        if self.expert_label_filter_var.get() not in label_values:
            self.expert_label_filter_var.set("All labels")
        if self.active_track_var.get() == "expert" and not window.reference_beat_samples.size:
            self.active_track_var.set("neurokit")
        if self.selected_peak_sample is not None and not (
            window.start_sample <= self.selected_peak_sample < window.end_sample
        ):
            self.selected_peak_sample = None
        self._draw()
        if window.reference_beat_samples.size:
            track = self.active_track_var.get()
            comparison = window.comparison_for(track)
            if comparison is None:
                self.status_var.set(
                    "Expert timestamps selected: morphology is a feature reference, not a prediction."
                )
            else:
                metrics = comparison.metrics()
                self.status_var.set(
                    f"{TRACK_DISPLAY_NAMES[track]} vs expert .atr (±75 ms): "
                    f"TP {metrics['tp']}, FP {metrics['fp']}, FN {metrics['fn']}, "
                    f"Se {metrics['sensitivity']:.3f}, PPV {metrics['ppv']:.3f}, "
                    f"F1 {metrics['f1']:.3f}"
                )
        else:
            self.status_var.set(
                "No expert .atr file: detector outputs and morphology are unverified context. "
                f"Manual NeuroKit labels auto-save to {self.review_store.output_path.name}."
            )

    def _draw(self) -> None:
        if not hasattr(self, "ax_ecg"):
            return
        self.ax_ecg.clear()
        self.ax_morph.clear()
        self.ax_rr.clear()
        window = self.window
        if window is None:
            self.ax_ecg.text(
                0.5,
                0.5,
                "Choose a dataset record or open an ECG to begin",
                ha="center",
                va="center",
                transform=self.ax_ecg.transAxes,
            )
            self.canvas.draw_idle()
            return

        active_track = self.active_track_var.get()
        if active_track == "expert" and not window.reference_beat_samples.size:
            active_track = "neurokit"
            self.active_track_var.set(active_track)
        times = window.times_s
        raw = window.raw_ecg
        self.ax_ecg.plot(times, raw, color="#17365D", linewidth=0.85, label="Raw ECG")
        if self.show_cleaned_var.get():
            self.ax_ecg.plot(
                times,
                window.cleaned_ecg,
                color="#888888",
                linewidth=0.7,
                alpha=0.8,
                label="NeuroKit-cleaned ECG",
            )

        selected_filter = self.expert_label_filter_var.get()
        selected_symbol = (
            selected_filter.split(":", 1)[0]
            if selected_filter != "All labels"
            else None
        )
        if selected_symbol is not None:
            for sample, symbol in zip(
                window.reference_beat_samples, window.reference_symbols
            ):
                if symbol == selected_symbol:
                    center = sample / window.sampling_rate_hz
                    self.ax_ecg.axvspan(
                        center - 0.06,
                        center + 0.06,
                        color="#FFE699",
                        alpha=0.45,
                        zorder=0,
                    )

        peaks = window.peaks_for(active_track)
        store_records = self.review_store.records if self.review_store is not None else {}
        if window.reference_beat_samples.size and self.show_track_vars["expert"].get():
            self._scatter_samples(
                window.reference_beat_samples,
                color="#111111",
                marker="d",
                label="Expert .atr beat",
                size=34,
                facecolors="none",
            )
            if window.end_s - window.start_s <= 60.0:
                amplitude_span = max(float(np.ptp(raw)), 1e-12)
                for sample, symbol in zip(
                    window.reference_beat_samples, window.reference_symbols
                ):
                    x, y = self._sample_xy(int(sample))
                    self.ax_ecg.text(
                        x,
                        y + 0.05 * amplitude_span,
                        symbol,
                        fontsize=7,
                        ha="center",
                        va="bottom",
                        color="#111111",
                        zorder=7,
                    )

        track_style = {
            "neurokit": ("#008C95", "x"),
            "unsw": ("#6A3D9A", "+"),
            "zhai": ("#CC6600", "1"),
        }
        for track, (color, marker) in track_style.items():
            if not self.show_track_vars[track].get():
                continue
            track_peaks = window.peaks_for(track)
            if track == active_track and window.reference_beat_samples.size:
                comparison = window.comparison_for(track)
                if comparison is not None:
                    self._scatter_samples(
                        comparison.matched_detection_samples,
                        color="#1B7F3A",
                        marker=marker,
                        label=f"{TRACK_DISPLAY_NAMES[track]} matched expert",
                        size=50,
                    )
                    self._scatter_samples(
                        comparison.false_positive_samples,
                        color="#C00000",
                        marker="X",
                        label=f"{TRACK_DISPLAY_NAMES[track]} false positive",
                        size=58,
                    )
                    self._scatter_samples(
                        comparison.missed_reference_samples,
                        color="#E59400",
                        marker="v",
                        label=f"Expert beat missed by {TRACK_DISPLAY_NAMES[track]}",
                        size=64,
                    )
                    continue
            if not window.reference_beat_samples.size and track == "neurokit":
                continue
            self._scatter_samples(
                track_peaks,
                color=color,
                marker=marker,
                label=TRACK_DISPLAY_NAMES[track],
                size=38,
                alpha=0.70 if track != active_track else 1.0,
            )

        if not window.reference_beat_samples.size and self.show_track_vars["neurokit"].get():
            unreviewed: list[int] = []
            confirmed: list[int] = []
            false: list[int] = []
            uncertain: list[int] = []
            for sample in peaks.tolist():
                record = store_records.get(sample)
                if record is None:
                    unreviewed.append(sample)
                elif record.label == "confirmed_r_peak":
                    confirmed.append(sample)
                elif record.label == "false_positive":
                    false.append(sample)
                else:
                    uncertain.append(sample)

            self._scatter_samples(unreviewed, color="#008C95", marker="x", label="NeuroKit: unreviewed", size=40)
            self._scatter_samples(confirmed, color="#1B7F3A", marker="o", label="Manually confirmed", size=42)
            self._scatter_samples(false, color="#C00000", marker="X", label="Marked false", size=55)
            self._scatter_samples(uncertain, color="#E59400", marker="D", label="Marked uncertain", size=42)

            manual = [
                record.sample_index
                for record in store_records.values()
                if record.source == "manual"
                and record.label == "manual_missed_r_peak"
                and window.start_sample <= record.sample_index < window.end_sample
            ]
            self._scatter_samples(manual, color="#7A3DB8", marker="^", label="Manually added missed R", size=62)

        if self.selected_region_samples is not None:
            first_region, last_region = self.selected_region_samples
            for axis in (self.ax_ecg, self.ax_morph, self.ax_rr):
                axis.axvspan(
                    first_region / window.sampling_rate_hz,
                    last_region / window.sampling_rate_hz,
                    color="#5B9BD5",
                    alpha=0.12,
                    zorder=0,
                )

        if self.cursor_sample is not None and window.start_sample <= self.cursor_sample < window.end_sample:
            self.ax_ecg.axvline(
                self.cursor_sample / window.sampling_rate_hz,
                color="#E59400",
                linewidth=1.0,
                linestyle="--",
                alpha=0.8,
                label="Review cursor",
            )
        if self.selected_peak_sample is not None and window.start_sample <= self.selected_peak_sample < window.end_sample:
            x, y = self._sample_xy(self.selected_peak_sample)
            self.ax_ecg.scatter(
                [x], [y], s=120, facecolors="none", edgecolors="#111111", linewidths=1.5, zorder=7
            )

        rr_ms = np.diff(peaks) * 1000.0 / window.sampling_rate_hz
        if rr_ms.size:
            self.ax_rr.plot(
                peaks[1:] / window.sampling_rate_hz,
                rr_ms,
                color="#17365D",
                marker=".",
                linewidth=0.8,
            )
            comparison = window.comparison_for(active_track)
            if comparison is not None and comparison.false_positive_samples.size:
                false_rr_mask = np.isin(peaks[1:], comparison.false_positive_samples)
                self.ax_rr.scatter(
                    peaks[1:][false_rr_mask] / window.sampling_rate_hz,
                    rr_ms[false_rr_mask],
                    color="#C00000",
                    marker="X",
                    s=36,
                    zorder=5,
                )

        morphology = window.morphology_results.get(active_track)
        if self.show_morphology_var.get() and morphology is not None:
            features = morphology.features
            visible = features[
                (features["window_end_time_s"] >= window.start_s)
                & (features["window_end_time_s"] < window.end_s)
                & features["published_varon_core_defined"].astype(bool)
            ]
            if not visible.empty:
                lambda_columns = [f"lambda{i}" for i in range(1, 6)]
                values = visible[lambda_columns].to_numpy(dtype=float)
                totals = np.sum(values, axis=1, keepdims=True)
                fractions = np.divide(
                    values,
                    totals,
                    out=np.full_like(values, np.nan),
                    where=totals > 0,
                )
                morphology_times = visible["window_end_time_s"].to_numpy(dtype=float)
                colors = ("#17365D", "#3D6FA8", "#70AD47", "#ED7D31", "#A52A2A")
                for index, color in enumerate(colors):
                    self.ax_morph.plot(
                        morphology_times,
                        fractions[:, index],
                        color=color,
                        linewidth=1.0,
                        label=f"λ{index + 1} energy fraction",
                    )
        if (
            self.show_morphology_var.get()
            and active_track != "expert"
            and "expert" in window.morphology_results
        ):
            expert_features = window.morphology_results["expert"].features
            expert_visible = expert_features[
                (expert_features["window_end_time_s"] >= window.start_s)
                & (expert_features["window_end_time_s"] < window.end_s)
                & expert_features["published_varon_core_defined"].astype(bool)
            ]
            if not expert_visible.empty:
                expert_values = expert_visible[
                    [f"lambda{i}" for i in range(1, 6)]
                ].to_numpy(dtype=float)
                expert_total = np.sum(expert_values, axis=1)
                expert_higher = np.divide(
                    np.sum(expert_values[:, 2:], axis=1),
                    expert_total,
                    out=np.full(expert_total.shape, np.nan),
                    where=expert_total > 0,
                )
                self.ax_morph.plot(
                    expert_visible["window_end_time_s"],
                    expert_higher,
                    color="#111111",
                    linestyle="--",
                    linewidth=1.3,
                    label="Expert-timestamp λ3–λ5 fraction",
                )
        if not self.show_morphology_var.get() or morphology is None:
            self.ax_morph.text(
                0.5,
                0.5,
                "Five-beat morphology display disabled or unavailable",
                transform=self.ax_morph.transAxes,
                ha="center",
                va="center",
            )

        self.ax_morph.set_ylabel("Eigenvalue\nenergy fraction")
        self.ax_morph.set_title(
            "Varon 2015: 120-ms waveforms × five beats (display-normalized; no seizure prediction)",
            fontsize=9,
        )
        self.ax_morph.grid(alpha=0.2)
        morph_handles, morph_labels = self.ax_morph.get_legend_handles_labels()
        if morph_handles:
            self.ax_morph.legend(
                morph_handles,
                morph_labels,
                loc="upper right",
                fontsize=7,
                ncol=3,
            )
        self.ax_rr.set_ylabel("RR (ms)")
        self.ax_rr.set_xlabel("Time from recording start (s)")
        self.ax_rr.grid(alpha=0.2)
        self.ax_ecg.set_ylabel(window.channel.physical_dimension or "EDF physical units")
        self.ax_ecg.set_title(
            f"{Path(window.recording_path).name} | {window.channel.label} | "
            f"active={TRACK_DISPLAY_NAMES[active_track]} | {window.input_format} | "
            f"{window.start_s:.3f}–{window.end_s:.3f} s"
        )
        self.ax_ecg.grid(alpha=0.2)
        handles, labels = self.ax_ecg.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            self.ax_ecg.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8, ncol=2)
        self.ax_ecg.set_xlim(window.start_s, window.end_s)
        self.hover_annotation = self.ax_ecg.annotate(
            "",
            xy=(0, 0),
            xytext=(12, 18),
            textcoords="offset points",
            bbox={"boxstyle": "round", "fc": "#FFFDEB", "ec": "#555555", "alpha": 0.95},
            arrowprops={"arrowstyle": "->", "color": "#555555"},
            fontsize=8,
            visible=False,
            zorder=20,
        )
        self.canvas.draw_idle()

    def _scatter_samples(
        self,
        samples: Iterable[int],
        *,
        color: str,
        marker: str,
        label: str,
        size: float,
        facecolors: str | None = None,
        alpha: float = 1.0,
    ) -> None:
        values = np.asarray(list(samples), dtype=np.int64)
        if values.size == 0:
            return
        xy = [self._sample_xy(int(sample)) for sample in values]
        options = {}
        if facecolors is not None:
            options["facecolors"] = facecolors
            options["edgecolors"] = color
        else:
            options["color"] = color
        self.ax_ecg.scatter(
            [item[0] for item in xy],
            [item[1] for item in xy],
            marker=marker,
            s=size,
            label=label,
            zorder=5,
            alpha=alpha,
            **options,
        )

    def _sample_xy(self, sample: int) -> tuple[float, float]:
        if self.window is None:
            raise RuntimeError("No visible window")
        local = int(sample) - self.window.start_sample
        local = min(max(local, 0), self.window.raw_ecg.size - 1)
        return sample / self.window.sampling_rate_hz, float(self.window.raw_ecg[local])

    def _on_plot_press(self, event) -> None:
        if (
            self.window is None
            or event.inaxes not in {self.ax_ecg, self.ax_morph, self.ax_rr}
            or event.xdata is None
        ):
            return
        sample = int(round(event.xdata * self.window.sampling_rate_hz))
        sample = min(max(sample, self.window.start_sample), self.window.end_sample - 1)
        if event.button == 3:
            self._region_drag_start_sample = sample
            return
        if event.button != 1:
            return
        self.cursor_sample = sample
        candidate_arrays = list(self.window.detector_peak_samples.values())
        if self.window.reference_beat_samples.size:
            candidate_arrays.append(self.window.reference_beat_samples)
        if self.review_store is not None:
            manual = np.asarray(
                [
                    record.sample_index
                    for record in self.review_store.records_between(
                        self.window.start_sample, self.window.end_sample
                    )
                    if record.source == "manual"
                ],
                dtype=np.int64,
            )
            if manual.size:
                candidate_arrays.append(manual)
        candidates = (
            np.unique(np.concatenate(candidate_arrays))
            if candidate_arrays
            else np.asarray([], dtype=np.int64)
        )
        if candidates.size:
            nearest = int(candidates[np.argmin(np.abs(candidates - sample))])
            if abs(nearest - sample) <= int(round(0.150 * self.window.sampling_rate_hz)):
                self.selected_peak_sample = nearest
            else:
                self.selected_peak_sample = None
        self._update_selection_text()
        self._draw()

    def _on_plot_release(self, event) -> None:
        if self.window is None or self._region_drag_start_sample is None:
            return
        start = self._region_drag_start_sample
        self._region_drag_start_sample = None
        if event.xdata is None:
            return
        end = int(round(event.xdata * self.window.sampling_rate_hz))
        end = min(max(end, self.window.start_sample), self.window.end_sample - 1)
        first, last = sorted((start, end))
        if last <= first:
            return
        self.selected_region_samples = (first, last)
        self._update_region_summary()
        self._draw()

    def _on_plot_motion(self, event) -> None:
        if (
            self.window is None
            or self._region_drag_start_sample is not None
            or event.inaxes is not self.ax_ecg
            or event.xdata is None
            or self.hover_annotation is None
        ):
            return
        sample = int(round(event.xdata * self.window.sampling_rate_hz))
        arrays = list(self.window.detector_peak_samples.values())
        if self.window.reference_beat_samples.size:
            arrays.append(self.window.reference_beat_samples)
        candidates = (
            np.unique(np.concatenate(arrays))
            if arrays
            else np.asarray([], dtype=np.int64)
        )
        if not candidates.size:
            return
        nearest = int(candidates[np.argmin(np.abs(candidates - sample))])
        tolerance = int(round(0.080 * self.window.sampling_rate_hz))
        if abs(nearest - sample) > tolerance:
            if self.hover_annotation.get_visible():
                self.hover_annotation.set_visible(False)
                self.canvas.draw_idle()
            return
        x, y = self._sample_xy(nearest)
        lines = [
            f"sample {nearest}",
            f"time {nearest / self.window.sampling_rate_hz:.3f} s",
        ]
        expert_index = np.flatnonzero(self.window.reference_beat_samples == nearest)
        if expert_index.size:
            symbol = self.window.reference_symbols[int(expert_index[0])]
            lines.append(f"expert {symbol}: {MIT_BIH_SYMBOL_NAMES.get(symbol, 'beat')}")
        for track in ("neurokit", "unsw", "zhai"):
            track_peaks = self.window.peaks_for(track)
            if track_peaks.size:
                offset_index = int(np.argmin(np.abs(track_peaks - nearest)))
                offset_samples = int(track_peaks[offset_index] - nearest)
                if abs(offset_samples) <= int(round(0.075 * self.window.sampling_rate_hz)):
                    lines.append(
                        f"{TRACK_DISPLAY_NAMES[track]} offset {1000.0 * offset_samples / self.window.sampling_rate_hz:+.1f} ms"
                    )
        morphology = self.window.morphology_results.get(self.active_track_var.get())
        if morphology is not None and not morphology.features.empty:
            index = (
                morphology.features["window_end_time_s"]
                - nearest / self.window.sampling_rate_hz
            ).abs().idxmin()
            row = morphology.features.loc[index]
            if abs(float(row["window_end_time_s"]) - nearest / self.window.sampling_rate_hz) <= 2.0:
                lines.append(
                    "five-beat λ1–λ5: "
                    + ", ".join(f"{float(row[f'lambda{i}']):.3g}" for i in range(1, 6))
                )
                lines.append("measurements only; no seizure class")
        self.hover_annotation.xy = (x, y)
        self.hover_annotation.set_text("\n".join(lines))
        self.hover_annotation.set_visible(True)
        self.canvas.draw_idle()

    def _update_region_summary(self) -> None:
        if self.window is None or self.selected_region_samples is None:
            return
        first, last = self.selected_region_samples
        symbol_counts: dict[str, int] = {}
        for sample, symbol in zip(
            self.window.reference_beat_samples, self.window.reference_symbols
        ):
            if first <= sample <= last:
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        symbol_text = (
            ", ".join(
                f"{symbol} ({MIT_BIH_SYMBOL_NAMES.get(symbol, 'beat')}) ×{count}"
                for symbol, count in sorted(symbol_counts.items())
            )
            if symbol_counts
            else "no expert beat labels"
        )
        track_counts = ", ".join(
            f"{TRACK_DISPLAY_NAMES[name]} {int(np.sum((samples >= first) & (samples <= last)))}"
            for name, samples in self.window.detector_peak_samples.items()
        )
        self.region_var.set(
            f"Region {first / self.window.sampling_rate_hz:.3f}–{last / self.window.sampling_rate_hz:.3f} s | "
            f"expert: {symbol_text} | detector events: {track_counts}"
        )

    def _clear_region(self) -> None:
        self.selected_region_samples = None
        self._region_drag_start_sample = None
        self.region_var.set(
            "Right-drag over the ECG to inspect a region and its expert beat labels."
        )
        self._draw()

    def _zoom(self, factor: float) -> None:
        if self.recording is None:
            return
        current_duration = self._duration_s()
        new_duration = min(300.0, max(1.0, current_duration * factor))
        center = (
            self.cursor_sample / self.window.sampling_rate_hz
            if self.window is not None and self.cursor_sample is not None
            else float(self.start_var.get()) + current_duration / 2.0
        )
        maximum_start = max(0.0, self.recording.duration_s - new_duration)
        new_start = min(max(0.0, center - new_duration / 2.0), maximum_start)
        self.duration_var.set(f"{new_duration:g}")
        self.start_var.set(new_start)
        self.start_text_var.set(f"{new_start:.3f}")
        self.selected_peak_sample = None
        self.cursor_sample = None
        self.selected_region_samples = None
        self.region_var.set(
            "Right-drag over the ECG to inspect a region and its expert beat labels."
        )
        self.analyze()

    def _update_selection_text(self) -> None:
        if self.window is None:
            self.selection_var.set("No peak selected")
            return
        if self.selected_peak_sample is None:
            if self.cursor_sample is None:
                self.selection_var.set("No peak selected")
            else:
                self.selection_var.set(
                    f"Cursor {self.cursor_sample / self.window.sampling_rate_hz:.3f} s; "
                    "press M to add a missed R peak"
                )
            return
        labels: list[str] = []
        if self.window.reference_beat_samples.size:
            matches = np.flatnonzero(
                self.window.reference_beat_samples == self.selected_peak_sample
            )
            if matches.size:
                symbol = self.window.reference_symbols[int(matches[0])]
                labels.append(
                    f"expert {symbol}: {MIT_BIH_SYMBOL_NAMES.get(symbol, 'beat')}"
                )
        record = self.review_store.records.get(self.selected_peak_sample) if self.review_store else None
        for track, track_peaks in self.window.detector_peak_samples.items():
            if self.selected_peak_sample in set(track_peaks.tolist()):
                comparison = self.window.comparison_for(track)
                if comparison is not None and self.selected_peak_sample in set(
                    comparison.false_positive_samples.tolist()
                ):
                    state = "false positive vs expert"
                elif comparison is not None and self.selected_peak_sample in set(
                    comparison.matched_detection_samples.tolist()
                ):
                    state = "matched expert"
                else:
                    state = "candidate"
                labels.append(f"{TRACK_DISPLAY_NAMES[track]} {state}")
        if record is not None:
            labels.append(f"manual review: {record.label}")
        label = "; ".join(labels) if labels else "unclassified cursor marker"
        self.selection_var.set(
            f"Selected sample {self.selected_peak_sample} | "
            f"{self.selected_peak_sample / self.window.sampling_rate_hz:.3f} s | {label}"
        )

    def _select_adjacent_peak(self, direction: int) -> None:
        if self.window is None:
            return
        peaks = self.window.peaks_for(self.active_track_var.get())
        if peaks.size == 0:
            return
        if self.selected_peak_sample is None:
            index = 0 if direction > 0 else peaks.size - 1
        else:
            current = int(np.searchsorted(peaks, self.selected_peak_sample))
            if current < peaks.size and peaks[current] == self.selected_peak_sample:
                index = min(max(current + direction, 0), peaks.size - 1)
            else:
                index = min(max(current, 0), peaks.size - 1)
        self.selected_peak_sample = int(peaks[index])
        self.cursor_sample = self.selected_peak_sample
        self._update_selection_text()
        self._draw()

    def _label_selected(self, label: str) -> None:
        if self.window is None or self.review_store is None or self.selected_peak_sample is None:
            self.status_var.set("Select a NeuroKit marker before assigning a detection label.")
            return
        if self.selected_peak_sample not in set(self.window.neurokit_peak_samples.tolist()):
            self.status_var.set("Manual missed-peak markers cannot be relabeled as NeuroKit detections.")
            return
        self.review_store.set_label(
            self.selected_peak_sample, source="neurokit", label=label
        )
        self._update_selection_text()
        self._draw()
        self.status_var.set(f"Saved {label} at {self.selected_peak_sample / self.window.sampling_rate_hz:.3f} s")

    def _add_missed_at_cursor(self) -> None:
        if self.window is None or self.review_store is None or self.cursor_sample is None:
            self.status_var.set("Click the ECG at the missing R wave, then press M.")
            return
        if self.window.reference_beat_samples.size:
            self.status_var.set(
                "This WFDB record already has expert .atr beats; orange markers are automatic misses."
            )
            return
        tolerance = int(round(0.150 * self.window.sampling_rate_hz))
        if self.window.neurokit_peak_samples.size and np.min(
            np.abs(self.window.neurokit_peak_samples - self.cursor_sample)
        ) <= tolerance:
            self.status_var.set(
                "A NeuroKit detection is within 150 ms; select and label that marker instead."
            )
            return
        record = self.review_store.set_label(
            self.cursor_sample, source="manual", label="manual_missed_r_peak"
        )
        self.selected_peak_sample = record.sample_index
        self._update_selection_text()
        self._draw()
        self.status_var.set(f"Saved manually added R peak at {record.time_s:.3f} s")

    def _clear_selected_label(self) -> None:
        if self.review_store is None:
            return
        target = self.selected_peak_sample if self.selected_peak_sample is not None else self.cursor_sample
        if target is None or not self.review_store.clear_label(target):
            self.status_var.set("The selected point has no saved manual label.")
            return
        self.selected_peak_sample = None
        self._update_selection_text()
        self._draw()
        self.status_var.set("Manual label removed from the review CSV.")

    def _shift_window(self, fraction: float) -> None:
        if self.recording is None:
            return
        try:
            duration = self._duration_s()
        except ValueError:
            return
        maximum_start = max(0.0, self.recording.duration_s - duration)
        new_start = min(max(0.0, float(self.start_var.get()) + fraction * duration), maximum_start)
        self.start_var.set(new_start)
        self.start_text_var.set(f"{new_start:.3f}")
        self.selected_peak_sample = None
        self.cursor_sample = None
        self.selected_region_samples = None
        self.analyze()

    def _start_from_slider(self) -> None:
        self.start_text_var.set(f"{float(self.start_var.get()):.3f}")
        self.selected_peak_sample = None
        self.cursor_sample = None
        self.analyze()

    def _start_from_entry(self) -> None:
        from tkinter import messagebox

        try:
            value = float(self.start_text_var.get())
        except ValueError:
            messagebox.showerror("Invalid start time", "Enter start time in seconds.")
            return
        if self.recording is not None:
            value = min(max(0.0, value), self.recording.duration_s)
        self.start_var.set(value)
        self.selected_peak_sample = None
        self.cursor_sample = None
        self.selected_region_samples = None
        self.analyze()

    def _open_review_csv(self) -> None:
        import os

        if self.review_store is None:
            return
        self.review_store.save()
        os.startfile(self.review_store.output_path)  # type: ignore[attr-defined]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Review NeuroKit, UNSW, Zhai, expert beats, RR, and five-beat "
            "morphology in EDF or WFDB ECG"
        )
    )
    parser.add_argument(
        "recording", nargs="?", help="Optional EDF or WFDB .hea record to open immediately"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    import tkinter as tk

    root = tk.Tk()
    RPeakReviewerApp(root, initial_recording=args.recording)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
