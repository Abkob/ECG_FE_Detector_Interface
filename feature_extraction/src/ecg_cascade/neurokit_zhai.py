"""Independent NeuroKit and Zhai RR--HRV tracks.

This module implements the architecture selected for the next audit:

raw ECG -> NeuroKit detector -> NeuroKit RR/HRV
        +-> Zhai detector     -> Zhai RR/HRV

The detector outputs are compared but never averaged, majority-voted, or
spliced beat by beat.  The code also deliberately avoids the previous local
positive-amplitude R-fiducial refinement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import NeuroKitZhaiConfig
from .edf import SignalSegment, load_ecg_segment
from .fusion import build_detector_association_table, build_fusion_ready_table
from .morphology import VaronMorphologyResult, extract_varon_morphology
from .peaks import DetectorRun, PeakAgreement, compare_peak_sequences, detect_neurokit_gradient
from .prsa_bprsa import PRSABPRSAResult, extract_prsa_bprsa_features
from .rr_hrv import extract_rr_hrv_windows
from .wfdb_io import WFDBSegment, load_wfdb_segment
from .zhai import ZhaiDetection, detect_zhai_template


@dataclass(frozen=True)
class TimestampTrackResult:
    """One detector's unmodified timestamps and downstream RR/HRV output."""

    name: str
    detector: DetectorRun
    events: pd.DataFrame
    rr_intervals: pd.DataFrame
    features: pd.DataFrame
    morphology: VaronMorphologyResult
    prsa_bprsa_context: PRSABPRSAResult
    fusion_ready: pd.DataFrame

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "detector": self.detector.summary(),
            "event_count": int(self.events.shape[0]),
            "supported_event_count": int(self.events["qrs_supported"].sum()),
            "event_support_fraction": _finite_or_none(
                float(self.events["qrs_supported"].mean())
                if not self.events.empty
                else np.nan
            ),
            "rr_interval_count": int(self.rr_intervals.shape[0]),
            "supported_rr_count": int(self.rr_intervals["rr_supported"].sum()),
            "rr_support_fraction": _finite_or_none(
                float(self.rr_intervals["rr_supported"].mean())
                if not self.rr_intervals.empty
                else np.nan
            ),
            "defined_hrv_rows": int(self.features["feature_defined"].sum()),
            "reliable_hrv_rows": int(self.features["feature_reliable"].sum()),
            "morphology": self.morphology.summary(),
            "prsa_bprsa_context": self.prsa_bprsa_context.summary(),
            "fusion_ready_row_count": int(self.fusion_ready.shape[0]),
            "fusion_defined_row_count": int(
                self.fusion_ready["fusion_measurements_defined"].sum()
            ),
            "timestamp_adjustment": "none_detector_output_used_directly",
        }


@dataclass(frozen=True)
class NeuroKitZhaiOrientationResult:
    orientation: str
    neurokit: TimestampTrackResult
    zhai: TimestampTrackResult
    zhai_detection: ZhaiDetection
    support_agreement: PeakAgreement
    audit_agreement: PeakAgreement
    detector_associations: pd.DataFrame

    def summary(self) -> dict[str, object]:
        return {
            "orientation": self.orientation,
            "neurokit": self.neurokit.summary(),
            "zhai": self.zhai.summary(),
            "zhai_intermediates": self.zhai_detection.summary(),
            "support_tolerance_agreement": self.support_agreement.to_dict(),
            "audit_tolerance_agreement": self.audit_agreement.to_dict(),
            "detector_association_rows": int(self.detector_associations.shape[0]),
            "timestamp_owner": "unselected_pending_annotation_validation",
        }


@dataclass(frozen=True)
class NeuroKitZhaiResult:
    selected_orientation: str
    selected: NeuroKitZhaiOrientationResult
    orientations: dict[str, NeuroKitZhaiOrientationResult]
    config: NeuroKitZhaiConfig
    metadata: dict[str, object]

    def summary(self) -> dict[str, object]:
        return {
            "architecture": (
                "independent_neurokit_and_zhai_RR_HRV_and_Varon_morphology_tracks_"
                "with_PRSA_BPRSA_context"
            ),
            "selected_orientation_for_display": self.selected_orientation,
            "timestamp_owner": "unselected_pending_annotation_validation",
            "automatic_timestamp_fusion_applied": False,
            "morphology_timestamp_fusion_applied": False,
            "raw_amplitude_argmax_refinement_applied": False,
            "config": self.config.to_dict(),
            "selected_orientation_result": self.selected.summary(),
            "orientation_summaries": {
                name: result.summary() for name, result in self.orientations.items()
            },
            "metadata": self.metadata,
        }


def run_neurokit_zhai_array(
    ecg: np.ndarray,
    sampling_rate_hz: float,
    *,
    segment_start_s: float = 0.0,
    config: NeuroKitZhaiConfig | None = None,
    source_metadata: dict[str, object] | None = None,
) -> NeuroKitZhaiResult:
    """Run both complete detector tracks without selecting a winner."""

    config = NeuroKitZhaiConfig() if config is None else config
    signal = np.asarray(ecg, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("ECG must be a one-dimensional array")
    if not np.all(np.isfinite(signal)):
        raise ValueError("ECG contains non-finite samples")
    if signal.size < int(np.ceil(3.0 * sampling_rate_hz)):
        raise ValueError("At least three seconds of ECG are required")
    if segment_start_s < 0:
        raise ValueError("segment_start_s must be non-negative")

    orientation_names = [config.processing_orientation]
    if config.compute_inverted_context:
        orientation_names.append(
            "inverted" if config.processing_orientation == "original" else "original"
        )
    orientations = {
        orientation: _run_orientation(
            signal,
            sampling_rate_hz,
            segment_start_s=segment_start_s,
            orientation=orientation,
            config=config,
            lead_name=_source_lead_name(source_metadata),
        )
        for orientation in dict.fromkeys(orientation_names)
    }
    selected = orientations[config.processing_orientation]
    metadata = {
        "sampling_rate_hz": float(sampling_rate_hz),
        "sample_count": int(signal.size),
        "segment_start_s": float(segment_start_s),
        "segment_duration_s": float(signal.size / sampling_rate_hz),
        "source": source_metadata or {},
        "package_versions": _package_versions(),
        "seizure_classifier_applied": False,
        "artifact_classifier_applied": False,
        "automatic_polarity_routing_applied": False,
        "automatic_detector_fusion_applied": False,
        "morphology_branch_applied_to_each_track_separately": True,
        "prsa_bprsa_attached_as_context_not_core_gate": True,
        "fusion_ready_tables_are_measurements_not_predictions": True,
        "interpretation": (
            "Detector agreement is context, not ground truth or an artifact label. "
            "Both RR/HRV and morphology tracks remain available until annotation "
            "validation selects a timestamp owner."
        ),
    }
    return NeuroKitZhaiResult(
        selected_orientation=config.processing_orientation,
        selected=selected,
        orientations=orientations,
        config=config,
        metadata=metadata,
    )


def run_neurokit_zhai_wfdb(
    record_path: str | Path,
    *,
    channel: int | str = 0,
    start_s: float = 0.0,
    duration_s: float | None = None,
    config: NeuroKitZhaiConfig | None = None,
) -> tuple[WFDBSegment, NeuroKitZhaiResult]:
    segment = load_wfdb_segment(
        record_path,
        channel=channel,
        start_s=start_s,
        duration_s=duration_s,
    )
    result = run_neurokit_zhai_array(
        segment.samples,
        segment.sampling_rate_hz,
        segment_start_s=segment.start_s,
        config=config,
        source_metadata={
            "kind": "WFDB",
            "record_path": segment.record_path,
            "channel_index": segment.channel_index,
            "channel_name": segment.channel_name,
            "start_sample": segment.start_sample,
        },
    )
    return segment, result


def run_neurokit_zhai_edf(
    edf_path: str | Path,
    *,
    channel_label: str | None,
    start_s: float,
    duration_s: float,
    config: NeuroKitZhaiConfig | None = None,
) -> tuple[SignalSegment, NeuroKitZhaiResult]:
    segment = load_ecg_segment(
        edf_path,
        channel_label=channel_label,
        start_s=start_s,
        duration_s=duration_s,
    )
    result = run_neurokit_zhai_array(
        segment.samples,
        segment.channel.sampling_rate_hz,
        segment_start_s=segment.start_s,
        config=config,
        source_metadata={
            "kind": "EDF",
            "path": segment.path,
            "channel": asdict(segment.channel),
        },
    )
    return segment, result


def save_neurokit_zhai_outputs(
    result: NeuroKitZhaiResult,
    output_dir: str | Path,
) -> dict[str, str]:
    """Save both tracks separately; no file is named a selected/fused result."""

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected = result.selected
    paths: dict[str, Path] = {
        "neurokit_events": output / "neurokit_events.csv",
        "neurokit_rr_hrv": output / "neurokit_rr_hrv.csv",
        "zhai_events": output / "zhai_events.csv",
        "zhai_rr_hrv": output / "zhai_rr_hrv.csv",
        "zhai_qrs_windows": output / "zhai_qrs_windows.csv",
        "zhai_template": output / "zhai_template.npy",
        "detector_associations": output / "detector_associations.csv",
        "neurokit_morphology_beats": output / "neurokit_morphology_beats.csv",
        "neurokit_varon_morphology": output / "neurokit_varon_morphology.csv",
        "neurokit_qrs_waveforms": output / "neurokit_qrs_waveforms.npy",
        "neurokit_fusion_ready": output / "neurokit_fusion_ready.csv",
        "neurokit_prsa_bprsa_context": output / "neurokit_prsa_bprsa_context.csv",
        "zhai_morphology_beats": output / "zhai_morphology_beats.csv",
        "zhai_varon_morphology": output / "zhai_varon_morphology.csv",
        "zhai_qrs_waveforms": output / "zhai_qrs_waveforms.npy",
        "zhai_fusion_ready": output / "zhai_fusion_ready.csv",
        "zhai_prsa_bprsa_context": output / "zhai_prsa_bprsa_context.csv",
        "summary": output / "neurokit_zhai_summary.json",
    }
    selected.neurokit.events.to_csv(paths["neurokit_events"], index=False)
    selected.neurokit.features.to_csv(paths["neurokit_rr_hrv"], index=False)
    selected.zhai.events.to_csv(paths["zhai_events"], index=False)
    selected.zhai.features.to_csv(paths["zhai_rr_hrv"], index=False)
    selected.detector_associations.to_csv(paths["detector_associations"], index=False)
    selected.neurokit.morphology.beat_context.to_csv(
        paths["neurokit_morphology_beats"], index=False
    )
    selected.neurokit.morphology.features.to_csv(
        paths["neurokit_varon_morphology"], index=False
    )
    np.save(
        paths["neurokit_qrs_waveforms"],
        selected.neurokit.morphology.beat_waveforms,
    )
    selected.neurokit.fusion_ready.to_csv(
        paths["neurokit_fusion_ready"], index=False
    )
    selected.neurokit.prsa_bprsa_context.features.to_csv(
        paths["neurokit_prsa_bprsa_context"], index=False
    )
    selected.zhai.morphology.beat_context.to_csv(
        paths["zhai_morphology_beats"], index=False
    )
    selected.zhai.morphology.features.to_csv(
        paths["zhai_varon_morphology"], index=False
    )
    np.save(paths["zhai_qrs_waveforms"], selected.zhai.morphology.beat_waveforms)
    selected.zhai.fusion_ready.to_csv(paths["zhai_fusion_ready"], index=False)
    selected.zhai.prsa_bprsa_context.features.to_csv(
        paths["zhai_prsa_bprsa_context"], index=False
    )
    pd.DataFrame(
        selected.zhai_detection.qrs_windows,
        columns=["start_sample_in_segment", "end_sample_exclusive_in_segment"],
    ).to_csv(paths["zhai_qrs_windows"], index=False)
    np.save(paths["zhai_template"], selected.zhai_detection.template)
    paths["summary"].write_text(
        json.dumps(result.summary(), indent=2, default=_json_default),
        encoding="utf-8",
    )
    return {name: str(path) for name, path in paths.items()}


def _run_orientation(
    signal: np.ndarray,
    sampling_rate_hz: float,
    *,
    segment_start_s: float,
    orientation: str,
    config: NeuroKitZhaiConfig,
    lead_name: str,
) -> NeuroKitZhaiOrientationResult:
    neurokit_detector = detect_neurokit_gradient(
        signal,
        sampling_rate_hz,
        orientation=orientation,
        minimum_delay_ms=config.neurokit_minimum_delay_ms,
        minimum_delay_inclusive=config.neurokit_minimum_delay_inclusive,
    )
    zhai_detection = detect_zhai_template(
        signal,
        sampling_rate_hz,
        orientation=orientation,
    )
    zhai_detector = zhai_detection.detector

    neurokit_events, neurokit_intervals = _timestamp_reliability_tables(
        neurokit_detector.peak_samples,
        zhai_detector.peak_samples,
        sampling_rate_hz=sampling_rate_hz,
        segment_start_s=segment_start_s,
        support_tolerance_ms=config.support_tolerance_ms,
        primary_name="neurokit",
        comparator_name="zhai",
    )
    zhai_events, zhai_intervals = _timestamp_reliability_tables(
        zhai_detector.peak_samples,
        neurokit_detector.peak_samples,
        sampling_rate_hz=sampling_rate_hz,
        segment_start_s=segment_start_s,
        support_tolerance_ms=config.support_tolerance_ms,
        primary_name="zhai",
        comparator_name="neurokit",
    )
    if not zhai_events.empty:
        score_by_sample = dict(
            zip(
                zhai_detector.peak_samples.tolist(),
                zhai_detection.correlation_peak_values.tolist(),
                strict=True,
            )
        )
        zhai_events["zhai_correlation"] = zhai_events[
            "primary_timestamp_sample"
        ].map(score_by_sample)
        zhai_events["zhai_absolute_correlation"] = zhai_events[
            "zhai_correlation"
        ].abs()
        absolute_score_by_sample = {
            sample: abs(score) for sample, score in score_by_sample.items()
        }
        neurokit_events["zhai_absolute_correlation"] = neurokit_events[
            "nearest_comparator_sample"
        ].map(absolute_score_by_sample)

    neurokit_features = extract_rr_hrv_windows(
        neurokit_intervals,
        window_size=config.hrv_window_rr_intervals,
        median_width=config.causal_rr_median_width,
        reliable_coverage=config.reliable_hrv_coverage,
    )
    zhai_features = extract_rr_hrv_windows(
        zhai_intervals,
        window_size=config.hrv_window_rr_intervals,
        median_width=config.causal_rr_median_width,
        reliable_coverage=config.reliable_hrv_coverage,
    )
    neurokit_prsa_bprsa = extract_prsa_bprsa_features(
        signal,
        neurokit_detector.peak_samples,
        sampling_rate_hz,
        segment_start_s=segment_start_s,
        orientation=orientation,
        rr_supported=neurokit_intervals["rr_supported"].to_numpy(dtype=bool),
        r_peak_supported=neurokit_events["qrs_supported"].to_numpy(dtype=bool),
        reliable_coverage=config.reliable_hrv_coverage,
        lead_name=lead_name,
        anchor_track="neurokit",
    )
    zhai_prsa_bprsa = extract_prsa_bprsa_features(
        signal,
        zhai_detector.peak_samples,
        sampling_rate_hz,
        segment_start_s=segment_start_s,
        orientation=orientation,
        rr_supported=zhai_intervals["rr_supported"].to_numpy(dtype=bool),
        r_peak_supported=zhai_events["qrs_supported"].to_numpy(dtype=bool),
        reliable_coverage=config.reliable_hrv_coverage,
        lead_name=lead_name,
        anchor_track="zhai",
    )
    neurokit_morphology = extract_varon_morphology(
        signal,
        neurokit_detector.peak_samples,
        sampling_rate_hz,
        anchor_track="neurokit",
        segment_start_s=segment_start_s,
        orientation=orientation,
        event_context=neurokit_events,
        pre_r_ms=config.varon_pre_r_ms,
        post_r_ms=config.varon_post_r_ms,
        stack_beats=config.varon_stack_beats,
        reliable_support_coverage=config.reliable_morphology_support_coverage,
    )
    zhai_morphology = extract_varon_morphology(
        signal,
        zhai_detector.peak_samples,
        sampling_rate_hz,
        anchor_track="zhai",
        segment_start_s=segment_start_s,
        orientation=orientation,
        event_context=zhai_events,
        pre_r_ms=config.varon_pre_r_ms,
        post_r_ms=config.varon_post_r_ms,
        stack_beats=config.varon_stack_beats,
        reliable_support_coverage=config.reliable_morphology_support_coverage,
    )
    neurokit_fusion = build_fusion_ready_table(
        neurokit_features,
        neurokit_morphology.features,
        anchor_track="neurokit",
        prsa_bprsa_context=neurokit_prsa_bprsa.features,
    )
    zhai_fusion = build_fusion_ready_table(
        zhai_features,
        zhai_morphology.features,
        anchor_track="zhai",
        prsa_bprsa_context=zhai_prsa_bprsa.features,
    )
    neurokit_track = TimestampTrackResult(
        name="neurokit",
        detector=neurokit_detector,
        events=neurokit_events,
        rr_intervals=neurokit_intervals,
        features=neurokit_features,
        morphology=neurokit_morphology,
        prsa_bprsa_context=neurokit_prsa_bprsa,
        fusion_ready=neurokit_fusion,
    )
    zhai_track = TimestampTrackResult(
        name="zhai2023",
        detector=zhai_detector,
        events=zhai_events,
        rr_intervals=zhai_intervals,
        features=zhai_features,
        morphology=zhai_morphology,
        prsa_bprsa_context=zhai_prsa_bprsa,
        fusion_ready=zhai_fusion,
    )
    support_agreement = compare_peak_sequences(
        neurokit_detector.peak_samples,
        zhai_detector.peak_samples,
        sampling_rate_hz=sampling_rate_hz,
        tolerance_ms=config.support_tolerance_ms,
    )
    audit_agreement = compare_peak_sequences(
        neurokit_detector.peak_samples,
        zhai_detector.peak_samples,
        sampling_rate_hz=sampling_rate_hz,
        tolerance_ms=config.audit_match_tolerance_ms,
    )
    detector_associations = build_detector_association_table(
        support_agreement,
        sampling_rate_hz=sampling_rate_hz,
        segment_start_s=segment_start_s,
        zhai_events=zhai_events,
    )
    return NeuroKitZhaiOrientationResult(
        orientation=orientation,
        neurokit=neurokit_track,
        zhai=zhai_track,
        zhai_detection=zhai_detection,
        support_agreement=support_agreement,
        audit_agreement=audit_agreement,
        detector_associations=detector_associations,
    )


def _timestamp_reliability_tables(
    primary_samples: np.ndarray,
    comparator_samples: np.ndarray,
    *,
    sampling_rate_hz: float,
    segment_start_s: float,
    support_tolerance_ms: float,
    primary_name: str,
    comparator_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build support context without moving or collapsing either timestamp set."""

    primary = np.unique(np.asarray(primary_samples, dtype=np.int64))
    comparator = np.unique(np.asarray(comparator_samples, dtype=np.int64))
    tolerance = int(np.rint(support_tolerance_ms * sampling_rate_hz / 1000.0))
    if tolerance < 1:
        raise ValueError("support_tolerance_ms is less than one sample")

    event_rows: list[dict[str, object]] = []
    supported: list[bool] = []
    for event_index, timestamp in enumerate(primary):
        timestamp = int(timestamp)
        left = np.searchsorted(comparator, timestamp - tolerance, side="left")
        right = np.searchsorted(comparator, timestamp + tolerance, side="right")
        matches = comparator[left:right]
        if matches.size:
            nearest = int(matches[np.argmin(np.abs(matches - timestamp))])
            offset_ms = (nearest - timestamp) * 1000.0 / sampling_rate_hz
        else:
            nearest = pd.NA
            offset_ms = np.nan
        is_supported = bool(matches.size == 1)
        supported.append(is_supported)
        event_rows.append(
            {
                "event_index": event_index,
                "primary_detector": primary_name,
                "comparator_detector": comparator_name,
                "primary_timestamp_sample": timestamp,
                "primary_timestamp_time_s": segment_start_s
                + timestamp / sampling_rate_hz,
                "comparator_match_count": int(matches.size),
                "nearest_comparator_sample": nearest,
                "comparator_offset_ms": offset_ms,
                "qrs_supported": is_supported,
                "timestamp_moved_after_detection": False,
            }
        )
    events = pd.DataFrame(event_rows, columns=_EVENT_COLUMNS)
    if not events.empty:
        events["nearest_comparator_sample"] = events[
            "nearest_comparator_sample"
        ].astype("Int64")

    interval_rows: list[dict[str, object]] = []
    for interval_index in range(max(0, primary.size - 1)):
        first = int(primary[interval_index])
        last = int(primary[interval_index + 1])
        left = np.searchsorted(comparator, first - tolerance, side="left")
        right = np.searchsorted(comparator, last + tolerance, side="right")
        comparator_count = int(right - left)
        start_supported = supported[interval_index]
        end_supported = supported[interval_index + 1]
        rr_supported = start_supported and end_supported and comparator_count == 2
        rr_samples = last - first
        reasons: list[str] = []
        if not start_supported:
            reasons.append("unsupported_start_qrs")
        if not end_supported:
            reasons.append("unsupported_end_qrs")
        if comparator_count != 2:
            reasons.append(f"comparator_count_{comparator_count}_not_2")
        interval_rows.append(
            {
                "rr_index": interval_index,
                "timestamp_owner_track": primary_name,
                "start_r_sample": first,
                "end_r_sample": last,
                "start_time_s": segment_start_s + first / sampling_rate_hz,
                "end_time_s": segment_start_s + last / sampling_rate_hz,
                "rr_samples": rr_samples,
                "rr_ms": rr_samples * 1000.0 / sampling_rate_hz,
                "heart_rate_bpm": 60.0 * sampling_rate_hz / rr_samples,
                "start_qrs_supported": start_supported,
                "end_qrs_supported": end_supported,
                "comparator_events_in_expanded_rr": comparator_count,
                "rr_supported": rr_supported,
                "reliability_reason": (
                    "supported" if rr_supported else ";".join(reasons)
                ),
            }
        )
    intervals = pd.DataFrame(interval_rows, columns=_INTERVAL_COLUMNS)
    return events, intervals


_EVENT_COLUMNS = [
    "event_index",
    "primary_detector",
    "comparator_detector",
    "primary_timestamp_sample",
    "primary_timestamp_time_s",
    "comparator_match_count",
    "nearest_comparator_sample",
    "comparator_offset_ms",
    "qrs_supported",
    "timestamp_moved_after_detection",
]

_INTERVAL_COLUMNS = [
    "rr_index",
    "timestamp_owner_track",
    "start_r_sample",
    "end_r_sample",
    "start_time_s",
    "end_time_s",
    "rr_samples",
    "rr_ms",
    "heart_rate_bpm",
    "start_qrs_supported",
    "end_qrs_supported",
    "comparator_events_in_expanded_rr",
    "rr_supported",
    "reliability_reason",
]


def _source_lead_name(source_metadata: dict[str, object] | None) -> str:
    """Keep lead identity beside the polarity-sensitive BPRSA context."""

    if not source_metadata:
        return "unknown"
    direct = source_metadata.get("channel_name")
    if direct is not None:
        return str(direct)
    channel = source_metadata.get("channel")
    if isinstance(channel, dict):
        for key in ("label", "name"):
            if channel.get(key) is not None:
                return str(channel[key])
    return "unknown"


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ["neurokit2", "numpy", "pandas", "scipy", "wfdb", "pyEDFlib"]:
        try:
            versions[name] = version(name)
        except Exception:  # pragma: no cover
            versions[name] = "unavailable"
    return versions


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _json_default(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
