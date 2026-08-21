"""Complete, auditable RR--HRV branch orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RRHRVConfig
from .edf import SignalSegment, load_ecg_segment
from .fiducials import FiducialCandidates, build_fiducial_candidates
from .peaks import (
    DetectorRun,
    PeakAgreement,
    compare_peak_sequences,
    detect_neurokit_gradient,
    detect_pan_tompkins_context,
    detect_unsw,
)
from .reliability import ReliabilityResult, assess_rr_reliability
from .rr_hrv import extract_rr_hrv_windows
from .wfdb_io import WFDBSegment, load_wfdb_segment


@dataclass(frozen=True)
class OrientationBranchResult:
    orientation: str
    primary: DetectorRun
    secondary_experimental: DetectorRun
    secondary_baseline: DetectorRun
    pan_context: dict[float, DetectorRun]
    reliability: ReliabilityResult
    baseline_reliability: ReliabilityResult
    legacy_tolerance_reliability: ReliabilityResult
    features: pd.DataFrame
    primary_events: pd.DataFrame
    fiducial_candidates: FiducialCandidates
    pan_agreements: dict[float, PeakAgreement]

    def summary(self) -> dict[str, object]:
        return {
            "orientation": self.orientation,
            "primary": self.primary.summary(),
            "secondary_experimental": self.secondary_experimental.summary(),
            "secondary_baseline": self.secondary_baseline.summary(),
            "reliability_current_50ms": self.reliability.summary(),
            "reliability_baseline_neurokit_300ms": self.baseline_reliability.summary(),
            "reliability_legacy_support_tolerance": self.legacy_tolerance_reliability.summary(),
            "fiducial_candidate_status": {
                "rr_owner": "ho_positive_current_baseline",
                "physiozoo_qrs_adjust": (
                    "implemented_as_positive_and_negative_named_ablations; "
                    "input sign is caller supplied"
                ),
                "physiozoo_rqrs_adapted_selected_sign": (
                    self.fiducial_candidates.physiozoo_rqrs_selected_sign
                ),
                "physiozoo_rqrs_adaptation_warning": (
                    "UNSW candidates replace the gqrs onsets used by rqrs"
                ),
                "rdeco_localization": (
                    "final 65-ms backward localization only; not the full "
                    "R-DECO detector or human correction GUI"
                ),
            },
            "mathematically_defined_hrv_rows": int(self.features["feature_defined"].sum()),
            "reliable_hrv_rows": int(self.features["feature_reliable"].sum()),
            "pan_context": {
                str(int(delay)): {
                    **run.summary(),
                    **_compact_agreement(self.pan_agreements[delay]),
                }
                for delay, run in self.pan_context.items()
            },
        }


@dataclass(frozen=True)
class RRHRVBranchResult:
    selected_orientation: str
    selected: OrientationBranchResult
    orientations: dict[str, OrientationBranchResult]
    polarity_context: pd.DataFrame
    config: RRHRVConfig
    metadata: dict[str, object]

    def summary(self) -> dict[str, object]:
        return {
            "selected_orientation": self.selected_orientation,
            "automatic_polarity_routing_applied": False,
            "config": self.config.to_dict(),
            "selected_branch": self.selected.summary(),
            "orientation_summaries": {
                name: result.summary() for name, result in self.orientations.items()
            },
            "metadata": self.metadata,
        }


def run_rr_hrv_array(
    ecg: np.ndarray,
    sampling_rate_hz: float,
    *,
    segment_start_s: float = 0.0,
    config: RRHRVConfig | None = None,
    source_metadata: dict[str, object] | None = None,
) -> RRHRVBranchResult:
    """Run the complete RR--HRV branch on a one-dimensional ECG array."""

    config = RRHRVConfig() if config is None else config
    signal = np.asarray(ecg, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("ECG must be a one-dimensional array")
    if signal.size < int(np.ceil(3.0 * sampling_rate_hz)):
        raise ValueError("At least three seconds of ECG are required")
    if not np.all(np.isfinite(signal)):
        raise ValueError("ECG contains non-finite samples")
    if sampling_rate_hz < 50:
        raise ValueError("UNSW requires a sampling rate of at least 50 Hz")
    if segment_start_s < 0:
        raise ValueError("segment_start_s must be non-negative")

    orientation_names = [config.processing_orientation]
    other_orientation = (
        "inverted" if config.processing_orientation == "original" else "original"
    )
    if config.compute_inverted_context:
        orientation_names.append(other_orientation)

    orientations: dict[str, OrientationBranchResult] = {}
    for orientation in dict.fromkeys(orientation_names):
        orientations[orientation] = _run_orientation(
            signal,
            sampling_rate_hz,
            segment_start_s=segment_start_s,
            orientation=orientation,
            config=config,
        )

    selected = orientations[config.processing_orientation]
    polarity_context = _make_polarity_context(
        orientations, selected_orientation=config.processing_orientation
    )
    package_names = [
        "neurokit2",
        "numpy",
        "pandas",
        "scipy",
        "wfdb",
        "pyEDFlib",
    ]
    package_versions: dict[str, str] = {}
    for name in package_names:
        try:
            package_versions[name] = version(name)
        except Exception:  # pragma: no cover - package metadata edge case
            package_versions[name] = "unavailable"
    metadata: dict[str, object] = {
        "sampling_rate_hz": float(sampling_rate_hz),
        "sample_count": int(signal.size),
        "segment_start_s": float(segment_start_s),
        "segment_duration_s": float(signal.size / sampling_rate_hz),
        "source": source_metadata or {},
        "package_versions": package_versions,
        "threshold_calibration_applied": False,
        "seizure_classifier_applied": False,
        "artifact_classifier_applied": False,
        "automatic_polarity_routing_applied": False,
        "polarity_context_interpretation": (
            "Higher detector agreement is descriptive only and does not prove "
            "correct polarity or authorize automatic switching."
        ),
    }
    return RRHRVBranchResult(
        selected_orientation=config.processing_orientation,
        selected=selected,
        orientations=orientations,
        polarity_context=polarity_context,
        config=config,
        metadata=metadata,
    )


def run_rr_hrv_edf(
    edf_path: str | Path,
    *,
    channel_label: str | None,
    start_s: float,
    duration_s: float,
    config: RRHRVConfig | None = None,
) -> tuple[SignalSegment, RRHRVBranchResult]:
    segment = load_ecg_segment(
        edf_path,
        channel_label=channel_label,
        start_s=start_s,
        duration_s=duration_s,
    )
    result = run_rr_hrv_array(
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


def run_rr_hrv_wfdb(
    record_path: str | Path,
    *,
    channel: int | str = 0,
    start_s: float = 0.0,
    duration_s: float | None = None,
    config: RRHRVConfig | None = None,
) -> tuple[WFDBSegment, RRHRVBranchResult]:
    segment = load_wfdb_segment(
        record_path,
        channel=channel,
        start_s=start_s,
        duration_s=duration_s,
    )
    result = run_rr_hrv_array(
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


def run_rr_hrv_segment(
    edf_path: str | Path,
    *,
    channel_label: str | None,
    start_s: float,
    duration_s: float,
    primary_method: str = "unsw",
) -> tuple[SignalSegment, pd.DataFrame, dict, dict[str, object]]:
    """Compatibility wrapper for the original notebook/CLI return shape."""

    if primary_method not in {"unsw", "pantompkins1985", "neurokit"}:
        raise ValueError("primary_method must be 'unsw', 'pantompkins1985', or 'neurokit'")
    segment, result = run_rr_hrv_edf(
        edf_path,
        channel_label=channel_label,
        start_s=start_s,
        duration_s=duration_s,
    )
    selected = result.selected
    pan_300 = selected.pan_context[300.0]
    compatibility_peaks = {
        "unsw": selected.primary.peak_samples,
        "pantompkins1985": pan_300.peak_samples,
        "neurokit": selected.secondary_experimental.peak_samples,
        "pan_tompkins": pan_300.peak_samples,
        "pan_cleaned": pan_300.analysis_signal,
        "neurokit_cleaned": selected.secondary_experimental.analysis_signal,
        "agreement": selected.pan_agreements[300.0],
        "complete_result": result,
    }
    metadata = result.summary()
    metadata["compatibility_primary_method_argument"] = primary_method
    metadata["compatibility_note"] = (
        "The complete architecture always uses UNSW primary; the legacy "
        "primary_method argument no longer changes RR ownership."
    )
    return segment, selected.features, compatibility_peaks, metadata


def _run_orientation(
    signal: np.ndarray,
    sampling_rate_hz: float,
    *,
    segment_start_s: float,
    orientation: str,
    config: RRHRVConfig,
) -> OrientationBranchResult:
    primary = detect_unsw(signal, sampling_rate_hz, orientation=orientation)
    secondary_experimental = detect_neurokit_gradient(
        signal,
        sampling_rate_hz,
        orientation=orientation,
        minimum_delay_ms=config.neurokit_experimental_min_delay_ms,
        minimum_delay_inclusive=config.min_delay_inclusive,
    )
    secondary_baseline = detect_neurokit_gradient(
        signal,
        sampling_rate_hz,
        orientation=orientation,
        minimum_delay_ms=config.neurokit_baseline_min_delay_ms,
        minimum_delay_inclusive=False,
    )
    pan_context = {
        float(delay): detect_pan_tompkins_context(
            signal,
            sampling_rate_hz,
            orientation=orientation,
            minimum_delay_ms=delay,
            minimum_delay_inclusive=config.min_delay_inclusive,
        )
        for delay in config.pan_context_min_delays_ms
    }

    reliability = assess_rr_reliability(
        primary.peak_samples,
        secondary_experimental.peak_samples,
        primary.analysis_signal,
        sampling_rate_hz=sampling_rate_hz,
        segment_start_s=segment_start_s,
        close_detection_exclusion_ms=config.close_detection_exclusion_ms,
        support_tolerance_ms=config.support_tolerance_ms,
        r_fiducial_refinement_ms=config.r_fiducial_refinement_ms,
    )
    baseline_reliability = assess_rr_reliability(
        primary.peak_samples,
        secondary_baseline.peak_samples,
        primary.analysis_signal,
        sampling_rate_hz=sampling_rate_hz,
        segment_start_s=segment_start_s,
        close_detection_exclusion_ms=config.close_detection_exclusion_ms,
        support_tolerance_ms=config.support_tolerance_ms,
        r_fiducial_refinement_ms=config.r_fiducial_refinement_ms,
    )
    legacy_reliability = assess_rr_reliability(
        primary.peak_samples,
        secondary_experimental.peak_samples,
        primary.analysis_signal,
        sampling_rate_hz=sampling_rate_hz,
        segment_start_s=segment_start_s,
        close_detection_exclusion_ms=config.close_detection_exclusion_ms,
        support_tolerance_ms=config.legacy_support_tolerance_ms,
        r_fiducial_refinement_ms=config.r_fiducial_refinement_ms,
    )

    intervals = reliability.rr_intervals.copy()
    intervals = _attach_reliability_ablation(
        intervals,
        baseline_reliability.rr_intervals,
        suffix="nk300_baseline",
    )
    intervals = _attach_reliability_ablation(
        intervals,
        legacy_reliability.rr_intervals,
        suffix="support150_ablation",
    )
    features = extract_rr_hrv_windows(
        intervals,
        window_size=config.hrv_window_rr_intervals,
        median_width=config.causal_rr_median_width,
        reliable_coverage=config.reliable_hrv_coverage,
    )

    primary_events = reliability.primary_events.copy()
    fiducial_candidates = build_fiducial_candidates(
        reliability.primary_collapsed_samples,
        primary.analysis_signal,
        sampling_rate_hz=sampling_rate_hz,
        tolerance_s=config.r_fiducial_refinement_ms / 1000.0,
    )
    primary_events["r_fiducial_ho_positive_sample"] = (
        fiducial_candidates.ho_positive
    )
    primary_events["r_fiducial_physiozoo_positive_sample"] = (
        fiducial_candidates.physiozoo_positive
    )
    primary_events["r_fiducial_physiozoo_negative_sample"] = (
        fiducial_candidates.physiozoo_negative
    )
    primary_events["r_fiducial_physiozoo_rqrs_adapted_sample"] = (
        fiducial_candidates.physiozoo_rqrs_adapted
    )
    primary_events["r_fiducial_rdeco_positive_backward_sample"] = (
        fiducial_candidates.rdeco_positive_backward
    )
    primary_events["r_fiducial_rdeco_negative_backward_sample"] = (
        fiducial_candidates.rdeco_negative_backward
    )
    primary_events["qrs_supported_nk300_baseline"] = (
        baseline_reliability.primary_events["qrs_supported"].to_numpy(dtype=bool)
        if len(baseline_reliability.primary_events) == len(primary_events)
        else False
    )
    primary_events["qrs_supported_support150_ablation"] = (
        legacy_reliability.primary_events["qrs_supported"].to_numpy(dtype=bool)
        if len(legacy_reliability.primary_events) == len(primary_events)
        else False
    )
    pan_agreements = {
        delay: compare_peak_sequences(
            reliability.primary_collapsed_samples,
            run.peak_samples,
            sampling_rate_hz=sampling_rate_hz,
            tolerance_ms=config.audit_match_tolerance_ms,
        )
        for delay, run in pan_context.items()
    }
    return OrientationBranchResult(
        orientation=orientation,
        primary=primary,
        secondary_experimental=secondary_experimental,
        secondary_baseline=secondary_baseline,
        pan_context=pan_context,
        reliability=reliability,
        baseline_reliability=baseline_reliability,
        legacy_tolerance_reliability=legacy_reliability,
        features=features,
        primary_events=primary_events,
        fiducial_candidates=fiducial_candidates,
        pan_agreements=pan_agreements,
    )


def _attach_reliability_ablation(
    current: pd.DataFrame,
    ablation: pd.DataFrame,
    *,
    suffix: str,
) -> pd.DataFrame:
    output = current.copy()
    if current.empty:
        output[f"rr_supported_{suffix}"] = pd.Series(dtype=bool)
        output[f"secondary_events_{suffix}"] = pd.Series(dtype=int)
        return output
    if len(current) != len(ablation):
        output[f"rr_supported_{suffix}"] = False
        output[f"secondary_events_{suffix}"] = -1
        return output
    output[f"rr_supported_{suffix}"] = ablation["rr_supported"].to_numpy(
        dtype=bool
    )
    output[f"secondary_events_{suffix}"] = ablation[
        "secondary_events_in_expanded_rr"
    ].to_numpy(dtype=int)
    return output


def _make_polarity_context(
    orientations: dict[str, OrientationBranchResult],
    *,
    selected_orientation: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, result in orientations.items():
        rows.append(
            {
                "orientation": name,
                "selected_for_rr_output": name == selected_orientation,
                "automatic_routing_applied": False,
                "unsw_primary_count": int(result.primary.peak_samples.size),
                "neurokit_250_count": int(
                    result.secondary_experimental.peak_samples.size
                ),
                "neurokit_300_count": int(result.secondary_baseline.peak_samples.size),
                "qrs_support_fraction_50ms_nk250": result.reliability.qrs_support_fraction,
                "rr_coverage_50ms_nk250": result.reliability.rr_coverage,
                "rr_coverage_50ms_nk300": result.baseline_reliability.rr_coverage,
                "rr_coverage_150ms_nk250_ablation": (
                    result.legacy_tolerance_reliability.rr_coverage
                ),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty and frame["rr_coverage_50ms_nk250"].notna().any():
        best_index = frame["rr_coverage_50ms_nk250"].astype(float).idxmax()
        frame["higher_coverage_orientation"] = False
        frame.loc[best_index, "higher_coverage_orientation"] = True
    else:
        frame["higher_coverage_orientation"] = False
    frame["higher_coverage_is_not_ground_truth"] = True
    return frame


def _compact_agreement(agreement: PeakAgreement) -> dict[str, object]:
    payload = agreement.to_dict()
    payload.pop("unmatched_primary_samples", None)
    payload.pop("unmatched_comparator_samples", None)
    return payload
