"""Deterministic patient-calibrated whole-beat morphology measurements.

Each beat is bounded by the midpoints to the preceding and following R peaks.
The pre-R and post-R portions are resampled separately so the R anchor remains
at a fixed column without erasing the original RR timing stored elsewhere.
Only the first requested complete beats form the fixed patient/lead templates;
later beats are the causal evaluation rows.

This module extracts measurements.  It does not train a seizure classifier or
convert morphology novelty into an artifact or seizure label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .peaks import orient_signal


@dataclass(frozen=True)
class PatientTemplateMorphologyResult:
    """Whole-beat waveforms, fixed templates, and per-beat distances."""

    anchor_track: str
    beat_waveforms: np.ndarray
    normalized_waveforms: np.ndarray
    raw_template: np.ndarray
    normalized_template: np.ndarray
    raw_template_bank: np.ndarray
    normalized_template_bank: np.ndarray
    template_member_counts: np.ndarray
    features: pd.DataFrame
    parameters: dict[str, Any]

    def summary(self) -> dict[str, object]:
        return {
            "anchor_track": self.anchor_track,
            "complete_midpoint_beat_count": int(self.features.shape[0]),
            "calibration_beat_count": int(
                self.features["calibration_member"].sum()
            ),
            "evaluation_beat_count": int(
                self.features["evaluation_eligible"].sum()
            ),
            "template_count": int(self.raw_template_bank.shape[0]),
            "template_member_counts": self.template_member_counts.tolist(),
            "waveform_shape": list(self.beat_waveforms.shape),
            "parameters": self.parameters,
            "seizure_probability_produced": False,
            "artifact_label_produced": False,
        }


def extract_patient_template_morphology(
    ecg: np.ndarray,
    anchor_peak_samples: np.ndarray,
    sampling_rate_hz: float,
    *,
    anchor_track: str,
    patient_id: str,
    lead_name: str,
    calibration_beats: int = 20,
    pre_r_points: int = 64,
    post_r_points: int = 128,
    dtw_band_fraction: float = 0.10,
    max_templates: int = 3,
    new_template_correlation_threshold: float = 0.90,
    segment_start_s: float = 0.0,
    orientation: str = "original",
) -> PatientTemplateMorphologyResult:
    """Build a fixed initial template and score complete later cardiac cycles.

    ``calibration_beats`` counts complete midpoint-bounded cycles, not raw R
    anchors.  The first and last R anchors cannot own complete cycles because
    one neighboring midpoint is unavailable.
    """

    signal = _validate_inputs(ecg, anchor_peak_samples, sampling_rate_hz)
    if not str(patient_id).strip():
        raise ValueError("patient_id must be non-empty")
    if not str(lead_name).strip():
        raise ValueError("lead_name must be non-empty")
    if calibration_beats < 1:
        raise ValueError("calibration_beats must be positive")
    if pre_r_points < 2 or post_r_points < 2:
        raise ValueError("pre_r_points and post_r_points must each be at least 2")
    if not 0 < dtw_band_fraction <= 1:
        raise ValueError("dtw_band_fraction must be in (0, 1]")
    if max_templates < 1:
        raise ValueError("max_templates must be positive")
    if not -1 <= new_template_correlation_threshold <= 1:
        raise ValueError("new_template_correlation_threshold must be in [-1, 1]")

    oriented = orient_signal(signal, orientation)
    peaks = np.unique(np.asarray(anchor_peak_samples, dtype=np.int64))
    waveform_points = pre_r_points + post_r_points
    raw_rows: list[np.ndarray] = []
    normalized_rows: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []

    for peak_index in range(1, peaks.size - 1):
        previous_peak = int(peaks[peak_index - 1])
        peak = int(peaks[peak_index])
        next_peak = int(peaks[peak_index + 1])
        start = (previous_peak + peak) // 2
        end_exclusive = (peak + next_peak) // 2
        if not (0 <= start < peak < end_exclusive <= oriented.size):
            continue
        pre = oriented[start : peak + 1]
        post = oriented[peak:end_exclusive]
        if pre.size < 2 or post.size < 2:
            continue
        resampled_pre = _resample(pre, pre_r_points + 1)[:-1]
        resampled_post = _resample(post, post_r_points)
        waveform = np.concatenate([resampled_pre, resampled_post])
        edge_count = max(1, waveform_points // 20)
        baseline = float(
            np.median(np.concatenate([waveform[:edge_count], waveform[-edge_count:]]))
        )
        centered = waveform - baseline
        norm = float(np.linalg.norm(centered))
        normalized = (
            centered / norm if norm > np.finfo(float).eps else np.zeros_like(centered)
        )
        raw_rows.append(centered)
        normalized_rows.append(normalized)
        metadata.append(
            {
                "beat_index": peak_index,
                "anchor_sample_in_segment": peak,
                "anchor_time_s": segment_start_s + peak / sampling_rate_hz,
                "cycle_start_sample_in_segment": start,
                "cycle_end_sample_exclusive_in_segment": end_exclusive,
                "pre_r_cycle_ms": 1000.0 * (peak - start) / sampling_rate_hz,
                "post_r_cycle_ms": 1000.0 * (end_exclusive - peak) / sampling_rate_hz,
                "cycle_duration_ms": 1000.0 * (end_exclusive - start) / sampling_rate_hz,
                "baseline_amplitude": baseline,
                "anchor_amplitude_from_baseline": float(oriented[peak] - baseline),
                "peak_to_peak_amplitude": float(np.ptp(centered)),
                "waveform_rms": float(np.sqrt(np.mean(centered * centered))),
            }
        )

    if len(raw_rows) <= calibration_beats:
        raise ValueError(
            "not enough complete midpoint-bounded beats for calibration plus "
            f"evaluation: complete={len(raw_rows)}, calibration={calibration_beats}"
        )

    raw_waveforms = np.vstack(raw_rows)
    normalized_waveforms = np.vstack(normalized_rows)
    raw_template_bank, normalized_template_bank, template_member_counts = (
        _build_template_bank(
            raw_waveforms[:calibration_beats],
            normalized_waveforms[:calibration_beats],
            max_templates=max_templates,
            new_template_correlation_threshold=new_template_correlation_threshold,
        )
    )
    dominant_template_index = int(np.argmax(template_member_counts))
    raw_template = raw_template_bank[dominant_template_index]
    normalized_template = normalized_template_bank[dominant_template_index]

    feature_rows: list[dict[str, object]] = []
    for row_index, (row, raw, normalized) in enumerate(
        zip(metadata, raw_waveforms, normalized_waveforms)
    ):
        correlations = np.asarray(
            [
                _correlation(normalized, template)
                for template in normalized_template_bank
            ],
            dtype=np.float64,
        )
        finite_correlations = np.where(np.isfinite(correlations), correlations, -np.inf)
        matched_template_index = int(np.argmax(finite_correlations))
        matched_raw_template = raw_template_bank[matched_template_index]
        matched_normalized_template = normalized_template_bank[matched_template_index]
        raw_residual = raw - matched_raw_template
        normalized_residual = normalized - matched_normalized_template
        correlation = float(correlations[matched_template_index])
        dtw_cost, warp_fraction = derivative_dtw_distance(
            normalized,
            matched_normalized_template,
            band_fraction=dtw_band_fraction,
        )
        if row_index > 0:
            previous = normalized_waveforms[row_index - 1]
            previous_rmse = float(np.sqrt(np.mean((normalized - previous) ** 2)))
            previous_correlation = _correlation(normalized, previous)
        else:
            previous_rmse = np.nan
            previous_correlation = np.nan
        feature_rows.append(
            {
                **row,
                "anchor_track": anchor_track,
                "patient_id": str(patient_id),
                "lead_name": str(lead_name),
                "waveform_row_index": row_index,
                "calibration_member": bool(row_index < calibration_beats),
                "evaluation_eligible": bool(row_index >= calibration_beats),
                "matched_template_index": matched_template_index,
                "matched_template_calibration_members": int(
                    template_member_counts[matched_template_index]
                ),
                "template_correlation": correlation,
                "template_normalized_rmse": float(
                    np.sqrt(np.mean(normalized_residual * normalized_residual))
                ),
                "template_raw_residual_rms": float(
                    np.sqrt(np.mean(raw_residual * raw_residual))
                ),
                "template_raw_residual_energy": float(raw_residual @ raw_residual),
                "template_derivative_dtw_cost": dtw_cost,
                "template_dtw_warp_fraction": warp_fraction,
                "previous_beat_normalized_rmse": previous_rmse,
                "previous_beat_correlation": previous_correlation,
                "seizure_probability_produced": False,
                "artifact_label_produced": False,
            }
        )

    parameters: dict[str, Any] = {
        "method": "patient_initial_median_midpoint_cycle_template",
        "patient_id": str(patient_id),
        "lead_name": str(lead_name),
        "anchor_track": anchor_track,
        "calibration_beat_count": int(calibration_beats),
        "template_update": "fixed_after_initial_calibration",
        "beat_boundaries": "midpoints_between_neighboring_R_anchors",
        "resampling": "pre_R_and_post_R_resampled_separately",
        "pre_r_points": int(pre_r_points),
        "post_r_points": int(post_r_points),
        "waveform_point_count": int(waveform_points),
        "polarity_handling": "preserved_no_silent_inversion",
        "baseline": "median_of_first_and_last_5_percent_of_resampled_cycle",
        "normalized_shape": "baseline_centered_then_unit_L2_norm",
        "amplitude_sensitive_features": True,
        "dtw": "derivative_DTW_with_Sakoe_Chiba_band",
        "dtw_band_fraction": float(dtw_band_fraction),
        "calibration_scores_are_for_context_not_unbiased_evaluation": True,
        "multiple_template_bank": "sequential_correlation_gated_median_templates",
        "maximum_template_count": int(max_templates),
        "resulting_template_count": int(raw_template_bank.shape[0]),
        "template_member_counts": template_member_counts.tolist(),
        "new_template_correlation_threshold": float(
            new_template_correlation_threshold
        ),
        "seizure_classifier": "none",
        "artifact_classifier": "none",
    }
    return PatientTemplateMorphologyResult(
        anchor_track=anchor_track,
        beat_waveforms=raw_waveforms,
        normalized_waveforms=normalized_waveforms,
        raw_template=raw_template,
        normalized_template=normalized_template,
        raw_template_bank=raw_template_bank,
        normalized_template_bank=normalized_template_bank,
        template_member_counts=template_member_counts,
        features=pd.DataFrame(feature_rows),
        parameters=parameters,
    )


def _build_template_bank(
    raw_waveforms: np.ndarray,
    normalized_waveforms: np.ndarray,
    *,
    max_templates: int,
    new_template_correlation_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign chronological calibration beats to a small median-template bank."""

    clusters: list[list[int]] = [[0]]
    for waveform_index in range(1, normalized_waveforms.shape[0]):
        templates = [
            _normalized_median(normalized_waveforms[members])
            for members in clusters
        ]
        correlations = np.asarray(
            [
                _correlation(normalized_waveforms[waveform_index], template)
                for template in templates
            ]
        )
        finite = np.where(np.isfinite(correlations), correlations, -np.inf)
        best_cluster = int(np.argmax(finite))
        if (
            float(finite[best_cluster]) < new_template_correlation_threshold
            and len(clusters) < max_templates
        ):
            clusters.append([waveform_index])
        else:
            clusters[best_cluster].append(waveform_index)

    raw_templates = np.vstack(
        [np.median(raw_waveforms[members], axis=0) for members in clusters]
    )
    normalized_templates = np.vstack(
        [_normalized_median(normalized_waveforms[members]) for members in clusters]
    )
    counts = np.asarray([len(members) for members in clusters], dtype=np.int64)
    return raw_templates, normalized_templates, counts


def _normalized_median(waveforms: np.ndarray) -> np.ndarray:
    template = np.median(waveforms, axis=0)
    norm = float(np.linalg.norm(template))
    return template / norm if norm > np.finfo(float).eps else template


def derivative_dtw_distance(
    left: np.ndarray,
    right: np.ndarray,
    *,
    band_fraction: float = 0.10,
) -> tuple[float, float]:
    """Return normalized derivative-DTW cost and non-diagonal path fraction."""

    first = np.diff(np.asarray(left, dtype=np.float64))
    second = np.diff(np.asarray(right, dtype=np.float64))
    if first.ndim != 1 or second.ndim != 1 or first.size == 0 or second.size == 0:
        raise ValueError("DTW inputs must be one-dimensional with at least 2 values")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("DTW inputs must be finite")
    if not 0 < band_fraction <= 1:
        raise ValueError("band_fraction must be in (0, 1]")

    n, m = first.size, second.size
    band = max(abs(n - m), int(np.ceil(max(n, m) * band_fraction)))
    costs = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    steps = np.full((n + 1, m + 1), -1, dtype=np.int8)
    costs[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - band), min(m, i + band) + 1):
            candidates = (costs[i - 1, j - 1], costs[i - 1, j], costs[i, j - 1])
            step = int(np.argmin(candidates))
            costs[i, j] = abs(first[i - 1] - second[j - 1]) + candidates[step]
            steps[i, j] = step

    if not np.isfinite(costs[n, m]):
        return np.nan, np.nan
    i, j = n, m
    path_length = 0
    non_diagonal = 0
    while i > 0 or j > 0:
        step = int(steps[i, j])
        if step < 0:
            return np.nan, np.nan
        path_length += 1
        if step == 0:
            i -= 1
            j -= 1
        elif step == 1:
            i -= 1
            non_diagonal += 1
        else:
            j -= 1
            non_diagonal += 1
    return float(costs[n, m] / path_length), float(non_diagonal / path_length)


def _resample(values: np.ndarray, output_size: int) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    source_axis = np.linspace(0.0, 1.0, source.size)
    output_axis = np.linspace(0.0, 1.0, output_size)
    return np.interp(output_axis, source_axis, source)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator <= np.finfo(float).eps:
        return np.nan
    return float((left_centered @ right_centered) / denominator)


def _validate_inputs(
    ecg: np.ndarray,
    anchor_peak_samples: np.ndarray,
    sampling_rate_hz: float,
) -> np.ndarray:
    signal = np.asarray(ecg, dtype=np.float64)
    if signal.ndim != 1 or signal.size == 0 or not np.all(np.isfinite(signal)):
        raise ValueError("ECG must be a non-empty finite one-dimensional array")
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")
    peaks = np.asarray(anchor_peak_samples)
    if peaks.ndim != 1:
        raise ValueError("anchor_peak_samples must be one-dimensional")
    if peaks.size and (np.any(peaks < 0) or np.any(peaks >= signal.size)):
        raise ValueError("anchor_peak_samples contain out-of-range positions")
    return signal
