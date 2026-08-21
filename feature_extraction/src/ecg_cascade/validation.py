"""Annotation-based validation helpers for the RR--HRV architecture."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .pipeline import RRHRVBranchResult


@dataclass(frozen=True)
class PeakMatch:
    detected_indices: np.ndarray
    reference_indices: np.ndarray


def match_peaks_to_reference(
    detected_samples: np.ndarray,
    reference_samples: np.ndarray,
    *,
    sampling_rate_hz: float,
    tolerance_ms: float = 75.0,
) -> PeakMatch:
    detected = np.unique(np.asarray(detected_samples, dtype=np.int64))
    reference = np.unique(np.asarray(reference_samples, dtype=np.int64))
    tolerance = int(np.rint(tolerance_ms * sampling_rate_hz / 1000.0))
    detected_matches: list[int] = []
    reference_matches: list[int] = []
    i = 0
    j = 0
    while i < detected.size and j < reference.size:
        delta = int(detected[i] - reference[j])
        if abs(delta) <= tolerance:
            detected_matches.append(i)
            reference_matches.append(j)
            i += 1
            j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    return PeakMatch(
        detected_indices=np.asarray(detected_matches, dtype=np.int64),
        reference_indices=np.asarray(reference_matches, dtype=np.int64),
    )


def peak_metrics(
    detected_samples: np.ndarray,
    reference_samples: np.ndarray,
    *,
    sampling_rate_hz: float,
    tolerance_ms: float = 75.0,
) -> dict[str, object]:
    detected = np.asarray(detected_samples, dtype=np.int64)
    reference = np.asarray(reference_samples, dtype=np.int64)
    matches = match_peaks_to_reference(
        detected,
        reference,
        sampling_rate_hz=sampling_rate_hz,
        tolerance_ms=tolerance_ms,
    )
    tp = int(matches.detected_indices.size)
    fp = int(detected.size - tp)
    fn = int(reference.size - tp)
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    f1 = (
        2 * sensitivity * ppv / (sensitivity + ppv)
        if np.isfinite(sensitivity + ppv) and sensitivity + ppv > 0
        else np.nan
    )
    if tp:
        timing_ms = (
            detected[matches.detected_indices] - reference[matches.reference_indices]
        ) * 1000.0 / sampling_rate_hz
        absolute_timing_ms = np.abs(timing_ms)
        mean_absolute = float(np.mean(absolute_timing_ms))
        median_absolute = float(np.median(absolute_timing_ms))
        p75_absolute = float(np.percentile(absolute_timing_ms, 75))
        p90_absolute = float(np.percentile(absolute_timing_ms, 90))
        p95_absolute = float(np.percentile(absolute_timing_ms, 95))
        p99_absolute = float(np.percentile(absolute_timing_ms, 99))
        maximum_absolute = float(np.max(absolute_timing_ms))
        mean_signed = float(np.mean(timing_ms))
        median_signed = float(np.median(timing_ms))
        signed_standard_deviation = float(np.std(timing_ms, ddof=0))
    else:
        mean_absolute = np.nan
        median_absolute = np.nan
        p75_absolute = np.nan
        p90_absolute = np.nan
        p95_absolute = np.nan
        p99_absolute = np.nan
        maximum_absolute = np.nan
        mean_signed = np.nan
        median_signed = np.nan
        signed_standard_deviation = np.nan
    return {
        "detected": int(detected.size),
        "reference": int(reference.size),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "sensitivity": sensitivity,
        "ppv": ppv,
        "f1": f1,
        "mean_absolute_timing_ms": mean_absolute,
        "median_absolute_timing_ms": median_absolute,
        "p75_absolute_timing_ms": p75_absolute,
        "p90_absolute_timing_ms": p90_absolute,
        "p95_absolute_timing_ms": p95_absolute,
        "p99_absolute_timing_ms": p99_absolute,
        "maximum_absolute_timing_ms": maximum_absolute,
        "mean_signed_timing_ms": mean_signed,
        "median_signed_timing_ms": median_signed,
        "signed_timing_standard_deviation_ms": signed_standard_deviation,
    }


def evaluate_branch_against_reference(
    result: RRHRVBranchResult,
    reference_samples: np.ndarray,
    *,
    sampling_rate_hz: float,
    tolerance_ms: float = 75.0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Evaluate every detector and the RR reliability claim separately."""

    branch = result.selected
    detector_rows: list[dict[str, object]] = []
    detector_specs = [
        ("unsw_initial", branch.primary.peak_samples),
        ("unsw_refined_r", branch.reliability.primary_refined_samples),
        ("physiozoo_qrs_adjust_positive", branch.fiducial_candidates.physiozoo_positive),
        ("physiozoo_qrs_adjust_negative", branch.fiducial_candidates.physiozoo_negative),
        ("physiozoo_rqrs_adapted", branch.fiducial_candidates.physiozoo_rqrs_adapted),
        ("rdeco_positive_backward", branch.fiducial_candidates.rdeco_positive_backward),
        ("rdeco_negative_backward", branch.fiducial_candidates.rdeco_negative_backward),
        ("neurokit_250", branch.secondary_experimental.peak_samples),
        ("neurokit_300", branch.secondary_baseline.peak_samples),
    ]
    detector_specs.extend(
        (f"pan_context_{int(delay)}", detector.peak_samples)
        for delay, detector in branch.pan_context.items()
    )
    for name, samples in detector_specs:
        detector_rows.append(
            {
                "detector": name,
                **peak_metrics(
                    samples,
                    reference_samples,
                    sampling_rate_hz=sampling_rate_hz,
                    tolerance_ms=tolerance_ms,
                ),
            }
        )
    rr_validation = _evaluate_rr_intervals(
        branch.features,
        branch.reliability.primary_refined_samples,
        reference_samples,
        sampling_rate_hz=sampling_rate_hz,
        tolerance_ms=tolerance_ms,
        quality_column="rr_supported",
    )
    baseline_validation = _evaluate_rr_intervals(
        branch.features,
        branch.reliability.primary_refined_samples,
        reference_samples,
        sampling_rate_hz=sampling_rate_hz,
        tolerance_ms=tolerance_ms,
        quality_column="rr_supported_nk300_baseline",
    )
    legacy_validation = _evaluate_rr_intervals(
        branch.features,
        branch.reliability.primary_refined_samples,
        reference_samples,
        sampling_rate_hz=sampling_rate_hz,
        tolerance_ms=tolerance_ms,
        quality_column="rr_supported_support150_ablation",
    )
    rr_validation.update(
        {f"nk300_{key}": value for key, value in baseline_validation.items()}
    )
    rr_validation.update(
        {f"support150_{key}": value for key, value in legacy_validation.items()}
    )
    return pd.DataFrame(detector_rows), rr_validation


def beat_symbol_sensitivity(
    detected_samples: np.ndarray,
    reference_samples: np.ndarray,
    reference_symbols: np.ndarray,
    *,
    sampling_rate_hz: float,
    tolerance_ms: float = 75.0,
) -> pd.DataFrame:
    matches = match_peaks_to_reference(
        detected_samples,
        reference_samples,
        sampling_rate_hz=sampling_rate_hz,
        tolerance_ms=tolerance_ms,
    )
    matched_reference = set(matches.reference_indices.tolist())
    rows: list[dict[str, object]] = []
    for symbol in sorted(set(reference_symbols.tolist())):
        indices = np.where(reference_symbols == symbol)[0]
        matched = sum(int(index) in matched_reference for index in indices)
        rows.append(
            {
                "symbol": symbol,
                "reference_count": int(indices.size),
                "matched_count": int(matched),
                "sensitivity": matched / indices.size if indices.size else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _evaluate_rr_intervals(
    intervals: pd.DataFrame,
    primary_refined: np.ndarray,
    reference: np.ndarray,
    *,
    sampling_rate_hz: float,
    tolerance_ms: float,
    quality_column: str,
) -> dict[str, object]:
    if intervals.empty:
        return {
            "rr_count": 0,
            "supported_rr_count": 0,
            "coverage": np.nan,
            "supported_rr_mae_ms": np.nan,
            "supported_rr_reference_evaluable_count": 0,
        }
    matches = match_peaks_to_reference(
        primary_refined,
        reference,
        sampling_rate_hz=sampling_rate_hz,
        tolerance_ms=tolerance_ms,
    )
    primary_to_reference = dict(
        zip(matches.detected_indices.tolist(), matches.reference_indices.tolist(), strict=True)
    )
    all_errors: list[float] = []
    supported_errors: list[float] = []
    for row_index, row in intervals.reset_index(drop=True).iterrows():
        if row_index not in primary_to_reference or row_index + 1 not in primary_to_reference:
            continue
        reference_start_index = primary_to_reference[row_index]
        reference_end_index = primary_to_reference[row_index + 1]
        if reference_end_index != reference_start_index + 1:
            continue
        reference_rr_ms = (
            reference[reference_end_index] - reference[reference_start_index]
        ) * 1000.0 / sampling_rate_hz
        error = float(row["rr_ms"] - reference_rr_ms)
        all_errors.append(error)
        if bool(row[quality_column]):
            supported_errors.append(error)
    supported_count = int(intervals[quality_column].sum())
    return {
        "rr_count": int(intervals.shape[0]),
        "supported_rr_count": supported_count,
        "coverage": float(intervals[quality_column].mean()),
        "all_evaluable_rr_count": len(all_errors),
        "all_evaluable_rr_mae_ms": (
            float(np.mean(np.abs(all_errors))) if all_errors else np.nan
        ),
        "supported_rr_reference_evaluable_count": len(supported_errors),
        "supported_rr_mae_ms": (
            float(np.mean(np.abs(supported_errors))) if supported_errors else np.nan
        ),
    }
