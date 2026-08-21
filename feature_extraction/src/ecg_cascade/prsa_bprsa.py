"""Varon-2015 phase-rectified signal averaging measurements.

This module implements the deterministic feature-extraction part of the
published 80-beat PRSA/BPRSA lane.  It deliberately does not implement the
paper's kernel spectral clustering seizure classifier.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from .peaks import orient_signal


@dataclass(frozen=True)
class PRSABPRSAResult:
    """Continuous 80-beat features and their phase-averaged curves."""

    features: pd.DataFrame
    curve_time_s: np.ndarray
    prsa_curves_ms: np.ndarray
    bprsa_curves_ms: np.ndarray
    parameters: dict[str, object]

    def summary(self) -> dict[str, object]:
        return {
            **self.parameters,
            "row_count": int(len(self.features)),
            "defined_feature_count": int(self.features["feature_defined"].sum()),
            "reliable_feature_count": int(self.features["feature_reliable"].sum()),
        }


def calculate_prsa_bprsa_feature_arrays(
    rr_ms: np.ndarray,
    r_peak_amplitudes: np.ndarray,
    *,
    end_time_s: np.ndarray | None = None,
    rr_supported: np.ndarray | None = None,
    r_amplitude_supported: np.ndarray | None = None,
    window_size: int = 80,
    resample_hz: float = 4.0,
    half_window_samples: int = 20,
    reliable_coverage: float = 1.0,
) -> PRSABPRSAResult:
    """Calculate causal moving-window PRSA and BPRSA features.

    Each row represents one RR-defined beat.  The first ``window_size - 1``
    rows are explicit warm-up rows.  Every later row uses only its latest 80
    RR intervals and corresponding R-peak amplitudes.  Within that window both
    tracks are cubic-spline resampled to 4 Hz.  Decreases in the driver
    (``y[i] < y[i-1]``) are anchor points, exactly as specified by Varon et al.

    The six paper-derived values are mean RR, population SDNN, local and
    long-range PRSA slopes, and local and long-range BPRSA slopes.  The paper
    specifies 4-Hz resampling but not its interpolation boundary convention;
    SciPy's not-a-knot cubic spline is therefore an explicit implementation
    choice recorded in ``parameters``.
    """

    rr = _finite_positive_vector(rr_ms, "rr_ms")
    ramp = _finite_vector(r_peak_amplitudes, "r_peak_amplitudes")
    if rr.shape != ramp.shape:
        raise ValueError("rr_ms and r_peak_amplitudes must have the same shape")
    if window_size < 3:
        raise ValueError("window_size must be at least three")
    if not np.isfinite(resample_hz) or resample_hz <= 0:
        raise ValueError("resample_hz must be finite and positive")
    if half_window_samples < 1:
        raise ValueError("half_window_samples must be positive")
    if not 0.0 <= reliable_coverage <= 1.0:
        raise ValueError("reliable_coverage must be between zero and one")

    if end_time_s is None:
        times = np.cumsum(rr, dtype=np.float64) / 1000.0
        time_axis_source = "cumulative_rr"
    else:
        times = _finite_vector(end_time_s, "end_time_s")
        if times.shape != rr.shape:
            raise ValueError("end_time_s must have one value per RR interval")
        if times.size > 1 and np.any(np.diff(times) <= 0):
            raise ValueError("end_time_s must be strictly increasing")
        time_axis_source = "provided_event_timestamps"

    rr_support = _support_vector(rr_supported, rr.size, "rr_supported")
    ramp_support = _support_vector(
        r_amplitude_supported, rr.size, "r_amplitude_supported"
    )
    curve_count = 2 * half_window_samples + 1
    curve_time_s = (
        np.arange(-half_window_samples, half_window_samples + 1, dtype=np.float64)
        / resample_hz
    )
    prsa_curves = np.full((rr.size, curve_count), np.nan, dtype=np.float64)
    bprsa_curves = np.full((rr.size, curve_count), np.nan, dtype=np.float64)
    frame = _feature_frame(rr, ramp, times, window_size)

    for end in range(window_size - 1, rr.size):
        first = end - window_size + 1
        window_times = times[first : end + 1]
        window_rr = rr[first : end + 1]
        window_ramp = ramp[first : end + 1]
        grid = _four_hz_grid(window_times[0], window_times[-1], resample_hz)
        resampled_rr = _cubic_resample(window_times, window_rr, grid)
        resampled_ramp = _cubic_resample(window_times, window_ramp, grid)

        prsa_curve, prsa_anchor_count = _phase_rectified_curve(
            resampled_rr,
            resampled_rr,
            half_window_samples,
        )
        bprsa_curve, bprsa_anchor_count = _phase_rectified_curve(
            resampled_ramp,
            resampled_rr,
            half_window_samples,
        )
        prsa_curves[end] = prsa_curve
        bprsa_curves[end] = bprsa_curve

        prsa_s = _local_slope(prsa_curve, half_window_samples)
        prsa_delta = _long_range_slope(prsa_curve)
        bprsa_s = _local_slope(bprsa_curve, half_window_samples)
        bprsa_delta = _long_range_slope(bprsa_curve)
        nonpositive_rr_count = int(np.sum(resampled_rr <= 0.0))
        rr_min = float(np.min(window_rr))
        rr_max = float(np.max(window_rr))
        ramp_min = float(np.min(window_ramp))
        ramp_max = float(np.max(window_ramp))
        rr_overshoot_fraction = float(
            np.mean((resampled_rr < rr_min) | (resampled_rr > rr_max))
        )
        ramp_overshoot_fraction = float(
            np.mean((resampled_ramp < ramp_min) | (resampled_ramp > ramp_max))
        )
        numerical_quality_pass = bool(nonpositive_rr_count == 0)
        rr_coverage = float(np.mean(rr_support[first : end + 1]))
        ramp_coverage = float(np.mean(ramp_support[first : end + 1]))
        mathematically_defined = bool(
            np.all(np.isfinite([prsa_s, prsa_delta, bprsa_s, bprsa_delta]))
        )

        frame.loc[
            end,
            [
                "window_start_time_s",
                "window_end_time_s",
                "window_elapsed_s",
                "mean_rr80_ms",
                "sdnn80_ms",
                "prsa_s_rr_ms_per_sample",
                "prsa_delta_rr_ms_per_sample",
                "bprsa_s_r_ms_per_sample",
                "bprsa_delta_r_ms_per_sample",
                "prsa_anchor_count",
                "bprsa_anchor_count",
                "resampled_sample_count",
                "resampled_rr_nonpositive_count",
                "rr_cubic_range_overshoot_fraction",
                "r_amplitude_cubic_range_overshoot_fraction",
                "numerical_quality_pass",
                "rr_reliability_coverage",
                "r_amplitude_reliability_coverage",
                "feature_defined",
                "feature_reliable",
            ],
        ] = [
            float(window_times[0]),
            float(window_times[-1]),
            float(window_times[-1] - window_times[0]),
            float(np.mean(window_rr)),
            float(np.std(window_rr, ddof=0)),
            prsa_s,
            prsa_delta,
            bprsa_s,
            bprsa_delta,
            prsa_anchor_count,
            bprsa_anchor_count,
            int(grid.size),
            nonpositive_rr_count,
            rr_overshoot_fraction,
            ramp_overshoot_fraction,
            numerical_quality_pass,
            rr_coverage,
            ramp_coverage,
            mathematically_defined,
            bool(
                mathematically_defined
                and numerical_quality_pass
                and rr_coverage >= reliable_coverage
                and ramp_coverage >= reliable_coverage
            ),
        ]

    for column in (
        "prsa_anchor_count",
        "bprsa_anchor_count",
        "resampled_sample_count",
        "resampled_rr_nonpositive_count",
    ):
        frame[column] = frame[column].astype(int)
    frame["numerical_quality_pass"] = frame["numerical_quality_pass"].astype(bool)
    frame["feature_defined"] = frame["feature_defined"].astype(bool)
    frame["feature_reliable"] = frame["feature_reliable"].astype(bool)
    return PRSABPRSAResult(
        features=frame,
        curve_time_s=curve_time_s,
        prsa_curves_ms=prsa_curves,
        bprsa_curves_ms=bprsa_curves,
        parameters={
            "method": "Varon-2015 PRSA/BPRSA feature extraction",
            "window_beats_with_rr": int(window_size),
            "window_step_beats": 1,
            "window_overlap_beats": int(window_size - 1),
            "resample_hz": float(resample_hz),
            "half_window_samples": int(half_window_samples),
            "half_window_seconds": float(half_window_samples / resample_hz),
            "curve_sample_count": int(curve_count),
            "anchor_rule": "driver[i] < driver[i-1]",
            "prsa_driver": "RR",
            "prsa_target": "RR",
            "bprsa_driver": "R-peak amplitude (ECG-derived respiration surrogate)",
            "bprsa_target": "RR",
            "interpolation": "scipy CubicSpline, not-a-knot",
            "numerical_quality_gate": (
                "resampled RR must remain strictly positive; cubic range "
                "overshoot is reported separately and does not alter the six features"
            ),
            "time_axis_source": time_axis_source,
            "classifier_applied": False,
            "calibration_applied": False,
        },
    )


def extract_prsa_bprsa_features(
    ecg: np.ndarray,
    r_peak_samples: np.ndarray,
    sampling_rate_hz: float,
    *,
    segment_start_s: float = 0.0,
    orientation: str = "original",
    rr_supported: np.ndarray | None = None,
    r_peak_supported: np.ndarray | None = None,
    window_size: int = 80,
    resample_hz: float = 4.0,
    half_window_samples: int = 20,
    reliable_coverage: float = 1.0,
    lead_name: str = "unknown",
    anchor_track: str = "unspecified",
) -> PRSABPRSAResult:
    """Build RR/R-amplitude tracks from ECG and run the 80-beat extractor."""

    signal = _finite_vector(ecg, "ecg")
    if not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be finite and positive")
    if not np.isfinite(segment_start_s) or segment_start_s < 0:
        raise ValueError("segment_start_s must be finite and non-negative")
    peaks = np.unique(np.asarray(r_peak_samples, dtype=np.int64))
    if peaks.size < 2:
        raise ValueError("At least two R peaks are required")
    if peaks[0] < 0 or peaks[-1] >= signal.size:
        raise ValueError("R peaks must lie inside the ECG segment")

    oriented = orient_signal(signal, orientation)
    rr = np.diff(peaks).astype(np.float64) * 1000.0 / sampling_rate_hz
    times = segment_start_s + peaks[1:].astype(np.float64) / sampling_rate_hz
    ramp = oriented[peaks[1:]]
    if r_peak_supported is None:
        ramp_support = None
    else:
        peak_support = _support_vector(
            r_peak_supported, peaks.size, "r_peak_supported"
        )
        ramp_support = peak_support[1:]

    result = calculate_prsa_bprsa_feature_arrays(
        rr,
        ramp,
        end_time_s=times,
        rr_supported=rr_supported,
        r_amplitude_supported=ramp_support,
        window_size=window_size,
        resample_hz=resample_hz,
        half_window_samples=half_window_samples,
        reliable_coverage=reliable_coverage,
    )
    frame = result.features.copy()
    frame["time_s"] = times
    frame["r_peak_sample_in_segment"] = peaks[1:]
    frame["r_peak_sample_in_file"] = np.rint(
        times * sampling_rate_hz
    ).astype(np.int64)
    parameters = {
        **result.parameters,
        "sampling_rate_hz": float(sampling_rate_hz),
        "segment_start_s": float(segment_start_s),
        "orientation": orientation,
        "lead_name": str(lead_name),
        "anchor_track": str(anchor_track),
        "r_amplitude_polarity_preserved_after_requested_orientation": True,
    }
    return PRSABPRSAResult(
        features=frame,
        curve_time_s=result.curve_time_s,
        prsa_curves_ms=result.prsa_curves_ms,
        bprsa_curves_ms=result.bprsa_curves_ms,
        parameters=parameters,
    )


def _feature_frame(
    rr: np.ndarray, ramp: np.ndarray, times: np.ndarray, window_size: int
) -> pd.DataFrame:
    count = rr.size
    return pd.DataFrame(
        {
            "end_time_s": times,
            "rr_ms": rr,
            "r_peak_amplitude": ramp,
            "window_beat_count": np.minimum(np.arange(1, count + 1), window_size),
            "window_start_time_s": np.full(count, np.nan),
            "window_end_time_s": np.full(count, np.nan),
            "window_elapsed_s": np.full(count, np.nan),
            "mean_rr80_ms": np.full(count, np.nan),
            "sdnn80_ms": np.full(count, np.nan),
            "prsa_s_rr_ms_per_sample": np.full(count, np.nan),
            "prsa_delta_rr_ms_per_sample": np.full(count, np.nan),
            "bprsa_s_r_ms_per_sample": np.full(count, np.nan),
            "bprsa_delta_r_ms_per_sample": np.full(count, np.nan),
            "prsa_anchor_count": np.zeros(count, dtype=int),
            "bprsa_anchor_count": np.zeros(count, dtype=int),
            "resampled_sample_count": np.zeros(count, dtype=int),
            "resampled_rr_nonpositive_count": np.zeros(count, dtype=int),
            "rr_cubic_range_overshoot_fraction": np.full(count, np.nan),
            "r_amplitude_cubic_range_overshoot_fraction": np.full(count, np.nan),
            "numerical_quality_pass": np.zeros(count, dtype=bool),
            "rr_reliability_coverage": np.full(count, np.nan),
            "r_amplitude_reliability_coverage": np.full(count, np.nan),
            "feature_defined": np.zeros(count, dtype=bool),
            "feature_reliable": np.zeros(count, dtype=bool),
        }
    )


def _phase_rectified_curve(
    driver: np.ndarray, target: np.ndarray, half_window_samples: int
) -> tuple[np.ndarray, int]:
    curve_count = 2 * half_window_samples + 1
    curve = np.full(curve_count, np.nan, dtype=np.float64)
    if driver.shape != target.shape or driver.size < curve_count:
        return curve, 0
    anchors = np.flatnonzero(driver[1:] < driver[:-1]) + 1
    anchors = anchors[
        (anchors >= half_window_samples)
        & (anchors + half_window_samples < target.size)
    ]
    if anchors.size == 0:
        return curve, 0
    segments = np.stack(
        [
            target[
                anchor - half_window_samples : anchor + half_window_samples + 1
            ]
            for anchor in anchors
        ]
    )
    return np.mean(segments, axis=0), int(anchors.size)


def _local_slope(curve: np.ndarray, center: int) -> float:
    if curve.size < 3 or not np.all(np.isfinite(curve[[center - 1, center + 1]])):
        return np.nan
    return float((curve[center + 1] - curve[center - 1]) / 2.0)


def _long_range_slope(curve: np.ndarray) -> float:
    if curve.size < 3 or not np.all(np.isfinite(curve[[0, -1]])):
        return np.nan
    return float((curve[-1] - curve[0]) / curve.size)


def _four_hz_grid(first_s: float, last_s: float, rate_hz: float) -> np.ndarray:
    count = int(np.floor((last_s - first_s) * rate_hz + 1e-9)) + 1
    return first_s + np.arange(max(1, count), dtype=np.float64) / rate_hz


def _cubic_resample(
    times: np.ndarray, values: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    if times.size < 2:
        return np.full(grid.size, np.nan, dtype=np.float64)
    return np.asarray(CubicSpline(times, values)(grid), dtype=np.float64)


def _finite_vector(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _finite_positive_vector(values: np.ndarray, name: str) -> np.ndarray:
    result = _finite_vector(values, name)
    if np.any(result <= 0):
        raise ValueError(f"{name} must be strictly positive")
    return result


def _support_vector(values: np.ndarray | None, size: int, name: str) -> np.ndarray:
    if values is None:
        return np.ones(size, dtype=bool)
    result = np.asarray(values, dtype=bool)
    if result.ndim != 1 or result.size != size:
        raise ValueError(f"{name} must have exactly {size} values")
    return result
