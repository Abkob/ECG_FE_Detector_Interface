"""Continuous RR/HRV measurements from the Jeppesen phase-3 method."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .peaks import PeakAgreement


def causal_median_filter(values: np.ndarray, width: int = 7) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if width <= 0:
        raise ValueError("Median-filter width must be positive")
    result = np.empty_like(values)
    for index in range(values.size):
        first = max(0, index - width + 1)
        result[index] = np.median(values[first : index + 1])
    return result


def poincare_axes(rr_ms: np.ndarray) -> tuple[float, float]:
    """Return standard Poincare SD1 and SD2 from successive RR pairs."""

    rr = np.asarray(rr_ms, dtype=np.float64)
    if rr.size < 3 or not np.all(np.isfinite(rr)):
        return np.nan, np.nan
    x1 = rr[:-1]
    x2 = rr[1:]
    sd1 = np.std((x2 - x1) / np.sqrt(2.0), ddof=1)
    sd2 = np.std((x2 + x1) / np.sqrt(2.0), ddof=1)
    return float(sd1), float(sd2)


def absolute_least_squares_slope(
    time_s: np.ndarray, heart_rate_bpm: np.ndarray
) -> float:
    time_s = np.asarray(time_s, dtype=np.float64)
    heart_rate_bpm = np.asarray(heart_rate_bpm, dtype=np.float64)
    if time_s.size < 2 or not np.all(np.isfinite(time_s + heart_rate_bpm)):
        return np.nan
    centered_time = time_s - np.mean(time_s)
    denominator = np.sum(centered_time**2)
    if denominator <= 0:
        return np.nan
    slope = np.sum(centered_time * (heart_rate_bpm - np.mean(heart_rate_bpm)))
    return float(abs(slope / denominator))


def calculate_jeppesen_feature_arrays(
    rr_ms: np.ndarray,
    *,
    end_time_s: np.ndarray | None = None,
    window_size: int = 100,
    median_width: int = 7,
) -> pd.DataFrame:
    """Vectorized form of the implemented Jeppesen 100-RR equations.

    This function is intended for multi-track validation.  It calculates the
    same continuous values as :func:`extract_rr_hrv_features` from an already
    constructed RR series.  If no coherent event-timestamp axis exists,
    ``end_time_s=None`` uses the cumulative RR tachogram as the time axis and
    exposes that choice to the caller.
    """

    rr = np.asarray(rr_ms, dtype=np.float64)
    if rr.ndim != 1:
        raise ValueError("rr_ms must be one-dimensional")
    if not np.all(np.isfinite(rr)) or np.any(rr <= 0):
        raise ValueError("RR intervals must be finite and strictly positive")
    if window_size < 3:
        raise ValueError("window_size must be at least three")
    if median_width <= 0:
        raise ValueError("median_width must be positive")
    if end_time_s is None:
        times = np.cumsum(rr) / 1000.0
    else:
        times = np.asarray(end_time_s, dtype=np.float64)
        if times.shape != rr.shape:
            raise ValueError("end_time_s must have one value per RR interval")
        if not np.all(np.isfinite(times)) or (
            times.size > 1 and np.any(np.diff(times) <= 0)
        ):
            raise ValueError("end_time_s must be finite and strictly increasing")

    filtered = causal_median_filter(rr, median_width)
    count = rr.size
    result = pd.DataFrame(
        {
            "rr_ms": rr,
            f"rr_median{median_width}_ms": filtered,
            "end_time_s": times,
            "sd1_raw_ms": np.full(count, np.nan),
            "sd2_raw_ms": np.full(count, np.nan),
            "csi100": np.full(count, np.nan),
            "sd1_filtered_ms": np.full(count, np.nan),
            "sd2_filtered_ms": np.full(count, np.nan),
            "modcsi100_filtered_ms": np.full(count, np.nan),
            "slope100_bpm_per_s": np.full(count, np.nan),
            "j1_csi_x_slope": np.full(count, np.nan),
            "j2_modcsi_filtered_x_slope": np.full(count, np.nan),
            "feature_defined": np.zeros(count, dtype=bool),
        }
    )
    if count < window_size:
        return result

    pair_width = window_size - 1
    root_two = np.sqrt(2.0)
    raw_minor = np.diff(rr) / root_two
    raw_major = (rr[1:] + rr[:-1]) / root_two
    filtered_minor = np.diff(filtered) / root_two
    filtered_major = (filtered[1:] + filtered[:-1]) / root_two
    sd1_raw = _rolling_sample_std(raw_minor, pair_width)
    sd2_raw = _rolling_sample_std(raw_major, pair_width)
    sd1_filtered = _rolling_sample_std(filtered_minor, pair_width)
    sd2_filtered = _rolling_sample_std(filtered_major, pair_width)
    slope = np.abs(
        _rolling_least_squares_slope(
            times,
            60000.0 / filtered,
            window_size,
        )
    )
    csi = np.divide(
        sd2_raw,
        sd1_raw,
        out=np.full_like(sd1_raw, np.nan),
        where=sd1_raw > 0,
    )
    modcsi = np.divide(
        4.0 * sd2_filtered**2,
        sd1_filtered,
        out=np.full_like(sd1_filtered, np.nan),
        where=sd1_filtered > 0,
    )
    j1 = csi * slope
    j2 = modcsi * slope
    first = window_size - 1
    for column, values in (
        ("sd1_raw_ms", sd1_raw),
        ("sd2_raw_ms", sd2_raw),
        ("csi100", csi),
        ("sd1_filtered_ms", sd1_filtered),
        ("sd2_filtered_ms", sd2_filtered),
        ("modcsi100_filtered_ms", modcsi),
        ("slope100_bpm_per_s", slope),
        ("j1_csi_x_slope", j1),
        ("j2_modcsi_filtered_x_slope", j2),
    ):
        result.loc[first:, column] = values
    result.loc[first:, "feature_defined"] = np.isfinite(j1) & np.isfinite(j2)
    result["feature_defined"] = result["feature_defined"].astype(bool)
    return result


def _rolling_sample_std(values: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size < width:
        return np.array([], dtype=np.float64)
    sums = _rolling_sum(values, width)
    squared_sums = _rolling_sum(values**2, width)
    variance = (squared_sums - sums**2 / width) / (width - 1)
    return np.sqrt(np.maximum(variance, 0.0))


def _rolling_least_squares_slope(
    x: np.ndarray,
    y: np.ndarray,
    width: int,
) -> np.ndarray:
    sum_x = _rolling_sum(x, width)
    sum_y = _rolling_sum(y, width)
    sum_xx = _rolling_sum(x * x, width)
    sum_xy = _rolling_sum(x * y, width)
    denominator = sum_xx - sum_x**2 / width
    numerator = sum_xy - sum_x * sum_y / width
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0,
    )


def _rolling_sum(values: np.ndarray, width: int) -> np.ndarray:
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    return cumulative[width:] - cumulative[:-width]


def extract_rr_hrv_features(
    primary_peak_samples: np.ndarray,
    *,
    sampling_rate_hz: float,
    segment_start_s: float,
    agreement: PeakAgreement,
    window_size: int = 100,
    median_width: int = 7,
) -> pd.DataFrame:
    peaks = np.unique(np.asarray(primary_peak_samples, dtype=np.int64))
    if peaks.size < 2:
        raise ValueError("At least two primary R peaks are required")
    if window_size < 3:
        raise ValueError("window_size must be at least 3")

    rr_ms = np.diff(peaks) * 1000.0 / sampling_rate_hz
    if np.any(rr_ms <= 0):
        raise ValueError("Primary R peaks did not produce strictly positive RR intervals")
    interval_end_time_s = segment_start_s + peaks[1:] / sampling_rate_hz
    heart_rate_bpm = 60000.0 / rr_ms
    rr_filtered_ms = causal_median_filter(rr_ms, median_width)
    filtered_hr_bpm = 60000.0 / rr_filtered_ms

    frame = pd.DataFrame(
        {
            "time_s": interval_end_time_s,
            "r_peak_sample_in_file": np.rint(
                interval_end_time_s * sampling_rate_hz
            ).astype(np.int64),
            "rr_ms": rr_ms,
            "heart_rate_bpm": heart_rate_bpm,
            f"rr_median{median_width}_ms": rr_filtered_ms,
            "window_rr_count": np.minimum(np.arange(1, rr_ms.size + 1), window_size),
            "sd1_raw_ms": np.nan,
            "sd2_raw_ms": np.nan,
            "csi100": np.nan,
            "sd1_filtered_ms": np.nan,
            "sd2_filtered_ms": np.nan,
            "modcsi100_filtered_ms": np.nan,
            "slope100_bpm_per_s": np.nan,
            "j1_csi_x_slope": np.nan,
            "j2_modcsi_filtered_x_slope": np.nan,
            "feature_defined": False,
            "window_has_detector_disagreement": False,
            "unmatched_primary_count": 0,
            "unmatched_comparator_count": 0,
        }
    )

    for end in range(window_size - 1, rr_ms.size):
        first = end - window_size + 1
        raw_window = rr_ms[first : end + 1]
        filtered_window = rr_filtered_ms[first : end + 1]
        time_window = interval_end_time_s[first : end + 1]
        filtered_hr_window = filtered_hr_bpm[first : end + 1]

        sd1_raw, sd2_raw = poincare_axes(raw_window)
        sd1_filtered, sd2_filtered = poincare_axes(filtered_window)
        csi = sd2_raw / sd1_raw if np.isfinite(sd1_raw) and sd1_raw > 0 else np.nan
        modcsi = (
            (4.0 * sd2_filtered) ** 2 / (4.0 * sd1_filtered)
            if np.isfinite(sd1_filtered) and sd1_filtered > 0
            else np.nan
        )
        slope = absolute_least_squares_slope(time_window, filtered_hr_window)
        j1 = csi * slope if np.isfinite(csi * slope) else np.nan
        j2 = modcsi * slope if np.isfinite(modcsi * slope) else np.nan

        window_start_sample = peaks[first]
        window_end_sample = peaks[end + 1]
        primary_disagreement_count = _count_in_closed_interval(
            agreement.unmatched_primary_samples,
            window_start_sample,
            window_end_sample,
        )
        comparator_disagreement_count = _count_in_closed_interval(
            agreement.unmatched_comparator_samples,
            window_start_sample,
            window_end_sample,
        )

        frame.loc[end, [
            "sd1_raw_ms",
            "sd2_raw_ms",
            "csi100",
            "sd1_filtered_ms",
            "sd2_filtered_ms",
            "modcsi100_filtered_ms",
            "slope100_bpm_per_s",
            "j1_csi_x_slope",
            "j2_modcsi_filtered_x_slope",
        ]] = [
            sd1_raw,
            sd2_raw,
            csi,
            sd1_filtered,
            sd2_filtered,
            modcsi,
            slope,
            j1,
            j2,
        ]
        frame.loc[end, "feature_defined"] = bool(np.isfinite(j1) and np.isfinite(j2))
        frame.loc[end, "unmatched_primary_count"] = primary_disagreement_count
        frame.loc[end, "unmatched_comparator_count"] = comparator_disagreement_count
        frame.loc[end, "window_has_detector_disagreement"] = bool(
            primary_disagreement_count or comparator_disagreement_count
        )

    return frame


def extract_rr_hrv_windows(
    rr_intervals: pd.DataFrame,
    *,
    window_size: int = 100,
    median_width: int = 7,
    reliable_coverage: float = 1.0,
) -> pd.DataFrame:
    """Add the published 100-RR Jeppesen measurements to an RR table.

    Feature values are calculated from every consecutive primary RR interval;
    unsupported intervals are never silently deleted.  A separate coverage and
    ``feature_reliable`` flag states whether the detector-support requirement
    was met.  This avoids converting detector disagreement into physiological
    missingness without recording it.
    """

    required = {"start_time_s", "end_time_s", "rr_ms", "heart_rate_bpm", "rr_supported"}
    missing = required.difference(rr_intervals.columns)
    if missing:
        raise ValueError(f"RR table is missing required columns: {sorted(missing)}")
    if window_size < 3:
        raise ValueError("window_size must be at least 3")
    if median_width <= 0:
        raise ValueError("median_width must be positive")
    if not 0 <= reliable_coverage <= 1:
        raise ValueError("reliable_coverage must be between 0 and 1")

    frame = rr_intervals.reset_index(drop=True).copy()
    rr_ms = frame["rr_ms"].to_numpy(dtype=np.float64)
    if rr_ms.size == 0:
        return _add_empty_window_columns(frame, median_width)
    if not np.all(np.isfinite(rr_ms)) or np.any(rr_ms <= 0):
        raise ValueError("RR intervals must be finite and strictly positive")

    end_time_s = frame["end_time_s"].to_numpy(dtype=np.float64)
    rr_supported = frame["rr_supported"].to_numpy(dtype=bool)
    filtered_rr_ms = causal_median_filter(rr_ms, median_width)
    filtered_hr_bpm = 60000.0 / filtered_rr_ms
    frame[f"rr_median{median_width}_ms"] = filtered_rr_ms
    frame[f"heart_rate_median{median_width}_bpm"] = filtered_hr_bpm
    frame["window_rr_count"] = np.minimum(
        np.arange(1, rr_ms.size + 1), window_size
    )
    frame["window_start_time_s"] = np.nan
    frame["window_end_time_s"] = np.nan
    frame["window_elapsed_s"] = np.nan
    frame["window_supported_rr_count"] = 0
    frame["rr_reliability_coverage"] = np.nan
    frame["window_contains_unsupported_rr"] = False
    frame["sd1_raw_ms"] = np.nan
    frame["sd2_raw_ms"] = np.nan
    frame["csi100"] = np.nan
    frame["sd1_filtered_ms"] = np.nan
    frame["sd2_filtered_ms"] = np.nan
    frame["modcsi100_filtered_ms"] = np.nan
    frame["slope100_bpm_per_s"] = np.nan
    frame["j1_csi_x_slope"] = np.nan
    frame["j2_modcsi_filtered_x_slope"] = np.nan
    frame["feature_defined"] = False
    frame["feature_reliable"] = False

    for end in range(window_size - 1, rr_ms.size):
        first = end - window_size + 1
        raw_window = rr_ms[first : end + 1]
        filtered_window = filtered_rr_ms[first : end + 1]
        time_window = end_time_s[first : end + 1]
        filtered_hr_window = filtered_hr_bpm[first : end + 1]
        support_window = rr_supported[first : end + 1]

        sd1_raw, sd2_raw = poincare_axes(raw_window)
        sd1_filtered, sd2_filtered = poincare_axes(filtered_window)
        csi = sd2_raw / sd1_raw if np.isfinite(sd1_raw) and sd1_raw > 0 else np.nan
        modcsi = (
            (4.0 * sd2_filtered) ** 2 / (4.0 * sd1_filtered)
            if np.isfinite(sd1_filtered) and sd1_filtered > 0
            else np.nan
        )
        slope = absolute_least_squares_slope(time_window, filtered_hr_window)
        j1 = csi * slope if np.isfinite(csi) and np.isfinite(slope) else np.nan
        j2 = modcsi * slope if np.isfinite(modcsi) and np.isfinite(slope) else np.nan
        supported_count = int(np.sum(support_window))
        coverage = supported_count / window_size
        mathematically_defined = bool(np.isfinite(j1) and np.isfinite(j2))

        frame.loc[end, [
            "window_start_time_s",
            "window_end_time_s",
            "window_elapsed_s",
            "window_supported_rr_count",
            "rr_reliability_coverage",
            "window_contains_unsupported_rr",
            "sd1_raw_ms",
            "sd2_raw_ms",
            "csi100",
            "sd1_filtered_ms",
            "sd2_filtered_ms",
            "modcsi100_filtered_ms",
            "slope100_bpm_per_s",
            "j1_csi_x_slope",
            "j2_modcsi_filtered_x_slope",
            "feature_defined",
            "feature_reliable",
        ]] = [
            float(frame.loc[first, "start_time_s"]),
            float(frame.loc[end, "end_time_s"]),
            float(np.sum(raw_window) / 1000.0),
            supported_count,
            coverage,
            bool(supported_count != window_size),
            sd1_raw,
            sd2_raw,
            csi,
            sd1_filtered,
            sd2_filtered,
            modcsi,
            slope,
            j1,
            j2,
            mathematically_defined,
            bool(mathematically_defined and coverage >= reliable_coverage),
        ]

    frame["window_supported_rr_count"] = frame[
        "window_supported_rr_count"
    ].astype(int)
    frame["window_contains_unsupported_rr"] = frame[
        "window_contains_unsupported_rr"
    ].astype(bool)
    frame["feature_defined"] = frame["feature_defined"].astype(bool)
    frame["feature_reliable"] = frame["feature_reliable"].astype(bool)
    return frame


def _add_empty_window_columns(frame: pd.DataFrame, median_width: int) -> pd.DataFrame:
    float_columns = [
        f"rr_median{median_width}_ms",
        f"heart_rate_median{median_width}_bpm",
        "window_start_time_s",
        "window_end_time_s",
        "window_elapsed_s",
        "rr_reliability_coverage",
        "sd1_raw_ms",
        "sd2_raw_ms",
        "csi100",
        "sd1_filtered_ms",
        "sd2_filtered_ms",
        "modcsi100_filtered_ms",
        "slope100_bpm_per_s",
        "j1_csi_x_slope",
        "j2_modcsi_filtered_x_slope",
    ]
    for column in float_columns:
        frame[column] = pd.Series(dtype=float)
    frame["window_rr_count"] = pd.Series(dtype=int)
    frame["window_supported_rr_count"] = pd.Series(dtype=int)
    frame["window_contains_unsupported_rr"] = pd.Series(dtype=bool)
    frame["feature_defined"] = pd.Series(dtype=bool)
    frame["feature_reliable"] = pd.Series(dtype=bool)
    return frame


def _count_in_closed_interval(values: np.ndarray, first: int, last: int) -> int:
    values = np.asarray(values, dtype=np.int64)
    left = np.searchsorted(values, first, side="left")
    right = np.searchsorted(values, last, side="right")
    return int(right - left)
