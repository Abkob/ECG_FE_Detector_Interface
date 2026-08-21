"""Published two-detector QRS support and RR reliability measurements."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReliabilityResult:
    primary_events: pd.DataFrame
    rr_intervals: pd.DataFrame
    primary_collapsed_samples: np.ndarray
    primary_refined_samples: np.ndarray
    secondary_collapsed_samples: np.ndarray
    support_tolerance_ms: float

    @property
    def rr_coverage(self) -> float:
        if self.rr_intervals.empty:
            return float("nan")
        return float(self.rr_intervals["rr_supported"].mean())

    @property
    def qrs_support_fraction(self) -> float:
        if self.primary_events.empty:
            return float("nan")
        return float(self.primary_events["qrs_supported"].mean())

    def summary(self) -> dict[str, object]:
        return {
            "support_tolerance_ms": self.support_tolerance_ms,
            "primary_event_count": int(self.primary_events.shape[0]),
            "secondary_event_count": int(self.secondary_collapsed_samples.size),
            "qrs_supported_count": int(self.primary_events["qrs_supported"].sum()),
            "qrs_support_fraction": _finite_or_none(self.qrs_support_fraction),
            "rr_interval_count": int(self.rr_intervals.shape[0]),
            "rr_supported_count": int(self.rr_intervals["rr_supported"].sum()),
            "rr_coverage": _finite_or_none(self.rr_coverage),
        }


def collapse_close_detections(
    peak_samples: np.ndarray,
    oriented_signal: np.ndarray,
    *,
    sampling_rate_hz: float,
    exclusion_ms: float = 150.0,
) -> np.ndarray:
    """Collapse detections separated by at most ``exclusion_ms``.

    The higher *oriented* signal amplitude is retained.  This is the close-beat
    rule used in the Ho analysis and keeps polarity handling explicit.
    """

    signal = np.asarray(oriented_signal, dtype=np.float64)
    peaks = np.unique(np.asarray(peak_samples, dtype=np.int64))
    if peaks.size == 0:
        return peaks
    if peaks[0] < 0 or peaks[-1] >= signal.size:
        raise ValueError("Peak sample lies outside the ECG")
    exclusion_samples = int(np.rint(exclusion_ms * sampling_rate_hz / 1000.0))
    collapsed = [int(peaks[0])]
    for peak in peaks[1:]:
        peak = int(peak)
        if peak - collapsed[-1] > exclusion_samples:
            collapsed.append(peak)
        elif signal[peak] > signal[collapsed[-1]]:
            collapsed[-1] = peak
    return np.asarray(collapsed, dtype=np.int64)


def refine_r_fiducials(
    peak_samples: np.ndarray,
    oriented_signal: np.ndarray,
    *,
    sampling_rate_hz: float,
    radius_ms: float = 50.0,
) -> np.ndarray:
    """Apply the author implementation's highest-amplitude refinement.

    The source analysis uses the half-open NumPy slice ``[peak-radius,
    peak+radius)`` and leaves edge events unchanged when the complete search
    window is unavailable.
    """

    signal = np.asarray(oriented_signal, dtype=np.float64)
    peaks = np.asarray(peak_samples, dtype=np.int64)
    radius = int(np.rint(radius_ms * sampling_rate_hz / 1000.0))
    refined = np.empty_like(peaks)
    for index, peak in enumerate(peaks):
        first = int(peak) - radius
        last = int(peak) + radius
        if first < 0 or last > signal.size or first >= last:
            refined[index] = peak
        else:
            refined[index] = first + int(np.argmax(signal[first:last]))
    if refined.size > 1 and np.any(np.diff(refined) <= 0):
        raise RuntimeError("R-fiducial refinement produced non-increasing peaks")
    return refined


def assess_rr_reliability(
    primary_samples: np.ndarray,
    secondary_samples: np.ndarray,
    oriented_signal: np.ndarray,
    *,
    sampling_rate_hz: float,
    segment_start_s: float = 0.0,
    close_detection_exclusion_ms: float = 150.0,
    support_tolerance_ms: float = 50.0,
    r_fiducial_refinement_ms: float = 50.0,
) -> ReliabilityResult:
    """Construct per-QRS support and per-RR reliability tables.

    A primary QRS is supported only when exactly one secondary event lies
    within ±``support_tolerance_ms``.  An RR interval is supported when both
    bounding primary QRS events are supported and exactly two secondary events
    occur from the tolerance-expanded start through end of that interval.
    """

    primary = collapse_close_detections(
        primary_samples,
        oriented_signal,
        sampling_rate_hz=sampling_rate_hz,
        exclusion_ms=close_detection_exclusion_ms,
    )
    secondary = collapse_close_detections(
        secondary_samples,
        oriented_signal,
        sampling_rate_hz=sampling_rate_hz,
        exclusion_ms=close_detection_exclusion_ms,
    )
    refined = refine_r_fiducials(
        primary,
        oriented_signal,
        sampling_rate_hz=sampling_rate_hz,
        radius_ms=r_fiducial_refinement_ms,
    )
    tolerance_samples = int(
        np.rint(support_tolerance_ms * sampling_rate_hz / 1000.0)
    )
    if tolerance_samples < 1:
        raise ValueError("support_tolerance_ms is less than one sample")

    event_rows: list[dict[str, object]] = []
    for event_index, (candidate, fiducial) in enumerate(
        zip(primary, refined, strict=True)
    ):
        left = np.searchsorted(secondary, candidate - tolerance_samples, side="left")
        right = np.searchsorted(secondary, candidate + tolerance_samples, side="right")
        matches = secondary[left:right]
        if matches.size:
            nearest = int(matches[np.argmin(np.abs(matches - candidate))])
            offset_ms = (nearest - candidate) * 1000.0 / sampling_rate_hz
        else:
            nearest = pd.NA
            offset_ms = np.nan
        event_rows.append(
            {
                "event_index": event_index,
                "primary_candidate_sample": int(candidate),
                "primary_candidate_time_s": segment_start_s
                + candidate / sampling_rate_hz,
                "r_fiducial_sample": int(fiducial),
                "r_fiducial_time_s": segment_start_s
                + fiducial / sampling_rate_hz,
                "secondary_match_count": int(matches.size),
                "nearest_secondary_sample": nearest,
                "secondary_offset_ms": offset_ms,
                "qrs_supported": bool(matches.size == 1),
            }
        )
    events = pd.DataFrame(event_rows)
    if not events.empty:
        events["nearest_secondary_sample"] = events[
            "nearest_secondary_sample"
        ].astype("Int64")

    interval_rows: list[dict[str, object]] = []
    for interval_index in range(max(0, primary.size - 1)):
        start_candidate = int(primary[interval_index])
        end_candidate = int(primary[interval_index + 1])
        start_fiducial = int(refined[interval_index])
        end_fiducial = int(refined[interval_index + 1])
        left = np.searchsorted(
            secondary, start_candidate - tolerance_samples, side="left"
        )
        right = np.searchsorted(
            secondary, end_candidate + tolerance_samples, side="right"
        )
        secondary_count = int(right - left)
        start_supported = bool(events.iloc[interval_index]["qrs_supported"])
        end_supported = bool(events.iloc[interval_index + 1]["qrs_supported"])
        supported = start_supported and end_supported and secondary_count == 2
        rr_samples = end_fiducial - start_fiducial
        reasons: list[str] = []
        if not start_supported:
            reasons.append("unsupported_start_qrs")
        if not end_supported:
            reasons.append("unsupported_end_qrs")
        if secondary_count != 2:
            reasons.append(f"secondary_count_{secondary_count}_not_2")
        interval_rows.append(
            {
                "rr_index": interval_index,
                "start_primary_candidate_sample": start_candidate,
                "end_primary_candidate_sample": end_candidate,
                "start_r_fiducial_sample": start_fiducial,
                "end_r_fiducial_sample": end_fiducial,
                "start_time_s": segment_start_s
                + start_fiducial / sampling_rate_hz,
                "end_time_s": segment_start_s + end_fiducial / sampling_rate_hz,
                "rr_samples": rr_samples,
                "rr_ms": rr_samples * 1000.0 / sampling_rate_hz,
                "heart_rate_bpm": 60.0 * sampling_rate_hz / rr_samples,
                "start_qrs_supported": start_supported,
                "end_qrs_supported": end_supported,
                "secondary_events_in_expanded_rr": secondary_count,
                "rr_supported": supported,
                "reliability_reason": "supported" if supported else ";".join(reasons),
            }
        )
    intervals = pd.DataFrame(interval_rows)
    return ReliabilityResult(
        primary_events=events,
        rr_intervals=intervals,
        primary_collapsed_samples=primary,
        primary_refined_samples=refined,
        secondary_collapsed_samples=secondary,
        support_tolerance_ms=float(support_tolerance_ms),
    )


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None
