"""UNSW-owned RR timestamps with NeuroKit reliability context.

This module never substitutes a NeuroKit timestamp into the RR sequence.
It implements a project adaptation of the primary/secondary comparison
structure described by Ho et al. and exports the qSQI Dice form
``2 * matches / (primary_count + secondary_count)``.

The score formula is published; applying it to UNSW and NeuroKit in centered
and trailing 10-second moving windows is an explicitly labelled adaptation.
Li's bSQI is the different Jaccard form and is implemented under its own name
in :mod:`ecg_cascade.signal_quality`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RRContextResult:
    """Unchanged UNSW timestamps plus beat/RR/window reliability context."""

    unsw_samples: np.ndarray
    neurokit_samples: np.ndarray
    events: pd.DataFrame
    rr_intervals: pd.DataFrame
    support_tolerance_ms: float
    agreement_window_s: float

    def summary(self) -> dict[str, object]:
        return {
            "unsw_event_count": int(self.unsw_samples.size),
            "neurokit_event_count": int(self.neurokit_samples.size),
            "support_tolerance_ms": float(self.support_tolerance_ms),
            "agreement_window_s": float(self.agreement_window_s),
            "supported_beat_fraction": (
                float(self.events["unsw_supported"].mean())
                if not self.events.empty
                else None
            ),
            "supported_rr_fraction": (
                float(self.rr_intervals["rr_supported"].mean())
                if not self.rr_intervals.empty
                else None
            ),
            "timestamp_owner": "unsw_initial_unchanged",
            "neurokit_role": "context_only_no_timestamp_substitution",
        }


def build_unsw_neurokit_rr_context(
    unsw_samples: np.ndarray,
    neurokit_samples: np.ndarray,
    *,
    sampling_rate_hz: float,
    support_tolerance_ms: float,
    agreement_window_s: float = 10.0,
) -> RRContextResult:
    """Calculate support fields while preserving every UNSW timestamp.

    Per-beat support follows the Ho structure but uses the caller-selected
    tolerance.  The extra requirement of exactly two NeuroKit events in the
    tolerance-expanded RR interval is a project rule, not Ho's published
    endpoint-only RR definition.  The exact 150-ms endpoint-only feature is
    ``rr_support_ho2024`` in :mod:`ecg_cascade.signal_quality`.

    The centered 10-second agreement score is offline context.  The trailing
    score uses only events at or before the current UNSW event and is suitable
    for a causal architecture.  Neither score is a calibrated probability.
    """

    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")
    if support_tolerance_ms <= 0:
        raise ValueError("support_tolerance_ms must be positive")
    if agreement_window_s <= 0:
        raise ValueError("agreement_window_s must be positive")
    unsw = _strictly_increasing(unsw_samples, "unsw_samples")
    neurokit = _strictly_increasing(neurokit_samples, "neurokit_samples")
    tolerance = max(
        1,
        int(np.rint(support_tolerance_ms * sampling_rate_hz / 1000.0)),
    )
    window = max(1, int(np.rint(agreement_window_s * sampling_rate_hz)))
    half_window = window // 2

    event_rows: list[dict[str, object]] = []
    for event_index, primary_sample in enumerate(unsw):
        left = int(
            np.searchsorted(neurokit, primary_sample - tolerance, side="left")
        )
        right = int(
            np.searchsorted(neurokit, primary_sample + tolerance, side="right")
        )
        nearby = neurokit[left:right]
        if nearby.size:
            nearest = int(nearby[np.argmin(np.abs(nearby - primary_sample))])
            offset_ms = (nearest - int(primary_sample)) * 1000.0 / sampling_rate_hz
        else:
            nearest = pd.NA
            offset_ms = np.nan

        centered_first = int(primary_sample) - half_window
        centered_last = int(primary_sample) + (window - half_window)
        trailing_first = int(primary_sample) - window
        trailing_last = int(primary_sample) + 1
        centered_score, centered_counts = _window_agreement(
            unsw,
            neurokit,
            first_sample=centered_first,
            last_sample_exclusive=centered_last,
            tolerance_samples=tolerance,
        )
        trailing_score, trailing_counts = _window_agreement(
            unsw,
            neurokit,
            first_sample=trailing_first,
            last_sample_exclusive=trailing_last,
            tolerance_samples=tolerance,
        )
        event_rows.append(
            {
                "event_index": event_index,
                "unsw_sample": int(primary_sample),
                "neurokit_match_count": int(nearby.size),
                "nearest_neurokit_sample": nearest,
                "neurokit_minus_unsw_ms": offset_ms,
                "unsw_supported": bool(nearby.size == 1),
                "centered_qsqi_fraction": centered_score,
                "centered_window_unsw_count": centered_counts[0],
                "centered_window_neurokit_count": centered_counts[1],
                "centered_window_match_count": centered_counts[2],
                "trailing_qsqi_fraction": trailing_score,
                "trailing_window_unsw_count": trailing_counts[0],
                "trailing_window_neurokit_count": trailing_counts[1],
                "trailing_window_match_count": trailing_counts[2],
            }
        )
    events = pd.DataFrame(event_rows)
    if not events.empty:
        events["nearest_neurokit_sample"] = events[
            "nearest_neurokit_sample"
        ].astype("Int64")

    interval_rows: list[dict[str, object]] = []
    for rr_index in range(max(0, unsw.size - 1)):
        start = int(unsw[rr_index])
        end = int(unsw[rr_index + 1])
        expanded_left = int(
            np.searchsorted(neurokit, start - tolerance, side="left")
        )
        expanded_right = int(
            np.searchsorted(neurokit, end + tolerance, side="right")
        )
        secondary_count = expanded_right - expanded_left
        start_supported = bool(events.iloc[rr_index]["unsw_supported"])
        end_supported = bool(events.iloc[rr_index + 1]["unsw_supported"])
        supported = start_supported and end_supported and secondary_count == 2
        reasons: list[str] = []
        if not start_supported:
            reasons.append("unsupported_start_unsw_event")
        if not end_supported:
            reasons.append("unsupported_end_unsw_event")
        if secondary_count != 2:
            reasons.append(f"expanded_interval_neurokit_count_{secondary_count}_not_2")
        rr_samples = end - start
        interval_rows.append(
            {
                "rr_index": rr_index,
                "start_unsw_sample": start,
                "end_unsw_sample": end,
                "rr_samples": rr_samples,
                "rr_ms": rr_samples * 1000.0 / sampling_rate_hz,
                "heart_rate_bpm": 60.0 * sampling_rate_hz / rr_samples,
                "start_unsw_supported": start_supported,
                "end_unsw_supported": end_supported,
                "neurokit_events_in_expanded_rr": int(secondary_count),
                "additional_neurokit_event": bool(secondary_count > 2),
                "rr_supported": supported,
                "reliability_reason": "supported" if supported else ";".join(reasons),
                "start_neurokit_minus_unsw_ms": float(
                    events.iloc[rr_index]["neurokit_minus_unsw_ms"]
                ),
                "end_neurokit_minus_unsw_ms": float(
                    events.iloc[rr_index + 1]["neurokit_minus_unsw_ms"]
                ),
                "centered_qsqi_fraction_at_end": float(
                    events.iloc[rr_index + 1]["centered_qsqi_fraction"]
                ),
                "trailing_qsqi_fraction_at_end": float(
                    events.iloc[rr_index + 1]["trailing_qsqi_fraction"]
                ),
            }
        )
    return RRContextResult(
        unsw_samples=unsw,
        neurokit_samples=neurokit,
        events=events,
        rr_intervals=pd.DataFrame(interval_rows),
        support_tolerance_ms=float(support_tolerance_ms),
        agreement_window_s=float(agreement_window_s),
    )


def detector_agreement_fraction(
    primary_samples: np.ndarray,
    secondary_samples: np.ndarray,
    *,
    tolerance_samples: int,
) -> float:
    """Return ``2M/(N_primary+N_secondary)`` using monotone matching."""

    primary = np.asarray(primary_samples, dtype=np.int64)
    secondary = np.asarray(secondary_samples, dtype=np.int64)
    denominator = int(primary.size + secondary.size)
    if denominator == 0:
        return float("nan")
    i = 0
    j = 0
    matches = 0
    while i < primary.size and j < secondary.size:
        delta = int(primary[i] - secondary[j])
        if abs(delta) <= tolerance_samples:
            matches += 1
            i += 1
            j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    return 2.0 * matches / denominator


def _window_agreement(
    primary: np.ndarray,
    secondary: np.ndarray,
    *,
    first_sample: int,
    last_sample_exclusive: int,
    tolerance_samples: int,
) -> tuple[float, tuple[int, int, int]]:
    primary_window = _slice_events(primary, first_sample, last_sample_exclusive)
    secondary_window = _slice_events(secondary, first_sample, last_sample_exclusive)
    score = detector_agreement_fraction(
        primary_window,
        secondary_window,
        tolerance_samples=tolerance_samples,
    )
    match_count = int(
        np.rint(score * (primary_window.size + secondary_window.size) / 2.0)
    ) if np.isfinite(score) else 0
    return score, (
        int(primary_window.size),
        int(secondary_window.size),
        match_count,
    )


def _slice_events(events: np.ndarray, first: int, last: int) -> np.ndarray:
    left = int(np.searchsorted(events, first, side="left"))
    right = int(np.searchsorted(events, last, side="left"))
    return events[left:right]


def _strictly_increasing(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size > 1 and np.any(np.diff(array) <= 0):
        raise ValueError(f"{name} must be strictly increasing and unique")
    return array.copy()
