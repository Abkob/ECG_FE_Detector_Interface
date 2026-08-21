"""Experimental same-source-per-RR fusion challenger.

This is a user-proposed project ablation, not a published fusion algorithm.
UNSW continues to own heartbeat existence.  Each RR interval independently
uses NeuroKit at both boundaries, else Zhai at both boundaries, else UNSW at
both boundaries.  A NeuroKit--UNSW or Zhai--UNSW interval is never formed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .fiducial_fusion import fuse_unsw_neurokit_zhai


@dataclass(frozen=True)
class IntervalConsistentFusionResult:
    """Per-event candidates and independently selected same-source intervals."""

    events: pd.DataFrame
    rr_intervals: pd.DataFrame
    association_tolerance_ms: float

    def summary(self) -> dict[str, object]:
        counts = self.rr_intervals["rr_source"].value_counts().to_dict()
        return {
            "rr_count": int(self.rr_intervals.shape[0]),
            "association_tolerance_ms": float(self.association_tolerance_ms),
            "rr_source_counts": {str(key): int(value) for key, value in counts.items()},
            "interval_source_transition_count": int(
                self.rr_intervals["interval_source_transition"].sum()
            ),
            "mixed_endpoint_interval_count": int(
                (~self.rr_intervals["same_source_endpoints"]).sum()
            ),
            "single_global_timestamp_sequence": False,
            "warning": (
                "Every RR has same-source boundaries, but consecutive RRs may "
                "change source and the result is not one global timestamp series."
            ),
        }


def fuse_rr_intervals_consistently(
    unsw_samples: np.ndarray,
    neurokit_samples: np.ndarray,
    zhai_samples: np.ndarray,
    *,
    sampling_rate_hz: float,
    association_tolerance_ms: float,
    zhai_correlation_values: np.ndarray | None = None,
) -> IntervalConsistentFusionResult:
    """Select each RR from one detector at both interval boundaries."""

    association = fuse_unsw_neurokit_zhai(
        unsw_samples,
        neurokit_samples,
        zhai_samples,
        sampling_rate_hz=sampling_rate_hz,
        association_tolerance_ms=association_tolerance_ms,
        zhai_correlation_values=zhai_correlation_values,
    )
    return build_interval_consistent_from_associations(
        association.events,
        sampling_rate_hz=sampling_rate_hz,
        association_tolerance_ms=association_tolerance_ms,
    )


def build_interval_consistent_from_associations(
    events: pd.DataFrame,
    *,
    sampling_rate_hz: float,
    association_tolerance_ms: float,
    enable_neurokit: bool = True,
    enable_zhai: bool = True,
) -> IntervalConsistentFusionResult:
    """Build the interval rule from a previously audited association table."""

    required = {"event_index", "unsw_sample", "neurokit_unique_sample", "zhai_unique_sample"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Association table is missing columns: {sorted(missing)}")
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")
    ordered = events.sort_values("event_index").reset_index(drop=True).copy()
    if not np.array_equal(
        ordered["event_index"].to_numpy(int),
        np.arange(ordered.shape[0], dtype=int),
    ):
        raise ValueError("event_index must be contiguous from zero")
    unsw = ordered["unsw_sample"].to_numpy(np.int64)
    if unsw.size > 1 and np.any(np.diff(unsw) <= 0):
        raise ValueError("UNSW samples must be strictly increasing")

    rows: list[dict[str, object]] = []
    previous_source: str | None = None
    previous_end_sample: int | None = None
    for rr_index in range(max(0, ordered.shape[0] - 1)):
        left = ordered.iloc[rr_index]
        right = ordered.iloc[rr_index + 1]
        source = "unsw"
        start = int(left["unsw_sample"])
        end = int(right["unsw_sample"])
        reason = "unsw_same_source_fallback"
        nk_available = enable_neurokit and pd.notna(left["neurokit_unique_sample"]) and pd.notna(
            right["neurokit_unique_sample"]
        )
        zhai_available = enable_zhai and pd.notna(left["zhai_unique_sample"]) and pd.notna(
            right["zhai_unique_sample"]
        )
        if nk_available:
            source = "neurokit"
            start = int(left["neurokit_unique_sample"])
            end = int(right["neurokit_unique_sample"])
            reason = "both_boundaries_unique_neurokit"
        elif zhai_available:
            source = "zhai"
            start = int(left["zhai_unique_sample"])
            end = int(right["zhai_unique_sample"])
            reason = "both_boundaries_unique_zhai_after_no_neurokit_pair"

        ordering_fallback = False
        if end <= start:
            source = "unsw"
            start = int(left["unsw_sample"])
            end = int(right["unsw_sample"])
            reason = "unsw_fallback_nonpositive_independent_rr"
            ordering_fallback = True
        transition = previous_source is not None and source != previous_source
        shared_boundary_discontinuity_ms = (
            (start - int(previous_end_sample)) * 1000.0 / sampling_rate_hz
            if previous_end_sample is not None
            else np.nan
        )
        rows.append(
            {
                "rr_index": rr_index,
                "start_sample": start,
                "end_sample": end,
                "rr_samples": end - start,
                "rr_ms": (end - start) * 1000.0 / sampling_rate_hz,
                "rr_source": source,
                "start_source": source,
                "end_source": source,
                "same_source_endpoints": True,
                "selection_reason": reason,
                "neurokit_pair_available": bool(nk_available),
                "zhai_pair_available": bool(zhai_available),
                "interval_source_transition": bool(transition),
                "previous_rr_source": previous_source,
                "shared_boundary_discontinuity_ms": shared_boundary_discontinuity_ms,
                "ordering_fallback": bool(ordering_fallback),
                "unsw_rr_ms": (
                    int(right["unsw_sample"]) - int(left["unsw_sample"])
                )
                * 1000.0
                / sampling_rate_hz,
            }
        )
        previous_source = source
        previous_end_sample = end
    return IntervalConsistentFusionResult(
        events=ordered,
        rr_intervals=pd.DataFrame(rows),
        association_tolerance_ms=float(association_tolerance_ms),
    )
