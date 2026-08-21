"""Experimental UNSW--NeuroKit--Zhai fiducial fusion.

UNSW owns heartbeat existence.  NeuroKit has first priority for timing when
there is a unique, bijective temporal association.  Zhai has second priority.
The original UNSW event is retained when neither independent detector can be
associated unambiguously.

This exact priority hierarchy is a project ablation, not a published method.
Ho et al. support the primary/secondary detector comparison used here, while
Zhai et al. support the independent template-correlation fiducials.  Neither
paper reports this three-source timestamp substitution rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FiducialFusionResult:
    """Selected timestamps and a complete audit trail for one ECG."""

    selected_samples: np.ndarray
    events: pd.DataFrame
    rr_intervals: pd.DataFrame
    association_tolerance_ms: float

    def summary(self) -> dict[str, object]:
        source_counts = self.events["selected_source"].value_counts().to_dict()
        switches = int(self.rr_intervals["source_transition"].sum())
        return {
            "event_count": int(self.selected_samples.size),
            "association_tolerance_ms": float(self.association_tolerance_ms),
            "source_counts": {str(k): int(v) for k, v in source_counts.items()},
            "source_switch_count": switches,
            "source_switch_fraction": (
                float(self.rr_intervals["source_transition"].mean())
                if not self.rr_intervals.empty
                else None
            ),
            "experimental_warning": (
                "The three-source priority/substitution rule is an ablation, "
                "not a published fusion algorithm."
            ),
        }


def fuse_unsw_neurokit_zhai(
    unsw_samples: np.ndarray,
    neurokit_samples: np.ndarray,
    zhai_samples: np.ndarray,
    *,
    sampling_rate_hz: float,
    association_tolerance_ms: float,
    zhai_correlation_values: np.ndarray | None = None,
) -> FiducialFusionResult:
    """Select one timestamp for every UNSW event without adding/removing beats.

    A candidate association is accepted only when exactly one candidate lies
    within the tolerance of the UNSW event *and* that candidate lies within the
    tolerance of exactly one UNSW event.  This bijective uniqueness prevents a
    candidate from being silently reused for two rapid adjacent beats.

    Priority is NeuroKit, then Zhai, then the original UNSW event.  Zhai
    correlation is exported as context and is not thresholded because no
    universal cutoff is established by the source paper.
    """

    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")
    if association_tolerance_ms <= 0:
        raise ValueError("association_tolerance_ms must be positive")
    unsw = _strictly_increasing(unsw_samples, "unsw_samples")
    neurokit = _strictly_increasing(neurokit_samples, "neurokit_samples")
    zhai = _strictly_increasing(zhai_samples, "zhai_samples")
    if zhai_correlation_values is None:
        correlations = np.full(zhai.size, np.nan, dtype=np.float64)
    else:
        correlations = np.asarray(zhai_correlation_values, dtype=np.float64)
        if correlations.ndim != 1 or correlations.size != zhai.size:
            raise ValueError(
                "zhai_correlation_values must contain one value per Zhai event"
            )

    tolerance_samples = max(
        1,
        int(np.rint(association_tolerance_ms * sampling_rate_hz / 1000.0)),
    )
    nk_index, nk_forward_count, nk_reverse_count = _unique_bijective_associations(
        unsw, neurokit, tolerance_samples
    )
    zhai_index, zhai_forward_count, zhai_reverse_count = _unique_bijective_associations(
        unsw, zhai, tolerance_samples
    )

    selected = unsw.copy()
    sources = np.full(unsw.size, "unsw", dtype=object)
    reasons = np.full(unsw.size, "no_unique_independent_match", dtype=object)
    for index in range(unsw.size):
        if nk_index[index] >= 0:
            selected[index] = neurokit[nk_index[index]]
            sources[index] = "neurokit"
            reasons[index] = "unique_bijective_neurokit_match"
        elif zhai_index[index] >= 0:
            selected[index] = zhai[zhai_index[index]]
            sources[index] = "zhai"
            reasons[index] = "unique_bijective_zhai_match_after_no_neurokit_match"

    # A timestamp substitution must never reverse heartbeat order.  If a rare
    # conflict occurs, revert all substituted members of that conflicting pair
    # to the already ordered UNSW events and preserve the fact in the audit.
    monotonicity_fallback = np.zeros(unsw.size, dtype=bool)
    for _ in range(max(1, unsw.size)):
        violations = np.flatnonzero(np.diff(selected) <= 0)
        if violations.size == 0:
            break
        for left in violations:
            for index in (int(left), int(left + 1)):
                if sources[index] != "unsw":
                    selected[index] = unsw[index]
                    sources[index] = "unsw"
                    reasons[index] = "unsw_fallback_to_preserve_monotonic_order"
                    monotonicity_fallback[index] = True
    if selected.size > 1 and np.any(np.diff(selected) <= 0):
        raise RuntimeError("Fusion could not preserve strictly increasing timestamps")

    event_rows: list[dict[str, object]] = []
    for index, primary_sample in enumerate(unsw):
        nk_candidate_index = int(nk_index[index])
        zhai_candidate_index = int(zhai_index[index])
        event_rows.append(
            {
                "event_index": index,
                "unsw_sample": int(primary_sample),
                "neurokit_unique_sample": (
                    int(neurokit[nk_candidate_index]) if nk_candidate_index >= 0 else pd.NA
                ),
                "zhai_unique_sample": (
                    int(zhai[zhai_candidate_index]) if zhai_candidate_index >= 0 else pd.NA
                ),
                "selected_sample": int(selected[index]),
                "selected_source": str(sources[index]),
                "selection_reason": str(reasons[index]),
                "neurokit_candidates_near_unsw": int(nk_forward_count[index]),
                "zhai_candidates_near_unsw": int(zhai_forward_count[index]),
                "neurokit_primary_events_near_candidate": (
                    int(nk_reverse_count[index]) if nk_candidate_index >= 0 else pd.NA
                ),
                "zhai_primary_events_near_candidate": (
                    int(zhai_reverse_count[index]) if zhai_candidate_index >= 0 else pd.NA
                ),
                "neurokit_minus_unsw_ms": (
                    (int(neurokit[nk_candidate_index]) - int(primary_sample))
                    * 1000.0
                    / sampling_rate_hz
                    if nk_candidate_index >= 0
                    else np.nan
                ),
                "zhai_minus_unsw_ms": (
                    (int(zhai[zhai_candidate_index]) - int(primary_sample))
                    * 1000.0
                    / sampling_rate_hz
                    if zhai_candidate_index >= 0
                    else np.nan
                ),
                "zhai_absolute_correlation": (
                    abs(float(correlations[zhai_candidate_index]))
                    if zhai_candidate_index >= 0
                    else np.nan
                ),
                "monotonicity_fallback": bool(monotonicity_fallback[index]),
            }
        )
    events = pd.DataFrame(event_rows)
    if not events.empty:
        events["neurokit_unique_sample"] = events["neurokit_unique_sample"].astype(
            "Int64"
        )
        events["zhai_unique_sample"] = events["zhai_unique_sample"].astype("Int64")
        events["neurokit_primary_events_near_candidate"] = events[
            "neurokit_primary_events_near_candidate"
        ].astype("Int64")
        events["zhai_primary_events_near_candidate"] = events[
            "zhai_primary_events_near_candidate"
        ].astype("Int64")

    rr_rows: list[dict[str, object]] = []
    for index in range(max(0, selected.size - 1)):
        start_source = str(sources[index])
        end_source = str(sources[index + 1])
        same_source = start_source == end_source
        selected_rr_samples = int(selected[index + 1] - selected[index])
        unsw_rr_samples = int(unsw[index + 1] - unsw[index])
        rr_rows.append(
            {
                "rr_index": index,
                "start_selected_sample": int(selected[index]),
                "end_selected_sample": int(selected[index + 1]),
                "rr_samples": selected_rr_samples,
                "rr_ms": selected_rr_samples * 1000.0 / sampling_rate_hz,
                "unsw_rr_samples": unsw_rr_samples,
                "unsw_rr_ms": unsw_rr_samples * 1000.0 / sampling_rate_hz,
                "start_source": start_source,
                "end_source": end_source,
                "same_source": same_source,
                "source_transition": not same_source,
                "same_independent_source": same_source
                and start_source in {"neurokit", "zhai"},
            }
        )
    return FiducialFusionResult(
        selected_samples=selected,
        events=events,
        rr_intervals=pd.DataFrame(rr_rows),
        association_tolerance_ms=float(association_tolerance_ms),
    )


def _unique_bijective_associations(
    primary: np.ndarray,
    candidates: np.ndarray,
    tolerance_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return candidate indices only for unique associations in both directions."""

    matched = np.full(primary.size, -1, dtype=np.int64)
    forward_counts = np.zeros(primary.size, dtype=np.int64)
    reverse_counts = np.zeros(primary.size, dtype=np.int64)
    for primary_index, primary_sample in enumerate(primary):
        left = int(
            np.searchsorted(candidates, primary_sample - tolerance_samples, side="left")
        )
        right = int(
            np.searchsorted(candidates, primary_sample + tolerance_samples, side="right")
        )
        forward_counts[primary_index] = right - left
        if right - left != 1:
            continue
        candidate_index = left
        candidate_sample = int(candidates[candidate_index])
        reverse_left = int(
            np.searchsorted(primary, candidate_sample - tolerance_samples, side="left")
        )
        reverse_right = int(
            np.searchsorted(primary, candidate_sample + tolerance_samples, side="right")
        )
        reverse_count = reverse_right - reverse_left
        reverse_counts[primary_index] = reverse_count
        if reverse_count == 1:
            matched[primary_index] = candidate_index
    return matched, forward_counts, reverse_counts


def _strictly_increasing(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size > 1 and np.any(np.diff(array) <= 0):
        raise ValueError(f"{name} must be strictly increasing and unique")
    return array.copy()
