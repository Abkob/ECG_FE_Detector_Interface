"""Expert-timestamp fidelity audit for the five-beat morphology branch.

This module measures whether detector-anchored Varon features reproduce the
features obtained from expert beat timestamps.  It does not interpret either
feature sequence as a seizure label and it does not convert MIT--BIH beat
symbols into a morphology-classifier target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .hrv_fidelity import (
    lin_concordance_correlation,
    symmetric_absolute_percentage_error,
)
from .morphology import VaronMorphologyResult, extract_varon_morphology
from .peaks import PeakAgreement, compare_peak_sequences


@dataclass(frozen=True)
class MorphologyFidelityAudit:
    """One candidate timestamp track compared with expert-timestamp features."""

    candidate_track: str
    agreement: PeakAgreement
    reference: VaronMorphologyResult
    candidate: VaronMorphologyResult
    windows: pd.DataFrame
    summary: dict[str, object]


def build_reference_context(
    candidate_samples: np.ndarray,
    reference_samples: np.ndarray,
    reference_symbols: np.ndarray,
    *,
    sampling_rate_hz: float,
    tolerance_ms: float = 75.0,
) -> tuple[pd.DataFrame, PeakAgreement]:
    """Attach one-to-one expert beat labels to candidate timestamps."""

    candidates = _sorted_unique(candidate_samples, "candidate_samples")
    references = _sorted_unique(reference_samples, "reference_samples")
    symbols = np.asarray(reference_symbols, dtype=object)
    if symbols.ndim != 1 or symbols.size != references.size:
        raise ValueError("reference_symbols must match reference_samples")
    agreement = compare_peak_sequences(
        candidates,
        references,
        sampling_rate_hz=sampling_rate_hz,
        tolerance_ms=tolerance_ms,
    )
    reference_symbol_by_sample = {
        int(sample): str(symbol)
        for sample, symbol in zip(references.tolist(), symbols.tolist())
    }
    matched_reference_by_candidate = {
        int(candidate): int(reference)
        for candidate, reference in zip(
            agreement.matched_primary_samples,
            agreement.matched_comparator_samples,
        )
    }
    rows: list[dict[str, object]] = []
    for sample in candidates:
        candidate = int(sample)
        reference = matched_reference_by_candidate.get(candidate)
        rows.append(
            {
                "primary_timestamp_sample": candidate,
                "qrs_supported": reference is not None,
                "comparator_match_count": 1 if reference is not None else 0,
                "comparator_offset_ms": (
                    1000.0 * (reference - candidate) / sampling_rate_hz
                    if reference is not None
                    else np.nan
                ),
                "reference_symbol": (
                    reference_symbol_by_sample[reference]
                    if reference is not None
                    else pd.NA
                ),
                "reference_match_error_ms": (
                    1000.0 * (candidate - reference) / sampling_rate_hz
                    if reference is not None
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows), agreement


def audit_morphology_timestamp_fidelity(
    ecg: np.ndarray,
    reference_samples: np.ndarray,
    reference_symbols: np.ndarray,
    candidate_samples: np.ndarray,
    sampling_rate_hz: float,
    *,
    candidate_track: str,
    tolerance_ms: float = 75.0,
    pre_r_ms: float = 60.0,
    post_r_ms: float = 60.0,
) -> MorphologyFidelityAudit:
    """Compare candidate- and expert-anchored five-beat eigenvalue sequences.

    A window is comparable only when all five expert beats have candidate
    matches and those candidates are consecutive in the candidate detector's
    own event stream.  This prevents a missed or extra beat from being hidden
    by aligning unrelated five-beat windows after the fact.
    """

    references = _sorted_unique(reference_samples, "reference_samples")
    candidates = _sorted_unique(candidate_samples, "candidate_samples")
    symbols = np.asarray(reference_symbols, dtype=object)
    if symbols.ndim != 1 or symbols.size != references.size:
        raise ValueError("reference_symbols must match reference_samples")

    candidate_context, agreement = build_reference_context(
        candidates,
        references,
        symbols,
        sampling_rate_hz=sampling_rate_hz,
        tolerance_ms=tolerance_ms,
    )
    reference_context = pd.DataFrame(
        {
            "primary_timestamp_sample": references,
            "qrs_supported": np.ones(references.size, dtype=bool),
            "comparator_match_count": np.ones(references.size, dtype=int),
            "comparator_offset_ms": np.zeros(references.size, dtype=float),
            "reference_symbol": symbols,
            "reference_match_error_ms": np.zeros(references.size, dtype=float),
        }
    )
    reference_result = extract_varon_morphology(
        ecg,
        references,
        sampling_rate_hz,
        anchor_track="expert_atr",
        event_context=reference_context,
        pre_r_ms=pre_r_ms,
        post_r_ms=post_r_ms,
    )
    candidate_result = extract_varon_morphology(
        ecg,
        candidates,
        sampling_rate_hz,
        anchor_track=candidate_track,
        event_context=candidate_context,
        pre_r_ms=pre_r_ms,
        post_r_ms=post_r_ms,
    )

    candidate_index_by_sample = {
        int(sample): index for index, sample in enumerate(candidates.tolist())
    }
    candidate_by_reference = {
        int(reference): int(candidate)
        for candidate, reference in zip(
            agreement.matched_primary_samples,
            agreement.matched_comparator_samples,
        )
    }
    candidate_features_by_end = {
        int(row.end_beat_index): row
        for row in candidate_result.features.itertuples(index=False)
    }
    reference_features_by_end = {
        int(row.end_beat_index): row
        for row in reference_result.features.itertuples(index=False)
    }

    rows: list[dict[str, object]] = []
    for end_reference_index in range(4, references.size):
        first_reference_index = end_reference_index - 4
        reference_window = references[first_reference_index : end_reference_index + 1]
        symbol_window = symbols[first_reference_index : end_reference_index + 1]
        mapped = [candidate_by_reference.get(int(sample)) for sample in reference_window]
        all_matched = all(sample is not None for sample in mapped)
        candidate_indices = (
            [candidate_index_by_sample[int(sample)] for sample in mapped]
            if all_matched
            else []
        )
        consecutive = bool(
            all_matched
            and np.array_equal(
                np.diff(np.asarray(candidate_indices, dtype=np.int64)),
                np.ones(4, dtype=np.int64),
            )
        )
        reference_feature = reference_features_by_end[end_reference_index]
        candidate_feature = (
            candidate_features_by_end.get(candidate_indices[-1]) if consecutive else None
        )
        capture_complete = bool(
            candidate_feature is not None
            and bool(candidate_feature.published_varon_core_defined)
            and bool(reference_feature.published_varon_core_defined)
        )
        comparable = bool(consecutive and capture_complete)
        if not all_matched:
            reason = "one_or_more_expert_beats_unmatched"
        elif not consecutive:
            reason = "extra_candidate_between_matched_expert_beats"
        elif not capture_complete:
            reason = "incomplete_120ms_capture"
        else:
            reason = "comparable"

        reference_values = np.asarray(
            [getattr(reference_feature, f"lambda{i}") for i in range(1, 6)],
            dtype=np.float64,
        )
        candidate_values = (
            np.asarray(
                [getattr(candidate_feature, f"lambda{i}") for i in range(1, 6)],
                dtype=np.float64,
            )
            if comparable and candidate_feature is not None
            else np.full(5, np.nan, dtype=np.float64)
        )
        reference_total = float(np.sum(reference_values))
        candidate_total = float(np.sum(candidate_values)) if comparable else np.nan
        relative_l1 = (
            float(np.sum(np.abs(candidate_values - reference_values)) / reference_total)
            if comparable and reference_total > 0
            else np.nan
        )
        reference_spectrum = (
            reference_values / reference_total
            if reference_total > 0
            else np.full(5, np.nan)
        )
        candidate_spectrum = (
            candidate_values / candidate_total
            if comparable and candidate_total > 0
            else np.full(5, np.nan)
        )
        row: dict[str, object] = {
            "reference_window_index": end_reference_index - 4,
            "reference_start_sample": int(reference_window[0]),
            "reference_end_sample": int(reference_window[-1]),
            "reference_end_time_s": float(reference_window[-1] / sampling_rate_hz),
            "reference_symbols": "|".join(str(value) for value in symbol_window),
            "reference_symbol_transition_count": int(
                np.sum(symbol_window[1:] != symbol_window[:-1])
            ),
            "reference_non_n_beat_count": int(np.sum(symbol_window != "N")),
            "all_five_expert_beats_matched": bool(all_matched),
            "candidate_beats_consecutive": consecutive,
            "morphology_comparable": comparable,
            "unavailable_reason": reason,
            "candidate_anchor_samples": (
                "|".join(str(int(value)) for value in mapped)
                if all_matched
                else ""
            ),
            "raw_eigenvalue_relative_l1_error": relative_l1,
            "exploratory_normalized_spectrum_l1_error": (
                float(np.sum(np.abs(candidate_spectrum - reference_spectrum)))
                if comparable
                else np.nan
            ),
        }
        for component in range(1, 6):
            row[f"reference_lambda{component}"] = float(reference_values[component - 1])
            row[f"candidate_lambda{component}"] = float(candidate_values[component - 1])
        rows.append(row)

    windows = pd.DataFrame(rows)
    summary = _summarize(
        agreement,
        windows,
        reference_count=references.size,
        candidate_count=candidates.size,
        candidate_track=candidate_track,
        sampling_rate_hz=sampling_rate_hz,
        tolerance_ms=tolerance_ms,
        pre_r_ms=pre_r_ms,
        post_r_ms=post_r_ms,
    )
    return MorphologyFidelityAudit(
        candidate_track=candidate_track,
        agreement=agreement,
        reference=reference_result,
        candidate=candidate_result,
        windows=windows,
        summary=summary,
    )


def _summarize(
    agreement: PeakAgreement,
    windows: pd.DataFrame,
    *,
    reference_count: int,
    candidate_count: int,
    candidate_track: str,
    sampling_rate_hz: float,
    tolerance_ms: float,
    pre_r_ms: float,
    post_r_ms: float,
) -> dict[str, object]:
    matched = int(agreement.matched_primary_samples.size)
    fp = int(agreement.unmatched_primary_samples.size)
    fn = int(agreement.unmatched_comparator_samples.size)
    comparable = windows["morphology_comparable"].to_numpy(dtype=bool) if not windows.empty else np.array([], dtype=bool)
    payload: dict[str, object] = {
        "candidate_track": candidate_track,
        "reference_beat_count": int(reference_count),
        "candidate_beat_count": int(candidate_count),
        "matched_beat_count": matched,
        "false_positive_count": fp,
        "missed_reference_count": fn,
        "beat_sensitivity": _divide(matched, matched + fn),
        "beat_ppv": _divide(matched, matched + fp),
        "beat_f1": _divide(2 * matched, 2 * matched + fp + fn),
        "median_absolute_timing_error_ms": (
            float(
                np.median(
                    np.abs(
                        1000.0
                        * (
                            agreement.matched_primary_samples
                            - agreement.matched_comparator_samples
                        )
                        / sampling_rate_hz
                    )
                )
            )
            if matched
            else np.nan
        ),
        "reference_five_beat_window_count": int(windows.shape[0]),
        "comparable_five_beat_window_count": int(np.sum(comparable)),
        "five_beat_window_coverage": _divide(int(np.sum(comparable)), int(windows.shape[0])),
        "median_raw_eigenvalue_relative_l1_error": (
            float(np.nanmedian(windows["raw_eigenvalue_relative_l1_error"]))
            if np.any(comparable)
            else np.nan
        ),
        "median_exploratory_normalized_spectrum_l1_error": (
            float(np.nanmedian(windows["exploratory_normalized_spectrum_l1_error"]))
            if np.any(comparable)
            else np.nan
        ),
        "match_tolerance_ms": float(tolerance_ms),
        "waveform_window_ms": float(pre_r_ms + post_r_ms),
        "seizure_accuracy_measured": False,
        "interpretation": (
            "Feature fidelity to expert-timestamp extraction; not seizure "
            "classification accuracy and not beat-type classification."
        ),
    }
    if np.any(comparable):
        paired = windows.loc[comparable]
        for component in range(1, 6):
            reference = paired[f"reference_lambda{component}"].to_numpy(dtype=float)
            candidate = paired[f"candidate_lambda{component}"].to_numpy(dtype=float)
            payload[f"lambda{component}_ccc"] = lin_concordance_correlation(
                reference, candidate
            )
            payload[f"lambda{component}_smape"] = symmetric_absolute_percentage_error(
                reference, candidate
            )
    else:
        for component in range(1, 6):
            payload[f"lambda{component}_ccc"] = np.nan
            payload[f"lambda{component}_smape"] = np.nan
    return payload


def _sorted_unique(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size and np.any(array < 0):
        raise ValueError(f"{name} cannot contain negative samples")
    if np.unique(array).size != array.size or np.any(np.diff(array) <= 0):
        raise ValueError(f"{name} must be strictly increasing and unique")
    return array


def _divide(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")
