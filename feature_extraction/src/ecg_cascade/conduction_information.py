"""Five-beat observational summaries of conduction/repolarization timing.

This module deliberately stops before prediction.  It converts the auditable
beat measurements from :mod:`ecg_cascade.conduction_timing` into causal,
short-window descriptions that a reviewer can inspect.  It does not place the
measurements in the seizure feature matrix, choose clinical thresholds, or
label a change as pathological.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .conduction_timing import extract_conduction_timing


_MEASUREMENTS: tuple[tuple[str, str, str | None], ...] = (
    ("p_duration_ms", "p_duration", "pr_support_end_sample"),
    ("pr_interval_ms", "pr_interval", "pr_support_end_sample"),
    ("pr_segment_ms", "pr_segment", "pr_support_end_sample"),
    ("qrs_duration_ms", "qrs_duration", "qrs_support_end_sample"),
    ("qt_interval_ms", "qt_interval", "qt_support_end_sample"),
    ("jt_interval_ms", "jt_interval", "qt_support_end_sample"),
    ("t_peak_to_end_ms", "t_peak_to_end", "qt_support_end_sample"),
    ("qtc_bazett_ms", "qtc_bazett", "qt_support_end_sample"),
    ("qtc_fridericia_ms", "qtc_fridericia", "qt_support_end_sample"),
    ("qtc_framingham_ms", "qtc_framingham", "qt_support_end_sample"),
)

_DIAB_PEAK_COLUMNS = {
    "p": "p_peak_sample_in_segment",
    "q": "q_peak_sample_in_segment",
    "r": "r_peak_sample_in_segment",
    "s": "s_peak_sample_in_segment",
    "t": "t_peak_sample_in_segment",
}


@dataclass(frozen=True)
class ConductionInformationResult:
    """Beat measurements plus causal five-beat information windows."""

    beat_measurements: pd.DataFrame
    information_windows: pd.DataFrame
    qt_nine_beat_context: pd.DataFrame
    parameters: dict[str, Any]
    diagnostics: dict[str, Any]

    def summary(self) -> dict[str, object]:
        return {
            "branch_name": "conduction_repolarization_information",
            "schema_version": "conduction_information_v2",
            "beat_count": int(self.beat_measurements.shape[0]),
            "window_endpoint_count": int(
                self.information_windows["beat_index"].nunique()
            ),
            "long_summary_row_count": int(self.information_windows.shape[0]),
            "complete_window_count": int(
                self.information_windows.loc[
                    self.information_windows[
                        f"history{self.parameters['window_beats']}_complete"
                    ],
                    "beat_index",
                ].nunique()
            ),
            "parameters": self.parameters,
            "diagnostics": self.diagnostics,
            "information_only": True,
            "final_feature_matrix_eligible": False,
            "seizure_probability_produced": False,
            "clinical_interpretation_produced": False,
            "abnormality_label_produced": False,
        }


def extract_conduction_information(
    pqrst_features: pd.DataFrame,
    sampling_rate_hz: float,
    *,
    window_beats: int = 5,
    minimum_valid_beats: int = 4,
    eligibility_column: str | None = None,
    landmark_method: str = "supplied_aligned_pqrst_landmarks",
) -> ConductionInformationResult:
    """Return descriptive timing measurements without producing model inputs.

    Windows are trailing and causal.  The row for beat ``i`` uses only beat
    ``i`` and earlier beats.  A non-overlapping previous-window comparison is
    emitted only after two complete windows are available.

    ``eligibility_column`` can carry the decision of the project's separate
    artifact/quality branch.  When supplied, ineligible beats remain in the
    beat table for audit but are excluded from every window calculation.
    This module does not create or reinterpret an artifact score.
    """

    if not isinstance(window_beats, int) or window_beats < 3:
        raise ValueError("window_beats must be an integer of at least 3")
    if not isinstance(minimum_valid_beats, int):
        raise TypeError("minimum_valid_beats must be an integer")
    if not 2 <= minimum_valid_beats <= window_beats:
        raise ValueError("minimum_valid_beats must be between 2 and window_beats")

    # Preserve the frozen v1 interval equations.  Its 30-beat output is not
    # exposed here; only the hand-testable beat measurements are reused.
    timing = extract_conduction_timing(
        pqrst_features,
        sampling_rate_hz,
        variability_window_beats=30,
        minimum_valid_fraction=0.8,
        landmark_method=landmark_method,
    )
    beats = timing.beat_features.copy()
    beats = _add_paper_aligned_peak_dynamics(
        beats,
        pqrst_features,
        sampling_rate_hz=float(sampling_rate_hz),
    )
    eligibility = _resolve_eligibility(
        pqrst_features,
        eligibility_column=eligibility_column,
    )
    beats["information_window_eligible"] = eligibility
    beats["information_only"] = True
    beats["final_feature_matrix_eligible"] = False
    beats["abnormality_label_produced"] = False

    wide_windows = _build_information_windows(
        beats,
        eligibility,
        sampling_rate_hz=float(sampling_rate_hz),
        window_beats=window_beats,
        minimum_valid_beats=minimum_valid_beats,
    )
    windows = _reshape_information_windows_long(
        wide_windows,
        window_beats=window_beats,
    )
    qt_nine_beat_context = _select_nine_beat_qt_context(wide_windows)
    parameters: dict[str, Any] = {
        "branch_name": "conduction_repolarization_information",
        "schema_version": "conduction_information_v2",
        "beat_measurement_schema": "conduction_timing_v1",
        "landmark_method": str(landmark_method),
        "sampling_rate_hz": float(sampling_rate_hz),
        "window_beats": int(window_beats),
        "minimum_valid_beats": int(minimum_valid_beats),
        "window_alignment": "causal_trailing_current_and_previous_beats",
        "window_output_layout": "long_one_row_per_measurement_per_window_endpoint",
        "previous_window_comparison": (
            "current_window_vs_immediately_preceding_non_overlapping_window"
        ),
        "eligibility_column": eligibility_column,
        "eligibility_source": (
            "caller_supplied_external_quality_branch"
            if eligibility_column is not None
            else "finite_measurements_only_no_artifact_gate"
        ),
        "five_beat_statistics": [
            "mean",
            "median",
            "sample_sd",
            "raw_mad",
            "iqr",
            "minimum",
            "maximum",
            "range",
            "rmssd_of_adjacent_valid_pairs",
            "median_absolute_successive_difference",
            "maximum_absolute_successive_difference",
            "theil_sen_slope_per_beat",
            "late_two_minus_early_two_median",
            "current_minus_prior_median",
        ],
        "direct_seizure_paper_features": {
            "reference": "Diab et al. 2025; DOI 10.1016/j.neucli.2025.103098",
            "per_beat_features": [
                "P_to_R, Q_to_R, S_to_R, T_to_R within-beat peak timing",
                "P_to_P, Q_to_Q, R_to_R, S_to_S, T_to_T interbeat timing",
            ],
            "aggregation": (
                "none; paper used the ordered raw values from 3-60 beats"
            ),
            "classifier_reproduced": False,
        },
        "nine_beat_qt_context": {
            "reference": (
                "Brotherstone et al. 2010; DOI "
                "10.1111/j.1528-1167.2009.02281.x"
            ),
            "method": (
                "mean QT and mean preceding RR over nine consecutive beats, "
                "then apply each QT-correction equation"
            ),
            "seizure_detector_reproduced": False,
        },
        "clinical_thresholds_applied": False,
        "patient_baseline_applied": False,
        "seizure_classifier_applied": False,
        "included_in_final_feature_matrix": False,
    }
    diagnostics: dict[str, Any] = {
        "measurement_gate": "information_only_not_pipeline_enabled",
        "interval_level_manual_validation": "pending",
        "target_seizure_validation": "not_run",
        "eligible_beat_count": int(eligibility.sum()),
        "ineligible_beat_count": int((~eligibility).sum()),
        "interpretation": (
            "Large spread or shift can reflect physiology, rate change, ectopy, "
            "lead movement, artifact, or landmark error; no cause is assigned."
        ),
        "bad_change_rule": (
            "none; clinical or seizure abnormality requires validated patient-, "
            "lead-, rate-, quality-, and measurement-error-aware references"
        ),
        "five_beat_scope": (
            "engineering display summary; not the aggregation used by the "
            "direct PQRST seizure paper"
        ),
        "qt_variability_scope": (
            "QTVI and STVQT are intentionally absent from five-beat output; "
            "published guidance requires longer, quality-controlled, steady-rate data"
        ),
    }
    return ConductionInformationResult(
        beat_measurements=beats,
        information_windows=windows,
        qt_nine_beat_context=qt_nine_beat_context,
        parameters=parameters,
        diagnostics=diagnostics,
    )


def _reshape_information_windows_long(
    wide: pd.DataFrame,
    *,
    window_beats: int,
) -> pd.DataFrame:
    """Convert the internal wide calculation into an inspectable long table."""

    rows: list[dict[str, object]] = []
    shared_columns = (
        "beat_index",
        "r_peak_time_s",
        "patient_id",
        "lead_name",
        "window_start_beat_index",
        f"history_beats{window_beats}",
        f"history{window_beats}_complete",
        f"eligible_beats{window_beats}",
        f"eligible_fraction{window_beats}",
        f"elapsed_time{window_beats}_s",
        f"previous_nonoverlapping_window{window_beats}_available",
        "information_only",
        "final_feature_matrix_eligible",
        "seizure_probability_produced",
        "clinical_interpretation_produced",
        "abnormality_label_produced",
    )
    for _, wide_row in wide.iterrows():
        shared = {column: wide_row[column] for column in shared_columns}
        for source_column, prefix, _ in _MEASUREMENTS:
            rows.append(
                {
                    **shared,
                    "measurement": prefix,
                    "source_beat_column": source_column,
                    "unit": "ms",
                    "valid_beats": wide_row[
                        f"{prefix}_valid_beats{window_beats}"
                    ],
                    "availability_fraction": wide_row[
                        f"{prefix}_availability_fraction{window_beats}"
                    ],
                    "summary_defined": wide_row[
                        f"{prefix}_summary{window_beats}_defined"
                    ],
                    "summary_status": wide_row[
                        f"{prefix}_summary{window_beats}_status"
                    ],
                    "support_end_sample": wide_row[
                        f"{prefix}_support_end_sample{window_beats}"
                    ],
                    "mean_ms": wide_row[f"{prefix}_mean{window_beats}_ms"],
                    "median_ms": wide_row[f"{prefix}_median{window_beats}_ms"],
                    "sample_sd_ms": wide_row[f"{prefix}_sd{window_beats}_ms"],
                    "mad_ms": wide_row[f"{prefix}_mad{window_beats}_ms"],
                    "iqr_ms": wide_row[f"{prefix}_iqr{window_beats}_ms"],
                    "minimum_ms": wide_row[f"{prefix}_min{window_beats}_ms"],
                    "maximum_ms": wide_row[f"{prefix}_max{window_beats}_ms"],
                    "range_ms": wide_row[f"{prefix}_range{window_beats}_ms"],
                    "rmssd_ms": wide_row[f"{prefix}_rmssd{window_beats}_ms"],
                    "median_abs_successive_difference_ms": wide_row[
                        f"{prefix}_median_abs_successive_difference{window_beats}_ms"
                    ],
                    "max_abs_successive_difference_ms": wide_row[
                        f"{prefix}_max_abs_successive_difference{window_beats}_ms"
                    ],
                    "theil_sen_slope_ms_per_beat": wide_row[
                        f"{prefix}_theil_sen_slope_per_beat{window_beats}_ms"
                    ],
                    "late2_minus_early2_median_ms": wide_row[
                        f"{prefix}_late2_minus_early2_median{window_beats}_ms"
                    ],
                    "current_minus_prior_median_ms": wide_row[
                        f"{prefix}_current_minus_prior_median{window_beats}_ms"
                    ],
                    "previous_window_comparison_defined": wide_row[
                        f"{prefix}_previous{window_beats}_comparison_defined"
                    ],
                    "median_shift_vs_previous_window_ms": wide_row[
                        f"{prefix}_median_shift_vs_previous{window_beats}_ms"
                    ],
                    "absolute_median_shift_vs_previous_window_ms": wide_row[
                        f"{prefix}_absolute_median_shift_vs_previous{window_beats}_ms"
                    ],
                    "robust_shift_scale_ms": wide_row[
                        f"{prefix}_robust_shift_scale{window_beats}_ms"
                    ],
                    "normalized_shift_vs_previous_window": wide_row[
                        f"{prefix}_normalized_shift_vs_previous{window_beats}"
                    ],
                }
            )
    frame = pd.DataFrame(rows)
    frame["support_end_sample"] = frame["support_end_sample"].astype("Int64")
    return frame


def _select_nine_beat_qt_context(wide: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "beat_index",
        "r_peak_time_s",
        "patient_id",
        "lead_name",
        "qt_rr_mean9_defined",
        "qt_interval_mean9_ms",
        "preceding_rr_mean9_ms",
        "qtc_bazett_from_means9_ms",
        "qtc_fridericia_from_means9_ms",
        "qtc_framingham_from_means9_ms",
        "information_only",
        "final_feature_matrix_eligible",
        "seizure_probability_produced",
        "clinical_interpretation_produced",
        "abnormality_label_produced",
    ]
    return wide[columns].copy()


def _resolve_eligibility(
    pqrst_features: pd.DataFrame,
    *,
    eligibility_column: str | None,
) -> np.ndarray:
    if eligibility_column is None:
        return np.ones(pqrst_features.shape[0], dtype=bool)
    if eligibility_column not in pqrst_features.columns:
        raise ValueError(f"eligibility column not found: {eligibility_column}")
    values = pqrst_features[eligibility_column]
    if values.isna().any():
        raise ValueError("eligibility column cannot contain missing values")
    if not pd.api.types.is_bool_dtype(values.dtype):
        unique = set(values.unique().tolist())
        if not unique.issubset({True, False, 0, 1}):
            raise ValueError("eligibility column must contain boolean values")
    return values.astype(bool).to_numpy()


def _build_information_windows(
    beats: pd.DataFrame,
    eligibility: np.ndarray,
    *,
    sampling_rate_hz: float,
    window_beats: int,
    minimum_valid_beats: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    resolution_ms = 1000.0 / sampling_rate_hz
    for end in range(beats.shape[0]):
        start = max(0, end - window_beats + 1)
        history_complete = end - start + 1 == window_beats
        current = beats.iloc[start : end + 1]
        current_eligibility = eligibility[start : end + 1]
        previous_start = start - window_beats
        previous_available = history_complete and previous_start >= 0
        previous = (
            beats.iloc[previous_start:start] if previous_available else None
        )
        previous_eligibility = (
            eligibility[previous_start:start] if previous_available else None
        )
        elapsed_time_s = (
            float(current.iloc[-1]["r_peak_time_s"])
            - float(current.iloc[0]["r_peak_time_s"])
            if current.shape[0] > 1
            else 0.0
        )
        output: dict[str, object] = {
            "beat_index": int(beats.iloc[end]["beat_index"]),
            "r_peak_time_s": float(beats.iloc[end]["r_peak_time_s"]),
            "patient_id": str(beats.iloc[end]["patient_id"]),
            "lead_name": str(beats.iloc[end]["lead_name"]),
            "window_start_beat_index": int(beats.iloc[start]["beat_index"]),
            f"history_beats{window_beats}": int(current.shape[0]),
            f"history{window_beats}_complete": bool(history_complete),
            f"eligible_beats{window_beats}": int(current_eligibility.sum()),
            f"eligible_fraction{window_beats}": float(
                current_eligibility.sum() / window_beats
            ),
            f"elapsed_time{window_beats}_s": elapsed_time_s,
            f"previous_nonoverlapping_window{window_beats}_available": bool(
                previous_available
            ),
            "information_only": True,
            "final_feature_matrix_eligible": False,
            "seizure_probability_produced": False,
            "clinical_interpretation_produced": False,
            "abnormality_label_produced": False,
        }
        for column, prefix, support_column in _MEASUREMENTS:
            values = current[column].to_numpy(dtype=float)
            usable = current_eligibility & np.isfinite(values)
            finite = values[usable]
            valid_count = int(finite.size)
            defined = history_complete and valid_count >= minimum_valid_beats
            output[f"{prefix}_valid_beats{window_beats}"] = valid_count
            output[f"{prefix}_availability_fraction{window_beats}"] = (
                valid_count / window_beats
            )
            output[f"{prefix}_summary{window_beats}_defined"] = bool(defined)
            output[f"{prefix}_summary{window_beats}_status"] = _summary_status(
                history_complete=history_complete,
                valid_count=valid_count,
                minimum_valid_beats=minimum_valid_beats,
            )
            _write_current_statistics(
                output,
                prefix=prefix,
                values=values,
                usable=usable,
                defined=defined,
                window_beats=window_beats,
            )
            if support_column is not None:
                support = pd.to_numeric(
                    current[support_column], errors="coerce"
                ).to_numpy(float)
                support_usable = usable & np.isfinite(support)
                output[f"{prefix}_support_end_sample{window_beats}"] = (
                    int(np.max(support[support_usable]))
                    if np.any(support_usable)
                    else pd.NA
                )

            previous_values = (
                previous[column].to_numpy(dtype=float)
                if previous is not None
                else np.asarray([], dtype=float)
            )
            previous_usable = (
                previous_eligibility & np.isfinite(previous_values)
                if previous_eligibility is not None
                else np.asarray([], dtype=bool)
            )
            _write_previous_window_comparison(
                output,
                prefix=prefix,
                current_values=values,
                current_usable=usable,
                previous_values=previous_values,
                previous_usable=previous_usable,
                previous_available=previous_available,
                minimum_valid_beats=minimum_valid_beats,
                window_beats=window_beats,
                resolution_ms=resolution_ms,
            )
        _write_nine_beat_qt_context(
            output,
            beats,
            eligibility,
            end=end,
        )
        rows.append(output)
    frame = pd.DataFrame(rows)
    support_columns = [
        column for column in frame.columns if "_support_end_sample" in column
    ]
    for column in support_columns:
        frame[column] = frame[column].astype("Int64")
    return frame


def _write_current_statistics(
    output: dict[str, object],
    *,
    prefix: str,
    values: np.ndarray,
    usable: np.ndarray,
    defined: bool,
    window_beats: int,
) -> None:
    names = (
        "mean",
        "median",
        "sd",
        "mad",
        "iqr",
        "min",
        "max",
        "range",
        "rmssd",
        "median_abs_successive_difference",
        "max_abs_successive_difference",
        "theil_sen_slope_per_beat",
        "late2_minus_early2_median",
        "current_minus_prior_median",
    )
    if not defined:
        for name in names:
            output[f"{prefix}_{name}{window_beats}_ms"] = np.nan
        return

    finite = values[usable]
    median = float(np.median(finite))
    output[f"{prefix}_mean{window_beats}_ms"] = float(np.mean(finite))
    output[f"{prefix}_median{window_beats}_ms"] = median
    output[f"{prefix}_sd{window_beats}_ms"] = (
        float(np.std(finite, ddof=1)) if finite.size >= 2 else np.nan
    )
    output[f"{prefix}_mad{window_beats}_ms"] = float(
        np.median(np.abs(finite - median))
    )
    output[f"{prefix}_iqr{window_beats}_ms"] = float(
        np.percentile(finite, 75) - np.percentile(finite, 25)
    )
    output[f"{prefix}_min{window_beats}_ms"] = float(np.min(finite))
    output[f"{prefix}_max{window_beats}_ms"] = float(np.max(finite))
    output[f"{prefix}_range{window_beats}_ms"] = float(np.ptp(finite))
    successive = _successive_differences(values, usable)
    output[f"{prefix}_rmssd{window_beats}_ms"] = (
        float(np.sqrt(np.mean(np.square(successive))))
        if successive.size
        else np.nan
    )
    absolute_successive = np.abs(successive)
    output[
        f"{prefix}_median_abs_successive_difference{window_beats}_ms"
    ] = (
        float(np.median(absolute_successive))
        if absolute_successive.size
        else np.nan
    )
    output[f"{prefix}_max_abs_successive_difference{window_beats}_ms"] = (
        float(np.max(absolute_successive)) if absolute_successive.size else np.nan
    )
    output[f"{prefix}_theil_sen_slope_per_beat{window_beats}_ms"] = (
        _theil_sen_slope(values, usable)
    )
    early = values[:2][usable[:2]]
    late = values[-2:][usable[-2:]]
    output[f"{prefix}_late2_minus_early2_median{window_beats}_ms"] = (
        float(np.median(late) - np.median(early))
        if early.size and late.size
        else np.nan
    )
    prior = values[:-1][usable[:-1]]
    current_valid = bool(usable[-1])
    output[f"{prefix}_current_minus_prior_median{window_beats}_ms"] = (
        float(values[-1] - np.median(prior))
        if current_valid and prior.size >= 3
        else np.nan
    )


def _write_previous_window_comparison(
    output: dict[str, object],
    *,
    prefix: str,
    current_values: np.ndarray,
    current_usable: np.ndarray,
    previous_values: np.ndarray,
    previous_usable: np.ndarray,
    previous_available: bool,
    minimum_valid_beats: int,
    window_beats: int,
    resolution_ms: float,
) -> None:
    current = current_values[current_usable]
    previous = previous_values[previous_usable]
    defined = (
        previous_available
        and current.size >= minimum_valid_beats
        and previous.size >= minimum_valid_beats
    )
    output[f"{prefix}_previous{window_beats}_comparison_defined"] = bool(defined)
    if not defined:
        output[f"{prefix}_median_shift_vs_previous{window_beats}_ms"] = np.nan
        output[f"{prefix}_absolute_median_shift_vs_previous{window_beats}_ms"] = np.nan
        output[f"{prefix}_robust_shift_scale{window_beats}_ms"] = np.nan
        output[f"{prefix}_normalized_shift_vs_previous{window_beats}"] = np.nan
        return

    current_median = float(np.median(current))
    previous_median = float(np.median(previous))
    shift = current_median - previous_median
    current_mad = float(np.median(np.abs(current - current_median)))
    previous_mad = float(np.median(np.abs(previous - previous_median)))
    # 1.4826 turns a normal-distribution MAD into an SD-like scale.  Sampling
    # resolution prevents division by zero in nearly constant synthetic data.
    robust_scale = max(1.4826 * max(current_mad, previous_mad), resolution_ms)
    output[f"{prefix}_median_shift_vs_previous{window_beats}_ms"] = shift
    output[f"{prefix}_absolute_median_shift_vs_previous{window_beats}_ms"] = abs(
        shift
    )
    output[f"{prefix}_robust_shift_scale{window_beats}_ms"] = robust_scale
    output[f"{prefix}_normalized_shift_vs_previous{window_beats}"] = (
        abs(shift) / robust_scale
    )


def _summary_status(
    *,
    history_complete: bool,
    valid_count: int,
    minimum_valid_beats: int,
) -> str:
    if not history_complete:
        return "insufficient_history"
    if valid_count < minimum_valid_beats:
        return "insufficient_valid_measurements"
    return "descriptive_summary_available"


def _add_paper_aligned_peak_dynamics(
    beats: pd.DataFrame,
    pqrst_features: pd.DataFrame,
    *,
    sampling_rate_hz: float,
) -> pd.DataFrame:
    """Add the exact relative peak-timing families used by Diab et al."""

    output = beats.copy()
    peaks: dict[str, np.ndarray] = {}
    for name, column in _DIAB_PEAK_COLUMNS.items():
        peaks[name] = (
            pd.to_numeric(pqrst_features[column], errors="coerce").to_numpy(float)
            if column in pqrst_features.columns
            else np.full(output.shape[0], np.nan, dtype=float)
        )
    scale = 1000.0 / sampling_rate_hz
    for name in ("p", "q", "s", "t"):
        relative = (peaks[name] - peaks["r"]) * scale
        valid = np.isfinite(peaks[name]) & np.isfinite(peaks["r"])
        relative[~valid] = np.nan
        output[f"diab_within_{name}_relative_to_r_ms"] = relative
    for name in ("p", "q", "r", "s", "t"):
        interval = np.full(output.shape[0], np.nan, dtype=float)
        if output.shape[0] > 1:
            valid = np.isfinite(peaks[name][1:]) & np.isfinite(peaks[name][:-1])
            difference = (peaks[name][1:] - peaks[name][:-1]) * scale
            interval[1:][valid] = difference[valid]
        output[f"diab_interbeat_{name}_to_{name}_ms"] = interval
    required = [
        f"diab_within_{name}_relative_to_r_ms" for name in ("p", "q", "s", "t")
    ] + [f"diab_interbeat_{name}_to_{name}_ms" for name in ("p", "q", "r", "s", "t")]
    output["diab_all_nine_peak_dynamics_defined"] = output[required].notna().all(axis=1)
    output["diab_peak_dynamics_are_raw_not_five_beat_averaged"] = True
    return output


def _write_nine_beat_qt_context(
    output: dict[str, object],
    beats: pd.DataFrame,
    eligibility: np.ndarray,
    *,
    end: int,
) -> None:
    """Add Brotherstone-style nine-beat QT/RR averaging as context."""

    start = end - 8
    if start < 0:
        output["qt_rr_mean9_defined"] = False
        output["qt_interval_mean9_ms"] = np.nan
        output["preceding_rr_mean9_ms"] = np.nan
        output["qtc_bazett_from_means9_ms"] = np.nan
        output["qtc_fridericia_from_means9_ms"] = np.nan
        output["qtc_framingham_from_means9_ms"] = np.nan
        return
    window = beats.iloc[start : end + 1]
    qt = window["qt_interval_ms"].to_numpy(float)
    rr = window["preceding_rr_ms"].to_numpy(float)
    usable = eligibility[start : end + 1] & np.isfinite(qt) & np.isfinite(rr)
    defined = bool(np.all(usable))
    output["qt_rr_mean9_defined"] = defined
    if not defined:
        output["qt_interval_mean9_ms"] = np.nan
        output["preceding_rr_mean9_ms"] = np.nan
        output["qtc_bazett_from_means9_ms"] = np.nan
        output["qtc_fridericia_from_means9_ms"] = np.nan
        output["qtc_framingham_from_means9_ms"] = np.nan
        return
    mean_qt = float(np.mean(qt))
    mean_rr = float(np.mean(rr))
    rr_s = mean_rr / 1000.0
    output["qt_interval_mean9_ms"] = mean_qt
    output["preceding_rr_mean9_ms"] = mean_rr
    output["qtc_bazett_from_means9_ms"] = mean_qt / np.sqrt(rr_s)
    output["qtc_fridericia_from_means9_ms"] = mean_qt / np.cbrt(rr_s)
    output["qtc_framingham_from_means9_ms"] = mean_qt + 154.0 * (1.0 - rr_s)


def _successive_differences(values: np.ndarray, usable: np.ndarray) -> np.ndarray:
    if values.size < 2:
        return np.asarray([], dtype=float)
    pairs = usable[:-1] & usable[1:]
    return values[1:][pairs] - values[:-1][pairs]


def _theil_sen_slope(values: np.ndarray, usable: np.ndarray) -> float:
    positions = np.flatnonzero(usable).astype(float)
    finite = values[usable]
    if finite.size < 2:
        return np.nan
    slopes: list[float] = []
    for left in range(finite.size - 1):
        for right in range(left + 1, finite.size):
            slopes.append(
                float(
                    (finite[right] - finite[left])
                    / (positions[right] - positions[left])
                )
            )
    return float(np.median(np.asarray(slopes, dtype=float)))
