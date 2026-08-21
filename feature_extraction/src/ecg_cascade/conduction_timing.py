"""Auditable conduction and repolarization timing measurements.

This branch consumes already aligned P/QRS/T landmarks.  It does not detect
beats, delineate waves, diagnose conduction disease, or classify seizures.
Trailing summaries use only the current and earlier beats; the causality of the
upstream landmark generator remains a separate validation responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


_SAMPLE_COLUMNS = {
    "p_onset": "p_onset_sample_in_segment",
    "p_offset": "p_offset_sample_in_segment",
    "qrs_onset": "qrs_onset_sample_in_segment",
    "r_peak": "r_peak_sample_in_segment",
    "qrs_offset": "qrs_offset_sample_in_segment",
    "t_peak": "t_peak_sample_in_segment",
    "t_offset": "t_offset_sample_in_segment",
}

_ROLLING_INTERVALS = (
    "pr_interval_ms",
    "qrs_duration_ms",
    "qt_interval_ms",
    "jt_interval_ms",
)


@dataclass(frozen=True)
class ConductionTimingResult:
    """Beatwise intervals plus causal trailing timing summaries."""

    beat_features: pd.DataFrame
    rolling_features: pd.DataFrame
    parameters: dict[str, Any]
    diagnostics: dict[str, Any]

    def summary(self) -> dict[str, object]:
        return {
            "branch_name": "conduction",
            "beat_count": int(self.beat_features.shape[0]),
            "rolling_row_count": int(self.rolling_features.shape[0]),
            "pr_defined_count": int(self.beat_features["pr_interval_defined"].sum()),
            "qrs_defined_count": int(self.beat_features["qrs_duration_defined"].sum()),
            "qt_defined_count": int(self.beat_features["qt_interval_defined"].sum()),
            "parameters": self.parameters,
            "diagnostics": self.diagnostics,
            "seizure_probability_produced": False,
            "clinical_interpretation_produced": False,
        }


def qt_correct_bazett(qt_ms: np.ndarray, rr_ms: np.ndarray) -> np.ndarray:
    """Return Bazett QTc in ms: ``QT / sqrt(RR_s)``."""

    qt, rr = _validate_qt_rr(qt_ms, rr_ms)
    output = np.full(qt.size, np.nan, dtype=np.float64)
    valid = np.isfinite(qt) & np.isfinite(rr)
    output[valid] = qt[valid] / np.sqrt(rr[valid] / 1000.0)
    return output


def qt_correct_fridericia(qt_ms: np.ndarray, rr_ms: np.ndarray) -> np.ndarray:
    """Return Fridericia QTc in ms: ``QT / cbrt(RR_s)``."""

    qt, rr = _validate_qt_rr(qt_ms, rr_ms)
    output = np.full(qt.size, np.nan, dtype=np.float64)
    valid = np.isfinite(qt) & np.isfinite(rr)
    output[valid] = qt[valid] / np.cbrt(rr[valid] / 1000.0)
    return output


def qt_correct_framingham(qt_ms: np.ndarray, rr_ms: np.ndarray) -> np.ndarray:
    """Return Framingham QTc in ms: ``QT_s + 0.154 * (1 - RR_s)``."""

    qt, rr = _validate_qt_rr(qt_ms, rr_ms)
    output = np.full(qt.size, np.nan, dtype=np.float64)
    valid = np.isfinite(qt) & np.isfinite(rr)
    output[valid] = qt[valid] + 154.0 * (1.0 - rr[valid] / 1000.0)
    return output


def qt_variability_index_rr(qt_ms: np.ndarray, rr_ms: np.ndarray) -> float:
    """Return the common RR-denominator QTVI adaptation.

    This is ``log10[(var(QT)/mean(QT)^2) / (var(RR)/mean(RR)^2)]``.
    It is intentionally named ``_rr`` because Berger's original 1997 QTVI used
    heart-rate variance; both variants are exported by this module.
    """

    qt, rr = _joint_finite_positive(qt_ms, rr_ms)
    if qt.size < 2:
        return np.nan
    return _normalized_variance_log_ratio(qt, rr)


def qt_variability_index_hr(qt_ms: np.ndarray, rr_ms: np.ndarray) -> float:
    """Return Berger-style QTVI using heart-rate variance in the denominator."""

    qt, rr = _joint_finite_positive(qt_ms, rr_ms)
    if qt.size < 2:
        return np.nan
    heart_rate_bpm = 60000.0 / rr
    return _normalized_variance_log_ratio(qt, heart_rate_bpm)


def short_term_variability_30(interval_ms: np.ndarray) -> float:
    """Return published 30-beat short-term variability in ms.

    ``STV30 = sum(abs(D[n+1] - D[n])) / (30 * sqrt(2))``.  Exactly 30
    finite positive consecutive intervals are required.
    """

    values = np.asarray(interval_ms, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("interval_ms must be one-dimensional")
    if values.size != 30:
        raise ValueError("STV30 requires exactly 30 consecutive intervals")
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        return np.nan
    return float(np.sum(np.abs(np.diff(values))) / (30.0 * np.sqrt(2.0)))


def extract_conduction_timing(
    pqrst_features: pd.DataFrame,
    sampling_rate_hz: float,
    *,
    variability_window_beats: int = 30,
    minimum_valid_fraction: float = 0.8,
    landmark_method: str = "supplied_aligned_pqrst_landmarks",
) -> ConductionTimingResult:
    """Build the standalone conduction timing branch from aligned landmarks.

    The QT for beat ``i`` is paired with the preceding RR interval ending at
    that beat.  Rolling windows are trailing and therefore never read a later
    row.  Upstream filters/delineators may still be non-causal; this function
    makes no claim about them.
    """

    if not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")
    if variability_window_beats < 2:
        raise ValueError("variability_window_beats must be at least 2")
    if not 0 < minimum_valid_fraction <= 1:
        raise ValueError("minimum_valid_fraction must be in (0, 1]")
    if variability_window_beats != 30:
        stv_status = "not_computed_requires_exact_30_beat_window"
    else:
        stv_status = "computed_only_for_30_complete_consecutive_qt_values"

    frame = _validate_input_frame(pqrst_features)
    beat_features = _build_beat_features(frame, sampling_rate_hz)
    minimum_valid_beats = int(
        np.ceil(variability_window_beats * minimum_valid_fraction)
    )
    rolling_features = _build_trailing_features(
        beat_features,
        window_beats=variability_window_beats,
        minimum_valid_beats=minimum_valid_beats,
    )

    parameters: dict[str, Any] = {
        "branch_name": "conduction",
        "schema_version": "conduction_timing_v1",
        "landmark_method": str(landmark_method),
        "sampling_rate_hz": float(sampling_rate_hz),
        "variability_window_beats": int(variability_window_beats),
        "minimum_valid_fraction": float(minimum_valid_fraction),
        "minimum_valid_beats": int(minimum_valid_beats),
        "window_alignment": "causal_trailing_current_and_previous_beats",
        "beat_timestamp": "r_peak_time_s_identifies_beat",
        "feature_availability": (
            "per_feature_support_end_sample_records_when_each_measurement_is_known"
        ),
        "qt_rr_pairing": "qt_of_current_beat_with_preceding_rr_ending_at_current_r",
        "qtc_formulas": ["bazett", "fridericia", "framingham"],
        "qtvi_variants": ["heart_rate_denominator", "rr_denominator"],
        "qtvi_window_warning": (
            "30-beat QTVI is an exploratory short-window calculation, not a "
            "replication of studies using substantially longer stationary records"
        ),
        "stv30_status": stv_status,
        "aggregation_causal_given_finalized_input_landmarks": True,
        "upstream_landmark_generation_causality_asserted": False,
        "threshold_calibration_applied": False,
        "seizure_classifier_applied": False,
        "clinical_thresholds_applied": False,
    }
    diagnostics: dict[str, Any] = {
        "interval_level_manual_validation": "pending",
        "target_seizure_validation": "not_run",
        "measurement_gate": "prototype_not_pipeline_enabled",
        "current_project_landmark_limit": (
            "P-onset and T-offset timing-error tails can exceed expected true "
            "beat-to-beat interval variation; variability features are exploratory"
        ),
        "rolling_rows_with_qrs_variability": int(
            rolling_features[
                f"qrs_duration_variability{variability_window_beats}_defined"
            ].sum()
        ),
        "rolling_rows_with_qt_variability": int(
            rolling_features[
                f"qt_interval_variability{variability_window_beats}_defined"
            ].sum()
        ),
    }
    return ConductionTimingResult(
        beat_features=beat_features,
        rolling_features=rolling_features,
        parameters=parameters,
        diagnostics=diagnostics,
    )


def _validate_input_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("pqrst_features must be a pandas DataFrame")
    required = {"beat_index", "patient_id", "lead_name", *_SAMPLE_COLUMNS.values()}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"pqrst_features missing columns: {missing}")
    output = frame.copy().reset_index(drop=True)
    beat_index = pd.to_numeric(output["beat_index"], errors="coerce").to_numpy(float)
    if not np.all(np.isfinite(beat_index)):
        raise ValueError("beat_index must be finite")
    if not np.array_equal(beat_index.astype(np.int64), np.arange(output.shape[0])):
        raise ValueError("beat_index must be contiguous from zero and time ordered")
    peaks = pd.to_numeric(
        output[_SAMPLE_COLUMNS["r_peak"]], errors="coerce"
    ).to_numpy(float)
    if not np.all(np.isfinite(peaks)) or np.any(peaks < 0):
        raise ValueError("R-peak samples must be finite and non-negative")
    if peaks.size > 1 and np.any(np.diff(peaks) <= 0):
        raise ValueError("R-peak samples must be strictly increasing")
    if output["patient_id"].nunique(dropna=False) != 1:
        raise ValueError("one call must contain one patient_id")
    if output["lead_name"].nunique(dropna=False) != 1:
        raise ValueError("one call must contain one lead_name")
    if not str(output.iloc[0]["patient_id"]).strip():
        raise ValueError("patient_id must be non-empty")
    if not str(output.iloc[0]["lead_name"]).strip():
        raise ValueError("lead_name must be non-empty")
    return output


def _build_beat_features(frame: pd.DataFrame, sampling_rate_hz: float) -> pd.DataFrame:
    samples = {
        name: pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        for name, column in _SAMPLE_COLUMNS.items()
    }
    for name, values in samples.items():
        if np.any(np.isinf(values)):
            raise ValueError(f"{name} cannot contain infinity")
        finite = values[np.isfinite(values)]
        if np.any(finite < 0):
            raise ValueError(f"{name} contains a negative sample")

    p_duration = _duration_ms(samples["p_onset"], samples["p_offset"], sampling_rate_hz)
    pr_interval = _duration_ms(samples["p_onset"], samples["qrs_onset"], sampling_rate_hz)
    pr_segment = _duration_ms(samples["p_offset"], samples["qrs_onset"], sampling_rate_hz)
    qrs_duration = _duration_ms(
        samples["qrs_onset"], samples["qrs_offset"], sampling_rate_hz
    )
    qrs_onset_to_r = _duration_ms(
        samples["qrs_onset"], samples["r_peak"], sampling_rate_hz
    )
    r_to_qrs_offset = _duration_ms(
        samples["r_peak"], samples["qrs_offset"], sampling_rate_hz
    )
    qt_interval = _duration_ms(samples["qrs_onset"], samples["t_offset"], sampling_rate_hz)
    jt_interval = _duration_ms(samples["qrs_offset"], samples["t_offset"], sampling_rate_hz)
    t_peak_to_end = _duration_ms(samples["t_peak"], samples["t_offset"], sampling_rate_hz)

    rr_ms = np.full(frame.shape[0], np.nan, dtype=np.float64)
    if frame.shape[0] > 1:
        rr_ms[1:] = np.diff(samples["r_peak"]) * 1000.0 / sampling_rate_hz
    heart_rate_bpm = np.full(frame.shape[0], np.nan, dtype=np.float64)
    valid_rr = np.isfinite(rr_ms) & (rr_ms > 0)
    heart_rate_bpm[valid_rr] = 60000.0 / rr_ms[valid_rr]

    qtc_bazett = qt_correct_bazett(qt_interval, rr_ms)
    qtc_fridericia = qt_correct_fridericia(qt_interval, rr_ms)
    qtc_framingham = qt_correct_framingham(qt_interval, rr_ms)

    rows = pd.DataFrame(
        {
            "beat_index": frame["beat_index"].to_numpy(np.int64),
            "patient_id": frame["patient_id"].astype(str).to_numpy(),
            "lead_name": frame["lead_name"].astype(str).to_numpy(),
            "anchor_track": (
                frame["anchor_track"].astype(str).to_numpy()
                if "anchor_track" in frame
                else np.repeat("unspecified", frame.shape[0])
            ),
            "r_peak_sample_in_segment": samples["r_peak"].astype(np.int64),
            "r_peak_time_s": (
                pd.to_numeric(frame["r_peak_time_s"], errors="coerce").to_numpy(float)
                if "r_peak_time_s" in frame
                else samples["r_peak"] / sampling_rate_hz
            ),
            "pr_support_end_sample": _nullable_integer(samples["qrs_onset"]),
            "qrs_support_end_sample": _nullable_integer(samples["qrs_offset"]),
            "qt_support_end_sample": _nullable_integer(samples["t_offset"]),
            "preceding_rr_ms": rr_ms,
            "instantaneous_heart_rate_bpm": heart_rate_bpm,
            "p_duration_ms": p_duration,
            "pr_interval_ms": pr_interval,
            "pr_segment_ms": pr_segment,
            "qrs_duration_ms": qrs_duration,
            "qrs_onset_to_r_ms": qrs_onset_to_r,
            "r_to_qrs_offset_ms": r_to_qrs_offset,
            "qt_interval_ms": qt_interval,
            "jt_interval_ms": jt_interval,
            "t_peak_to_end_ms": t_peak_to_end,
            "qtc_bazett_ms": qtc_bazett,
            "qtc_fridericia_ms": qtc_fridericia,
            "qtc_framingham_ms": qtc_framingham,
            "preceding_rr_defined": valid_rr,
            "pr_interval_defined": np.isfinite(pr_interval),
            "qrs_duration_defined": np.isfinite(qrs_duration),
            "qt_interval_defined": np.isfinite(qt_interval),
            "jt_interval_defined": np.isfinite(jt_interval),
            "qtc_defined": np.isfinite(qt_interval) & valid_rr,
            "all_raw_wave_intervals_defined": (
                np.isfinite(pr_interval)
                & np.isfinite(qrs_duration)
                & np.isfinite(qt_interval)
                & np.isfinite(jt_interval)
            ),
            "seizure_probability_produced": False,
            "clinical_interpretation_produced": False,
        }
    )
    for column in (
        "pr_support_end_sample",
        "qrs_support_end_sample",
        "qt_support_end_sample",
    ):
        rows[column] = rows[column].astype("Int64")

    for name in _ROLLING_INTERVALS:
        values = rows[name].to_numpy(float)
        delta = np.full(values.size, np.nan, dtype=np.float64)
        if values.size > 1:
            adjacent = np.isfinite(values[1:]) & np.isfinite(values[:-1])
            difference = values[1:] - values[:-1]
            delta[1:][adjacent] = difference[adjacent]
        prefix = name.removesuffix("_ms")
        rows[f"delta_{prefix}_ms"] = delta
        rows[f"absolute_delta_{prefix}_ms"] = np.abs(delta)
    return rows


def _build_trailing_features(
    beats: pd.DataFrame,
    *,
    window_beats: int,
    minimum_valid_beats: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for end in range(beats.shape[0]):
        start = max(0, end - window_beats + 1)
        window = beats.iloc[start : end + 1]
        history_complete = window.shape[0] == window_beats
        output: dict[str, object] = {
            "beat_index": int(beats.iloc[end]["beat_index"]),
            "r_peak_time_s": float(beats.iloc[end]["r_peak_time_s"]),
            "window_start_beat_index": int(beats.iloc[start]["beat_index"]),
            f"history_beats{window_beats}": int(window.shape[0]),
            f"history{window_beats}_complete": bool(history_complete),
            "seizure_probability_produced": False,
            "clinical_interpretation_produced": False,
        }
        for column in _ROLLING_INTERVALS:
            prefix = column.removesuffix("_ms")
            values = window[column].to_numpy(float)
            valid_count = int(np.isfinite(values).sum())
            available = history_complete and valid_count >= minimum_valid_beats
            output[f"{prefix}_valid_beats{window_beats}"] = valid_count
            output[f"{prefix}_availability_fraction{window_beats}"] = (
                valid_count / window_beats
            )
            output[f"{prefix}_variability{window_beats}_defined"] = bool(available)
            if available:
                finite = values[np.isfinite(values)]
                output[f"{prefix}_mean{window_beats}_ms"] = float(np.mean(finite))
                output[f"{prefix}_median{window_beats}_ms"] = float(np.median(finite))
                output[f"{prefix}_sd{window_beats}_ms"] = float(np.std(finite, ddof=1))
                output[f"{prefix}_rmssd{window_beats}_ms"] = _adjacent_rmssd(values)
            else:
                output[f"{prefix}_mean{window_beats}_ms"] = np.nan
                output[f"{prefix}_median{window_beats}_ms"] = np.nan
                output[f"{prefix}_sd{window_beats}_ms"] = np.nan
                output[f"{prefix}_rmssd{window_beats}_ms"] = np.nan

        qt = window["qt_interval_ms"].to_numpy(float)
        rr = window["preceding_rr_ms"].to_numpy(float)
        joint = np.isfinite(qt) & np.isfinite(rr) & (qt > 0) & (rr > 0)
        joint_count = int(joint.sum())
        qtvi_available = history_complete and joint_count >= minimum_valid_beats
        qtvi_hr = (
            qt_variability_index_hr(qt[joint], rr[joint]) if qtvi_available else np.nan
        )
        qtvi_rr = (
            qt_variability_index_rr(qt[joint], rr[joint]) if qtvi_available else np.nan
        )
        output[f"qt_rr_joint_valid_beats{window_beats}"] = joint_count
        output[f"exploratory_qtvi_hr{window_beats}_defined"] = bool(np.isfinite(qtvi_hr))
        output[f"exploratory_qtvi_rr{window_beats}_defined"] = bool(np.isfinite(qtvi_rr))
        output[f"exploratory_qtvi_hr{window_beats}"] = qtvi_hr
        output[f"exploratory_qtvi_rr{window_beats}"] = qtvi_rr

        if window_beats == 30 and history_complete:
            stv = short_term_variability_30(qt)
        else:
            stv = np.nan
        output["qt_stv30_defined"] = bool(np.isfinite(stv))
        output["qt_stv30_ms"] = stv

        pr = window["pr_interval_ms"].to_numpy(float)
        heart_rate = window["instantaneous_heart_rate_bpm"].to_numpy(float)
        pr_hr = np.isfinite(pr) & np.isfinite(heart_rate)
        pr_hr_count = int(pr_hr.sum())
        regression_available = (
            history_complete
            and pr_hr_count >= minimum_valid_beats
            and np.ptp(heart_rate[pr_hr]) > 0
        )
        output[f"pr_hr_joint_valid_beats{window_beats}"] = pr_hr_count
        output[f"pr_hr_relation{window_beats}_defined"] = bool(regression_available)
        if regression_available:
            output[f"pr_vs_hr_slope{window_beats}_ms_per_bpm"] = float(
                np.polyfit(heart_rate[pr_hr], pr[pr_hr], 1)[0]
            )
            output[f"pr_hr_pearson_r{window_beats}"] = float(
                np.corrcoef(heart_rate[pr_hr], pr[pr_hr])[0, 1]
            )
        else:
            output[f"pr_vs_hr_slope{window_beats}_ms_per_bpm"] = np.nan
            output[f"pr_hr_pearson_r{window_beats}"] = np.nan
        rows.append(output)
    return pd.DataFrame(rows)


def _duration_ms(start: np.ndarray, end: np.ndarray, sampling_rate_hz: float) -> np.ndarray:
    output = np.full(start.size, np.nan, dtype=np.float64)
    valid = np.isfinite(start) & np.isfinite(end) & (end >= start)
    output[valid] = (end[valid] - start[valid]) * 1000.0 / sampling_rate_hz
    return output


def _nullable_integer(values: np.ndarray) -> pd.array:
    return pd.array(
        [int(np.rint(value)) if np.isfinite(value) else pd.NA for value in values],
        dtype="Int64",
    )


def _adjacent_rmssd(values: np.ndarray) -> float:
    if values.size < 2:
        return np.nan
    valid_pairs = np.isfinite(values[:-1]) & np.isfinite(values[1:])
    if not np.any(valid_pairs):
        return np.nan
    differences = values[1:][valid_pairs] - values[:-1][valid_pairs]
    return float(np.sqrt(np.mean(np.square(differences))))


def _validate_qt_rr(
    qt_ms: np.ndarray, rr_ms: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    qt = np.asarray(qt_ms, dtype=np.float64)
    rr = np.asarray(rr_ms, dtype=np.float64)
    if qt.ndim != 1 or rr.ndim != 1 or qt.size != rr.size:
        raise ValueError("qt_ms and rr_ms must be equal-length one-dimensional arrays")
    if np.any(np.isinf(qt)) or np.any(np.isinf(rr)):
        raise ValueError("qt_ms and rr_ms cannot contain infinity")
    if np.any(qt[np.isfinite(qt)] <= 0) or np.any(rr[np.isfinite(rr)] <= 0):
        raise ValueError("finite QT and RR values must be positive")
    return qt, rr


def _joint_finite_positive(
    qt_ms: np.ndarray, rr_ms: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    qt, rr = _validate_qt_rr(qt_ms, rr_ms)
    valid = np.isfinite(qt) & np.isfinite(rr)
    return qt[valid], rr[valid]


def _normalized_variance_log_ratio(numerator: np.ndarray, denominator: np.ndarray) -> float:
    numerator_variance = float(np.var(numerator, ddof=1))
    denominator_variance = float(np.var(denominator, ddof=1))
    numerator_mean = float(np.mean(numerator))
    denominator_mean = float(np.mean(denominator))
    if numerator_variance <= 0 or denominator_variance <= 0:
        return np.nan
    ratio = (numerator_variance / numerator_mean**2) / (
        denominator_variance / denominator_mean**2
    )
    return float(np.log10(ratio)) if ratio > 0 else np.nan
