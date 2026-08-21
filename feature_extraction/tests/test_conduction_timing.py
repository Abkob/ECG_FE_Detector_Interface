import numpy as np
import pandas as pd
import pytest

from ecg_cascade.conduction_timing import (
    extract_conduction_timing,
    qt_correct_bazett,
    qt_correct_framingham,
    qt_correct_fridericia,
    qt_variability_index_hr,
    qt_variability_index_rr,
    short_term_variability_30,
)


def _pqrst_fixture(beat_count: int = 35) -> pd.DataFrame:
    r = 1000 + np.arange(beat_count) * 1000
    return pd.DataFrame(
        {
            "beat_index": np.arange(beat_count),
            "patient_id": "fixture",
            "lead_name": "II",
            "anchor_track": "expert",
            "r_peak_sample_in_segment": r,
            "r_peak_time_s": r / 1000.0,
            "p_onset_sample_in_segment": r - 200,
            "p_offset_sample_in_segment": r - 120,
            "qrs_onset_sample_in_segment": r - 80,
            "qrs_offset_sample_in_segment": r + 40,
            "t_peak_sample_in_segment": r + 220,
            "t_offset_sample_in_segment": r + 400,
        }
    )


def test_qt_correction_formulas_keep_units_explicit():
    qt = np.asarray([400.0])
    rr = np.asarray([800.0])
    assert np.isclose(qt_correct_bazett(qt, rr)[0], 447.2135955)
    assert np.isclose(qt_correct_fridericia(qt, rr)[0], 430.886938)
    assert np.isclose(qt_correct_framingham(qt, rr)[0], 430.8)


def test_beatwise_conduction_intervals_are_hand_computable():
    result = extract_conduction_timing(_pqrst_fixture(), 1000.0)
    first = result.beat_features.iloc[0]
    row = result.beat_features.iloc[1]
    assert first["qt_interval_defined"]
    assert not first["qtc_defined"]
    assert row["preceding_rr_ms"] == 1000.0
    assert row["p_duration_ms"] == 80.0
    assert row["pr_interval_ms"] == 120.0
    assert row["pr_segment_ms"] == 40.0
    assert row["qrs_duration_ms"] == 120.0
    assert row["qrs_onset_to_r_ms"] == 80.0
    assert row["r_to_qrs_offset_ms"] == 40.0
    assert row["qt_interval_ms"] == 480.0
    assert row["jt_interval_ms"] == 360.0
    assert row["qtc_fridericia_ms"] == 480.0
    assert row["pr_support_end_sample"] == 1920
    assert row["qt_support_end_sample"] == 2400


def test_missing_t_offset_remains_missing_instead_of_becoming_zero():
    fixture = _pqrst_fixture()
    fixture.loc[5, "t_offset_sample_in_segment"] = pd.NA
    result = extract_conduction_timing(fixture, 1000.0)
    row = result.beat_features.iloc[5]
    assert np.isnan(row["qt_interval_ms"])
    assert np.isnan(row["jt_interval_ms"])
    assert np.isnan(row["qtc_bazett_ms"])
    assert not row["qt_interval_defined"]
    assert pd.isna(row["qt_support_end_sample"])


def test_trailing_window_is_causal_and_requires_history():
    fixture = _pqrst_fixture()
    baseline = extract_conduction_timing(fixture, 1000.0)
    changed = fixture.copy()
    changed.loc[34, "qrs_offset_sample_in_segment"] += 20
    rerun = extract_conduction_timing(changed, 1000.0)
    column = "qrs_duration_mean30_ms"
    assert np.isnan(baseline.rolling_features.loc[28, column])
    assert baseline.rolling_features.loc[29, column] == 120.0
    pd.testing.assert_series_equal(
        baseline.rolling_features.loc[:33, column],
        rerun.rolling_features.loc[:33, column],
    )
    assert rerun.rolling_features.loc[34, column] > baseline.rolling_features.loc[34, column]


def test_published_stv30_equation_and_qtvi_zero_variance_edges():
    intervals = np.arange(400.0, 430.0)
    expected = 29.0 / (30.0 * np.sqrt(2.0))
    assert np.isclose(short_term_variability_30(intervals), expected)
    assert np.isnan(qt_variability_index_rr(np.repeat(400.0, 4), np.arange(900.0, 940.0, 10)))
    assert np.isnan(qt_variability_index_hr(np.arange(400.0, 440.0, 10), np.repeat(1000.0, 4)))

    result = extract_conduction_timing(_pqrst_fixture(), 1000.0)
    row = result.rolling_features.iloc[29]
    assert not row["exploratory_qtvi_hr30_defined"]
    assert not row["exploratory_qtvi_rr30_defined"]


def test_invalid_schema_and_nonpositive_qt_inputs_are_rejected():
    with pytest.raises(ValueError, match="missing columns"):
        extract_conduction_timing(pd.DataFrame({"beat_index": [0]}), 1000.0)
    with pytest.raises(ValueError, match="must be positive"):
        qt_correct_bazett(np.asarray([0.0]), np.asarray([1000.0]))
    fixture = _pqrst_fixture()
    fixture["p_onset_sample_in_segment"] = fixture[
        "p_onset_sample_in_segment"
    ].astype(float)
    fixture.loc[2, "p_onset_sample_in_segment"] = np.inf
    with pytest.raises(ValueError, match="cannot contain infinity"):
        extract_conduction_timing(fixture, 1000.0)


def test_pr_heart_rate_relation_uses_only_trailing_jointly_valid_beats():
    fixture = _pqrst_fixture(30)
    rr_ms = 800 + 10 * np.arange(30)
    r = np.cumsum(rr_ms).astype(float)
    heart_rate = 60000.0 / rr_ms
    pr_ms = 100.0 + 0.5 * heart_rate
    fixture["r_peak_sample_in_segment"] = r
    fixture["r_peak_time_s"] = r / 1000.0
    fixture["qrs_onset_sample_in_segment"] = r - 80.0
    fixture["qrs_offset_sample_in_segment"] = r + 40.0
    fixture["p_onset_sample_in_segment"] = r - 80.0 - pr_ms
    fixture["p_offset_sample_in_segment"] = r - 120.0
    fixture["t_peak_sample_in_segment"] = r + 220.0
    fixture["t_offset_sample_in_segment"] = r + 400.0

    result = extract_conduction_timing(fixture, 1000.0)
    row = result.rolling_features.iloc[-1]
    assert row["pr_hr_relation30_defined"]
    assert np.isclose(row["pr_vs_hr_slope30_ms_per_bpm"], 0.5)
