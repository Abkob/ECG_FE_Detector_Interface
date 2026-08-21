import numpy as np
import pandas as pd
import pytest

from ecg_cascade.conduction_information import extract_conduction_information


def _pqrst_fixture(beat_count: int = 12) -> pd.DataFrame:
    r = 1000 + np.arange(beat_count) * 1000
    qrs_duration = np.repeat(120.0, beat_count)
    qrs_onset = r - 70.0
    return pd.DataFrame(
        {
            "beat_index": np.arange(beat_count),
            "patient_id": "fixture",
            "lead_name": "II",
            "anchor_track": "expert",
            "r_peak_sample_in_segment": r,
            "r_peak_time_s": r / 1000.0,
            "p_onset_sample_in_segment": r - 220.0,
            "p_peak_sample_in_segment": r - 180.0,
            "p_offset_sample_in_segment": r - 130.0,
            "q_peak_sample_in_segment": r - 40.0,
            "qrs_onset_sample_in_segment": qrs_onset,
            "s_peak_sample_in_segment": r + 40.0,
            "qrs_offset_sample_in_segment": qrs_onset + qrs_duration,
            "t_peak_sample_in_segment": r + 250.0,
            "t_offset_sample_in_segment": r + 400.0,
        }
    )


def test_diab_peak_dynamics_match_published_definitions():
    result = extract_conduction_information(_pqrst_fixture(), 1000.0)
    first = result.beat_measurements.iloc[0]
    second = result.beat_measurements.iloc[1]

    assert first["diab_within_p_relative_to_r_ms"] == -180.0
    assert first["diab_within_q_relative_to_r_ms"] == -40.0
    assert first["diab_within_s_relative_to_r_ms"] == 40.0
    assert first["diab_within_t_relative_to_r_ms"] == 250.0
    assert np.isnan(first["diab_interbeat_r_to_r_ms"])
    for name in ("p", "q", "r", "s", "t"):
        assert second[f"diab_interbeat_{name}_to_{name}_ms"] == 1000.0
    assert second["diab_all_nine_peak_dynamics_defined"]
    assert second["diab_peak_dynamics_are_raw_not_five_beat_averaged"]


def test_five_beat_summary_exposes_robust_spread_and_trend():
    fixture = _pqrst_fixture()
    durations = np.asarray([100.0, 102.0, 104.0, 106.0, 300.0])
    fixture.loc[:4, "qrs_offset_sample_in_segment"] = (
        fixture.loc[:4, "qrs_onset_sample_in_segment"].to_numpy(float) + durations
    )
    result = extract_conduction_information(fixture, 1000.0)
    row = result.information_windows.query(
        "beat_index == 4 and measurement == 'qrs_duration'"
    ).iloc[0]

    assert row["history5_complete"]
    assert row["summary_defined"]
    assert np.isclose(row["mean_ms"], 142.4)
    assert row["median_ms"] == 104.0
    assert row["mad_ms"] == 2.0
    assert row["range_ms"] == 200.0
    assert row["theil_sen_slope_ms_per_beat"] == 2.0
    assert row["current_minus_prior_median_ms"] == 197.0


def test_nonoverlapping_five_beat_shift_and_nine_beat_qt_context():
    fixture = _pqrst_fixture(10)
    first = np.asarray([100.0, 101.0, 102.0, 103.0, 104.0])
    second = np.asarray([120.0, 121.0, 122.0, 123.0, 124.0])
    durations = np.concatenate([first, second])
    fixture["qrs_offset_sample_in_segment"] = (
        fixture["qrs_onset_sample_in_segment"].to_numpy(float) + durations
    )
    result = extract_conduction_information(fixture, 1000.0)
    row = result.information_windows.query(
        "beat_index == 9 and measurement == 'qrs_duration'"
    ).iloc[0]
    qt_row = result.qt_nine_beat_context.iloc[9]

    assert row["previous_window_comparison_defined"]
    assert row["median_shift_vs_previous_window_ms"] == 20.0
    assert row["absolute_median_shift_vs_previous_window_ms"] == 20.0
    assert qt_row["qt_rr_mean9_defined"]
    assert qt_row["qt_interval_mean9_ms"] == 470.0
    assert qt_row["preceding_rr_mean9_ms"] == 1000.0
    assert qt_row["qtc_fridericia_from_means9_ms"] == 470.0


def test_external_quality_eligibility_is_consumed_not_invented():
    fixture = _pqrst_fixture()
    fixture["quality_eligible"] = True
    fixture.loc[4, "quality_eligible"] = False
    fixture.loc[4, "qrs_offset_sample_in_segment"] = (
        fixture.loc[4, "qrs_onset_sample_in_segment"] + 500.0
    )
    result = extract_conduction_information(
        fixture,
        1000.0,
        eligibility_column="quality_eligible",
    )
    row = result.information_windows.query(
        "beat_index == 4 and measurement == 'qrs_duration'"
    ).iloc[0]

    assert row["eligible_beats5"] == 4
    assert row["valid_beats"] == 4
    assert row["mean_ms"] == 120.0
    assert not result.beat_measurements.loc[4, "information_window_eligible"]
    assert result.summary()["final_feature_matrix_eligible"] is False
    assert result.summary()["abnormality_label_produced"] is False


def test_window_output_is_causal_and_rejects_bad_eligibility_schema():
    fixture = _pqrst_fixture()
    baseline = extract_conduction_information(fixture, 1000.0)
    changed = fixture.copy()
    changed.loc[11, "t_offset_sample_in_segment"] += 100.0
    rerun = extract_conduction_information(changed, 1000.0)
    baseline_qt = baseline.information_windows.query(
        "measurement == 'qt_interval' and beat_index <= 10"
    ).reset_index(drop=True)
    rerun_qt = rerun.information_windows.query(
        "measurement == 'qt_interval' and beat_index <= 10"
    ).reset_index(drop=True)
    pd.testing.assert_series_equal(
        baseline_qt["median_ms"],
        rerun_qt["median_ms"],
    )

    fixture["bad_quality"] = "maybe"
    with pytest.raises(ValueError, match="boolean"):
        extract_conduction_information(
            fixture,
            1000.0,
            eligibility_column="bad_quality",
        )
