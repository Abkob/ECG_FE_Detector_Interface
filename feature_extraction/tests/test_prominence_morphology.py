import numpy as np

from ecg_cascade.prominence_morphology import build_pqrst_feature_table


def test_interpretable_pqrst_features_preserve_signed_amplitude_and_duration():
    fs = 1000.0
    signal = np.zeros(2000)
    signal[900] = 0.2
    signal[1000] = -1.0
    signal[1020] = 0.4
    signal[1200] = -0.3
    peaks = np.asarray([1000])
    landmarks = {
        "p_onset": np.asarray([850.0]),
        "p_peak": np.asarray([900.0]),
        "p_offset": np.asarray([930.0]),
        "qrs_onset": np.asarray([970.0]),
        "q_peak": np.asarray([990.0]),
        "r_peak": np.asarray([1000.0]),
        "s_peak": np.asarray([1020.0]),
        "qrs_offset": np.asarray([1040.0]),
        "t_onset": np.asarray([1100.0]),
        "t_peak": np.asarray([1200.0]),
        "t_offset": np.asarray([1300.0]),
    }

    frame = build_pqrst_feature_table(
        signal,
        peaks,
        landmarks,
        fs,
        anchor_track="expert",
        patient_id="fixture",
        lead_name="II",
    )
    row = frame.iloc[0]
    assert row["p_duration_ms"] == 80.0
    assert row["pr_interval_ms"] == 120.0
    assert row["qrs_duration_ms"] == 70.0
    assert row["qt_interval_ms"] == 330.0
    assert row["qrs_polarity"] == "negative"
    assert row["r_anchor_amplitude_from_baseline"] == -1.0
    assert row["t_peak_amplitude_from_baseline"] == -0.3
    assert row["complete_pqrst"]
    assert row["physiological_order_valid"]
    assert row["complete_qs_peaks"]
    assert row["q_s_peak_order_valid"]
    assert row["q_peak_sample_in_segment"] == 990
    assert row["s_peak_sample_in_segment"] == 1020


def test_missing_t_onset_is_retained_as_missing_with_confidence_flags():
    signal = np.zeros(2000)
    peaks = np.asarray([1000])
    landmarks = {
        "p_onset": np.asarray([850.0]),
        "p_peak": np.asarray([900.0]),
        "p_offset": np.asarray([930.0]),
        "qrs_onset": np.asarray([970.0]),
        "r_peak": np.asarray([1000.0]),
        "qrs_offset": np.asarray([1040.0]),
        "t_onset": np.asarray([np.nan]),
        "t_peak": np.asarray([1200.0]),
        "t_offset": np.asarray([1300.0]),
    }

    frame = build_pqrst_feature_table(
        signal,
        peaks,
        landmarks,
        1000.0,
        anchor_track="expert",
        patient_id="fixture",
        lead_name="II",
    )
    row = frame.iloc[0]
    assert not row["complete_t"]
    assert not row["complete_pqrst"]
    assert not row["physiological_order_valid"]
    assert np.isnan(row["st_interval_ms"])
    assert np.isclose(row["landmark_availability_fraction"], 7 / 8)
