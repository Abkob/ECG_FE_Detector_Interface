import numpy as np

from ecg_cascade.peaks import compare_peak_sequences
from ecg_cascade.rr_hrv import (
    absolute_least_squares_slope,
    calculate_jeppesen_feature_arrays,
    causal_median_filter,
    extract_rr_hrv_features,
    poincare_axes,
)


def test_causal_median_filter_never_uses_future_values():
    values = np.array([1.0, 100.0, 2.0, 3.0])
    filtered = causal_median_filter(values, width=3)
    assert filtered.tolist() == [1.0, 50.5, 2.0, 3.0]


def test_poincare_constant_rr_has_zero_axes():
    sd1, sd2 = poincare_axes(np.full(100, 800.0))
    assert np.isclose(sd1, 0.0, atol=1e-12)
    assert np.isclose(sd2, 0.0, atol=1e-12)


def test_absolute_slope_uses_elapsed_time_not_beat_number():
    time_s = np.array([0.0, 1.0, 2.0, 3.0])
    heart_rate = np.array([60.0, 62.0, 64.0, 66.0])
    assert np.isclose(absolute_least_squares_slope(time_s, heart_rate), 2.0)


def test_jeppesen_features_start_only_after_100_rr_intervals():
    fs = 1000.0
    rr_ms = 800.0 + 20.0 * np.sin(np.linspace(0, 8 * np.pi, 130))
    peak_times_ms = np.concatenate([[0.0], np.cumsum(rr_ms)])
    peaks = np.rint(peak_times_ms * fs / 1000.0).astype(int)
    agreement = compare_peak_sequences(
        peaks,
        peaks,
        sampling_rate_hz=fs,
    )

    frame = extract_rr_hrv_features(
        peaks,
        sampling_rate_hz=fs,
        segment_start_s=10.0,
        agreement=agreement,
    )

    assert len(frame) == 130
    assert frame.loc[:98, "j1_csi_x_slope"].isna().all()
    assert frame.loc[:98, "j2_modcsi_filtered_x_slope"].isna().all()
    assert frame.loc[99:, "feature_defined"].all()
    assert not frame["window_has_detector_disagreement"].any()


def test_vectorized_feature_arrays_match_audited_peak_implementation():
    fs = 360.0
    rr_ms = 780.0 + 45.0 * np.sin(np.linspace(0, 12 * np.pi, 240))
    rr_samples = np.maximum(1, np.rint(rr_ms * fs / 1000.0)).astype(int)
    peaks = np.concatenate([[0], np.cumsum(rr_samples)])
    agreement = compare_peak_sequences(peaks, peaks, sampling_rate_hz=fs)
    audited = extract_rr_hrv_features(
        peaks,
        sampling_rate_hz=fs,
        segment_start_s=0.0,
        agreement=agreement,
    )
    vectorized = calculate_jeppesen_feature_arrays(
        np.diff(peaks) * 1000.0 / fs,
        end_time_s=peaks[1:] / fs,
    )
    columns = [
        "sd1_raw_ms",
        "sd2_raw_ms",
        "csi100",
        "sd1_filtered_ms",
        "sd2_filtered_ms",
        "modcsi100_filtered_ms",
        "slope100_bpm_per_s",
        "j1_csi_x_slope",
        "j2_modcsi_filtered_x_slope",
    ]
    for column in columns:
        assert np.allclose(
            audited[column],
            vectorized[column],
            equal_nan=True,
            rtol=1e-8,
            atol=1e-9,
        )
