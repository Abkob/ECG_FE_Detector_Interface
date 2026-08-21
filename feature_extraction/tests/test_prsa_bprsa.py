import numpy as np
import pytest

from ecg_cascade.prsa_bprsa import (
    calculate_prsa_bprsa_feature_arrays,
    extract_prsa_bprsa_features,
)


def _manual_curve(driver: np.ndarray, target: np.ndarray, half: int) -> tuple[np.ndarray, int]:
    anchors = np.flatnonzero(driver[1:] < driver[:-1]) + 1
    anchors = anchors[(anchors >= half) & (anchors + half < target.size)]
    segments = np.stack([target[index - half : index + half + 1] for index in anchors])
    return np.mean(segments, axis=0), int(anchors.size)


def test_published_anchor_rule_and_slope_equations_are_exact():
    count = 80
    times = np.arange(count, dtype=float) / 4.0
    phase = np.arange(count, dtype=float)
    rr = 800.0 + 35.0 * np.sin(2.0 * np.pi * phase / 11.0)
    ramp = 1.2 + 0.3 * np.sin(2.0 * np.pi * phase / 17.0 + 0.4)

    result = calculate_prsa_bprsa_feature_arrays(
        rr,
        ramp,
        end_time_s=times,
        window_size=80,
        resample_hz=4.0,
        half_window_samples=4,
    )
    row = result.features.iloc[-1]
    expected_prsa, prsa_count = _manual_curve(rr, rr, 4)
    expected_bprsa, bprsa_count = _manual_curve(ramp, rr, 4)

    assert not result.features.iloc[:-1]["feature_defined"].any()
    assert row["feature_defined"]
    assert row["numerical_quality_pass"]
    assert row["resampled_rr_nonpositive_count"] == 0
    assert row["mean_rr80_ms"] == pytest.approx(np.mean(rr))
    assert row["sdnn80_ms"] == pytest.approx(np.std(rr, ddof=0))
    assert row["prsa_anchor_count"] == prsa_count
    assert row["bprsa_anchor_count"] == bprsa_count
    assert result.prsa_curves_ms[-1] == pytest.approx(expected_prsa)
    assert result.bprsa_curves_ms[-1] == pytest.approx(expected_bprsa)
    assert row["prsa_s_rr_ms_per_sample"] == pytest.approx(
        (expected_prsa[5] - expected_prsa[3]) / 2.0
    )
    assert row["prsa_delta_rr_ms_per_sample"] == pytest.approx(
        (expected_prsa[-1] - expected_prsa[0]) / expected_prsa.size
    )
    assert row["bprsa_s_r_ms_per_sample"] == pytest.approx(
        (expected_bprsa[5] - expected_bprsa[3]) / 2.0
    )
    assert row["bprsa_delta_r_ms_per_sample"] == pytest.approx(
        (expected_bprsa[-1] - expected_bprsa[0]) / expected_bprsa.size
    )


def test_appending_future_beats_cannot_change_existing_windows():
    phase = np.arange(140, dtype=float)
    rr = 780.0 + 30.0 * np.sin(phase / 3.1) + 8.0 * np.cos(phase / 7.2)
    ramp = 1.0 + 0.2 * np.cos(phase / 4.5)
    times = np.cumsum(rr) / 1000.0

    prefix = calculate_prsa_bprsa_feature_arrays(
        rr[:110], ramp[:110], end_time_s=times[:110]
    )
    complete = calculate_prsa_bprsa_feature_arrays(rr, ramp, end_time_s=times)
    columns = [
        "mean_rr80_ms",
        "sdnn80_ms",
        "prsa_s_rr_ms_per_sample",
        "prsa_delta_rr_ms_per_sample",
        "bprsa_s_r_ms_per_sample",
        "bprsa_delta_r_ms_per_sample",
    ]

    assert complete.features.loc[:109, columns].to_numpy() == pytest.approx(
        prefix.features[columns].to_numpy(), nan_ok=True
    )
    assert complete.prsa_curves_ms[:110] == pytest.approx(
        prefix.prsa_curves_ms, nan_ok=True
    )
    assert complete.bprsa_curves_ms[:110] == pytest.approx(
        prefix.bprsa_curves_ms, nan_ok=True
    )


def test_bprsa_is_lead_polarity_sensitive_but_prsa_is_not():
    phase = np.arange(100, dtype=float)
    rr = 810.0 + 45.0 * np.sin(phase / 3.3)
    ramp = np.sin(phase / 4.7) + 0.25 * np.sin(phase / 1.8)
    times = np.cumsum(rr) / 1000.0

    positive = calculate_prsa_bprsa_feature_arrays(rr, ramp, end_time_s=times)
    inverted = calculate_prsa_bprsa_feature_arrays(rr, -ramp, end_time_s=times)

    assert positive.prsa_curves_ms == pytest.approx(
        inverted.prsa_curves_ms, nan_ok=True
    )
    assert not np.allclose(
        positive.bprsa_curves_ms[-1], inverted.bprsa_curves_ms[-1]
    )


def test_reliability_is_separate_from_mathematical_definition():
    phase = np.arange(80, dtype=float)
    rr = 800.0 + 25.0 * np.sin(phase / 2.9)
    ramp = np.cos(phase / 3.7)
    times = np.cumsum(rr) / 1000.0
    support = np.ones(80, dtype=bool)
    support[12] = False

    result = calculate_prsa_bprsa_feature_arrays(
        rr,
        ramp,
        end_time_s=times,
        rr_supported=support,
    )
    row = result.features.iloc[-1]

    assert row["feature_defined"]
    assert not row["feature_reliable"]
    assert row["rr_reliability_coverage"] == pytest.approx(79 / 80)


def test_ecg_wrapper_uses_requested_orientation_and_explicit_peak_amplitudes():
    fs = 100.0
    peaks = np.arange(10, 911, 10, dtype=int)
    signal = np.zeros(1000, dtype=float)
    signal[peaks] = np.linspace(0.2, 1.2, peaks.size)

    original = extract_prsa_bprsa_features(signal, peaks, fs, orientation="original")
    inverted = extract_prsa_bprsa_features(signal, peaks, fs, orientation="inverted")

    assert original.features["r_peak_amplitude"].to_numpy() == pytest.approx(
        signal[peaks[1:]]
    )
    assert inverted.features["r_peak_amplitude"].to_numpy() == pytest.approx(
        -signal[peaks[1:]]
    )
    assert original.parameters["calibration_applied"] is False
    assert original.parameters["classifier_applied"] is False


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError, match="same shape"):
        calculate_prsa_bprsa_feature_arrays(np.ones(80), np.ones(79))
    with pytest.raises(ValueError, match="strictly positive"):
        calculate_prsa_bprsa_feature_arrays(np.r_[np.ones(79), 0.0], np.ones(80))
