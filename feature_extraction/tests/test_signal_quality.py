import numpy as np

from ecg_cascade.signal_quality import (
    bsqi_li2008_jaccard,
    extract_signal_quality_window,
    extract_trailing_signal_quality_windows,
    galeotti_baseline_wander,
    galeotti_powerline,
    galeotti_residual_noise,
    learn_menon_fourier_template,
    menon_fourier_score,
    pearson_kurtosis,
    population_skewness,
    qsqi_zhao2018_dice,
    rr_support_ho2024,
    template_correlation_orphanidou,
)


def _periodic_ecg(fs: int = 250, seconds: int = 10):
    count = fs * seconds
    peaks = np.arange(fs, count, fs, dtype=int)
    samples = np.arange(count)
    ecg = np.full(count, 2.0)
    for peak in peaks:
        ecg += 1000.0 * np.exp(-0.5 * ((samples - peak) / 5.0) ** 2)
    return ecg, peaks


def test_population_moments_use_biased_standardized_definitions():
    values = np.array([-2.0, -1.0, 0.0, 1.0, 4.0])
    centered = values - values.mean()
    scale = np.sqrt(np.mean(centered**2))

    assert np.isclose(population_skewness(values), np.mean((centered / scale) ** 3))
    assert np.isclose(pearson_kurtosis(values), np.mean((centered / scale) ** 4))
    assert np.isnan(population_skewness(np.ones(5)))
    assert np.isnan(pearson_kurtosis(np.ones(5)))


def test_clifford_spectral_features_respond_to_known_sine_bands():
    fs = 250
    time = np.arange(10 * fs) / fs
    qrs_band = np.sin(2.0 * np.pi * 10.0 * time)
    baseline_band = np.sin(2.0 * np.pi * 0.5 * time)

    qrs = extract_signal_quality_window(qrs_band, fs, uv_per_input_unit=1.0)
    baseline = extract_signal_quality_window(
        baseline_band, fs, uv_per_input_unit=1.0
    )

    assert qrs.values["psqi_clifford"] > 0.99
    assert qrs.values["bassqi_clifford"] > 0.99
    assert baseline.values["bassqi_clifford"] < 0.05


def test_flatline_and_redmond_dilated_rail_masks_have_known_durations():
    fs = 100
    ecg = np.linspace(-5.0, 5.0, 5 * fs)
    ecg[:fs] = 0.0
    ecg[3 * fs] = 10.0

    result = extract_signal_quality_window(
        ecg,
        fs,
        uv_per_input_unit=1000.0,
        rail_min_input_units=-10.0,
        rail_max_input_units=10.0,
    )

    assert np.isclose(result.values["flat_fraction"], 0.2)
    assert np.isclose(result.values["longest_flat_run_s"], 1.0)
    assert np.isclose(result.values["rail_fraction"], 201 / 500)
    assert np.isclose(result.values["longest_rail_run_s"], 2.01)


def test_langley_saturation_comparator_requires_more_than_200_ms_above_2mv():
    fs = 1000
    ecg = np.zeros(2 * fs)
    ecg[100:300] = 2.1  # exactly 200 ms: does not qualify
    ecg[500:701] = -2.1  # 201 ms and absolute amplitude above 2 mV

    result = extract_signal_quality_window(
        ecg,
        fs,
        uv_per_input_unit=1000.0,
    )

    assert np.isclose(result.values["saturation_fraction_langley2011"], 201 / 2000)
    assert np.isclose(
        result.values["longest_saturation_run_s_langley2011"], 0.201
    )


def test_bsqi_qsqi_and_ho_support_remain_distinct():
    primary = np.array([1000, 2000, 3000])
    # The 2500 event is not near a primary event.  It must not invalidate Ho's
    # endpoint-only RR support rule.
    secondary = np.array([1001, 1999, 2500, 3001])

    assert np.isclose(bsqi_li2008_jaccard(primary, secondary, 1000.0), 3 / 4)
    assert np.isclose(
        qsqi_zhao2018_dice(primary, secondary, 1000.0, tolerance_ms=75.0),
        6 / 7,
    )
    support = rr_support_ho2024(primary, secondary, 1000.0)
    assert support.primary_supported.tolist() == [True, True, True]
    assert support.rr_supported.tolist() == [True, True]


def test_orphanidou_template_correlation_is_one_for_identical_beats():
    ecg, peaks = _periodic_ecg()
    assert np.isclose(template_correlation_orphanidou(ecg, peaks), 1.0)


def test_galeotti_baseline_and_powerline_recover_controlled_components():
    fs = 250
    count = 10 * fs
    time = np.arange(count) / fs
    peaks = np.arange(fs, count, fs, dtype=int)
    onsets = peaks - 25
    ecg = 2.0 + 10.0 * np.sin(2.0 * np.pi * 50.0 * time)

    baseline = galeotti_baseline_wander(
        ecg, onsets, fs, mains_frequency_hz=50.0
    )
    mains = galeotti_powerline(
        ecg - baseline.estimate,
        peaks,
        fs,
        mains_frequency_hz=50.0,
    )

    assert np.isclose(baseline.rms, 2.0, atol=1e-10)
    assert np.isclose(mains.rms, 10.0 / np.sqrt(2.0), atol=1e-10)


def test_galeotti_residual_uses_external_groups_without_deleting_beats():
    ecg, peaks = _periodic_ecg()
    labels = np.zeros(peaks.size, dtype=int)
    labels[-1] = 1

    residual = galeotti_residual_noise(ecg, peaks, labels)

    assert residual.rms < 1e-12
    assert np.isclose(residual.excluded_fraction, 1 / peaks.size)
    assert residual.used_beat_count == peaks.size - 1
    assert residual.complete_beat_count == peaks.size


def test_menon_fifth_order_template_converges_and_scores_repeated_ecg():
    ecg, peaks = _periodic_ecg()
    learned = learn_menon_fourier_template(ecg, peaks, order=5)

    assert learned.converged
    assert learned.beats_used == 3
    assert learned.coefficients.size == 11
    score = menon_fourier_score(ecg, learned.template, peaks.size, 250.0)
    assert np.isfinite(score)
    assert score > 0


def test_combined_result_marks_missing_prerequisites_instead_of_imputing_zero():
    fs = 250
    time = np.arange(10 * fs) / fs
    result = extract_signal_quality_window(
        np.sin(2.0 * np.pi * 10.0 * time),
        fs,
        uv_per_input_unit=1.0,
    )

    for feature in (
        "rail_fraction",
        "bsqi_li2008_jaccard_150ms",
        "template_corr_orphanidou",
        "baseline_rms_galeotti",
        "mains_rms_galeotti",
        "residual_rms_galeotti",
    ):
        assert not result.available[feature]
        assert np.isnan(result.values[feature])
        assert result.reasons[feature] != "calculated"

    record = result.to_record()
    assert record["rail_fraction"] is None
    assert record["rail_fraction__available"] is False
    assert record["rail_fraction__reason"] == "adc_rail_metadata_missing"


def test_combined_result_computes_every_conditional_feature_with_metadata():
    fs = 250
    ecg, peaks = _periodic_ecg(fs)
    time = np.arange(ecg.size) / fs
    ecg = ecg + 5.0 * np.sin(2.0 * np.pi * 50.0 * time)
    onsets = peaks - 25
    secondary = peaks + 2
    labels = np.zeros(peaks.size, dtype=int)

    result = extract_signal_quality_window(
        ecg,
        fs,
        uv_per_input_unit=1.0,
        primary_peak_samples=peaks,
        secondary_peak_samples=secondary,
        qrs_onset_samples=onsets,
        other_lead_peak_samples=[peaks + 1],
        beat_group_labels=labels,
        rail_min_input_units=-5000.0,
        rail_max_input_units=5000.0,
        mains_frequency_hz=50.0,
        hf_threshold_uv=100.0,
    )

    expected = (
        "finite_fraction",
        "rail_fraction",
        "longest_rail_run_s",
        "flat_fraction",
        "longest_flat_run_s",
        "absmax_uv",
        "peak_to_peak_uv",
        "rms_uv",
        "hf_rms_uv",
        "hf_contaminated_fraction",
        "bassqi_clifford",
        "psqi_clifford",
        "skewness",
        "pearson_kurtosis",
        "bsqi_li2008_jaccard_150ms",
        "qsqi_zhao2018_dice_75ms",
        "rr_support_ho2024_150ms",
        "isqi_li2008_max_jaccard_150ms",
        "template_corr_orphanidou",
        "baseline_rms_galeotti",
        "mains_rms_galeotti",
        "residual_rms_galeotti",
        "dominant_beat_excluded_fraction",
        "menon_fourier_score",
    )
    for feature in expected:
        assert result.available[feature], (feature, result.reasons[feature])
        assert np.isfinite(result.values[feature])

    assert result.parameters["wavelet_features_enabled"] is False
    assert result.parameters["classifier_score_produced"] is False


def test_trailing_windows_do_not_read_samples_or_events_from_the_future():
    fs = 250
    time = np.arange(12 * fs) / fs
    first_ten_seconds = np.sin(2.0 * np.pi * 10.0 * time[: 10 * fs])
    appended_future = np.full(2 * fs, 1e6)
    peaks_first = np.arange(fs, 10 * fs, fs)

    short = extract_trailing_signal_quality_windows(
        first_ten_seconds,
        fs,
        uv_per_input_unit=1.0,
        primary_peak_samples=peaks_first,
        window_s=10.0,
        step_s=1.0,
    )
    long = extract_trailing_signal_quality_windows(
        np.concatenate((first_ten_seconds, appended_future)),
        fs,
        uv_per_input_unit=1.0,
        primary_peak_samples=np.append(peaks_first, 11 * fs),
        window_s=10.0,
        step_s=1.0,
    )

    assert short.window_end_samples_exclusive.tolist() == [10 * fs]
    assert long.window_end_samples_exclusive[:1].tolist() == [10 * fs]
    assert short.windows[0].values.keys() == long.windows[0].values.keys()
    for feature, value in short.windows[0].values.items():
        assert np.isclose(value, long.windows[0].values[feature], equal_nan=True)
