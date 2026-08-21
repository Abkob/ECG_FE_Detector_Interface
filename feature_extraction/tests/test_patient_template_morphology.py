import numpy as np

from ecg_cascade.patient_template_morphology import (
    derivative_dtw_distance,
    extract_patient_template_morphology,
)


def _synthetic_cycles(*, changed_last: bool = False, scaled_last: bool = False):
    fs = 250.0
    peaks = np.arange(250, 2751, 250, dtype=np.int64)
    ecg = np.zeros(3000, dtype=np.float64)
    axis = np.arange(-125, 125)
    beat = (
        0.12 * np.exp(-((axis + 55) / 12) ** 2)
        + 1.0 * np.exp(-(axis / 5) ** 2)
        - 0.25 * np.exp(-((axis - 9) / 6) ** 2)
        + 0.30 * np.exp(-((axis - 70) / 20) ** 2)
    )
    for index, peak in enumerate(peaks):
        current = beat.copy()
        if changed_last and index == len(peaks) - 2:
            current += -0.65 * np.exp(-((axis - 70) / 18) ** 2)
        if scaled_last and index == len(peaks) - 2:
            current *= 2.0
        ecg[peak - 125 : peak + 125] += current
    return ecg, peaks, fs


def test_identical_whole_beats_match_fixed_initial_template():
    ecg, peaks, fs = _synthetic_cycles()
    result = extract_patient_template_morphology(
        ecg,
        peaks,
        fs,
        anchor_track="expert",
        patient_id="fixture",
        lead_name="II",
        calibration_beats=4,
    )

    evaluation = result.features[result.features["evaluation_eligible"]]
    assert result.beat_waveforms.shape[1] == 192
    assert np.allclose(evaluation["template_correlation"], 1.0)
    assert np.allclose(evaluation["template_normalized_rmse"], 0.0, atol=1e-12)
    assert np.allclose(evaluation["template_derivative_dtw_cost"], 0.0, atol=1e-12)


def test_shape_change_increases_template_distances():
    baseline_ecg, peaks, fs = _synthetic_cycles()
    changed_ecg, _, _ = _synthetic_cycles(changed_last=True)
    baseline = extract_patient_template_morphology(
        baseline_ecg,
        peaks,
        fs,
        anchor_track="expert",
        patient_id="fixture",
        lead_name="II",
        calibration_beats=4,
    )
    changed = extract_patient_template_morphology(
        changed_ecg,
        peaks,
        fs,
        anchor_track="expert",
        patient_id="fixture",
        lead_name="II",
        calibration_beats=4,
    )

    target = len(changed.features) - 1
    assert (
        changed.features.loc[target, "template_normalized_rmse"]
        > baseline.features.loc[target, "template_normalized_rmse"]
    )
    assert (
        changed.features.loc[target, "template_derivative_dtw_cost"]
        > baseline.features.loc[target, "template_derivative_dtw_cost"]
    )


def test_amplitude_change_preserves_shape_score_and_changes_raw_residual():
    ecg, peaks, fs = _synthetic_cycles(scaled_last=True)
    result = extract_patient_template_morphology(
        ecg,
        peaks,
        fs,
        anchor_track="expert",
        patient_id="fixture",
        lead_name="II",
        calibration_beats=4,
    )

    target = result.features.iloc[-1]
    assert target["template_correlation"] > 0.999
    assert target["template_normalized_rmse"] < 1e-12
    assert target["template_raw_residual_rms"] > 0


def test_derivative_dtw_reports_warp_for_shifted_shape():
    waveform = np.exp(-((np.arange(100) - 50) / 5) ** 2)
    shifted = np.roll(waveform, 3)
    identical_cost, identical_warp = derivative_dtw_distance(waveform, waveform)
    shifted_cost, shifted_warp = derivative_dtw_distance(waveform, shifted)

    assert identical_cost == 0.0
    assert identical_warp == 0.0
    assert shifted_cost > 0.0
    assert shifted_warp > 0.0


def test_calibration_can_retain_two_recurring_template_shapes():
    ecg, peaks, fs = _synthetic_cycles()
    axis = np.arange(-125, 125)
    alternate = -0.9 * np.exp(-(axis / 7) ** 2)
    # Peak indices 2 and 4 become different calibration morphologies while
    # indices 1..4 are the four complete calibration cycles.
    for peak_index in (2, 4):
        peak = int(peaks[peak_index])
        ecg[peak - 125 : peak + 125] = alternate
    result = extract_patient_template_morphology(
        ecg,
        peaks,
        fs,
        anchor_track="expert",
        patient_id="fixture",
        lead_name="II",
        calibration_beats=4,
        max_templates=3,
        new_template_correlation_threshold=0.90,
    )

    assert result.raw_template_bank.shape[0] == 2
    assert sorted(result.template_member_counts.tolist()) == [2, 2]
