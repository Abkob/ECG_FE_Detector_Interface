import numpy as np
import pandas as pd

from ecg_cascade.fusion import (
    build_detector_association_table,
    build_fusion_ready_table,
)
from ecg_cascade.morphology import (
    calibrate_patient_asymmetric_varon_window,
    calibrate_patient_symmetric_varon_width,
    extract_patient_average_varon_morphology,
    extract_patient_asymmetric_varon_morphology,
    extract_patient_required_width_varon_morphology,
    extract_varon_morphology,
)
from ecg_cascade.peaks import compare_peak_sequences


def _repeated_waveform_ecg() -> tuple[np.ndarray, np.ndarray, float]:
    fs = 250.0
    peaks = np.arange(250, 2251, 250, dtype=np.int64)
    ecg = np.zeros(2500, dtype=np.float64)
    local = np.array([-0.2, 0.1, 1.0, -0.4, -0.1])
    for peak in peaks:
        ecg[peak - 2 : peak + 3] = local
    return ecg, peaks, fs


def _event_context(peaks: np.ndarray, *, unsupported_index: int | None = None):
    supported = np.ones(peaks.size, dtype=bool)
    if unsupported_index is not None:
        supported[unsupported_index] = False
    return pd.DataFrame(
        {
            "primary_timestamp_sample": peaks,
            "qrs_supported": supported,
            "comparator_match_count": supported.astype(int),
            "comparator_offset_ms": np.where(supported, 4.0, np.nan),
            "zhai_absolute_correlation": np.where(supported, 0.95, np.nan),
        }
    )


def test_varon_identical_beats_have_one_dominant_eigenvalue():
    ecg, peaks, fs = _repeated_waveform_ecg()
    result = extract_varon_morphology(
        ecg,
        peaks,
        fs,
        anchor_track="synthetic",
        event_context=_event_context(peaks),
    )

    first = result.features.iloc[0]
    waveform = result.beat_waveforms[0]
    expected_lambda1 = 5.0 * float(waveform @ waveform)
    assert np.isclose(first["lambda1"], expected_lambda1)
    assert np.allclose(
        first[["lambda2", "lambda3", "lambda4", "lambda5"]].astype(float),
        0.0,
        atol=1e-12,
    )
    assert first["published_varon_core_defined"]
    assert first["support_context_pass"]


def test_detector_disagreement_is_context_and_does_not_erase_morphology():
    ecg, peaks, fs = _repeated_waveform_ecg()
    result = extract_varon_morphology(
        ecg,
        peaks,
        fs,
        anchor_track="synthetic",
        event_context=_event_context(peaks, unsupported_index=2),
    )

    first = result.features.iloc[0]
    assert first["published_varon_core_defined"]
    assert np.isfinite(first["lambda1"])
    assert np.isclose(first["detector_support_fraction"], 0.8)
    assert not first["support_context_pass"]
    assert first["window_has_detector_disagreement"]


def test_varon_gram_eigenvalues_are_invariant_to_global_signal_inversion():
    ecg, peaks, fs = _repeated_waveform_ecg()
    original = extract_varon_morphology(
        ecg,
        peaks,
        fs,
        anchor_track="synthetic",
        orientation="original",
        event_context=_event_context(peaks),
    )
    inverted = extract_varon_morphology(
        ecg,
        peaks,
        fs,
        anchor_track="synthetic",
        orientation="inverted",
        event_context=_event_context(peaks),
    )
    columns = [f"lambda{index}" for index in range(1, 6)]
    assert np.allclose(original.features[columns], inverted.features[columns])


def test_patient_average_varon_changes_only_symmetric_capture_width():
    ecg, peaks, fs = _repeated_waveform_ecg()
    result = extract_patient_average_varon_morphology(
        ecg,
        peaks,
        fs,
        np.asarray([136.0, 140.0, 144.0]),
        anchor_track="expert",
        patient_id="fixture_patient",
        lead_name="II",
        event_context=_event_context(peaks),
    )

    assert result.parameters["calibration_mean_qrs_duration_ms"] == 140.0
    assert result.parameters["pre_r_ms_requested"] == 70.0
    assert result.parameters["post_r_ms_requested"] == 70.0
    assert result.parameters["waveform_sample_count"] == 35
    assert result.parameters["fixed_120ms_control_modified"] is False
    assert list(result.features.filter(regex=r"^lambda[1-5]$").columns) == [
        "lambda1",
        "lambda2",
        "lambda3",
        "lambda4",
        "lambda5",
    ]


def test_patient_average_varon_rejects_missing_calibration():
    ecg, peaks, fs = _repeated_waveform_ecg()
    with np.testing.assert_raises(ValueError):
        extract_patient_average_varon_morphology(
            ecg,
            peaks,
            fs,
            np.asarray([]),
            anchor_track="expert",
            patient_id="fixture_patient",
            lead_name="II",
        )


def test_required_symmetric_width_accounts_for_off_center_r_anchor():
    width, required = calibrate_patient_symmetric_varon_width(
        np.asarray([30.0, 45.0]),
        np.asarray([70.0, 50.0]),
        statistic="mean",
    )

    assert required.tolist() == [140.0, 100.0]
    assert width == 120.0


def test_required_width_varon_uses_one_calibrated_symmetric_parameter():
    ecg, peaks, fs = _repeated_waveform_ecg()
    result = extract_patient_required_width_varon_morphology(
        ecg,
        peaks,
        fs,
        np.asarray([30.0, 45.0]),
        np.asarray([70.0, 50.0]),
        anchor_track="expert",
        patient_id="fixture_patient",
        lead_name="II",
        statistic="max",
    )

    assert result.parameters["calibrated_symmetric_width_ms"] == 140.0
    assert result.parameters["pre_r_ms_requested"] == 70.0
    assert result.parameters["post_r_ms_requested"] == 70.0
    assert result.parameters["calibration_qrs_count"] == 2


def test_asymmetric_calibration_keeps_pre_and_post_distributions_separate():
    pre_ms, post_ms = calibrate_patient_asymmetric_varon_window(
        np.asarray([30.0, 40.0, 50.0]),
        np.asarray([60.0, 70.0, 80.0]),
        statistic="p95",
    )

    assert np.isclose(pre_ms, 49.0)
    assert np.isclose(post_ms, 79.0)


def test_asymmetric_varon_honors_each_sampled_side_independently():
    ecg, peaks, fs = _repeated_waveform_ecg()
    result = extract_patient_asymmetric_varon_morphology(
        ecg,
        peaks,
        fs,
        np.full(5, 40.0),
        np.full(5, 80.0),
        anchor_track="expert",
        patient_id="fixture_patient",
        lead_name="II",
        event_context=_event_context(peaks),
    )

    assert result.parameters["pre_r_samples"] == 10
    assert result.parameters["post_r_samples"] == 20
    assert result.parameters["waveform_sample_count"] == 31
    assert result.parameters["capture_sample_rule"] == (
        "ceil_each_asymmetric_side_and_include_anchor_once"
    )
    assert result.parameters["insufficient_calibration_fallback_used"] is False
    assert (
        result.beat_context["capture_start_sample_in_segment"].iloc[0]
        == peaks[0] - 10
    )
    assert (
        result.beat_context["capture_end_sample_exclusive_in_segment"].iloc[0]
        == peaks[0] + 21
    )


def test_asymmetric_varon_falls_back_when_calibration_is_too_small():
    ecg, peaks, fs = _repeated_waveform_ecg()
    result = extract_patient_asymmetric_varon_morphology(
        ecg,
        peaks,
        fs,
        np.asarray([40.0, 45.0, 50.0]),
        np.asarray([70.0, 75.0, 80.0]),
        anchor_track="expert",
        patient_id="fixture_patient",
        lead_name="II",
    )

    assert result.parameters["insufficient_calibration_fallback_used"] is True
    assert result.parameters["pre_r_ms_requested"] == 60.0
    assert result.parameters["post_r_ms_requested"] == 60.0
    assert result.parameters["waveform_sample_count"] == 30
    assert result.parameters["capture_sample_rule"] == (
        "published_total_length_then_symmetric_split"
    )


def test_fusion_table_retains_timescales_and_does_not_make_prediction():
    ecg, peaks, fs = _repeated_waveform_ecg()
    morphology = extract_varon_morphology(
        ecg,
        peaks,
        fs,
        anchor_track="synthetic",
        event_context=_event_context(peaks),
    )
    hrv = pd.DataFrame(
        {
            "end_time_s": peaks[1:] / fs,
            "feature_defined": [False] * 3 + [True] * (peaks.size - 4),
            "feature_reliable": [False] * 3 + [True] * (peaks.size - 4),
            "rr_ms": np.diff(peaks) * 1000.0 / fs,
            "j1_csi_x_slope": np.arange(peaks.size - 1, dtype=float),
        }
    )
    joined = build_fusion_ready_table(
        hrv,
        morphology.features,
        anchor_track="synthetic",
    )

    assert "morph_lambda1" in joined
    assert "hrv_j1_csi_x_slope" in joined
    assert np.allclose(joined["hrv_age_s"], 0.0)
    assert not joined["seizure_probability_produced"].any()
    assert not joined["published_varon_5s_reproduction"].any()
    assert not joined["prsa_context_available"].any()
    assert not joined["prsa_context_affects_core_fusion_gate"].any()


def test_prsa_bprsa_is_causal_context_and_does_not_gate_core_fusion():
    morphology = pd.DataFrame(
        {
            "window_end_time_s": [10.0, 20.0, 30.0],
            "published_varon_core_defined": [True, True, True],
            "support_context_pass": [True, True, True],
            "lambda1": [1.0, 2.0, 3.0],
        }
    )
    hrv = pd.DataFrame(
        {
            "end_time_s": [9.0, 19.0, 29.0],
            "feature_defined": [True, True, True],
            "feature_reliable": [True, True, True],
            "rr_ms": [800.0, 790.0, 780.0],
        }
    )
    context = pd.DataFrame(
        {
            "end_time_s": [15.0, 25.0, 35.0],
            "feature_defined": [True, True, True],
            "feature_reliable": [False, True, True],
            "mean_rr80_ms": [801.0, 802.0, 999.0],
            "sdnn80_ms": [20.0, 21.0, 99.0],
            "prsa_s_rr_ms_per_sample": [1.0, 2.0, 99.0],
            "prsa_delta_rr_ms_per_sample": [3.0, 4.0, 99.0],
            "bprsa_s_r_ms_per_sample": [5.0, 6.0, 99.0],
            "bprsa_delta_r_ms_per_sample": [7.0, 8.0, 99.0],
            "prsa_anchor_count": [10, 11, 12],
        }
    )

    joined = build_fusion_ready_table(
        hrv,
        morphology,
        anchor_track="synthetic",
        prsa_bprsa_context=context,
    )

    assert joined["fusion_measurements_defined"].all()
    assert joined["fusion_context_pass"].all()
    assert not joined.loc[0, "prsa_context_available"]
    assert joined.loc[1, "prsa_context_measurement_time_s"] == 15.0
    assert joined.loc[1, "prsa_context_mean_rr80_ms"] == 801.0
    assert not joined.loc[1, "prsa_context_usable"]
    assert joined.loc[2, "prsa_context_measurement_time_s"] == 25.0
    assert joined.loc[2, "prsa_context_mean_rr80_ms"] == 802.0
    assert joined.loc[2, "prsa_context_usable"]
    assert joined.loc[2, "prsa_context_computationally_usable"]
    assert not joined["prsa_context_signal_quality_attached"].any()
    assert not joined["prsa_context_model_eligible"].any()
    assert joined.loc[2, "prsa_context_prsa_anchor_count"] == 11
    assert (
        joined["prsa_context_role"]
        == "autonomic_cardiorespiratory_context_only"
    ).all()


def test_association_table_preserves_detector_only_events():
    neurokit = np.array([100, 200, 300])
    zhai = np.array([102, 298, 450])
    agreement = compare_peak_sequences(
        neurokit,
        zhai,
        sampling_rate_hz=1000.0,
        tolerance_ms=5.0,
    )
    associations = build_detector_association_table(
        agreement,
        sampling_rate_hz=1000.0,
        segment_start_s=0.0,
    )

    assert associations["agreement_status"].value_counts().to_dict() == {
        "matched": 2,
        "neurokit_only": 1,
        "zhai_only": 1,
    }
    assert not associations["timestamp_selected"].any()
    assert not associations["artifact_label_produced"].any()
