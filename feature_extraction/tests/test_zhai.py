import numpy as np

from ecg_cascade import NeuroKitZhaiConfig, run_neurokit_zhai_array
from ecg_cascade.zhai import detect_zhai_template


def _synthetic_qrs_train(fs: float = 250.0) -> tuple[np.ndarray, np.ndarray]:
    sample_count = int(12 * fs)
    samples = np.arange(sample_count)
    reference = np.arange(int(fs), int(11 * fs) + 1, int(fs))
    ecg = np.zeros(sample_count, dtype=float)
    for peak in reference:
        ecg += 1.2 * np.exp(-0.5 * ((samples - peak) / 3.0) ** 2)
        ecg -= 0.35 * np.exp(-0.5 * ((samples - (peak + 8)) / 5.0) ** 2)
    ecg += 0.02 * np.sin(2 * np.pi * samples / (3 * fs))
    return ecg, reference


def test_zhai_full_track_localizes_synthetic_qrs_without_positive_refinement():
    ecg, reference = _synthetic_qrs_train()
    result = detect_zhai_template(ecg, 250.0)

    assert result.peak_samples.tolist() == reference.tolist()
    assert result.template.size == 31
    assert result.detector.parameters["raw_amplitude_argmax_refinement"] is False
    assert result.detector.parameters["localization"] == (
        "maximum_absolute_normalized_cross_correlation"
    )
    assert np.all(np.abs(result.correlation_peak_values) > 0.99)


def test_zhai_template_matching_is_invariant_to_whole_record_inversion():
    ecg, _ = _synthetic_qrs_train()
    original = detect_zhai_template(ecg, 250.0, orientation="original")
    inverted = detect_zhai_template(ecg, 250.0, orientation="inverted")

    assert inverted.peak_samples.tolist() == original.peak_samples.tolist()
    assert np.allclose(
        np.abs(inverted.correlation_peak_values),
        np.abs(original.correlation_peak_values),
        atol=1e-10,
    )


def test_two_track_architecture_keeps_timestamps_and_rr_outputs_separate():
    ecg, _ = _synthetic_qrs_train()
    result = run_neurokit_zhai_array(
        ecg,
        250.0,
        config=NeuroKitZhaiConfig(
            compute_inverted_context=False,
            hrv_window_rr_intervals=5,
        ),
    )

    neurokit = result.selected.neurokit
    zhai = result.selected.zhai
    assert result.summary()["timestamp_owner"] == (
        "unselected_pending_annotation_validation"
    )
    assert not result.summary()["automatic_timestamp_fusion_applied"]
    assert not result.summary()["raw_amplitude_argmax_refinement_applied"]
    assert neurokit.events["primary_timestamp_sample"].tolist() == (
        neurokit.detector.peak_samples.tolist()
    )
    assert zhai.events["primary_timestamp_sample"].tolist() == (
        zhai.detector.peak_samples.tolist()
    )
    assert set(neurokit.rr_intervals["timestamp_owner_track"]) == {"neurokit"}
    assert set(zhai.rr_intervals["timestamp_owner_track"]) == {"zhai"}
    assert "zhai_absolute_correlation" in zhai.events
    assert int(zhai.features["window_rr_count"].max()) == 5
    # Perfectly constant synthetic RR has SD1=0, so CSI is intentionally
    # undefined rather than forced to an arbitrary finite number.
    assert not zhai.features["feature_defined"].any()
