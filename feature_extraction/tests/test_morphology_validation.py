import numpy as np

from ecg_cascade.config import NeuroKitZhaiConfig
from ecg_cascade.morphology_validation import (
    audit_morphology_timestamp_fidelity,
    build_reference_context,
)


def _signal() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    fs = 250.0
    peaks = np.arange(250, 2251, 250, dtype=np.int64)
    symbols = np.asarray(["N", "N", "N", "V", "N", "N", "N", "N", "N"], dtype=object)
    ecg = np.zeros(2500, dtype=float)
    template = np.asarray([-0.1, 0.2, 1.0, -0.5, -0.15])
    for index, peak in enumerate(peaks):
        waveform = template if symbols[index] == "N" else np.asarray([0.2, -0.8, -1.2, 0.4, 0.1])
        ecg[peak - 2 : peak + 3] = waveform
    return ecg, peaks, symbols, fs


def test_identical_expert_and_candidate_timestamps_have_perfect_fidelity():
    ecg, peaks, symbols, fs = _signal()
    audit = audit_morphology_timestamp_fidelity(
        ecg,
        peaks,
        symbols,
        peaks.copy(),
        fs,
        candidate_track="same",
    )

    assert audit.summary["beat_f1"] == 1.0
    assert audit.summary["five_beat_window_coverage"] == 1.0
    assert audit.summary["median_raw_eigenvalue_relative_l1_error"] == 0.0
    assert audit.summary["lambda1_ccc"] == 1.0
    assert not audit.summary["seizure_accuracy_measured"]


def test_extra_candidate_breaks_strict_five_beat_stream_windows():
    ecg, peaks, symbols, fs = _signal()
    candidates = np.sort(np.concatenate([peaks, np.asarray([875])]))
    audit = audit_morphology_timestamp_fidelity(
        ecg,
        peaks,
        symbols,
        candidates,
        fs,
        candidate_track="extra",
    )

    affected = audit.windows[
        audit.windows["unavailable_reason"]
        == "extra_candidate_between_matched_expert_beats"
    ]
    assert not affected.empty
    assert audit.summary["five_beat_window_coverage"] < 1.0


def test_reference_context_carries_expert_symbol_and_timing_error():
    _, peaks, symbols, fs = _signal()
    candidates = peaks + 2
    context, agreement = build_reference_context(
        candidates,
        peaks,
        symbols,
        sampling_rate_hz=fs,
    )

    assert agreement.matched_primary_samples.size == peaks.size
    assert context["reference_symbol"].tolist() == symbols.tolist()
    assert np.allclose(context["reference_match_error_ms"], 8.0)


def test_default_varon_capture_is_exactly_120_ms_after_sample_rounding():
    ecg, peaks, symbols, fs = _signal()
    audit = audit_morphology_timestamp_fidelity(
        ecg,
        peaks,
        symbols,
        peaks,
        fs,
        candidate_track="same",
    )

    assert audit.candidate.parameters["pre_r_ms_requested"] == 60.0
    assert audit.candidate.parameters["post_r_ms_requested"] == 60.0
    assert audit.candidate.beat_waveforms.shape[1] == 30


def test_integrated_neurokit_zhai_pipeline_defaults_to_2015_window():
    config = NeuroKitZhaiConfig()

    assert config.varon_pre_r_ms == 60.0
    assert config.varon_post_r_ms == 60.0
    assert "Varon_2015" in str(config.to_dict()["morphology_core"])
