from pathlib import Path

import pytest

from ecg_cascade.rpeak_viewer import (
    ReviewStore,
    detect_wfdb_window,
    discover_recordings,
)


RECORD_207 = (
    Path(__file__).resolve().parents[2]
    / "Datasets"
    / "mit-bih-arrhythmia-database-1.0"
    / "mit-bih-arrhythmia-database-1.0.0"
    / "207.hea"
)


def test_review_store_round_trip_preserves_manual_decisions(tmp_path: Path):
    recording = tmp_path / "example.edf"
    recording.touch()
    store = ReviewStore(
        recording_path=recording,
        channel="ECG",
        sampling_rate_hz=256.0,
        output_dir=tmp_path / "reviews",
    )

    store.set_label(2560, source="neurokit", label="false_positive")
    store.set_label(3000, source="manual", label="manual_missed_r_peak")

    reloaded = ReviewStore(
        recording_path=recording,
        channel="ECG",
        sampling_rate_hz=256.0,
        output_dir=tmp_path / "reviews",
    )
    assert reloaded.records[2560].time_s == 10.0
    assert reloaded.records[2560].label == "false_positive"
    assert reloaded.records[3000].source == "manual"
    assert reloaded.output_path.is_file()


def test_review_store_clear_label_is_persistent(tmp_path: Path):
    recording = tmp_path / "example.edf"
    recording.touch()
    store = ReviewStore(
        recording_path=recording,
        channel="ECG lead",
        sampling_rate_hz=250.0,
        output_dir=tmp_path / "reviews",
    )
    store.set_label(500, source="neurokit", label="uncertain")

    assert store.clear_label(500)
    assert not store.clear_label(500)

    reloaded = ReviewStore(
        recording_path=recording,
        channel="ECG lead",
        sampling_rate_hz=250.0,
        output_dir=tmp_path / "reviews",
    )
    assert reloaded.records == {}


def test_record_discovery_returns_one_entry_per_edf_or_wfdb_header(tmp_path: Path):
    (tmp_path / "100.hea").touch()
    (tmp_path / "100.dat").touch()
    (tmp_path / "study.edf").touch()
    (tmp_path / "ignore.csv").touch()

    assert [path.name for path in discover_recordings(tmp_path)] == [
        "100.hea",
        "study.edf",
    ]


@pytest.mark.skipif(not RECORD_207.is_file(), reason="local MIT-BIH record 207 unavailable")
def test_record_207_loads_neurokit_and_expert_atr_beats():
    window = detect_wfdb_window(
        RECORD_207,
        channel_label="MLII",
        start_s=0.0,
        duration_s=15.0,
    )

    assert window.input_format == "WFDB"
    assert window.sampling_rate_hz == 360.0
    assert window.neurokit_peak_samples.size == 12
    assert window.reference_beat_samples.size == 14
    assert window.matched_detection_samples.size == 12
    assert window.false_positive_samples.size == 0
    assert window.missed_reference_samples.size == 2
    assert set(window.detector_peak_samples) == {"neurokit", "unsw", "zhai"}
    assert set(window.reference_comparisons) == {"neurokit", "unsw", "zhai"}
    assert set(window.morphology_results) == {"neurokit", "unsw", "zhai", "expert"}
    assert window.morphology_results["expert"].parameters["pre_r_ms_requested"] == 60.0
