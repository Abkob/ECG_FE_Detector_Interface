from pathlib import Path

import numpy as np
import pytest

from ecg_cascade.clinical_demo_data import (
    DemoSegment,
    build_preview_payload,
    bundled_demo_record,
    feature_definitions,
    inspect_signal_source,
    load_signal_segment,
    run_demo_analysis,
)


DEMO_RECORD = bundled_demo_record()


@pytest.mark.parametrize("calibration_beats", [0, 1, 2, 4, 101])
def test_demo_rejects_calibration_that_cannot_build_five_qrs_varon(calibration_beats: int):
    segment = DemoSegment(
        samples=np.zeros(360, dtype=float),
        sampling_rate_hz=360.0,
        start_s=0.0,
        channel="MLII",
        source_name="validation-only",
        source_kind="ARRAY",
        unit="mV",
    )

    with pytest.raises(ValueError, match="Varon lane requires five aligned QRS beats"):
        run_demo_analysis(segment, calibration_beats=calibration_beats)


@pytest.mark.skipif(not DEMO_RECORD.is_file(), reason="bundled MIT-BIH record unavailable")
def test_bundled_demo_runs_real_morphology_hrv_and_combined_rows():
    source = inspect_signal_source(DEMO_RECORD)
    assert source.kind == "WFDB"
    assert "MLII" in source.channels
    assert source.sampling_rate_hz == 360.0
    assert source.channel_units[source.channels.index("MLII")] == "mV"

    segment = load_signal_segment(
        source,
        channel="MLII",
        start_s=0.0,
        duration_s=120.0,
    )
    payload = run_demo_analysis(segment, timestamp_track="neurokit")

    assert payload["summary"]["r_peak_count"] > 100
    assert payload["summary"]["morphology_window_count"] > 90
    assert payload["summary"]["defined_hrv_count"] > 0
    assert payload["summary"]["defined_prsa_bprsa_count"] > 0
    assert payload["summary"]["combined_defined_count"] > 0
    assert payload["interpretation"]["classifier_applied"] is False
    assert payload["interpretation"]["seizure_probability_produced"] is False
    assert len(payload["feature_definitions"]) == 52
    assert payload["summary"]["signal_quality_window_count"] == 23
    assert payload["summary"]["conduction_beat_count"] == payload["summary"]["r_peak_count"]
    assert payload["signal_quality"]["artifact_label_produced"] is False
    assert payload["signal_quality"]["threshold_applied"] is False
    assert payload["signal_quality"]["amplitude_calibrated"] is True
    assert payload["signal_quality"]["rows"][-1]["available_count"] > 0
    assert payload["conduction"]["information_only"] is True
    assert payload["conduction"]["final_feature_matrix_eligible"] is False
    assert payload["conduction"]["abnormality_label_produced"] is False
    assert len(payload["conduction"]["beats"]) == payload["summary"]["r_peak_count"]
    assert len(payload["conduction"]["five_beat"]) > len(payload["conduction"]["beats"])
    assert payload["prsa_bprsa"]["paper_feature_count"] == 6
    assert payload["prsa_bprsa"]["calibration_applied"] is False
    assert payload["prsa_bprsa"]["classifier_applied"] is False
    assert len(payload["prsa_bprsa"]["curve_time_s"]) == 41

    detail = payload["morphology_detail"]
    assert "fixed_varon" not in detail
    calibration = detail["patient_calibration"]
    assert calibration["requested_beats"] == 20
    assert calibration["used_beats"] == 20
    assert calibration["manual_ground_truth"] is False
    assert calibration["boundary_source"].startswith("automatic Emrich-2024")
    assert calibration["frozen_at_time_s"] is not None
    assert calibration["applied_pre_r_ms"] > 0
    assert calibration["applied_post_r_ms"] > 0
    assert detail["patient_template"]["used_calibration_beats"] == 20
    assert detail["patient_template"]["evaluation_beat_count"] > 0
    assert detail["pqrst"]["complete_qrs_count"] > 100

    before_freeze = [
        row
        for row in payload["fusion"]
        if row["time_s"] < calibration["frozen_at_time_s"]
    ]
    assert before_freeze
    assert all(row["values"]["cal_varon_lambda1"] is None for row in before_freeze)
    assert all(row["values"]["template_correlation"] is None for row in before_freeze)

    complete = next(row for row in payload["fusion"] if row["measurements_defined"])
    assert complete["values"]["cal_varon_lambda1"] is not None
    assert complete["values"]["template_correlation"] is not None
    assert complete["values"]["pqrst_qrs_duration_ms"] is not None
    assert complete["values"]["hrv_sd1_raw_ms"] is not None
    assert complete["values"]["hrv_j2_modcsi_filtered_x_slope"] is not None
    assert complete["values"]["prsa_s_rr_ms_per_sample"] is not None
    assert complete["values"]["bprsa_s_r_ms_per_sample"] is not None
    assert complete["prsa_bprsa_defined"]
    assert complete["prsa_bprsa_usable"] == complete["prsa_bprsa_reliable"]
    assert complete["prsa_bprsa_computationally_usable"] == complete["prsa_bprsa_reliable"]
    assert complete["prsa_bprsa_signal_quality_attached"] is False
    assert complete["prsa_bprsa_model_eligible"] is False
    assert complete["prsa_bprsa_role"] == "autonomic_cardiorespiratory_context_only"
    assert complete["prsa_bprsa_affects_core_gate"] is False
    assert complete["values"]["quality_bassqi"] is not None
    assert complete["values"]["quality_bsqi"] is not None
    assert complete["values"]["conduction_jt_interval_ms"] is not None
    assert complete["values"]["conduction_qtc_fridericia_ms"] is not None
    assert complete["values"]["conduction_qt_median5_ms"] is not None
    assert complete["signal_quality_context_available"] is True
    assert complete["signal_quality_artifact_label_produced"] is False
    assert complete["signal_quality_affects_core_gate"] is False
    assert complete["conduction_information_available"] is True
    assert complete["conduction_final_matrix_eligible"] is False
    assert complete["conduction_abnormality_label_produced"] is False
    assert complete["conduction_affects_core_gate"] is False


def test_csv_time_column_infers_rate_and_preview_is_finite(tmp_path: Path):
    path = tmp_path / "signal.csv"
    time = np.arange(0.0, 5.0, 1.0 / 100.0)
    signal = np.sin(2 * np.pi * 1.2 * time)
    path.write_text(
        "time_s,ECG\n" + "\n".join(f"{t:.5f},{value:.8f}" for t, value in zip(time, signal, strict=True)),
        encoding="utf-8",
    )

    source = inspect_signal_source(path)
    assert source.kind == "CSV"
    assert source.channels == ("ECG",)
    assert source.sampling_rate_hz == pytest.approx(100.0)
    assert source.channel_units == ("unknown",)

    segment = load_signal_segment(
        source,
        channel="ECG",
        start_s=0.0,
        duration_s=5.0,
    )
    preview = build_preview_payload(segment, maximum_points=100)
    assert len(preview["times_s"]) <= 100
    assert all(np.isfinite(preview["values"]))


def test_segment_duration_has_no_artificial_upper_limit(tmp_path: Path):
    path = tmp_path / "signal.csv"
    time = np.arange(0.0, 5.0, 1.0 / 100.0)
    signal = np.sin(2 * np.pi * 1.2 * time)
    path.write_text(
        "time_s,ECG\n"
        + "\n".join(
            f"{t:.5f},{value:.8f}"
            for t, value in zip(time, signal, strict=True)
        ),
        encoding="utf-8",
    )

    source = inspect_signal_source(path)
    segment = load_signal_segment(
        source,
        channel="ECG",
        start_s=0.0,
        duration_s=3_600.0,
    )

    assert segment.duration_s == pytest.approx(5.0)


def test_feature_matrix_definitions_keep_morphology_and_hrv_visibly_separate():
    definitions = feature_definitions()
    assert [item["label"] for item in definitions[:5]] == ["λ1", "λ2", "λ3", "λ4", "λ5"]
    assert {item["branch"] for item in definitions[:5]} == {"Varon patient"}
    assert {item["branch"] for item in definitions[5:10]} == {"Template"}
    assert {item["branch"] for item in definitions[10:16]} == {"P–QRS–T"}
    assert {item["branch"] for item in definitions[16:25]} == {"HRV"}
    assert {item["branch"] for item in definitions[25:31]} == {"PRSA/BPRSA"}
    assert {item["branch"] for item in definitions[31:39]} == {"Signal quality"}
    assert {item["branch"] for item in definitions[39:]} == {"Conduction"}
    assert {item["role"] for item in definitions[:25]} == {"core_measurement"}
    assert {item["role"] for item in definitions[25:31]} == {"context"}
    assert {item["role"] for item in definitions[31:39]} == {"quality_context"}
    assert {item["role"] for item in definitions[39:]} == {"information_only"}
