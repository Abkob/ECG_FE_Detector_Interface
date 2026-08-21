"""Data preparation for the clinician-facing ECG branch web demo.

The browser is deliberately a presentation layer.  All values exposed here
come from the existing NeuroKit--Zhai, RR/HRV, morphology, and causal
measurement-join implementation.  This module does not add a classifier or
turn detector agreement into a clinical label.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wfdb

from .config import NeuroKitZhaiConfig
from .conduction_information import extract_conduction_information
from .conduction_timing import extract_conduction_timing
from .edf import inspect_edf, load_ecg_segment
from .morphology import extract_patient_asymmetric_varon_morphology
from .neurokit_zhai import run_neurokit_zhai_array
from .patient_template_morphology import extract_patient_template_morphology
from .peaks import orient_signal
from .prsa_bprsa import extract_prsa_bprsa_features
from .prominence_morphology import extract_prominence_morphology
from .signal_quality import extract_trailing_signal_quality_windows
from .wfdb_io import load_wfdb_segment


SUPPORTED_SOURCE_SUFFIXES = {".edf", ".hea", ".dat", ".csv", ".txt"}


@dataclass(frozen=True)
class SignalSource:
    """An inspected signal source that can be safely presented by the UI."""

    path: str
    kind: str
    name: str
    channels: tuple[str, ...]
    sampling_rate_hz: float | None
    duration_s: float | None
    channel_units: tuple[str, ...]
    csv_time_column: str | None = None

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "channels": list(self.channels),
            "sampling_rate_hz": _finite_or_none(self.sampling_rate_hz),
            "duration_s": _finite_or_none(self.duration_s),
            "channel_units": {
                channel: unit
                for channel, unit in zip(self.channels, self.channel_units, strict=True)
            },
            "csv_time_column": self.csv_time_column,
        }


@dataclass(frozen=True)
class DemoSegment:
    """One finite, one-dimensional signal segment selected by the user."""

    samples: np.ndarray
    sampling_rate_hz: float
    start_s: float
    channel: str
    source_name: str
    source_kind: str
    unit: str

    @property
    def duration_s(self) -> float:
        return float(self.samples.size / self.sampling_rate_hz)


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def bundled_demo_record() -> Path:
    return (
        project_root()
        / "Datasets"
        / "mit-bih-arrhythmia-database-1.0"
        / "mit-bih-arrhythmia-database-1.0.0"
        / "100.hea"
    )


def inspect_signal_source(
    path: str | Path,
    *,
    csv_sampling_rate_hz: float | None = None,
) -> SignalSource:
    """Inspect EDF, WFDB, or numeric CSV input without running the branches."""

    candidate = Path(path).expanduser().resolve()
    suffix = candidate.suffix.casefold()
    if suffix not in SUPPORTED_SOURCE_SUFFIXES:
        raise ValueError(
            "Supported inputs are EDF, WFDB (.hea + .dat), and numeric CSV/TXT."
        )

    if suffix == ".edf":
        recording = inspect_edf(candidate)
        if not recording.channels:
            raise ValueError("The EDF contains no signal channels.")
        rates = {round(channel.sampling_rate_hz, 9) for channel in recording.channels}
        common_rate = next(iter(rates)) if len(rates) == 1 else None
        return SignalSource(
            path=recording.path,
            kind="EDF",
            name=candidate.name,
            channels=tuple(channel.label for channel in recording.channels),
            sampling_rate_hz=common_rate,
            duration_s=float(recording.duration_s),
            channel_units=tuple(
                channel.physical_dimension or "unknown"
                for channel in recording.channels
            ),
        )

    if suffix in {".hea", ".dat"}:
        record_path = candidate.with_suffix("")
        header_path = record_path.with_suffix(".hea")
        if not header_path.is_file():
            raise FileNotFoundError(
                f"WFDB header not found: {header_path.name}. Select the matching .hea file."
            )
        header = wfdb.rdheader(str(record_path))
        header_units = tuple(
            str(unit or "unknown") for unit in (getattr(header, "units", None) or [])
        )
        if len(header_units) != len(header.sig_name):
            header_units = tuple("unknown" for _ in header.sig_name)
        return SignalSource(
            path=str(record_path),
            kind="WFDB",
            name=header_path.name,
            channels=tuple(str(name) for name in header.sig_name),
            sampling_rate_hz=float(header.fs),
            duration_s=float(header.sig_len / header.fs),
            channel_units=header_units,
        )

    frame = _read_csv(candidate)
    time_column, inferred_rate = _infer_csv_time_axis(frame)
    channels = _numeric_signal_columns(frame, time_column=time_column)
    if not channels:
        raise ValueError(
            "No fully numeric signal column was found in the CSV/TXT file."
        )
    rate = _validate_sampling_rate(csv_sampling_rate_hz, allow_none=True)
    if rate is None:
        rate = inferred_rate
    duration = float(len(frame) / rate) if rate is not None else None
    return SignalSource(
        path=str(candidate),
        kind="CSV",
        name=candidate.name,
        channels=tuple(channels),
        sampling_rate_hz=rate,
        duration_s=duration,
        channel_units=tuple("unknown" for _ in channels),
        csv_time_column=time_column,
    )


def load_signal_segment(
    source: SignalSource,
    *,
    channel: str,
    start_s: float,
    duration_s: float,
    csv_sampling_rate_hz: float | None = None,
) -> DemoSegment:
    """Load a finite segment from an already inspected signal source."""

    if not np.isfinite(start_s) or start_s < 0:
        raise ValueError("Segment start must be a non-negative number of seconds.")
    if not np.isfinite(duration_s) or not 3 <= duration_s <= 300:
        raise ValueError("Segment duration must be between 3 and 300 seconds.")
    if channel not in source.channels:
        raise ValueError(f"Unknown channel {channel!r}; choose one shown in the app.")
    channel_unit = source.channel_units[source.channels.index(channel)]

    if source.kind == "WFDB":
        segment = load_wfdb_segment(
            source.path,
            channel=channel,
            start_s=start_s,
            duration_s=duration_s,
        )
        return DemoSegment(
            samples=segment.samples,
            sampling_rate_hz=segment.sampling_rate_hz,
            start_s=segment.start_s,
            channel=segment.channel_name,
            source_name=source.name,
            source_kind=source.kind,
            unit=channel_unit,
        )

    if source.kind == "EDF":
        segment = load_ecg_segment(
            source.path,
            channel_label=channel,
            start_s=start_s,
            duration_s=duration_s,
        )
        return DemoSegment(
            samples=segment.samples,
            sampling_rate_hz=segment.channel.sampling_rate_hz,
            start_s=segment.start_s,
            channel=segment.channel.label,
            source_name=source.name,
            source_kind=source.kind,
            unit=channel_unit,
        )

    rate = _validate_sampling_rate(
        csv_sampling_rate_hz if csv_sampling_rate_hz is not None else source.sampling_rate_hz,
        allow_none=False,
    )
    frame = _read_csv(Path(source.path))
    values = pd.to_numeric(frame[channel], errors="coerce").to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"CSV column {channel!r} contains missing or non-numeric samples."
        )
    first = int(round(start_s * rate))
    if first >= values.size:
        raise ValueError("Segment start lies beyond the CSV signal.")
    last = min(values.size, first + int(round(duration_s * rate)))
    samples = values[first:last]
    if samples.size < int(np.ceil(3 * rate)):
        raise ValueError("At least three seconds of ECG are required.")
    return DemoSegment(
        samples=samples,
        sampling_rate_hz=rate,
        start_s=float(first / rate),
        channel=channel,
        source_name=source.name,
        source_kind=source.kind,
        unit=channel_unit,
    )


def build_preview_payload(segment: DemoSegment, *, maximum_points: int = 3000) -> dict[str, Any]:
    times, values = _plot_signal(segment, maximum_points=maximum_points)
    return {
        "source": segment.source_name,
        "kind": segment.source_kind,
        "channel": segment.channel,
        "sampling_rate_hz": float(segment.sampling_rate_hz),
        "start_s": float(segment.start_s),
        "duration_s": float(segment.duration_s),
        "unit": segment.unit,
        "times_s": times,
        "values": values,
    }


def run_demo_analysis(
    segment: DemoSegment,
    *,
    timestamp_track: str = "neurokit",
    calibration_beats: int = 20,
) -> dict[str, Any]:
    """Run the real branch code and serialize an animation-ready result."""

    if timestamp_track not in {"neurokit", "zhai"}:
        raise ValueError("timestamp_track must be 'neurokit' or 'zhai'.")
    if not 5 <= int(calibration_beats) <= 100:
        raise ValueError(
            "calibration_beats must be between 5 and 100; "
            "the Varon lane requires five aligned QRS beats."
        )
    calibration_beats = int(calibration_beats)
    config = NeuroKitZhaiConfig(compute_inverted_context=False)
    result = run_neurokit_zhai_array(
        segment.samples,
        segment.sampling_rate_hz,
        segment_start_s=segment.start_s,
        config=config,
        source_metadata={
            "kind": segment.source_kind,
            "name": segment.source_name,
            "channel": segment.channel,
            "presentation": "clinician_web_demo",
        },
    )
    track = result.selected.neurokit if timestamp_track == "neurokit" else result.selected.zhai
    anchor_samples = track.events["primary_timestamp_sample"].to_numpy(dtype=np.int64)
    patient_id = Path(segment.source_name).stem
    orientation = str(result.selected_orientation)

    prsa_bprsa_result = extract_prsa_bprsa_features(
        segment.samples,
        anchor_samples,
        segment.sampling_rate_hz,
        segment_start_s=segment.start_s,
        orientation=orientation,
        rr_supported=track.rr_intervals["rr_supported"].to_numpy(dtype=bool),
        r_peak_supported=track.events["qrs_supported"].to_numpy(dtype=bool),
        lead_name=segment.channel,
        anchor_track=timestamp_track,
    )

    prominence = extract_prominence_morphology(
        segment.samples,
        anchor_samples,
        segment.sampling_rate_hz,
        anchor_track=timestamp_track,
        patient_id=patient_id,
        lead_name=segment.channel,
        segment_start_s=segment.start_s,
        orientation=orientation,
    )
    calibration = _calibration_payload(
        segment,
        prominence.features,
        requested_beats=calibration_beats,
        orientation=orientation,
    )
    calibrated_varon = None
    if calibration["used_beats"] == calibration_beats and calibration_beats >= 5:
        calibrated_varon = extract_patient_asymmetric_varon_morphology(
            segment.samples,
            anchor_samples,
            segment.sampling_rate_hz,
            np.asarray(calibration["onset_to_r_ms"], dtype=np.float64),
            np.asarray(calibration["r_to_offset_ms"], dtype=np.float64),
            anchor_track=timestamp_track,
            patient_id=patient_id,
            lead_name=segment.channel,
            statistic="p95",
            minimum_calibration_beats=5,
            segment_start_s=segment.start_s,
            orientation=orientation,
            event_context=track.events,
        )
        calibration.update(
            {
                "calibrated_pre_r_ms": _finite_or_none(
                    calibrated_varon.parameters["calibrated_pre_r_ms"]
                ),
                "calibrated_post_r_ms": _finite_or_none(
                    calibrated_varon.parameters["calibrated_post_r_ms"]
                ),
                "applied_pre_r_ms": _finite_or_none(
                    calibrated_varon.parameters["pre_r_ms_realized"]
                ),
                "applied_post_r_ms": _finite_or_none(
                    calibrated_varon.parameters["post_r_ms_realized"]
                ),
                "calibration_sufficient": True,
            }
        )

    template = _run_template_lane(
        segment,
        anchor_samples,
        timestamp_track=timestamp_track,
        patient_id=patient_id,
        orientation=orientation,
        calibration_beats=calibration_beats,
    )
    signal_times, signal_values = _plot_signal(segment, maximum_points=75000)

    events = [
        {
            "time_s": _number(row.primary_timestamp_time_s),
            "supported": bool(row.qrs_supported),
        }
        for row in track.events.itertuples(index=False)
    ]
    rr = [
        {
            "time_s": _number(row.end_time_s),
            "rr_ms": _number(row.rr_ms),
            "heart_rate_bpm": _number(row.heart_rate_bpm),
            "supported": bool(row.rr_supported),
        }
        for row in track.rr_intervals.itertuples(index=False)
    ]
    hrv = [
        {
            "time_s": _number(row.end_time_s),
            "rr_ms": _number(row.rr_ms),
            "heart_rate_bpm": _number(row.heart_rate_bpm),
            "window_rr_count": int(row.window_rr_count),
            "sd1_ms": _finite_or_none(row.sd1_raw_ms),
            "sd2_ms": _finite_or_none(row.sd2_raw_ms),
            "csi": _finite_or_none(row.csi100),
            "modcsi_ms": _finite_or_none(row.modcsi100_filtered_ms),
            "slope_bpm_per_s": _finite_or_none(row.slope100_bpm_per_s),
            "j1": _finite_or_none(row.j1_csi_x_slope),
            "j2": _finite_or_none(row.j2_modcsi_filtered_x_slope),
            "defined": bool(row.feature_defined),
            "reliable": bool(row.feature_reliable),
            "coverage": _finite_or_none(row.rr_reliability_coverage),
        }
        for row in track.features.itertuples(index=False)
    ]
    prsa_bprsa = _serialize_prsa_bprsa(prsa_bprsa_result)
    calibrated_morphology = (
        _serialize_varon_features(calibrated_varon.features)
        if calibrated_varon is not None
        else []
    )
    calibrated_beats = (
        {
            "times_s": [
                _number(value)
                for value in calibrated_varon.beat_context["anchor_time_s"].to_numpy(
                    dtype=float
                )
            ],
            "waveform_time_ms": _waveform_time_axis_ms(calibrated_varon.parameters),
            "waveforms": [
                [_finite_or_none(value) for value in row]
                for row in calibrated_varon.beat_waveforms
            ],
        }
        if calibrated_varon is not None
        else {"times_s": [], "waveform_time_ms": [], "waveforms": []}
    )
    pqrst = _serialize_prominence(prominence)
    secondary_track = (
        result.selected.zhai if timestamp_track == "neurokit" else result.selected.neurokit
    )
    microvolts_per_unit = _microvolts_per_input_unit(segment.unit)
    quality_result = extract_trailing_signal_quality_windows(
        segment.samples,
        segment.sampling_rate_hz,
        uv_per_input_unit=(microvolts_per_unit or 1.0),
        window_s=10.0,
        step_s=5.0,
        primary_peak_samples=anchor_samples,
        secondary_peak_samples=secondary_track.detector.peak_samples,
        qrs_onset_samples=(
            prominence.features["qrs_onset_sample_in_segment"]
            .dropna()
            .to_numpy(dtype=np.int64)
        ),
    )
    quality_frame = quality_result.to_frame()
    if not quality_frame.empty:
        quality_frame["window_start_time_s"] = (
            segment.start_s
            + quality_frame["window_start_sample"].to_numpy(dtype=float)
            / segment.sampling_rate_hz
        )
        quality_frame["window_end_time_s"] = (
            segment.start_s
            + quality_frame["window_end_sample_exclusive"].to_numpy(dtype=float)
            / segment.sampling_rate_hz
        )
    quality = _serialize_signal_quality(
        quality_frame,
        input_unit=segment.unit,
        microvolts_per_input_unit=microvolts_per_unit,
    )

    conduction_timing = extract_conduction_timing(
        prominence.features,
        segment.sampling_rate_hz,
        variability_window_beats=30,
        minimum_valid_fraction=0.8,
        landmark_method=str(prominence.parameters["method"]),
    )
    conduction_information = extract_conduction_information(
        prominence.features,
        segment.sampling_rate_hz,
        window_beats=5,
        minimum_valid_beats=4,
        eligibility_column=None,
        landmark_method=str(prominence.parameters["method"]),
    )
    conduction = _serialize_conduction(
        conduction_timing,
        conduction_information,
    )
    fusion = _fusion_rows(
        track.fusion_ready,
        calibrated_features=(
            calibrated_varon.features if calibrated_varon is not None else None
        ),
        template_features=template.get("feature_frame"),
        pqrst_features=prominence.features,
        prsa_bprsa_features=prsa_bprsa_result.features,
        quality_features=quality_frame,
        conduction_beats=conduction_information.beat_measurements,
        conduction_five_beat=conduction_information.information_windows,
        conduction_qt_nine=conduction_information.qt_nine_beat_context,
        calibration_frozen_at_s=calibration.get("frozen_at_time_s"),
    )
    template.pop("feature_frame", None)

    agreement = result.selected.support_agreement.to_dict()
    return {
        "source": {
            "name": segment.source_name,
            "kind": segment.source_kind,
            "channel": segment.channel,
            "sampling_rate_hz": float(segment.sampling_rate_hz),
            "start_s": float(segment.start_s),
            "duration_s": float(segment.duration_s),
            "unit": segment.unit,
        },
        "track": {
            "key": timestamp_track,
            "label": "NeuroKit" if timestamp_track == "neurokit" else "Zhai 2023",
            "status": "display track only; timestamp owner is not clinically selected",
        },
        "signal": {"times_s": signal_times, "values": signal_values},
        "events": events,
        "rr": rr,
        "hrv": hrv,
        "prsa_bprsa": prsa_bprsa,
        "signal_quality": quality,
        "conduction": conduction,
        "morphology": calibrated_morphology,
        "morphology_beats": calibrated_beats,
        "morphology_detail": {
            "patient_calibration": calibration,
            "calibrated_varon": {
                "status": (
                    "implemented patient-calibrated Varon lane"
                    if calibrated_varon is not None
                    else "unavailable: fewer than five complete automatic QRS boundaries"
                ),
                "features": calibrated_morphology,
                "beats": calibrated_beats,
            },
            "patient_template": template,
            "pqrst": pqrst,
        },
        "fusion": fusion,
        "summary": {
            "r_peak_count": len(events),
            "rr_interval_count": len(rr),
            "morphology_window_count": len(calibrated_morphology),
            "defined_hrv_count": int(track.features["feature_defined"].sum()),
            "defined_prsa_bprsa_count": int(
                prsa_bprsa_result.features["feature_defined"].sum()
            ),
            "reliable_prsa_bprsa_count": int(
                prsa_bprsa_result.features["feature_reliable"].sum()
            ),
            "signal_quality_window_count": int(quality_frame.shape[0]),
            "conduction_beat_count": int(
                conduction_information.beat_measurements.shape[0]
            ),
            "conduction_five_beat_endpoint_count": int(
                conduction_information.information_windows["beat_index"].nunique()
            ),
            "combined_defined_count": int(
                sum(bool(row["measurements_defined"]) for row in fusion)
            ),
            "detector_matched_count": int(agreement["matched_count"]),
            "detector_support_tolerance_ms": float(config.support_tolerance_ms),
        },
        "feature_definitions": feature_definitions(),
        "interpretation": {
            "classifier_applied": False,
            "seizure_probability_produced": False,
            "artifact_label_produced": False,
            "signal_quality_threshold_applied": False,
            "signal_quality_is_measurement_context": True,
            "conduction_information_only": True,
            "conduction_final_feature_matrix_eligible": False,
            "conduction_abnormality_label_produced": False,
            "fusion_operation": "causal same-track measurement alignment only",
            "animation": "browser playback of the completed deterministic extraction",
            "qrs_boundary_ground_truth": False,
            "prsa_bprsa_classifier_applied": False,
            "prsa_bprsa_calibration_applied": False,
            "prsa_bprsa_matrix_role": (
                "autonomic/cardiorespiratory context; does not gate core "
                "morphology--RR/HRV row validity"
            ),
            "prsa_bprsa_provenance": (
                "Varon-2015 80-beat feature extractor: RR-triggered PRSA and "
                "R-amplitude-triggered BPRSA; no KSC classifier"
            ),
            "qrs_boundary_provenance": (
                "automatic Emrich-2024 prominence delineation via NeuroKit; "
                "MIT-BIH beat labels do not supply manual QRS onset/offset"
            ),
        },
    }


def feature_definitions() -> list[dict[str, str]]:
    """Columns used by the combined feature matrix, in display order."""

    return [
        _feature("cal_varon_lambda1", "λ1", "Varon patient", "energy", "Dominant five-QRS eigenvalue after the patient/lead asymmetric crop freezes"),
        _feature("cal_varon_lambda2", "λ2", "Varon patient", "energy", "Second calibrated-crop eigenvalue"),
        _feature("cal_varon_lambda3", "λ3", "Varon patient", "energy", "Third calibrated-crop eigenvalue"),
        _feature("cal_varon_lambda4", "λ4", "Varon patient", "energy", "Fourth calibrated-crop eigenvalue"),
        _feature("cal_varon_lambda5", "λ5", "Varon patient", "energy", "Fifth calibrated-crop eigenvalue"),
        _feature("template_correlation", "Corr", "Template", "r", "Correlation to the closest frozen patient/lead whole-cycle template"),
        _feature("template_normalized_rmse", "Shape RMSE", "Template", "norm", "Normalized residual from the closest template"),
        _feature("template_raw_residual_rms", "Raw residual", "Template", "amplitude", "Amplitude-sensitive residual RMS from the closest template"),
        _feature("template_derivative_dtw_cost", "D-DTW", "Template", "cost", "Derivative-DTW cost with a constrained warping band"),
        _feature("template_dtw_warp_fraction", "Warp", "Template", "fraction", "Fraction of the DTW path away from the diagonal"),
        _feature("pqrst_qrs_duration_ms", "QRS", "P–QRS–T", "ms", "Automatically delineated QRS onset-to-offset duration"),
        _feature("pqrst_pr_interval_ms", "PR", "P–QRS–T", "ms", "Automatically delineated P-onset to QRS-onset interval"),
        _feature("pqrst_qt_interval_ms", "QT", "P–QRS–T", "ms", "Automatically delineated QRS-onset to T-offset interval"),
        _feature("pqrst_qrs_peak_to_peak_amplitude", "QRS p-p", "P–QRS–T", "amplitude", "QRS peak-to-peak amplitude with preserved lead polarity"),
        _feature("pqrst_qrs_signed_area", "QRS area", "P–QRS–T", "amp·s", "Baseline-relative signed QRS area"),
        _feature("pqrst_landmark_availability_fraction", "Landmarks", "P–QRS–T", "fraction", "Available automatic P/QRS/T landmarks for the beat"),
        _feature("hrv_rr_ms", "RR", "HRV", "ms", "Time between the latest two R peaks"),
        _feature("hrv_heart_rate_bpm", "HR", "HRV", "bpm", "Instantaneous heart rate"),
        _feature("hrv_sd1_raw_ms", "SD1", "HRV", "ms", "Short-term beat-to-beat variation"),
        _feature("hrv_sd2_raw_ms", "SD2", "HRV", "ms", "Longer-axis Poincaré variation"),
        _feature("hrv_csi100", "CSI", "HRV", "ratio", "SD2-to-SD1 ratio over 100 RR intervals"),
        _feature("hrv_modcsi100_filtered_ms", "ModCSI", "HRV", "ms", "Filtered modified CSI"),
        _feature("hrv_slope100_bpm_per_s", "Slope", "HRV", "bpm/s", "Absolute heart-rate trend"),
        _feature("hrv_j1_csi_x_slope", "J1", "HRV", "index", "CSI multiplied by heart-rate slope"),
        _feature("hrv_j2_modcsi_filtered_x_slope", "J2", "HRV", "index", "Modified CSI multiplied by slope"),
        _feature("prsa_mean_rr80_ms", "Mean RR80", "PRSA/BPRSA", "ms", "Mean of the latest 80 RR intervals using the paper's population convention", role="context"),
        _feature("prsa_sdnn80_ms", "SDNN80", "PRSA/BPRSA", "ms", "Population standard deviation of the latest 80 RR intervals", role="context"),
        _feature("prsa_s_rr_ms_per_sample", "S-RR", "PRSA/BPRSA", "ms/sample", "Local PRSA slope across the samples immediately before and after the acceleration anchor", role="context"),
        _feature("prsa_delta_rr_ms_per_sample", "Δ-RR", "PRSA/BPRSA", "ms/sample", "Long-range PRSA slope across the full ±5-second curve", role="context"),
        _feature("bprsa_s_r_ms_per_sample", "S-R", "PRSA/BPRSA", "ms/sample", "Local RR response when R-peak amplitude is the BPRSA driver", role="context"),
        _feature("bprsa_delta_r_ms_per_sample", "Δ-R", "PRSA/BPRSA", "ms/sample", "Long-range RR response across the R-amplitude-triggered BPRSA curve", role="context"),
        _feature("quality_finite_fraction", "Finite", "Signal quality", "fraction", "Fraction of samples that are finite in the causal trailing 10-second window", role="quality_context"),
        _feature("quality_flat_fraction", "Flat", "Signal quality", "fraction", "Fraction of adjacent samples inside an exactly flat run; a measurement, not an artifact label", role="quality_context"),
        _feature("quality_bassqi", "basSQI", "Signal quality", "ratio", "Clifford baseline-band power ratio for the trailing signal window", role="quality_context"),
        _feature("quality_psqi", "pSQI", "Signal quality", "ratio", "Clifford QRS-band power ratio for the trailing signal window", role="quality_context"),
        _feature("quality_bsqi", "bSQI", "Signal quality", "Jaccard", "Li-style agreement between the two independent R-peak detectors at 150 ms", role="quality_context"),
        _feature("quality_qsqi", "qSQI", "Signal quality", "Dice", "Zhao-style detector agreement at 75 ms", role="quality_context"),
        _feature("quality_rr_support", "RR support", "Signal quality", "fraction", "Ho-style fraction of primary RR intervals supported by the comparison detector", role="quality_context"),
        _feature("quality_template_corr", "Beat corr", "Signal quality", "r", "Orphanidou-style within-window beat-template correlation", role="quality_context"),
        _feature("conduction_jt_interval_ms", "JT", "Conduction", "ms", "Automatically delineated QRS-offset to T-offset interval", role="information_only"),
        _feature("conduction_t_peak_to_end_ms", "Tpeak-end", "Conduction", "ms", "Automatically delineated T-peak to T-offset interval", role="information_only"),
        _feature("conduction_qtc_bazett_ms", "QTc B", "Conduction", "ms", "Beatwise Bazett-corrected QT paired with the preceding RR interval", role="information_only"),
        _feature("conduction_qtc_fridericia_ms", "QTc Fd", "Conduction", "ms", "Beatwise Fridericia-corrected QT paired with the preceding RR interval", role="information_only"),
        _feature("conduction_qtc_framingham_ms", "QTc Fh", "Conduction", "ms", "Beatwise Framingham-corrected QT paired with the preceding RR interval", role="information_only"),
        _feature("conduction_delta_pr_ms", "ΔPR", "Conduction", "ms", "Change in automatic PR interval from the preceding beat", role="information_only"),
        _feature("conduction_delta_qt_ms", "ΔQT", "Conduction", "ms", "Change in automatic QT interval from the preceding beat", role="information_only"),
        _feature("conduction_p_to_r_ms", "P→R", "Conduction", "ms", "Diab-aligned within-beat P-peak timing relative to R", role="information_only"),
        _feature("conduction_t_to_r_ms", "T→R", "Conduction", "ms", "Diab-aligned within-beat T-peak timing relative to R", role="information_only"),
        _feature("conduction_qt_median5_ms", "QT med5", "Conduction", "ms", "Median automatic QT over the causal trailing five-beat information window", role="information_only"),
        _feature("conduction_qt_range5_ms", "QT range5", "Conduction", "ms", "Range of automatic QT in the causal trailing five-beat information window", role="information_only"),
        _feature("conduction_qt_slope5_ms_per_beat", "QT slope5", "Conduction", "ms/beat", "Theil-Sen QT slope across the causal trailing five-beat information window", role="information_only"),
        _feature("conduction_qtc_fridericia_mean9_ms", "QTc Fd9", "Conduction", "ms", "Fridericia correction after averaging QT and preceding RR over nine consecutive beats", role="information_only"),
    ]


def _fusion_rows(
    frame: pd.DataFrame,
    *,
    calibrated_features: pd.DataFrame | None = None,
    template_features: pd.DataFrame | None = None,
    pqrst_features: pd.DataFrame | None = None,
    prsa_bprsa_features: pd.DataFrame | None = None,
    quality_features: pd.DataFrame | None = None,
    conduction_beats: pd.DataFrame | None = None,
    conduction_five_beat: pd.DataFrame | None = None,
    conduction_qt_nine: pd.DataFrame | None = None,
    calibration_frozen_at_s: float | None = None,
) -> list[dict[str, Any]]:
    keys = [definition["key"] for definition in feature_definitions()]
    qt_five_beat = (
        conduction_five_beat.loc[
            conduction_five_beat["measurement"] == "qt_interval"
        ].reset_index(drop=True)
        if conduction_five_beat is not None and not conduction_five_beat.empty
        else None
    )
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        values = {key: _finite_or_none(record.get(key)) for key in keys}
        time_s = _finite_or_none(record.get("fusion_time_s"))
        calibrated = _latest_frame_record(
            calibrated_features, "window_end_time_s", time_s
        )
        template = _latest_frame_record(template_features, "anchor_time_s", time_s)
        pqrst = _latest_frame_record(pqrst_features, "r_peak_time_s", time_s)
        prsa_bprsa = _latest_frame_record(
            prsa_bprsa_features, "end_time_s", time_s
        )
        quality = _latest_frame_record(
            quality_features, "window_end_time_s", time_s
        )
        conduction_beat = _latest_frame_record(
            conduction_beats, "r_peak_time_s", time_s
        )
        conduction_five = _latest_frame_record(
            qt_five_beat, "r_peak_time_s", time_s
        )
        conduction_nine = _latest_frame_record(
            conduction_qt_nine, "r_peak_time_s", time_s
        )
        calibration_ready = bool(
            time_s is not None
            and calibration_frozen_at_s is not None
            and time_s >= calibration_frozen_at_s
        )
        for index in range(1, 6):
            values[f"cal_varon_lambda{index}"] = (
                _finite_or_none(calibrated.get(f"lambda{index}"))
                if calibrated is not None and calibration_ready
                else None
            )
        template_ready = bool(template and template.get("evaluation_eligible", False))
        for source_key, target_key in (
            ("template_correlation", "template_correlation"),
            ("template_normalized_rmse", "template_normalized_rmse"),
            ("template_raw_residual_rms", "template_raw_residual_rms"),
            ("template_derivative_dtw_cost", "template_derivative_dtw_cost"),
            ("template_dtw_warp_fraction", "template_dtw_warp_fraction"),
        ):
            values[target_key] = (
                _finite_or_none(template.get(source_key)) if template_ready else None
            )
        for source_key, target_key in (
            ("qrs_duration_ms", "pqrst_qrs_duration_ms"),
            ("pr_interval_ms", "pqrst_pr_interval_ms"),
            ("qt_interval_ms", "pqrst_qt_interval_ms"),
            ("qrs_peak_to_peak_amplitude", "pqrst_qrs_peak_to_peak_amplitude"),
            ("qrs_signed_area", "pqrst_qrs_signed_area"),
            ("landmark_availability_fraction", "pqrst_landmark_availability_fraction"),
        ):
            values[target_key] = (
                _finite_or_none(pqrst.get(source_key)) if pqrst is not None else None
            )
        prsa_defined = bool(
            prsa_bprsa is not None and prsa_bprsa.get("feature_defined", False)
        )
        prsa_reliable = bool(
            prsa_bprsa is not None
            and prsa_bprsa.get("feature_reliable", False)
        )
        for source_key, target_key in (
            ("mean_rr80_ms", "prsa_mean_rr80_ms"),
            ("sdnn80_ms", "prsa_sdnn80_ms"),
            ("prsa_s_rr_ms_per_sample", "prsa_s_rr_ms_per_sample"),
            ("prsa_delta_rr_ms_per_sample", "prsa_delta_rr_ms_per_sample"),
            ("bprsa_s_r_ms_per_sample", "bprsa_s_r_ms_per_sample"),
            ("bprsa_delta_r_ms_per_sample", "bprsa_delta_r_ms_per_sample"),
        ):
            values[target_key] = (
                _finite_or_none(prsa_bprsa.get(source_key)) if prsa_defined else None
            )
        quality_map = (
            ("finite_fraction", "quality_finite_fraction"),
            ("flat_fraction", "quality_flat_fraction"),
            ("bassqi_clifford", "quality_bassqi"),
            ("psqi_clifford", "quality_psqi"),
            ("bsqi_li2008_jaccard_150ms", "quality_bsqi"),
            ("qsqi_zhao2018_dice_75ms", "quality_qsqi"),
            ("rr_support_ho2024_150ms", "quality_rr_support"),
            ("template_corr_orphanidou", "quality_template_corr"),
        )
        for source_key, target_key in quality_map:
            values[target_key] = (
                _finite_or_none(quality.get(source_key))
                if quality is not None
                and bool(quality.get(f"{source_key}__available", False))
                else None
            )
        for source_key, target_key in (
            ("jt_interval_ms", "conduction_jt_interval_ms"),
            ("t_peak_to_end_ms", "conduction_t_peak_to_end_ms"),
            ("qtc_bazett_ms", "conduction_qtc_bazett_ms"),
            ("qtc_fridericia_ms", "conduction_qtc_fridericia_ms"),
            ("qtc_framingham_ms", "conduction_qtc_framingham_ms"),
            ("delta_pr_interval_ms", "conduction_delta_pr_ms"),
            ("delta_qt_interval_ms", "conduction_delta_qt_ms"),
            ("diab_within_p_relative_to_r_ms", "conduction_p_to_r_ms"),
            ("diab_within_t_relative_to_r_ms", "conduction_t_to_r_ms"),
        ):
            values[target_key] = (
                _finite_or_none(conduction_beat.get(source_key))
                if conduction_beat is not None
                else None
            )
        for source_key, target_key in (
            ("median_ms", "conduction_qt_median5_ms"),
            ("range_ms", "conduction_qt_range5_ms"),
            ("theil_sen_slope_ms_per_beat", "conduction_qt_slope5_ms_per_beat"),
        ):
            values[target_key] = (
                _finite_or_none(conduction_five.get(source_key))
                if conduction_five is not None
                and bool(conduction_five.get("summary_defined", False))
                else None
            )
        values["conduction_qtc_fridericia_mean9_ms"] = (
            _finite_or_none(conduction_nine.get("qtc_fridericia_from_means9_ms"))
            if conduction_nine is not None
            and bool(conduction_nine.get("qt_rr_mean9_defined", False))
            else None
        )
        morphology_detail_defined = bool(
            calibration_ready and calibrated is not None and template_ready and pqrst is not None
        )
        calibrated_defined = bool(
            calibrated is not None
            and calibration_ready
            and calibrated.get("published_varon_core_defined", False)
        )
        rows.append(
            {
                "time_s": time_s,
                "values": values,
                "morphology_defined": calibrated_defined,
                "hrv_defined": bool(record.get("hrv_feature_defined", False)),
                "hrv_reliable": bool(record.get("hrv_feature_reliable", False)),
                "prsa_bprsa_defined": prsa_defined,
                "prsa_bprsa_reliable": prsa_reliable,
                "prsa_bprsa_computationally_usable": bool(
                    prsa_defined and prsa_reliable
                ),
                "prsa_bprsa_usable": bool(prsa_defined and prsa_reliable),
                "prsa_bprsa_signal_quality_attached": False,
                "prsa_bprsa_model_eligible": False,
                "prsa_bprsa_role": "autonomic_cardiorespiratory_context_only",
                "prsa_bprsa_affects_core_gate": False,
                "signal_quality_context_available": bool(
                    any(values[target] is not None for _, target in quality_map)
                ),
                "signal_quality_artifact_label_produced": False,
                "signal_quality_affects_core_gate": False,
                "conduction_information_available": bool(
                    conduction_beat is not None
                    and any(
                        values[key] is not None
                        for key in (
                            "conduction_jt_interval_ms",
                            "conduction_qtc_fridericia_ms",
                            "conduction_qt_median5_ms",
                        )
                    )
                ),
                "conduction_final_matrix_eligible": False,
                "conduction_abnormality_label_produced": False,
                "conduction_affects_core_gate": False,
                "measurements_defined": bool(
                    record.get("hrv_feature_defined", False)
                    and morphology_detail_defined
                ),
                "context_pass": bool(record.get("fusion_context_pass", False)),
            }
        )
    return rows


def _latest_frame_record(
    frame: pd.DataFrame | None,
    time_column: str,
    time_s: float | None,
) -> dict[str, Any] | None:
    if frame is None or frame.empty or time_s is None or time_column not in frame:
        return None
    times = frame[time_column].to_numpy(dtype=float)
    index = int(np.searchsorted(times, time_s, side="right") - 1)
    if index < 0:
        return None
    return frame.iloc[index].to_dict()


def _calibration_payload(
    segment: DemoSegment,
    features: pd.DataFrame,
    *,
    requested_beats: int,
    orientation: str,
) -> dict[str, Any]:
    """Serialize the chronological automatic-boundary calibration exercise."""

    onset_column = "qrs_onset_sample_in_segment"
    offset_column = "qrs_offset_sample_in_segment"
    valid = features.loc[
        features["complete_qrs"].astype(bool)
        & features[onset_column].notna()
        & features[offset_column].notna()
    ].copy()
    valid = valid.loc[
        (valid[onset_column].astype(float) <= valid["r_peak_sample_in_segment"])
        & (valid[offset_column].astype(float) >= valid["r_peak_sample_in_segment"])
    ]
    selected = valid.iloc[:requested_beats].copy()
    rate = float(segment.sampling_rate_hz)
    before = (
        1000.0
        * (
            selected["r_peak_sample_in_segment"].to_numpy(dtype=float)
            - selected[onset_column].to_numpy(dtype=float)
        )
        / rate
    )
    after = (
        1000.0
        * (
            selected[offset_column].to_numpy(dtype=float)
            - selected["r_peak_sample_in_segment"].to_numpy(dtype=float)
        )
        / rate
    )
    beats: list[dict[str, Any]] = []
    for row, pre_ms, post_ms in zip(
        selected.to_dict(orient="records"), before.tolist(), after.tolist()
    ):
        onset = int(row[onset_column])
        peak = int(row["r_peak_sample_in_segment"])
        offset = int(row[offset_column])
        beats.append(
            {
                "beat_index": int(row["beat_index"]),
                "onset_time_s": float(segment.start_s + onset / rate),
                "r_time_s": float(segment.start_s + peak / rate),
                "offset_time_s": float(segment.start_s + offset / rate),
                "onset_to_r_ms": float(pre_ms),
                "r_to_offset_ms": float(post_ms),
                "qrs_duration_ms": float(pre_ms + post_ms),
                "qrs_polarity": str(row["qrs_polarity"]),
            }
        )

    visualization = _calibration_waveforms(
        segment,
        selected,
        before,
        after,
        orientation=orientation,
    )
    sufficient = bool(selected.shape[0] == requested_beats and requested_beats >= 5)
    frozen_at = beats[-1]["offset_time_s"] if sufficient else None
    calibrated_pre = float(np.quantile(before, 0.95)) if before.size else None
    calibrated_post = float(np.quantile(after, 0.95)) if after.size else None
    return {
        "requested_beats": int(requested_beats),
        "used_beats": int(selected.shape[0]),
        "eligible_automatic_qrs_beats": int(valid.shape[0]),
        "minimum_beats_for_calibrated_varon": 5,
        "statistic": "separate marginal p95 before and after R",
        "calibrated_pre_r_ms": calibrated_pre,
        "calibrated_post_r_ms": calibrated_post,
        "applied_pre_r_ms": None,
        "applied_post_r_ms": None,
        "calibration_sufficient": sufficient,
        "frozen_at_time_s": frozen_at,
        "evaluation_begins_after_s": frozen_at,
        "causal_rule": (
            "first chronological eligible beats only; freeze once the final "
            "calibration QRS offset is observed; do not update on evaluation beats"
        ),
        "patient_and_lead_specific": True,
        "lead_name": segment.channel,
        "boundary_source": (
            "automatic Emrich-2024 prominence delineation via NeuroKit"
        ),
        "manual_ground_truth": False,
        "ground_truth_warning": (
            "This demonstrates calibration mechanics. MIT-BIH beat annotations "
            "do not provide manual QRS onset/offset boundaries."
        ),
        "onset_to_r_ms": [float(value) for value in before],
        "r_to_offset_ms": [float(value) for value in after],
        "beats": beats,
        "visualization": visualization,
    }


def _calibration_waveforms(
    segment: DemoSegment,
    selected: pd.DataFrame,
    before_ms: np.ndarray,
    after_ms: np.ndarray,
    *,
    orientation: str,
) -> dict[str, Any]:
    if selected.empty:
        return {"waveform_time_ms": [], "waveforms": []}
    rate = float(segment.sampling_rate_hz)
    pre_ms = max(100.0, float(np.max(before_ms)) + 25.0)
    post_ms = max(100.0, float(np.max(after_ms)) + 25.0)
    pre_samples = int(np.ceil(pre_ms * rate / 1000.0))
    post_samples = int(np.ceil(post_ms * rate / 1000.0))
    signal = orient_signal(np.asarray(segment.samples, dtype=np.float64), orientation)
    rows: list[list[float | None]] = []
    for peak in selected["r_peak_sample_in_segment"].to_numpy(dtype=np.int64):
        first = int(peak) - pre_samples
        last = int(peak) + post_samples + 1
        if first < 0 or last > signal.size:
            rows.append([None] * (pre_samples + post_samples + 1))
        else:
            rows.append([_finite_or_none(value) for value in signal[first:last]])
    axis = 1000.0 * np.arange(-pre_samples, post_samples + 1) / rate
    return {
        "waveform_time_ms": np.round(axis, 4).tolist(),
        "waveforms": rows,
        "display_transform": "none; raw oriented lead amplitude is overlaid",
    }


def _run_template_lane(
    segment: DemoSegment,
    anchor_samples: np.ndarray,
    *,
    timestamp_track: str,
    patient_id: str,
    orientation: str,
    calibration_beats: int,
) -> dict[str, Any]:
    complete_cycle_upper_bound = max(0, int(anchor_samples.size) - 2)
    if complete_cycle_upper_bound <= calibration_beats:
        return {
            "available": False,
            "status": (
                f"needs at least {calibration_beats + 1} complete midpoint cycles; "
                f"at most {complete_cycle_upper_bound} are available"
            ),
            "requested_calibration_beats": int(calibration_beats),
            "used_calibration_beats": 0,
            "feature_frame": None,
            "features": [],
            "waveform_phase": [],
            "waveforms": [],
            "template_bank": [],
        }
    try:
        result = extract_patient_template_morphology(
            segment.samples,
            anchor_samples,
            segment.sampling_rate_hz,
            anchor_track=timestamp_track,
            patient_id=patient_id,
            lead_name=segment.channel,
            calibration_beats=calibration_beats,
            segment_start_s=segment.start_s,
            orientation=orientation,
        )
    except ValueError as exc:
        return {
            "available": False,
            "status": str(exc),
            "requested_calibration_beats": int(calibration_beats),
            "used_calibration_beats": 0,
            "feature_frame": None,
            "features": [],
            "waveform_phase": [],
            "waveforms": [],
            "template_bank": [],
        }
    feature_rows = []
    for row in result.features.to_dict(orient="records"):
        feature_rows.append(
            {
                "time_s": _number(row["anchor_time_s"]),
                "waveform_row_index": int(row["waveform_row_index"]),
                "calibration_member": bool(row["calibration_member"]),
                "evaluation_eligible": bool(row["evaluation_eligible"]),
                "matched_template_index": int(row["matched_template_index"]),
                "template_correlation": _finite_or_none(row["template_correlation"]),
                "template_normalized_rmse": _finite_or_none(
                    row["template_normalized_rmse"]
                ),
                "template_raw_residual_rms": _finite_or_none(
                    row["template_raw_residual_rms"]
                ),
                "template_derivative_dtw_cost": _finite_or_none(
                    row["template_derivative_dtw_cost"]
                ),
                "template_dtw_warp_fraction": _finite_or_none(
                    row["template_dtw_warp_fraction"]
                ),
                "previous_beat_normalized_rmse": _finite_or_none(
                    row["previous_beat_normalized_rmse"]
                ),
                "previous_beat_correlation": _finite_or_none(
                    row["previous_beat_correlation"]
                ),
                "cycle_duration_ms": _finite_or_none(row["cycle_duration_ms"]),
                "peak_to_peak_amplitude": _finite_or_none(
                    row["peak_to_peak_amplitude"]
                ),
            }
        )
    pre_points = int(result.parameters["pre_r_points"])
    post_points = int(result.parameters["post_r_points"])
    phase = np.concatenate(
        [
            np.linspace(-1.0, 0.0, pre_points, endpoint=False),
            np.linspace(0.0, 1.0, post_points),
        ]
    )
    return {
        "available": True,
        "status": "implemented deterministic challenger",
        "requested_calibration_beats": int(calibration_beats),
        "used_calibration_beats": int(
            result.features["calibration_member"].sum()
        ),
        "evaluation_beat_count": int(
            result.features["evaluation_eligible"].sum()
        ),
        "frozen_at_time_s": _number(
            result.features.loc[
                result.features["calibration_member"], "anchor_time_s"
            ].iloc[-1]
        ),
        "template_count": int(result.raw_template_bank.shape[0]),
        "template_member_counts": result.template_member_counts.tolist(),
        "boundary_rule": "midpoints between neighboring R anchors",
        "resampling": "pre-R and post-R resampled separately; R stays fixed",
        "normalization": "baseline-centered and unit-L2 only for shape distances",
        "polarity": "preserved; no silent lead inversion",
        "dtw": "derivative DTW, 10% Sakoe–Chiba band",
        "feature_frame": result.features,
        "features": feature_rows,
        "waveform_phase": np.round(phase, 5).tolist(),
        "waveforms": [
            [_finite_or_none(value) for value in row]
            for row in result.beat_waveforms
        ],
        "template_bank": [
            [_finite_or_none(value) for value in row]
            for row in result.raw_template_bank
        ],
    }


def _serialize_varon_features(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "time_s": _number(row["window_end_time_s"]),
            "start_beat_index": int(row["start_beat_index"]),
            "end_beat_index": int(row["end_beat_index"]),
            **{
                f"lambda{index}": _finite_or_none(row[f"lambda{index}"])
                for index in range(1, 6)
            },
            "defined": bool(row["published_varon_core_defined"]),
            "support_pass": bool(row["support_context_pass"]),
        }
        for row in frame.to_dict(orient="records")
    ]


def _serialize_signal_quality(
    frame: pd.DataFrame,
    *,
    input_unit: str,
    microvolts_per_input_unit: float | None,
) -> dict[str, Any]:
    """Expose values and computability separately; never create a quality label."""

    amplitude_features = {
        "absmax_uv",
        "peak_to_peak_uv",
        "rms_uv",
        "hf_rms_uv",
    }
    rows: list[dict[str, Any]] = []
    availability_columns = [
        column for column in frame.columns if column.endswith("__available")
    ]
    feature_names = [column.removesuffix("__available") for column in availability_columns]
    for record in frame.to_dict(orient="records"):
        values: dict[str, Any] = {}
        available: dict[str, bool] = {}
        reasons: dict[str, str] = {}
        for name in feature_names:
            is_available = bool(record.get(f"{name}__available", False))
            reason = str(record.get(f"{name}__reason", "not_calculated"))
            if name in amplitude_features and microvolts_per_input_unit is None:
                is_available = False
                reason = "input_amplitude_unit_unknown"
            values[name] = (
                _finite_or_none(record.get(name)) if is_available else None
            )
            available[name] = is_available
            reasons[name] = reason
        rows.append(
            {
                "time_s": _finite_or_none(record.get("window_end_time_s")),
                "window_start_s": _finite_or_none(record.get("window_start_time_s")),
                "window_end_s": _finite_or_none(record.get("window_end_time_s")),
                "values": values,
                "available": available,
                "reasons": reasons,
                "available_count": int(sum(available.values())),
                "feature_count": int(len(available)),
            }
        )
    return {
        "status": "continuous signal-quality measurements; no clean/artifact decision",
        "window_s": 10.0,
        "step_s": 5.0,
        "window_alignment": "causal trailing; endpoint excludes later samples",
        "input_unit": input_unit,
        "microvolts_per_input_unit": _finite_or_none(microvolts_per_input_unit),
        "amplitude_calibrated": microvolts_per_input_unit is not None,
        "threshold_applied": False,
        "artifact_label_produced": False,
        "rows": rows,
    }


def _serialize_conduction(timing: Any, information: Any) -> dict[str, Any]:
    """Serialize the audit-friendly beat, 5-beat, 9-beat and 30-beat views."""

    beat_columns = (
        "beat_index",
        "r_peak_time_s",
        "preceding_rr_ms",
        "instantaneous_heart_rate_bpm",
        "p_duration_ms",
        "pr_interval_ms",
        "pr_segment_ms",
        "qrs_duration_ms",
        "qt_interval_ms",
        "jt_interval_ms",
        "t_peak_to_end_ms",
        "qtc_bazett_ms",
        "qtc_fridericia_ms",
        "qtc_framingham_ms",
        "delta_pr_interval_ms",
        "delta_qrs_duration_ms",
        "delta_qt_interval_ms",
        "delta_jt_interval_ms",
        "diab_within_p_relative_to_r_ms",
        "diab_within_q_relative_to_r_ms",
        "diab_within_s_relative_to_r_ms",
        "diab_within_t_relative_to_r_ms",
        "diab_interbeat_p_to_p_ms",
        "diab_interbeat_q_to_q_ms",
        "diab_interbeat_r_to_r_ms",
        "diab_interbeat_s_to_s_ms",
        "diab_interbeat_t_to_t_ms",
        "pr_interval_defined",
        "qrs_duration_defined",
        "qt_interval_defined",
        "jt_interval_defined",
        "qtc_defined",
        "diab_all_nine_peak_dynamics_defined",
    )
    rolling_columns = (
        "beat_index",
        "r_peak_time_s",
        "history_beats30",
        "history30_complete",
        "pr_interval_variability30_defined",
        "pr_interval_sd30_ms",
        "qrs_duration_variability30_defined",
        "qrs_duration_sd30_ms",
        "qrs_duration_rmssd30_ms",
        "qt_interval_variability30_defined",
        "qt_interval_sd30_ms",
        "qt_interval_rmssd30_ms",
        "qt_stv30_defined",
        "qt_stv30_ms",
        "exploratory_qtvi_hr30_defined",
        "exploratory_qtvi_hr30",
        "exploratory_qtvi_rr30_defined",
        "exploratory_qtvi_rr30",
        "pr_hr_relation30_defined",
        "pr_vs_hr_slope30_ms_per_bpm",
        "pr_hr_pearson_r30",
    )
    five_columns = (
        "beat_index",
        "r_peak_time_s",
        "measurement",
        "history_beats5",
        "history5_complete",
        "valid_beats",
        "availability_fraction",
        "summary_defined",
        "summary_status",
        "mean_ms",
        "median_ms",
        "sample_sd_ms",
        "range_ms",
        "rmssd_ms",
        "theil_sen_slope_ms_per_beat",
        "late2_minus_early2_median_ms",
        "median_shift_vs_previous_window_ms",
    )
    nine_columns = (
        "beat_index",
        "r_peak_time_s",
        "qt_rr_mean9_defined",
        "qt_interval_mean9_ms",
        "preceding_rr_mean9_ms",
        "qtc_bazett_from_means9_ms",
        "qtc_fridericia_from_means9_ms",
        "qtc_framingham_from_means9_ms",
    )
    return {
        "status": "automatic conduction/repolarization measurements for explanation only",
        "landmark_source": timing.parameters["landmark_method"],
        "manual_ground_truth": False,
        "information_only": True,
        "final_feature_matrix_eligible": False,
        "abnormality_label_produced": False,
        "five_beat_purpose": "slow, local descriptive change summary",
        "nine_beat_purpose": "paper-aligned QT/RR averaging context",
        "thirty_beat_purpose": "trailing variability context; exploratory here",
        "beats": _serialize_frame_columns(information.beat_measurements, beat_columns),
        "five_beat": _serialize_frame_columns(information.information_windows, five_columns),
        "nine_beat": _serialize_frame_columns(information.qt_nine_beat_context, nine_columns),
        "thirty_beat": _serialize_frame_columns(timing.rolling_features, rolling_columns),
        "diagnostics": information.diagnostics,
    }


def _serialize_frame_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> list[dict[str, Any]]:
    present = [column for column in columns if column in frame.columns]
    rows: list[dict[str, Any]] = []
    for record in frame[present].to_dict(orient="records"):
        rows.append({key: _json_scalar(value) for key, value in record.items()})
    return rows


def _json_scalar(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _finite_or_none(value)
    return str(value)


def _serialize_prsa_bprsa(result: Any) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(result.features.to_dict(orient="records")):
        defined = bool(row["feature_defined"])
        rows.append(
            {
                "time_s": _number(row["end_time_s"]),
                "window_beat_count": int(row["window_beat_count"]),
                "window_elapsed_s": _finite_or_none(row["window_elapsed_s"]),
                "mean_rr80_ms": _finite_or_none(row["mean_rr80_ms"]),
                "sdnn80_ms": _finite_or_none(row["sdnn80_ms"]),
                "prsa_s_rr_ms_per_sample": _finite_or_none(
                    row["prsa_s_rr_ms_per_sample"]
                ),
                "prsa_delta_rr_ms_per_sample": _finite_or_none(
                    row["prsa_delta_rr_ms_per_sample"]
                ),
                "bprsa_s_r_ms_per_sample": _finite_or_none(
                    row["bprsa_s_r_ms_per_sample"]
                ),
                "bprsa_delta_r_ms_per_sample": _finite_or_none(
                    row["bprsa_delta_r_ms_per_sample"]
                ),
                "prsa_anchor_count": int(row["prsa_anchor_count"]),
                "bprsa_anchor_count": int(row["bprsa_anchor_count"]),
                "numerical_quality_pass": bool(row["numerical_quality_pass"]),
                "resampled_rr_nonpositive_count": int(
                    row["resampled_rr_nonpositive_count"]
                ),
                "rr_cubic_range_overshoot_fraction": _finite_or_none(
                    row["rr_cubic_range_overshoot_fraction"]
                ),
                "r_amplitude_cubic_range_overshoot_fraction": _finite_or_none(
                    row["r_amplitude_cubic_range_overshoot_fraction"]
                ),
                "rr_coverage": _finite_or_none(row["rr_reliability_coverage"]),
                "r_amplitude_coverage": _finite_or_none(
                    row["r_amplitude_reliability_coverage"]
                ),
                "defined": defined,
                "reliable": bool(row["feature_reliable"]),
                "prsa_curve_ms": (
                    [_finite_or_none(value) for value in result.prsa_curves_ms[index]]
                    if defined
                    else []
                ),
                "bprsa_curve_ms": (
                    [_finite_or_none(value) for value in result.bprsa_curves_ms[index]]
                    if defined
                    else []
                ),
            }
        )
    return {
        "status": "implemented 80-beat Varon-2015 PRSA/BPRSA feature lane",
        "paper_feature_count": 6,
        "calibration_applied": False,
        "classifier_applied": False,
        "curve_time_s": np.round(result.curve_time_s, 6).tolist(),
        "parameters": result.parameters,
        "rows": rows,
    }


def _serialize_prominence(result: Any) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    landmark_names = (
        "p_onset",
        "p_peak",
        "p_offset",
        "qrs_onset",
        "qrs_offset",
        "t_onset",
        "t_peak",
        "t_offset",
    )
    for row in result.features.to_dict(orient="records"):
        rows.append(
            {
                "time_s": _number(row["r_peak_time_s"]),
                "samples": {
                    name: (
                        int(row[f"{name}_sample_in_segment"])
                        if not pd.isna(row.get(f"{name}_sample_in_segment"))
                        else None
                    )
                    for name in landmark_names
                },
                "landmark_availability_fraction": _finite_or_none(
                    row["landmark_availability_fraction"]
                ),
                "complete_qrs": bool(row["complete_qrs"]),
                "complete_pqrst": bool(row["complete_pqrst"]),
                "physiological_order_valid": bool(
                    row["physiological_order_valid"]
                ),
                "p_duration_ms": _finite_or_none(row["p_duration_ms"]),
                "pr_interval_ms": _finite_or_none(row["pr_interval_ms"]),
                "qrs_duration_ms": _finite_or_none(row["qrs_duration_ms"]),
                "st_interval_ms": _finite_or_none(row["st_interval_ms"]),
                "qt_interval_ms": _finite_or_none(row["qt_interval_ms"]),
                "r_to_t_peak_ms": _finite_or_none(row["r_to_t_peak_ms"]),
                "p_peak_amplitude_from_baseline": _finite_or_none(
                    row["p_peak_amplitude_from_baseline"]
                ),
                "r_anchor_amplitude_from_baseline": _finite_or_none(
                    row["r_anchor_amplitude_from_baseline"]
                ),
                "t_peak_amplitude_from_baseline": _finite_or_none(
                    row["t_peak_amplitude_from_baseline"]
                ),
                "qrs_peak_to_peak_amplitude": _finite_or_none(
                    row["qrs_peak_to_peak_amplitude"]
                ),
                "qrs_signed_area": _finite_or_none(row["qrs_signed_area"]),
                "qrs_polarity": str(row["qrs_polarity"]),
            }
        )
    return {
        "available": True,
        "status": "implemented automatic measurement lane",
        "method": result.parameters["method"],
        "reference": result.parameters["reference"],
        "reference_url": result.parameters["reference_url"],
        "manual_ground_truth": False,
        "boundary_source": "automatic prominence delineation",
        "polarity": result.parameters["polarity_handling"],
        "complete_qrs_count": int(result.features["complete_qrs"].sum()),
        "complete_pqrst_count": int(result.features["complete_pqrst"].sum()),
        "rows": rows,
    }


def _feature(
    key: str,
    label: str,
    branch: str,
    unit: str,
    meaning: str,
    *,
    role: str = "core_measurement",
) -> dict[str, str]:
    return {
        "key": key,
        "label": label,
        "branch": branch,
        "unit": unit,
        "meaning": meaning,
        "role": role,
    }


def _plot_signal(segment: DemoSegment, *, maximum_points: int) -> tuple[list[float], list[float]]:
    stride = max(1, int(np.ceil(segment.samples.size / maximum_points)))
    indices = np.arange(0, segment.samples.size, stride, dtype=np.int64)
    times = segment.start_s + indices / segment.sampling_rate_hz
    return (
        np.round(times, 6).tolist(),
        np.round(segment.samples[indices], 7).tolist(),
    )


def _waveform_time_axis_ms(parameters: dict[str, Any]) -> list[float]:
    pre_samples = int(parameters["pre_r_samples"])
    post_samples = int(parameters["post_r_samples"])
    pre_ms = float(parameters["pre_r_ms_realized"])
    post_ms = float(parameters["post_r_ms_realized"])
    if pre_samples + post_samples + 1 <= 1:
        return [0.0]
    return np.round(
        np.linspace(-pre_ms, post_ms, pre_samples + post_samples + 1), 4
    ).tolist()


def _microvolts_per_input_unit(unit: str) -> float | None:
    normalized = str(unit or "").strip().casefold().replace("μ", "u").replace("µ", "u")
    normalized = normalized.replace("microvolts", "uv").replace("microvolt", "uv")
    normalized = normalized.replace("millivolts", "mv").replace("millivolt", "mv")
    normalized = normalized.replace("volts", "v").replace("volt", "v")
    return {"v": 1_000_000.0, "mv": 1_000.0, "uv": 1.0}.get(normalized)


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, sep=None, engine="python")
    except Exception as exc:
        raise ValueError(f"Could not read {path.name} as CSV/TXT: {exc}") from exc
    if frame.empty or len(frame) < 3:
        raise ValueError("The CSV/TXT signal must contain at least three rows.")
    return frame


def _numeric_signal_columns(frame: pd.DataFrame, *, time_column: str | None) -> list[str]:
    columns: list[str] = []
    for column in frame.columns:
        name = str(column)
        if name == time_column:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().all():
            columns.append(name)
    return columns


def _infer_csv_time_axis(frame: pd.DataFrame) -> tuple[str | None, float | None]:
    candidates = [
        str(column)
        for column in frame.columns
        if str(column).strip().casefold()
        in {"time", "time_s", "times", "seconds", "timestamp", "time_ms"}
    ]
    for column in candidates:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(values)) or values.size < 3:
            continue
        differences = np.diff(values)
        if np.any(differences <= 0):
            continue
        step = float(np.median(differences))
        if "ms" in column.casefold():
            step /= 1000.0
        if step > 0 and np.isfinite(step):
            rate = 1.0 / step
            if 0.1 <= rate <= 100000:
                return column, rate
    return None, None


def _validate_sampling_rate(value: float | None, *, allow_none: bool) -> float | None:
    if value is None:
        if allow_none:
            return None
        raise ValueError("Enter the CSV sampling rate in Hz.")
    rate = float(value)
    if not np.isfinite(rate) or not 1 <= rate <= 100000:
        raise ValueError("Sampling rate must be between 1 and 100,000 Hz.")
    return rate


def _finite_or_none(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _number(value: Any) -> float:
    number = _finite_or_none(value)
    if number is None:
        raise ValueError("Expected a finite numeric result from the feature pipeline.")
    return number
