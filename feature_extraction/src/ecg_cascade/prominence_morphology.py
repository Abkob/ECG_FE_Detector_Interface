"""Interpretable P-QRS-T features from deterministic prominence delineation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import neurokit2 as nk
import numpy as np
import pandas as pd

from .delineation_validation import LANDMARK_NAMES, associate_landmarks_with_anchors
from .peaks import orient_signal


_NEUROKIT_KEYS = {
    "p_onset": "ECG_P_Onsets",
    "p_peak": "ECG_P_Peaks",
    "p_offset": "ECG_P_Offsets",
    "qrs_onset": "ECG_R_Onsets",
    "qrs_offset": "ECG_R_Offsets",
    "t_onset": "ECG_T_Onsets",
    "t_peak": "ECG_T_Peaks",
    "t_offset": "ECG_T_Offsets",
}

_OPTIONAL_PEAK_KEYS = {
    "q_peak": "ECG_Q_Peaks",
    "s_peak": "ECG_S_Peaks",
}


@dataclass(frozen=True)
class ProminenceMorphologyResult:
    """Per-beat landmarks and manually interpretable morphology features."""

    anchor_track: str
    landmarks: Mapping[str, np.ndarray]
    features: pd.DataFrame
    parameters: dict[str, Any]

    def summary(self) -> dict[str, object]:
        return {
            "anchor_track": self.anchor_track,
            "beat_count": int(self.features.shape[0]),
            "complete_qrs_count": int(self.features["complete_qrs"].sum()),
            "complete_qs_peak_count": int(self.features["complete_qs_peaks"].sum()),
            "q_s_peak_order_pass_count": int(
                self.features["q_s_peak_order_valid"].sum()
            ),
            "complete_pqrst_count": int(self.features["complete_pqrst"].sum()),
            "physiological_order_pass_count": int(
                self.features["physiological_order_valid"].sum()
            ),
            "parameters": self.parameters,
            "seizure_probability_produced": False,
            "artifact_label_produced": False,
        }


def extract_prominence_morphology(
    ecg: np.ndarray,
    anchor_peak_samples: np.ndarray,
    sampling_rate_hz: float,
    *,
    anchor_track: str,
    patient_id: str,
    lead_name: str,
    segment_start_s: float = 0.0,
    orientation: str = "original",
    cleaning_method: str = "neurokit",
) -> ProminenceMorphologyResult:
    """Run NeuroKit's Emrich-2024 prominence delineator and extract features."""

    signal = _validate_inputs(ecg, anchor_peak_samples, sampling_rate_hz)
    if not str(patient_id).strip() or not str(lead_name).strip():
        raise ValueError("patient_id and lead_name must be non-empty")
    oriented = orient_signal(signal, orientation)
    peaks = np.unique(np.asarray(anchor_peak_samples, dtype=np.int64))
    cleaned = np.asarray(
        nk.ecg_clean(
            oriented,
            sampling_rate=sampling_rate_hz,
            method=cleaning_method,
        ),
        dtype=np.float64,
    )
    _, waves = nk.ecg_delineate(
        cleaned,
        rpeaks=peaks,
        sampling_rate=sampling_rate_hz,
        method="prominence",
        check=False,
    )
    landmarks = _align_neurokit_landmarks(waves, peaks, sampling_rate_hz, signal.size)
    features = build_pqrst_feature_table(
        oriented,
        peaks,
        landmarks,
        sampling_rate_hz,
        anchor_track=anchor_track,
        patient_id=patient_id,
        lead_name=lead_name,
        segment_start_s=segment_start_s,
    )
    parameters: dict[str, Any] = {
        "method": "emrich2024_prominence_via_neurokit2_0_2_13",
        "reference": (
            "Emrich et al. 2024, Physiology-informed ECG delineation based "
            "on peak prominence"
        ),
        "reference_url": (
            "https://eurasip.org/Proceedings/Eusipco/Eusipco2024/"
            "pdfs/0001402.pdf"
        ),
        "patient_id": str(patient_id),
        "lead_name": str(lead_name),
        "anchor_track": anchor_track,
        "r_peaks_are_inputs": True,
        "cleaning_method": cleaning_method,
        "polarity_handling": "preserved_no_silent_inversion",
        "missing_landmarks_retained": True,
        "q_s_peaks_retained_for_paper_aligned_peak_dynamics": True,
        "q_s_peak_manual_validation_status": "pending",
        "matched_lead_gate_result": (
            "QRS accepted for feature extraction; P/T retained with per-beat "
            "availability/order flags and separate reported benchmark limits"
        ),
        "seizure_classifier": "none",
        "artifact_classifier": "none",
    }
    return ProminenceMorphologyResult(
        anchor_track=anchor_track,
        landmarks=landmarks,
        features=features,
        parameters=parameters,
    )


def build_pqrst_feature_table(
    ecg: np.ndarray,
    anchor_peak_samples: np.ndarray,
    landmarks: Mapping[str, np.ndarray],
    sampling_rate_hz: float,
    *,
    anchor_track: str,
    patient_id: str,
    lead_name: str,
    segment_start_s: float = 0.0,
) -> pd.DataFrame:
    """Convert aligned landmark arrays into one auditable row per R anchor."""

    signal = _validate_inputs(ecg, anchor_peak_samples, sampling_rate_hz)
    peaks = np.unique(np.asarray(anchor_peak_samples, dtype=np.int64))
    aligned = _validate_aligned_landmarks(landmarks, peaks.size, signal.size)
    rows: list[dict[str, object]] = []
    for index, r_peak in enumerate(peaks.tolist()):
        values = {
            name: _optional_sample(aligned[name][index])
            for name in LANDMARK_NAMES
            if name != "r_peak"
        }
        values["r_peak"] = int(r_peak)
        q_peak = (
            _optional_sample(aligned["q_peak"][index])
            if "q_peak" in aligned
            else None
        )
        s_peak = (
            _optional_sample(aligned["s_peak"][index])
            if "s_peak" in aligned
            else None
        )
        available_count = sum(
            values[name] is not None for name in LANDMARK_NAMES if name != "r_peak"
        )
        complete_qrs = bool(
            values["qrs_onset"] is not None and values["qrs_offset"] is not None
        )
        complete_p = all(values[name] is not None for name in ("p_onset", "p_peak", "p_offset"))
        complete_t = all(values[name] is not None for name in ("t_onset", "t_peak", "t_offset"))
        complete_pqrst = bool(complete_p and complete_qrs and complete_t)
        ordered_samples = [values.get(name) for name in LANDMARK_NAMES]
        physiological_order_valid = bool(
            complete_pqrst
            and all(
                int(left) <= int(right)
                for left, right in zip(ordered_samples, ordered_samples[1:])
            )
        )
        qrs_onset = values["qrs_onset"]
        qrs_offset = values["qrs_offset"]
        complete_qs_peaks = bool(q_peak is not None and s_peak is not None)
        q_s_peak_order_valid = bool(
            complete_qs_peaks
            and qrs_onset is not None
            and qrs_offset is not None
            and int(qrs_onset) <= int(q_peak) <= int(r_peak)
            and int(r_peak) <= int(s_peak) <= int(qrs_offset)
        )

        baseline = _local_baseline(signal, qrs_onset, r_peak, sampling_rate_hz)
        qrs = (
            signal[int(qrs_onset) : int(qrs_offset) + 1] - baseline
            if complete_qrs and int(qrs_onset) <= int(qrs_offset)
            else np.asarray([], dtype=np.float64)
        )
        positive_excursion = float(np.max(qrs)) if qrs.size else np.nan
        negative_excursion = float(np.min(qrs)) if qrs.size else np.nan
        qrs_polarity = (
            "positive"
            if qrs.size and positive_excursion >= abs(negative_excursion)
            else "negative"
            if qrs.size
            else "missing"
        )
        row: dict[str, object] = {
            "beat_index": index,
            "anchor_track": anchor_track,
            "patient_id": str(patient_id),
            "lead_name": str(lead_name),
            "r_peak_sample_in_segment": int(r_peak),
            "r_peak_time_s": segment_start_s + r_peak / sampling_rate_hz,
            **{
                f"{name}_sample_in_segment": (
                    int(values[name]) if values[name] is not None else pd.NA
                )
                for name in LANDMARK_NAMES
                if name != "r_peak"
            },
            "q_peak_sample_in_segment": int(q_peak) if q_peak is not None else pd.NA,
            "s_peak_sample_in_segment": int(s_peak) if s_peak is not None else pd.NA,
            "landmark_availability_fraction": available_count / 8.0,
            "complete_p": complete_p,
            "complete_qrs": complete_qrs,
            "complete_t": complete_t,
            "complete_pqrst": complete_pqrst,
            "physiological_order_valid": physiological_order_valid,
            "complete_qs_peaks": complete_qs_peaks,
            "q_s_peak_order_valid": q_s_peak_order_valid,
            "p_duration_ms": _duration_ms(values["p_onset"], values["p_offset"], sampling_rate_hz),
            "pr_interval_ms": _duration_ms(values["p_onset"], qrs_onset, sampling_rate_hz),
            "qrs_duration_ms": _duration_ms(qrs_onset, qrs_offset, sampling_rate_hz),
            "st_interval_ms": _duration_ms(qrs_offset, values["t_onset"], sampling_rate_hz),
            "qt_interval_ms": _duration_ms(qrs_onset, values["t_offset"], sampling_rate_hz),
            "r_to_t_peak_ms": _duration_ms(r_peak, values["t_peak"], sampling_rate_hz),
            "baseline_amplitude": baseline,
            "p_peak_amplitude_from_baseline": _amplitude(signal, values["p_peak"], baseline),
            "r_anchor_amplitude_from_baseline": float(signal[r_peak] - baseline),
            "t_peak_amplitude_from_baseline": _amplitude(signal, values["t_peak"], baseline),
            "qrs_peak_to_peak_amplitude": float(np.ptp(qrs)) if qrs.size else np.nan,
            "qrs_signed_area": float(np.trapezoid(qrs) / sampling_rate_hz) if qrs.size else np.nan,
            "qrs_polarity": qrs_polarity,
            "seizure_probability_produced": False,
            "artifact_label_produced": False,
        }
        rows.append(row)
    frame = pd.DataFrame(rows)
    for column in frame.columns:
        if column.endswith("_sample_in_segment") and column != "r_peak_sample_in_segment":
            frame[column] = frame[column].astype("Int64")
    return frame


def _align_neurokit_landmarks(
    waves: Mapping[str, object],
    peaks: np.ndarray,
    sampling_rate_hz: float,
    signal_size: int,
) -> dict[str, np.ndarray]:
    output = {
        name: np.full(peaks.size, np.nan, dtype=np.float64)
        for name in LANDMARK_NAMES
    }
    output["r_peak"] = peaks.astype(np.float64)
    output.update(
        {
            name: np.full(peaks.size, np.nan, dtype=np.float64)
            for name in _OPTIONAL_PEAK_KEYS
        }
    )
    for landmark, key in {**_NEUROKIT_KEYS, **_OPTIONAL_PEAK_KEYS}.items():
        raw = np.asarray(list(waves.get(key, [])), dtype=np.float64)
        if raw.size == peaks.size:
            valid = np.isfinite(raw) & (raw >= 0) & (raw < signal_size)
            output[landmark][valid] = np.rint(raw[valid])
            continue
        finite = raw[np.isfinite(raw) & (raw >= 0) & (raw < signal_size)]
        finite = np.unique(np.rint(finite).astype(np.int64))
        associations = associate_landmarks_with_anchors(
            finite,
            peaks,
            landmark=landmark,
            sampling_rate_hz=sampling_rate_hz,
        )
        for sample, anchor_index in zip(finite.tolist(), associations.tolist()):
            if anchor_index >= 0 and not np.isfinite(output[landmark][anchor_index]):
                output[landmark][anchor_index] = sample
    return output


def _validate_aligned_landmarks(
    landmarks: Mapping[str, np.ndarray],
    beat_count: int,
    signal_size: int,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for name in LANDMARK_NAMES:
        if name == "r_peak":
            continue
        if name not in landmarks:
            raise ValueError(f"landmarks missing {name}")
        values = np.asarray(landmarks[name], dtype=np.float64)
        if values.ndim != 1 or values.size != beat_count:
            raise ValueError(f"{name} must contain one aligned value per beat")
        finite = values[np.isfinite(values)]
        if np.any(finite < 0) or np.any(finite >= signal_size):
            raise ValueError(f"{name} contains out-of-range samples")
        output[name] = values
    for name in _OPTIONAL_PEAK_KEYS:
        if name not in landmarks:
            continue
        values = np.asarray(landmarks[name], dtype=np.float64)
        if values.ndim != 1 or values.size != beat_count:
            raise ValueError(f"{name} must contain one aligned value per beat")
        finite = values[np.isfinite(values)]
        if np.any(finite < 0) or np.any(finite >= signal_size):
            raise ValueError(f"{name} contains out-of-range samples")
        output[name] = values
    return output


def _optional_sample(value: float) -> int | None:
    return int(np.rint(value)) if np.isfinite(value) else None


def _duration_ms(
    start: int | None,
    end: int | None,
    sampling_rate_hz: float,
) -> float:
    if start is None or end is None or end < start:
        return np.nan
    return float(1000.0 * (end - start) / sampling_rate_hz)


def _local_baseline(
    signal: np.ndarray,
    qrs_onset: int | None,
    r_peak: int,
    sampling_rate_hz: float,
) -> float:
    endpoint = int(qrs_onset) if qrs_onset is not None else int(r_peak)
    width = max(1, int(np.rint(40.0 * sampling_rate_hz / 1000.0)))
    start = max(0, endpoint - width)
    return float(np.median(signal[start:endpoint])) if endpoint > start else 0.0


def _amplitude(signal: np.ndarray, sample: int | None, baseline: float) -> float:
    return float(signal[sample] - baseline) if sample is not None else np.nan


def _validate_inputs(
    ecg: np.ndarray,
    anchor_peak_samples: np.ndarray,
    sampling_rate_hz: float,
) -> np.ndarray:
    signal = np.asarray(ecg, dtype=np.float64)
    if signal.ndim != 1 or signal.size == 0 or not np.all(np.isfinite(signal)):
        raise ValueError("ECG must be a non-empty finite one-dimensional array")
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")
    peaks = np.asarray(anchor_peak_samples)
    if peaks.ndim != 1:
        raise ValueError("anchor_peak_samples must be one-dimensional")
    if peaks.size and (np.any(peaks < 0) or np.any(peaks >= signal.size)):
        raise ValueError("anchor_peak_samples contain out-of-range positions")
    return signal
