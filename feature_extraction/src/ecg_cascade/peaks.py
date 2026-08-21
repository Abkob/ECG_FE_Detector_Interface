"""R-peak detectors and detector-agreement measurements."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import neurokit2 as nk
import numpy as np
from scipy import signal as scipy_signal


@dataclass(frozen=True)
class PeakDetection:
    method: str
    peak_samples: np.ndarray
    cleaned_ecg: np.ndarray


@dataclass(frozen=True)
class DetectorRun:
    """One detector output with explicit signal orientation and parameters."""

    method: str
    orientation: str
    peak_samples: np.ndarray
    analysis_signal: np.ndarray
    parameters: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, object]:
        return {
            "method": self.method,
            "orientation": self.orientation,
            "peak_count": int(self.peak_samples.size),
            "parameters": self.parameters,
        }


@dataclass(frozen=True)
class PeakAgreement:
    matched_primary_samples: np.ndarray
    matched_comparator_samples: np.ndarray
    unmatched_primary_samples: np.ndarray
    unmatched_comparator_samples: np.ndarray
    agreement_f1: float
    median_timing_difference_ms: float
    unmatched_primary_fraction: float
    unmatched_comparator_fraction: float

    def to_dict(self) -> dict:
        return {
            "agreement_f1": _finite_or_none(self.agreement_f1),
            "median_timing_difference_ms": _finite_or_none(
                self.median_timing_difference_ms
            ),
            "unmatched_primary_fraction": _finite_or_none(
                self.unmatched_primary_fraction
            ),
            "unmatched_comparator_fraction": _finite_or_none(
                self.unmatched_comparator_fraction
            ),
            "matched_count": int(self.matched_primary_samples.size),
            "primary_count": int(
                self.matched_primary_samples.size
                + self.unmatched_primary_samples.size
            ),
            "comparator_count": int(
                self.matched_comparator_samples.size
                + self.unmatched_comparator_samples.size
            ),
            "unmatched_primary_samples": self.unmatched_primary_samples.tolist(),
            "unmatched_comparator_samples": self.unmatched_comparator_samples.tolist(),
        }


def detect_r_peaks(
    ecg: np.ndarray, sampling_rate_hz: float, *, method: str
) -> PeakDetection:
    """Run one named NeuroKit2 cleaning/detection pipeline."""

    signal = np.asarray(ecg, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("ECG must be a one-dimensional array")
    if signal.size < int(3 * sampling_rate_hz):
        raise ValueError("At least three seconds of ECG are required")
    if not np.all(np.isfinite(signal)):
        raise ValueError("ECG contains non-finite samples")

    cleaned = np.asarray(
        nk.ecg_clean(signal, sampling_rate=sampling_rate_hz, method=method),
        dtype=np.float64,
    )
    _, info = nk.ecg_peaks(
        cleaned,
        sampling_rate=sampling_rate_hz,
        method=method,
        correct_artifacts=False,
    )
    peaks = np.asarray(info["ECG_R_Peaks"], dtype=np.int64)
    return PeakDetection(method=method, peak_samples=peaks, cleaned_ecg=cleaned)


def orient_signal(ecg: np.ndarray, orientation: str) -> np.ndarray:
    """Return the requested polarity without choosing it automatically."""

    signal = _validate_ecg(ecg)
    if orientation == "original":
        return signal.copy()
    if orientation == "inverted":
        return -signal
    raise ValueError("orientation must be 'original' or 'inverted'")


def detect_unsw(
    ecg: np.ndarray,
    sampling_rate_hz: float,
    *,
    orientation: str,
) -> DetectorRun:
    """Run the Khamis/UNSW detector distributed with NeuroKit2 0.2.13.

    The UNSW implementation performs its own documented signal cleaning, so
    it receives the oriented raw ECG rather than a separately cleaned copy.
    """

    oriented = orient_signal(ecg, orientation)
    info = nk.ecg_findpeaks(
        oriented,
        sampling_rate=sampling_rate_hz,
        method="khamis2016",
    )
    peaks = np.unique(np.asarray(info["ECG_R_Peaks"], dtype=np.int64))
    return DetectorRun(
        method="khamis2016_unsw",
        orientation=orientation,
        peak_samples=peaks,
        analysis_signal=oriented,
        parameters={"implementation": "NeuroKit2_0.2.13_author_Python_port"},
    )


def detect_neurokit_gradient(
    ecg: np.ndarray,
    sampling_rate_hz: float,
    *,
    orientation: str,
    minimum_delay_ms: float,
    minimum_delay_inclusive: bool,
) -> DetectorRun:
    """Run the NeuroKit gradient detector with an explicit delay comparator.

    NeuroKit's public detector uses a strict ``>`` comparison.  This local
    wrapper reproduces its documented algorithm but exposes ``>=`` as a named
    experiment, which is required to distinguish exactly 250 ms at 240 bpm.
    """

    oriented = orient_signal(ecg, orientation)
    cleaned = np.asarray(
        nk.ecg_clean(oriented, sampling_rate=sampling_rate_hz, method="neurokit"),
        dtype=np.float64,
    )
    peaks = _neurokit_gradient_findpeaks(
        cleaned,
        sampling_rate_hz,
        minimum_delay_ms=minimum_delay_ms,
        inclusive=minimum_delay_inclusive,
    )
    return DetectorRun(
        method="neurokit_gradient",
        orientation=orientation,
        peak_samples=peaks,
        analysis_signal=cleaned,
        parameters={
            "minimum_delay_ms": float(minimum_delay_ms),
            "minimum_delay_samples": int(
                np.rint(minimum_delay_ms * sampling_rate_hz / 1000.0)
            ),
            "comparison": ">=" if minimum_delay_inclusive else ">",
            "cleaning": "NeuroKit2_neurokit",
        },
    )


def detect_pan_tompkins_context(
    ecg: np.ndarray,
    sampling_rate_hz: float,
    *,
    orientation: str,
    minimum_delay_ms: float,
    minimum_delay_inclusive: bool,
) -> DetectorRun:
    """Run the tested NeuroKit Pan--Tompkins path as context, never a veto."""

    oriented = orient_signal(ecg, orientation)
    cleaned = np.asarray(
        nk.ecg_clean(
            oriented,
            sampling_rate=sampling_rate_hz,
            method="pantompkins1985",
        ),
        dtype=np.float64,
    )
    gradient = np.diff(cleaned)
    squared = gradient * gradient
    integration_samples = max(1, int(0.12 * sampling_rate_hz))
    integrated = _moving_window_average(squared, integration_samples)
    integrated[: int(0.2 * sampling_rate_hz)] = 0
    peaks = _adaptive_energy_peakdetect(
        integrated,
        sampling_rate_hz,
        minimum_delay_ms=minimum_delay_ms,
        minimum_delay_inclusive=minimum_delay_inclusive,
        missed_peak_minimum_ms=250.0,
    )
    return DetectorRun(
        method="neurokit_pantompkins1985_port",
        orientation=orientation,
        peak_samples=peaks,
        analysis_signal=cleaned,
        parameters={
            "integration_window_ms": 120.0,
            "minimum_delay_ms": float(minimum_delay_ms),
            "minimum_delay_samples": int(
                np.rint(minimum_delay_ms * sampling_rate_hz / 1000.0)
            ),
            "comparison": ">=" if minimum_delay_inclusive else ">",
            "missed_peak_minimum_ms": 250.0,
            "role": "context_only",
        },
    )


def _neurokit_gradient_findpeaks(
    cleaned: np.ndarray,
    sampling_rate_hz: float,
    *,
    minimum_delay_ms: float,
    inclusive: bool,
) -> np.ndarray:
    smooth_window_s = 0.1
    average_window_s = 0.75
    gradient_threshold_weight = 1.5
    minimum_length_weight = 0.4

    absolute_gradient = np.abs(np.gradient(cleaned))
    smooth_size = max(1, int(np.rint(smooth_window_s * sampling_rate_hz)))
    average_size = max(1, int(np.rint(average_window_s * sampling_rate_hz)))
    smooth_gradient = np.asarray(
        nk.signal_smooth(absolute_gradient, kernel="boxcar", size=smooth_size),
        dtype=np.float64,
    )
    average_gradient = np.asarray(
        nk.signal_smooth(smooth_gradient, kernel="boxcar", size=average_size),
        dtype=np.float64,
    )
    threshold = gradient_threshold_weight * average_gradient
    qrs_mask = smooth_gradient > threshold
    beginnings = np.where((~qrs_mask[:-1]) & qrs_mask[1:])[0]
    endings = np.where(qrs_mask[:-1] & (~qrs_mask[1:]))[0]
    if beginnings.size == 0:
        return np.array([], dtype=np.int64)
    endings = endings[endings > beginnings[0]]
    number = min(beginnings.size, endings.size)
    if number == 0:
        return np.array([], dtype=np.int64)
    minimum_length = (
        np.mean(endings[:number] - beginnings[:number]) * minimum_length_weight
    )
    minimum_delay_samples = int(
        np.rint(minimum_delay_ms * sampling_rate_hz / 1000.0)
    )
    peaks: list[int] = []
    for beginning, ending in zip(beginnings[:number], endings[:number], strict=True):
        if ending - beginning < minimum_length:
            continue
        region = cleaned[beginning:ending]
        local_maxima, properties = scipy_signal.find_peaks(
            region, prominence=(None, None)
        )
        if local_maxima.size == 0:
            continue
        peak = int(beginning + local_maxima[np.argmax(properties["prominences"])])
        if not peaks:
            peaks.append(peak)
            continue
        separation = peak - peaks[-1]
        accepted = (
            separation >= minimum_delay_samples
            if inclusive
            else separation > minimum_delay_samples
        )
        if accepted:
            peaks.append(peak)
    return np.asarray(peaks, dtype=np.int64)


def _moving_window_average(values: np.ndarray, window_size: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values.copy()
    window_size = max(1, int(window_size))
    result = np.convolve(values, np.ones(window_size), mode="same") / window_size
    head_size = min(window_size - 1, values.size)
    if head_size:
        result[:head_size] = np.cumsum(values[:head_size]) / np.arange(
            1, head_size + 1
        )
    return result


def _adaptive_energy_peakdetect(
    detection: np.ndarray,
    sampling_rate_hz: float,
    *,
    minimum_delay_ms: float,
    minimum_delay_inclusive: bool,
    missed_peak_minimum_ms: float,
) -> np.ndarray:
    minimum_distance = int(
        np.rint(minimum_delay_ms * sampling_rate_hz / 1000.0)
    )
    missed_distance = int(
        np.rint(missed_peak_minimum_ms * sampling_rate_hz / 1000.0)
    )
    candidates, _ = scipy_signal.find_peaks(detection, plateau_size=(1, 1))
    signal_peaks: list[int] = []
    signal_level = 0.0
    noise_level = 0.0
    last_peak = 0
    last_candidate_index = -1

    def far_enough(candidate: int, reference: int) -> bool:
        separation = candidate - reference
        return separation >= minimum_distance if minimum_delay_inclusive else separation > minimum_distance

    for index, candidate in enumerate(candidates):
        candidate = int(candidate)
        value = float(detection[candidate])
        threshold_1 = noise_level + 0.25 * (signal_level - noise_level)
        if value > threshold_1 and far_enough(candidate, last_peak):
            signal_peaks.append(candidate)
            if len(signal_peaks) > 9:
                average_rr = (signal_peaks[-2] - signal_peaks[-10]) // 8
                missed_threshold = int(1.66 * average_rr)
                if candidate - last_peak > missed_threshold:
                    missed = candidates[last_candidate_index + 1 : index]
                    missed = missed[
                        (missed > last_peak + missed_distance)
                        & (missed < candidate - missed_distance)
                    ]
                    threshold_2 = 0.5 * threshold_1
                    missed = missed[detection[missed] > threshold_2]
                    if missed.size:
                        signal_peaks[-1] = int(missed[np.argmax(detection[missed])])
                        signal_peaks.append(candidate)
            last_peak = candidate
            last_candidate_index = index
            signal_level = 0.125 * value + 0.875 * signal_level
        else:
            noise_level = 0.125 * value + 0.875 * noise_level
    return np.asarray(signal_peaks, dtype=np.int64)


def _validate_ecg(ecg: np.ndarray) -> np.ndarray:
    signal = np.asarray(ecg, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("ECG must be a one-dimensional array")
    if signal.size == 0:
        raise ValueError("ECG is empty")
    if not np.all(np.isfinite(signal)):
        raise ValueError("ECG contains non-finite samples")
    return signal


def compare_peak_sequences(
    primary_samples: np.ndarray,
    comparator_samples: np.ndarray,
    *,
    sampling_rate_hz: float,
    tolerance_ms: float = 75.0,
) -> PeakAgreement:
    """Monotone one-to-one temporal matching within the published tolerance."""

    primary = np.unique(np.asarray(primary_samples, dtype=np.int64))
    comparator = np.unique(np.asarray(comparator_samples, dtype=np.int64))
    tolerance_samples = int(round(tolerance_ms * sampling_rate_hz / 1000.0))
    if tolerance_samples < 1:
        raise ValueError("Matching tolerance is less than one sample")

    matched_primary: list[int] = []
    matched_comparator: list[int] = []
    unmatched_primary: list[int] = []
    unmatched_comparator: list[int] = []
    i = 0
    j = 0
    while i < primary.size and j < comparator.size:
        delta = int(primary[i] - comparator[j])
        if abs(delta) <= tolerance_samples:
            matched_primary.append(int(primary[i]))
            matched_comparator.append(int(comparator[j]))
            i += 1
            j += 1
        elif delta < 0:
            unmatched_primary.append(int(primary[i]))
            i += 1
        else:
            unmatched_comparator.append(int(comparator[j]))
            j += 1
    unmatched_primary.extend(primary[i:].tolist())
    unmatched_comparator.extend(comparator[j:].tolist())

    mp = np.asarray(matched_primary, dtype=np.int64)
    mc = np.asarray(matched_comparator, dtype=np.int64)
    up = np.asarray(unmatched_primary, dtype=np.int64)
    uc = np.asarray(unmatched_comparator, dtype=np.int64)
    matched_count = mp.size
    primary_count = primary.size
    comparator_count = comparator.size
    denominator = primary_count + comparator_count

    agreement = 2.0 * matched_count / denominator if denominator else np.nan
    timing_ms = (
        float(np.median(np.abs(mp - mc)) * 1000.0 / sampling_rate_hz)
        if matched_count
        else np.nan
    )
    primary_fraction = (
        float(up.size / primary_count) if primary_count else np.nan
    )
    comparator_fraction = (
        float(uc.size / comparator_count) if comparator_count else np.nan
    )
    return PeakAgreement(
        matched_primary_samples=mp,
        matched_comparator_samples=mc,
        unmatched_primary_samples=up,
        unmatched_comparator_samples=uc,
        agreement_f1=float(agreement),
        median_timing_difference_ms=timing_ms,
        unmatched_primary_fraction=primary_fraction,
        unmatched_comparator_fraction=comparator_fraction,
    )


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None
