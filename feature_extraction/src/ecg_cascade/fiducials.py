"""Published and source-audited QRS-to-fiducial adjustment candidates.

These functions deliberately separate QRS-event detection from the later
choice of a raw-ECG timing landmark.  None of the functions below decides
whether a beat is physiological, artifactual, or epileptic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FiducialCandidates:
    """Alternative timestamps calculated from the same QRS candidates.

    ``physiozoo_rqrs_*`` reproduces the polarity-selection and forward-search
    logic of PhysioZoo ``rqrs`` *after* QRS candidates have been supplied.
    Because this project supplies UNSW candidates rather than PhysioZoo's
    gqrs onsets, it is an explicitly named adaptation, not an exact rqrs run.
    """

    ho_positive: np.ndarray
    physiozoo_positive: np.ndarray
    physiozoo_negative: np.ndarray
    physiozoo_rqrs_adapted: np.ndarray
    physiozoo_rqrs_selected_sign: int
    physiozoo_rqrs_median_delta_to_positive: float
    physiozoo_rqrs_median_delta_to_negative: float
    rdeco_positive_backward: np.ndarray
    rdeco_negative_backward: np.ndarray


def ho_positive_refinement(
    peak_samples: np.ndarray,
    ecg: np.ndarray,
    *,
    sampling_rate_hz: float,
    radius_ms: float = 50.0,
) -> np.ndarray:
    """Reproduce the current Ho-author-code positive maximum adjustment.

    The search uses the half-open NumPy interval ``[q-r, q+r)``.  Candidates
    too close to an edge are left unchanged, matching the existing project
    implementation.
    """

    signal, peaks = _validate_inputs(ecg, peak_samples, sampling_rate_hz)
    radius = int(np.rint(radius_ms * sampling_rate_hz / 1000.0))
    if radius < 1:
        raise ValueError("radius_ms is less than one sample")
    refined = np.empty_like(peaks)
    for index, peak in enumerate(peaks):
        first = int(peak) - radius
        last = int(peak) + radius
        if first < 0 or last > signal.size or first >= last:
            refined[index] = peak
        else:
            refined[index] = first + int(np.argmax(signal[first:last]))
    return _require_increasing(refined, "Ho positive refinement")


def physiozoo_qrs_adjust(
    peak_samples: np.ndarray,
    ecg: np.ndarray,
    *,
    sampling_rate_hz: float,
    input_sign: int,
    tolerance_s: float = 0.050,
) -> np.ndarray:
    """Python reproduction of PhysioZoo ``mhrv.ecg.qrs_adjust``.

    ``input_sign=1`` chooses the local maximum and ``input_sign=-1`` chooses
    the local minimum.  The sign is an input supplied by the caller; this
    function does not infer polarity.  PhysioZoo uses ``ceil(fs * tol)`` and
    an inclusive search interval on both sides of every QRS candidate.
    """

    signal, peaks = _validate_inputs(ecg, peak_samples, sampling_rate_hz)
    if input_sign not in {-1, 1}:
        raise ValueError("input_sign must be -1 or 1")
    if tolerance_s <= 0:
        raise ValueError("tolerance_s must be positive")
    radius = int(np.ceil(sampling_rate_hz * tolerance_s))
    adjusted = np.empty_like(peaks)
    for index, peak in enumerate(peaks):
        first = max(0, int(peak) - radius)
        last = min(signal.size - 1, int(peak) + radius)
        local = signal[first : last + 1]
        local_index = int(np.argmax(local) if input_sign > 0 else np.argmin(local))
        adjusted[index] = first + local_index
    return _require_increasing(adjusted, "PhysioZoo qrs_adjust")


def physiozoo_rqrs_polarity_adjust_from_candidates(
    peak_samples: np.ndarray,
    ecg: np.ndarray,
    *,
    sampling_rate_hz: float,
    forward_window_s: float = 0.056,
) -> tuple[np.ndarray, int, float, float]:
    """Apply PhysioZoo ``rqrs``'s global polarity rule to supplied candidates.

    The public rqrs source runs gqrs first, searches forward from each gqrs
    event for both extrema, and chooses one polarity for the whole recording.
    Here the supplied candidates are UNSW events, so the returned result is a
    source-faithful *adaptation* of the post-detection rule, not an rqrs
    replication.  The default 56-ms window is the PhysioZoo default.
    """

    signal, peaks = _validate_inputs(ecg, peak_samples, sampling_rate_hz)
    if forward_window_s <= 0:
        raise ValueError("forward_window_s must be positive")
    window = int(np.ceil(forward_window_s * sampling_rate_hz))
    maxima = np.empty_like(peaks)
    minima = np.empty_like(peaks)
    for index, peak in enumerate(peaks):
        last = min(signal.size - 1, int(peak) + window)
        local = signal[int(peak) : last + 1]
        maxima[index] = int(peak) + int(np.argmax(local))
        minima[index] = int(peak) + int(np.argmin(local))
    candidate_values = signal[peaks]
    delta_positive = float(np.median(np.abs(candidate_values - signal[maxima])))
    delta_negative = float(np.median(np.abs(candidate_values - signal[minima])))
    if delta_positive > delta_negative:
        selected = maxima
        sign = 1
    else:
        selected = minima
        sign = -1
    return (
        _require_increasing(selected, "PhysioZoo rqrs adapted adjustment"),
        sign,
        delta_positive,
        delta_negative,
    )


def rdeco_final_localization(
    peak_samples: np.ndarray,
    ecg: np.ndarray,
    *,
    sampling_rate_hz: float,
    inverted: bool = False,
    backward_window_s: float = 0.065,
) -> np.ndarray:
    """Reproduce R-DECO's final ``peaks_in_ecg`` localization stage.

    This is not the full R-DECO envelope detector or its human GUI.  The
    public MATLAB source looks backward 65 ms and chooses a positive maximum;
    its user-controlled ``inverted`` option flips the entire signal first.
    The function is included only as a named localization ablation.
    """

    signal, peaks = _validate_inputs(ecg, peak_samples, sampling_rate_hz)
    oriented = -signal if inverted else signal
    window = int(np.rint(backward_window_s * sampling_rate_hz))
    if window < 1:
        raise ValueError("backward_window_s is less than one sample")
    adjusted = np.empty_like(peaks)
    for index, peak in enumerate(peaks):
        first = max(0, int(peak) - window)
        local = oriented[first : int(peak) + 1]
        adjusted[index] = first + int(np.argmax(local))
    return _require_increasing(adjusted, "R-DECO final localization")


def build_fiducial_candidates(
    peak_samples: np.ndarray,
    ecg: np.ndarray,
    *,
    sampling_rate_hz: float,
    tolerance_s: float = 0.050,
) -> FiducialCandidates:
    """Calculate every currently audited fiducial hypothesis."""

    ho = ho_positive_refinement(
        peak_samples,
        ecg,
        sampling_rate_hz=sampling_rate_hz,
        radius_ms=tolerance_s * 1000.0,
    )
    positive = physiozoo_qrs_adjust(
        peak_samples,
        ecg,
        sampling_rate_hz=sampling_rate_hz,
        input_sign=1,
        tolerance_s=tolerance_s,
    )
    negative = physiozoo_qrs_adjust(
        peak_samples,
        ecg,
        sampling_rate_hz=sampling_rate_hz,
        input_sign=-1,
        tolerance_s=tolerance_s,
    )
    rqrs, sign, delta_positive, delta_negative = (
        physiozoo_rqrs_polarity_adjust_from_candidates(
            peak_samples,
            ecg,
            sampling_rate_hz=sampling_rate_hz,
        )
    )
    return FiducialCandidates(
        ho_positive=ho,
        physiozoo_positive=positive,
        physiozoo_negative=negative,
        physiozoo_rqrs_adapted=rqrs,
        physiozoo_rqrs_selected_sign=sign,
        physiozoo_rqrs_median_delta_to_positive=delta_positive,
        physiozoo_rqrs_median_delta_to_negative=delta_negative,
        rdeco_positive_backward=rdeco_final_localization(
            peak_samples,
            ecg,
            sampling_rate_hz=sampling_rate_hz,
            inverted=False,
        ),
        rdeco_negative_backward=rdeco_final_localization(
            peak_samples,
            ecg,
            sampling_rate_hz=sampling_rate_hz,
            inverted=True,
        ),
    )


def _validate_inputs(
    ecg: np.ndarray,
    peak_samples: np.ndarray,
    sampling_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(ecg, dtype=np.float64)
    peaks = np.asarray(peak_samples, dtype=np.int64)
    if signal.ndim != 1:
        raise ValueError("ecg must be one-dimensional")
    if peaks.ndim != 1:
        raise ValueError("peak_samples must be one-dimensional")
    if not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")
    if not np.all(np.isfinite(signal)):
        raise ValueError("ecg contains non-finite samples")
    if peaks.size and (peaks[0] < 0 or peaks[-1] >= signal.size):
        raise ValueError("peak sample lies outside ecg")
    if peaks.size > 1 and np.any(np.diff(peaks) <= 0):
        raise ValueError("peak_samples must be strictly increasing")
    return signal, peaks


def _require_increasing(peaks: np.ndarray, label: str) -> np.ndarray:
    peaks = np.asarray(peaks, dtype=np.int64)
    if peaks.size > 1 and np.any(np.diff(peaks) <= 0):
        raise RuntimeError(f"{label} produced non-increasing timestamps")
    return peaks
