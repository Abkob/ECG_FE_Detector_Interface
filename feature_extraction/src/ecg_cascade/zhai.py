"""Paper-traceable reimplementation of Zhai et al. (2023).

The implementation in this module is an independent detector/localizer.  It
does not take NeuroKit or UNSW candidates as inputs and it does not relocate a
candidate to the largest positive ECG sample.  This is important: combining a
NeuroKit candidate generator with only the final Zhai cross-correlation step
would be a new, unvalidated hybrid rather than the published pipeline.

Reference
---------
D. Zhai, X. Bao, X. Long, T. Ru, and G. Zhou, "Precise detection and
localization of R-peaks from ECG signals," Mathematical Biosciences and
Engineering, 20(11), 19191--19208, 2023. doi:10.3934/mbe.2023848.

No official author source code was located.  Consequently, details that the
paper leaves ambiguous are exposed in ``parameters`` and are covered by unit
tests instead of being presented as author code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as scipy_signal

from .peaks import DetectorRun, orient_signal


@dataclass(frozen=True)
class ZhaiDetection:
    """Full intermediate state of one Zhai detector run."""

    detector: DetectorRun
    squared_signal: np.ndarray
    qrs_envelope: np.ndarray
    dynamic_threshold: np.ndarray
    qrs_mask: np.ndarray
    qrs_windows: np.ndarray
    template: np.ndarray
    template_center_sample: int | None
    template_source_window_index: int | None
    correlation_signal: np.ndarray
    correlation_peak_values: np.ndarray

    @property
    def peak_samples(self) -> np.ndarray:
        return self.detector.peak_samples

    def summary(self) -> dict[str, object]:
        payload = self.detector.summary()
        payload.update(
            {
                "qrs_window_count": int(self.qrs_windows.shape[0]),
                "template_sample_count": int(self.template.size),
                "template_center_sample": self.template_center_sample,
                "template_source_window_index": self.template_source_window_index,
                "median_absolute_correlation": (
                    float(np.median(np.abs(self.correlation_peak_values)))
                    if self.correlation_peak_values.size
                    else None
                ),
            }
        )
        return payload


def detect_zhai_template(
    ecg: np.ndarray,
    sampling_rate_hz: float,
    *,
    orientation: str = "original",
) -> ZhaiDetection:
    """Run the complete Zhai 2023 detector and template localizer.

    The defaults reproduce the values reported in the paper:

    * second-order, forward--backward 5--35 Hz Butterworth band-pass;
    * squaring followed by a second-order, forward--backward 5 Hz low-pass;
    * 400-ms threshold-update blocks with a forward 2-s amplitude context;
    * QRS windows widened to at least 200 ms;
    * one 120-ms template selected from the first five QRS windows;
    * maximum *absolute* normalized cross-correlation in each QRS window;
    * rejection of the lower-correlation member when an RR interval is less
      than 0.4 times the current mean RR interval.

    This is an offline algorithm: ``sosfiltfilt`` and the forward 2-s context
    are non-causal.  It is appropriate for the present pre-recorded data, not
    a claim of real-time deployability.
    """

    raw = _validate_signal(ecg, sampling_rate_hz)
    oriented = orient_signal(raw, orientation)

    bandpass = scipy_signal.butter(
        2,
        [5.0, 35.0],
        btype="bandpass",
        fs=sampling_rate_hz,
        output="sos",
    )
    filtered = np.asarray(
        scipy_signal.sosfiltfilt(bandpass, oriented), dtype=np.float64
    )
    squared = filtered * filtered
    lowpass = scipy_signal.butter(
        2,
        5.0,
        btype="lowpass",
        fs=sampling_rate_hz,
        output="sos",
    )
    envelope = np.asarray(
        scipy_signal.sosfiltfilt(lowpass, squared), dtype=np.float64
    )

    threshold, initial_mask = _dynamic_qrs_threshold(
        envelope,
        sampling_rate_hz=sampling_rate_hz,
        update_window_s=0.400,
        amplitude_context_s=2.0,
    )
    initial_windows = _mask_to_windows(initial_mask)
    windows = _paper_window_cleanup(
        initial_windows,
        signal_length=filtered.size,
        sampling_rate_hz=sampling_rate_hz,
        minimum_width_s=0.200,
        duplicate_context_s=0.400,
    )
    final_mask = np.zeros(filtered.size, dtype=bool)
    for first, last in windows:
        final_mask[int(first) : int(last)] = True

    template_length = _odd_sample_count(0.120, sampling_rate_hz)
    template, template_center, source_index = _select_template(
        filtered,
        windows,
        template_length=template_length,
        candidate_count=5,
    )

    if template.size == 0:
        correlation = np.full(filtered.size, np.nan, dtype=np.float64)
        peaks = np.array([], dtype=np.int64)
        peak_values = np.array([], dtype=np.float64)
    else:
        correlation = _normalized_moving_correlation(filtered, template)
        peaks, peak_values = _localize_windows_by_correlation(
            correlation,
            windows,
            half_template=template.size // 2,
        )
        peaks, peak_values = _remove_implausibly_close_peaks(
            peaks,
            peak_values,
            rr_fraction=0.4,
        )

    parameters: dict[str, object] = {
        "implementation": "paper_reimplementation_no_author_code_located",
        "reference_doi": "10.3934/mbe.2023848",
        "filter_order": 2,
        "zero_phase_filtering": True,
        "bandpass_hz": [5.0, 35.0],
        "envelope_lowpass_hz": 5.0,
        "threshold_update_ms": 400.0,
        "threshold_amplitude_context_ms": 2000.0,
        "threshold_amplitude_context_direction": "forward_offline",
        "minimum_qrs_window_ms": 200.0,
        "duplicate_window_context_ms": 400.0,
        "template_duration_ms_requested": 120.0,
        "template_samples_odd": int(template_length),
        "template_candidate_windows": 5,
        "localization": "maximum_absolute_normalized_cross_correlation",
        "raw_amplitude_argmax_refinement": False,
        "short_rr_rule": "remove_lower_abs_correlation_if_RR_below_0.4_mean_RR",
        "role": "independent_detection_and_localization_track",
        "causal": False,
    }
    detector = DetectorRun(
        method="zhai2023_template_matching",
        orientation=orientation,
        peak_samples=peaks,
        analysis_signal=filtered,
        parameters=parameters,
    )
    return ZhaiDetection(
        detector=detector,
        squared_signal=squared,
        qrs_envelope=envelope,
        dynamic_threshold=threshold,
        qrs_mask=final_mask,
        qrs_windows=windows,
        template=template,
        template_center_sample=template_center,
        template_source_window_index=source_index,
        correlation_signal=correlation,
        correlation_peak_values=peak_values,
    )


def _dynamic_qrs_threshold(
    envelope: np.ndarray,
    *,
    sampling_rate_hz: float,
    update_window_s: float,
    amplitude_context_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Implement equations (8)--(9) and the paper's 400-ms updates."""

    block = max(1, int(np.rint(update_window_s * sampling_rate_hz)))
    context = max(block, int(np.rint(amplitude_context_s * sampling_rate_hz)))
    threshold = np.empty(envelope.size, dtype=np.float64)
    mask = np.zeros(envelope.size, dtype=bool)
    running_maximum_mean = 0.0

    for block_index, first in enumerate(range(0, envelope.size, block), start=1):
        last = min(envelope.size, first + block)
        local_maximum = float(np.max(envelope[first:last]))
        running_maximum_mean = (
            local_maximum
            + running_maximum_mean * float(block_index - 1)
        ) / float(block_index)
        context_last = min(envelope.size, first + context)
        context_maximum = float(np.max(envelope[first:context_last]))
        value = max(
            0.3 * local_maximum + 0.1 * running_maximum_mean,
            0.05 * context_maximum,
        )
        threshold[first:last] = value
        mask[first:last] = envelope[first:last] > value
    return threshold, mask


def _mask_to_windows(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    if starts.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.column_stack([starts, ends]).astype(np.int64, copy=False)


def _paper_window_cleanup(
    windows: np.ndarray,
    *,
    signal_length: int,
    sampling_rate_hz: float,
    minimum_width_s: float,
    duplicate_context_s: float,
) -> np.ndarray:
    """Apply the paper's small-window, duplicate-window, and widening rules.

    The paper says to remove the narrower window when another window lies
    within 0.4 s but does not specify a tie rule.  Equal-width ties retain the
    earlier window here; this deterministic interpretation is recorded as a
    reimplementation detail rather than attributed to the authors.
    """

    windows = np.asarray(windows, dtype=np.int64)
    if windows.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    widths = windows[:, 1] - windows[:, 0]
    windows = windows[widths >= 0.25 * float(np.mean(widths))]
    if windows.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    duplicate_samples = int(np.rint(duplicate_context_s * sampling_rate_hz))
    retained: list[np.ndarray] = []
    for window in windows:
        current = window.copy()
        if not retained:
            retained.append(current)
            continue
        previous = retained[-1]
        previous_center = (int(previous[0]) + int(previous[1]) - 1) // 2
        current_center = (int(current[0]) + int(current[1]) - 1) // 2
        if current_center - previous_center < duplicate_samples:
            previous_width = int(previous[1] - previous[0])
            current_width = int(current[1] - current[0])
            if current_width > previous_width:
                retained[-1] = current
        else:
            retained.append(current)

    minimum_samples = max(1, int(np.rint(minimum_width_s * sampling_rate_hz)))
    widened: list[tuple[int, int]] = []
    for first, last in retained:
        first = int(first)
        last = int(last)
        if last - first < minimum_samples:
            center = (first + last - 1) // 2
            first = center - minimum_samples // 2
            last = first + minimum_samples
            if first < 0:
                last -= first
                first = 0
            if last > signal_length:
                first -= last - signal_length
                last = signal_length
                first = max(0, first)
        widened.append((first, last))
    return np.asarray(widened, dtype=np.int64)


def _select_template(
    filtered: np.ndarray,
    windows: np.ndarray,
    *,
    template_length: int,
    candidate_count: int,
) -> tuple[np.ndarray, int | None, int | None]:
    half = template_length // 2
    candidates: list[tuple[int, float, int]] = []
    for window_index, (first, last) in enumerate(windows[:candidate_count]):
        first = int(first)
        last = int(last)
        if first >= last:
            continue
        # The paper selects the initial template by median absolute R amplitude.
        # This absolute extremum is used only to seed one template; it is not
        # repeated as the per-beat final localization rule.
        center = first + int(np.argmax(np.abs(filtered[first:last])))
        if center - half < 0 or center + half + 1 > filtered.size:
            continue
        candidates.append((center, float(abs(filtered[center])), window_index))
    if not candidates:
        return np.array([], dtype=np.float64), None, None
    amplitudes = np.asarray([candidate[1] for candidate in candidates])
    median_amplitude = float(np.median(amplitudes))
    selected = int(np.argmin(np.abs(amplitudes - median_amplitude)))
    center, _, source_index = candidates[selected]
    template = filtered[center - half : center + half + 1].copy()
    return template, int(center), int(source_index)


def _normalized_moving_correlation(
    filtered: np.ndarray,
    template: np.ndarray,
) -> np.ndarray:
    """Pearson correlation for each sample-centered template alignment."""

    template = np.asarray(template, dtype=np.float64)
    centered_template = template - float(np.mean(template))
    template_norm = float(np.sqrt(np.sum(centered_template**2)))
    output = np.full(filtered.size, np.nan, dtype=np.float64)
    if template_norm <= np.finfo(float).eps:
        return output

    numerator = scipy_signal.correlate(
        filtered,
        centered_template,
        mode="same",
        method="fft",
    )
    kernel = np.ones(template.size, dtype=np.float64)
    rolling_sum = np.convolve(filtered, kernel, mode="same")
    rolling_square_sum = np.convolve(filtered * filtered, kernel, mode="same")
    variance_sum = rolling_square_sum - rolling_sum * rolling_sum / template.size
    variance_sum = np.maximum(variance_sum, 0.0)
    denominator = template_norm * np.sqrt(variance_sum)
    valid = denominator > np.finfo(float).eps
    output[valid] = numerator[valid] / denominator[valid]
    half = template.size // 2
    output[:half] = np.nan
    output[filtered.size - half :] = np.nan
    return np.clip(output, -1.0, 1.0)


def _localize_windows_by_correlation(
    correlation: np.ndarray,
    windows: np.ndarray,
    *,
    half_template: int,
) -> tuple[np.ndarray, np.ndarray]:
    peaks: list[int] = []
    values: list[float] = []
    for first, last in windows:
        first = max(int(first), half_template)
        last = min(int(last), correlation.size - half_template)
        if first >= last:
            continue
        local = correlation[first:last]
        finite = np.isfinite(local)
        if not finite.any():
            continue
        score = np.where(finite, np.abs(local), -np.inf)
        peak = first + int(np.argmax(score))
        peaks.append(peak)
        values.append(float(correlation[peak]))
    return np.asarray(peaks, dtype=np.int64), np.asarray(values, dtype=np.float64)


def _remove_implausibly_close_peaks(
    peaks: np.ndarray,
    correlation_values: np.ndarray,
    *,
    rr_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the paper's lower-Cp rejection for unusually short RR pairs."""

    peaks = np.asarray(peaks, dtype=np.int64)
    values = np.asarray(correlation_values, dtype=np.float64)
    if peaks.size < 3:
        return peaks, values
    keep = np.ones(peaks.size, dtype=bool)
    while np.sum(keep) >= 3:
        retained_indices = np.flatnonzero(keep)
        retained_peaks = peaks[retained_indices]
        rr = np.diff(retained_peaks)
        mean_rr = float(np.mean(rr))
        close = np.flatnonzero(rr < rr_fraction * mean_rr)
        if close.size == 0:
            break
        pair_start = int(close[0])
        left_index = int(retained_indices[pair_start])
        right_index = int(retained_indices[pair_start + 1])
        if abs(values[left_index]) >= abs(values[right_index]):
            keep[right_index] = False
        else:
            keep[left_index] = False
    return peaks[keep], values[keep]


def _odd_sample_count(duration_s: float, sampling_rate_hz: float) -> int:
    count = max(3, int(np.rint(duration_s * sampling_rate_hz)))
    if count % 2 == 0:
        count += 1
    return count


def _validate_signal(ecg: np.ndarray, sampling_rate_hz: float) -> np.ndarray:
    signal = np.asarray(ecg, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("ECG must be a one-dimensional array")
    if not np.all(np.isfinite(signal)):
        raise ValueError("ECG contains non-finite samples")
    if sampling_rate_hz <= 70.0:
        raise ValueError("Zhai's 35-Hz low-pass edge requires sampling_rate_hz > 70")
    if signal.size < int(np.ceil(3.0 * sampling_rate_hz)):
        raise ValueError("At least three seconds of ECG are required")
    return signal
