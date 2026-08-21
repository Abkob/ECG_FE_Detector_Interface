"""Beat-aligned QRS morphology measurements for the ECG cascade.

The primary published seizure-specific core follows Varon et al. (2015):
extract a symmetric 120-ms waveform around every explicit timestamp, stack
five consecutive complexes, form ``Q @ Q.T``, and retain its five ordered
eigenvalues.  The earlier 2013 80-ms window remains reproducible only by
explicitly requesting ``pre_r_ms=40`` and ``post_r_ms=40``.

This module deliberately does not classify a morphology as seizure, artifact,
PVC, or bundle-branch block.  Detector support and template correlation are
carried beside the measurements as context and never delete a waveform.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from .peaks import orient_signal


@dataclass(frozen=True)
class VaronMorphologyResult:
    """One detector track's beat waveforms and five-beat PCA measurements."""

    anchor_track: str
    beat_waveforms: np.ndarray
    beat_context: pd.DataFrame
    features: pd.DataFrame
    parameters: dict[str, Any]

    def summary(self) -> dict[str, object]:
        defined = (
            int(self.features["published_varon_core_defined"].sum())
            if not self.features.empty
            else 0
        )
        support_pass = (
            int(self.features["support_context_pass"].sum())
            if not self.features.empty
            else 0
        )
        return {
            "anchor_track": self.anchor_track,
            "beat_count": int(self.beat_context.shape[0]),
            "complete_beat_waveform_count": int(
                self.beat_context["qrs_capture_complete"].sum()
            ),
            "five_beat_window_count": int(self.features.shape[0]),
            "defined_varon_window_count": defined,
            "support_context_pass_count": support_pass,
            "waveform_shape": list(self.beat_waveforms.shape),
            "parameters": self.parameters,
            "seizure_probability_produced": False,
            "artifact_label_produced": False,
        }


def extract_varon_morphology(
    ecg: np.ndarray,
    anchor_peak_samples: np.ndarray,
    sampling_rate_hz: float,
    *,
    anchor_track: str,
    segment_start_s: float = 0.0,
    orientation: str = "original",
    event_context: pd.DataFrame | None = None,
    pre_r_ms: float = 60.0,
    post_r_ms: float = 60.0,
    stack_beats: int = 5,
    reliable_support_coverage: float = 1.0,
) -> VaronMorphologyResult:
    """Extract the published five-QRS eigenvalue feature sequence.

    ``event_context`` may contain detector-agreement and Zhai-correlation
    fields.  These fields are summarized over each five-beat window but do not
    determine whether the eigenvalues are calculated.

    The paper describes PCA and explicitly shows the matrix ``Q @ Q.T``.  The
    implementation therefore uses that uncentered Gram matrix.  It does not
    add unreported per-beat centering or amplitude normalization.
    """

    signal = _validate_inputs(ecg, anchor_peak_samples, sampling_rate_hz)
    oriented = orient_signal(signal, orientation)
    peaks = np.unique(np.asarray(anchor_peak_samples, dtype=np.int64))
    if pre_r_ms <= 0 or post_r_ms <= 0:
        raise ValueError("pre_r_ms and post_r_ms must be positive")
    if stack_beats != 5:
        raise ValueError(
            "The paper-traceable Varon core requires exactly five beats; "
            "use a separate experimental function for other window sizes"
        )
    if not 0 <= reliable_support_coverage <= 1:
        raise ValueError("reliable_support_coverage must be between 0 and 1")

    symmetric_capture = bool(np.isclose(pre_r_ms, post_r_ms, rtol=0.0, atol=1e-12))
    if symmetric_capture:
        waveform_samples = max(
            3,
            int(np.rint((pre_r_ms + post_r_ms) * sampling_rate_hz / 1000.0)),
        )
        # A sampled symmetric window can have an even number of values.
        # Preserve the published total duration/sample-count rule exactly,
        # include the anchor once, and place the unavoidable one-sample parity
        # difference on the post-anchor side.
        pre_samples = waveform_samples // 2
        post_samples = waveform_samples - pre_samples - 1
        capture_sample_rule = "published_total_length_then_symmetric_split"
    else:
        # The experimental asymmetric branch calibrates each side separately.
        # Outward rounding prevents a requested boundary from being lost merely
        # because it lies between samples.  The fixed 60+60-ms control never
        # enters this path and therefore retains its exact published behavior.
        pre_samples = max(
            1, int(np.ceil(pre_r_ms * sampling_rate_hz / 1000.0))
        )
        post_samples = max(
            1, int(np.ceil(post_r_ms * sampling_rate_hz / 1000.0))
        )
        waveform_samples = pre_samples + post_samples + 1
        capture_sample_rule = "ceil_each_asymmetric_side_and_include_anchor_once"
    waveforms = np.full(
        (peaks.size, waveform_samples), np.nan, dtype=np.float64
    )
    context_by_sample = _event_context_by_sample(event_context)

    beat_rows: list[dict[str, object]] = []
    for beat_index, peak in enumerate(peaks):
        peak = int(peak)
        first = peak - pre_samples
        last_exclusive = peak + post_samples + 1
        complete = bool(first >= 0 and last_exclusive <= oriented.size)
        waveform = np.array([], dtype=np.float64)
        if complete:
            waveform = oriented[first:last_exclusive]
            waveforms[beat_index] = waveform
        context = context_by_sample.get(peak, {})
        beat_rows.append(
            {
                "beat_index": beat_index,
                "anchor_track": anchor_track,
                "anchor_sample_in_segment": peak,
                "anchor_time_s": segment_start_s + peak / sampling_rate_hz,
                "capture_start_sample_in_segment": first,
                "capture_end_sample_exclusive_in_segment": last_exclusive,
                "qrs_capture_complete": complete,
                "waveform_row_index": beat_index,
                "detector_support_available": "qrs_supported" in context,
                "qrs_supported": context.get("qrs_supported", pd.NA),
                "comparator_match_count": context.get(
                    "comparator_match_count", pd.NA
                ),
                "comparator_offset_ms": context.get(
                    "comparator_offset_ms", np.nan
                ),
                "zhai_correlation": context.get("zhai_correlation", np.nan),
                "zhai_absolute_correlation": context.get(
                    "zhai_absolute_correlation", np.nan
                ),
                "reference_symbol": context.get("reference_symbol", pd.NA),
                "reference_match_error_ms": context.get(
                    "reference_match_error_ms", np.nan
                ),
                # These are transparent waveform descriptors for audit/context.
                # They are not claimed as Varon seizure features.
                "exploratory_anchor_amplitude": (
                    float(oriented[peak]) if 0 <= peak < oriented.size else np.nan
                ),
                "exploratory_peak_to_peak_amplitude": (
                    float(np.ptp(waveform)) if complete else np.nan
                ),
                "exploratory_waveform_rms": (
                    float(np.sqrt(np.mean(waveform * waveform)))
                    if complete
                    else np.nan
                ),
                "exploratory_max_abs_slope_per_s": (
                    float(np.max(np.abs(np.diff(waveform))) * sampling_rate_hz)
                    if complete and waveform.size > 1
                    else np.nan
                ),
            }
        )
    beat_context = pd.DataFrame(beat_rows, columns=_BEAT_COLUMNS)
    _apply_nullable_types(beat_context)

    feature_rows: list[dict[str, object]] = []
    for window_index, last_index in enumerate(range(stack_beats - 1, peaks.size)):
        first_index = last_index - stack_beats + 1
        window_context = beat_context.iloc[first_index : last_index + 1]
        window_waveforms = waveforms[first_index : last_index + 1]
        complete_count = int(window_context["qrs_capture_complete"].sum())
        defined = bool(complete_count == stack_beats)
        eigenvalues = np.full(stack_beats, np.nan, dtype=np.float64)
        if defined:
            gram = window_waveforms @ window_waveforms.T
            gram = 0.5 * (gram + gram.T)
            eigenvalues = np.linalg.eigvalsh(gram)[::-1]
            eigenvalues = np.maximum(eigenvalues, 0.0)

        support_available = bool(
            window_context["detector_support_available"].all()
        )
        if support_available:
            support_values = window_context["qrs_supported"].astype(bool)
            supported_count: int | object = int(support_values.sum())
            support_fraction = float(supported_count / stack_beats)
            support_pass: bool | object = bool(
                support_fraction >= reliable_support_coverage
            )
        else:
            supported_count = pd.NA
            support_fraction = np.nan
            support_pass = False

        offsets = pd.to_numeric(
            window_context["comparator_offset_ms"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        correlations = pd.to_numeric(
            window_context["zhai_absolute_correlation"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        finite_offsets = np.abs(offsets[np.isfinite(offsets)])
        finite_correlations = correlations[np.isfinite(correlations)]
        reference_symbols = [
            str(value)
            for value in window_context["reference_symbol"].tolist()
            if not pd.isna(value) and str(value)
        ]
        transition_count = sum(
            left != right
            for left, right in zip(reference_symbols, reference_symbols[1:])
        )
        row: dict[str, object] = {
            "morphology_window_index": window_index,
            "anchor_track": anchor_track,
            "start_beat_index": first_index,
            "end_beat_index": last_index,
            "start_anchor_sample_in_segment": int(peaks[first_index]),
            "end_anchor_sample_in_segment": int(peaks[last_index]),
            "window_start_time_s": float(
                segment_start_s + peaks[first_index] / sampling_rate_hz
            ),
            "window_end_time_s": float(
                segment_start_s + peaks[last_index] / sampling_rate_hz
            ),
            "window_elapsed_s": float(
                (peaks[last_index] - peaks[first_index]) / sampling_rate_hz
            ),
            "five_anchor_samples": "|".join(
                str(int(value)) for value in peaks[first_index : last_index + 1]
            ),
            "beat_count": stack_beats,
            "complete_capture_count": complete_count,
            "complete_capture_fraction": complete_count / stack_beats,
            "qrs_window_truncation_present": bool(complete_count < stack_beats),
            "published_varon_core_defined": defined,
            "detector_support_context_available": support_available,
            "supported_beat_count": supported_count,
            "detector_support_fraction": support_fraction,
            "support_context_pass": support_pass,
            "window_has_detector_disagreement": bool(
                support_available and support_fraction < 1.0
            ),
            "median_absolute_detector_offset_ms": (
                float(np.median(finite_offsets))
                if finite_offsets.size
                else np.nan
            ),
            "maximum_absolute_detector_offset_ms": (
                float(np.max(finite_offsets)) if finite_offsets.size else np.nan
            ),
            "minimum_zhai_absolute_correlation": (
                float(np.min(finite_correlations))
                if finite_correlations.size
                else np.nan
            ),
            "median_zhai_absolute_correlation": (
                float(np.median(finite_correlations))
                if finite_correlations.size
                else np.nan
            ),
            "reference_label_context_available": bool(
                len(reference_symbols) == stack_beats
            ),
            "reference_symbols": "|".join(reference_symbols),
            "reference_symbol_transition_count": int(transition_count),
            "reference_non_n_beat_count": int(
                sum(symbol != "N" for symbol in reference_symbols)
            ),
            "delineation_available": False,
            "delineation_method": "not_run",
            "seizure_probability_produced": False,
            "artifact_label_produced": False,
        }
        row.update(
            {
                f"lambda{component}": float(eigenvalues[component - 1])
                for component in range(1, stack_beats + 1)
            }
        )
        total_energy = float(np.sum(eigenvalues)) if defined else np.nan
        row["eigenvalue_total_energy"] = total_energy
        row["exploratory_lambda1_energy_fraction"] = (
            float(eigenvalues[0] / total_energy)
            if defined and total_energy > 0
            else np.nan
        )
        row["exploratory_lambda3_to_5_energy_fraction"] = (
            float(np.sum(eigenvalues[2:]) / total_energy)
            if defined and total_energy > 0
            else np.nan
        )
        feature_rows.append(row)

    features = pd.DataFrame(feature_rows, columns=_FEATURE_COLUMNS)
    if not features.empty:
        features["supported_beat_count"] = features[
            "supported_beat_count"
        ].astype("Int64")

    is_published_120ms = bool(pre_r_ms == 60.0 and post_r_ms == 60.0)
    parameters: dict[str, Any] = {
        "method": (
            "varon2015_five_qrs_120ms_gram_eigenvalues"
            if is_published_120ms
            else "varon2015_core_with_experimental_capture_window"
        ),
        "reference": (
            "Varon et al. 2015, Detection of epileptic seizures by means "
            "of morphological changes in the ECG"
        ),
        "reference_url": "https://doi.org/10.1016/j.jelectrocard.2015.08.020",
        "pre_r_ms_requested": float(pre_r_ms),
        "post_r_ms_requested": float(post_r_ms),
        "pre_r_samples": pre_samples,
        "post_r_samples": post_samples,
        "pre_r_ms_realized": float(1000.0 * pre_samples / sampling_rate_hz),
        "post_r_ms_realized": float(1000.0 * post_samples / sampling_rate_hz),
        "capture_sample_rule": capture_sample_rule,
        "waveform_sample_count": waveform_samples,
        "stack_beats": stack_beats,
        "matrix": "uncentered_Q_Q_transpose_as_shown_in_paper",
        "eigenvalue_order": "descending",
        "input_preprocessing": (
            "caller_supplied_morphology_signal; Varon reports a 50-Hz notch "
            "but does not specify enough filter details to reproduce it here"
        ),
        "earlier_2013_80ms_ablation": {
            "pre_r_ms": 40.0,
            "post_r_ms": 40.0,
            "enabled": bool(pre_r_ms == 40.0 and post_r_ms == 40.0),
        },
        "five_second_resampling_and_normalization": (
            "not_applied; paper does not fully specify normalization and this "
            "module returns pre-classifier measurements"
        ),
        "delineation": "not_run_separate_future_lane",
        "reliable_support_coverage": float(reliable_support_coverage),
        "seizure_classifier": "none",
        "artifact_classifier": "none",
    }
    return VaronMorphologyResult(
        anchor_track=anchor_track,
        beat_waveforms=waveforms,
        beat_context=beat_context,
        features=features,
        parameters=parameters,
    )


def extract_patient_average_varon_morphology(
    ecg: np.ndarray,
    anchor_peak_samples: np.ndarray,
    sampling_rate_hz: float,
    calibration_qrs_durations_ms: np.ndarray,
    *,
    anchor_track: str,
    patient_id: str,
    lead_name: str,
    segment_start_s: float = 0.0,
    orientation: str = "original",
    event_context: pd.DataFrame | None = None,
    stack_beats: int = 5,
    reliable_support_coverage: float = 1.0,
) -> VaronMorphologyResult:
    """Run the Varon core with one patient-calibrated symmetric QRS width.

    This experimental branch changes only the capture duration.  It computes
    the arithmetic mean of manually or otherwise externally supplied total QRS
    durations and places half of that duration on either side of each R anchor.
    It does not estimate separate onset-to-R or R-to-offset durations, build a
    waveform template, or alter the published five-beat Gram eigenspectrum.

    The caller owns the calibration/test split.  Calibration durations must
    come from data available before the ECG passed in for evaluation when the
    function is used in a causal experiment.
    """

    durations = np.asarray(calibration_qrs_durations_ms, dtype=np.float64)
    if durations.ndim != 1:
        raise ValueError("calibration_qrs_durations_ms must be one-dimensional")
    if durations.size == 0:
        raise ValueError("at least one calibration QRS duration is required")
    if not np.all(np.isfinite(durations)) or np.any(durations <= 0):
        raise ValueError("calibration QRS durations must be finite and positive")
    if not str(patient_id).strip():
        raise ValueError("patient_id must be non-empty")
    if not str(lead_name).strip():
        raise ValueError("lead_name must be non-empty")

    mean_duration_ms = float(np.mean(durations))
    half_duration_ms = mean_duration_ms / 2.0
    result = extract_varon_morphology(
        ecg,
        anchor_peak_samples,
        sampling_rate_hz,
        anchor_track=anchor_track,
        segment_start_s=segment_start_s,
        orientation=orientation,
        event_context=event_context,
        pre_r_ms=half_duration_ms,
        post_r_ms=half_duration_ms,
        stack_beats=stack_beats,
        reliable_support_coverage=reliable_support_coverage,
    )
    parameters = dict(result.parameters)
    parameters.update(
        {
            "method": "patient_average_width_varon_five_qrs_gram_eigenvalues",
            "experimental_branch": True,
            "fixed_120ms_control_modified": False,
            "patient_id": str(patient_id),
            "lead_name": str(lead_name),
            "calibration_parameter": "arithmetic_mean_total_qrs_duration_ms",
            "calibration_qrs_count": int(durations.size),
            "calibration_mean_qrs_duration_ms": mean_duration_ms,
            "calibration_min_qrs_duration_ms": float(np.min(durations)),
            "calibration_max_qrs_duration_ms": float(np.max(durations)),
            "capture_placement": "symmetric_about_R_anchor",
            "pre_r_ms_requested": half_duration_ms,
            "post_r_ms_requested": half_duration_ms,
        }
    )
    return replace(result, parameters=parameters)


def calibrate_patient_symmetric_varon_width(
    calibration_onset_to_r_ms: np.ndarray,
    calibration_r_to_offset_ms: np.ndarray,
    *,
    statistic: str = "p95",
) -> tuple[float, np.ndarray]:
    """Derive one R-centered width that accounts for QRS asymmetry.

    A symmetric window captures one calibration QRS only when its total width
    is at least ``2 * max(onset_to_R, R_to_offset)``.  ``statistic`` selects
    the arithmetic mean, 95th percentile, or maximum of those per-beat required
    widths.  The output remains one millisecond parameter for the patient/lead.
    """

    before = np.asarray(calibration_onset_to_r_ms, dtype=np.float64)
    after = np.asarray(calibration_r_to_offset_ms, dtype=np.float64)
    if before.ndim != 1 or after.ndim != 1 or before.size != after.size:
        raise ValueError("calibration boundary arrays must be one-dimensional and equal")
    if before.size == 0:
        raise ValueError("at least one calibration QRS boundary pair is required")
    if (
        not np.all(np.isfinite(before))
        or not np.all(np.isfinite(after))
        or np.any(before < 0)
        or np.any(after < 0)
    ):
        raise ValueError("calibration boundary distances must be finite and non-negative")
    required_widths = 2.0 * np.maximum(before, after)
    normalized_statistic = str(statistic).casefold()
    if normalized_statistic == "mean":
        width_ms = float(np.mean(required_widths))
    elif normalized_statistic == "p95":
        width_ms = float(np.quantile(required_widths, 0.95))
    elif normalized_statistic == "max":
        width_ms = float(np.max(required_widths))
    else:
        raise ValueError("statistic must be one of: mean, p95, max")
    return width_ms, required_widths


def calibrate_patient_asymmetric_varon_window(
    calibration_onset_to_r_ms: np.ndarray,
    calibration_r_to_offset_ms: np.ndarray,
    *,
    statistic: str = "p95",
) -> tuple[float, float]:
    """Calibrate independent patient/lead-specific pre- and post-R extents.

    Unlike :func:`calibrate_patient_symmetric_varon_width`, this experimental
    rule does not collapse QRS geometry into one number.  It summarizes the
    onset-to-R and R-to-offset calibration distributions independently.  A
    marginal p95 on each side does not mathematically guarantee 95% joint QRS
    capture, so held-out complete-capture performance must still be measured.
    """

    before = np.asarray(calibration_onset_to_r_ms, dtype=np.float64)
    after = np.asarray(calibration_r_to_offset_ms, dtype=np.float64)
    if before.ndim != 1 or after.ndim != 1 or before.size != after.size:
        raise ValueError("calibration boundary arrays must be one-dimensional and equal")
    if before.size == 0:
        raise ValueError("at least one calibration QRS boundary pair is required")
    if (
        not np.all(np.isfinite(before))
        or not np.all(np.isfinite(after))
        or np.any(before < 0)
        or np.any(after < 0)
    ):
        raise ValueError("calibration boundary distances must be finite and non-negative")

    normalized_statistic = str(statistic).casefold()
    if normalized_statistic == "mean":
        summarize = np.mean
    elif normalized_statistic == "p95":
        summarize = lambda values: np.quantile(values, 0.95)
    elif normalized_statistic == "max":
        summarize = np.max
    else:
        raise ValueError("statistic must be one of: mean, p95, max")
    return float(summarize(before)), float(summarize(after))


def extract_patient_asymmetric_varon_morphology(
    ecg: np.ndarray,
    anchor_peak_samples: np.ndarray,
    sampling_rate_hz: float,
    calibration_onset_to_r_ms: np.ndarray,
    calibration_r_to_offset_ms: np.ndarray,
    *,
    anchor_track: str,
    patient_id: str,
    lead_name: str,
    statistic: str = "p95",
    minimum_calibration_beats: int = 5,
    fallback_pre_r_ms: float = 60.0,
    fallback_post_r_ms: float = 60.0,
    segment_start_s: float = 0.0,
    orientation: str = "original",
    event_context: pd.DataFrame | None = None,
    stack_beats: int = 5,
    reliable_support_coverage: float = 1.0,
) -> VaronMorphologyResult:
    """Run the Varon core with independently calibrated pre/post-R extents.

    Calibration is patient- and lead-specific.  When fewer than
    ``minimum_calibration_beats`` are supplied, the function explicitly falls
    back to the requested fixed window.  The caller remains responsible for a
    causal calibration/evaluation split and for the provenance of boundaries.
    """

    before = np.asarray(calibration_onset_to_r_ms, dtype=np.float64)
    after = np.asarray(calibration_r_to_offset_ms, dtype=np.float64)
    calibrated_pre_ms, calibrated_post_ms = (
        calibrate_patient_asymmetric_varon_window(
            before,
            after,
            statistic=statistic,
        )
    )
    if not str(patient_id).strip() or not str(lead_name).strip():
        raise ValueError("patient_id and lead_name must be non-empty")
    if minimum_calibration_beats < 1:
        raise ValueError("minimum_calibration_beats must be positive")
    if fallback_pre_r_ms <= 0 or fallback_post_r_ms <= 0:
        raise ValueError("fallback window extents must be positive")

    fallback_used = bool(before.size < minimum_calibration_beats)
    pre_r_ms = float(fallback_pre_r_ms if fallback_used else calibrated_pre_ms)
    post_r_ms = float(fallback_post_r_ms if fallback_used else calibrated_post_ms)
    result = extract_varon_morphology(
        ecg,
        anchor_peak_samples,
        sampling_rate_hz,
        anchor_track=anchor_track,
        segment_start_s=segment_start_s,
        orientation=orientation,
        event_context=event_context,
        pre_r_ms=pre_r_ms,
        post_r_ms=post_r_ms,
        stack_beats=stack_beats,
        reliable_support_coverage=reliable_support_coverage,
    )
    parameters = dict(result.parameters)
    parameters.update(
        {
            "method": "patient_asymmetric_window_varon_five_qrs",
            "experimental_branch": True,
            "fixed_120ms_control_modified": False,
            "patient_id": str(patient_id),
            "lead_name": str(lead_name),
            "calibration_parameter": (
                f"separate_{str(statistic).casefold()}_onsetR_and_Roffset"
            ),
            "calibration_qrs_count": int(before.size),
            "minimum_calibration_beats": int(minimum_calibration_beats),
            "insufficient_calibration_fallback_used": fallback_used,
            "calibrated_pre_r_ms": calibrated_pre_ms,
            "calibrated_post_r_ms": calibrated_post_ms,
            "fallback_pre_r_ms": float(fallback_pre_r_ms),
            "fallback_post_r_ms": float(fallback_post_r_ms),
            "capture_placement": "patient_lead_specific_asymmetric_about_R_anchor",
            "pre_r_ms_requested": pre_r_ms,
            "post_r_ms_requested": post_r_ms,
        }
    )
    return replace(result, parameters=parameters)


def extract_patient_required_width_varon_morphology(
    ecg: np.ndarray,
    anchor_peak_samples: np.ndarray,
    sampling_rate_hz: float,
    calibration_onset_to_r_ms: np.ndarray,
    calibration_r_to_offset_ms: np.ndarray,
    *,
    anchor_track: str,
    patient_id: str,
    lead_name: str,
    statistic: str = "p95",
    segment_start_s: float = 0.0,
    orientation: str = "original",
    event_context: pd.DataFrame | None = None,
    stack_beats: int = 5,
    reliable_support_coverage: float = 1.0,
) -> VaronMorphologyResult:
    """Run the Varon core with one asymmetry-aware patient width."""

    width_ms, required_widths = calibrate_patient_symmetric_varon_width(
        calibration_onset_to_r_ms,
        calibration_r_to_offset_ms,
        statistic=statistic,
    )
    if not str(patient_id).strip() or not str(lead_name).strip():
        raise ValueError("patient_id and lead_name must be non-empty")
    half_width_ms = width_ms / 2.0
    result = extract_varon_morphology(
        ecg,
        anchor_peak_samples,
        sampling_rate_hz,
        anchor_track=anchor_track,
        segment_start_s=segment_start_s,
        orientation=orientation,
        event_context=event_context,
        pre_r_ms=half_width_ms,
        post_r_ms=half_width_ms,
        stack_beats=stack_beats,
        reliable_support_coverage=reliable_support_coverage,
    )
    parameters = dict(result.parameters)
    parameters.update(
        {
            "method": "patient_required_symmetric_width_varon_five_qrs",
            "experimental_branch": True,
            "fixed_120ms_control_modified": False,
            "patient_id": str(patient_id),
            "lead_name": str(lead_name),
            "calibration_parameter": (
                f"{str(statistic).casefold()}_of_2x_max_onsetR_Roffset"
            ),
            "calibration_qrs_count": int(required_widths.size),
            "calibration_required_width_mean_ms": float(np.mean(required_widths)),
            "calibration_required_width_p95_ms": float(
                np.quantile(required_widths, 0.95)
            ),
            "calibration_required_width_max_ms": float(np.max(required_widths)),
            "calibrated_symmetric_width_ms": width_ms,
            "capture_placement": "symmetric_about_R_anchor",
            "pre_r_ms_requested": half_width_ms,
            "post_r_ms_requested": half_width_ms,
        }
    )
    return replace(result, parameters=parameters)


def _event_context_by_sample(
    event_context: pd.DataFrame | None,
) -> dict[int, dict[str, object]]:
    if event_context is None or event_context.empty:
        return {}
    required = {"primary_timestamp_sample"}
    missing = required.difference(event_context.columns)
    if missing:
        raise ValueError(
            f"event_context is missing required columns: {sorted(missing)}"
        )
    if event_context["primary_timestamp_sample"].duplicated().any():
        raise ValueError("event_context contains duplicate primary timestamps")
    return {
        int(row["primary_timestamp_sample"]): row.to_dict()
        for _, row in event_context.iterrows()
    }


def _apply_nullable_types(frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    frame["qrs_supported"] = frame["qrs_supported"].astype("boolean")
    frame["comparator_match_count"] = frame["comparator_match_count"].astype(
        "Int64"
    )


def _validate_inputs(
    ecg: np.ndarray,
    anchor_peak_samples: np.ndarray,
    sampling_rate_hz: float,
) -> np.ndarray:
    signal = np.asarray(ecg, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("ECG must be a one-dimensional array")
    if signal.size == 0 or not np.all(np.isfinite(signal)):
        raise ValueError("ECG must be non-empty and finite")
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")
    peaks = np.asarray(anchor_peak_samples)
    if peaks.ndim != 1:
        raise ValueError("anchor_peak_samples must be one-dimensional")
    if peaks.size and (
        np.any(peaks < 0) or np.any(peaks >= signal.size)
    ):
        raise ValueError("anchor_peak_samples contain out-of-range positions")
    return signal


_BEAT_COLUMNS = [
    "beat_index",
    "anchor_track",
    "anchor_sample_in_segment",
    "anchor_time_s",
    "capture_start_sample_in_segment",
    "capture_end_sample_exclusive_in_segment",
    "qrs_capture_complete",
    "waveform_row_index",
    "detector_support_available",
    "qrs_supported",
    "comparator_match_count",
    "comparator_offset_ms",
    "zhai_correlation",
    "zhai_absolute_correlation",
    "reference_symbol",
    "reference_match_error_ms",
    "exploratory_anchor_amplitude",
    "exploratory_peak_to_peak_amplitude",
    "exploratory_waveform_rms",
    "exploratory_max_abs_slope_per_s",
]

_FEATURE_COLUMNS = [
    "morphology_window_index",
    "anchor_track",
    "start_beat_index",
    "end_beat_index",
    "start_anchor_sample_in_segment",
    "end_anchor_sample_in_segment",
    "window_start_time_s",
    "window_end_time_s",
    "window_elapsed_s",
    "five_anchor_samples",
    "beat_count",
    "complete_capture_count",
    "complete_capture_fraction",
    "qrs_window_truncation_present",
    "published_varon_core_defined",
    "detector_support_context_available",
    "supported_beat_count",
    "detector_support_fraction",
    "support_context_pass",
    "window_has_detector_disagreement",
    "median_absolute_detector_offset_ms",
    "maximum_absolute_detector_offset_ms",
    "minimum_zhai_absolute_correlation",
    "median_zhai_absolute_correlation",
    "reference_label_context_available",
    "reference_symbols",
    "reference_symbol_transition_count",
    "reference_non_n_beat_count",
    "delineation_available",
    "delineation_method",
    "seizure_probability_produced",
    "artifact_label_produced",
    "lambda1",
    "lambda2",
    "lambda3",
    "lambda4",
    "lambda5",
    "eigenvalue_total_energy",
    "exploratory_lambda1_energy_fraction",
    "exploratory_lambda3_to_5_energy_fraction",
]
