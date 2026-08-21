"""Paper-traceable ECG signal-quality measurements for one analysis window.

The module intentionally returns measurements and availability state, not a
binary clean/artifact decision.  Thresholds that were tuned to a particular
device are accepted only as explicit inputs.  Likewise, ADC-rail, QRS-onset,
and beat-group features remain unavailable when their prerequisites are not
supplied.

The default analysis unit is one 10-second ECG window.  Callers own the window
alignment (centred, trailing/causal, or disjoint) and must record that choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy import interpolate, ndimage, signal as scipy_signal


@dataclass(frozen=True)
class SignalQualityWindowResult:
    """Continuous feature values plus per-feature computability metadata."""

    values: dict[str, float]
    available: dict[str, bool]
    reasons: dict[str, str]
    parameters: dict[str, Any]
    intermediates: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def to_record(self) -> dict[str, object]:
        """Return a flat, serialization-friendly feature/validity record."""

        record: dict[str, object] = {}
        for name, value in self.values.items():
            record[name] = float(value) if np.isfinite(value) else None
            record[f"{name}__available"] = bool(self.available[name])
            record[f"{name}__reason"] = self.reasons[name]
        return record


@dataclass(frozen=True)
class SignalQualitySeriesResult:
    """Ordered trailing-window results with explicit sample boundaries."""

    window_start_samples: np.ndarray
    window_end_samples_exclusive: np.ndarray
    windows: tuple[SignalQualityWindowResult, ...]

    def to_frame(self):
        """Return one flat row per window (pandas imported only on demand)."""

        import pandas as pd

        rows: list[dict[str, object]] = []
        for first, last, result in zip(
            self.window_start_samples,
            self.window_end_samples_exclusive,
            self.windows,
            strict=True,
        ):
            row = {
                "window_start_sample": int(first),
                "window_end_sample_exclusive": int(last),
            }
            row.update(result.to_record())
            rows.append(row)
        return pd.DataFrame(rows)


@dataclass(frozen=True)
class HoRRSupport:
    """Ho-style primary-beat and primary-RR endpoint support."""

    primary_supported: np.ndarray
    rr_supported: np.ndarray
    secondary_count_near_primary: np.ndarray

    @property
    def supported_rr_fraction(self) -> float:
        if self.rr_supported.size == 0:
            return float("nan")
        return float(np.mean(self.rr_supported))


@dataclass(frozen=True)
class GaleottiBaselineResult:
    rms: float
    estimate: np.ndarray
    anchor_samples: np.ndarray
    anchor_values: np.ndarray


@dataclass(frozen=True)
class GaleottiPowerlineResult:
    rms: float
    estimate: np.ndarray


@dataclass(frozen=True)
class GaleottiResidualResult:
    rms: float
    excluded_fraction: float
    used_beat_count: int
    complete_beat_count: int


@dataclass(frozen=True)
class MenonTemplateResult:
    template: np.ndarray
    coefficients: np.ndarray
    converged: bool
    beats_used: int
    final_relative_coefficient_change: float


class _FeatureBuilder:
    def __init__(self) -> None:
        self.values: dict[str, float] = {}
        self.available: dict[str, bool] = {}
        self.reasons: dict[str, str] = {}

    def add(
        self,
        name: str,
        value: float | None = None,
        *,
        reason: str = "calculated",
    ) -> None:
        finite = value is not None and np.isfinite(value)
        self.values[name] = float(value) if finite else float("nan")
        self.available[name] = bool(finite)
        if not finite and reason == "calculated":
            reason = "numerical_result_undefined"
        self.reasons[name] = reason


def extract_trailing_signal_quality_windows(
    ecg: np.ndarray,
    sampling_rate_hz: float,
    *,
    uv_per_input_unit: float,
    window_s: float = 10.0,
    step_s: float = 1.0,
    primary_peak_samples: np.ndarray | None = None,
    secondary_peak_samples: np.ndarray | None = None,
    qrs_onset_samples: np.ndarray | None = None,
    other_lead_peak_samples: Sequence[np.ndarray] | None = None,
    beat_group_labels: np.ndarray | None = None,
    **window_options: Any,
) -> SignalQualitySeriesResult:
    """Apply :func:`extract_signal_quality_window` to causal trailing windows.

    A row timestamp belongs at ``window_end_sample_exclusive``.  No samples at
    or after that boundary enter the row.  Event arrays use recording-relative
    sample numbers and are shifted to each local window before calculation.
    """

    x = np.asarray(ecg, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("ecg must be one-dimensional")
    if sampling_rate_hz <= 0 or window_s <= 0 or step_s <= 0:
        raise ValueError("sampling_rate_hz, window_s, and step_s must be positive")
    window_samples = max(1, int(np.rint(window_s * sampling_rate_hz)))
    step_samples = max(1, int(np.rint(step_s * sampling_rate_hz)))
    if x.size < window_samples:
        return SignalQualitySeriesResult(
            window_start_samples=np.array([], dtype=np.int64),
            window_end_samples_exclusive=np.array([], dtype=np.int64),
            windows=(),
        )

    primary_global = _optional_events(
        primary_peak_samples, x.size, "primary_peak_samples"
    )
    secondary_global = _optional_events(
        secondary_peak_samples, x.size, "secondary_peak_samples"
    )
    onsets_global = _optional_events(qrs_onset_samples, x.size, "qrs_onset_samples")
    other_global = None
    if other_lead_peak_samples is not None:
        other_global = [
            _events(events, x.size, f"other_lead_peak_samples[{index}]")
            for index, events in enumerate(other_lead_peak_samples)
        ]
    labels_global = None
    if beat_group_labels is not None:
        if primary_global is None:
            raise ValueError("beat_group_labels require primary_peak_samples")
        labels_global = np.asarray(beat_group_labels)
        if labels_global.ndim != 1 or labels_global.size != primary_global.size:
            raise ValueError("beat_group_labels must match unique primary peak count")

    starts: list[int] = []
    ends: list[int] = []
    results: list[SignalQualityWindowResult] = []
    for last in range(window_samples, x.size + 1, step_samples):
        first = last - window_samples
        primary_local, primary_keep = _window_events(primary_global, first, last)
        secondary_local, _ = _window_events(secondary_global, first, last)
        onsets_local, _ = _window_events(onsets_global, first, last)
        other_local = None
        if other_global is not None:
            other_local = [
                _window_events(events, first, last)[0] for events in other_global
            ]
        labels_local = None
        if labels_global is not None and primary_keep is not None:
            labels_local = labels_global[primary_keep]
        result = extract_signal_quality_window(
            x[first:last],
            sampling_rate_hz,
            uv_per_input_unit=uv_per_input_unit,
            primary_peak_samples=primary_local,
            secondary_peak_samples=secondary_local,
            qrs_onset_samples=onsets_local,
            other_lead_peak_samples=other_local,
            beat_group_labels=labels_local,
            **window_options,
        )
        starts.append(first)
        ends.append(last)
        results.append(result)
    return SignalQualitySeriesResult(
        window_start_samples=np.asarray(starts, dtype=np.int64),
        window_end_samples_exclusive=np.asarray(ends, dtype=np.int64),
        windows=tuple(results),
    )


def extract_signal_quality_window(
    ecg: np.ndarray,
    sampling_rate_hz: float,
    *,
    uv_per_input_unit: float,
    primary_peak_samples: np.ndarray | None = None,
    secondary_peak_samples: np.ndarray | None = None,
    qrs_onset_samples: np.ndarray | None = None,
    other_lead_peak_samples: Sequence[np.ndarray] | None = None,
    beat_group_labels: np.ndarray | None = None,
    rail_min_input_units: float | None = None,
    rail_max_input_units: float | None = None,
    flat_tolerance_input_units: float = 0.0,
    mains_frequency_hz: float | None = None,
    hf_threshold_uv: float | None = None,
    zhao_tolerance_ms: float = 75.0,
) -> SignalQualityWindowResult:
    """Calculate the combined v0 feature vector for one ECG window.

    ``uv_per_input_unit`` is mandatory so amplitude-valued outputs cannot
    silently mix volts, millivolts, microvolts, or ADC counts.  Peak and onset
    arrays are relative to the first sample of this window.
    """

    x = np.asarray(ecg, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("ecg must be a non-empty one-dimensional array")
    if not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")
    if not np.isfinite(uv_per_input_unit) or uv_per_input_unit <= 0:
        raise ValueError("uv_per_input_unit must be positive")
    if flat_tolerance_input_units < 0:
        raise ValueError("flat_tolerance_input_units must be non-negative")

    fs = float(sampling_rate_hz)
    finite = np.isfinite(x)
    finite_x = x[finite]
    x_uv = x * float(uv_per_input_unit)
    builder = _FeatureBuilder()
    intermediates: dict[str, np.ndarray] = {}

    builder.add("finite_fraction", float(np.mean(finite)))

    if finite_x.size:
        finite_uv = finite_x * uv_per_input_unit
        builder.add("absmax_uv", float(np.max(np.abs(finite_uv))))
        builder.add("peak_to_peak_uv", float(np.ptp(finite_uv)))
        builder.add("rms_uv", float(np.sqrt(np.mean(finite_uv * finite_uv))))
    else:
        for name in ("absmax_uv", "peak_to_peak_uv", "rms_uv"):
            builder.add(name, reason="no_finite_samples")

    # Langley/Di Marco's 12-lead comparator is kept under a source-specific
    # name.  It is not used as a universal limited-lead threshold.
    saturation_mask = _qualified_true_mask(
        finite & (np.abs(x_uv) > 2000.0),
        minimum_samples=int(np.floor(0.200 * fs)) + 1,
    )
    if finite_x.size:
        builder.add(
            "saturation_fraction_langley2011", float(np.mean(saturation_mask))
        )
        builder.add(
            "longest_saturation_run_s_langley2011",
            _longest_true_run(saturation_mask) / fs,
        )
        intermediates["saturation_mask_langley2011"] = saturation_mask
    else:
        builder.add("saturation_fraction_langley2011", reason="no_finite_samples")
        builder.add(
            "longest_saturation_run_s_langley2011", reason="no_finite_samples"
        )

    flat_mask = _qualified_flat_mask(
        x,
        fs,
        tolerance=float(flat_tolerance_input_units),
        minimum_duration_s=1.0,
    )
    if finite_x.size:
        builder.add("flat_fraction", float(np.mean(flat_mask)))
        builder.add("longest_flat_run_s", _longest_true_run(flat_mask) / fs)
        intermediates["flat_mask"] = flat_mask
    else:
        builder.add("flat_fraction", reason="no_finite_samples")
        builder.add("longest_flat_run_s", reason="no_finite_samples")

    if rail_min_input_units is None or rail_max_input_units is None:
        builder.add("rail_fraction", reason="adc_rail_metadata_missing")
        builder.add("longest_rail_run_s", reason="adc_rail_metadata_missing")
    elif not rail_min_input_units < rail_max_input_units:
        raise ValueError("rail_min_input_units must be below rail_max_input_units")
    elif not finite_x.size:
        builder.add("rail_fraction", reason="no_finite_samples")
        builder.add("longest_rail_run_s", reason="no_finite_samples")
    else:
        rail_span = rail_max_input_units - rail_min_input_units
        margin = 0.01 * rail_span
        raw_rail = finite & (
            (x <= rail_min_input_units + margin)
            | (x >= rail_max_input_units - margin)
        )
        dilation_samples = max(1, int(np.rint(fs)))
        rail_mask = ndimage.binary_dilation(
            raw_rail,
            structure=np.ones(2 * dilation_samples + 1, dtype=bool),
        )
        builder.add("rail_fraction", float(np.mean(rail_mask)))
        builder.add("longest_rail_run_s", _longest_true_run(rail_mask) / fs)
        intermediates["raw_rail_mask"] = raw_rail
        intermediates["rail_mask_redmond_dilated_1s"] = rail_mask

    full_finite = bool(np.all(finite))
    if full_finite and x.size >= max(8, int(np.ceil(2 * fs))):
        try:
            freqs, psd = _welch_psd(x_uv, fs)
            builder.add("bassqi_clifford", clifford_baseline_sqi(freqs, psd))
            builder.add("psqi_clifford", clifford_qrs_power_sqi(freqs, psd))
            sdr = li_qrs_power_ratio(freqs, psd)
            builder.add("sdr_li2008", sdr)
            builder.add(
                "ssqi_li2008_binary",
                float(0.5 <= sdr <= 0.8) if np.isfinite(sdr) else None,
                reason="published_0.5_to_0.8_rule" if np.isfinite(sdr) else "zero_band_power",
            )
            intermediates["welch_frequency_hz"] = freqs
            intermediates["welch_psd_uv2_per_hz"] = psd
        except ValueError as exc:
            for name in (
                "bassqi_clifford",
                "psqi_clifford",
                "sdr_li2008",
                "ssqi_li2008_binary",
            ):
                builder.add(name, reason=str(exc))
    else:
        reason = "nonfinite_samples" if not full_finite else "window_shorter_than_2s"
        for name in (
            "bassqi_clifford",
            "psqi_clifford",
            "sdr_li2008",
            "ssqi_li2008_binary",
        ):
            builder.add(name, reason=reason)

    if finite_x.size >= 2 and float(np.std(finite_x)) > 0:
        skew = population_skewness(finite_x)
        kurtosis = pearson_kurtosis(finite_x)
        builder.add("skewness", skew)
        builder.add("pearson_kurtosis", kurtosis)
        builder.add(
            "ksqi_li2008_binary",
            float(kurtosis > 5.0),
            reason="published_pearson_kurtosis_gt_5_rule",
        )
    else:
        for name in ("skewness", "pearson_kurtosis", "ksqi_li2008_binary"):
            builder.add(name, reason="zero_variance_or_too_few_samples")

    if full_finite:
        try:
            hf_envelope = redmond_high_frequency_envelope(x_uv, fs)
            builder.add("hf_rms_uv", float(np.sqrt(np.mean(hf_envelope**2))))
            intermediates["hf_envelope_uv"] = hf_envelope
            if hf_threshold_uv is None:
                builder.add(
                    "hf_contaminated_fraction",
                    reason="device_specific_hf_threshold_not_supplied",
                )
            elif hf_threshold_uv <= 0:
                raise ValueError("hf_threshold_uv must be positive")
            else:
                builder.add(
                    "hf_contaminated_fraction",
                    float(np.mean(hf_envelope > hf_threshold_uv)),
                )
        except ValueError as exc:
            builder.add("hf_rms_uv", reason=str(exc))
            builder.add("hf_contaminated_fraction", reason=str(exc))
    else:
        builder.add("hf_rms_uv", reason="nonfinite_samples")
        builder.add("hf_contaminated_fraction", reason="nonfinite_samples")

    primary = _optional_events(primary_peak_samples, x.size, "primary_peak_samples")
    secondary = _optional_events(
        secondary_peak_samples, x.size, "secondary_peak_samples"
    )
    if primary is None or secondary is None:
        for name in (
            "bsqi_li2008_jaccard_150ms",
            "qsqi_zhao2018_dice_75ms",
            "rr_support_ho2024_150ms",
        ):
            builder.add(name, reason="two_detector_peak_sequences_required")
    else:
        builder.add(
            "bsqi_li2008_jaccard_150ms",
            bsqi_li2008_jaccard(primary, secondary, fs, tolerance_ms=150.0),
            reason="no_detected_events" if primary.size + secondary.size == 0 else "calculated",
        )
        qsqi_name = f"qsqi_zhao2018_dice_{zhao_tolerance_ms:g}ms"
        builder.add(
            qsqi_name,
            qsqi_zhao2018_dice(
                primary,
                secondary,
                fs,
                tolerance_ms=zhao_tolerance_ms,
            ),
            reason="no_detected_events" if primary.size + secondary.size == 0 else "calculated",
        )
        # Keep a stable v0 column when the recommended Kristof comparator
        # tolerance is used.
        if np.isclose(zhao_tolerance_ms, 75.0):
            builder.values["qsqi_zhao2018_dice_75ms"] = builder.values.pop(qsqi_name)
            builder.available["qsqi_zhao2018_dice_75ms"] = builder.available.pop(qsqi_name)
            builder.reasons["qsqi_zhao2018_dice_75ms"] = builder.reasons.pop(qsqi_name)
        ho = rr_support_ho2024(primary, secondary, fs, tolerance_ms=150.0)
        builder.add(
            "rr_support_ho2024_150ms",
            ho.supported_rr_fraction,
            reason="fewer_than_two_primary_events" if primary.size < 2 else "calculated",
        )
        intermediates["ho_primary_supported"] = ho.primary_supported
        intermediates["ho_rr_supported"] = ho.rr_supported

    if primary is None:
        builder.add("template_corr_orphanidou", reason="primary_peaks_required")
        builder.add("menon_fourier_score", reason="primary_peaks_required")
    elif not full_finite:
        builder.add("template_corr_orphanidou", reason="nonfinite_samples")
        builder.add("menon_fourier_score", reason="nonfinite_samples")
    else:
        builder.add(
            "template_corr_orphanidou",
            template_correlation_orphanidou(x_uv, primary),
            reason="insufficient_complete_nonconstant_beats",
        )
        try:
            learned = learn_menon_fourier_template(x_uv, primary, order=5)
            intermediates["menon_fourier_template"] = learned.template
            if not learned.converged:
                builder.add(
                    "menon_fourier_score",
                    reason="fourier_coefficients_did_not_converge_within_5pct",
                )
            elif x.size < int(np.ceil(2 * fs)):
                builder.add("menon_fourier_score", reason="window_shorter_than_2s")
            else:
                builder.add(
                    "menon_fourier_score",
                    menon_fourier_score(x_uv, learned.template, primary.size, fs),
                )
        except ValueError as exc:
            builder.add("menon_fourier_score", reason=str(exc))

    if primary is None or other_lead_peak_samples is None:
        builder.add(
            "isqi_li2008_max_jaccard_150ms",
            reason="synchronized_other_lead_peaks_required",
        )
        builder.add(
            "isqi_primary_unmatched_fraction_at_best_lead",
            reason="synchronized_other_lead_peaks_required",
        )
        builder.add(
            "isqi_other_unmatched_fraction_at_best_lead",
            reason="synchronized_other_lead_peaks_required",
        )
    else:
        agreements: list[tuple[float, float, float]] = []
        tolerance = _tolerance_samples(fs, 150.0)
        for index, events in enumerate(other_lead_peak_samples):
            other = _events(events, x.size, f"other_lead_peak_samples[{index}]")
            matches = monotone_match_count(primary, other, tolerance)
            union = primary.size + other.size - matches
            score = matches / union if union else np.nan
            primary_unmatched = (
                (primary.size - matches) / primary.size if primary.size else np.nan
            )
            other_unmatched = (
                (other.size - matches) / other.size if other.size else np.nan
            )
            agreements.append((float(score), float(primary_unmatched), float(other_unmatched)))
        finite_agreements = [item for item in agreements if np.isfinite(item[0])]
        if not finite_agreements:
            for name in (
                "isqi_li2008_max_jaccard_150ms",
                "isqi_primary_unmatched_fraction_at_best_lead",
                "isqi_other_unmatched_fraction_at_best_lead",
            ):
                builder.add(name, reason="no_detected_events")
        else:
            best = max(finite_agreements, key=lambda item: item[0])
            builder.add("isqi_li2008_max_jaccard_150ms", best[0])
            builder.add("isqi_primary_unmatched_fraction_at_best_lead", best[1])
            builder.add("isqi_other_unmatched_fraction_at_best_lead", best[2])

    onsets = _optional_events(qrs_onset_samples, x.size, "qrs_onset_samples")
    baseline_result: GaleottiBaselineResult | None = None
    if not full_finite:
        builder.add("baseline_rms_galeotti", reason="nonfinite_samples")
    elif onsets is None:
        builder.add("baseline_rms_galeotti", reason="qrs_onsets_required")
    elif mains_frequency_hz not in (50.0, 60.0):
        builder.add(
            "baseline_rms_galeotti",
            reason="mains_frequency_must_be_explicitly_50_or_60_hz",
        )
    else:
        try:
            baseline_result = galeotti_baseline_wander(
                x_uv,
                onsets,
                fs,
                mains_frequency_hz=mains_frequency_hz,
            )
            builder.add("baseline_rms_galeotti", baseline_result.rms)
            intermediates["baseline_estimate_galeotti_uv"] = baseline_result.estimate
        except ValueError as exc:
            builder.add("baseline_rms_galeotti", reason=str(exc))

    powerline_result: GaleottiPowerlineResult | None = None
    if mains_frequency_hz is None:
        builder.add("mains_rms_galeotti", reason="mains_frequency_metadata_missing")
    elif baseline_result is None:
        builder.add("mains_rms_galeotti", reason="baseline_estimate_unavailable")
    elif primary is None:
        builder.add("mains_rms_galeotti", reason="primary_peaks_required")
    else:
        try:
            baseline_removed = x_uv - baseline_result.estimate
            powerline_result = galeotti_powerline(
                baseline_removed,
                primary,
                fs,
                mains_frequency_hz=mains_frequency_hz,
            )
            builder.add("mains_rms_galeotti", powerline_result.rms)
            intermediates["mains_estimate_galeotti_uv"] = powerline_result.estimate
        except ValueError as exc:
            builder.add("mains_rms_galeotti", reason=str(exc))

    if baseline_result is None or powerline_result is None:
        builder.add("residual_rms_galeotti", reason="baseline_and_mains_estimates_required")
        builder.add(
            "dominant_beat_excluded_fraction",
            reason="baseline_and_mains_estimates_required",
        )
    elif primary is None or beat_group_labels is None:
        builder.add(
            "residual_rms_galeotti",
            reason="paper_does_not_freeze_grouping_rule_external_beat_groups_required",
        )
        builder.add(
            "dominant_beat_excluded_fraction",
            reason="paper_does_not_freeze_grouping_rule_external_beat_groups_required",
        )
    else:
        try:
            residual_signal = (
                x_uv - baseline_result.estimate - powerline_result.estimate
            )
            residual = galeotti_residual_noise(
                residual_signal,
                primary,
                beat_group_labels,
            )
            builder.add("residual_rms_galeotti", residual.rms)
            builder.add(
                "dominant_beat_excluded_fraction", residual.excluded_fraction
            )
        except ValueError as exc:
            builder.add("residual_rms_galeotti", reason=str(exc))
            builder.add("dominant_beat_excluded_fraction", reason=str(exc))

    parameters = {
        "feature_set": "signal_quality_v0_paper_named",
        "window_samples": int(x.size),
        "window_duration_s": float(x.size / fs),
        "sampling_rate_hz": fs,
        "uv_per_input_unit": float(uv_per_input_unit),
        "flat_minimum_duration_s": 1.0,
        "flat_tolerance_input_units": float(flat_tolerance_input_units),
        "langley_saturation_threshold_uv": 2000.0,
        "langley_saturation_minimum_duration_rule": "strictly_greater_than_200ms",
        "rail_margin_fraction": 0.01,
        "rail_dilation_s_each_side": 1.0,
        "welch_window": "hann_periodic",
        "welch_segment_s": 4.0,
        "redmond_hf_pipeline": {
            "notch_hz": 50.0,
            "notch_q_project_frozen": 30.0,
            "elliptic_highpass_order": 5,
            "highpass_hz": 40.0,
            "ripple_db_project_frozen": 0.5,
            "stop_attenuation_db_project_frozen": 40.0,
            "smoothing_hamming_ms": 50.0,
        },
        "li_bsqi_tolerance_ms": 150.0,
        "zhao_qsqi_tolerance_ms": float(zhao_tolerance_ms),
        "ho_rr_support_tolerance_ms": 150.0,
        "menon_fourier_order": 5,
        "menon_coefficient_convergence_fraction": 0.05,
        "mains_frequency_hz": mains_frequency_hz,
        "wavelet_features_enabled": False,
        "classifier_score_produced": False,
    }
    return SignalQualityWindowResult(
        values=builder.values,
        available=builder.available,
        reasons=builder.reasons,
        parameters=parameters,
        intermediates=intermediates,
    )


def population_skewness(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    centered = values - np.mean(values)
    scale = float(np.sqrt(np.mean(centered**2)))
    if values.size == 0 or scale == 0 or not np.isfinite(scale):
        return float("nan")
    return float(np.mean((centered / scale) ** 3))


def pearson_kurtosis(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    centered = values - np.mean(values)
    scale = float(np.sqrt(np.mean(centered**2)))
    if values.size == 0 or scale == 0 or not np.isfinite(scale):
        return float("nan")
    return float(np.mean((centered / scale) ** 4))


def clifford_baseline_sqi(frequencies_hz: np.ndarray, psd: np.ndarray) -> float:
    """Return ``1 - P(0--1 Hz) / P(0--40 Hz)``."""

    numerator = _band_power(frequencies_hz, psd, 0.0, 1.0)
    denominator = _band_power(frequencies_hz, psd, 0.0, 40.0)
    return float(1.0 - numerator / denominator) if denominator > 0 else float("nan")


def clifford_qrs_power_sqi(frequencies_hz: np.ndarray, psd: np.ndarray) -> float:
    """Return Clifford's ``P(5--15 Hz) / P(5--40 Hz)``."""

    numerator = _band_power(frequencies_hz, psd, 5.0, 15.0)
    denominator = _band_power(frequencies_hz, psd, 5.0, 40.0)
    return float(numerator / denominator) if denominator > 0 else float("nan")


def li_qrs_power_ratio(frequencies_hz: np.ndarray, psd: np.ndarray) -> float:
    """Return Li's distinct ``P(5--14 Hz) / P(5--50 Hz)`` SDR."""

    numerator = _band_power(frequencies_hz, psd, 5.0, 14.0)
    denominator = _band_power(frequencies_hz, psd, 5.0, 50.0)
    return float(numerator / denominator) if denominator > 0 else float("nan")


def redmond_high_frequency_envelope(ecg_uv: np.ndarray, sampling_rate_hz: float) -> np.ndarray:
    """Return the continuous Redmond-style >40-Hz RMS envelope.

    The source paper gives the filter families/order and smoothing duration.
    Q, ripple, and stop-band attenuation are therefore explicit project-frozen
    transfer parameters recorded by the combined extractor.
    """

    x = np.asarray(ecg_uv, dtype=np.float64)
    fs = float(sampling_rate_hz)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("hf_pipeline_requires_finite_1d_signal")
    if fs <= 100.0:
        raise ValueError("sampling_rate_must_exceed_100hz_for_50hz_notch")
    notch_b, notch_a = scipy_signal.iirnotch(50.0, 30.0, fs=fs)
    sos = scipy_signal.ellip(
        5,
        0.5,
        40.0,
        40.0,
        btype="highpass",
        fs=fs,
        output="sos",
    )
    try:
        filtered = scipy_signal.filtfilt(notch_b, notch_a, x)
        filtered = scipy_signal.sosfiltfilt(sos, filtered)
    except ValueError as exc:
        raise ValueError("window_too_short_for_zero_phase_hf_filters") from exc
    smooth_samples = max(1, int(np.rint(0.050 * fs)))
    kernel = np.hamming(smooth_samples)
    kernel /= np.sum(kernel)
    mean_square = scipy_signal.convolve(filtered**2, kernel, mode="same")
    return np.sqrt(np.maximum(mean_square, 0.0))


def bsqi_li2008_jaccard(
    primary_samples: np.ndarray,
    secondary_samples: np.ndarray,
    sampling_rate_hz: float,
    *,
    tolerance_ms: float = 150.0,
) -> float:
    primary = np.unique(np.asarray(primary_samples, dtype=np.int64))
    secondary = np.unique(np.asarray(secondary_samples, dtype=np.int64))
    matches = monotone_match_count(
        primary, secondary, _tolerance_samples(sampling_rate_hz, tolerance_ms)
    )
    denominator = primary.size + secondary.size - matches
    return float(matches / denominator) if denominator else float("nan")


def qsqi_zhao2018_dice(
    primary_samples: np.ndarray,
    secondary_samples: np.ndarray,
    sampling_rate_hz: float,
    *,
    tolerance_ms: float,
) -> float:
    primary = np.unique(np.asarray(primary_samples, dtype=np.int64))
    secondary = np.unique(np.asarray(secondary_samples, dtype=np.int64))
    matches = monotone_match_count(
        primary, secondary, _tolerance_samples(sampling_rate_hz, tolerance_ms)
    )
    denominator = primary.size + secondary.size
    return float(2.0 * matches / denominator) if denominator else float("nan")


def rr_support_ho2024(
    primary_samples: np.ndarray,
    secondary_samples: np.ndarray,
    sampling_rate_hz: float,
    *,
    tolerance_ms: float = 150.0,
) -> HoRRSupport:
    """Apply Ho's endpoint rule without the project's extra interval-count gate."""

    primary = np.unique(np.asarray(primary_samples, dtype=np.int64))
    secondary = np.unique(np.asarray(secondary_samples, dtype=np.int64))
    tolerance = _tolerance_samples(sampling_rate_hz, tolerance_ms)
    counts = np.zeros(primary.size, dtype=np.int64)
    for index, event in enumerate(primary):
        left = np.searchsorted(secondary, event - tolerance, side="left")
        right = np.searchsorted(secondary, event + tolerance, side="right")
        counts[index] = int(right - left)
    supported = counts == 1
    return HoRRSupport(
        primary_supported=supported,
        rr_supported=supported[:-1] & supported[1:],
        secondary_count_near_primary=counts,
    )


def monotone_match_count(
    primary_samples: np.ndarray,
    secondary_samples: np.ndarray,
    tolerance_samples: int,
) -> int:
    if tolerance_samples < 0:
        raise ValueError("tolerance_samples must be non-negative")
    primary = np.unique(np.asarray(primary_samples, dtype=np.int64))
    secondary = np.unique(np.asarray(secondary_samples, dtype=np.int64))
    i = j = matches = 0
    while i < primary.size and j < secondary.size:
        delta = int(primary[i] - secondary[j])
        if abs(delta) <= tolerance_samples:
            matches += 1
            i += 1
            j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    return matches


def template_correlation_orphanidou(ecg: np.ndarray, peak_samples: np.ndarray) -> float:
    """Average beat-to-mean-template Pearson correlation in a 10-s segment."""

    beats, _ = _median_rr_beats(ecg, peak_samples)
    if beats.shape[0] < 2:
        return float("nan")
    template = np.mean(beats, axis=0)
    template_centered = template - np.mean(template)
    template_norm = float(np.linalg.norm(template_centered))
    if template_norm == 0:
        return float("nan")
    correlations: list[float] = []
    for beat in beats:
        centered = beat - np.mean(beat)
        denominator = float(np.linalg.norm(centered) * template_norm)
        if denominator > 0:
            correlations.append(float(np.dot(centered, template_centered) / denominator))
    return float(np.mean(correlations)) if correlations else float("nan")


def galeotti_baseline_wander(
    ecg: np.ndarray,
    qrs_onset_samples: np.ndarray,
    sampling_rate_hz: float,
    *,
    mains_frequency_hz: float,
) -> GaleottiBaselineResult:
    x = np.asarray(ecg, dtype=np.float64)
    onsets = _events(qrs_onset_samples, x.size, "qrs_onset_samples")
    if mains_frequency_hz == 50.0:
        anchor_duration_s = 0.020
    elif mains_frequency_hz == 60.0:
        anchor_duration_s = 0.016
    else:
        raise ValueError("mains_frequency_must_be_50_or_60_hz")
    anchor_width = max(1, int(np.rint(anchor_duration_s * sampling_rate_hz)))
    valid_onsets = onsets[onsets >= anchor_width]
    if valid_onsets.size < 4:
        raise ValueError("at_least_four_valid_qrs_onsets_required_for_cubic_spline")
    anchors = np.array(
        [np.mean(x[int(onset) - anchor_width : int(onset)]) for onset in valid_onsets],
        dtype=np.float64,
    )
    spline = interpolate.CubicSpline(valid_onsets, anchors, bc_type="natural")
    estimate = np.asarray(spline(np.arange(x.size)), dtype=np.float64)
    return GaleottiBaselineResult(
        rms=float(np.sqrt(np.mean(estimate**2))),
        estimate=estimate,
        anchor_samples=valid_onsets,
        anchor_values=anchors,
    )


def galeotti_powerline(
    baseline_removed_ecg: np.ndarray,
    peak_samples: np.ndarray,
    sampling_rate_hz: float,
    *,
    mains_frequency_hz: float,
) -> GaleottiPowerlineResult:
    x = np.asarray(baseline_removed_ecg, dtype=np.float64)
    peaks = _events(peak_samples, x.size, "peak_samples")
    fs = float(sampling_rate_hz)
    if peaks.size < 2:
        raise ValueError("at_least_two_peaks_required_for_beat_specific_mains_fit")
    if mains_frequency_hz not in (50.0, 60.0):
        raise ValueError("mains_frequency_must_be_50_or_60_hz")
    if fs <= 2.0 * mains_frequency_hz:
        raise ValueError("sampling_rate_below_mains_nyquist_requirement")
    boundaries = np.concatenate(
        ([0], np.rint((peaks[:-1] + peaks[1:]) / 2.0).astype(int), [x.size])
    )
    estimate = np.zeros_like(x)
    omega = 2.0 * np.pi * mains_frequency_hz / fs
    for first, last in zip(boundaries[:-1], boundaries[1:], strict=True):
        first = int(first)
        last = int(last)
        if last - first < 2:
            continue
        indices = np.arange(first, last, dtype=np.float64)
        design = np.column_stack((np.sin(omega * indices), np.cos(omega * indices)))
        coefficients, *_ = np.linalg.lstsq(design, x[first:last], rcond=None)
        estimate[first:last] = design @ coefficients
    return GaleottiPowerlineResult(
        rms=float(np.sqrt(np.mean(estimate**2))),
        estimate=estimate,
    )


def galeotti_residual_noise(
    corrected_ecg: np.ndarray,
    peak_samples: np.ndarray,
    beat_group_labels: np.ndarray,
) -> GaleottiResidualResult:
    """Calculate residual RMS after externally supplied morphology grouping."""

    peaks = np.unique(np.asarray(peak_samples, dtype=np.int64))
    labels = np.asarray(beat_group_labels)
    if labels.ndim != 1 or labels.size != peaks.size:
        raise ValueError("beat_group_labels_must_match_unique_peak_count")
    beats, kept_indices = _median_rr_beats(corrected_ecg, peaks)
    if beats.shape[0] < 2:
        raise ValueError("at_least_two_complete_beats_required")
    complete_labels = labels[kept_indices]
    unique_labels, counts = np.unique(complete_labels, return_counts=True)
    dominant_label = unique_labels[int(np.argmax(counts))]
    dominant = beats[complete_labels == dominant_label]
    if dominant.shape[0] < 2:
        raise ValueError("at_least_two_dominant_group_beats_required")
    template = np.median(dominant, axis=0)
    per_beat_rms = np.sqrt(np.mean((dominant - template) ** 2, axis=1))
    return GaleottiResidualResult(
        rms=float(np.mean(per_beat_rms)),
        excluded_fraction=float(1.0 - dominant.shape[0] / beats.shape[0]),
        used_beat_count=int(dominant.shape[0]),
        complete_beat_count=int(beats.shape[0]),
    )


def learn_menon_fourier_template(
    ecg: np.ndarray,
    peak_samples: np.ndarray,
    *,
    order: int = 5,
    convergence_fraction: float = 0.05,
) -> MenonTemplateResult:
    beats, _ = _median_rr_beats(ecg, peak_samples)
    if beats.shape[0] < 2:
        raise ValueError("at_least_two_complete_beats_required_for_fourier_template")
    previous: np.ndarray | None = None
    final_change = float("inf")
    converged = False
    coefficients = np.array([], dtype=np.float64)
    used = beats.shape[0]
    for count in range(1, beats.shape[0] + 1):
        representative = np.mean(beats[:count], axis=0)
        coefficients, fitted = _fourier_fit(representative, order)
        if previous is not None:
            denominator = max(float(np.linalg.norm(previous)), np.finfo(float).eps)
            final_change = float(np.linalg.norm(coefficients - previous) / denominator)
            if count >= 3 and final_change <= convergence_fraction:
                converged = True
                used = count
                return MenonTemplateResult(
                    template=fitted,
                    coefficients=coefficients,
                    converged=True,
                    beats_used=count,
                    final_relative_coefficient_change=final_change,
                )
        previous = coefficients
    representative = np.mean(beats, axis=0)
    coefficients, fitted = _fourier_fit(representative, order)
    return MenonTemplateResult(
        template=fitted,
        coefficients=coefficients,
        converged=converged,
        beats_used=used,
        final_relative_coefficient_change=final_change,
    )


def menon_fourier_score(
    ecg: np.ndarray,
    fourier_template: np.ndarray,
    detected_qrs_count: int,
    sampling_rate_hz: float,
) -> float:
    x = np.asarray(ecg, dtype=np.float64)
    template = np.asarray(fourier_template, dtype=np.float64)
    if x.ndim != 1 or template.ndim != 1 or template.size == 0:
        raise ValueError("ecg_and_template_must_be_nonempty_1d_arrays")
    if detected_qrs_count <= 0:
        return float("nan")
    if x.size < int(np.ceil(2 * sampling_rate_hz)):
        raise ValueError("segment_must_be_at_least_2s")
    correlation = scipy_signal.correlate(x, template, mode="same", method="fft")
    return float(np.trapezoid(correlation**2, dx=1.0 / sampling_rate_hz) / detected_qrs_count)


def _welch_psd(values: np.ndarray, sampling_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    if sampling_rate_hz / 2.0 < 50.0:
        raise ValueError("sampling_rate_nyquist_below_50hz_published_band")
    nperseg = min(values.size, max(8, int(np.rint(4.0 * sampling_rate_hz))))
    frequencies, psd = scipy_signal.welch(
        values,
        fs=sampling_rate_hz,
        window="hann_periodic",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
    )
    return np.asarray(frequencies), np.asarray(psd)


def _band_power(
    frequencies_hz: np.ndarray,
    psd: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> float:
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    power = np.asarray(psd, dtype=np.float64)
    if frequencies.ndim != 1 or power.shape != frequencies.shape:
        raise ValueError("frequencies_and_psd_must_be_matching_1d_arrays")
    if frequencies.size < 2 or frequencies[0] > low_hz or frequencies[-1] < high_hz:
        raise ValueError("psd_does_not_cover_requested_band")
    inside = (frequencies > low_hz) & (frequencies < high_hz)
    band_f = np.concatenate(([low_hz], frequencies[inside], [high_hz]))
    band_p = np.concatenate(
        ([np.interp(low_hz, frequencies, power)], power[inside], [np.interp(high_hz, frequencies, power)])
    )
    return float(np.trapezoid(band_p, band_f))


def _qualified_flat_mask(
    values: np.ndarray,
    sampling_rate_hz: float,
    *,
    tolerance: float,
    minimum_duration_s: float,
) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    mask = np.zeros(x.size, dtype=bool)
    if x.size == 0:
        return mask
    finite = np.isfinite(x)
    same = finite[:-1] & finite[1:] & (np.abs(np.diff(x)) <= tolerance)
    minimum_samples = max(1, int(np.ceil(minimum_duration_s * sampling_rate_hz)))
    start = 0
    while start < x.size:
        end = start + 1
        while end < x.size and same[end - 1]:
            end += 1
        if end - start >= minimum_samples:
            mask[start:end] = True
        start = end
    return mask


def _longest_true_run(mask: np.ndarray) -> int:
    values = np.asarray(mask, dtype=bool)
    if values.size == 0 or not np.any(values):
        return 0
    padded = np.concatenate(([False], values, [False])).astype(np.int8)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return int(np.max(ends - starts))


def _qualified_true_mask(mask: np.ndarray, *, minimum_samples: int) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    qualified = np.zeros(values.size, dtype=bool)
    if minimum_samples <= 0:
        raise ValueError("minimum_samples must be positive")
    padded = np.concatenate(([False], values, [False])).astype(np.int8)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    for first, last in zip(starts, ends, strict=True):
        if last - first >= minimum_samples:
            qualified[first:last] = True
    return qualified


def _median_rr_beats(
    ecg: np.ndarray, peak_samples: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(ecg, dtype=np.float64)
    peaks = np.unique(np.asarray(peak_samples, dtype=np.int64))
    if x.ndim != 1 or peaks.size < 2:
        return np.empty((0, 0), dtype=np.float64), np.array([], dtype=np.int64)
    width = max(3, int(np.rint(np.median(np.diff(peaks)))))
    pre = width // 2
    post = width - pre
    beats: list[np.ndarray] = []
    indices: list[int] = []
    for index, peak in enumerate(peaks):
        first = int(peak) - pre
        last = int(peak) + post
        if first >= 0 and last <= x.size:
            beat = x[first:last]
            if beat.size == width and np.all(np.isfinite(beat)):
                beats.append(beat)
                indices.append(index)
    if not beats:
        return np.empty((0, width), dtype=np.float64), np.array([], dtype=np.int64)
    return np.stack(beats), np.asarray(indices, dtype=np.int64)


def _fourier_fit(values: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    if order < 1:
        raise ValueError("order must be positive")
    y = np.asarray(values, dtype=np.float64)
    phase = np.linspace(0.0, 2.0 * np.pi, y.size, endpoint=False)
    columns = [np.ones(y.size)]
    for harmonic in range(1, order + 1):
        columns.extend((np.cos(harmonic * phase), np.sin(harmonic * phase)))
    design = np.column_stack(columns)
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coefficients, np.asarray(design @ coefficients, dtype=np.float64)


def _tolerance_samples(sampling_rate_hz: float, tolerance_ms: float) -> int:
    if sampling_rate_hz <= 0 or tolerance_ms <= 0:
        raise ValueError("sampling_rate_hz and tolerance_ms must be positive")
    return max(1, int(np.rint(tolerance_ms * sampling_rate_hz / 1000.0)))


def _events(values: np.ndarray, signal_size: int, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    events = np.unique(raw.astype(np.int64, copy=False))
    if np.any(events < 0) or np.any(events >= signal_size):
        raise ValueError(f"{name} contains samples outside the ECG window")
    return events


def _optional_events(
    values: np.ndarray | None, signal_size: int, name: str
) -> np.ndarray | None:
    return None if values is None else _events(values, signal_size, name)


def _window_events(
    events: np.ndarray | None, first: int, last: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if events is None:
        return None, None
    keep = (events >= first) & (events < last)
    return events[keep] - first, keep
