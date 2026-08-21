"""Diagnostic plots for visual branch validation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .edf import SignalSegment
from .peaks import PeakAgreement
from .pipeline import RRHRVBranchResult
from .neurokit_zhai import NeuroKitZhaiResult


def save_rr_hrv_diagnostic(
    output_path: str | Path,
    *,
    segment: SignalSegment,
    features: pd.DataFrame,
    pan_peaks: np.ndarray,
    neurokit_peaks: np.ndarray,
    agreement: PeakAgreement,
    events: list[tuple[float, float]],
    feature_peak_source: str,
) -> None:
    fs = segment.channel.sampling_rate_hz
    signal = segment.samples
    pan_peaks = np.asarray(pan_peaks, dtype=np.int64)
    neurokit_peaks = np.asarray(neurokit_peaks, dtype=np.int64)

    disagreement = np.sort(
        np.concatenate(
            [agreement.unmatched_primary_samples, agreement.unmatched_comparator_samples]
        )
    )
    events_in_segment = [
        event
        for event in events
        if event[1] >= segment.start_s
        and event[0] <= segment.start_s + segment.duration_s
    ]
    if events_in_segment:
        event_start, event_end = events_in_segment[0]
        center_sample = int(round(((event_start + event_end) / 2 - segment.start_s) * fs))
    else:
        center_sample = int(disagreement[0]) if disagreement.size else signal.size // 2
    half_width = int(round(10.0 * fs))
    first = max(0, center_sample - half_width)
    last = min(signal.size, center_sample + half_width)
    detail_time = segment.start_s + np.arange(first, last) / fs

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True)
    axes[0].plot(detail_time, signal[first:last], color="#17365D", linewidth=0.8)
    for peaks, color, marker, label in [
        (pan_peaks, "#C00000", "o", "Pan-Tompkins detection"),
        (neurokit_peaks, "#008C95", "x", "NeuroKit detection"),
    ]:
        visible = peaks[(peaks >= first) & (peaks < last)]
        axes[0].scatter(
            segment.start_s + visible / fs,
            signal[visible],
            s=25,
            marker=marker,
            color=color,
            label=label,
            zorder=3,
        )
    axes[0].set_ylabel(segment.channel.physical_dimension or "EDF physical units")
    axes[0].set_title("Twenty-second R-peak audit view")
    axes[0].legend(loc="upper right")

    axes[1].plot(
        features["time_s"], features["heart_rate_bpm"], color="#17365D", linewidth=0.9
    )
    axes[1].set_ylabel("Heart rate (bpm)")
    axes[1].set_title(f"RR-derived heart rate; peak source: {feature_peak_source}")

    axes[2].plot(
        features["time_s"],
        features["j1_csi_x_slope"],
        label="J1 = CSI100 x slope",
        color="#C00000",
        linewidth=0.9,
    )
    axes[2].plot(
        features["time_s"],
        features["j2_modcsi_filtered_x_slope"],
        label="J2 = filtered ModCSI100 x slope",
        color="#008C95",
        linewidth=0.9,
    )
    axes[2].set_ylabel("Continuous index")
    axes[2].set_xlabel("Time from EDF start (s)")
    axes[2].set_title("RR/HRV branch outputs (no seizure threshold applied)")
    axes[2].legend(loc="upper right")

    for axis in axes[1:]:
        for event_start, event_end in events:
            axis.axvspan(event_start, event_end, color="#E6A700", alpha=0.18)
        axis.grid(alpha=0.2)
    axes[0].grid(alpha=0.2)
    fig.suptitle(
        f"{Path(segment.path).name} | {segment.channel.label} | "
        f"R-peak agreement={agreement.agreement_f1:.3f}",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def save_neurokit_zhai_diagnostic(
    output_path: str | Path,
    *,
    ecg: np.ndarray,
    result: NeuroKitZhaiResult,
    title: str,
    detail_start_s: float = 0.0,
    detail_duration_s: float = 10.0,
) -> None:
    """Plot detector outputs and literal Zhai intermediate transformations."""

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    signal = np.asarray(ecg, dtype=np.float64)
    fs = float(result.metadata["sampling_rate_hz"])
    segment_start_s = float(result.metadata["segment_start_s"])
    selected = result.selected
    first = max(0, int(np.rint(detail_start_s * fs)))
    last = min(signal.size, first + int(np.rint(detail_duration_s * fs)))
    if first >= last:
        raise ValueError("Requested diagnostic detail interval is empty")
    local_samples = np.arange(first, last)
    time = segment_start_s + local_samples / fs

    fig, axes = plt.subplots(4, 1, figsize=(15, 12), constrained_layout=True)
    axes[0].plot(time, signal[first:last], color="#17365D", linewidth=0.8)
    for peaks, color, marker, label in [
        (
            selected.neurokit.detector.peak_samples,
            "#008C95",
            "x",
            "NeuroKit timestamp",
        ),
        (
            selected.zhai.detector.peak_samples,
            "#D88400",
            "o",
            "Zhai cross-correlation timestamp",
        ),
    ]:
        visible = peaks[(peaks >= first) & (peaks < last)]
        axes[0].scatter(
            segment_start_s + visible / fs,
            signal[visible],
            s=28,
            marker=marker,
            color=color,
            facecolors="none" if marker == "o" else color,
            label=label,
            zorder=4,
        )
    axes[0].set_title("Independent detector timestamps on the raw ECG")
    axes[0].set_ylabel("ECG amplitude")
    axes[0].legend(loc="upper right")

    zhai = selected.zhai_detection
    axes[1].plot(
        time,
        zhai.qrs_envelope[first:last],
        color="#17365D",
        linewidth=0.9,
        label="Zhai squared/low-pass QRS envelope",
    )
    axes[1].plot(
        time,
        zhai.dynamic_threshold[first:last],
        color="#C00000",
        linewidth=0.9,
        label="Dynamic threshold",
    )
    for window_first, window_last in zhai.qrs_windows:
        if window_last <= first or window_first >= last:
            continue
        axes[1].axvspan(
            segment_start_s + max(int(window_first), first) / fs,
            segment_start_s + min(int(window_last), last) / fs,
            color="#E6A700",
            alpha=0.12,
        )
    axes[1].set_title("Zhai QRS-envelope windows; shading is not an artifact label")
    axes[1].set_ylabel("Envelope")
    axes[1].legend(loc="upper right")

    correlation_in_qrs_windows = np.where(
        zhai.qrs_mask[first:last],
        zhai.correlation_signal[first:last],
        np.nan,
    )
    axes[2].plot(
        time,
        correlation_in_qrs_windows,
        color="#7A5195",
        linewidth=0.9,
        label="Normalized template correlation",
    )
    zhai_visible_mask = (
        (zhai.peak_samples >= first) & (zhai.peak_samples < last)
    )
    visible_peaks = zhai.peak_samples[zhai_visible_mask]
    visible_values = zhai.correlation_peak_values[zhai_visible_mask]
    axes[2].scatter(
        segment_start_s + visible_peaks / fs,
        visible_values,
        color="#D88400",
        s=25,
        label="Maximum |correlation| per QRS window",
        zorder=4,
    )
    axes[2].axhline(0, color="black", linewidth=0.5)
    axes[2].set_ylim(-1.05, 1.05)
    axes[2].set_title(
        "Zhai localization: whole-template alignment, not raw positive amplitude"
    )
    axes[2].set_ylabel("Correlation")
    axes[2].legend(loc="lower right")

    for track, color, label in [
        (selected.neurokit, "#008C95", "NeuroKit RR"),
        (selected.zhai, "#D88400", "Zhai RR"),
    ]:
        if not track.rr_intervals.empty:
            axes[3].plot(
                track.rr_intervals["end_time_s"],
                track.rr_intervals["rr_ms"],
                color=color,
                linewidth=0.75,
                alpha=0.85,
                label=label,
            )
    axes[3].set_title("Separate RR series; no averaging or automatic fusion")
    axes[3].set_ylabel("RR (ms)")
    axes[3].set_xlabel("Time from recording start (s)")
    axes[3].legend(loc="upper right")

    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle(
        f"{title} | NeuroKit--Zhai audit agreement "
        f"F1={selected.audit_agreement.agreement_f1:.3f}",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def save_neurokit_zhai_morphology_diagnostic(
    output_path: str | Path,
    *,
    ecg: np.ndarray,
    result: NeuroKitZhaiResult,
    title: str,
    detail_start_s: float = 0.0,
    detail_duration_s: float = 10.0,
) -> None:
    """Plot the two anchor tracks and their separate morphology/HRV outputs."""

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    signal = np.asarray(ecg, dtype=np.float64)
    fs = float(result.metadata["sampling_rate_hz"])
    segment_start_s = float(result.metadata["segment_start_s"])
    selected = result.selected
    first = max(0, int(np.rint(detail_start_s * fs)))
    last = min(signal.size, first + int(np.rint(detail_duration_s * fs)))
    if first >= last:
        raise ValueError("Requested diagnostic detail interval is empty")
    time = segment_start_s + np.arange(first, last) / fs

    fig, axes = plt.subplots(5, 1, figsize=(15, 16), constrained_layout=True)
    axes[0].plot(time, signal[first:last], color="#17365D", linewidth=0.8)
    for track, color, marker, label in [
        (selected.neurokit, "#008C95", "x", "NeuroKit anchor"),
        (selected.zhai, "#D88400", "o", "Zhai anchor"),
    ]:
        peaks = track.detector.peak_samples
        visible = peaks[(peaks >= first) & (peaks < last)]
        axes[0].scatter(
            segment_start_s + visible / fs,
            signal[visible],
            s=28,
            marker=marker,
            color=color,
            facecolors="none" if marker == "o" else color,
            label=label,
            zorder=4,
        )
    axes[0].set_title("Two candidate timestamp tracks; neither is selected")
    axes[0].set_ylabel("ECG amplitude")
    axes[0].legend(loc="upper right")

    _plot_first_varon_stack(
        axes[1], selected.neurokit.morphology, fs, color="#008C95"
    )
    axes[1].set_title("First complete NeuroKit-anchored five-QRS matrix")
    _plot_first_varon_stack(
        axes[2], selected.zhai.morphology, fs, color="#D88400"
    )
    axes[2].set_title("First complete Zhai-anchored five-QRS matrix")

    for track, color, label in [
        (selected.neurokit, "#008C95", "NeuroKit"),
        (selected.zhai, "#D88400", "Zhai"),
    ]:
        morphology = track.morphology.features
        if morphology.empty:
            continue
        for component, linestyle in [(3, "-"), (4, "--"), (5, ":")]:
            axes[3].plot(
                morphology["window_end_time_s"],
                morphology[f"lambda{component}"],
                color=color,
                linestyle=linestyle,
                linewidth=0.8,
                alpha=0.9,
                label=f"{label} lambda{component}",
            )
    axes[3].set_title(
        "Published Varon morphology components lambda3-lambda5; no seizure label"
    )
    axes[3].set_ylabel("Raw Gram eigenvalue")
    axes[3].legend(loc="upper right", ncol=2)

    for track, color, label in [
        (selected.neurokit, "#008C95", "NeuroKit"),
        (selected.zhai, "#D88400", "Zhai"),
    ]:
        features = track.features
        if features.empty:
            continue
        axes[4].plot(
            features["end_time_s"],
            features["j1_csi_x_slope"],
            color=color,
            linewidth=0.8,
            label=f"{label} J1",
        )
        axes[4].plot(
            features["end_time_s"],
            features["j2_modcsi_filtered_x_slope"],
            color=color,
            linestyle="--",
            linewidth=0.8,
            label=f"{label} J2",
        )
    axes[4].set_title("Separate Jeppesen RR/HRV measurements")
    axes[4].set_ylabel("J index")
    axes[4].set_xlabel("Time from recording start (s)")
    axes[4].legend(loc="upper right", ncol=2)

    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle(
        f"{title} | dual-track RR/HRV + Varon morphology audit",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _plot_first_varon_stack(axis, morphology, sampling_rate_hz: float, *, color: str):
    defined = morphology.features[
        morphology.features["published_varon_core_defined"]
    ]
    if defined.empty:
        axis.text(
            0.5,
            0.5,
            "No complete five-beat morphology window",
            transform=axis.transAxes,
            ha="center",
            va="center",
        )
        return
    first_index = int(defined.iloc[0]["start_beat_index"])
    last_index = int(defined.iloc[0]["end_beat_index"])
    waveforms = morphology.beat_waveforms[first_index : last_index + 1]
    pre_samples = int(morphology.parameters["pre_r_samples"])
    relative_ms = (
        (np.arange(waveforms.shape[1]) - pre_samples)
        * 1000.0
        / sampling_rate_hz
    )
    for offset, waveform in enumerate(waveforms):
        axis.plot(
            relative_ms,
            waveform,
            color=color,
            linewidth=0.8,
            alpha=0.35 + 0.13 * offset,
            label=f"beat {first_index + offset}",
        )
    axis.axvline(0.0, color="black", linewidth=0.7, linestyle=":")
    axis.set_xlabel("Time relative to detector anchor (ms)")
    axis.set_ylabel("ECG amplitude")
    axis.legend(loc="upper right", ncol=5, fontsize=7)


def save_complete_rr_hrv_diagnostic(
    output_path: str | Path,
    *,
    ecg: np.ndarray,
    result: RRHRVBranchResult,
    title: str,
) -> None:
    """Plot detector outputs, RR reliability, coverage, and HRV measurements."""

    signal = np.asarray(ecg, dtype=np.float64)
    selected = result.selected
    features = selected.features
    fs = float(result.metadata["sampling_rate_hz"])
    start_s = float(result.metadata["segment_start_s"])
    unsupported = selected.reliability.rr_intervals
    unsupported = unsupported.loc[~unsupported["rr_supported"]]
    if not unsupported.empty:
        center_s = float(unsupported.iloc[0]["end_time_s"])
    elif selected.reliability.primary_refined_samples.size:
        middle = selected.reliability.primary_refined_samples.size // 2
        center_s = start_s + selected.reliability.primary_refined_samples[middle] / fs
    else:
        center_s = start_s + signal.size / (2 * fs)
    center_sample = int(np.rint((center_s - start_s) * fs))
    half_width = int(np.rint(7.5 * fs))
    first = max(0, center_sample - half_width)
    last = min(signal.size, center_sample + half_width)
    detail_time = start_s + np.arange(first, last) / fs

    fig, axes = plt.subplots(4, 1, figsize=(15, 13), constrained_layout=True)
    axes[0].plot(detail_time, signal[first:last], color="#17365D", linewidth=0.8, label="Raw ECG")
    marker_specs = [
        (selected.reliability.primary_refined_samples, "#111111", "o", "UNSW refined R"),
        (selected.secondary_experimental.peak_samples, "#008C95", "x", "NeuroKit 250 ms"),
        (selected.secondary_baseline.peak_samples, "#7A5195", "+", "NeuroKit 300 ms"),
    ]
    for delay, detector in sorted(selected.pan_context.items(), reverse=True):
        marker_specs.append(
            (
                detector.peak_samples,
                "#C00000" if delay == 300 else "#E6A700",
                "v" if delay == 300 else "^",
                f"Pan context {delay:.0f} ms",
            )
        )
    for peaks, color, marker, label in marker_specs:
        peaks = np.asarray(peaks, dtype=np.int64)
        visible = peaks[(peaks >= first) & (peaks < last)]
        if visible.size:
            axes[0].scatter(
                start_s + visible / fs,
                signal[visible],
                s=25,
                marker=marker,
                color=color,
                label=label,
                zorder=3,
            )
    axes[0].set_title("Fifteen-second detector audit view")
    axes[0].set_ylabel("ECG amplitude")
    axes[0].legend(loc="upper right", ncols=2, fontsize=8)

    supported = features["rr_supported"].to_numpy(dtype=bool)
    axes[1].plot(
        features["end_time_s"],
        features["heart_rate_bpm"],
        color="#B8B8B8",
        linewidth=0.8,
    )
    if len(features):
        axes[1].scatter(
            features.loc[supported, "end_time_s"],
            features.loc[supported, "heart_rate_bpm"],
            s=9,
            color="#008C95",
            label="Supported RR",
        )
        axes[1].scatter(
            features.loc[~supported, "end_time_s"],
            features.loc[~supported, "heart_rate_bpm"],
            s=14,
            color="#C00000",
            label="Unsupported RR retained",
        )
    axes[1].set_title("Primary RR-derived heart rate and reliability mask")
    axes[1].set_ylabel("Heart rate (bpm)")
    axes[1].legend(loc="upper right")

    window = result.config.hrv_window_rr_intervals
    current_coverage = features["rr_supported"].rolling(window).mean()
    baseline_coverage = features["rr_supported_nk300_baseline"].rolling(window).mean()
    legacy_coverage = features["rr_supported_support150_ablation"].rolling(window).mean()
    axes[2].plot(
        features["end_time_s"], current_coverage, color="#008C95", label="NK250 + 50-ms support"
    )
    axes[2].plot(
        features["end_time_s"], baseline_coverage, color="#7A5195", label="NK300 + 50-ms support"
    )
    axes[2].plot(
        features["end_time_s"], legacy_coverage, color="#E6A700", label="NK250 + 150-ms ablation"
    )
    axes[2].axhline(
        result.config.reliable_hrv_coverage,
        color="#C00000",
        linestyle="--",
        linewidth=0.8,
        label="Reliable-feature threshold",
    )
    axes[2].set_ylim(-0.03, 1.03)
    axes[2].set_ylabel("100-RR coverage")
    axes[2].set_title("Reliability ablations; coverage is not artifact probability")
    axes[2].legend(loc="lower right", ncols=2, fontsize=8)

    axes[3].plot(
        features["end_time_s"],
        features["j1_csi_x_slope"],
        color="#C00000",
        linewidth=0.9,
        label="J1 = CSI100 x slope",
    )
    axes[3].plot(
        features["end_time_s"],
        features["j2_modcsi_filtered_x_slope"],
        color="#008C95",
        linewidth=0.9,
        label="J2 = filtered ModCSI100 x slope",
    )
    unreliable = features["feature_defined"] & (~features["feature_reliable"])
    if unreliable.any():
        axes[3].scatter(
            features.loc[unreliable, "end_time_s"],
            features.loc[unreliable, "j1_csi_x_slope"],
            facecolors="none",
            edgecolors="#E6A700",
            s=18,
            label="Defined but reliability criterion failed",
        )
    axes[3].set_title("Jeppesen 100-RR branch outputs; no seizure threshold")
    axes[3].set_ylabel("Continuous measurement")
    axes[3].set_xlabel("Time from recording start (s)")
    axes[3].legend(loc="upper right", fontsize=8)

    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle(
        f"{title} | selected polarity={result.selected_orientation} | "
        f"RR coverage={selected.reliability.rr_coverage:.3f}",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
