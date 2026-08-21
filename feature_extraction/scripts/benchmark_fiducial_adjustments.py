"""Benchmark published QRS-to-fiducial adjustment candidates on MIT-BIH.

This experiment starts from the same UNSW QRS candidates and changes only the
raw-ECG localization rule.  It does not claim to reproduce the full PhysioZoo
rqrs (which starts from gqrs) or the full R-DECO detector/GUI.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb

from ecg_cascade.fiducials import build_fiducial_candidates
from ecg_cascade.peaks import detect_unsw
from ecg_cascade.reliability import collapse_close_detections
from ecg_cascade.validation import match_peaks_to_reference, peak_metrics
from ecg_cascade.wfdb_io import load_wfdb_beat_annotations, load_wfdb_segment


ROOT = Path(__file__).resolve().parents[2]
DATABASE = (
    ROOT
    / "Datasets"
    / "mit-bih-arrhythmia-database-1.0"
    / "mit-bih-arrhythmia-database-1.0.0"
)
OUTPUT = ROOT / "output" / "rr_hrv_architecture" / "fiducial_ablation_v1"
DIFFICULT = ["108", "113", "207", "222", "231"]

METHOD_LABELS = {
    "unsw_initial": "UNSW event",
    "ho_positive": "Previous: Ho +max",
    "physiozoo_positive": "PhysioZoo + sign",
    "physiozoo_negative": "PhysioZoo - sign",
    "physiozoo_rqrs_adapted": "PhysioZoo rqrs rule\n(adapted to UNSW)",
    "rdeco_positive_backward": "R-DECO final +\n(not full detector)",
    "rdeco_negative_backward": "R-DECO final inverted\n(not full detector)",
    "reference_selected_record_sign_upper_bound": "Reference-selected sign\n(optimistic upper bound)",
}


def _records() -> list[str]:
    return [line.strip() for line in (DATABASE / "RECORDS").read_text().splitlines() if line.strip()]


def _rr_errors_ms(
    detected: np.ndarray,
    reference: np.ndarray,
    *,
    fs: float,
    tolerance_ms: float = 75.0,
) -> np.ndarray:
    matches = match_peaks_to_reference(
        detected,
        reference,
        sampling_rate_hz=fs,
        tolerance_ms=tolerance_ms,
    )
    errors: list[float] = []
    for position in range(matches.detected_indices.size - 1):
        detected_start = int(matches.detected_indices[position])
        detected_end = int(matches.detected_indices[position + 1])
        reference_start = int(matches.reference_indices[position])
        reference_end = int(matches.reference_indices[position + 1])
        if detected_end != detected_start + 1 or reference_end != reference_start + 1:
            continue
        detected_rr = detected[detected_end] - detected[detected_start]
        reference_rr = reference[reference_end] - reference[reference_start]
        errors.append((detected_rr - reference_rr) * 1000.0 / fs)
    return np.asarray(errors, dtype=float)


def _timing_errors_ms(
    detected: np.ndarray,
    reference: np.ndarray,
    *,
    fs: float,
) -> np.ndarray:
    matches = match_peaks_to_reference(
        detected,
        reference,
        sampling_rate_hz=fs,
        tolerance_ms=75.0,
    )
    return (
        detected[matches.detected_indices] - reference[matches.reference_indices]
    ) * 1000.0 / fs


def _choose_reference_selected_sign(
    positive: np.ndarray,
    negative: np.ndarray,
    reference: np.ndarray,
    *,
    fs: float,
) -> tuple[np.ndarray, int]:
    """Same-record reference selection: upper bound, never deployment logic."""

    candidates = []
    for sign, values in [(1, positive), (-1, negative)]:
        metrics = peak_metrics(values, reference, sampling_rate_hz=fs, tolerance_ms=75.0)
        candidates.append(
            (
                float(metrics["f1"]),
                -float(metrics["median_absolute_timing_ms"]),
                sign,
                values,
            )
        )
    _, _, sign, values = max(candidates, key=lambda row: (row[0], row[1]))
    return values, sign


def run() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    record_rows: list[dict[str, object]] = []
    timing_by_method: dict[str, list[np.ndarray]] = {}
    rr_by_method: dict[str, list[np.ndarray]] = {}
    pooled_counts: dict[str, dict[str, int]] = {}
    selected_sign_counts = {"positive": 0, "negative": 0}

    for record_name in _records():
        record_path = DATABASE / record_name
        segment = load_wfdb_segment(record_path, channel=0)
        reference, _ = load_wfdb_beat_annotations(record_path)
        run = detect_unsw(segment.samples, segment.sampling_rate_hz, orientation="original")
        primary = collapse_close_detections(
            run.peak_samples,
            run.analysis_signal,
            sampling_rate_hz=segment.sampling_rate_hz,
            exclusion_ms=150.0,
        )
        candidates = build_fiducial_candidates(
            primary,
            run.analysis_signal,
            sampling_rate_hz=segment.sampling_rate_hz,
        )
        upper_bound, upper_sign = _choose_reference_selected_sign(
            candidates.physiozoo_positive,
            candidates.physiozoo_negative,
            reference,
            fs=segment.sampling_rate_hz,
        )
        methods = {
            "unsw_initial": primary,
            "ho_positive": candidates.ho_positive,
            "physiozoo_positive": candidates.physiozoo_positive,
            "physiozoo_negative": candidates.physiozoo_negative,
            "physiozoo_rqrs_adapted": candidates.physiozoo_rqrs_adapted,
            "rdeco_positive_backward": candidates.rdeco_positive_backward,
            "rdeco_negative_backward": candidates.rdeco_negative_backward,
            "reference_selected_record_sign_upper_bound": upper_bound,
        }
        selected_sign_counts[
            "positive" if candidates.physiozoo_rqrs_selected_sign > 0 else "negative"
        ] += 1
        for method, values in methods.items():
            metrics = peak_metrics(
                values,
                reference,
                sampling_rate_hz=segment.sampling_rate_hz,
                tolerance_ms=75.0,
            )
            rr_errors = _rr_errors_ms(
                values,
                reference,
                fs=segment.sampling_rate_hz,
            )
            timing_errors = _timing_errors_ms(
                values,
                reference,
                fs=segment.sampling_rate_hz,
            )
            record_rows.append(
                {
                    "record": record_name,
                    "method": method,
                    **metrics,
                    "rr_evaluable_count": int(rr_errors.size),
                    "rr_mae_ms": float(np.mean(np.abs(rr_errors))) if rr_errors.size else np.nan,
                    "rqrs_adapted_selected_sign": candidates.physiozoo_rqrs_selected_sign,
                    "rqrs_delta_positive_mV": candidates.physiozoo_rqrs_median_delta_to_positive,
                    "rqrs_delta_negative_mV": candidates.physiozoo_rqrs_median_delta_to_negative,
                    "reference_selected_sign_upper_bound": upper_sign,
                }
            )
            timing_by_method.setdefault(method, []).append(timing_errors)
            rr_by_method.setdefault(method, []).append(rr_errors)
            totals = pooled_counts.setdefault(method, {"tp": 0, "fp": 0, "fn": 0})
            for key in totals:
                totals[key] += int(metrics[key])

    record_frame = pd.DataFrame(record_rows)
    pooled_rows = []
    for method, counts in pooled_counts.items():
        sensitivity = counts["tp"] / (counts["tp"] + counts["fn"])
        ppv = counts["tp"] / (counts["tp"] + counts["fp"])
        f1 = 2 * sensitivity * ppv / (sensitivity + ppv)
        timing = np.concatenate(timing_by_method[method])
        rr_errors = np.concatenate(rr_by_method[method])
        pooled_rows.append(
            {
                "method": method,
                **counts,
                "sensitivity": sensitivity,
                "ppv": ppv,
                "f1": f1,
                "median_absolute_timing_ms": float(np.median(np.abs(timing))),
                "p95_absolute_timing_ms": float(np.percentile(np.abs(timing), 95)),
                "rr_evaluable_count": int(rr_errors.size),
                "rr_mae_ms": float(np.mean(np.abs(rr_errors))),
                "rr_p95_absolute_error_ms": float(np.percentile(np.abs(rr_errors), 95)),
            }
        )
    pooled_frame = pd.DataFrame(pooled_rows)
    status = {
        "database": str(DATABASE),
        "records": _records(),
        "sampling_rate_hz": 360,
        "channel": "first channel for every record",
        "match_tolerance_ms": 75,
        "physiozoo_qrs_adjust": (
            "source-faithful local min/max reproduction; input sign supplied"
        ),
        "physiozoo_rqrs_adapted": (
            "global polarity and 56-ms forward-search logic applied to UNSW "
            "events, not to the gqrs onsets used by the original rqrs"
        ),
        "rqrs_adapted_sign_counts": selected_sign_counts,
        "rdeco": {
            "original_matlab_executed": False,
            "reason": "MATLAB/Octave is not installed in this environment",
            "tested_component": (
                "public peaks_in_ecg 65-ms backward positive-maximum stage, "
                "with and without the source's user-controlled global inversion"
            ),
            "manual_correction_benchmarked": False,
            "manual_correction_reason": (
                "R-DECO correction is a human GUI operation; automatic use "
                "would not constitute expert correction"
            ),
        },
        "upper_bound_warning": (
            "Reference-selected sign uses each test record's annotations and is "
            "optimistically biased; it is not a deployable selector."
        ),
    }
    record_frame.to_csv(OUTPUT / "fiducial_metrics_by_record.csv", index=False)
    pooled_frame.to_csv(OUTPUT / "fiducial_metrics_pooled.csv", index=False)
    (OUTPUT / "method_status.json").write_text(json.dumps(status, indent=2))
    _plot_pooled(pooled_frame)
    _plot_difficult(record_frame)
    _plot_timing_error_identity()
    _plot_record_207_excerpt()
    return record_frame, pooled_frame, status


def _plot_pooled(pooled: pd.DataFrame) -> None:
    methods = [
        "unsw_initial",
        "ho_positive",
        "physiozoo_positive",
        "physiozoo_negative",
        "physiozoo_rqrs_adapted",
        "rdeco_positive_backward",
        "rdeco_negative_backward",
        "reference_selected_record_sign_upper_bound",
    ]
    frame = pooled.set_index("method").loc[methods]
    labels = [METHOD_LABELS[method] for method in methods]
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
    colors = ["#3d5a80", "#c44536", "#ee9b00", "#0a9396", "#6a4c93", "#bb3e03", "#005f73", "#777777"]
    axes[0].bar(np.arange(len(methods)), frame["f1"], color=colors)
    axes[0].set_ylim(min(0.95, float(frame["f1"].min()) - 0.005), 1.001)
    axes[0].set_ylabel("Pooled F1 (higher is better)")
    axes[1].bar(np.arange(len(methods)), frame["median_absolute_timing_ms"], color=colors)
    axes[1].set_ylabel("Median absolute timing error (ms; lower is better)")
    axes[2].bar(np.arange(len(methods)), frame["rr_mae_ms"], color=colors)
    axes[2].set_ylabel("Evaluable RR mean absolute error (ms; lower is better)")
    for axis in axes:
        axis.set_xticks(np.arange(len(methods)), labels, rotation=45, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Same UNSW events, different fiducial rules: MIT-BIH 48-record audit")
    figure.savefig(OUTPUT / "pooled_fiducial_ablation.png", dpi=220)
    plt.close(figure)


def _plot_difficult(record_frame: pd.DataFrame) -> None:
    methods = [
        "unsw_initial",
        "ho_positive",
        "physiozoo_positive",
        "physiozoo_negative",
        "physiozoo_rqrs_adapted",
        "rdeco_positive_backward",
        "rdeco_negative_backward",
    ]
    figure, axes = plt.subplots(len(DIFFICULT), 2, figsize=(15, 17), constrained_layout=True)
    for row, record in enumerate(DIFFICULT):
        frame = record_frame[
            (record_frame["record"] == record) & record_frame["method"].isin(methods)
        ].set_index("method").loc[methods]
        labels = [METHOD_LABELS[method].replace("\n", " ") for method in methods]
        x = np.arange(len(methods))
        axes[row, 0].bar(x, frame["f1"], color="#457b9d")
        axes[row, 0].set_ylim(max(0.7, float(frame["f1"].min()) - 0.02), 1.002)
        axes[row, 0].set_ylabel(f"{record}: F1")
        axes[row, 1].bar(x, frame["median_absolute_timing_ms"], color="#e76f51")
        axes[row, 1].set_ylabel(f"{record}: median |timing error| (ms)")
        for axis in axes[row]:
            axis.set_xticks(x, labels, rotation=35, ha="right", fontsize=7)
            axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Five difficult records: polarity and window rules change timing differently")
    figure.savefig(OUTPUT / "five_difficult_fiducial_ablation.png", dpi=220)
    plt.close(figure)


def _plot_timing_error_identity() -> None:
    beat_errors = np.array([20.0, 0.0, 20.0])
    rr_errors = np.diff(beat_errors)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].stem([1, 2, 3], beat_errors, basefmt=" ")
    axes[0].set(xticks=[1, 2, 3], xlabel="Beat", ylabel="Fiducial error e_i (ms)")
    axes[0].set_title("Beat timestamps: +20, 0, +20 ms")
    axes[1].stem([1, 2], rr_errors, basefmt=" ", linefmt="#c44536", markerfmt="o")
    axes[1].set(xticks=[1, 2], xlabel="RR interval", ylabel="RR error e_i - e_(i-1) (ms)")
    axes[1].set_title("Changing error becomes false RR variation")
    for axis in axes:
        axis.axhline(0, color="black", lw=0.8)
        axis.grid(alpha=0.25)
    figure.savefig(OUTPUT / "timing_error_propagates_to_rr.png", dpi=220)
    plt.close(figure)


def _plot_record_207_excerpt() -> None:
    record_path = DATABASE / "207"
    start_s, end_s = 1742.5, 1752.5
    header = wfdb.rdheader(str(record_path))
    fs = float(header.fs)
    start = int(round(start_s * fs))
    end = int(round(end_s * fs))
    full = load_wfdb_segment(record_path, channel=0)
    reference, _ = load_wfdb_beat_annotations(record_path)
    run = detect_unsw(full.samples, fs, orientation="original")
    primary = collapse_close_detections(
        run.peak_samples,
        run.analysis_signal,
        sampling_rate_hz=fs,
        exclusion_ms=150.0,
    )
    candidates = build_fiducial_candidates(primary, run.analysis_signal, sampling_rate_hz=fs)
    series = {
        "Expert": reference,
        "UNSW event": primary,
        "PhysioZoo +": candidates.physiozoo_positive,
        "PhysioZoo -": candidates.physiozoo_negative,
        f"rqrs adapted (sign {candidates.physiozoo_rqrs_selected_sign:+d})": candidates.physiozoo_rqrs_adapted,
    }
    colors = ["black", "#457b9d", "#e76f51", "#2a9d8f", "#6a4c93"]
    markers = ["D", "|", "^", "v", "x"]
    time = np.arange(start, end) / fs
    figure, axis = plt.subplots(figsize=(16, 5.5), constrained_layout=True)
    axis.plot(time, full.samples[start:end], color="#45566c", lw=0.9, label="MLII ECG")
    for (label, samples), color, marker in zip(series.items(), colors, markers, strict=True):
        shown = samples[(samples >= start) & (samples < end)]
        axis.scatter(
            shown / fs,
            full.samples[shown],
            s=45,
            marker=marker,
            color=color,
            label=label,
            zorder=4,
        )
    axis.set(xlabel="Time from recording start (s)", ylabel="ECG (mV)")
    axis.set_title("Record 207: the same UNSW QRS events yield different polarity-dependent fiducials")
    axis.grid(alpha=0.2)
    axis.legend(ncol=3, fontsize=8)
    figure.savefig(OUTPUT / "record_207_physiozoo_candidates.png", dpi=220)
    plt.close(figure)


if __name__ == "__main__":
    records, pooled, status = run()
    print(pooled.to_string(index=False))
    print(json.dumps(status["rdeco"], indent=2))
