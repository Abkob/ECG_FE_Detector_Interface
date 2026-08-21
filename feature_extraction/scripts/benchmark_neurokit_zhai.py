"""Benchmark independent NeuroKit and Zhai tracks on annotated WFDB ECGs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ecg_cascade.peaks import detect_neurokit_gradient
from ecg_cascade.validation import (
    PeakMatch,
    beat_symbol_sensitivity,
    match_peaks_to_reference,
    peak_metrics,
)
from ecg_cascade.wfdb_io import load_wfdb_beat_annotations, load_wfdb_segment
from ecg_cascade.zhai import detect_zhai_template


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--records", nargs="*")
    parser.add_argument("--channel", default="0")
    parser.add_argument("--orientation", choices=["original", "inverted"], default="original")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    database = args.database_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = args.records or sorted(path.stem for path in database.glob("*.hea"))
    channel = _parse_channel(args.channel)

    metric_rows: list[dict[str, object]] = []
    symbol_rows: list[pd.DataFrame] = []
    timing_errors: dict[tuple[str, float], list[np.ndarray]] = {}
    rr_errors: dict[tuple[str, float], list[np.ndarray]] = {}
    matched_timing_rows: list[dict[str, object]] = []
    unmatched_detected_rows: list[dict[str, object]] = []
    missed_reference_rows: list[dict[str, object]] = []
    for record_name in records:
        record_path = database / record_name
        segment = load_wfdb_segment(record_path, channel=channel)
        reference, symbols = load_wfdb_beat_annotations(
            record_path,
            start_sample=segment.start_sample,
            end_sample=segment.start_sample + segment.samples.size,
        )
        neurokit = detect_neurokit_gradient(
            segment.samples,
            segment.sampling_rate_hz,
            orientation=args.orientation,
            minimum_delay_ms=300.0,
            minimum_delay_inclusive=False,
        )
        zhai = detect_zhai_template(
            segment.samples,
            segment.sampling_rate_hz,
            orientation=args.orientation,
        )
        detectors = {
            "neurokit_300_strict": neurokit.peak_samples,
            "zhai2023_reimplementation": zhai.peak_samples,
        }
        for detector_name, detected in detectors.items():
            for tolerance_ms in [75.0, 25.0, 1000.0 / segment.sampling_rate_hz]:
                metrics = peak_metrics(
                    detected,
                    reference,
                    sampling_rate_hz=segment.sampling_rate_hz,
                    tolerance_ms=tolerance_ms,
                )
                matches = match_peaks_to_reference(
                    detected,
                    reference,
                    sampling_rate_hz=segment.sampling_rate_hz,
                    tolerance_ms=tolerance_ms,
                )
                rr_error = _consecutive_rr_errors_ms(
                    detected,
                    reference,
                    matches.detected_indices,
                    matches.reference_indices,
                    sampling_rate_hz=segment.sampling_rate_hz,
                )
                metric_rows.append(
                    {
                        "record": record_name,
                        "channel": segment.channel_name,
                        "sampling_rate_hz": segment.sampling_rate_hz,
                        "detector": detector_name,
                        "tolerance_ms": tolerance_ms,
                        **metrics,
                        "evaluable_consecutive_rr": int(rr_error.size),
                        "rr_mae_ms": (
                            float(np.mean(np.abs(rr_error)))
                            if rr_error.size
                            else np.nan
                        ),
                        "rr_p95_absolute_error_ms": (
                            float(np.percentile(np.abs(rr_error), 95))
                            if rr_error.size
                            else np.nan
                        ),
                        "zhai_median_absolute_correlation": (
                            float(np.median(np.abs(zhai.correlation_peak_values)))
                            if detector_name == "zhai2023_reimplementation"
                            and zhai.correlation_peak_values.size
                            else np.nan
                        ),
                    }
                )
                errors = (
                    detected[matches.detected_indices]
                    - reference[matches.reference_indices]
                ) * 1000.0 / segment.sampling_rate_hz
                timing_errors.setdefault((detector_name, tolerance_ms), []).append(errors)
                rr_errors.setdefault((detector_name, tolerance_ms), []).append(rr_error)
                if np.isclose(tolerance_ms, 75.0):
                    _append_event_level_audit_rows(
                        record_name=record_name,
                        channel_name=segment.channel_name,
                        sampling_rate_hz=segment.sampling_rate_hz,
                        detector_name=detector_name,
                        detected=detected,
                        reference=reference,
                        symbols=symbols,
                        matches=matches,
                        matched_rows=matched_timing_rows,
                        unmatched_detected_rows=unmatched_detected_rows,
                        missed_reference_rows=missed_reference_rows,
                    )
            symbol_frame = beat_symbol_sensitivity(
                detected,
                reference,
                symbols,
                sampling_rate_hz=segment.sampling_rate_hz,
                tolerance_ms=75.0,
            )
            symbol_frame.insert(0, "detector", detector_name)
            symbol_frame.insert(0, "record", record_name)
            symbol_rows.append(symbol_frame)
        print(f"completed {record_name}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    symbols = pd.concat(symbol_rows, ignore_index=True)
    pooled_rows: list[dict[str, object]] = []
    for (detector_name, tolerance_ms), group in metrics.groupby(
        ["detector", "tolerance_ms"], sort=True
    ):
        tp = int(group["tp"].sum())
        fp = int(group["fp"].sum())
        fn = int(group["fn"].sum())
        sensitivity = tp / (tp + fn) if tp + fn else np.nan
        ppv = tp / (tp + fp) if tp + fp else np.nan
        f1 = (
            2 * sensitivity * ppv / (sensitivity + ppv)
            if np.isfinite(sensitivity + ppv) and sensitivity + ppv > 0
            else np.nan
        )
        errors = np.concatenate(timing_errors[(detector_name, tolerance_ms)])
        pooled_rr_errors = np.concatenate(rr_errors[(detector_name, tolerance_ms)])
        pooled_rows.append(
            {
                "detector": detector_name,
                "tolerance_ms": tolerance_ms,
                "records": int(group.shape[0]),
                "reference": int(group["reference"].sum()),
                "detected": int(group["detected"].sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "sensitivity": sensitivity,
                "ppv": ppv,
                "f1": f1,
                "mean_absolute_timing_ms": float(np.mean(np.abs(errors))),
                "median_absolute_timing_ms": float(np.median(np.abs(errors))),
                "p75_absolute_timing_ms": float(
                    np.percentile(np.abs(errors), 75)
                ),
                "p90_absolute_timing_ms": float(
                    np.percentile(np.abs(errors), 90)
                ),
                "p95_absolute_timing_ms": float(np.percentile(np.abs(errors), 95)),
                "p99_absolute_timing_ms": float(
                    np.percentile(np.abs(errors), 99)
                ),
                "maximum_absolute_timing_ms": float(np.max(np.abs(errors))),
                "mean_signed_timing_ms": float(np.mean(errors)),
                "median_signed_timing_ms": float(np.median(errors)),
                "signed_timing_standard_deviation_ms": float(
                    np.std(errors, ddof=0)
                ),
                "evaluable_consecutive_rr": int(pooled_rr_errors.size),
                "rr_mae_ms": float(np.mean(np.abs(pooled_rr_errors))),
                "rr_median_absolute_error_ms": float(
                    np.median(np.abs(pooled_rr_errors))
                ),
                "rr_p95_absolute_error_ms": float(
                    np.percentile(np.abs(pooled_rr_errors), 95)
                ),
                "rr_median_signed_error_ms": float(np.median(pooled_rr_errors)),
            }
        )
    pooled = pd.DataFrame(pooled_rows)
    metrics.to_csv(output / "record_metrics.csv", index=False)
    symbols.to_csv(output / "record_symbol_sensitivity.csv", index=False)
    pooled.to_csv(output / "pooled_metrics.csv", index=False)
    pd.DataFrame(matched_timing_rows).to_csv(
        output / "matched_rpeak_timing_errors_75ms.csv", index=False
    )
    pd.DataFrame(unmatched_detected_rows).to_csv(
        output / "unmatched_detected_events_75ms.csv", index=False
    )
    pd.DataFrame(missed_reference_rows).to_csv(
        output / "missed_expert_beats_75ms.csv", index=False
    )
    _save_distance_table(metrics, output)
    _save_benchmark_figures(metrics, pooled, output)
    (output / "benchmark_metadata.json").write_text(
        json.dumps(
            {
                "records": records,
                "database_dir": str(database),
                "channel": args.channel,
                "orientation": args.orientation,
                "automatic_timestamp_fusion": False,
                "raw_amplitude_argmax_refinement": False,
                "timing_distance_definition": (
                    "detected_minus_expert_for_monotone_one_to_one_matches; "
                    "absolute summaries condition on matches within tolerance"
                ),
                "primary_timing_report_tolerance_ms": 75.0,
                "unmatched_event_warning": (
                    "FP and FN do not receive a true timing error because no "
                    "validated one-to-one correspondence exists"
                ),
                "zhai_reference_doi": "10.3934/mbe.2023848",
                "zhai_implementation_status": (
                    "paper reimplementation; no official author source code located"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(pooled.to_string(index=False))


def _append_event_level_audit_rows(
    *,
    record_name: str,
    channel_name: str,
    sampling_rate_hz: float,
    detector_name: str,
    detected: np.ndarray,
    reference: np.ndarray,
    symbols: np.ndarray,
    matches: PeakMatch,
    matched_rows: list[dict[str, object]],
    unmatched_detected_rows: list[dict[str, object]],
    missed_reference_rows: list[dict[str, object]],
) -> None:
    """Store exact 75-ms matches and preserve both kinds of unmatched event."""

    matched_detected = set(matches.detected_indices.tolist())
    matched_reference = set(matches.reference_indices.tolist())
    for detected_index, reference_index in zip(
        matches.detected_indices,
        matches.reference_indices,
        strict=True,
    ):
        detected_sample = int(detected[int(detected_index)])
        reference_sample = int(reference[int(reference_index)])
        signed_error_ms = (
            (detected_sample - reference_sample) * 1000.0 / sampling_rate_hz
        )
        matched_rows.append(
            {
                "record": record_name,
                "channel": channel_name,
                "sampling_rate_hz": sampling_rate_hz,
                "detector": detector_name,
                "detected_index": int(detected_index),
                "expert_index": int(reference_index),
                "detected_sample": detected_sample,
                "expert_sample": reference_sample,
                "detected_time_s": detected_sample / sampling_rate_hz,
                "expert_time_s": reference_sample / sampling_rate_hz,
                "expert_symbol": str(symbols[int(reference_index)]),
                "detected_minus_expert_ms": signed_error_ms,
                "absolute_timing_error_ms": abs(signed_error_ms),
                "match_tolerance_ms": 75.0,
            }
        )

    for detected_index, detected_sample_value in enumerate(detected):
        if detected_index in matched_detected:
            continue
        detected_sample = int(detected_sample_value)
        nearest_index = _nearest_index(reference, detected_sample)
        nearest_sample = int(reference[nearest_index])
        unmatched_detected_rows.append(
            {
                "record": record_name,
                "channel": channel_name,
                "sampling_rate_hz": sampling_rate_hz,
                "detector": detector_name,
                "detected_index": detected_index,
                "detected_sample": detected_sample,
                "detected_time_s": detected_sample / sampling_rate_hz,
                "nearest_expert_index_context_only": nearest_index,
                "nearest_expert_sample_context_only": nearest_sample,
                "nearest_expert_symbol_context_only": str(symbols[nearest_index]),
                "nearest_expert_signed_distance_ms_context_only": (
                    (detected_sample - nearest_sample)
                    * 1000.0
                    / sampling_rate_hz
                ),
                "validated_correspondence": False,
                "reason": "no_one_to_one_expert_match_within_75ms",
            }
        )

    for reference_index, reference_sample_value in enumerate(reference):
        if reference_index in matched_reference:
            continue
        reference_sample = int(reference_sample_value)
        nearest_index = _nearest_index(detected, reference_sample)
        nearest_sample = int(detected[nearest_index])
        missed_reference_rows.append(
            {
                "record": record_name,
                "channel": channel_name,
                "sampling_rate_hz": sampling_rate_hz,
                "detector": detector_name,
                "expert_index": reference_index,
                "expert_sample": reference_sample,
                "expert_time_s": reference_sample / sampling_rate_hz,
                "expert_symbol": str(symbols[reference_index]),
                "nearest_detection_index_context_only": nearest_index,
                "nearest_detection_sample_context_only": nearest_sample,
                "nearest_detection_signed_distance_ms_context_only": (
                    (nearest_sample - reference_sample)
                    * 1000.0
                    / sampling_rate_hz
                ),
                "validated_correspondence": False,
                "reason": "no_one_to_one_detection_match_within_75ms",
            }
        )


def _nearest_index(sorted_samples: np.ndarray, target: int) -> int:
    if sorted_samples.size == 0:
        raise ValueError("Cannot find a nearest event in an empty sequence")
    right = int(np.searchsorted(sorted_samples, target, side="left"))
    candidates = [index for index in [right - 1, right] if 0 <= index < sorted_samples.size]
    return min(candidates, key=lambda index: abs(int(sorted_samples[index]) - target))


def _save_distance_table(metrics: pd.DataFrame, output: Path) -> None:
    """Create a readable one-row-per-record 75-ms timing audit."""

    at_75 = metrics[np.isclose(metrics["tolerance_ms"], 75.0)].copy()
    at_75["record_number"] = at_75["record"].astype(int)
    at_75 = at_75.sort_values(["record_number", "detector"])
    detector_prefix = {
        "neurokit_300_strict": "neurokit",
        "zhai2023_reimplementation": "zhai",
    }
    measurement_columns = [
        "detected",
        "reference",
        "tp",
        "fp",
        "fn",
        "sensitivity",
        "ppv",
        "f1",
        "mean_absolute_timing_ms",
        "median_absolute_timing_ms",
        "p75_absolute_timing_ms",
        "p90_absolute_timing_ms",
        "p95_absolute_timing_ms",
        "p99_absolute_timing_ms",
        "maximum_absolute_timing_ms",
        "mean_signed_timing_ms",
        "median_signed_timing_ms",
        "signed_timing_standard_deviation_ms",
        "evaluable_consecutive_rr",
        "rr_mae_ms",
        "rr_p95_absolute_error_ms",
    ]
    rows: list[dict[str, object]] = []
    for record_name, record_group in at_75.groupby("record", sort=False):
        row: dict[str, object] = {
            "record": record_name,
            "channel": str(record_group.iloc[0]["channel"]),
            "sampling_rate_hz": float(record_group.iloc[0]["sampling_rate_hz"]),
            "timing_match_tolerance_ms": 75.0,
            "timing_statistics_condition": (
                "one_to_one_true_positives_only; FP_and_FN_reported_separately"
            ),
        }
        for _, detector_row in record_group.iterrows():
            prefix = detector_prefix[str(detector_row["detector"])]
            for column in measurement_columns:
                row[f"{prefix}_{column}"] = detector_row[column]
        rows.append(row)
    distance = pd.DataFrame(rows).sort_values(
        "record", key=lambda values: values.astype(int)
    )
    distance.to_csv(output / "record_rpeak_distance_75ms.csv", index=False)

    x = np.arange(distance.shape[0])
    fig, axes = plt.subplots(2, 1, figsize=(17, 10), constrained_layout=True)
    for prefix, color, label in [
        ("neurokit", "#008C95", "NeuroKit"),
        ("zhai", "#D88400", "Zhai reimplementation"),
    ]:
        axes[0].plot(
            x,
            distance[f"{prefix}_median_absolute_timing_ms"],
            color=color,
            marker="o",
            markersize=3,
            linewidth=1.0,
            label=f"{label} median absolute error",
        )
        axes[0].plot(
            x,
            distance[f"{prefix}_p95_absolute_timing_ms"],
            color=color,
            linestyle="--",
            linewidth=1.0,
            label=f"{label} 95th percentile",
        )
        axes[1].plot(
            x,
            distance[f"{prefix}_f1"],
            color=color,
            marker="o",
            markersize=3,
            linewidth=1.0,
            label=f"{label} F1",
        )
    axes[0].set_ylabel("Distance from matched expert R annotation (ms)")
    axes[0].set_title(
        "Matched-beat timing distance; dashed lines show the 95th percentile"
    )
    axes[0].set_ylim(bottom=-1)
    axes[0].legend(loc="upper right", ncol=2)
    axes[1].set_ylabel("F1 at +/-75 ms")
    axes[1].set_xlabel("MIT--BIH record")
    axes[1].set_ylim(0.70, 1.005)
    axes[1].legend(loc="lower left")
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(distance["record"], rotation=90)
        axis.grid(alpha=0.2)
    fig.suptitle(
        "All 48 records: timing cannot be interpreted without missed/extra beats",
        fontsize=14,
    )
    fig.savefig(output / "record_rpeak_distance_to_expert_75ms.png", dpi=190)
    plt.close(fig)

    ranking_rows: list[pd.DataFrame] = []
    for prefix, label in [("neurokit", "NeuroKit"), ("zhai", "Zhai")]:
        subset = distance[
            [
                "record",
                f"{prefix}_p95_absolute_timing_ms",
                f"{prefix}_median_absolute_timing_ms",
                f"{prefix}_fp",
                f"{prefix}_fn",
                f"{prefix}_f1",
            ]
        ].copy()
        subset.columns = [
            "record",
            "p95_absolute_timing_ms",
            "median_absolute_timing_ms",
            "fp",
            "fn",
            "f1",
        ]
        subset.insert(1, "detector", label)
        ranking_rows.append(subset)
    ranking = pd.concat(ranking_rows, ignore_index=True).sort_values(
        ["p95_absolute_timing_ms", "median_absolute_timing_ms"],
        ascending=False,
    )
    ranking.to_csv(output / "worst_record_rpeak_timing_ranking_75ms.csv", index=False)


def _save_benchmark_figures(
    metrics: pd.DataFrame,
    pooled: pd.DataFrame,
    output: Path,
) -> None:
    at_75 = metrics[np.isclose(metrics["tolerance_ms"], 75.0)].copy()
    pivot = at_75.pivot(index="record", columns="detector", values="f1")
    pivot = pivot.sort_index(key=lambda values: values.astype(int))
    x = np.arange(pivot.shape[0])
    fig, axis = plt.subplots(figsize=(16, 6), constrained_layout=True)
    axis.plot(
        x,
        pivot["neurokit_300_strict"],
        marker="o",
        markersize=3,
        linewidth=1,
        color="#008C95",
        label="NeuroKit 300-ms strict",
    )
    axis.plot(
        x,
        pivot["zhai2023_reimplementation"],
        marker="o",
        markersize=3,
        linewidth=1,
        color="#D88400",
        label="Zhai 2023 reimplementation",
    )
    axis.set_xticks(x)
    axis.set_xticklabels(pivot.index, rotation=90)
    axis.set_ylim(0.74, 1.005)
    axis.set_ylabel("Record-level F1 at 75 ms")
    axis.set_xlabel("MIT--BIH record")
    axis.set_title(
        "NeuroKit and Zhai fail on different records; neither track universally dominates"
    )
    axis.grid(alpha=0.2)
    axis.legend(loc="lower left")
    fig.savefig(output / "record_f1_comparison_75ms.png", dpi=180)
    plt.close(fig)

    difficult = ["108", "113", "207", "222", "231"]
    label_offsets = {
        "108": (22, 4),
        "113": (4, 42),
        "207": (10, 8),
        "222": (10, 8),
        "231": (40, -14),
    }
    difficult_metrics = at_75[at_75["record"].isin(difficult)]
    fig, axis = plt.subplots(figsize=(10, 7), constrained_layout=True)
    for record in difficult:
        group = difficult_metrics[difficult_metrics["record"] == record].set_index(
            "detector"
        )
        if not {
            "neurokit_300_strict",
            "zhai2023_reimplementation",
        }.issubset(group.index):
            continue
        neurokit = group.loc["neurokit_300_strict"]
        zhai = group.loc["zhai2023_reimplementation"]
        axis.annotate(
            "",
            xy=(zhai["fp"], zhai["fn"]),
            xytext=(neurokit["fp"], neurokit["fn"]),
            arrowprops={"arrowstyle": "->", "color": "#999999", "lw": 1.2},
        )
        axis.scatter(
            neurokit["fp"],
            neurokit["fn"],
            color="#008C95",
            marker="x",
            s=65,
        )
        axis.scatter(
            zhai["fp"],
            zhai["fn"],
            color="#D88400",
            marker="D",
            s=45,
        )
        axis.annotate(
            record,
            (zhai["fp"], zhai["fn"]),
            xytext=label_offsets[record],
            textcoords="offset points",
            fontsize=9,
        )
    axis.scatter([], [], color="#008C95", marker="x", label="NeuroKit")
    axis.scatter([], [], color="#D88400", marker="D", label="Zhai")
    axis.set_xlabel("False positives")
    axis.set_ylabel("False negatives")
    axis.set_title(
        "Five difficult records: arrow shows change from NeuroKit to Zhai"
    )
    axis.grid(alpha=0.2)
    axis.legend(loc="upper right")
    fig.savefig(output / "difficult_records_fp_fn_change.png", dpi=180)
    plt.close(fig)

    tolerance_order = sorted(pooled["tolerance_ms"].unique())
    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    width = 0.36
    for offset, (detector, color, label) in enumerate(
        [
            ("neurokit_300_strict", "#008C95", "NeuroKit"),
            ("zhai2023_reimplementation", "#D88400", "Zhai"),
        ]
    ):
        values = (
            pooled[pooled["detector"] == detector]
            .set_index("tolerance_ms")
            .loc[tolerance_order, "f1"]
            .to_numpy()
        )
        axis.bar(
            np.arange(len(tolerance_order)) + (offset - 0.5) * width,
            values,
            width=width,
            color=color,
            label=label,
        )
    axis.set_xticks(np.arange(len(tolerance_order)))
    axis.set_xticklabels([f"{value:.2f} ms" for value in tolerance_order])
    axis.set_ylim(0.78, 1.0)
    axis.set_ylabel("Pooled F1")
    axis.set_xlabel("Matching tolerance")
    axis.set_title("Exact-timing conclusions change with evaluation tolerance")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    fig.savefig(output / "pooled_f1_by_tolerance.png", dpi=180)
    plt.close(fig)


def _parse_channel(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def _consecutive_rr_errors_ms(
    detected: np.ndarray,
    reference: np.ndarray,
    detected_indices: np.ndarray,
    reference_indices: np.ndarray,
    *,
    sampling_rate_hz: float,
) -> np.ndarray:
    """RR error where both detected and expert events are consecutive."""

    if detected_indices.size < 2:
        return np.array([], dtype=np.float64)
    consecutive = (
        (np.diff(detected_indices) == 1) & (np.diff(reference_indices) == 1)
    )
    first = np.flatnonzero(consecutive)
    if first.size == 0:
        return np.array([], dtype=np.float64)
    detected_rr = (
        detected[detected_indices[first + 1]] - detected[detected_indices[first]]
    )
    reference_rr = (
        reference[reference_indices[first + 1]] - reference[reference_indices[first]]
    )
    return (detected_rr - reference_rr) * 1000.0 / sampling_rate_hz


if __name__ == "__main__":
    main()
