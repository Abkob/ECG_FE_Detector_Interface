"""All-record ablation of experimental UNSW--NeuroKit--Zhai fusion.

The script deliberately keeps the published detector outputs intact.  UNSW
owns event existence; independently produced NeuroKit or Zhai timestamps may
replace an UNSW timestamp only after a unique bijective temporal association.
No expert annotation is used by the fusion rule itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ecg_cascade.fiducial_fusion import (
    FiducialFusionResult,
    fuse_unsw_neurokit_zhai,
)
from ecg_cascade.peaks import (
    detect_neurokit_gradient,
    detect_unsw,
)
from ecg_cascade.reliability import collapse_close_detections
from ecg_cascade.validation import match_peaks_to_reference, peak_metrics
from ecg_cascade.wfdb_io import load_wfdb_beat_annotations, load_wfdb_segment
from ecg_cascade.zhai import detect_zhai_template


DIFFICULT_RECORDS = ["108", "113", "207", "222", "231"]
ASSOCIATION_TOLERANCES_MS = (50.0, 75.0, 100.0, 150.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--records", nargs="*")
    parser.add_argument("--channel", default="0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    database = args.database_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = args.records or [
        line.strip()
        for line in (database / "RECORDS").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    channel = _parse_channel(args.channel)

    record_rows: list[dict[str, object]] = []
    event_audit_frames: list[pd.DataFrame] = []
    error_store: dict[str, dict[str, list[np.ndarray]]] = {}

    for record_name in records:
        record_path = database / record_name
        segment = load_wfdb_segment(record_path, channel=channel)
        reference, _ = load_wfdb_beat_annotations(
            record_path,
            start_sample=segment.start_sample,
            end_sample=segment.start_sample + segment.samples.size,
        )
        fs = float(segment.sampling_rate_hz)

        unsw_run = detect_unsw(segment.samples, fs, orientation="original")
        unsw = collapse_close_detections(
            unsw_run.peak_samples,
            unsw_run.analysis_signal,
            sampling_rate_hz=fs,
            exclusion_ms=150.0,
        )
        nk250 = detect_neurokit_gradient(
            segment.samples,
            fs,
            orientation="original",
            minimum_delay_ms=250.0,
            minimum_delay_inclusive=True,
        ).peak_samples
        nk300 = detect_neurokit_gradient(
            segment.samples,
            fs,
            orientation="original",
            minimum_delay_ms=300.0,
            minimum_delay_inclusive=False,
        ).peak_samples
        zhai = detect_zhai_template(segment.samples, fs, orientation="original")

        baselines = {
            "unsw_initial": unsw,
            "neurokit_250_inclusive": nk250,
            "neurokit_300_strict": nk300,
            "zhai2023_reimplementation": zhai.peak_samples,
        }
        for method, samples in baselines.items():
            row, errors = _evaluate_method(
                record_name=record_name,
                channel_name=segment.channel_name,
                fs=fs,
                method=method,
                detected=samples,
                reference=reference,
                fusion=None,
                association_tolerance_ms=np.nan,
            )
            record_rows.append(row)
            _store_errors(error_store, method, errors)

        for tolerance_ms in ASSOCIATION_TOLERANCES_MS:
            zhai_only = fuse_unsw_neurokit_zhai(
                unsw,
                np.array([], dtype=np.int64),
                zhai.peak_samples,
                sampling_rate_hz=fs,
                association_tolerance_ms=tolerance_ms,
                zhai_correlation_values=zhai.correlation_peak_values,
            )
            method = f"substitution_zhai_only_tol{int(tolerance_ms):03d}"
            row, errors = _evaluate_method(
                record_name=record_name,
                channel_name=segment.channel_name,
                fs=fs,
                method=method,
                detected=zhai_only.selected_samples,
                reference=reference,
                fusion=zhai_only,
                association_tolerance_ms=tolerance_ms,
            )
            record_rows.append(row)
            _store_errors(error_store, method, errors)

        for nk_label, nk_samples in [("nk250", nk250), ("nk300", nk300)]:
            for tolerance_ms in ASSOCIATION_TOLERANCES_MS:
                nk_only = fuse_unsw_neurokit_zhai(
                    unsw,
                    nk_samples,
                    np.array([], dtype=np.int64),
                    sampling_rate_hz=fs,
                    association_tolerance_ms=tolerance_ms,
                )
                method = f"substitution_{nk_label}_only_tol{int(tolerance_ms):03d}"
                row, errors = _evaluate_method(
                    record_name=record_name,
                    channel_name=segment.channel_name,
                    fs=fs,
                    method=method,
                    detected=nk_only.selected_samples,
                    reference=reference,
                    fusion=nk_only,
                    association_tolerance_ms=tolerance_ms,
                )
                record_rows.append(row)
                _store_errors(error_store, method, errors)

                fusion = fuse_unsw_neurokit_zhai(
                    unsw,
                    nk_samples,
                    zhai.peak_samples,
                    sampling_rate_hz=fs,
                    association_tolerance_ms=tolerance_ms,
                    zhai_correlation_values=zhai.correlation_peak_values,
                )
                method = f"fusion_{nk_label}_zhai_tol{int(tolerance_ms):03d}"
                row, errors = _evaluate_method(
                    record_name=record_name,
                    channel_name=segment.channel_name,
                    fs=fs,
                    method=method,
                    detected=fusion.selected_samples,
                    reference=reference,
                    fusion=fusion,
                    association_tolerance_ms=tolerance_ms,
                )
                record_rows.append(row)
                _store_errors(error_store, method, errors)
                audit = fusion.events.copy()
                audit.insert(0, "association_tolerance_ms", tolerance_ms)
                audit.insert(0, "neurokit_configuration", nk_label)
                audit.insert(0, "record", record_name)
                event_audit_frames.append(audit)

        print(f"completed {record_name}", flush=True)

    record_metrics = pd.DataFrame(record_rows)
    pooled = _pool_metrics(record_metrics, error_store)
    record_metrics.to_csv(output / "record_metrics.csv", index=False)
    pooled.to_csv(output / "pooled_metrics.csv", index=False)
    pd.concat(event_audit_frames, ignore_index=True).to_csv(
        output / "fusion_event_decisions.csv.gz",
        index=False,
        compression="gzip",
    )
    difficult = record_metrics[
        record_metrics["record"].isin(DIFFICULT_RECORDS)
    ].copy()
    difficult.to_csv(output / "five_difficult_records.csv", index=False)
    decision = _build_decision_summary(pooled)
    (output / "decision_summary.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    _save_plots(record_metrics, pooled, decision, output)
    (output / "benchmark_metadata.json").write_text(
        json.dumps(
            {
                "database_dir": str(database),
                "records": records,
                "channel": args.channel,
                "expert_match_tolerance_ms": 75.0,
                "association_tolerances_ms": list(ASSOCIATION_TOLERANCES_MS),
                "unsw_role": "beat_existence_owner",
                "timestamp_priority": ["neurokit", "zhai", "unsw"],
                "neurokit_variants": {
                    "nk250": "250_ms_inclusive_project_experiment",
                    "nk300": "300_ms_strict_unmodified_baseline",
                },
                "zhai_role": "independent_paper_reimplementation_fiducial",
                "zhai_correlation_threshold": None,
                "zhai_threshold_reason": (
                    "No universal high-confidence correlation cutoff was "
                    "reported; correlation is exported, not thresholded."
                ),
                "raw_amplitude_argmax_used": False,
                "automatic_expert_informed_selection": False,
                "experimental_status": (
                    "Exact three-source priority/substitution is a project "
                    "ablation, not a published architecture."
                ),
                "published_support": {
                    "Ho_2024_DOI": "10.22489/CinC.2024.084",
                    "Zhai_2023_DOI": "10.3934/mbe.2023848",
                    "Khamis_2016_DOI": "10.1109/TBME.2016.2549060",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(pooled.to_string(index=False))
    print(json.dumps(decision, indent=2))


def _evaluate_method(
    *,
    record_name: str,
    channel_name: str,
    fs: float,
    method: str,
    detected: np.ndarray,
    reference: np.ndarray,
    fusion: FiducialFusionResult | None,
    association_tolerance_ms: float,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    metrics = peak_metrics(
        detected,
        reference,
        sampling_rate_hz=fs,
        tolerance_ms=75.0,
    )
    matches = match_peaks_to_reference(
        detected,
        reference,
        sampling_rate_hz=fs,
        tolerance_ms=75.0,
    )
    timing_errors = (
        detected[matches.detected_indices] - reference[matches.reference_indices]
    ) * 1000.0 / fs
    rr_all, rr_same_source, rr_same_independent = _rr_errors(
        detected,
        reference,
        matches.detected_indices,
        matches.reference_indices,
        sampling_rate_hz=fs,
        fusion=fusion,
    )
    if fusion is None:
        source_counts = {"neurokit": 0, "zhai": 0, "unsw": 0}
        switches = 0
        switch_fraction = np.nan
        monotonicity_fallbacks = 0
    else:
        counts = fusion.events["selected_source"].value_counts()
        source_counts = {
            "neurokit": int(counts.get("neurokit", 0)),
            "zhai": int(counts.get("zhai", 0)),
            "unsw": int(counts.get("unsw", 0)),
        }
        switches = int(fusion.rr_intervals["source_transition"].sum())
        switch_fraction = float(fusion.rr_intervals["source_transition"].mean())
        monotonicity_fallbacks = int(fusion.events["monotonicity_fallback"].sum())
    row = {
        "record": record_name,
        "channel": channel_name,
        "sampling_rate_hz": fs,
        "method": method,
        "association_tolerance_ms": association_tolerance_ms,
        **metrics,
        "rr_evaluable_count": int(rr_all.size),
        "rr_mae_ms": _mae(rr_all),
        "rr_p95_absolute_error_ms": _p95(rr_all),
        "same_source_rr_evaluable_count": int(rr_same_source.size),
        "same_source_rr_mae_ms": _mae(rr_same_source),
        "source_transition_rr_evaluable_count": int(
            rr_all.size - rr_same_source.size
        ),
        "source_transition_rr_mae_ms": _complement_mae(
            rr_all, rr_same_source
        ),
        "same_independent_source_rr_evaluable_count": int(rr_same_independent.size),
        "same_independent_source_rr_mae_ms": _mae(rr_same_independent),
        "selected_neurokit_count": source_counts["neurokit"],
        "selected_zhai_count": source_counts["zhai"],
        "selected_unsw_count": source_counts["unsw"],
        "source_switch_count": switches,
        "source_switch_fraction": switch_fraction,
        "monotonicity_fallback_count": monotonicity_fallbacks,
    }
    return row, {
        "timing": timing_errors,
        "rr_all": rr_all,
        "rr_same_source": rr_same_source,
        "rr_same_independent": rr_same_independent,
    }


def _rr_errors(
    detected: np.ndarray,
    reference: np.ndarray,
    detected_indices: np.ndarray,
    reference_indices: np.ndarray,
    *,
    sampling_rate_hz: float,
    fusion: FiducialFusionResult | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_errors: list[float] = []
    same_source_errors: list[float] = []
    same_independent_errors: list[float] = []
    for position in range(max(0, detected_indices.size - 1)):
        detected_start = int(detected_indices[position])
        detected_end = int(detected_indices[position + 1])
        reference_start = int(reference_indices[position])
        reference_end = int(reference_indices[position + 1])
        if detected_end != detected_start + 1 or reference_end != reference_start + 1:
            continue
        error = (
            (detected[detected_end] - detected[detected_start])
            - (reference[reference_end] - reference[reference_start])
        ) * 1000.0 / sampling_rate_hz
        all_errors.append(float(error))
        if fusion is None:
            same_source_errors.append(float(error))
            same_independent_errors.append(float(error))
            continue
        rr_row = fusion.rr_intervals.iloc[detected_start]
        if bool(rr_row["same_source"]):
            same_source_errors.append(float(error))
        if bool(rr_row["same_independent_source"]):
            same_independent_errors.append(float(error))
    return (
        np.asarray(all_errors, dtype=float),
        np.asarray(same_source_errors, dtype=float),
        np.asarray(same_independent_errors, dtype=float),
    )


def _store_errors(
    store: dict[str, dict[str, list[np.ndarray]]],
    method: str,
    errors: dict[str, np.ndarray],
) -> None:
    destination = store.setdefault(
        method,
        {"timing": [], "rr_all": [], "rr_same_source": [], "rr_same_independent": []},
    )
    for name, values in errors.items():
        destination[name].append(values)


def _pool_metrics(
    records: pd.DataFrame,
    error_store: dict[str, dict[str, list[np.ndarray]]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, group in records.groupby("method", sort=True):
        tp = int(group["tp"].sum())
        fp = int(group["fp"].sum())
        fn = int(group["fn"].sum())
        sensitivity = tp / (tp + fn)
        ppv = tp / (tp + fp)
        f1 = 2 * sensitivity * ppv / (sensitivity + ppv)
        errors = error_store[method]
        timing = _concatenate_nonempty(errors["timing"])
        rr_all = _concatenate_nonempty(errors["rr_all"])
        rr_same = _concatenate_nonempty(errors["rr_same_source"])
        rr_independent = _concatenate_nonempty(errors["rr_same_independent"])
        detected = int(group["detected"].sum())
        rows.append(
            {
                "method": method,
                "records": int(group.shape[0]),
                "reference": int(group["reference"].sum()),
                "detected": detected,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "sensitivity": sensitivity,
                "ppv": ppv,
                "f1": f1,
                "mean_absolute_timing_ms": _mae(timing),
                "median_absolute_timing_ms": (
                    float(np.median(np.abs(timing))) if timing.size else np.nan
                ),
                "p95_absolute_timing_ms": _p95(timing),
                "rr_evaluable_count": int(rr_all.size),
                "rr_mae_ms": _mae(rr_all),
                "rr_p95_absolute_error_ms": _p95(rr_all),
                "same_source_rr_evaluable_count": int(rr_same.size),
                "same_source_rr_mae_ms": _mae(rr_same),
                "source_transition_rr_evaluable_count": int(
                    rr_all.size - rr_same.size
                ),
                "source_transition_rr_mae_ms": _complement_mae(
                    rr_all, rr_same
                ),
                "same_independent_source_rr_evaluable_count": int(rr_independent.size),
                "same_independent_source_rr_mae_ms": _mae(rr_independent),
                "selected_neurokit_count": int(group["selected_neurokit_count"].sum()),
                "selected_zhai_count": int(group["selected_zhai_count"].sum()),
                "selected_unsw_count": int(group["selected_unsw_count"].sum()),
                "source_switch_count": int(group["source_switch_count"].sum()),
                "source_switch_fraction": (
                    float(group["source_switch_count"].sum()) / max(1, detected - group.shape[0])
                    if int(
                        group[
                            [
                                "selected_neurokit_count",
                                "selected_zhai_count",
                                "selected_unsw_count",
                            ]
                        ].to_numpy().sum()
                    )
                    else np.nan
                ),
                "monotonicity_fallback_count": int(
                    group["monotonicity_fallback_count"].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _build_decision_summary(pooled: pd.DataFrame) -> dict[str, object]:
    unsw = pooled.loc[pooled["method"] == "unsw_initial"].iloc[0]
    fusion = pooled[pooled["method"].str.startswith("fusion_")].copy()
    fusion = fusion.sort_values(
        ["rr_mae_ms", "f1", "p95_absolute_timing_ms"],
        ascending=[True, False, True],
    )
    best = fusion.iloc[0]
    pareto = fusion[
        (fusion["f1"] >= float(unsw["f1"]))
        & (fusion["rr_mae_ms"] <= float(unsw["rr_mae_ms"]))
        & (fusion["p95_absolute_timing_ms"] <= float(unsw["p95_absolute_timing_ms"]))
    ]
    return {
        "descriptive_best_rr_fusion": str(best["method"]),
        "automatic_acceptance": False,
        "strictly_dominates_unsw_count": int(pareto.shape[0]),
        "strictly_dominates_unsw_methods": pareto["method"].tolist(),
        "unsw": _summary_values(unsw),
        "best_rr_fusion": _summary_values(best),
        "warning": (
            "Choosing a tolerance on these same 48 records is development-set "
            "selection and requires independent validation."
        ),
    }


def _summary_values(row: pd.Series) -> dict[str, object]:
    fields = [
        "method",
        "f1",
        "mean_absolute_timing_ms",
        "median_absolute_timing_ms",
        "p95_absolute_timing_ms",
        "rr_mae_ms",
        "rr_p95_absolute_error_ms",
        "selected_neurokit_count",
        "selected_zhai_count",
        "selected_unsw_count",
        "source_switch_count",
        "source_switch_fraction",
    ]
    payload: dict[str, object] = {}
    for field in fields:
        value = row[field]
        payload[field] = str(value) if field == "method" else _finite_or_none(value)
    return payload


def _save_plots(
    records: pd.DataFrame,
    pooled: pd.DataFrame,
    decision: dict[str, object],
    output: Path,
) -> None:
    records = records.copy()
    records["record"] = records["record"].astype(str)
    best_method = str(decision["descriptive_best_rr_fusion"])
    displayed = pooled[
        pooled["method"].isin(
            [
                "unsw_initial",
                "neurokit_250_inclusive",
                "neurokit_300_strict",
                "zhai2023_reimplementation",
                best_method,
            ]
        )
    ].copy()
    labels = displayed["method"].str.replace("_", " ").tolist()
    x = np.arange(displayed.shape[0])
    fig, axes = plt.subplots(3, 1, figsize=(13, 12), constrained_layout=True)
    axes[0].bar(x, displayed["f1"], color="#315A7D")
    axes[0].set_ylabel("Pooled F1")
    axes[0].set_ylim(min(0.96, float(displayed["f1"].min()) - 0.005), 1.0)
    axes[1].bar(x, displayed["mean_absolute_timing_ms"], color="#008C95")
    axes[1].set_ylabel("Mean absolute timing error (ms)")
    axes[2].bar(x, displayed["rr_mae_ms"], color="#D88400")
    axes[2].set_ylabel("Consecutive RR MAE (ms)")
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=18, ha="right")
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("UNSW--NeuroKit--Zhai fusion ablation: all 48 MIT--BIH records")
    fig.savefig(output / "pooled_fusion_comparison.png", dpi=190)
    plt.close(fig)

    difficult = records[
        records["record"].isin(DIFFICULT_RECORDS)
        & records["method"].isin(["unsw_initial", best_method])
    ].copy()
    pivot_fp = difficult.pivot(index="record", columns="method", values="fp").loc[
        DIFFICULT_RECORDS
    ]
    pivot_fn = difficult.pivot(index="record", columns="method", values="fn").loc[
        DIFFICULT_RECORDS
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    width = 0.36
    x = np.arange(len(DIFFICULT_RECORDS))
    for offset, (method, color, label) in enumerate(
        [
            ("unsw_initial", "#315A7D", "UNSW initial"),
            (best_method, "#D88400", "Descriptive best fusion"),
        ]
    ):
        axes[0].bar(x + (offset - 0.5) * width, pivot_fp[method], width, color=color, label=label)
        axes[1].bar(x + (offset - 0.5) * width, pivot_fn[method], width, color=color, label=label)
    for axis, title in zip(axes, ["False positives", "False negatives"], strict=True):
        axis.set_xticks(x)
        axis.set_xticklabels(DIFFICULT_RECORDS)
        axis.set_xlabel("MIT--BIH record")
        axis.set_ylabel("Count")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
        axis.legend()
    fig.suptitle("Five difficult records: UNSW event ownership versus substituted timing")
    fig.savefig(output / "five_difficult_fusion_errors.png", dpi=190)
    plt.close(fig)

    fusion = pooled[pooled["method"].str.startswith("fusion_")].copy()
    source_total = fusion[
        ["selected_neurokit_count", "selected_zhai_count", "selected_unsw_count"]
    ].sum(axis=1)
    fig, axis = plt.subplots(figsize=(14, 6), constrained_layout=True)
    bottom = np.zeros(fusion.shape[0])
    for column, color, label in [
        ("selected_neurokit_count", "#008C95", "NeuroKit timestamp"),
        ("selected_zhai_count", "#D88400", "Zhai timestamp"),
        ("selected_unsw_count", "#777777", "UNSW fallback"),
    ]:
        fraction = fusion[column].to_numpy() / source_total.to_numpy()
        axis.bar(np.arange(fusion.shape[0]), fraction, bottom=bottom, color=color, label=label)
        bottom += fraction
    axis.set_xticks(np.arange(fusion.shape[0]))
    axis.set_xticklabels(fusion["method"].str.replace("fusion_", ""), rotation=35, ha="right")
    axis.set_ylabel("Fraction of retained UNSW events")
    axis.set_ylim(0, 1)
    axis.set_title("Which detector supplied each final timestamp?")
    axis.legend(loc="upper right")
    axis.grid(axis="y", alpha=0.2)
    fig.savefig(output / "fusion_timestamp_source_usage.png", dpi=190)
    plt.close(fig)

    comparison = records[
        records["method"].isin(["unsw_initial", best_method])
    ].pivot(index="record", columns="method", values="rr_mae_ms")
    comparison = comparison.sort_index(key=lambda values: values.astype(int))
    delta = comparison[best_method] - comparison["unsw_initial"]
    colors = np.where(delta <= 0, "#2A8C6A", "#C73E32")
    fig, axis = plt.subplots(figsize=(16, 6), constrained_layout=True)
    axis.bar(np.arange(delta.size), delta.to_numpy(), color=colors)
    axis.axhline(0, color="black", linewidth=0.9)
    axis.set_xticks(np.arange(delta.size))
    axis.set_xticklabels(delta.index, rotation=90)
    axis.set_ylabel("Fusion RR MAE minus UNSW RR MAE (ms)")
    axis.set_xlabel("MIT--BIH record")
    axis.set_title(
        "Pooled improvement is not universal: green improved, red deteriorated"
    )
    axis.grid(axis="y", alpha=0.2)
    fig.savefig(output / "record_rr_mae_change_from_unsw.png", dpi=190)
    plt.close(fig)

    transition_methods = [
        "substitution_nk300_only_tol050",
        "substitution_zhai_only_tol050",
        "fusion_nk300_zhai_tol050",
    ]
    transition = pooled.set_index("method").loc[transition_methods]
    transition_mae = []
    for _, row in transition.iterrows():
        all_count = int(row["rr_evaluable_count"])
        same_count = int(row["same_source_rr_evaluable_count"])
        count = all_count - same_count
        transition_mae.append(
            (
                all_count * float(row["rr_mae_ms"])
                - same_count * float(row["same_source_rr_mae_ms"])
            )
            / count
        )
    x = np.arange(len(transition_methods))
    width = 0.36
    fig, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    axis.bar(
        x - width / 2,
        transition["same_source_rr_mae_ms"],
        width,
        color="#2A8C6A",
        label="Both RR boundaries use same source",
    )
    axis.bar(
        x + width / 2,
        transition_mae,
        width,
        color="#C73E32",
        label="RR crosses timestamp sources",
    )
    axis.set_xticks(x)
    axis.set_xticklabels(
        ["NeuroKit only", "Zhai only", "NeuroKit then Zhai"]
    )
    axis.set_ylabel("RR MAE (ms)")
    axis.set_title("Timestamp-source switching creates large RR errors")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    fig.savefig(output / "source_transition_rr_error.png", dpi=190)
    plt.close(fig)


def _concatenate_nonempty(arrays: list[np.ndarray]) -> np.ndarray:
    nonempty = [array for array in arrays if array.size]
    return np.concatenate(nonempty) if nonempty else np.array([], dtype=float)


def _mae(values: np.ndarray) -> float:
    return float(np.mean(np.abs(values))) if values.size else np.nan


def _p95(values: np.ndarray) -> float:
    return float(np.percentile(np.abs(values), 95)) if values.size else np.nan


def _complement_mae(all_values: np.ndarray, retained_values: np.ndarray) -> float:
    """MAE of the excluded subset from counts and absolute-error sums."""

    count = int(all_values.size - retained_values.size)
    if count <= 0:
        return np.nan
    absolute_sum = float(np.abs(all_values).sum() - np.abs(retained_values).sum())
    return absolute_sum / count


def _finite_or_none(value: object) -> float | int | None:
    numeric = float(value)
    if not np.isfinite(numeric):
        return None
    return int(numeric) if numeric.is_integer() else numeric


def _parse_channel(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


if __name__ == "__main__":
    main()
