"""Benchmark unchanged UNSW RR timestamps with NeuroKit context on MIT--BIH."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ecg_cascade.peaks import detect_neurokit_gradient, detect_unsw
from ecg_cascade.reliability import collapse_close_detections
from ecg_cascade.rr_reliability_context import build_unsw_neurokit_rr_context
from ecg_cascade.validation import match_peaks_to_reference, peak_metrics
from ecg_cascade.wfdb_io import load_wfdb_beat_annotations, load_wfdb_segment


TOLERANCES_MS = (50.0, 75.0, 100.0, 150.0)
AGREEMENT_WINDOW_S = 10.0
DIFFICULT_RECORDS = ["108", "113", "207", "222", "231"]


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
    rr_audits: list[pd.DataFrame] = []
    pooled_errors: dict[float, dict[str, list[np.ndarray]]] = {
        tolerance: {"all": [], "supported": [], "unsupported": []}
        for tolerance in TOLERANCES_MS
    }
    for record_name in records:
        record_path = database / record_name
        segment = load_wfdb_segment(record_path, channel=channel)
        reference, symbols = load_wfdb_beat_annotations(
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
        neurokit = detect_neurokit_gradient(
            segment.samples,
            fs,
            orientation="original",
            minimum_delay_ms=300.0,
            minimum_delay_inclusive=False,
        ).peak_samples
        detector_metrics = peak_metrics(
            unsw,
            reference,
            sampling_rate_hz=fs,
            tolerance_ms=75.0,
        )

        for tolerance_ms in TOLERANCES_MS:
            context = build_unsw_neurokit_rr_context(
                unsw,
                neurokit,
                sampling_rate_hz=fs,
                support_tolerance_ms=tolerance_ms,
                agreement_window_s=AGREEMENT_WINDOW_S,
            )
            audit, errors = _audit_rr_against_reference(
                context.rr_intervals,
                unsw,
                reference,
                symbols,
                sampling_rate_hz=fs,
            )
            audit.insert(0, "support_tolerance_ms", tolerance_ms)
            audit.insert(0, "record", record_name)
            rr_audits.append(audit)
            for key, values in errors.items():
                pooled_errors[tolerance_ms][key].append(values)

            supported = context.rr_intervals["rr_supported"].to_numpy(bool)
            evaluable = audit["expert_evaluable"].to_numpy(bool)
            qsqi = audit.loc[evaluable, "trailing_qsqi_fraction_at_end"].to_numpy(float)
            absolute_error = audit.loc[evaluable, "absolute_rr_error_ms"].to_numpy(float)
            spearman = _spearman(qsqi, absolute_error)
            record_rows.append(
                {
                    "record": record_name,
                    "channel": segment.channel_name,
                    "sampling_rate_hz": fs,
                    "support_tolerance_ms": tolerance_ms,
                    **detector_metrics,
                    "rr_count": int(context.rr_intervals.shape[0]),
                    "supported_rr_count": int(supported.sum()),
                    "supported_rr_coverage": float(supported.mean()),
                    "evaluable_rr_count": int(errors["all"].size),
                    "all_rr_mae_ms": _mae(errors["all"]),
                    "supported_evaluable_rr_count": int(errors["supported"].size),
                    "supported_rr_mae_ms": _mae(errors["supported"]),
                    "unsupported_evaluable_rr_count": int(errors["unsupported"].size),
                    "unsupported_rr_mae_ms": _mae(errors["unsupported"]),
                    "supported_to_unsupported_mae_ratio": _safe_ratio(
                        _mae(errors["supported"]),
                        _mae(errors["unsupported"]),
                    ),
                    "trailing_qsqi_spearman_with_absolute_rr_error": spearman,
                    "mean_trailing_qsqi": float(
                        context.events["trailing_qsqi_fraction"].mean()
                    ),
                    "mean_centered_qsqi": float(
                        context.events["centered_qsqi_fraction"].mean()
                    ),
                }
            )
        print(f"completed {record_name}", flush=True)

    records_frame = pd.DataFrame(record_rows)
    rr_audit = pd.concat(rr_audits, ignore_index=True)
    pooled = _pool(records_frame, pooled_errors, rr_audit)
    bins = _qsqi_error_bins(rr_audit)
    records_frame.to_csv(output / "record_metrics.csv", index=False)
    pooled.to_csv(output / "pooled_metrics.csv", index=False)
    bins.to_csv(output / "trailing_qsqi_error_bins.csv", index=False)
    rr_audit.to_csv(
        output / "rr_reliability_audit.csv.gz", index=False, compression="gzip"
    )
    records_frame[
        records_frame["record"].isin(DIFFICULT_RECORDS)
    ].to_csv(output / "five_difficult_records.csv", index=False)
    decision = _decision_summary(pooled)
    (output / "decision_summary.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    _save_plots(records_frame, pooled, bins, output)
    (output / "benchmark_metadata.json").write_text(
        json.dumps(
            {
                "records": records,
                "database_dir": str(database),
                "channel": args.channel,
                "timestamp_owner": "unsw_initial_unchanged",
                "neurokit_role": "context_only_no_timestamp_substitution",
                "neurokit_configuration": "300_ms_strict_unmodified_baseline",
                "support_tolerances_ms": list(TOLERANCES_MS),
                "expert_match_tolerance_ms": 75.0,
                "agreement_formula": "2M/(N_UNSW+N_NeuroKit)",
                "agreement_window_s": AGREEMENT_WINDOW_S,
                "centered_window_status": "offline_bSQI_style_context",
                "trailing_window_status": (
                    "causal_project_adaptation_not_Zhao_2018_literal_protocol"
                ),
                "agreement_is_probability": False,
                "agreement_is_artifact_label": False,
                "raw_amplitude_argmax_used": False,
                "published_support": {
                    "Ho_2024_DOI": "10.22489/CinC.2024.084",
                    "Zhao_Zhang_2018_DOI": "10.3389/fphys.2018.00727",
                    "Li_Mark_Clifford_2008_DOI": "10.1088/0967-3334/29/1/008",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(pooled.to_string(index=False))
    print(json.dumps(decision, indent=2))


def _audit_rr_against_reference(
    intervals: pd.DataFrame,
    unsw: np.ndarray,
    reference: np.ndarray,
    symbols: np.ndarray,
    *,
    sampling_rate_hz: float,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    matches = match_peaks_to_reference(
        unsw,
        reference,
        sampling_rate_hz=sampling_rate_hz,
        tolerance_ms=75.0,
    )
    unsw_to_reference = dict(
        zip(matches.detected_indices.tolist(), matches.reference_indices.tolist(), strict=True)
    )
    audit = intervals.copy()
    expert_evaluable = np.zeros(intervals.shape[0], dtype=bool)
    expert_rr_ms = np.full(intervals.shape[0], np.nan)
    signed_error_ms = np.full(intervals.shape[0], np.nan)
    start_symbol = np.full(intervals.shape[0], "", dtype=object)
    end_symbol = np.full(intervals.shape[0], "", dtype=object)
    for rr_index in range(intervals.shape[0]):
        if rr_index not in unsw_to_reference or rr_index + 1 not in unsw_to_reference:
            continue
        reference_start = unsw_to_reference[rr_index]
        reference_end = unsw_to_reference[rr_index + 1]
        if reference_end != reference_start + 1:
            continue
        truth = (
            reference[reference_end] - reference[reference_start]
        ) * 1000.0 / sampling_rate_hz
        expert_evaluable[rr_index] = True
        expert_rr_ms[rr_index] = truth
        signed_error_ms[rr_index] = float(intervals.iloc[rr_index]["rr_ms"]) - truth
        start_symbol[rr_index] = str(symbols[reference_start])
        end_symbol[rr_index] = str(symbols[reference_end])
    audit["expert_evaluable"] = expert_evaluable
    audit["expert_rr_ms"] = expert_rr_ms
    audit["signed_rr_error_ms"] = signed_error_ms
    audit["absolute_rr_error_ms"] = np.abs(signed_error_ms)
    audit["expert_start_symbol"] = start_symbol
    audit["expert_end_symbol"] = end_symbol
    supported_mask = audit["rr_supported"].to_numpy(bool)
    return audit, {
        "all": signed_error_ms[expert_evaluable],
        "supported": signed_error_ms[expert_evaluable & supported_mask],
        "unsupported": signed_error_ms[expert_evaluable & ~supported_mask],
    }


def _pool(
    records: pd.DataFrame,
    error_store: dict[float, dict[str, list[np.ndarray]]],
    rr_audit: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for tolerance, group in records.groupby("support_tolerance_ms", sort=True):
        errors = {
            key: _concatenate_nonempty(values)
            for key, values in error_store[float(tolerance)].items()
        }
        tp = int(group["tp"].sum())
        fp = int(group["fp"].sum())
        fn = int(group["fn"].sum())
        sensitivity = tp / (tp + fn)
        ppv = tp / (tp + fp)
        f1 = 2 * sensitivity * ppv / (sensitivity + ppv)
        supported_count = int(group["supported_rr_count"].sum())
        rr_count = int(group["rr_count"].sum())
        supported_mae = _mae(errors["supported"])
        unsupported_mae = _mae(errors["unsupported"])
        evaluable_audit = rr_audit[
            np.isclose(rr_audit["support_tolerance_ms"], tolerance)
            & rr_audit["expert_evaluable"]
        ]
        rows.append(
            {
                "support_tolerance_ms": tolerance,
                "records": int(group.shape[0]),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "f1": f1,
                "rr_count": rr_count,
                "supported_rr_count": supported_count,
                "supported_rr_coverage": supported_count / rr_count,
                "evaluable_rr_count": int(errors["all"].size),
                "all_unsw_rr_mae_ms": _mae(errors["all"]),
                "supported_evaluable_rr_count": int(errors["supported"].size),
                "supported_unsw_rr_mae_ms": supported_mae,
                "unsupported_evaluable_rr_count": int(errors["unsupported"].size),
                "unsupported_unsw_rr_mae_ms": unsupported_mae,
                "supported_to_unsupported_mae_ratio": _safe_ratio(
                    supported_mae, unsupported_mae
                ),
                "mean_record_supported_coverage": float(
                    group["supported_rr_coverage"].mean()
                ),
                "median_record_supported_coverage": float(
                    group["supported_rr_coverage"].median()
                ),
                "mean_record_trailing_qsqi": float(group["mean_trailing_qsqi"].mean()),
                "mean_record_qsqi_error_spearman": float(
                    group["trailing_qsqi_spearman_with_absolute_rr_error"].mean()
                ),
                "pooled_trailing_qsqi_error_spearman": _spearman(
                    evaluable_audit["trailing_qsqi_fraction_at_end"].to_numpy(float),
                    evaluable_audit["absolute_rr_error_ms"].to_numpy(float),
                ),
                "pooled_centered_qsqi_error_spearman": _spearman(
                    evaluable_audit["centered_qsqi_fraction_at_end"].to_numpy(float),
                    evaluable_audit["absolute_rr_error_ms"].to_numpy(float),
                ),
            }
        )
    return pd.DataFrame(rows)


def _qsqi_error_bins(audit: pd.DataFrame) -> pd.DataFrame:
    evaluable = audit[audit["expert_evaluable"]].copy()
    boundaries = [-np.inf, 0.80, 0.90, 0.95, 0.999999999, np.inf]
    labels = ["<0.80", "0.80-<0.90", "0.90-<0.95", "0.95-<1.00", "1.00"]
    evaluable["trailing_qsqi_bin"] = pd.cut(
        evaluable["trailing_qsqi_fraction_at_end"],
        bins=boundaries,
        labels=labels,
        right=False,
    )
    rows: list[dict[str, object]] = []
    for (tolerance, score_bin), group in evaluable.groupby(
        ["support_tolerance_ms", "trailing_qsqi_bin"], observed=False
    ):
        rows.append(
            {
                "support_tolerance_ms": tolerance,
                "trailing_qsqi_bin": str(score_bin),
                "evaluable_rr_count": int(group.shape[0]),
                "rr_mae_ms": (
                    float(group["absolute_rr_error_ms"].mean())
                    if not group.empty
                    else np.nan
                ),
                "median_absolute_rr_error_ms": (
                    float(group["absolute_rr_error_ms"].median())
                    if not group.empty
                    else np.nan
                ),
                "supported_fraction": (
                    float(group["rr_supported"].mean()) if not group.empty else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _decision_summary(pooled: pd.DataFrame) -> dict[str, object]:
    best = pooled.sort_values(
        ["supported_unsw_rr_mae_ms", "supported_rr_coverage"],
        ascending=[True, False],
    ).iloc[0]
    coverage_candidates = pooled[pooled["supported_rr_coverage"] >= 0.90]
    best_coverage = (
        coverage_candidates.sort_values("supported_unsw_rr_mae_ms").iloc[0]
        if not coverage_candidates.empty
        else None
    )
    return {
        "timestamp_owner": "unsw_initial_unchanged",
        "full_stream_metrics_equal_unsw": True,
        "descriptive_lowest_supported_rr_mae_tolerance_ms": float(
            best["support_tolerance_ms"]
        ),
        "descriptive_lowest_supported_rr_mae_ms": float(
            best["supported_unsw_rr_mae_ms"]
        ),
        "coverage_at_that_tolerance": float(best["supported_rr_coverage"]),
        "best_tolerance_with_at_least_90_percent_pooled_coverage_ms": (
            float(best_coverage["support_tolerance_ms"])
            if best_coverage is not None
            else None
        ),
        "automatic_artifact_interpretation": False,
        "automatic_acceptance": False,
        "warning": (
            "Tolerance selection on MIT--BIH is development-set selection; "
            "patient-specific held-out validation remains required."
        ),
    }


def _save_plots(
    records: pd.DataFrame,
    pooled: pd.DataFrame,
    bins: pd.DataFrame,
    output: Path,
) -> None:
    x = np.arange(pooled.shape[0])
    labels = [f"+/-{int(value)} ms" for value in pooled["support_tolerance_ms"]]
    fig, left = plt.subplots(figsize=(11, 6), constrained_layout=True)
    right = left.twinx()
    left.plot(
        x,
        pooled["supported_unsw_rr_mae_ms"],
        color="#2A8C6A",
        marker="o",
        label="Supported UNSW RR MAE",
    )
    left.plot(
        x,
        pooled["unsupported_unsw_rr_mae_ms"],
        color="#C73E32",
        marker="o",
        label="Unsupported UNSW RR MAE",
    )
    right.bar(
        x,
        pooled["supported_rr_coverage"],
        color="#315A7D",
        alpha=0.25,
        label="Supported coverage",
    )
    left.set_xticks(x)
    left.set_xticklabels(labels)
    left.set_ylabel("RR MAE (ms)")
    right.set_ylabel("Supported RR coverage")
    right.set_ylim(0, 1.05)
    left.set_title("NeuroKit context selects a lower-error subset without changing UNSW RR")
    left.grid(alpha=0.2)
    handles1, labels1 = left.get_legend_handles_labels()
    handles2, labels2 = right.get_legend_handles_labels()
    left.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    fig.savefig(output / "tolerance_coverage_rr_error.png", dpi=190)
    plt.close(fig)

    selected_bins = bins[np.isclose(bins["support_tolerance_ms"], 50.0)]
    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    axis.bar(
        np.arange(selected_bins.shape[0]),
        selected_bins["rr_mae_ms"],
        color="#6A5A9C",
    )
    axis.set_xticks(np.arange(selected_bins.shape[0]))
    axis.set_xticklabels(selected_bins["trailing_qsqi_bin"])
    axis.set_xlabel("Trailing 10-second detector-agreement score")
    axis.set_ylabel("UNSW RR MAE (ms)")
    axis.set_title("qSQI is agreement context, not a monotonic correctness probability")
    axis.grid(axis="y", alpha=0.2)
    fig.savefig(output / "trailing_qsqi_vs_rr_error_50ms.png", dpi=190)
    plt.close(fig)

    records = records.copy()
    records["record"] = records["record"].astype(str)
    at_50 = records[np.isclose(records["support_tolerance_ms"], 50.0)].sort_values(
        "record", key=lambda values: values.astype(int)
    )
    colors = [
        "#C73E32" if record in DIFFICULT_RECORDS else "#315A7D"
        for record in at_50["record"]
    ]
    fig, axis = plt.subplots(figsize=(16, 6), constrained_layout=True)
    axis.bar(np.arange(at_50.shape[0]), at_50["supported_rr_coverage"], color=colors)
    axis.set_xticks(np.arange(at_50.shape[0]))
    axis.set_xticklabels(at_50["record"], rotation=90)
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("Supported RR fraction")
    axis.set_xlabel("MIT--BIH record")
    axis.set_title("Coverage is strongly record dependent; difficult records shown in red")
    axis.grid(axis="y", alpha=0.2)
    fig.savefig(output / "record_supported_coverage_50ms.png", dpi=190)
    plt.close(fig)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3 or np.unique(x[finite]).size < 2:
        return np.nan
    return float(stats.spearmanr(x[finite], y[finite]).statistic)


def _concatenate_nonempty(arrays: list[np.ndarray]) -> np.ndarray:
    arrays = [array for array in arrays if array.size]
    return np.concatenate(arrays) if arrays else np.array([], dtype=float)


def _mae(values: np.ndarray) -> float:
    return float(np.mean(np.abs(values))) if values.size else np.nan


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return np.nan
    return numerator / denominator


def _parse_channel(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


if __name__ == "__main__":
    main()
