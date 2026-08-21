"""Compare long-window HRV from expert beats and unchanged UNSW timestamps."""

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
import wfdb

from ecg_cascade.hrv_fidelity import (
    lin_concordance_correlation,
    symmetric_absolute_percentage_error,
    threshold_agreement_counts,
)
from ecg_cascade.peaks import compare_peak_sequences
from ecg_cascade.rr_hrv import extract_rr_hrv_features
from ecg_cascade.validation import match_peaks_to_reference
from ecg_cascade.wfdb_io import load_wfdb_beat_annotations


FEATURES = {
    "sd1_raw_ms": "SD1 raw",
    "sd2_raw_ms": "SD2 raw",
    "csi100": "CSI100",
    "sd1_filtered_ms": "SD1 filtered",
    "sd2_filtered_ms": "SD2 filtered",
    "modcsi100_filtered_ms": "ModCSI100 filtered",
    "slope100_bpm_per_s": "HR slope100",
    "j1_csi_x_slope": "J1 = CSI x slope",
    "j2_modcsi_filtered_x_slope": "J2 = ModCSI x slope",
}
THRESHOLD_FEATURES = ("j1_csi_x_slope", "j2_modcsi_filtered_x_slope")
SUBSETS = (
    "all",
    "support_ge_0.95",
    "support_eq_1.00",
    "reference_all_N",
    "reference_has_non_N",
)
DIFFICULT_RECORDS = ("108", "113", "207", "222", "231")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-dir", required=True, type=Path)
    parser.add_argument("--context-audit", required=True, type=Path)
    parser.add_argument(
        "--paired-input",
        type=Path,
        help="Reuse a previously generated paired_expert_unsw_hrv_windows CSV",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--support-tolerance-ms", type=float, default=50.0)
    parser.add_argument("--expert-match-tolerance-ms", type=float, default=75.0)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--median-width", type=int, default=7)
    parser.add_argument(
        "--threshold-quantiles",
        type=float,
        nargs="*",
        default=(0.90, 0.95, 0.99),
    )
    parser.add_argument("--records", nargs="*")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    database = args.database_dir.expanduser().resolve()
    audit_path = args.context_audit.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.window_size < 3:
        raise ValueError("window-size must be at least three")
    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)
    quantiles = tuple(float(value) for value in args.threshold_quantiles)
    if any(not 0 < value < 1 for value in quantiles):
        raise ValueError("threshold quantiles must lie strictly between 0 and 1")

    if args.paired_input is not None:
        paired_input = args.paired_input.expanduser().resolve()
        paired = pd.read_csv(paired_input, low_memory=False)
        paired["record"] = paired["record"].astype(str)
        available_records = paired["record"].drop_duplicates().tolist()
        records = [str(value) for value in (args.records or available_records)]
        paired = paired[paired["record"].isin(records)].copy()
    else:
        context = pd.read_csv(audit_path, low_memory=False)
        context = context[
            np.isclose(
                context["support_tolerance_ms"].to_numpy(float),
                args.support_tolerance_ms,
            )
        ].copy()
        available_records = context["record"].astype(str).drop_duplicates().tolist()
        records = [str(value) for value in (args.records or available_records)]
        paired_records: list[pd.DataFrame] = []
        for record in records:
            record_intervals = context[context["record"].astype(str) == record].copy()
            if record_intervals.empty:
                raise ValueError(f"No context rows for record {record}")
            paired_records.append(
                _pair_record_windows(
                    database / record,
                    record,
                    record_intervals,
                    window_size=args.window_size,
                    median_width=args.median_width,
                    expert_match_tolerance_ms=args.expert_match_tolerance_ms,
                )
            )
            print(f"completed {record}", flush=True)
        paired = pd.concat(paired_records, ignore_index=True)
    record_summary = _summarize_by_record(paired)
    pooled_summary = _summarize_pooled(paired, record_summary)
    record_thresholds, pooled_thresholds = _threshold_audit(
        paired,
        quantiles=quantiles,
    )
    paired.to_csv(output / "paired_expert_unsw_hrv_windows.csv.gz", index=False)
    record_summary.to_csv(output / "record_feature_agreement.csv", index=False)
    pooled_summary.to_csv(output / "pooled_feature_agreement.csv", index=False)
    record_thresholds.to_csv(output / "record_threshold_agreement.csv", index=False)
    pooled_thresholds.to_csv(output / "pooled_threshold_agreement.csv", index=False)
    (output / "decision_summary.json").write_text(
        json.dumps(
            _decision_summary(pooled_summary, pooled_thresholds),
            indent=2,
        ),
        encoding="utf-8",
    )
    paired[paired["record"].isin(DIFFICULT_RECORDS)].to_csv(
        output / "five_difficult_record_windows.csv.gz",
        index=False,
        compression="gzip",
    )
    _save_plots(paired, pooled_summary, pooled_thresholds, output)
    metadata = {
        "database_dir": str(database),
        "input_context_audit": str(audit_path),
        "reused_paired_input": (
            str(args.paired_input.expanduser().resolve())
            if args.paired_input is not None
            else None
        ),
        "records": records,
        "record_count": len(records),
        "algorithm_timestamp_owner": "unsw_initial_unchanged",
        "expert_source": "MIT-BIH atr expert beat annotations",
        "support_tolerance_ms": args.support_tolerance_ms,
        "expert_match_tolerance_ms": args.expert_match_tolerance_ms,
        "window_size_rr": args.window_size,
        "median_width_rr": args.median_width,
        "threshold_quantiles": quantiles,
        "threshold_status": (
            "descriptive expert-derived per-record sensitivity analysis; "
            "not a seizure or clinical abnormality threshold"
        ),
        "window_independence": (
            "false; adjacent 100-RR windows overlap by 99 intervals"
        ),
        "expert_all_beats_status": (
            "reference QRS sequence includes annotated ectopic and conduction beats; "
            "this isolates detector-to-reference feature fidelity and is not pure NN-HRV"
        ),
        "primary_outcome": "Lin concordance correlation coefficient",
        "ccc_above_0_8_status": (
            "operational very-strong agreement criterion from Weinstein et al. 2026, "
            "not proof of seizure-equivalent decisions"
        ),
    }
    (output / "benchmark_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("\nPOOLED FEATURE AGREEMENT")
    print(
        pooled_summary[
            pooled_summary["feature"].isin(THRESHOLD_FEATURES)
        ].to_string(index=False)
    )
    print("\nPOOLED THRESHOLD AGREEMENT")
    print(pooled_thresholds.to_string(index=False))


def _pair_record_windows(
    record_path: Path,
    record_name: str,
    intervals: pd.DataFrame,
    *,
    window_size: int,
    median_width: int,
    expert_match_tolerance_ms: float,
) -> pd.DataFrame:
    intervals = intervals.sort_values("rr_index").reset_index(drop=True)
    expected = np.arange(intervals.shape[0], dtype=int)
    if not np.array_equal(intervals["rr_index"].to_numpy(int), expected):
        raise ValueError(f"Record {record_name} has noncontiguous RR indices")
    starts = intervals["start_unsw_sample"].to_numpy(np.int64)
    ends = intervals["end_unsw_sample"].to_numpy(np.int64)
    if starts.size == 0 or not np.array_equal(starts[1:], ends[:-1]):
        raise ValueError(f"Record {record_name} does not form one UNSW sequence")
    unsw = np.concatenate(([starts[0]], ends))
    supported = _boolean_array(intervals["rr_supported"])
    trailing_qsqi = intervals["trailing_qsqi_fraction_at_end"].to_numpy(float)

    header = wfdb.rdheader(str(record_path))
    fs = float(header.fs)
    reference, symbols = load_wfdb_beat_annotations(record_path)
    unsw_features = extract_rr_hrv_features(
        unsw,
        sampling_rate_hz=fs,
        segment_start_s=0.0,
        agreement=compare_peak_sequences(unsw, unsw, sampling_rate_hz=fs),
        window_size=window_size,
        median_width=median_width,
    )
    expert_features = extract_rr_hrv_features(
        reference,
        sampling_rate_hz=fs,
        segment_start_s=0.0,
        agreement=compare_peak_sequences(reference, reference, sampling_rate_hz=fs),
        window_size=window_size,
        median_width=median_width,
    )
    matches = match_peaks_to_reference(
        unsw,
        reference,
        sampling_rate_hz=fs,
        tolerance_ms=expert_match_tolerance_ms,
    )

    rows: list[dict[str, object]] = []
    for detected_index, reference_index in zip(
        matches.detected_indices,
        matches.reference_indices,
        strict=True,
    ):
        detected_index = int(detected_index)
        reference_index = int(reference_index)
        if detected_index < window_size or reference_index < window_size:
            continue
        algorithm_row = unsw_features.iloc[detected_index - 1]
        expert_row = expert_features.iloc[reference_index - 1]
        if not bool(algorithm_row["feature_defined"]):
            continue
        if not bool(expert_row["feature_defined"]):
            continue
        support_window = supported[detected_index - window_size : detected_index]
        reference_symbol_window = symbols[
            reference_index - window_size : reference_index + 1
        ]
        algorithm_duration = (
            unsw[detected_index] - unsw[detected_index - window_size]
        ) / fs
        expert_duration = (
            reference[reference_index] - reference[reference_index - window_size]
        ) / fs
        row: dict[str, object] = {
            "record": record_name,
            "sampling_rate_hz": fs,
            "unsw_peak_index": detected_index,
            "expert_peak_index": reference_index,
            "anchor_time_s": reference[reference_index] / fs,
            "anchor_timing_error_ms": (
                unsw[detected_index] - reference[reference_index]
            )
            * 1000.0
            / fs,
            "unsw_window_duration_s": algorithm_duration,
            "expert_window_duration_s": expert_duration,
            "window_duration_error_s": algorithm_duration - expert_duration,
            "rr_support_coverage": float(np.mean(support_window)),
            "all_100_rr_supported": bool(np.all(support_window)),
            "trailing_qsqi_at_anchor": float(trailing_qsqi[detected_index - 1]),
            "reference_non_N_beat_count": int(
                np.sum(reference_symbol_window != "N")
            ),
            "reference_all_N": bool(np.all(reference_symbol_window == "N")),
        }
        for feature in FEATURES:
            row[f"expert__{feature}"] = float(expert_row[feature])
            row[f"algorithm__{feature}"] = float(algorithm_row[feature])
        rows.append(row)
    return pd.DataFrame(rows)


def _summarize_by_record(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subset in SUBSETS:
        selected = paired[_subset_mask(paired, subset)]
        for record, group in selected.groupby("record", sort=True):
            for feature in FEATURES:
                rows.append(
                    _agreement_row(
                        group,
                        feature=feature,
                        subset=subset,
                        record=str(record),
                    )
                )
    return pd.DataFrame(rows)


def _summarize_pooled(
    paired: pd.DataFrame,
    record_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subset in SUBSETS:
        selected = paired[_subset_mask(paired, subset)]
        for feature in FEATURES:
            row = _agreement_row(selected, feature=feature, subset=subset)
            record_group = record_summary[
                (record_summary["subset"] == subset)
                & (record_summary["feature"] == feature)
            ]
            finite_record_ccc = record_group["ccc"].to_numpy(float)
            finite_record_ccc = finite_record_ccc[np.isfinite(finite_record_ccc)]
            row["records"] = int(record_group["record"].nunique())
            row["median_record_ccc"] = (
                float(np.median(finite_record_ccc))
                if finite_record_ccc.size
                else np.nan
            )
            row["record_fraction_ccc_ge_0_8"] = (
                float(np.mean(finite_record_ccc >= 0.8))
                if finite_record_ccc.size
                else np.nan
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _agreement_row(
    frame: pd.DataFrame,
    *,
    feature: str,
    subset: str,
    record: str | None = None,
) -> dict[str, object]:
    truth = frame[f"expert__{feature}"].to_numpy(float)
    measured = frame[f"algorithm__{feature}"].to_numpy(float)
    finite = np.isfinite(truth) & np.isfinite(measured)
    truth = truth[finite]
    measured = measured[finite]
    error = measured - truth
    return {
        **({"record": record} if record is not None else {}),
        "subset": subset,
        "feature": feature,
        "feature_label": FEATURES[feature],
        "paired_windows": int(truth.size),
        "ccc": lin_concordance_correlation(truth, measured),
        "pearson_r": _correlation(truth, measured, method="pearson"),
        "spearman_rho": _correlation(truth, measured, method="spearman"),
        "expert_median": _percentile(truth, 50),
        "algorithm_median": _percentile(measured, 50),
        "mean_signed_error": float(np.mean(error)) if error.size else np.nan,
        "mae": float(np.mean(np.abs(error))) if error.size else np.nan,
        "median_absolute_error": _percentile(np.abs(error), 50),
        "p95_absolute_error": _percentile(np.abs(error), 95),
        "smape_fraction": symmetric_absolute_percentage_error(truth, measured),
    }


def _threshold_audit(
    paired: pd.DataFrame,
    *,
    quantiles: tuple[float, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    record_rows: list[dict[str, object]] = []
    for record, record_frame in paired.groupby("record", sort=True):
        for feature in THRESHOLD_FEATURES:
            raw_truth_all = record_frame[f"expert__{feature}"].to_numpy(float)
            raw_measured_all = record_frame[f"algorithm__{feature}"].to_numpy(float)
            finite_all = np.isfinite(raw_truth_all) & np.isfinite(raw_measured_all)
            truth_all = raw_truth_all[finite_all]
            if truth_all.size < 200:
                continue
            for quantile in quantiles:
                threshold = float(np.quantile(truth_all, quantile))
                all_n = int(truth_all.size)
                all_expert_positive = int(np.sum(truth_all > threshold))
                for subset in ("all", "support_ge_0.95", "support_eq_1.00"):
                    selected = record_frame[_subset_mask(record_frame, subset)]
                    counts = threshold_agreement_counts(
                        selected[f"expert__{feature}"].to_numpy(float),
                        selected[f"algorithm__{feature}"].to_numpy(float),
                        threshold=threshold,
                    )
                    record_rows.append(
                        {
                            "record": str(record),
                            "feature": feature,
                            "feature_label": FEATURES[feature],
                            "subset": subset,
                            "expert_threshold_quantile": quantile,
                            "expert_derived_threshold": threshold,
                            "all_record_evaluable_n": all_n,
                            "all_record_expert_positive": all_expert_positive,
                            "selected_window_coverage": _divide(
                                int(counts["n"]), all_n
                            ),
                            "selected_expert_positive_coverage": _divide(
                                int(counts["tp"]) + int(counts["fn"]),
                                all_expert_positive,
                            ),
                            "hard_veto_sensitivity": _divide(
                                int(counts["tp"]), all_expert_positive
                            ),
                            "hard_veto_f1": _divide(
                                2 * int(counts["tp"]),
                                2 * int(counts["tp"])
                                + int(counts["fp"])
                                + all_expert_positive
                                - int(counts["tp"]),
                            ),
                            **counts,
                        }
                    )
    record_thresholds = pd.DataFrame(record_rows)
    pooled_rows: list[dict[str, object]] = []
    group_columns = ["feature", "feature_label", "subset", "expert_threshold_quantile"]
    for keys, group in record_thresholds.groupby(group_columns, sort=True):
        feature, label, subset, quantile = keys
        tp = int(group["tp"].sum())
        fp = int(group["fp"].sum())
        fn = int(group["fn"].sum())
        tn = int(group["tn"].sum())
        all_n = int(group["all_record_evaluable_n"].sum())
        all_expert_positive = int(group["all_record_expert_positive"].sum())
        selected_expert_positive = tp + fn
        pooled_rows.append(
            {
                "feature": feature,
                "feature_label": label,
                "subset": subset,
                "expert_threshold_quantile": float(quantile),
                "records": int(group["record"].nunique()),
                "contributing_records": int(group.loc[group["n"] > 0, "record"].nunique()),
                "n": tp + fp + fn + tn,
                "all_evaluable_n": all_n,
                "window_coverage": _divide(tp + fp + fn + tn, all_n),
                "all_expert_positive": all_expert_positive,
                "selected_expert_positive": selected_expert_positive,
                "expert_positive_coverage": _divide(
                    selected_expert_positive, all_expert_positive
                ),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "sensitivity": _divide(tp, tp + fn),
                "ppv": _divide(tp, tp + fp),
                "specificity": _divide(tn, tn + fp),
                "f1": _divide(2 * tp, 2 * tp + fp + fn),
                "hard_veto_sensitivity": _divide(tp, all_expert_positive),
                "hard_veto_f1": _divide(
                    2 * tp,
                    2 * tp + fp + all_expert_positive - tp,
                ),
            }
        )
    return record_thresholds, pd.DataFrame(pooled_rows)


def _save_plots(
    paired: pd.DataFrame,
    pooled: pd.DataFrame,
    thresholds: pd.DataFrame,
    output: Path,
) -> None:
    _plot_concordance(pooled, output)
    _plot_j_features(paired, output)
    _plot_thresholds(thresholds, output)
    _plot_difficult_timelines(paired, output)


def _decision_summary(
    features: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> dict[str, object]:
    all_features = features[features["subset"] == "all"].set_index("feature")
    all_thresholds = thresholds[thresholds["subset"] == "all"].set_index(
        ["feature", "expert_threshold_quantile"]
    )
    fully_supported = thresholds[
        (thresholds["subset"] == "support_eq_1.00")
        & np.isclose(thresholds["expert_threshold_quantile"], 0.95)
    ].set_index("feature")
    return {
        "continue_to_patient_specific_validation": True,
        "validated_seizure_threshold": False,
        "j1_pooled_ccc": float(all_features.loc["j1_csi_x_slope", "ccc"]),
        "j2_pooled_ccc": float(
            all_features.loc["j2_modcsi_filtered_x_slope", "ccc"]
        ),
        "j1_95th_percentile_threshold_f1": float(
            all_thresholds.loc[("j1_csi_x_slope", 0.95), "f1"]
        ),
        "j2_95th_percentile_threshold_f1": float(
            all_thresholds.loc[("j2_modcsi_filtered_x_slope", 0.95), "f1"]
        ),
        "j1_99th_percentile_threshold_f1": float(
            all_thresholds.loc[("j1_csi_x_slope", 0.99), "f1"]
        ),
        "j2_99th_percentile_threshold_f1": float(
            all_thresholds.loc[("j2_modcsi_filtered_x_slope", 0.99), "f1"]
        ),
        "all_100_rr_supported_window_coverage": float(
            fully_supported.loc["j1_csi_x_slope", "window_coverage"]
        ),
        "j1_95th_expert_high_window_coverage_if_all_supported_required": float(
            fully_supported.loc["j1_csi_x_slope", "expert_positive_coverage"]
        ),
        "j2_95th_expert_high_window_coverage_if_all_supported_required": float(
            fully_supported.loc[
                "j2_modcsi_filtered_x_slope", "expert_positive_coverage"
            ]
        ),
        "hard_detector_agreement_veto_accepted": False,
        "warning": (
            "MIT-BIH has no seizure labels; quantile thresholds are a "
            "measurement-preservation stress test, not seizure validation."
        ),
    }


def _plot_concordance(pooled: pd.DataFrame, output: Path) -> None:
    shown = pooled[pooled["subset"].isin(["all", "support_ge_0.95", "support_eq_1.00"])]
    fig, axes = plt.subplots(2, 1, figsize=(13, 10), constrained_layout=True)
    labels = list(FEATURES.values())
    x = np.arange(len(labels))
    width = 0.25
    colors = ["#3569A8", "#D88400", "#2E8B57"]
    for index, subset in enumerate(("all", "support_ge_0.95", "support_eq_1.00")):
        group = shown[shown["subset"] == subset].set_index("feature")
        values = [group.loc[key, "ccc"] for key in FEATURES]
        record_values = [group.loc[key, "median_record_ccc"] for key in FEATURES]
        axes[0].bar(x + (index - 1) * width, values, width, label=subset, color=colors[index])
        axes[1].bar(
            x + (index - 1) * width,
            record_values,
            width,
            label=subset,
            color=colors[index],
        )
    for axis, title in zip(
        axes,
        ("Pooled-window Lin CCC", "Median record-level Lin CCC"),
        strict=True,
    ):
        axis.axhline(0.8, color="#B22222", linestyle="--", linewidth=1.3, label="CCC = 0.8")
        axis.set_ylim(-0.05, 1.05)
        axis.set_ylabel("Concordance with expert HRV")
        axis.set_title(title)
        axis.set_xticks(x, labels, rotation=24, ha="right")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(ncol=4, fontsize=9)
    fig.suptitle("Expert annotations versus unchanged-UNSW 100-RR features")
    fig.savefig(output / "hrv_feature_concordance.png", dpi=190)
    plt.close(fig)


def _plot_j_features(paired: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for axis, feature in zip(axes, THRESHOLD_FEATURES, strict=True):
        truth = paired[f"expert__{feature}"].to_numpy(float)
        measured = paired[f"algorithm__{feature}"].to_numpy(float)
        finite = np.isfinite(truth) & np.isfinite(measured)
        truth = np.log1p(np.maximum(truth[finite], 0.0))
        measured = np.log1p(np.maximum(measured[finite], 0.0))
        limit = float(max(np.percentile(truth, 99.8), np.percentile(measured, 99.8)))
        axis.hexbin(truth, measured, gridsize=65, mincnt=1, bins="log", cmap="viridis")
        axis.plot([0, limit], [0, limit], color="white", linewidth=1.2, linestyle="--")
        axis.set_xlim(0, limit)
        axis.set_ylim(0, limit)
        axis.set_xlabel(f"Expert log(1 + {FEATURES[feature]})")
        axis.set_ylabel(f"UNSW log(1 + {FEATURES[feature]})")
        axis.set_title(FEATURES[feature])
    fig.suptitle("Same matched heartbeat, separate trailing 100-RR histories")
    fig.savefig(output / "j1_j2_expert_vs_unsw.png", dpi=190)
    plt.close(fig)


def _plot_thresholds(thresholds: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    colors = {"all": "#3569A8", "support_ge_0.95": "#D88400", "support_eq_1.00": "#2E8B57"}
    for axis, feature in zip(axes, THRESHOLD_FEATURES, strict=True):
        group = thresholds[thresholds["feature"] == feature]
        for subset, subset_group in group.groupby("subset", sort=False):
            subset_group = subset_group.sort_values("expert_threshold_quantile")
            axis.plot(
                100 * subset_group["expert_threshold_quantile"],
                subset_group["f1"],
                marker="o",
                color=colors[str(subset)],
                label=str(subset),
            )
            if str(subset) != "all":
                axis.plot(
                    100 * subset_group["expert_threshold_quantile"],
                    subset_group["hard_veto_f1"],
                    marker="x",
                    linestyle="--",
                    color=colors[str(subset)],
                    alpha=0.8,
                    label=f"{subset}: if vetoed windows count as missed",
                )
        axis.set_ylim(0, 1.02)
        axis.set_xlabel("Per-record expert-derived percentile threshold")
        axis.set_ylabel("Same-threshold window F1")
        axis.set_title(FEATURES[feature])
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Threshold-crossing preservation; descriptive, not seizure detection")
    fig.savefig(output / "j1_j2_threshold_agreement.png", dpi=190)
    plt.close(fig)


def _plot_difficult_timelines(paired: pd.DataFrame, output: Path) -> None:
    records = [record for record in ("108", "207") if record in set(paired["record"])]
    if not records:
        return
    fig, axes = plt.subplots(
        len(records),
        2,
        figsize=(15, 4.5 * len(records)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, record in enumerate(records):
        group = paired[paired["record"] == record]
        for column_index, feature in enumerate(THRESHOLD_FEATURES):
            axis = axes[row_index, column_index]
            time_min = group["anchor_time_s"] / 60.0
            axis.plot(
                time_min,
                np.log1p(np.maximum(group[f"expert__{feature}"], 0.0)),
                color="#202020",
                linewidth=1.1,
                label="Expert beats",
            )
            axis.plot(
                time_min,
                np.log1p(np.maximum(group[f"algorithm__{feature}"], 0.0)),
                color="#C23B22",
                linewidth=0.9,
                alpha=0.8,
                label="UNSW timestamps",
            )
            axis.set_title(f"Record {record}: {FEATURES[feature]}")
            axis.set_xlabel("Recording time (minutes)")
            axis.set_ylabel("log(1 + feature)")
            axis.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.suptitle("Difficult-record long-run HRV trajectories")
    fig.savefig(output / "difficult_record_hrv_timelines.png", dpi=190)
    plt.close(fig)


def _subset_mask(frame: pd.DataFrame, subset: str) -> np.ndarray:
    if subset == "all":
        return np.ones(frame.shape[0], dtype=bool)
    if subset == "support_ge_0.95":
        return frame["rr_support_coverage"].to_numpy(float) >= 0.95
    if subset == "support_eq_1.00":
        return _boolean_array(frame["all_100_rr_supported"])
    if subset == "reference_all_N":
        return _boolean_array(frame["reference_all_N"])
    if subset == "reference_has_non_N":
        return ~_boolean_array(frame["reference_all_N"])
    raise ValueError(f"Unknown subset {subset}")


def _boolean_array(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(bool)
    values = series.astype(str).str.strip().str.casefold()
    valid = values.isin(["true", "false"])
    if not bool(valid.all()):
        raise ValueError("rr_supported contains non-Boolean values")
    return (values == "true").to_numpy(bool)


def _correlation(x: np.ndarray, y: np.ndarray, *, method: str) -> float:
    if x.size < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    if method == "pearson":
        return float(stats.pearsonr(x, y).statistic)
    return float(stats.spearmanr(x, y).statistic)


def _percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values.size else float("nan")


def _divide(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


if __name__ == "__main__":
    main()
