"""Explain the RR+HRV method choice from the saved all-48 audit outputs.

This is a post-processing script.  It does not rerun any detector.  It combines
the saved beat/RR/HRV scorecards with threshold-independent ROC and
precision-recall areas calculated from the paired expert/algorithmic HRV
windows.

The current architecture is named carefully throughout this script:

* UNSW owns the event grid, which preserves event coverage and beat F1.
* For each complete RR interval, timestamp priority is NeuroKit, then Zhai,
  then the original UNSW endpoints.

The AUC task is measurement-fidelity analysis.  Positive labels mean that the
expert-derived HRV feature is above its within-record quantile; they are not
seizure labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FEATURES = {
    "j1_csi_x_slope": "J1",
    "j2_modcsi_filtered_x_slope": "J2",
}
SELECTED_METHOD = "interval_neurokit_zhai_unsw"
SELECTED_CONFIGURATION = "nk300"
SELECTED_TOLERANCE_MS = 50.0

# RR and HRV tables use different names for the same availability-conditioned
# pure lanes.  Canonical names let one scorecard show both parts honestly.
METHOD_ALIASES = {
    "pure_neurokit_rr_lane": "pure_neurokit_lane",
    "pure_neurokit_window_lane": "pure_neurokit_lane",
    "pure_zhai_rr_lane": "pure_zhai_lane",
    "pure_zhai_window_lane": "pure_zhai_lane",
}

METHOD_LABELS = {
    "unsw_baseline": "Unchanged UNSW baseline",
    "neurokit_300_strict": "Prior standalone NeuroKit (300 ms strict)",
    "zhai2023_reimplementation": "Prior standalone Zhai reimplementation",
    "fusion_nk300_zhai_tol050": "Prior per-beat NeuroKit/Zhai fusion",
    "original_per_beat_fusion": "Old per-beat fusion on UNSW grid",
    "interval_neurokit_unsw": "UNSW grid: NeuroKit then UNSW per RR",
    "interval_zhai_unsw": "UNSW grid: Zhai then UNSW per RR",
    "interval_neurokit_zhai_unsw": (
        "UNSW grid: NeuroKit then Zhai then UNSW per RR"
    ),
    "pure_neurokit_lane": "Pure NeuroKit precision/context lane",
    "pure_zhai_lane": "Pure Zhai precision/context lane",
    "window_priority_neurokit_zhai_unsw": (
        "Whole-window NeuroKit/Zhai/UNSW priority"
    ),
}

CURRENT_METHOD_ORDER = (
    "unsw_baseline",
    "original_per_beat_fusion",
    "interval_neurokit_unsw",
    "interval_zhai_unsw",
    "interval_neurokit_zhai_unsw",
    "pure_neurokit_lane",
    "pure_zhai_lane",
    "window_priority_neurokit_zhai_unsw",
)


def build_parser() -> argparse.ArgumentParser:
    project = Path(__file__).resolve().parents[1]
    default_audit = project / "outputs" / "interval_consistent_fusion_all48_v1"
    default_baseline = (
        project
        / "outputs"
        / "expert_unsw_hrv_fidelity_all48_v1"
        / "paired_expert_unsw_hrv_windows.csv.gz"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Build a reproducible RR+HRV method-selection scorecard and explain "
            "the UNSW-owned NeuroKit-to-Zhai-to-UNSW hierarchy."
        )
    )
    parser.add_argument("--audit-dir", type=Path, default=default_audit)
    parser.add_argument("--baseline-paired", type=Path, default=default_baseline)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project / "outputs" / "method_selection_analysis_v1",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.95,
        help="Within-record expert quantile used to define high-HRV labels.",
    )
    parser.add_argument(
        "--selected-method",
        default=SELECTED_METHOD,
        help="Method highlighted as the current working candidate.",
    )
    return parser


def _canonical_method(method: str) -> str:
    return METHOD_ALIASES.get(str(method), str(method))


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], source: Path) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def _binary_curve_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Return AUROC, trapezoidal PR-AUC, and average precision without sklearn."""
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    labels = labels[finite]
    scores = scores[finite]
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if not positives or not negatives:
        return {
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "average_precision": float("nan"),
        }

    # Mann-Whitney form of AUROC with average ranks gives correct tie handling.
    ranks = pd.Series(scores).rank(method="average").to_numpy(float)
    positive_rank_sum = float(ranks[labels == 1].sum())
    roc_auc = (
        positive_rank_sum - positives * (positives + 1) / 2
    ) / (positives * negatives)

    # Keep the last observation at each distinct descending score so every PR
    # point represents a complete threshold, including all tied observations.
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)
    distinct_ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), labels.size - 1]
    tp = tp[distinct_ends].astype(float)
    fp = fp[distinct_ends].astype(float)
    recall = tp / positives
    precision = tp / (tp + fp)
    recall = np.r_[0.0, recall]
    precision = np.r_[1.0, precision]
    pr_auc = float(np.trapezoid(precision, recall))
    average_precision = float(np.sum(np.diff(recall) * precision[1:]))
    return {
        "roc_auc": float(roc_auc),
        "pr_auc": pr_auc,
        "average_precision": average_precision,
    }


def _read_paired_windows(audit_dir: Path, baseline_path: Path) -> pd.DataFrame:
    selected_path = audit_dir / "selected_configuration_paired_hrv.csv.gz"
    selected = pd.read_csv(selected_path, low_memory=False)
    _require_columns(selected, ("record", "method"), selected_path)
    selected["record"] = selected["record"].astype(str)

    baseline = pd.read_csv(baseline_path, low_memory=False)
    _require_columns(baseline, ("record",), baseline_path)
    baseline["record"] = baseline["record"].astype(str)
    baseline["method"] = "unsw_baseline"

    columns = ["record", "method"]
    for feature in FEATURES:
        columns.extend((f"expert__{feature}", f"algorithm__{feature}"))
    _require_columns(selected, columns, selected_path)
    _require_columns(baseline, columns, baseline_path)
    return pd.concat((baseline[columns], selected[columns]), ignore_index=True)


def calculate_auc_metrics(paired: pd.DataFrame, quantile: float) -> pd.DataFrame:
    """Calculate record-normalized HRV discrimination and fixed-threshold F1.

    Each record receives its own expert-derived threshold.  The continuous AUC
    score is algorithm_value / expert_threshold, so score > 1 exactly matches
    the saved same-threshold classification rule.
    """
    rows: list[dict[str, object]] = []
    for method, method_frame in paired.groupby("method", sort=False):
        for feature, short_name in FEATURES.items():
            expert_column = f"expert__{feature}"
            algorithm_column = f"algorithm__{feature}"
            pooled_labels: list[np.ndarray] = []
            pooled_scores: list[np.ndarray] = []
            all_evaluable_n = 0
            all_expert_positive = 0
            available_expert_positive = 0
            macro_roc: list[float] = []
            macro_pr: list[float] = []

            for _, record_frame in method_frame.groupby("record", sort=False):
                truth_all = record_frame[expert_column].to_numpy(float)
                measured_all = record_frame[algorithm_column].to_numpy(float)
                finite_truth = np.isfinite(truth_all)
                truth_for_threshold = truth_all[finite_truth]
                if truth_for_threshold.size < 2:
                    continue
                threshold = float(np.quantile(truth_for_threshold, quantile))
                if not np.isfinite(threshold) or threshold <= 0:
                    raise ValueError(
                        f"Non-positive {short_name} threshold for method {method}: "
                        f"{threshold}"
                    )
                all_evaluable_n += int(truth_for_threshold.size)
                all_expert_positive += int(np.sum(truth_for_threshold > threshold))

                finite_pair = finite_truth & np.isfinite(measured_all)
                truth = truth_all[finite_pair]
                measured = measured_all[finite_pair]
                labels = (truth > threshold).astype(np.int8)
                scores = measured / threshold
                available_expert_positive += int(labels.sum())
                pooled_labels.append(labels)
                pooled_scores.append(scores)

                record_curves = _binary_curve_metrics(labels, scores)
                if np.isfinite(record_curves["roc_auc"]):
                    macro_roc.append(record_curves["roc_auc"])
                    macro_pr.append(record_curves["pr_auc"])

            if not pooled_labels:
                continue
            labels = np.concatenate(pooled_labels)
            scores = np.concatenate(pooled_scores)
            predicted = scores > 1.0
            tp = int(np.sum(predicted & (labels == 1)))
            fp = int(np.sum(predicted & (labels == 0)))
            fn = int(np.sum((~predicted) & (labels == 1)))
            tn = int(np.sum((~predicted) & (labels == 0)))
            curves = _binary_curve_metrics(labels, scores)
            rows.append(
                {
                    "method": method,
                    "canonical_method": _canonical_method(method),
                    "feature": feature,
                    "feature_short": short_name,
                    "expert_threshold_quantile": quantile,
                    "available_windows": int(labels.size),
                    "all_evaluable_windows": all_evaluable_n,
                    "window_coverage": _safe_divide(labels.size, all_evaluable_n),
                    "all_expert_positive": all_expert_positive,
                    "available_expert_positive": available_expert_positive,
                    "expert_positive_coverage": _safe_divide(
                        available_expert_positive, all_expert_positive
                    ),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "sensitivity": _safe_divide(tp, tp + fn),
                    "ppv": _safe_divide(tp, tp + fp),
                    "same_threshold_f1": _safe_divide(2 * tp, 2 * tp + fp + fn),
                    "hard_veto_f1": _safe_divide(
                        2 * tp, 2 * tp + fp + all_expert_positive - tp
                    ),
                    **curves,
                    "macro_record_roc_auc": _finite_mean(macro_roc),
                    "macro_record_pr_auc": _finite_mean(macro_pr),
                }
            )
    return pd.DataFrame(rows)


def _primary_setting(frame: pd.DataFrame) -> pd.DataFrame:
    baseline = frame["method"].eq("unsw_baseline")
    selected = frame["neurokit_configuration"].eq(SELECTED_CONFIGURATION) & np.isclose(
        frame["association_tolerance_ms"].fillna(-1).to_numpy(float),
        SELECTED_TOLERANCE_MS,
    )
    return frame[baseline | selected].copy()


def validate_reconstructed_f1(
    auc_metrics: pd.DataFrame,
    saved_thresholds: pd.DataFrame,
    quantile: float,
) -> None:
    saved = saved_thresholds[
        np.isclose(saved_thresholds["expert_threshold_quantile"], quantile)
    ][["method", "feature", "f1", "hard_veto_f1"]]
    check = auc_metrics.merge(saved, on=["method", "feature"], how="inner")
    if check.empty:
        raise ValueError("Could not match reconstructed AUC rows to saved F1 rows")
    f1_error = np.abs(check["same_threshold_f1"] - check["f1"])
    veto_error = np.abs(check["hard_veto_f1_x"] - check["hard_veto_f1_y"])
    if float(f1_error.max()) > 1e-10 or float(veto_error.max()) > 1e-10:
        raise ValueError(
            "Reconstructed threshold classification does not match saved F1; "
            f"maximum errors were F1={f1_error.max():.3g}, "
            f"hard-veto F1={veto_error.max():.3g}"
        )


def build_scorecard(
    audit_dir: Path,
    auc_metrics: pd.DataFrame,
    quantile: float,
    selected_method: str,
) -> pd.DataFrame:
    comparison = pd.read_csv(audit_dir / "complete_method_comparison.csv")
    rr = _primary_setting(pd.read_csv(audit_dir / "pooled_rr_metrics.csv"))
    hrv = _primary_setting(pd.read_csv(audit_dir / "pooled_hrv_metrics.csv"))
    thresholds = _primary_setting(pd.read_csv(audit_dir / "pooled_threshold_metrics.csv"))
    thresholds = thresholds[
        np.isclose(thresholds["expert_threshold_quantile"], quantile)
    ]

    comparison["canonical_method"] = comparison["method"].map(_canonical_method)
    rr["canonical_method"] = rr["method"].map(_canonical_method)
    hrv["canonical_method"] = hrv["method"].map(_canonical_method)
    thresholds["canonical_method"] = thresholds["method"].map(_canonical_method)

    methods = list(dict.fromkeys(comparison["canonical_method"].tolist()))
    for method in auc_metrics["canonical_method"].tolist():
        if method not in methods:
            methods.append(method)

    rows: list[dict[str, object]] = []
    for method in methods:
        comp = comparison[comparison["canonical_method"].eq(method)]
        rr_row = rr[rr["canonical_method"].eq(method)]
        hrv_rows = hrv[hrv["canonical_method"].eq(method)]
        threshold_rows = thresholds[thresholds["canonical_method"].eq(method)]
        auc_rows = auc_metrics[auc_metrics["canonical_method"].eq(method)]

        def first(frame: pd.DataFrame, column: str) -> float:
            if frame.empty or column not in frame:
                return float("nan")
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            return float(values.iloc[0]) if not values.empty else float("nan")

        def by_feature(frame: pd.DataFrame, feature: str, column: str) -> float:
            selected = frame[frame["feature"].eq(feature)]
            return first(selected, column)

        j1_ccc = by_feature(hrv_rows, "j1_csi_x_slope", "pooled_ccc")
        j2_ccc = by_feature(hrv_rows, "j2_modcsi_filtered_x_slope", "pooled_ccc")
        j1_f1 = by_feature(threshold_rows, "j1_csi_x_slope", "f1")
        j2_f1 = by_feature(threshold_rows, "j2_modcsi_filtered_x_slope", "f1")
        j1_veto = by_feature(threshold_rows, "j1_csi_x_slope", "hard_veto_f1")
        j2_veto = by_feature(
            threshold_rows, "j2_modcsi_filtered_x_slope", "hard_veto_f1"
        )
        j1_roc = by_feature(auc_rows, "j1_csi_x_slope", "roc_auc")
        j2_roc = by_feature(auc_rows, "j2_modcsi_filtered_x_slope", "roc_auc")
        j1_pr = by_feature(auc_rows, "j1_csi_x_slope", "pr_auc")
        j2_pr = by_feature(auc_rows, "j2_modcsi_filtered_x_slope", "pr_auc")

        label = METHOD_LABELS.get(method)
        if label is None and not comp.empty:
            label = str(comp.iloc[0]["label"])
        rows.append(
            {
                "method": method,
                "label": label or method,
                "selected_working_candidate": method == selected_method,
                "event_f1": first(comp, "beat_or_event_f1"),
                "mean_timing_error_ms": first(comp, "mean_timing_error_ms"),
                "rr_availability": first(rr_row, "rr_availability"),
                "rr_mae_ms": first(rr_row, "rr_mae_ms"),
                "successive_rr_change_mae_ms": first(
                    rr_row, "successive_rr_change_mae_ms"
                ),
                "source_transition_change_mae_ms": first(
                    rr_row, "source_transition_successive_change_mae_ms"
                ),
                "mixed_endpoint_rr_count": first(
                    rr_row, "mixed_endpoint_rr_count"
                ),
                "hrv_window_coverage": first(hrv_rows, "window_coverage"),
                "j1_ccc": j1_ccc,
                "j2_ccc": j2_ccc,
                "mean_ccc": _finite_mean((j1_ccc, j2_ccc)),
                "j1_threshold_f1": j1_f1,
                "j2_threshold_f1": j2_f1,
                "mean_threshold_f1": _finite_mean((j1_f1, j2_f1)),
                "j1_hard_veto_f1": j1_veto,
                "j2_hard_veto_f1": j2_veto,
                "mean_hard_veto_f1": _finite_mean((j1_veto, j2_veto)),
                "j1_roc_auc": j1_roc,
                "j2_roc_auc": j2_roc,
                "mean_roc_auc": _finite_mean((j1_roc, j2_roc)),
                "j1_pr_auc": j1_pr,
                "j2_pr_auc": j2_pr,
                "mean_pr_auc": _finite_mean((j1_pr, j2_pr)),
            }
        )
    return pd.DataFrame(rows)


def _metric(value: object, digits: int = 4, percent: bool = False) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(numeric):
        return "—"
    if percent:
        return f"{100 * numeric:.1f}%"
    return f"{numeric:.{digits}f}"


def _markdown_table(rows: list[list[str]], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(
    scorecard: pd.DataFrame,
    output_path: Path,
    quantile: float,
    selected_method: str,
) -> None:
    indexed = scorecard.set_index("method")
    selected = indexed.loc[selected_method]
    baseline = indexed.loc["unsw_baseline"]
    old = indexed.loc["original_per_beat_fusion"]
    nk_unsw = indexed.loc["interval_neurokit_unsw"]
    pure_nk = indexed.loc["pure_neurokit_lane"]
    pure_zhai = indexed.loc["pure_zhai_lane"]

    rr_improvement = 1 - selected["rr_mae_ms"] / baseline["rr_mae_ms"]
    change_improvement = (
        1
        - selected["successive_rr_change_mae_ms"]
        / baseline["successive_rr_change_mae_ms"]
    )
    old_rr_improvement = 1 - selected["rr_mae_ms"] / old["rr_mae_ms"]
    versus_nk = nk_unsw["rr_mae_ms"] - selected["rr_mae_ms"]

    table_rows: list[list[str]] = []
    for method in CURRENT_METHOD_ORDER:
        if method not in indexed.index:
            continue
        row = indexed.loc[method]
        name = str(row["label"])
        if method == selected_method:
            name = f"**{name}**"
        table_rows.append(
            [
                name,
                _metric(row["rr_availability"], percent=True),
                _metric(row["rr_mae_ms"], 3),
                _metric(row["successive_rr_change_mae_ms"], 3),
                _metric(row["hrv_window_coverage"], percent=True),
                _metric(row["mean_ccc"], 4),
                _metric(row["mean_threshold_f1"], 4),
                _metric(row["mean_roc_auc"], 4),
                _metric(row["mean_pr_auc"], 4),
            ]
        )

    q_label = f"q{int(round(quantile * 100))}"
    table = _markdown_table(
        table_rows,
        [
            "Method",
            "RR availability",
            "RR MAE ms",
            "ΔRR MAE ms",
            "HRV coverage",
            "Mean CCC",
            f"Mean {q_label} F1",
            "Mean ROC-AUC",
            "Mean PR-AUC",
        ],
    )

    report = f"""# RR+HRV method-selection analysis

## Decision in one sentence

Retain the **UNSW-owned NeuroKit → Zhai → UNSW same-source-per-RR hierarchy**
as the current complete-coverage candidate, while keeping unchanged UNSW as a
parallel baseline and the pure NeuroKit/Zhai outputs as precision/context lanes.

“UNSW–NK–Zhai” describes ownership: UNSW supplies the event grid. The actual
timestamp priority for a complete RR interval is NeuroKit first, Zhai second,
and UNSW as the coverage-preserving fallback.

## What was compared

The saved 48-record MIT–BIH audit was analyzed at the predeclared NeuroKit
300-ms configuration and ±50-ms association tolerance. The scorecard combines:

- beat/event F1 and timing error;
- RR availability, RR MAE, successive-RR-change MAE, and transition error;
- J1/J2 HRV concordance (Lin CCC);
- same-threshold {q_label} sensitivity, PPV, and F1;
- ROC-AUC and PR-AUC calculated from continuous J1/J2 values; and
- HRV-window and expert-high-window coverage.

For AUC, a positive label means the expert-derived feature exceeded its
within-record {quantile:.0%} threshold. The continuous score is the algorithmic
feature divided by that record's expert-derived threshold. A score above 1
therefore reproduces the saved {q_label} F1 decision exactly.

## Primary-setting scorecard

{table}

Conditional pure-lane metrics must be interpreted with coverage. Their strong
CCC, F1, and AUC values describe the easier windows where a complete pure lane
exists; they do not describe a complete replacement.

## Why this architecture was retained

1. **UNSW remains the event owner.** Its pooled event F1 is
   **{_metric(selected['event_f1'], 6)}**, and the hierarchy preserves 100% RR
   and HRV-window coverage on the common event grid.
2. **Same-source-per-RR construction fixes the old fusion defect.** The selected
   hierarchy has **{_metric(selected['mixed_endpoint_rr_count'], 0)}** mixed-endpoint
   RRs versus **{_metric(old['mixed_endpoint_rr_count'], 0)}** for old per-beat
   fusion. RR MAE falls from **{_metric(old['rr_mae_ms'], 3)} ms** to
   **{_metric(selected['rr_mae_ms'], 3)} ms** ({old_rr_improvement:.1%} lower).
3. **It improves the unchanged baseline's interval fidelity.** RR MAE is
   {rr_improvement:.1%} lower than UNSW alone, and successive-RR-change MAE is
   {change_improvement:.1%} lower.
4. **NeuroKit is first because it supplies the strongest broadly available
   timing refinement.** Adding Zhai reduces RR MAE by another
   **{versus_nk:.4f} ms** relative to NeuroKit→UNSW. That aggregate increment is
   small, so Zhai is best described as a complementary fallback, not the sole
   reason the architecture works.
5. **UNSW is the final fallback because coverage matters.** Pure NeuroKit and
   pure Zhai HRV coverage are only **{_metric(pure_nk['hrv_window_coverage'], percent=True)}**
   and **{_metric(pure_zhai['hrv_window_coverage'], percent=True)}**. Treating
   missing expert-high windows as missed positives lowers their mean hard-veto
   F1 to **{_metric(pure_nk['mean_hard_veto_f1'])}** and
   **{_metric(pure_zhai['mean_hard_veto_f1'])}**, respectively.
6. **The selected stream preserves strong continuous ranking.** At {q_label},
   J1/J2 ROC-AUC are **{_metric(selected['j1_roc_auc'], 5)} /
   {_metric(selected['j2_roc_auc'], 5)}** and PR-AUC are
   **{_metric(selected['j1_pr_auc'], 5)} /
   {_metric(selected['j2_pr_auc'], 5)}**. The corresponding fixed-threshold F1
   values are **{_metric(selected['j1_threshold_f1'], 5)} /
   {_metric(selected['j2_threshold_f1'], 5)}**.

## What the comparison does and does not prove

The combined hierarchy is the **lowest-RR-MAE full-coverage interval-consistent
candidate**, but it is not a universal metric winner. NeuroKit→UNSW has a
slightly lower successive-change MAE (**{_metric(nk_unsw['successive_rr_change_mae_ms'], 4)}**
versus **{_metric(selected['successive_rr_change_mae_ms'], 4)} ms**), a higher
mean {q_label} F1 (**{_metric(nk_unsw['mean_threshold_f1'], 4)}** versus
**{_metric(selected['mean_threshold_f1'], 4)}**), and a higher mean PR-AUC
(**{_metric(nk_unsw['mean_pr_auc'], 4)}** versus
**{_metric(selected['mean_pr_auc'], 4)}**). Unchanged UNSW also has a slightly
higher mean {q_label} F1 (**{_metric(baseline['mean_threshold_f1'], 4)}**).

The selected hierarchy's transition-specific successive-change MAE is
**{_metric(selected['source_transition_change_mae_ms'], 3)} ms**, improved from
**{_metric(old['source_transition_change_mae_ms'], 3)} ms** for old per-beat
fusion but still materially above the unchanged baseline's global
successive-change MAE of **{_metric(baseline['successive_rr_change_mae_ms'], 3)}
ms**. Therefore, “landed on” means **retained as the most complete architecture
candidate for further validation**, not statistically proven superior. This is
why unchanged UNSW remains a parallel baseline.

## Why AUC alone does not select the winner

ROC-AUC measures ranking over all thresholds, while the intended downstream
system must operate at one locked patient-specific threshold. PR-AUC is more
informative for the rare high-HRV windows, but it is still conditional on the
available windows. Selection therefore uses AUC alongside fixed-threshold F1,
coverage, RR error, transition behavior, and architectural consistency.

## Important limitations

- These are HRV measurement-fidelity labels, **not seizure labels**.
- The 100-RR windows overlap by 99 intervals and are not independent samples.
- Thresholds and methods were examined on the same 48-record development set.
- The selected hierarchy does not dominate unchanged UNSW on every metric:
  J2 {q_label} F1 is **{_metric(selected['j2_threshold_f1'], 5)}** versus
  **{_metric(baseline['j2_threshold_f1'], 5)}** for baseline, and transition
  error still needs target-ECG validation.
- The result supports a **current working candidate**, not a clinically locked
  seizure detector or an automatically accepted replacement.

## Reproduce

```powershell
python scripts/analyze_rr_hrv_method_selection.py
```

Generated data files:

- `method_auc_metrics.csv`: per-method, per-feature AUC and threshold counts;
- `method_selection_scorecard.csv`: joined branch-level comparison;
- `method_selection_overview.png`: compact visual trade-off summary; and
- `analysis_metadata.json`: definitions, inputs, and warnings.
"""
    output_path.write_text(report, encoding="utf-8")


def plot_scorecard(
    scorecard: pd.DataFrame, output_path: Path, selected_method: str
) -> None:
    current = scorecard[scorecard["method"].isin(CURRENT_METHOD_ORDER)].copy()
    current["short_label"] = current["method"].map(
        {
            "unsw_baseline": "UNSW",
            "original_per_beat_fusion": "Old fusion",
            "interval_neurokit_unsw": "NK→UNSW",
            "interval_zhai_unsw": "Zhai→UNSW",
            "interval_neurokit_zhai_unsw": "NK→Zhai→UNSW",
            "pure_neurokit_lane": "Pure NK",
            "pure_zhai_lane": "Pure Zhai",
            "window_priority_neurokit_zhai_unsw": "Window priority",
        }
    )
    colors = {
        method: "#1F77B4" if method == selected_method else "#A6A6A6"
        for method in current["method"]
    }
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    panels = (
        ("rr_mae_ms", "RR MAE (ms; lower is better)"),
        ("mean_threshold_f1", "Mean J1/J2 q95 F1"),
        ("mean_roc_auc", "Mean J1/J2 ROC-AUC"),
        ("mean_pr_auc", "Mean J1/J2 PR-AUC"),
    )
    for axis, (column, title) in zip(axes.flat, panels):
        data = current[np.isfinite(current[column])]
        panel_colors = [colors[method] for method in data["method"]]
        axis.bar(data["short_label"], data[column], color=panel_colors)
        axis.set_title(title, fontweight="bold")
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.25)
        if column != "rr_mae_ms" and not data.empty:
            lower = max(0.0, float(data[column].min()) - 0.08)
            axis.set_ylim(lower, 1.01)
    figure.suptitle(
        "RR+HRV method selection: performance must be read with coverage",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = build_parser().parse_args()
    if not 0 < args.quantile < 1:
        raise ValueError("--quantile must be strictly between 0 and 1")
    audit_dir = args.audit_dir.expanduser().resolve()
    baseline_path = args.baseline_paired.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paired = _read_paired_windows(audit_dir, baseline_path)
    auc_metrics = calculate_auc_metrics(paired, args.quantile)
    saved_thresholds = _primary_setting(
        pd.read_csv(audit_dir / "pooled_threshold_metrics.csv")
    )
    validate_reconstructed_f1(auc_metrics, saved_thresholds, args.quantile)
    scorecard = build_scorecard(
        audit_dir,
        auc_metrics,
        quantile=args.quantile,
        selected_method=args.selected_method,
    )
    if args.selected_method not in set(scorecard["method"]):
        raise ValueError(f"Selected method is absent from scorecard: {args.selected_method}")

    auc_path = output_dir / "method_auc_metrics.csv"
    scorecard_path = output_dir / "method_selection_scorecard.csv"
    report_path = output_dir / "method_selection_report.md"
    plot_path = output_dir / "method_selection_overview.png"
    auc_metrics.to_csv(auc_path, index=False)
    scorecard.to_csv(scorecard_path, index=False)
    write_report(
        scorecard,
        report_path,
        quantile=args.quantile,
        selected_method=args.selected_method,
    )
    plot_scorecard(scorecard, plot_path, selected_method=args.selected_method)

    metadata = {
        "selected_method": args.selected_method,
        "architecture_name": "UNSW-owned NeuroKit-to-Zhai-to-UNSW per RR",
        "neurokit_configuration": SELECTED_CONFIGURATION,
        "association_tolerance_ms": SELECTED_TOLERANCE_MS,
        "expert_threshold_quantile": args.quantile,
        "auc_label": "expert HRV feature above its within-record quantile",
        "auc_score": "algorithm HRV feature divided by record-specific expert threshold",
        "auc_interpretation": "measurement fidelity, not seizure classification",
        "f1_validation": "reconstructed F1 and hard-veto F1 matched saved audit values",
        "inputs": {
            "audit_dir": str(audit_dir),
            "baseline_paired": str(baseline_path),
        },
        "outputs": [
            str(auc_path),
            str(scorecard_path),
            str(report_path),
            str(plot_path),
        ],
        "warnings": [
            "100-RR windows overlap by 99 intervals and are not independent.",
            "High-HRV labels are not seizure labels.",
            "Pure-lane AUC and F1 are conditional on available windows.",
            "Development-set comparison requires patient-specific held-out validation.",
        ],
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    selected = scorecard.set_index("method").loc[args.selected_method]
    print(f"Wrote method-selection analysis to {output_dir}")
    print(
        "Selected candidate mean metrics: "
        f"ROC-AUC={selected['mean_roc_auc']:.6f}, "
        f"PR-AUC={selected['mean_pr_auc']:.6f}, "
        f"threshold F1={selected['mean_threshold_f1']:.6f}, "
        f"RR MAE={selected['rr_mae_ms']:.6f} ms"
    )


if __name__ == "__main__":
    main()
