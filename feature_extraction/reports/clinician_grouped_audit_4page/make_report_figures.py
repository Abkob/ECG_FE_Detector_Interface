"""Create reproducible figures for the four-page clinician audit report."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "feature_extraction" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecg_cascade.rigorous_audit import metric_record  # noqa: E402


AUDIT = ROOT / "feature_extraction" / "outputs" / "comprehensive_branch_matrix_v1" / "rigorous_audit_v2"
MATRIX = ROOT / "feature_extraction" / "outputs" / "comprehensive_branch_matrix_v1"
HERE = Path(__file__).resolve().parent

ORDER = ["artifact", "seizure", "abnormal_beat", "noise_active", "signal_unusable"]
LABELS = {
    "artifact": "Artifact / degraded",
    "seizure": "Ictal seizure",
    "abnormal_beat": "Abnormal beat",
    "noise_active": "Active noise",
    "signal_unusable": "Unusable signal",
}
TEAL = "#0A7C74"
NAVY = "#193946"
ORANGE = "#D67A35"
RED = "#B94A48"
GRAY = "#70858D"
LIGHT = "#DDE9E8"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.3,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "axes.edgecolor": "#AFC1C5",
            "axes.linewidth": 0.7,
            "xtick.color": "#425E67",
            "ytick.color": "#294750",
            "text.color": NAVY,
        }
    )


def performance_overview(selected: pd.DataFrame, baselines: pd.DataFrame) -> None:
    frame = selected.set_index("target_id").loc[ORDER]
    missing = (
        baselines.loc[baselines["baseline"].eq("feature missingness only")]
        .set_index("target_id")
        .loc[ORDER]
    )
    y = np.arange(len(ORDER))[::-1]
    row_ba = frame["balanced_accuracy"].to_numpy()
    low = frame["balanced_accuracy_group_bootstrap_ci_low"].to_numpy()
    high = frame["balanced_accuracy_group_bootstrap_ci_high"].to_numpy()
    group_ba = frame["group_macro_balanced_accuracy"].to_numpy()
    miss_ba = missing["group_macro_balanced_accuracy"].to_numpy()

    fig, ax = plt.subplots(figsize=(7.05, 2.15))
    ax.errorbar(
        row_ba,
        y,
        xerr=np.vstack([row_ba - low, high - row_ba]),
        fmt="o",
        color=TEAL,
        ecolor="#73AFA9",
        capsize=3,
        lw=1.4,
        label="Row-pooled BA (95% group bootstrap)",
        zorder=3,
    )
    ax.scatter(group_ba, y, marker="D", s=29, color=NAVY, label="Group-macro BA", zorder=4)
    ax.scatter(miss_ba, y, marker="^", s=32, color=ORANGE, label="Missingness-only group BA", zorder=4)
    for idx, (_, row) in enumerate(frame.iterrows()):
        ax.text(1.008, y[idx], f"g={int(row['groups'])}", va="center", ha="left", fontsize=7.4, color=GRAY)
    ax.axvline(0.5, color="#A9B9BD", ls="--", lw=0.9)
    ax.set_yticks(y, [LABELS[key] for key in ORDER])
    ax.set_xlim(0.45, 1.035)
    ax.set_xlabel("Balanced accuracy (0.50 = chance for a balanced binary task)")
    ax.grid(axis="x", color="#E6EEEE", lw=0.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False, fontsize=7.2)
    fig.tight_layout(pad=0.5)
    fig.savefig(HERE / "performance_overview.png", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def group_variability(cases: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    rows: list[dict[str, object]] = []
    for (target, group_id), group in cases.groupby(["target_id", "lineage_group_id"], sort=True):
        metrics = metric_record(group["y_true"].to_numpy(int), group["y_pred"].to_numpy(int))
        rows.append({"target_id": target, "group_id": group_id, **metrics})
    frame = pd.DataFrame(rows)
    summary: dict[str, dict[str, float | int]] = {}

    fig, ax = plt.subplots(figsize=(7.05, 2.35))
    rng = np.random.default_rng(20260904)
    for index, target in enumerate(ORDER):
        target_rows = frame.loc[frame["target_id"].eq(target)]
        mixed = target_rows["balanced_accuracy"].dropna().to_numpy(float)
        single = int(target_rows["balanced_accuracy"].isna().sum())
        jitter = rng.uniform(-0.12, 0.12, size=mixed.size)
        ax.scatter(index + jitter, mixed, s=22, color=TEAL, edgecolor="white", linewidth=0.45, alpha=0.9, zorder=3)
        if mixed.size:
            q1, median, q3 = np.quantile(mixed, [0.25, 0.5, 0.75])
            ax.vlines(index, q1, q3, color=NAVY, lw=5, alpha=0.75, zorder=2)
            ax.hlines(median, index - 0.18, index + 0.18, color=NAVY, lw=1.5, zorder=4)
            summary[target] = {
                "groups": int(target_rows.shape[0]),
                "mixed_groups": int(mixed.size),
                "single_class_groups": single,
                "median_balanced_accuracy": float(median),
                "q1": float(q1),
                "q3": float(q3),
                "minimum": float(mixed.min()),
                "maximum": float(mixed.max()),
                "below_060": int(np.sum(mixed < 0.60)),
                "at_least_080": int(np.sum(mixed >= 0.80)),
            }
        ax.text(index, 0.465, f"{mixed.size} mixed\n{single} single-class", ha="center", va="top", fontsize=6.8, color=GRAY)
    ax.axhline(0.5, color="#A9B9BD", ls="--", lw=0.9)
    ax.set_ylim(0.42, 1.025)
    ax.set_ylabel("Per-lineage balanced accuracy")
    ax.set_xticks(np.arange(len(ORDER)), ["Artifact", "Seizure", "Abnormal\nbeat", "Noise", "Unusable"])
    ax.grid(axis="y", color="#E6EEEE", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.6)
    fig.savefig(HERE / "group_variability.png", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return summary


def mixed_context(cases: pd.DataFrame, observations: pd.DataFrame) -> dict[str, object]:
    merged = cases.merge(
        observations[["row_id", "target_noise_active_binary", "target_abnormal_beat_binary"]],
        on="row_id",
        how="left",
        validate="many_to_one",
    )
    values: list[tuple[str, float, int, int]] = []

    abnormal = merged.loc[(merged["target_id"].eq("abnormal_beat")) & (merged["dataset_key"].eq("nstdb"))]
    for noise_state, label in ((0.0, "Abnormal detector\nclean interval"), (1.0, "Abnormal detector\nactive noise")):
        group = abnormal.loc[abnormal["target_noise_active_binary"].eq(noise_state)]
        positive = group.loc[group["y_true"].eq(1)]
        values.append((label, float((positive["y_pred"] == 1).mean()), int((positive["y_pred"] == 1).sum()), int(positive.shape[0])))

    noise = merged.loc[(merged["target_id"].eq("noise_active")) & merged["target_abnormal_beat_binary"].notna()]
    for beat_state, label in ((0.0, "Noise detector\nnormal beats"), (1.0, "Noise detector\nabnormal beats")):
        group = noise.loc[noise["target_abnormal_beat_binary"].eq(beat_state)]
        positive = group.loc[group["y_true"].eq(1)]
        values.append((label, float((positive["y_pred"] == 1).mean()), int((positive["y_pred"] == 1).sum()), int(positive.shape[0])))

    artifact = merged.loc[(merged["target_id"].eq("artifact")) & (merged["dataset_key"].eq("nstdb")) & merged["target_abnormal_beat_binary"].notna()]
    for beat_state, label in ((0.0, "Artifact detector\nnormal beats"), (1.0, "Artifact detector\nabnormal beats")):
        group = artifact.loc[artifact["target_abnormal_beat_binary"].eq(beat_state)]
        positive = group.loc[group["y_true"].eq(1)]
        values.append((label, float((positive["y_pred"] == 1).mean()), int((positive["y_pred"] == 1).sum()), int(positive.shape[0])))

    labels = [item[0] for item in values]
    rates = [item[1] for item in values]
    fig, ax = plt.subplots(figsize=(7.05, 2.2))
    x = np.arange(len(values))
    colors = [NAVY, RED, NAVY, RED, NAVY, RED]
    bars = ax.bar(x, rates, width=0.68, color=colors, alpha=0.9)
    for bar, (_, rate, detected, total) in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.022, f"{rate:.0%}\n({detected}/{total})", ha="center", va="bottom", fontsize=7.2)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Sensitivity within context")
    ax.set_xticks(x, labels)
    ax.grid(axis="y", color="#E6EEEE", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0.8, color="#A9B9BD", ls=":", lw=0.9)
    fig.tight_layout(pad=0.6)
    fig.savefig(HERE / "mixed_context.png", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    crosstab = pd.crosstab(abnormal["target_noise_active_binary"], abnormal["y_true"])
    return {
        "bars": [
            {"context": label.replace("\n", " "), "sensitivity": rate, "detected": detected, "positive_total": total}
            for label, rate, detected, total in values
        ],
        "nstdb_known_cross_tab": {
            "clean_normal": int(crosstab.loc[0.0, 0]),
            "clean_abnormal": int(crosstab.loc[0.0, 1]),
            "noisy_normal": int(crosstab.loc[1.0, 0]),
            "noisy_abnormal": int(crosstab.loc[1.0, 1]),
        },
    }


def main() -> None:
    _style()
    selected = pd.read_csv(AUDIT / "selected_experiments.csv")
    baselines = pd.read_csv(AUDIT / "leakage_and_naive_baselines.csv")
    cases = pd.read_csv(AUDIT / "selected_case_predictions.csv.gz", low_memory=False)
    observations = pd.read_csv(
        MATRIX / "observations_with_labels.csv",
        low_memory=False,
        usecols=["row_id", "target_noise_active_binary", "target_abnormal_beat_binary"],
    )
    performance_overview(selected, baselines)
    group_summary = group_variability(cases)
    mixed_summary = mixed_context(cases, observations)
    (HERE / "report_statistics.json").write_text(
        json.dumps({"group_summary": group_summary, "mixed_context": mixed_summary}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
