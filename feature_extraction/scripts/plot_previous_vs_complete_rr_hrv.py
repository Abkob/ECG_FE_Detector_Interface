"""Plot FP/FN error exchange for the previous and complete RR-HRV architectures.

The previous architecture is represented by NeuroKit alone and the historical
0--100 ms Pan--Tompkins hard gate. The complete architecture contributes the
UNSW primary detector before and after R-fiducial refinement. All values are
read from the saved MIT-BIH expert-annotation audits; no metrics are recomputed
or entered manually.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


RECORDS = [108, 113, 207, 222, 231]
METHOD_ORDER = [
    "Previous: NeuroKit alone",
    "Previous: NK + PT hard gate",
    "New: UNSW initial",
    "New: UNSW refined R",
]

STYLE = {
    "Previous: NeuroKit alone": {"color": "#c43c35", "marker": "o"},
    "Previous: NK + PT hard gate": {"color": "#128a8a", "marker": "X"},
    "New: UNSW initial": {"color": "#2f6db0", "marker": "^"},
    "New: UNSW refined R": {"color": "#d58b13", "marker": "D"},
}


def load_comparison(previous_csv: Path, new_csv: Path) -> pd.DataFrame:
    previous = pd.read_csv(previous_csv)
    previous = previous[previous["record"].isin(RECORDS)].copy()
    previous["architecture"] = previous["method"].map(
        {
            "NeuroKit alone": "Previous: NeuroKit alone",
            "NeuroKit + PT hard gate 0..100 ms": "Previous: NK + PT hard gate",
        }
    )

    new = pd.read_csv(new_csv)
    new = new[
        new["record"].isin(RECORDS)
        & new["detector"].isin(["unsw_initial", "unsw_refined_r"])
    ].copy()
    new["architecture"] = new["detector"].map(
        {
            "unsw_initial": "New: UNSW initial",
            "unsw_refined_r": "New: UNSW refined R",
        }
    )

    columns = ["record", "architecture", "tp", "fp", "fn", "f1"]
    comparison = pd.concat(
        [
            previous.rename(
                columns={"TP": "tp", "FP": "fp", "FN": "fn", "F1": "f1"}
            )[columns],
            new[columns],
        ],
        ignore_index=True,
    )
    comparison["architecture"] = pd.Categorical(
        comparison["architecture"], categories=METHOD_ORDER, ordered=True
    )
    return comparison.sort_values(["record", "architecture"]).reset_index(drop=True)


def plot_shared_scale(comparison: pd.DataFrame, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(11.8, 7.4), constrained_layout=True)

    for record in RECORDS:
        rows = comparison[comparison["record"] == record].set_index("architecture")
        old_start = rows.loc["Previous: NeuroKit alone"]
        old_gate = rows.loc["Previous: NK + PT hard gate"]
        new_start = rows.loc["New: UNSW initial"]
        new_refined = rows.loc["New: UNSW refined R"]

        axis.plot(
            [old_start.fp, old_gate.fp],
            [old_start.fn, old_gate.fn],
            color="#8a8a8a",
            linewidth=1.3,
            alpha=0.65,
            zorder=1,
        )
        axis.plot(
            [new_start.fp, new_refined.fp],
            [new_start.fn, new_refined.fn],
            color=STYLE["New: UNSW refined R"]["color"],
            linewidth=1.3,
            linestyle="--",
            alpha=0.8,
            zorder=1,
        )

    for method in METHOD_ORDER:
        rows = comparison[comparison["architecture"] == method]
        axis.scatter(
            rows["fp"],
            rows["fn"],
            s=78,
            edgecolor="white",
            linewidth=0.7,
            label=method,
            zorder=3,
            **STYLE[method],
        )

    zoom_axis = inset_axes(
        axis,
        width="43%",
        height="34%",
        loc="lower right",
        borderpad=2.0,
    )
    refined = comparison[comparison["architecture"] == "New: UNSW refined R"]
    initial = comparison[comparison["architecture"] == "New: UNSW initial"]
    zoom_axis.scatter(
        initial["fp"],
        initial["fn"],
        s=60,
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
        **STYLE["New: UNSW initial"],
    )
    zoom_axis.scatter(
        refined["fp"],
        refined["fn"],
        s=65,
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
        **STYLE["New: UNSW refined R"],
    )
    zoom_offsets = {
        108: (6, 8),
        113: (6, 8),
        207: (-28, 8),
        222: (6, 8),
        231: (6, 8),
    }
    for row in refined.itertuples(index=False):
        dx, dy = zoom_offsets[int(row.record)]
        zoom_axis.annotate(
            str(int(row.record)),
            (row.fp, row.fn),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
        )
    zoom_axis.set_xlim(-5, 165)
    zoom_axis.set_ylim(-2, 17)
    zoom_axis.set_title("New UNSW outputs: magnified", fontsize=9)
    zoom_axis.set_xlabel("FP", fontsize=8)
    zoom_axis.set_ylabel("FN", fontsize=8)
    zoom_axis.tick_params(labelsize=7)
    zoom_axis.grid(True, color="#d8d8d8", linewidth=0.5, alpha=0.7)

    axis.set_xlabel("False positives (lower is better)")
    axis.set_ylabel("False negatives (lower is better)")
    axis.set_title("Five difficult records: previous versus complete RR-HRV architecture")
    axis.grid(True, color="#d8d8d8", linewidth=0.6, alpha=0.7)
    axis.set_xlim(left=-25)
    axis.set_ylim(bottom=-35)
    axis.legend(loc="upper right", frameon=True)
    axis.text(
        0.01,
        0.99,
        "Grey line: effect of the old hard gate   |   Dashed gold: effect of R-fiducial refinement",
        transform=axis.transAxes,
        va="top",
        fontsize=9,
        color="#555555",
    )
    fig.savefig(output_path, dpi=220)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_small_multiples(comparison: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.5), constrained_layout=True)
    axes = axes.ravel()

    for axis, record in zip(axes, RECORDS, strict=False):
        rows = comparison[comparison["record"] == record].set_index("architecture")
        old_start = rows.loc["Previous: NeuroKit alone"]
        old_gate = rows.loc["Previous: NK + PT hard gate"]
        new_start = rows.loc["New: UNSW initial"]
        new_refined = rows.loc["New: UNSW refined R"]

        axis.plot(
            [old_start.fp, old_gate.fp],
            [old_start.fn, old_gate.fn],
            color="#8a8a8a",
            linewidth=1.4,
            zorder=1,
        )
        axis.plot(
            [new_start.fp, new_refined.fp],
            [new_start.fn, new_refined.fn],
            color=STYLE["New: UNSW refined R"]["color"],
            linestyle="--",
            linewidth=1.4,
            zorder=1,
        )

        for method in METHOD_ORDER:
            row = rows.loc[method]
            axis.scatter(
                row.fp,
                row.fn,
                s=78,
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
                **STYLE[method],
            )
            short = {
                "Previous: NeuroKit alone": "NK",
                "Previous: NK + PT hard gate": "Gate",
                "New: UNSW initial": "UNSW-i",
                "New: UNSW refined R": "UNSW-r",
            }[method]
            same_unsw = (
                new_start.fp == new_refined.fp and new_start.fn == new_refined.fn
            )
            if same_unsw and method == "New: UNSW initial":
                continue
            if same_unsw and method == "New: UNSW refined R":
                short = "UNSW-i/r"
            annotation_offsets = {
                "Previous: NeuroKit alone": (6, 6),
                "Previous: NK + PT hard gate": (6, 6),
                "New: UNSW initial": (-65, 12),
                "New: UNSW refined R": (6, -26) if not same_unsw else (6, 8),
            }
            custom_offsets = {
                (113, "Previous: NK + PT hard gate"): (10, 20),
                (113, "New: UNSW refined R"): (10, -20),
                (207, "New: UNSW initial"): (-105, 22),
                (207, "New: UNSW refined R"): (-10, 48),
                (222, "New: UNSW initial"): (10, 42),
                (222, "New: UNSW refined R"): (10, 8),
                (231, "Previous: NK + PT hard gate"): (10, 20),
                (231, "New: UNSW refined R"): (20, -20),
            }
            axis.annotate(
                f"{short}\nFP {int(row.fp)}, FN {int(row.fn)}",
                (row.fp, row.fn),
                xytext=custom_offsets.get((record, method), annotation_offsets[method]),
                textcoords="offset points",
                fontsize=7.5,
                arrowprops=(
                    {"arrowstyle": "-", "color": "#777777", "linewidth": 0.7}
                    if method.startswith("New:") and not same_unsw
                    else None
                ),
            )

        axis.set_title(f"Record {record}", fontweight="bold")
        axis.set_xlabel("False positives")
        axis.set_ylabel("False negatives")
        axis.grid(True, color="#d8d8d8", linewidth=0.6, alpha=0.7)
        x_max = max(rows["fp"].max() * 1.18, 20)
        y_max = max(rows["fn"].max() * 1.16, 20)
        axis.set_xlim(-0.08 * x_max, x_max)
        axis.set_ylim(-0.12 * y_max, y_max)

    legend_axis = axes[-1]
    legend_axis.axis("off")
    handles = []
    for method in METHOD_ORDER:
        handles.append(
            plt.Line2D(
                [],
                [],
                linestyle="none",
                markersize=8,
                markeredgecolor="white",
                label=method,
                **STYLE[method],
            )
        )
    legend_axis.legend(handles=handles, loc="center", frameon=False, fontsize=10)
    legend_axis.text(
        0.5,
        0.22,
        "Axes are scaled separately per record.\nUse the shared-scale plot for cross-record magnitude.",
        transform=legend_axis.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        color="#555555",
    )

    fig.suptitle(
        "How each architecture changed FP/FN errors",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=220)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-csv", type=Path, required=True)
    parser.add_argument("--new-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison = load_comparison(args.previous_csv, args.new_csv)
    comparison.to_csv(args.output_dir / "five_record_architecture_comparison.csv", index=False)
    plot_shared_scale(
        comparison,
        args.output_dir / "five_record_error_exchange_shared_scale.png",
    )
    plot_small_multiples(
        comparison,
        args.output_dir / "five_record_error_exchange_small_multiples.png",
    )


if __name__ == "__main__":
    main()
