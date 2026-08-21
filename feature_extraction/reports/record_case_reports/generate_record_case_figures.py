"""Generate per-record figures for the five MIT--BIH detector case reports.

The script deliberately distinguishes expert beat annotations from detector
outputs.  Marker agreement in a short strip is explanatory; full-record
metrics remain the quantitative result.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb

from ecg_cascade.peaks import detect_neurokit_gradient, detect_unsw
from ecg_cascade.reliability import assess_rr_reliability


ROOT = Path(__file__).resolve().parents[3]
DATABASE = ROOT / "Datasets" / "mit-bih-arrhythmia-database-1.0" / "mit-bih-arrhythmia-database-1.0.0"
FIGURES = Path(__file__).resolve().parent / "figures"
METRICS = ROOT / "output" / "rr_hrv_architecture" / "mitdb_all48_v1" / "detector_metrics.csv"

WINDOWS = {
    "108": (1093.0, 1101.0),
    "113": (893.5, 901.5),
    "207": (505.5, 513.5),
    "222": (39.0, 47.0),
    "231": (462.0, 470.0),
}

BEAT_SYMBOLS = {
    "N", "L", "R", "A", "a", "J", "S", "V", "F", "e", "j", "E", "/", "f", "Q",
}


def y_at(signal: np.ndarray, samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=int)
    return signal[np.clip(samples, 0, signal.size - 1)]


def current_metric(metrics: pd.DataFrame, record: str, detector: str) -> pd.Series:
    row = metrics[(metrics["record"].astype(str) == record) & (metrics["detector"] == detector)]
    if len(row) != 1:
        raise RuntimeError(f"Expected one metric row for {record} {detector}; got {len(row)}")
    return row.iloc[0]


def make_strip(record: str, metrics: pd.DataFrame) -> None:
    path = str(DATABASE / record)
    rec = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")
    fs = float(rec.fs)
    signal = np.asarray(rec.p_signal[:, 0], dtype=float)

    primary = detect_unsw(signal, fs, orientation="original")
    secondary = detect_neurokit_gradient(
        signal,
        fs,
        orientation="original",
        minimum_delay_ms=250.0,
        minimum_delay_inclusive=True,
    )
    reliability = assess_rr_reliability(
        primary.peak_samples,
        secondary.peak_samples,
        primary.analysis_signal,
        sampling_rate_hz=fs,
        close_detection_exclusion_ms=150.0,
        support_tolerance_ms=50.0,
        r_fiducial_refinement_ms=50.0,
    )

    start_s, stop_s = WINDOWS[record]
    start = int(round(start_s * fs))
    stop = int(round(stop_s * fs))
    time = np.arange(start, stop) / fs

    expert_samples = np.asarray(
        [s for s, symbol in zip(ann.sample, ann.symbol, strict=True) if symbol in BEAT_SYMBOLS],
        dtype=int,
    )
    expert_symbols = np.asarray(
        [symbol for symbol in ann.symbol if symbol in BEAT_SYMBOLS], dtype=object
    )
    expert_mask = (expert_samples >= start) & (expert_samples < stop)
    expert_window = expert_samples[expert_mask]
    symbols_window = expert_symbols[expert_mask]

    events = reliability.primary_events
    primary_samples = events["primary_candidate_sample"].to_numpy(dtype=int)
    refined_samples = events["r_fiducial_sample"].to_numpy(dtype=int)
    support = events["qrs_supported"].to_numpy(dtype=bool)
    event_mask = (primary_samples >= start) & (primary_samples < stop)
    secondary_samples = reliability.secondary_collapsed_samples
    secondary_window = secondary_samples[(secondary_samples >= start) & (secondary_samples < stop)]

    m_initial = current_metric(metrics, record, "unsw_initial")
    m_refined = current_metric(metrics, record, "unsw_refined_r")

    fig, ax = plt.subplots(figsize=(12.4, 5.5), constrained_layout=True)
    ax.plot(time, signal[start:stop], color="#274c77", linewidth=1.05, label="Raw first lead")
    ax.scatter(
        expert_window / fs,
        y_at(signal, expert_window),
        marker="D",
        facecolors="white",
        edgecolors="black",
        linewidths=1.1,
        s=55,
        zorder=6,
        label="Expert beat annotation",
    )
    for sample, symbol in zip(expert_window, symbols_window, strict=True):
        ax.annotate(
            str(symbol),
            (sample / fs, signal[sample]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="black",
        )

    win_primary = primary_samples[event_mask]
    win_refined = refined_samples[event_mask]
    win_support = support[event_mask]
    ax.scatter(
        win_primary / fs,
        y_at(signal, win_primary),
        marker="^",
        s=52,
        color="#2a9d8f",
        zorder=5,
        label="UNSW initial QRS event",
    )
    ax.scatter(
        win_refined / fs,
        y_at(signal, win_refined),
        marker="x",
        s=60,
        linewidths=1.8,
        color="#e76f51",
        zorder=7,
        label="Refined R fiducial",
    )
    unsupported = win_primary[~win_support]
    if unsupported.size:
        ax.scatter(
            unsupported / fs,
            y_at(signal, unsupported),
            marker="o",
            facecolors="none",
            edgecolors="#d00000",
            linewidths=1.7,
            s=105,
            zorder=8,
            label="Primary lacks exact ±50-ms support",
        )
    if secondary_window.size:
        ymin, ymax = np.nanpercentile(signal[start:stop], [1, 99])
        marker_y = ymin - 0.10 * (ymax - ymin)
        ax.scatter(
            secondary_window / fs,
            np.full(secondary_window.size, marker_y),
            marker="|",
            s=150,
            linewidths=1.8,
            color="#6a4c93",
            label="NeuroKit-gradient context",
        )

    ax.set_title(
        f"Record {record}: complete RR–HRV architecture, explanatory {stop_s-start_s:.0f}-s strip\n"
        f"Full record: UNSW initial F1={m_initial.f1:.4f}; refined F1={m_refined.f1:.4f}"
    )
    ax.set_xlabel("Time from recording start (s)")
    ax.set_ylabel(f"{rec.sig_name[0]} amplitude ({rec.units[0]})")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    fig.savefig(FIGURES / f"record_{record}_complete_architecture_strip.png", dpi=220)
    plt.close(fig)


def make_metric_chart(record: str, metrics: pd.DataFrame) -> None:
    rows = [
        current_metric(metrics, record, "neurokit_300"),
        current_metric(metrics, record, "unsw_initial"),
        current_metric(metrics, record, "unsw_refined_r"),
    ]
    labels = ["NeuroKit\n300 ms", "UNSW\ninitial", "UNSW\nrefined R"]
    fp = np.asarray([float(row.fp) for row in rows])
    fn = np.asarray([float(row.fn) for row in rows])
    f1 = np.asarray([float(row.f1) for row in rows])
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8.6, 4.7), constrained_layout=True)
    width = 0.34
    bars_fp = ax.bar(x - width / 2, fp, width, color="#e76f51", label="False positives")
    bars_fn = ax.bar(x + width / 2, fn, width, color="#457b9d", label="False negatives")
    for bars in (bars_fp, bars_fn):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{int(bar.get_height())}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    for index, score in enumerate(f1):
        ax.text(index, max(fp[index], fn[index]) * 0.53, f"F1={score:.4f}", ha="center", fontsize=9,
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.82, "edgecolor": "0.75"})
    ax.set_xticks(x, labels)
    ax.set_ylabel("Full-record error count")
    ax.set_title(f"Record {record}: current full-record detector audit")
    ax.grid(axis="y", alpha=0.22)
    ax.legend()
    fig.savefig(FIGURES / f"record_{record}_current_metric_comparison.png", dpi=220)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(METRICS)
    metrics["record"] = metrics["record"].astype(str)
    for record in WINDOWS:
        make_strip(record, metrics)
        make_metric_chart(record, metrics)
        print(f"Wrote record {record} figures")


if __name__ == "__main__":
    main()
