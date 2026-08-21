"""Plot the five worst NeuroKit records before and after the PT support gate.

This reproduces the fixed audit configuration:

* NeuroKit2 0.2.13 ``neurokit`` cleaning and peak detection;
* ``correct_artifacts=False``;
* expert MIT-BIH ``atr`` beat symbols;
* WFDB one-to-one matching within +/-75 ms;
* hard gate retains a NeuroKit peak only if a Pan-Tompkins event satisfies
  0 <= t_PT - t_NK <= 100 ms.

The hard gate is visualized as an ablation.  The script does not recommend it
as the final detector and does not interpret detector disagreement as artifact.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import neurokit2
import numpy as np
import wfdb
from wfdb import processing

from ecg_cascade.peaks import detect_r_peaks
from ecg_cascade.rpeak_viewer import MIT_BIH_BEAT_SYMBOLS


RECORDS = ("113", "222", "207", "231", "108")
MATCH_TOLERANCE_MS = 75.0
GATE_MIN_MS = 0.0
GATE_MAX_MS = 100.0
EXCERPT_DURATION_S = 8.0

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MITDB_DIR = (
    PROJECT_ROOT
    / "Datasets"
    / "mit-bih-arrhythmia-database-1.0"
    / "mit-bih-arrhythmia-database-1.0.0"
)
OUTPUT_DIR = PROJECT_ROOT / "output" / "rpeak_pan_gate_comparison"
PDF_OUTPUT_DIR = PROJECT_ROOT / "output" / "pdf"


@dataclass(frozen=True)
class Classification:
    detected: np.ndarray
    true_detection_samples: np.ndarray
    matched_reference_samples: np.ndarray
    false_detection_samples: np.ndarray
    missed_reference_samples: np.ndarray
    sensitivity: float
    ppv: float
    f1: float

    @property
    def tp(self) -> int:
        return int(self.true_detection_samples.size)

    @property
    def fp(self) -> int:
        return int(self.false_detection_samples.size)

    @property
    def fn(self) -> int:
        return int(self.missed_reference_samples.size)


@dataclass(frozen=True)
class RecordAudit:
    name: str
    lead: str
    fs: float
    signal: np.ndarray
    reference: np.ndarray
    reference_symbols: tuple[str, ...]
    neurokit_peaks: np.ndarray
    pan_events: np.ndarray
    gated_peaks: np.ndarray
    baseline: Classification
    gated: Classification
    excerpt_start_s: float


def classify(reference: np.ndarray, detected: np.ndarray, fs: float) -> Classification:
    comparison = processing.compare_annotations(
        np.asarray(reference, dtype=np.int64),
        np.asarray(detected, dtype=np.int64),
        window_width=max(1, int(round(MATCH_TOLERANCE_MS * fs / 1000.0))),
    )
    sensitivity = float(comparison.sensitivity)
    ppv = float(comparison.positive_predictivity)
    f1 = 2.0 * sensitivity * ppv / (sensitivity + ppv) if sensitivity + ppv else np.nan
    return Classification(
        detected=np.asarray(detected, dtype=np.int64),
        true_detection_samples=np.asarray(comparison.matched_test_sample, dtype=np.int64),
        matched_reference_samples=np.asarray(comparison.matched_ref_sample, dtype=np.int64),
        false_detection_samples=np.asarray(comparison.unmatched_test_sample, dtype=np.int64),
        missed_reference_samples=np.asarray(comparison.unmatched_ref_sample, dtype=np.int64),
        sensitivity=sensitivity,
        ppv=ppv,
        f1=float(f1),
    )


def apply_pan_support_gate(neurokit_peaks: np.ndarray, pan_events: np.ndarray, fs: float) -> np.ndarray:
    """Retain peaks with any PT event in the fixed forward timing interval."""

    peaks = np.asarray(neurokit_peaks, dtype=np.int64)
    events = np.asarray(pan_events, dtype=np.int64)
    lower = int(round(GATE_MIN_MS * fs / 1000.0))
    upper = int(round(GATE_MAX_MS * fs / 1000.0))
    event_indices = np.searchsorted(events, peaks + lower, side="left")
    in_range = event_indices < events.size
    supported = np.zeros(peaks.size, dtype=bool)
    supported[in_range] = events[event_indices[in_range]] <= peaks[in_range] + upper
    return peaks[supported]


def densest_error_window(
    *,
    duration_s: float,
    fs: float,
    baseline: Classification,
    gated: Classification,
) -> float:
    """Choose a reproducible excerpt rich in detections changed by the gate."""

    # Because the gated output is a strict subset of the NeuroKit output, these
    # samples are exactly the candidates that the gate removed.  Centering the
    # excerpt on them makes the before/after mechanism visible: in records 113,
    # 222, and 231 the removed detections are mostly false positives; in 207 and
    # 108 many are true detections that become false negatives.
    event_samples = np.setdiff1d(
        baseline.detected,
        gated.detected,
        assume_unique=True,
    )
    if event_samples.size == 0:
        event_samples = np.concatenate(
            [baseline.false_detection_samples, baseline.missed_reference_samples]
        )
    event_times = np.sort(event_samples / fs)
    maximum_start = max(0.0, duration_s - EXCERPT_DURATION_S)
    if event_times.size == 0:
        return min(2.0, maximum_start)

    candidates = np.arange(2.0, max(2.01, maximum_start) + 0.001, 0.5)
    if candidates.size == 0:
        return 0.0
    counts = np.asarray(
        [
            np.count_nonzero((event_times >= start) & (event_times < start + EXCERPT_DURATION_S))
            for start in candidates
        ]
    )
    return float(candidates[int(np.argmax(counts))])


def load_record(name: str) -> RecordAudit:
    record_base = MITDB_DIR / name
    record = wfdb.rdrecord(str(record_base), channels=[0], physical=True)
    signal = np.asarray(record.p_signal[:, 0], dtype=np.float64)
    fs = float(record.fs)
    annotation = wfdb.rdann(str(record_base), "atr")
    beat_mask = np.asarray(
        [symbol in MIT_BIH_BEAT_SYMBOLS for symbol in annotation.symbol], dtype=bool
    )
    reference = np.asarray(annotation.sample, dtype=np.int64)[beat_mask]
    reference_symbols = tuple(
        str(symbol) for symbol, keep in zip(annotation.symbol, beat_mask) if keep
    )
    neurokit_peaks = detect_r_peaks(signal, fs, method="neurokit").peak_samples
    pan_events = detect_r_peaks(signal, fs, method="pantompkins1985").peak_samples
    gated_peaks = apply_pan_support_gate(neurokit_peaks, pan_events, fs)
    baseline = classify(reference, neurokit_peaks, fs)
    gated = classify(reference, gated_peaks, fs)
    duration_s = signal.size / fs
    excerpt_start = densest_error_window(
        duration_s=duration_s,
        fs=fs,
        baseline=baseline,
        gated=gated,
    )
    return RecordAudit(
        name=name,
        lead=str(record.sig_name[0]),
        fs=fs,
        signal=signal,
        reference=reference,
        reference_symbols=reference_symbols,
        neurokit_peaks=neurokit_peaks,
        pan_events=pan_events,
        gated_peaks=gated_peaks,
        baseline=baseline,
        gated=gated,
        excerpt_start_s=excerpt_start,
    )


def write_metrics(records: list[RecordAudit]) -> Path:
    path = OUTPUT_DIR / "five_record_neurokit_vs_pan_gate_metrics.csv"
    fields = [
        "record",
        "lead",
        "method",
        "reference_beats",
        "detected_beats",
        "TP",
        "FP",
        "FN",
        "sensitivity",
        "PPV",
        "F1",
        "excerpt_start_s",
        "excerpt_end_s",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            for method, result in [
                ("NeuroKit alone", record.baseline),
                ("NeuroKit + PT hard gate 0..100 ms", record.gated),
            ]:
                writer.writerow(
                    {
                        "record": record.name,
                        "lead": record.lead,
                        "method": method,
                        "reference_beats": record.reference.size,
                        "detected_beats": result.detected.size,
                        "TP": result.tp,
                        "FP": result.fp,
                        "FN": result.fn,
                        "sensitivity": f"{result.sensitivity:.9f}",
                        "PPV": f"{result.ppv:.9f}",
                        "F1": f"{result.f1:.9f}",
                        "excerpt_start_s": f"{record.excerpt_start_s:.3f}",
                        "excerpt_end_s": f"{record.excerpt_start_s + EXCERPT_DURATION_S:.3f}",
                    }
                )
    return path


def make_summary_figure(records: list[RecordAudit]) -> plt.Figure:
    names = [record.name for record in records]
    baseline_color = "#008C95"
    gate_color = "#C00000"
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=False)
    figure.subplots_adjust(left=0.065, right=0.985, top=0.90, bottom=0.10, hspace=0.30, wspace=0.18)
    figure.suptitle(
        "Five worst NeuroKit records: effect of the proposed Pan-Tompkins hard-support gate",
        fontsize=17,
        fontweight="bold",
    )

    for axis, metric, title in [
        (axes[0, 0], "f1", "F1: overall detection balance"),
        (axes[0, 1], "sensitivity", "Sensitivity: fraction of expert beats found"),
        (axes[1, 0], "ppv", "PPV: fraction of reported detections that are correct"),
    ]:
        baseline_values = np.asarray([getattr(record.baseline, metric) for record in records])
        gate_values = np.asarray([getattr(record.gated, metric) for record in records])
        positions = np.arange(len(records))
        axis.bar(positions - 0.18, baseline_values, width=0.36, color=baseline_color, label="NeuroKit alone")
        axis.bar(positions + 0.18, gate_values, width=0.36, color=gate_color, label="NeuroKit + PT 0-100 ms hard gate")
        for position, before, after in zip(positions, baseline_values, gate_values):
            axis.text(position - 0.18, before + 0.018, f"{before:.3f}", ha="center", va="bottom", fontsize=9)
            axis.text(position + 0.18, after + 0.018, f"{after:.3f}", ha="center", va="bottom", fontsize=9)
        axis.set_xticks(positions, names)
        axis.set_ylim(0, 1.12)
        axis.set_xlabel("MIT-BIH record")
        axis.set_ylabel(metric.upper() if metric == "f1" else metric.capitalize())
        axis.set_title(title, fontweight="bold")
        axis.grid(axis="y", alpha=0.22)
    axes[0, 0].legend(loc="upper center", fontsize=9)

    error_axis = axes[1, 1]
    for index, record in enumerate(records):
        error_axis.annotate(
            "",
            xy=(record.gated.fp, record.gated.fn),
            xytext=(record.baseline.fp, record.baseline.fn),
            arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 1.5},
        )
        error_axis.scatter(record.baseline.fp, record.baseline.fn, color=baseline_color, s=65, zorder=3)
        error_axis.scatter(record.gated.fp, record.gated.fn, color=gate_color, marker="X", s=75, zorder=3)
        error_axis.annotate(
            record.name,
            xy=(record.gated.fp, record.gated.fn),
            xytext=(6, 5 + 10 * index),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )
    error_axis.set_xlabel("False positives (lower is better)")
    error_axis.set_ylabel("False negatives (lower is better)")
    error_axis.set_title("Error exchange caused by hard gating", fontweight="bold")
    error_axis.grid(alpha=0.22)
    error_axis.scatter([], [], color=baseline_color, s=65, label="NeuroKit alone")
    error_axis.scatter([], [], color=gate_color, marker="X", s=75, label="After hard gate")
    error_axis.legend(loc="upper right")

    figure.text(
        0.5,
        0.025,
        "Gate definition: retain a NeuroKit peak only when 0 ≤ t_PT − t_NK ≤ 100 ms. "
        "Expert matching: WFDB one-to-one comparison within ±75 ms. The gate is an ablation, not ground truth.",
        ha="center",
        fontsize=10,
    )
    return figure


def _visible(samples: np.ndarray, first: int, last: int) -> np.ndarray:
    values = np.asarray(samples, dtype=np.int64)
    return values[(values >= first) & (values < last)]


def _scatter_samples(
    axis: plt.Axes,
    signal: np.ndarray,
    fs: float,
    samples: np.ndarray,
    *,
    color: str,
    marker: str,
    label: str,
    size: float,
    facecolors: str | None = None,
    zorder: int = 4,
) -> None:
    values = np.asarray(samples, dtype=np.int64)
    if values.size == 0:
        return
    options: dict[str, object] = {"edgecolors": color, "linewidths": 1.2} if facecolors == "none" else {"color": color}
    if facecolors == "none":
        options["facecolors"] = "none"
    axis.scatter(values / fs, signal[values], marker=marker, s=size, label=label, zorder=zorder, **options)


def plot_excerpt(
    axis: plt.Axes,
    record: RecordAudit,
    result: Classification,
    *,
    gated: bool,
) -> None:
    first = int(round(record.excerpt_start_s * record.fs))
    last = min(record.signal.size, int(round((record.excerpt_start_s + EXCERPT_DURATION_S) * record.fs)))
    samples = np.arange(first, last)
    axis.plot(samples / record.fs, record.signal[first:last], color="#17365D", linewidth=0.8, label="ECG")
    _scatter_samples(
        axis,
        record.signal,
        record.fs,
        _visible(record.reference, first, last),
        color="#111111",
        marker="d",
        label="Expert .atr beat",
        size=34,
        facecolors="none",
    )
    _scatter_samples(
        axis,
        record.signal,
        record.fs,
        _visible(result.true_detection_samples, first, last),
        color="#1B7F3A",
        marker="x",
        label="Matched NeuroKit detection",
        size=45,
    )
    _scatter_samples(
        axis,
        record.signal,
        record.fs,
        _visible(result.false_detection_samples, first, last),
        color="#C00000",
        marker="X",
        label="NeuroKit false positive",
        size=58,
    )
    _scatter_samples(
        axis,
        record.signal,
        record.fs,
        _visible(result.missed_reference_samples, first, last),
        color="#E59400",
        marker="v",
        label="Expert beat missed",
        size=64,
        zorder=5,
    )
    if gated:
        _scatter_samples(
            axis,
            record.signal,
            record.fs,
            _visible(record.pan_events, first, last),
            color="#7A3DB8",
            marker="|",
            label="Pan-Tompkins event",
            size=85,
            zorder=3,
        )
    axis.set_xlim(first / record.fs, last / record.fs)
    axis.grid(alpha=0.18)
    axis.set_ylabel(record.lead + " (mV)")
    mode = "NeuroKit alone" if not gated else "After PT hard gate (0-100 ms)"
    axis.set_title(
        f"Record {record.name} — {mode}\n"
        f"record-level TP={result.tp:,}, FP={result.fp:,}, FN={result.fn:,}, "
        f"Se={result.sensitivity:.3f}, PPV={result.ppv:.3f}, F1={result.f1:.3f}",
        fontsize=10,
        fontweight="bold",
    )


def make_examples_figure(records: list[RecordAudit]) -> plt.Figure:
    figure, axes = plt.subplots(len(records), 2, figsize=(18, 18.5), constrained_layout=False)
    figure.subplots_adjust(left=0.055, right=0.99, top=0.945, bottom=0.085, hspace=0.62, wspace=0.12)
    figure.suptitle(
        "Same ECG excerpts before and after the proposed Pan-Tompkins support gate",
        fontsize=17,
        fontweight="bold",
    )
    for row, record in enumerate(records):
        plot_excerpt(axes[row, 0], record, record.baseline, gated=False)
        plot_excerpt(axes[row, 1], record, record.gated, gated=True)
        axes[row, 0].set_xlabel("Time from record start (s)")
        axes[row, 1].set_xlabel("Time from record start (s)")
    by_label = {}
    for axis in axes.ravel():
        handles, labels = axis.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            by_label.setdefault(label, handle)
    figure.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=5,
        frameon=True,
        fontsize=10,
    )
    figure.text(
        0.5,
        0.013,
        "Each row uses exactly the same automatically selected 8-second excerpt in both columns. "
        "The excerpt is chosen by the highest combined concentration of baseline and gated errors.",
        ha="center",
        fontsize=10,
    )
    return figure


def make_record_figure(record: RecordAudit) -> plt.Figure:
    """Create a publication-sized before/after figure for one record."""

    figure, axes = plt.subplots(1, 2, figsize=(15.5, 5.1), constrained_layout=False)
    figure.subplots_adjust(left=0.065, right=0.99, top=0.82, bottom=0.24, wspace=0.13)
    figure.suptitle(
        f"MIT-BIH record {record.name}: same ECG excerpt before and after hard gating",
        fontsize=16,
        fontweight="bold",
    )
    plot_excerpt(axes[0], record, record.baseline, gated=False)
    plot_excerpt(axes[1], record, record.gated, gated=True)
    for axis in axes:
        axis.set_xlabel("Time from record start (s)")

    by_label: dict[str, object] = {}
    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            by_label.setdefault(label, handle)
    figure.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.075),
        ncol=3,
        frameon=True,
        fontsize=9,
    )
    figure.text(
        0.5,
        0.02,
        "Same automatically selected 8-second excerpt in both panels. "
        "The titles report full-record counts; markers show only this excerpt.",
        ha="center",
        fontsize=9.5,
    )
    return figure


def write_readme(records: list[RecordAudit]) -> Path:
    path = OUTPUT_DIR / "README.txt"
    lines = [
        "NeuroKit versus Pan-Tompkins hard-gate plot",
        "==============================================",
        "",
        f"NeuroKit2 version: {neurokit2.__version__}",
        f"Records: {', '.join(RECORDS)}",
        "Expert comparison: WFDB processing.compare_annotations, +/-75 ms",
        "Gate: retain NeuroKit peak when 0 <= t_PT - t_NK <= 100 ms",
        "Artifact correction: disabled",
        "",
        "Selected excerpts:",
    ]
    for record in records:
        lines.append(
            f"  {record.name}: {record.excerpt_start_s:.3f} to "
            f"{record.excerpt_start_s + EXCERPT_DURATION_S:.3f} s"
        )
    lines.extend(
        [
            "",
            "Interpretation boundary:",
            "The hard gate is an ablation. Pan-Tompkins support is not ground truth,",
            "detector disagreement is not an artifact label, and improved PPV may be",
            "purchased by a clinically unacceptable loss of sensitivity.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PDF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = [load_record(name) for name in RECORDS]
    metrics_path = write_metrics(records)
    readme_path = write_readme(records)

    summary = make_summary_figure(records)
    examples = make_examples_figure(records)
    summary_png = OUTPUT_DIR / "five_record_metric_comparison.png"
    examples_png = OUTPUT_DIR / "five_record_detection_examples.png"
    pdf_path = PDF_OUTPUT_DIR / "five_record_neurokit_vs_pan_gate.pdf"
    summary.savefig(summary_png, dpi=220, bbox_inches="tight")
    examples.savefig(examples_png, dpi=220, bbox_inches="tight")
    record_figure_paths: list[Path] = []
    for record in records:
        record_figure = make_record_figure(record)
        record_path = OUTPUT_DIR / f"record_{record.name}_before_after.png"
        record_figure.savefig(record_path, dpi=220, bbox_inches="tight")
        plt.close(record_figure)
        record_figure_paths.append(record_path)
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(summary, bbox_inches="tight")
        pdf.savefig(examples, bbox_inches="tight")
    plt.close(summary)
    plt.close(examples)

    for record in records:
        print(
            f"{record.name}: baseline F1={record.baseline.f1:.5f}; "
            f"gate F1={record.gated.f1:.5f}; excerpt={record.excerpt_start_s:.1f}s"
        )
    for path in [
        summary_png,
        examples_png,
        *record_figure_paths,
        pdf_path,
        metrics_path,
        readme_path,
    ]:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
