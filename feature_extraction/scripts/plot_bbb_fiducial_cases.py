"""Plot real LBBB/RBBB timing cases for the RR-HRV report."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wfdb

from ecg_cascade.fiducials import build_fiducial_candidates
from ecg_cascade.peaks import detect_unsw
from ecg_cascade.reliability import collapse_close_detections
from ecg_cascade.validation import match_peaks_to_reference
from ecg_cascade.wfdb_io import load_wfdb_beat_annotations, load_wfdb_segment


ROOT = Path(__file__).resolve().parents[2]
DATABASE = (
    ROOT
    / "Datasets"
    / "mit-bih-arrhythmia-database-1.0"
    / "mit-bih-arrhythmia-database-1.0.0"
)
OUTPUT = ROOT / "output" / "rr_hrv_architecture" / "fiducial_ablation_v1"

CASES = [
    ("207", "L", np.array([5.56, 55.56, 58.33]), "LBBB: positive-lobe failure"),
    ("109", "L", np.array([11.11, 2.78, 2.78]), "LBBB: refinement succeeds"),
    ("118", "R", np.array([22.22, 5.56, 5.56]), "RBBB: refinement succeeds"),
]

METHOD_STYLES = {
    "UNSW initial": ("#2166ac", "o"),
    "PhysioZoo +": ("#d17c00", "^"),
    "R-DECO + component": ("#7b3294", "s"),
}


def candidates_for_record(record: str):
    path = DATABASE / record
    segment = load_wfdb_segment(path, channel=0)
    reference, symbols = load_wfdb_beat_annotations(path)
    unsw = detect_unsw(segment.samples, segment.sampling_rate_hz, orientation="original")
    primary = collapse_close_detections(
        unsw.peak_samples,
        unsw.analysis_signal,
        sampling_rate_hz=segment.sampling_rate_hz,
        exclusion_ms=150.0,
    )
    adjusted = build_fiducial_candidates(
        primary,
        unsw.analysis_signal,
        sampling_rate_hz=segment.sampling_rate_hz,
    )
    methods = {
        "UNSW initial": primary,
        "PhysioZoo +": adjusted.physiozoo_positive,
        "R-DECO + component": adjusted.rdeco_positive_backward,
    }
    return segment, reference, symbols, methods


def choose_case(record: str, symbol: str, target_absolute_errors: np.ndarray):
    segment, reference, symbols, methods = candidates_for_record(record)
    by_method: dict[str, dict[int, int]] = {}
    for name, values in methods.items():
        matches = match_peaks_to_reference(
            values,
            reference,
            sampling_rate_hz=segment.sampling_rate_hz,
            tolerance_ms=75.0,
        )
        by_method[name] = {
            int(reference_index): int(values[detected_index])
            for detected_index, reference_index in zip(
                matches.detected_indices,
                matches.reference_indices,
                strict=True,
            )
        }
    common = set.intersection(*(set(mapping) for mapping in by_method.values()))
    eligible = [
        index
        for index in sorted(common)
        if symbols[index] == symbol
        and reference[index] > int(0.3 * segment.sampling_rate_hz)
        and reference[index] < segment.samples.size - int(0.3 * segment.sampling_rate_hz)
    ]
    if not eligible:
        raise RuntimeError(f"No common {symbol} beat in record {record}")
    scores = []
    for reference_index in eligible:
        detected = np.array(
            [by_method[name][reference_index] for name in METHOD_STYLES], dtype=int
        )
        errors = np.abs(detected - reference[reference_index]) * 1000.0 / segment.sampling_rate_hz
        scores.append(float(np.sum(np.abs(errors - target_absolute_errors))))
    chosen = eligible[int(np.argmin(scores))]
    selected = {name: mapping[chosen] for name, mapping in by_method.items()}
    return segment, reference, symbols, methods, chosen, selected


def plot_timing_cases(selected_cases) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(11.5, 9.6), constrained_layout=True)
    for axis, case, selected_case in zip(axes, CASES, selected_cases, strict=True):
        record, symbol, _, description = case
        segment, reference, _, _, reference_index, selected = selected_case
        expert = int(reference[reference_index])
        fs = segment.sampling_rate_hz
        radius = int(np.rint(0.18 * fs))
        first = expert - radius
        last = expert + radius
        samples = np.arange(first, last + 1)
        time_ms = (samples - expert) * 1000.0 / fs
        axis.plot(time_ms, segment.samples[first : last + 1], color="#33495f", lw=1.3)
        axis.axvline(0, color="black", lw=1.2, ls="--", label="Expert R fiducial")
        unsw_time_ms = (selected["UNSW initial"] - expert) * 1000.0 / fs
        axis.axvspan(
            unsw_time_ms - 50.0,
            unsw_time_ms + 50.0,
            color="#f6d55c",
            alpha=0.13,
            label="PhysioZoo +/-50-ms search" if axis is axes[0] else None,
        )
        for name, (color, marker) in METHOD_STYLES.items():
            sample = selected[name]
            error_ms = (sample - expert) * 1000.0 / fs
            axis.scatter(
                [error_ms],
                [segment.samples[sample]],
                s=75,
                marker=marker,
                color=color,
                edgecolor="white",
                linewidth=0.7,
                zorder=5,
                label=f"{name}: {error_ms:+.1f} ms",
            )
        axis.set_title(f"Record {record}, expert {symbol} beat, channel 0 = MLII - {description}")
        axis.set_ylabel("ECG (mV)")
        axis.grid(alpha=0.2)
        axis.legend(loc="best", fontsize=8, ncol=2)
    axes[-1].set_xlabel("Time relative to expert R-wave annotation (ms)")
    figure.suptitle(
        "Same QRS event, different timestamp rules: a broad match tolerance can conceal lobe switching",
        fontsize=14,
    )
    figure.savefig(OUTPUT / "bbb_three_fiducial_cases.png", dpi=240)
    plt.close(figure)


def plot_lead_polarity(selected_cases) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(11.5, 9.2), constrained_layout=True)
    for row, (case, selected_case) in enumerate(zip(CASES, selected_cases, strict=True)):
        record, symbol, _, _ = case
        _, reference, _, _, reference_index, _ = selected_case
        expert = int(reference[reference_index])
        header = wfdb.rdheader(str(DATABASE / record))
        fs = float(header.fs)
        radius = int(np.rint(0.18 * fs))
        first = expert - radius
        last = expert + radius + 1
        samples = np.arange(first, last)
        time_ms = (samples - expert) * 1000.0 / fs
        data = wfdb.rdrecord(
            str(DATABASE / record),
            sampfrom=first,
            sampto=last,
            channels=[0, 1],
            physical=True,
        ).p_signal
        for column in range(2):
            axis = axes[row, column]
            axis.plot(time_ms, data[:, column], color="#33495f", lw=1.3)
            axis.axvline(0, color="black", lw=1.1, ls="--")
            axis.axhline(0, color="#888888", lw=0.6, alpha=0.5)
            audit_label = "audited" if column == 0 else "not used in this audit"
            axis.set_title(
                f"Record {record}, {symbol}: channel {column} {header.sig_name[column]} ({audit_label})"
            )
            axis.set_ylabel("ECG (mV)")
            axis.grid(alpha=0.2)
    for axis in axes[-1, :]:
        axis.set_xlabel("Time relative to the channel-0 expert annotation (ms)")
    figure.suptitle(
        "The same heartbeat can have different polarity and lobes in MLII and V1",
        fontsize=14,
    )
    figure.savefig(OUTPUT / "bbb_same_beats_mlII_v1.png", dpi=240)
    plt.close(figure)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selected_cases = [choose_case(record, symbol, targets) for record, symbol, targets, _ in CASES]
    plot_timing_cases(selected_cases)
    plot_lead_polarity(selected_cases)
    for case, selected_case in zip(CASES, selected_cases, strict=True):
        record, symbol, _, _ = case
        segment, reference, _, _, reference_index, selected = selected_case
        expert = int(reference[reference_index])
        errors = {
            name: (sample - expert) * 1000.0 / segment.sampling_rate_hz
            for name, sample in selected.items()
        }
        print(record, symbol, "expert sample", expert, errors)


if __name__ == "__main__":
    main()
