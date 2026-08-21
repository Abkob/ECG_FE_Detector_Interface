"""Benchmark detector-anchored morphology against expert-timestamp features.

The output measures feature-extraction fidelity on MIT--BIH.  It is not a
seizure benchmark because MIT--BIH does not contain seizure labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecg_cascade.hrv_fidelity import (  # noqa: E402
    lin_concordance_correlation,
    symmetric_absolute_percentage_error,
)
from ecg_cascade.morphology_validation import (  # noqa: E402
    audit_morphology_timestamp_fidelity,
)
from ecg_cascade.peaks import (  # noqa: E402
    detect_neurokit_gradient,
    detect_unsw,
)
from ecg_cascade.wfdb_io import load_wfdb_beat_annotations  # noqa: E402
from ecg_cascade.zhai import detect_zhai_template  # noqa: E402


def _default_dataset() -> Path:
    return (
        PROJECT.parent
        / "Datasets"
        / "mit-bih-arrhythmia-database-1.0"
        / "mit-bih-arrhythmia-database-1.0.0"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=_default_dataset())
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "outputs" / "morphology_fidelity_all48_v1",
    )
    parser.add_argument(
        "--records",
        nargs="*",
        default=None,
        help="Optional record stems; default is every numeric .hea record.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = (
        [str(value) for value in args.records]
        if args.records
        else sorted(path.stem for path in dataset.glob("*.hea") if path.stem.isdigit())
    )
    if not records:
        raise FileNotFoundError(f"No numeric WFDB headers found in {dataset}")

    record_rows: list[dict[str, object]] = []
    window_frames: list[pd.DataFrame] = []
    timing_frames: list[pd.DataFrame] = []
    for record_id in records:
        base = dataset / record_id
        header = wfdb.rdheader(str(base))
        record = wfdb.rdrecord(str(base), channels=[0], physical=True)
        if record.p_signal is None:
            raise ValueError(f"{record_id}: no physical ECG returned")
        ecg = np.asarray(record.p_signal[:, 0], dtype=np.float64)
        fs = float(header.fs)
        reference_samples, reference_symbols = load_wfdb_beat_annotations(base)
        tracks = {
            "unsw": detect_unsw(ecg, fs, orientation="original").peak_samples,
            "neurokit": detect_neurokit_gradient(
                ecg,
                fs,
                orientation="original",
                minimum_delay_ms=300.0,
                minimum_delay_inclusive=False,
            ).peak_samples,
            "zhai": detect_zhai_template(ecg, fs, orientation="original").peak_samples,
        }
        for track, candidates in tracks.items():
            audit = audit_morphology_timestamp_fidelity(
                ecg,
                reference_samples,
                reference_symbols,
                candidates,
                fs,
                candidate_track=track,
            )
            record_rows.append(
                {
                    "record_id": record_id,
                    "channel": str(header.sig_name[0]),
                    "sampling_rate_hz": fs,
                    **audit.summary,
                }
            )
            windows = audit.windows.copy()
            windows.insert(0, "candidate_track", track)
            windows.insert(0, "record_id", record_id)
            window_frames.append(windows)
            timing_frames.append(
                pd.DataFrame(
                    {
                        "record_id": record_id,
                        "candidate_track": track,
                        "candidate_sample": audit.agreement.matched_primary_samples,
                        "reference_sample": audit.agreement.matched_comparator_samples,
                        "timing_error_ms": 1000.0
                        * (
                            audit.agreement.matched_primary_samples
                            - audit.agreement.matched_comparator_samples
                        )
                        / fs,
                    }
                )
            )
        print(f"completed {record_id}", flush=True)

    record_table = pd.DataFrame(record_rows)
    windows = pd.concat(window_frames, ignore_index=True)
    timing = pd.concat(timing_frames, ignore_index=True)
    record_table.to_csv(output / "record_summary.csv", index=False)
    windows.to_csv(output / "window_fidelity.csv", index=False)
    timing.to_csv(output / "matched_beat_timing.csv", index=False)

    pooled_rows: list[dict[str, object]] = []
    for track in ("unsw", "neurokit", "zhai"):
        records_track = record_table[record_table["candidate_track"] == track]
        window_track = windows[windows["candidate_track"] == track]
        timing_track = timing[timing["candidate_track"] == track]
        tp = int(records_track["matched_beat_count"].sum())
        fp = int(records_track["false_positive_count"].sum())
        fn = int(records_track["missed_reference_count"].sum())
        comparable = window_track["morphology_comparable"].astype(bool)
        row: dict[str, object] = {
            "candidate_track": track,
            "records": int(records_track.shape[0]),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "beat_sensitivity": _divide(tp, tp + fn),
            "beat_ppv": _divide(tp, tp + fp),
            "beat_f1": _divide(2 * tp, 2 * tp + fp + fn),
            "median_absolute_timing_error_ms": float(
                np.median(np.abs(timing_track["timing_error_ms"]))
            ),
            "five_beat_window_coverage": float(comparable.mean()),
            "median_raw_eigenvalue_relative_l1_error": float(
                window_track.loc[comparable, "raw_eigenvalue_relative_l1_error"].median()
            ),
            "median_exploratory_normalized_spectrum_l1_error": float(
                window_track.loc[
                    comparable, "exploratory_normalized_spectrum_l1_error"
                ].median()
            ),
            "seizure_accuracy_measured": False,
        }
        for component in range(1, 6):
            reference = window_track.loc[
                comparable, f"reference_lambda{component}"
            ].to_numpy(dtype=float)
            candidate = window_track.loc[
                comparable, f"candidate_lambda{component}"
            ].to_numpy(dtype=float)
            row[f"lambda{component}_ccc"] = lin_concordance_correlation(
                reference, candidate
            )
            row[f"lambda{component}_smape"] = symmetric_absolute_percentage_error(
                reference, candidate
            )
        pooled_rows.append(row)
    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(output / "pooled_summary.csv", index=False)
    _plot_summary(pooled, output / "morphology_fidelity_summary.png")
    (output / "audit_scope.json").write_text(
        json.dumps(
            {
                "dataset": str(dataset),
                "records": records,
                "channel": 0,
                "expert_annotation": "atr beat symbols",
                "matching_tolerance_ms": 75.0,
                "morphology": "Varon 2015 five-beat uncentered Q @ Q.T, 120 ms",
                "claim": "feature fidelity to expert timestamps",
                "not_measured": [
                    "seizure sensitivity",
                    "seizure PPV",
                    "artifact classification",
                    "beat-type classification",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(pooled.to_string(index=False), flush=True)
    return 0


def _plot_summary(pooled: pd.DataFrame, path: Path) -> None:
    labels = [str(value) for value in pooled["candidate_track"]]
    x = np.arange(len(labels))
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].bar(x - 0.18, pooled["beat_f1"], 0.36, label="Beat F1")
    axes[0, 0].bar(
        x + 0.18,
        pooled["five_beat_window_coverage"],
        0.36,
        label="Five-beat coverage",
    )
    axes[0, 0].set_ylim(0, 1.02)
    axes[0, 0].legend()
    axes[0, 0].set_title("Event detection and strict morphology availability")

    axes[0, 1].bar(x, pooled["median_absolute_timing_error_ms"])
    axes[0, 1].set_ylabel("ms")
    axes[0, 1].set_title("Median absolute candidate-to-expert timing error")

    width = 0.15
    for component in range(1, 6):
        axes[1, 0].bar(
            x + (component - 3) * width,
            pooled[f"lambda{component}_ccc"],
            width,
            label=f"λ{component}",
        )
    axes[1, 0].axhline(1.0, color="#555555", linewidth=0.8)
    axes[1, 0].set_title("Raw eigenvalue concordance on comparable windows")
    axes[1, 0].legend(ncol=5, fontsize=8)

    axes[1, 1].bar(
        x,
        pooled["median_exploratory_normalized_spectrum_l1_error"],
    )
    axes[1, 1].set_title("Median normalized eigen-spectrum L1 error")
    axes[1, 1].set_ylabel("lower is better")
    for axis in axes.flat:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        "Morphology feature fidelity to expert timestamps — not seizure accuracy"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _divide(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
