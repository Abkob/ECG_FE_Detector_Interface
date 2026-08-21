"""Stratify expert-anchor PRSA/BPRSA windows by MIT-BIH beat labels.

This is a confounding audit.  It asks whether feature/numerical behavior occurs
in windows containing non-normal annotated beats.  It does not call those
beats artifact and does not measure seizure discrimination.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecg_cascade.wfdb_io import load_wfdb_beat_annotations  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=Path,
        default=(
            PROJECT
            / "outputs"
            / "prsa_bprsa_context_audit_20260819"
            / "mit_expert_anchor_features.csv.gz"
        ),
    )
    parser.add_argument(
        "--mitdb",
        type=Path,
        default=(
            PROJECT.parent
            / "Datasets"
            / "mit-bih-arrhythmia-database-1.0"
            / "mit-bih-arrhythmia-database-1.0.0"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "outputs" / "prsa_bprsa_arrhythmia_context_20260819",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    features = pd.read_csv(args.features.expanduser().resolve())
    mitdb = args.mitdb.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []

    for record_id, record in features.groupby("record_id", sort=True):
        record_id = str(record_id)
        base = mitdb / record_id
        peaks, symbols = load_wfdb_beat_annotations(base)
        fs = float(_sampling_rate(base))
        peak_times = peaks.astype(float) / fs
        record = record.sort_values("end_time_s", kind="stable").reset_index(drop=True)
        # Row zero is the first completed 80-RR window, ending at expert beat
        # index 80.  Validate that convention against timestamps after CSV
        # round-tripping instead of using floating-point search insertion.
        end_indices = np.arange(80, 80 + record.shape[0], dtype=int)
        aligned = (
            end_indices[-1] < peak_times.size
            and np.allclose(
                peak_times[end_indices],
                record["end_time_s"].to_numpy(float),
                rtol=0.0,
                atol=1e-9,
            )
        )
        if not aligned:
            raise ValueError(f"{record_id}: feature rows do not align to expert beats")
        abnormal_counts = np.empty(record.shape[0], dtype=int)
        endpoint_symbols: list[str] = []
        for row_index, end_index in enumerate(end_indices):
            window_symbols = symbols[end_index - 80 : end_index + 1]
            abnormal_counts[row_index] = int(np.sum(window_symbols != "N"))
            endpoint_symbols.append(str(symbols[end_index]))
        record["window_expert_beat_count"] = 81
        record["window_non_normal_beat_count"] = abnormal_counts
        record["window_non_normal_beat_fraction"] = abnormal_counts / 81.0
        record["contains_non_normal_beat"] = abnormal_counts > 0
        record["endpoint_beat_symbol"] = endpoint_symbols
        frames.append(record)

    windows = pd.concat(frames, ignore_index=True)
    windows.to_csv(
        output / "expert_anchor_windows_with_arrhythmia_context.csv.gz",
        index=False,
        compression="gzip",
    )
    strata = (
        windows.assign(
            rhythm_stratum=np.where(
                windows["contains_non_normal_beat"],
                "contains_non_normal_beat",
                "all_N_beats",
            )
        )
        .groupby("rhythm_stratum", as_index=False)
        .agg(
            window_count=("record_id", "size"),
            record_count=("record_id", "nunique"),
            numerical_quality_pass_count=("numerical_quality_pass", "sum"),
            numerical_quality_pass_fraction=("numerical_quality_pass", "mean"),
            median_non_normal_beat_fraction=("window_non_normal_beat_fraction", "median"),
            median_sdnn80_ms=("sdnn80_ms", "median"),
            median_prsa_s_rr_ms_per_sample=("prsa_s_rr_ms_per_sample", "median"),
            median_bprsa_s_r_ms_per_sample=("bprsa_s_r_ms_per_sample", "median"),
        )
    )
    strata.to_csv(output / "rhythm_stratum_summary.csv", index=False)
    failures = ~windows["numerical_quality_pass"].to_numpy(bool)
    abnormal = windows["contains_non_normal_beat"].to_numpy(bool)
    summary = {
        "record_count": int(windows["record_id"].nunique()),
        "window_count": int(windows.shape[0]),
        "windows_containing_non_normal_beats": int(abnormal.sum()),
        "fraction_containing_non_normal_beats": float(abnormal.mean()),
        "numerical_quality_failure_count": int(failures.sum()),
        "quality_failures_containing_non_normal_beats": int(np.sum(failures & abnormal)),
        "quality_failure_fraction_with_non_normal_beats": (
            float(np.mean(abnormal[failures])) if failures.any() else np.nan
        ),
        "all_normal_window_quality_pass_fraction": float(
            windows.loc[~abnormal, "numerical_quality_pass"].mean()
        ),
        "non_normal_window_quality_pass_fraction": float(
            windows.loc[abnormal, "numerical_quality_pass"].mean()
        ),
        "interpretation": (
            "non-normal beats are a physiological/arrhythmia context and potential "
            "seizure-model confound, not automatically artifact"
        ),
        "seizure_accuracy_measured": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "SUMMARY.md").write_text(
        "\n".join(
            [
                "# PRSA/BPRSA arrhythmia-context audit",
                "",
                f"- Expert-anchor windows: {summary['window_count']}",
                f"- Windows containing at least one non-N beat: {summary['windows_containing_non_normal_beats']} ({100 * summary['fraction_containing_non_normal_beats']:.2f}%)",
                f"- Cubic-RR numerical failures: {summary['numerical_quality_failure_count']}",
                f"- Failures whose 81-beat context contained a non-N beat: {summary['quality_failures_containing_non_normal_beats']} ({100 * summary['quality_failure_fraction_with_non_normal_beats']:.2f}%)",
                f"- Quality pass for all-N versus non-normal windows: {100 * summary['all_normal_window_quality_pass_fraction']:.3f}% / {100 * summary['non_normal_window_quality_pass_fraction']:.3f}%",
                "- Non-normal beats are retained as physiological/confounding context, not automatically rejected as artifact.",
                "- This does not measure seizure accuracy.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


def _sampling_rate(base: Path) -> float:
    import wfdb

    return float(wfdb.rdheader(str(base)).fs)


if __name__ == "__main__":
    raise SystemExit(main())
