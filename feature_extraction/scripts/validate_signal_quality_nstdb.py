"""Validate signal-quality v0 behavior on the local MIT-BIH NSTDB copy.

This is a controlled degradation diagnostic, not artifact-classifier
validation.  NSTDB supplies known electrode-motion SNR levels and retained
beat annotations, but it does not provide the window-level labels needed to
train or calibrate the final BUT-QDB quality classifier.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ecg_cascade.peaks import detect_neurokit_gradient, detect_unsw
from ecg_cascade.signal_quality import extract_signal_quality_window
from ecg_cascade.wfdb_io import load_wfdb_beat_annotations, load_wfdb_segment


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
DEFAULT_NSTDB = (
    WORKSPACE
    / "Datasets"
    / "mit-bih-noise-stress-test-database-1.0"
    / "mit-bih-noise-stress-test-database-1.0.0"
)
DEFAULT_MITDB = (
    WORKSPACE
    / "Datasets"
    / "mit-bih-arrhythmia-database-1.0"
    / "mit-bih-arrhythmia-database-1.0.0"
)
NOISY_RECORDS = (
    ("e24", 24),
    ("e18", 18),
    ("e12", 12),
    ("e06", 6),
    ("e00", 0),
    ("e_6", -6),
)


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        rendered = []
        for value in row:
            if isinstance(value, float):
                rendered.append("" if np.isnan(value) else f"{value:.6g}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nstdb-root", type=Path, default=DEFAULT_NSTDB)
    parser.add_argument("--mitdb-root", type=Path, default=DEFAULT_MITDB)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "outputs" / "signal_quality_nstdb_v0",
    )
    parser.add_argument("--start-s", type=float, default=360.0)
    parser.add_argument("--duration-s", type=float, default=10.0)
    return parser.parse_args()


def _window_annotations(record: Path, start_s: float, duration_s: float, fs: float):
    first = int(np.rint(start_s * fs))
    last = first + int(np.rint(duration_s * fs))
    samples, _ = load_wfdb_beat_annotations(
        record,
        start_sample=first,
        end_sample=last,
    )
    return samples


def _measure_record(
    record: Path,
    *,
    source: str,
    condition: str,
    snr_db: float,
    start_s: float,
    duration_s: float,
) -> dict[str, object]:
    segment = load_wfdb_segment(
        record,
        channel=0,
        start_s=start_s,
        duration_s=duration_s,
    )
    fs = segment.sampling_rate_hz
    reference = _window_annotations(record, start_s, duration_s, fs)
    detector_error = ""
    try:
        primary = detect_unsw(segment.samples, fs, orientation="original").peak_samples
        secondary = detect_neurokit_gradient(
            segment.samples,
            fs,
            orientation="original",
            minimum_delay_ms=250.0,
            minimum_delay_inclusive=True,
        ).peak_samples
    except Exception as exc:
        primary = reference
        secondary = np.array([], dtype=np.int64)
        detector_error = f"{type(exc).__name__}: {exc}"
    result = extract_signal_quality_window(
        segment.samples,
        fs,
        uv_per_input_unit=1000.0,
        primary_peak_samples=primary,
        secondary_peak_samples=secondary,
    )
    row: dict[str, object] = {
        "source_record": source,
        "condition": condition,
        "snr_db": snr_db,
        "record": record.name,
        "start_s": start_s,
        "duration_s": segment.duration_s,
        "sampling_rate_hz": fs,
        "reference_beat_count": int(reference.size),
        "primary_beat_count": int(primary.size),
        "secondary_beat_count": int(secondary.size),
        "detector_error": detector_error,
        "available_feature_count": int(sum(result.available.values())),
        "total_feature_count": int(len(result.values)),
    }
    row.update(result.values)
    for feature, available in result.available.items():
        row[f"{feature}__available"] = available
        row[f"{feature}__reason"] = result.reasons[feature]
    return row


def main() -> int:
    args = parse_args()
    nstdb = args.nstdb_root.expanduser().resolve()
    mitdb = args.mitdb_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not nstdb.is_dir():
        raise FileNotFoundError(nstdb)
    if not mitdb.is_dir():
        raise FileNotFoundError(mitdb)
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for source in ("118", "119"):
        rows.append(
            _measure_record(
                mitdb / source,
                source=source,
                condition="clean_source",
                snr_db=np.nan,
                start_s=args.start_s,
                duration_s=args.duration_s,
            )
        )
        for suffix, snr_db in NOISY_RECORDS:
            rows.append(
                _measure_record(
                    nstdb / f"{source}{suffix}",
                    source=source,
                    condition="electrode_motion",
                    snr_db=float(snr_db),
                    start_s=args.start_s,
                    duration_s=args.duration_s,
                )
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "window_features.csv", index=False)

    candidate_features = [
        "rms_uv",
        "hf_rms_uv",
        "bassqi_clifford",
        "psqi_clifford",
        "pearson_kurtosis",
        "bsqi_li2008_jaccard_150ms",
        "qsqi_zhao2018_dice_75ms",
        "rr_support_ho2024_150ms",
        "template_corr_orphanidou",
        "menon_fourier_score",
    ]
    noisy = frame[frame["condition"] == "electrode_motion"]
    correlations = []
    for feature in candidate_features:
        valid = noisy[["snr_db", feature]].dropna()
        correlations.append(
            {
                "feature": feature,
                "evaluable_windows": int(valid.shape[0]),
                "spearman_with_snr_db": (
                    float(valid[feature].corr(valid["snr_db"], method="spearman"))
                    if valid.shape[0] >= 3
                    else np.nan
                ),
            }
        )
    correlation_frame = pd.DataFrame(correlations)
    correlation_frame.to_csv(output / "snr_rank_diagnostics.csv", index=False)

    report = [
        "# Signal-quality v0 NSTDB diagnostic",
        "",
        f"- Window: {args.start_s:g}--{args.start_s + args.duration_s:g} s; this lies in NSTDB's first post-learning noisy interval.",
        "- Records: clean MIT-BIH 118/119 and six electrode-motion SNR copies per record.",
        "- Purpose: verify calculability, units, detector behavior, and directional response to controlled degradation.",
        "- This does not validate an artifact classifier or a universal threshold. BUT-QDB labels remain required for that stage.",
        "- Missing ADC rails, QRS onsets, acquisition mains metadata, and morphology groups remain explicitly unavailable.",
        "",
        "## SNR rank diagnostics",
        "",
        _markdown_table(correlation_frame),
        "",
        "A positive correlation means the value tends to rise as signal-to-noise ratio improves; it is not accuracy or causality.",
    ]
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(output)
    print(correlation_frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
