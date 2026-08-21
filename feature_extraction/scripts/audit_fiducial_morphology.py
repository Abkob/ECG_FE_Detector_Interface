"""Cross-check fiducial adjustments by MIT-BIH beat morphology.

This audit supplements the pooled fiducial benchmark.  It evaluates every
method on every first-channel MIT-BIH record, reports expert-symbol-specific
sensitivity and timing, and compares the project's matcher with WFDB's
nearest-annotation comparator.  MIT-BIH does not provide beatwise QRS onset
and offset annotations, so the L/R symbols are morphology strata rather than
direct QRS-duration measurements.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from wfdb.processing import compare_annotations

from ecg_cascade.fiducials import build_fiducial_candidates
from ecg_cascade.peaks import detect_unsw
from ecg_cascade.reliability import collapse_close_detections
from ecg_cascade.validation import match_peaks_to_reference, peak_metrics
from ecg_cascade.wfdb_io import load_wfdb_beat_annotations, load_wfdb_segment


ROOT = Path(__file__).resolve().parents[2]
DATABASE = (
    ROOT
    / "Datasets"
    / "mit-bih-arrhythmia-database-1.0"
    / "mit-bih-arrhythmia-database-1.0.0"
)
OUTPUT = ROOT / "output" / "rr_hrv_architecture" / "fiducial_ablation_v1"


def records() -> list[str]:
    return [
        line.strip()
        for line in (DATABASE / "RECORDS").read_text().splitlines()
        if line.strip()
    ]


def methods_for_record(record_name: str) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, float]:
    path = DATABASE / record_name
    segment = load_wfdb_segment(path, channel=0)
    reference, symbols = load_wfdb_beat_annotations(path)
    unsw = detect_unsw(segment.samples, segment.sampling_rate_hz, orientation="original")
    primary = collapse_close_detections(
        unsw.peak_samples,
        unsw.analysis_signal,
        sampling_rate_hz=segment.sampling_rate_hz,
        exclusion_ms=150.0,
    )
    candidates = build_fiducial_candidates(
        primary,
        unsw.analysis_signal,
        sampling_rate_hz=segment.sampling_rate_hz,
    )
    return (
        {
            "unsw_initial": primary,
            "ho_positive": candidates.ho_positive,
            "physiozoo_positive": candidates.physiozoo_positive,
            "physiozoo_negative": candidates.physiozoo_negative,
            "physiozoo_rqrs_adapted": candidates.physiozoo_rqrs_adapted,
            "rdeco_positive_backward": candidates.rdeco_positive_backward,
            "rdeco_negative_backward": candidates.rdeco_negative_backward,
        },
        reference,
        symbols,
        segment.sampling_rate_hz,
    )


def run() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    symbol_accumulator: dict[tuple[str, str], dict[str, list[float] | int]] = {}
    matcher_accumulator: dict[str, dict[str, int]] = {}
    record_rows: list[dict[str, object]] = []
    record_symbol_rows: list[dict[str, object]] = []

    for record_name in records():
        methods, reference, symbols, fs = methods_for_record(record_name)
        tolerance_samples = int(np.rint(75.0 * fs / 1000.0))
        for method, detected in methods.items():
            project_matches = match_peaks_to_reference(
                detected,
                reference,
                sampling_rate_hz=fs,
                tolerance_ms=75.0,
            )
            project_metrics = peak_metrics(
                detected,
                reference,
                sampling_rate_hz=fs,
                tolerance_ms=75.0,
            )
            # WFDB uses a strict '< window_width' comparison.  Adding one
            # sample makes it equivalent to this project's inclusive +/-27
            # sample rule at 360 Hz.
            wfdb_matches = compare_annotations(
                reference,
                detected,
                window_width=tolerance_samples + 1,
            )
            counts = matcher_accumulator.setdefault(
                method,
                {
                    "project_tp": 0,
                    "wfdb_tp": 0,
                    "project_fp": 0,
                    "wfdb_fp": 0,
                    "common_pairs": 0,
                    "project_only_pairs": 0,
                    "wfdb_only_pairs": 0,
                },
            )
            counts["project_tp"] += int(project_metrics["tp"])
            counts["project_fp"] += int(project_metrics["fp"])
            counts["wfdb_tp"] += int(wfdb_matches.tp)
            counts["wfdb_fp"] += int(wfdb_matches.fp)
            project_pairs = set(
                zip(
                    project_matches.reference_indices.tolist(),
                    project_matches.detected_indices.tolist(),
                    strict=True,
                )
            )
            wfdb_pairs = set(
                zip(
                    wfdb_matches.matched_ref_inds.tolist(),
                    wfdb_matches.matched_test_inds.tolist(),
                    strict=True,
                )
            )
            counts["common_pairs"] += len(project_pairs & wfdb_pairs)
            counts["project_only_pairs"] += len(project_pairs - wfdb_pairs)
            counts["wfdb_only_pairs"] += len(wfdb_pairs - project_pairs)

            matched_by_reference = {
                int(reference_index): int(detected_index)
                for detected_index, reference_index in zip(
                    project_matches.detected_indices,
                    project_matches.reference_indices,
                    strict=True,
                )
            }
            for symbol in np.unique(symbols):
                reference_indices = np.flatnonzero(symbols == symbol)
                key = (method, str(symbol))
                entry = symbol_accumulator.setdefault(
                    key,
                    {"reference": 0, "matched": 0, "timing_ms": []},
                )
                entry["reference"] = int(entry["reference"]) + int(reference_indices.size)
                record_symbol_timing: list[float] = []
                for reference_index in reference_indices:
                    detected_index = matched_by_reference.get(int(reference_index))
                    if detected_index is None:
                        continue
                    entry["matched"] = int(entry["matched"]) + 1
                    error_ms = (
                        int(detected[detected_index]) - int(reference[reference_index])
                    ) * 1000.0 / fs
                    timing = entry["timing_ms"]
                    assert isinstance(timing, list)
                    timing.append(float(error_ms))
                    record_symbol_timing.append(float(error_ms))
                record_symbol_array = np.asarray(record_symbol_timing, dtype=float)
                record_symbol_rows.append(
                    {
                        "record": record_name,
                        "method": method,
                        "symbol": str(symbol),
                        "reference_count": int(reference_indices.size),
                        "matched_count": int(record_symbol_array.size),
                        "sensitivity": (
                            record_symbol_array.size / reference_indices.size
                            if reference_indices.size
                            else np.nan
                        ),
                        "median_absolute_timing_ms": (
                            float(np.median(np.abs(record_symbol_array)))
                            if record_symbol_array.size
                            else np.nan
                        ),
                        "p95_absolute_timing_ms": (
                            float(np.percentile(np.abs(record_symbol_array), 95))
                            if record_symbol_array.size
                            else np.nan
                        ),
                        "absolute_timing_over_20_ms": int(
                            (np.abs(record_symbol_array) > 20.0).sum()
                        ),
                        "absolute_timing_over_40_ms": int(
                            (np.abs(record_symbol_array) > 40.0).sum()
                        ),
                        "absolute_timing_over_50_ms": int(
                            (np.abs(record_symbol_array) > 50.0).sum()
                        ),
                    }
                )

            record_rows.append(
                {
                    "record": record_name,
                    "method": method,
                    "tp": int(project_metrics["tp"]),
                    "fp": int(project_metrics["fp"]),
                    "fn": int(project_metrics["fn"]),
                    "f1": float(project_metrics["f1"]),
                    "median_absolute_timing_ms": float(
                        project_metrics["median_absolute_timing_ms"]
                    ),
                    "p95_absolute_timing_ms": float(
                        project_metrics["p95_absolute_timing_ms"]
                    ),
                }
            )

    symbol_rows: list[dict[str, object]] = []
    for (method, symbol), entry in sorted(symbol_accumulator.items()):
        reference_count = int(entry["reference"])
        matched_count = int(entry["matched"])
        timing = np.asarray(entry["timing_ms"], dtype=float)
        symbol_rows.append(
            {
                "method": method,
                "symbol": symbol,
                "reference_count": reference_count,
                "matched_count": matched_count,
                "sensitivity": matched_count / reference_count,
                "median_signed_timing_ms": float(np.median(timing)) if timing.size else np.nan,
                "median_absolute_timing_ms": float(np.median(np.abs(timing))) if timing.size else np.nan,
                "p95_absolute_timing_ms": float(np.percentile(np.abs(timing), 95)) if timing.size else np.nan,
                "absolute_timing_over_20_ms": int((np.abs(timing) > 20.0).sum()),
                "absolute_timing_over_40_ms": int((np.abs(timing) > 40.0).sum()),
                "absolute_timing_over_50_ms": int((np.abs(timing) > 50.0).sum()),
            }
        )

    matcher_rows = [
        {"method": method, **counts, "tp_difference": counts["wfdb_tp"] - counts["project_tp"]}
        for method, counts in sorted(matcher_accumulator.items())
    ]
    symbol_frame = pd.DataFrame(symbol_rows)
    record_frame = pd.DataFrame(record_rows)
    record_symbol_frame = pd.DataFrame(record_symbol_rows)
    matcher_frame = pd.DataFrame(matcher_rows)

    symbol_frame.to_csv(OUTPUT / "fiducial_metrics_by_symbol.csv", index=False)
    record_frame.to_csv(OUTPUT / "fiducial_record_crosscheck.csv", index=False)
    record_symbol_frame.to_csv(
        OUTPUT / "fiducial_metrics_by_record_and_symbol.csv", index=False
    )
    matcher_frame.to_csv(OUTPUT / "fiducial_matcher_crosscheck.csv", index=False)

    pivot = record_frame.pivot(index="record", columns="method", values="f1")
    initial = pivot["unsw_initial"]
    comparisons = {}
    for method in pivot.columns:
        delta = pivot[method] - initial
        comparisons[method] = {
            "records_better_f1_than_unsw": int((delta > 1e-12).sum()),
            "records_equal_f1_to_unsw": int((np.abs(delta) <= 1e-12).sum()),
            "records_worse_f1_than_unsw": int((delta < -1e-12).sum()),
        }
    summary = {
        "records": len(records()),
        "channel": "first channel only",
        "qrs_duration_available": False,
        "morphology_proxy": "MIT-BIH expert symbols; L=LBBB beat, R=RBBB beat",
        "matcher_crosscheck": matcher_rows,
        "record_f1_comparison": comparisons,
    }
    (OUTPUT / "fiducial_morphology_audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(symbol_frame[symbol_frame["symbol"].isin(["L", "R"])].to_string(index=False))
    print("\nMatcher cross-check")
    print(matcher_frame.to_string(index=False))
    print("\nRecord-level F1 directions versus UNSW initial")
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    run()
