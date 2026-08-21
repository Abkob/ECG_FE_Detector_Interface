"""Benchmark a patient/lead-specific asymmetric Varon capture window.

The first ``k`` complete manual QRS triplets in each record calibrate separate
onset-to-R and R-to-offset extents.  Later annotated beats are the only test
beats.  The proposed p95 window, its minimum-five-beat fallback, the earlier
symmetric p95 correction, and a maximum-window sensitivity analysis are
compared with both the nominal 60+60-ms geometry and the exact sampled window
used by the current fixed Varon extractor.

This is a QRS-boundary/crop experiment.  It does not estimate automatic
boundaries, calculate seizure sensitivity, or establish that a larger crop
improves the downstream Varon Gram eigenvalues.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import wfdb

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecg_cascade.delineation_validation import (  # noqa: E402
    ManualDelineationRecord,
    associate_landmarks_with_anchors,
    fixed_qrs_capture_rows,
    load_ludb_delineation_record,
    load_qtdb_delineation_record,
)
from ecg_cascade.morphology import (  # noqa: E402
    calibrate_patient_asymmetric_varon_window,
    calibrate_patient_symmetric_varon_width,
)


FIXED_EXTRACTOR = "fixed_120ms_current_extractor"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ludb",
        type=Path,
        default=PROJECT.parent
        / "Datasets"
        / "lobachevsky-university-electrocardiography-database-1.0.1",
    )
    parser.add_argument(
        "--qtdb",
        type=Path,
        default=PROJECT.parent / "Datasets" / "qt-database-1.0.0",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "outputs" / "patient_asymmetric_varon_capture_v3",
    )
    parser.add_argument(
        "--calibration-beats",
        nargs="+",
        type=int,
        default=[3, 5, 10, 20],
    )
    parser.add_argument("--minimum-p95-calibration-beats", type=int, default=5)
    parser.add_argument("--ludb-lead", default="ii")
    parser.add_argument("--qtdb-lead", default="MLII")
    parser.add_argument("--ludb-records", nargs="*", default=None)
    parser.add_argument("--qtdb-records", nargs="*", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    calibration_sizes = sorted(set(int(value) for value in args.calibration_beats))
    if not calibration_sizes or calibration_sizes[0] < 1:
        raise ValueError("calibration beat counts must be positive")
    if args.minimum_p95_calibration_beats < 1:
        raise ValueError("minimum p95 calibration beats must be positive")

    ludb = args.ludb.expanduser().resolve()
    qtdb = args.qtdb.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    ludb_records = (
        [str(value) for value in args.ludb_records]
        if args.ludb_records
        else _read_records(ludb)
    )
    qtdb_records = (
        [str(value) for value in args.qtdb_records]
        if args.qtdb_records
        else _qtdb_records_for_channel0(qtdb, args.qtdb_lead)
    )
    loaders = [
        (
            "ludb",
            ludb_records,
            lambda record_id: load_ludb_delineation_record(
                ludb, record_id, lead=args.ludb_lead
            ),
        ),
        (
            "qtdb",
            qtdb_records,
            lambda record_id: load_qtdb_delineation_record(
                qtdb, record_id, required_channel0=args.qtdb_lead
            ),
        ),
    ]

    beat_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    for dataset, record_ids, loader in loaders:
        for record_number, record_id in enumerate(record_ids, start=1):
            try:
                record = loader(record_id)
                manual = pd.DataFrame(fixed_qrs_capture_rows(record))
            except Exception as exc:  # preserve record-level failures for audit
                skipped_rows.append(
                    {
                        "dataset": dataset,
                        "record_id": record_id,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(f"failed {dataset} {record_id}: {exc}", flush=True)
                continue
            if manual.empty:
                skipped_rows.append(
                    {
                        "dataset": dataset,
                        "record_id": record.record_id,
                        "reason": "no complete manual QRS onset/R/offset triplets",
                    }
                )
                continue

            manual = manual.sort_values("r_peak_sample", kind="stable")
            p_offsets = _landmark_by_anchor(record, "p_offset")
            t_onsets = _landmark_by_anchor(record, "t_onset")
            for calibration_count in calibration_sizes:
                if len(manual) <= calibration_count:
                    continue
                calibration = manual.iloc[:calibration_count]
                evaluation = manual.iloc[calibration_count:]
                windows = _calibrated_windows(
                    calibration,
                    record.sampling_rate_hz,
                    minimum_p95_calibration_beats=(
                        args.minimum_p95_calibration_beats
                    ),
                )
                for window in windows:
                    calibration_capture = _capture_metrics(
                        calibration,
                        float(window["realized_pre_r_ms"]),
                        float(window["realized_post_r_ms"]),
                    )
                    calibration_rows.append(
                        {
                            "dataset": dataset,
                            "record_id": record.record_id,
                            "lead_name": record.lead_name,
                            "sampling_rate_hz": record.sampling_rate_hz,
                            "calibration_beat_count": calibration_count,
                            "evaluation_beat_count": int(len(evaluation)),
                            **window,
                            "calibration_complete_capture_fraction": float(
                                calibration_capture["complete"].mean()
                            ),
                            "calibration_mean_missed_qrs_ms": float(
                                calibration_capture["missed_ms"].mean()
                            ),
                        }
                    )
                    for row in evaluation.itertuples(index=False):
                        beat_rows.append(
                            _beat_result(
                                dataset=dataset,
                                record=record,
                                calibration_count=calibration_count,
                                row=row,
                                window=window,
                                p_offset_sample=p_offsets.get(int(row.anchor_index)),
                                t_onset_sample=t_onsets.get(int(row.anchor_index)),
                            )
                        )
            print(
                f"completed {dataset} {record.record_id} "
                f"({record_number}/{len(record_ids)})",
                flush=True,
            )

    beats = pd.DataFrame(beat_rows)
    calibrations = pd.DataFrame(calibration_rows)
    skipped = pd.DataFrame(skipped_rows, columns=["dataset", "record_id", "reason"])
    summary = _summarize(beats, calibrations)
    beats.to_csv(output / "held_out_beat_windows.csv", index=False)
    calibrations.to_csv(output / "record_calibrations.csv", index=False)
    summary.to_csv(output / "capture_summary.csv", index=False)
    skipped.to_csv(output / "skipped_records.csv", index=False)

    scope = {
        "purpose": "patient/lead-calibrated asymmetric Varon crop audit",
        "ground_truth": {
            "ludb": "lead-specific manual cardiologist QRS onset/R/offset",
            "qtdb": (
                "selected q1c manual QRS boundaries associated with the "
                "record's expert atr R-anchor stream"
            ),
            "mitdb": (
                "not used: MIT-BIH Arrhythmia beat annotations do not provide "
                "manual beatwise QRS onset and offset ground truth"
            ),
        },
        "calibration_beats": calibration_sizes,
        "test_beats": "strictly later annotated QRS beats from the same record",
        "proposed_rule": (
            "separate NumPy-linear p95 of onset-to-R and R-to-offset; ceil "
            "each side to samples; include R once"
        ),
        "minimum_p95_calibration_beats": args.minimum_p95_calibration_beats,
        "fallback": "current fixed 120-ms extractor when k is below the minimum",
        "lead_scope": {
            "ludb": args.ludb_lead,
            "qtdb": f"channel 0 restricted to {args.qtdb_lead}",
            "transfer": "no cross-lead or cross-patient calibration",
        },
        "variants": sorted(summary["variant"].unique().tolist()) if not summary.empty else [],
        "extra_non_qrs_ms_definition": (
            "max(pre-onset_to_R,0)+max(post-R_to_offset,0)"
        ),
        "missed_qrs_ms_definition": (
            "max(onset_to_R-pre,0)+max(R_to_offset-post,0)"
        ),
        "p_t_overlap": (
            "reported only where manual P offset or T onset is available; "
            "overlap is geometric and does not prove contamination"
        ),
        "not_measured": [
            "automatic R-peak performance",
            "automatic QRS-boundary performance",
            "Varon eigenvalue stability or normalization",
            "seizure sensitivity, PPV, false alarms, or prediction",
        ],
    }
    (output / "benchmark_scope.json").write_text(
        json.dumps(scope, indent=2), encoding="utf-8"
    )
    (output / "SUMMARY.md").write_text(
        _build_markdown_summary(summary, scope, skipped), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    return 0


def _calibrated_windows(
    calibration: pd.DataFrame,
    sampling_rate_hz: float,
    *,
    minimum_p95_calibration_beats: int,
) -> list[dict[str, object]]:
    before = calibration["onset_lead_ms"].to_numpy(dtype=float)
    after = calibration["offset_lag_ms"].to_numpy(dtype=float)
    symmetric_p95_ms, _ = calibrate_patient_symmetric_varon_width(
        before, after, statistic="p95"
    )
    asymmetric_p95 = calibrate_patient_asymmetric_varon_window(
        before, after, statistic="p95"
    )
    asymmetric_max = calibrate_patient_asymmetric_varon_window(
        before, after, statistic="max"
    )

    fixed_actual = _sample_symmetric_total(120.0, sampling_rate_hz)
    symmetric_actual = _sample_symmetric_total(
        symmetric_p95_ms, sampling_rate_hz
    )
    p95_actual = _sample_asymmetric_sides(
        *asymmetric_p95, sampling_rate_hz
    )
    maximum_actual = _sample_asymmetric_sides(
        *asymmetric_max, sampling_rate_hz
    )
    fallback_used = len(calibration) < minimum_p95_calibration_beats
    safeguarded = fixed_actual if fallback_used else p95_actual

    return [
        _window_row(
            "fixed_120ms_nominal",
            requested_pre_ms=60.0,
            requested_post_ms=60.0,
            realized_pre_ms=60.0,
            realized_post_ms=60.0,
            pre_samples=None,
            post_samples=None,
            sample_rule="nominal_millisecond_geometry_from_prior_audit",
        ),
        _window_row(
            FIXED_EXTRACTOR,
            requested_pre_ms=60.0,
            requested_post_ms=60.0,
            **fixed_actual,
        ),
        _window_row(
            "required_p95_symmetric_current_extractor",
            requested_pre_ms=symmetric_p95_ms / 2.0,
            requested_post_ms=symmetric_p95_ms / 2.0,
            **symmetric_actual,
        ),
        _window_row(
            "separate_p95_asymmetric",
            requested_pre_ms=asymmetric_p95[0],
            requested_post_ms=asymmetric_p95[1],
            **p95_actual,
        ),
        _window_row(
            "separate_p95_asymmetric_min5_fallback",
            requested_pre_ms=(60.0 if fallback_used else asymmetric_p95[0]),
            requested_post_ms=(60.0 if fallback_used else asymmetric_p95[1]),
            fallback_used=fallback_used,
            **safeguarded,
        ),
        _window_row(
            "separate_max_asymmetric_sensitivity",
            requested_pre_ms=asymmetric_max[0],
            requested_post_ms=asymmetric_max[1],
            **maximum_actual,
        ),
    ]


def _window_row(
    variant: str,
    *,
    requested_pre_ms: float,
    requested_post_ms: float,
    realized_pre_ms: float,
    realized_post_ms: float,
    pre_samples: int | None,
    post_samples: int | None,
    sample_rule: str,
    fallback_used: bool = False,
) -> dict[str, object]:
    return {
        "variant": variant,
        "requested_pre_r_ms": float(requested_pre_ms),
        "requested_post_r_ms": float(requested_post_ms),
        "requested_window_ms": float(requested_pre_ms + requested_post_ms),
        "realized_pre_r_ms": float(realized_pre_ms),
        "realized_post_r_ms": float(realized_post_ms),
        "realized_window_ms": float(realized_pre_ms + realized_post_ms),
        "pre_r_samples": pre_samples,
        "post_r_samples": post_samples,
        "sample_rule": sample_rule,
        "fallback_used": bool(fallback_used),
    }


def _sample_symmetric_total(
    total_ms: float, sampling_rate_hz: float
) -> dict[str, object]:
    waveform_samples = max(
        3, int(np.rint(total_ms * sampling_rate_hz / 1000.0))
    )
    pre_samples = waveform_samples // 2
    post_samples = waveform_samples - pre_samples - 1
    return {
        "realized_pre_ms": 1000.0 * pre_samples / sampling_rate_hz,
        "realized_post_ms": 1000.0 * post_samples / sampling_rate_hz,
        "pre_samples": pre_samples,
        "post_samples": post_samples,
        "sample_rule": "published_total_length_then_symmetric_split",
    }


def _sample_asymmetric_sides(
    pre_ms: float, post_ms: float, sampling_rate_hz: float
) -> dict[str, object]:
    pre_samples = max(1, int(np.ceil(pre_ms * sampling_rate_hz / 1000.0)))
    post_samples = max(1, int(np.ceil(post_ms * sampling_rate_hz / 1000.0)))
    return {
        "realized_pre_ms": 1000.0 * pre_samples / sampling_rate_hz,
        "realized_post_ms": 1000.0 * post_samples / sampling_rate_hz,
        "pre_samples": pre_samples,
        "post_samples": post_samples,
        "sample_rule": "ceil_each_asymmetric_side_and_include_anchor_once",
    }


def _capture_metrics(
    rows: pd.DataFrame, pre_ms: float, post_ms: float
) -> pd.DataFrame:
    before = rows["onset_lead_ms"].to_numpy(dtype=float)
    after = rows["offset_lag_ms"].to_numpy(dtype=float)
    missed = np.maximum(before - pre_ms, 0.0) + np.maximum(
        after - post_ms, 0.0
    )
    extra = np.maximum(pre_ms - before, 0.0) + np.maximum(
        post_ms - after, 0.0
    )
    return pd.DataFrame(
        {
            "complete": (before <= pre_ms) & (after <= post_ms),
            "missed_ms": missed,
            "extra_ms": extra,
        }
    )


def _beat_result(
    *,
    dataset: str,
    record: ManualDelineationRecord,
    calibration_count: int,
    row: object,
    window: dict[str, object],
    p_offset_sample: int | None,
    t_onset_sample: int | None,
) -> dict[str, object]:
    pre_ms = float(window["realized_pre_r_ms"])
    post_ms = float(window["realized_post_r_ms"])
    onset_lead_ms = float(row.onset_lead_ms)
    offset_lag_ms = float(row.offset_lag_ms)
    complete = onset_lead_ms <= pre_ms and offset_lag_ms <= post_ms
    missed = max(onset_lead_ms - pre_ms, 0.0) + max(
        offset_lag_ms - post_ms, 0.0
    )
    extra = max(pre_ms - onset_lead_ms, 0.0) + max(
        post_ms - offset_lag_ms, 0.0
    )
    r_peak = int(row.r_peak_sample)
    p_distance_ms = (
        1000.0 * (r_peak - p_offset_sample) / record.sampling_rate_hz
        if p_offset_sample is not None and p_offset_sample <= r_peak
        else np.nan
    )
    t_distance_ms = (
        1000.0 * (t_onset_sample - r_peak) / record.sampling_rate_hz
        if t_onset_sample is not None and t_onset_sample >= r_peak
        else np.nan
    )
    return {
        "dataset": dataset,
        "record_id": record.record_id,
        "lead_name": record.lead_name,
        "sampling_rate_hz": record.sampling_rate_hz,
        "calibration_beat_count": calibration_count,
        "evaluation_anchor_index": int(row.anchor_index),
        "evaluation_r_peak_sample": r_peak,
        "evaluation_onset_lead_ms": onset_lead_ms,
        "evaluation_offset_lag_ms": offset_lag_ms,
        "evaluation_qrs_duration_ms": float(row.qrs_duration_ms),
        **window,
        "complete_qrs_captured": bool(complete),
        "missed_qrs_ms": float(missed),
        "extra_non_qrs_ms": float(extra),
        "p_offset_available": bool(np.isfinite(p_distance_ms)),
        "p_offset_to_r_ms": float(p_distance_ms),
        "p_wave_overlap": (
            bool(pre_ms >= p_distance_ms) if np.isfinite(p_distance_ms) else pd.NA
        ),
        "t_onset_available": bool(np.isfinite(t_distance_ms)),
        "r_to_t_onset_ms": float(t_distance_ms),
        "t_wave_overlap": (
            bool(post_ms >= t_distance_ms) if np.isfinite(t_distance_ms) else pd.NA
        ),
    }


def _summarize(
    beats: pd.DataFrame, calibrations: pd.DataFrame
) -> pd.DataFrame:
    if beats.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    group_columns = ["dataset", "calibration_beat_count", "variant"]
    for keys, group in beats.groupby(group_columns, sort=True):
        dataset, calibration_count, variant = keys
        calibration_group = calibrations.loc[
            (calibrations["dataset"] == dataset)
            & (calibrations["calibration_beat_count"] == calibration_count)
            & (calibrations["variant"] == variant)
        ]
        captured = group["complete_qrs_captured"].astype(bool)
        successes = int(captured.sum())
        ci_low, ci_high = _wilson_interval(successes, len(group))
        record_capture = group.groupby("record_id")[
            "complete_qrs_captured"
        ].mean()
        p_rows = group.loc[group["p_offset_available"].astype(bool)]
        t_rows = group.loc[group["t_onset_available"].astype(bool)]
        rows.append(
            {
                "dataset": dataset,
                "calibration_beat_count": int(calibration_count),
                "variant": variant,
                "records": int(group["record_id"].nunique()),
                "held_out_beats": int(len(group)),
                "median_requested_pre_r_ms": float(
                    calibration_group["requested_pre_r_ms"].median()
                ),
                "median_requested_post_r_ms": float(
                    calibration_group["requested_post_r_ms"].median()
                ),
                "median_realized_window_ms": float(
                    calibration_group["realized_window_ms"].median()
                ),
                "p95_realized_window_ms_across_records": float(
                    calibration_group["realized_window_ms"].quantile(0.95)
                ),
                "pooled_capture_fraction": float(captured.mean()),
                "pooled_capture_wilson95_low": ci_low,
                "pooled_capture_wilson95_high": ci_high,
                "record_macro_capture_fraction": float(record_capture.mean()),
                "mean_missed_qrs_ms": float(group["missed_qrs_ms"].mean()),
                "p95_missed_qrs_ms": float(group["missed_qrs_ms"].quantile(0.95)),
                "mean_extra_non_qrs_ms": float(
                    group["extra_non_qrs_ms"].mean()
                ),
                "p_offset_available_beats": int(len(p_rows)),
                "p_wave_overlap_fraction_where_available": (
                    float(p_rows["p_wave_overlap"].astype(bool).mean())
                    if len(p_rows)
                    else np.nan
                ),
                "t_onset_available_beats": int(len(t_rows)),
                "t_wave_overlap_fraction_where_available": (
                    float(t_rows["t_wave_overlap"].astype(bool).mean())
                    if len(t_rows)
                    else np.nan
                ),
            }
        )
    summary = pd.DataFrame(rows)

    beat_keys = [
        "dataset",
        "record_id",
        "calibration_beat_count",
        "evaluation_anchor_index",
    ]
    fixed = beats.loc[
        beats["variant"] == FIXED_EXTRACTOR,
        beat_keys + ["complete_qrs_captured"],
    ].rename(columns={"complete_qrs_captured": "fixed_captured"})
    gains: dict[tuple[str, int, str], tuple[int, int]] = {}
    for keys, group in beats.groupby(group_columns, sort=True):
        compared = group.merge(fixed, on=beat_keys, how="left", validate="one_to_one")
        current = compared["complete_qrs_captured"].astype(bool)
        control = compared["fixed_captured"].astype(bool)
        gains[keys] = (int((current & ~control).sum()), int((control & ~current).sum()))
    summary["beats_gained_vs_fixed_extractor"] = [
        gains[(row.dataset, row.calibration_beat_count, row.variant)][0]
        for row in summary.itertuples(index=False)
    ]
    summary["beats_lost_vs_fixed_extractor"] = [
        gains[(row.dataset, row.calibration_beat_count, row.variant)][1]
        for row in summary.itertuples(index=False)
    ]
    fixed_capture = summary.loc[
        summary["variant"] == FIXED_EXTRACTOR,
        ["dataset", "calibration_beat_count", "pooled_capture_fraction"],
    ].rename(columns={"pooled_capture_fraction": "fixed_capture_fraction"})
    summary = summary.merge(
        fixed_capture,
        on=["dataset", "calibration_beat_count"],
        how="left",
        validate="many_to_one",
    )
    summary["capture_change_vs_fixed_extractor"] = (
        summary["pooled_capture_fraction"] - summary["fixed_capture_fraction"]
    )
    return summary.drop(columns=["fixed_capture_fraction"])


def _landmark_by_anchor(
    record: ManualDelineationRecord, landmark: str
) -> dict[int, int]:
    indices = associate_landmarks_with_anchors(
        record.landmarks[landmark],
        record.anchor_samples,
        landmark=landmark,
        sampling_rate_hz=record.sampling_rate_hz,
    )
    return {
        int(index): int(sample)
        for sample, index in zip(record.landmarks[landmark], indices)
        if index >= 0
    }


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return np.nan, np.nan
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return float(center - half), float(center + half)


def _build_markdown_summary(
    summary: pd.DataFrame,
    scope: dict[str, object],
    skipped: pd.DataFrame,
) -> str:
    lines = [
        "# Patient-specific asymmetric Varon window benchmark",
        "",
        "The first k manually delineated QRS complexes in each record calibrate "
        "the window; only later delineated beats are evaluated.",
        "",
        "| Dataset | k | Variant | Records | Beats | Median window (ms) | "
        "Capture | Change vs sampled fixed | Mean extra (ms) |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.calibration_beat_count} | {row.variant} | "
            f"{row.records} | {row.held_out_beats} | "
            f"{row.median_realized_window_ms:.1f} | "
            f"{100.0 * row.pooled_capture_fraction:.1f}% | "
            f"{100.0 * row.capture_change_vs_fixed_extractor:+.1f} pp | "
            f"{row.mean_extra_non_qrs_ms:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- LUDB and QTDB provide the manual QRS boundaries used here. MIT-BIH "
            "Arrhythmia was not used because its beat annotations are not "
            "beatwise manual QRS-onset/QRS-offset ground truth.",
            "- P- and T-wave overlap is evaluated only when the corresponding "
            "manual landmark exists; geometric overlap is a risk indicator, not "
            "proof that a feature is corrupted.",
            "- This benchmark measures crop geometry only. It does not show "
            "seizure accuracy or that the Varon eigenvalues improve.",
            f"- Skipped record/load failures: {len(skipped)}.",
            "",
            "Exact machine-readable scope is in `benchmark_scope.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def _read_records(dataset: Path) -> list[str]:
    return [
        value.strip()
        for value in (dataset / "RECORDS").read_text(encoding="utf-8").splitlines()
        if value.strip()
    ]


def _qtdb_records_for_channel0(dataset: Path, lead: str) -> list[str]:
    output = []
    for record_id in _read_records(dataset):
        header = wfdb.rdheader(str(dataset / record_id))
        if str(header.sig_name[0]).casefold() == lead.casefold():
            output.append(record_id)
    return output


if __name__ == "__main__":
    raise SystemExit(main())
