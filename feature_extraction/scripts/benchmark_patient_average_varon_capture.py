"""Benchmark patient-calibrated symmetric Varon widths against 120 ms.

For every LUDB lead-II or matched QTDB MLII record, the first ``k`` complete
manual QRS annotations calibrate one symmetric width.  The original proposal
uses mean total duration.  Corrected variants summarize the per-beat required
width ``2 * max(onset-to-R, R-to-offset)`` by its mean, p95, or maximum.  Every
variant is evaluated only on later manual QRS complexes from the same record.
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
    fixed_qrs_capture_rows,
    load_ludb_delineation_record,
    load_qtdb_delineation_record,
)


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
        default=PROJECT / "outputs" / "patient_symmetric_varon_capture_v2",
    )
    parser.add_argument(
        "--calibration-beats",
        nargs="+",
        type=int,
        default=[3, 5, 10, 20],
    )
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
    for dataset, record_ids, loader in loaders:
        for record_number, record_id in enumerate(record_ids, start=1):
            record = loader(record_id)
            manual = pd.DataFrame(fixed_qrs_capture_rows(record))
            if manual.empty:
                print(
                    f"skipped {dataset} {record.record_id}: no complete manual "
                    "QRS onset/R/offset triplets",
                    flush=True,
                )
                continue
            manual = manual.sort_values("r_peak_sample", kind="stable")
            for calibration_count in calibration_sizes:
                if len(manual) <= calibration_count:
                    continue
                calibration = manual.iloc[:calibration_count]
                evaluation = manual.iloc[calibration_count:]
                personalized_width_ms = float(
                    calibration["qrs_duration_ms"].mean()
                )
                half_width_ms = personalized_width_ms / 2.0
                required_widths_ms = 2.0 * np.maximum(
                    calibration["onset_lead_ms"].to_numpy(dtype=float),
                    calibration["offset_lag_ms"].to_numpy(dtype=float),
                )
                required_mean_width_ms = float(np.mean(required_widths_ms))
                required_p95_width_ms = float(
                    np.quantile(required_widths_ms, 0.95)
                )
                required_max_width_ms = float(np.max(required_widths_ms))
                calibration_rows.append(
                    {
                        "dataset": dataset,
                        "record_id": record.record_id,
                        "lead_name": record.lead_name,
                        "calibration_beat_count": calibration_count,
                        "evaluation_beat_count": int(len(evaluation)),
                        "calibration_mean_qrs_duration_ms": personalized_width_ms,
                        "calibration_min_qrs_duration_ms": float(
                            calibration["qrs_duration_ms"].min()
                        ),
                        "calibration_max_qrs_duration_ms": float(
                            calibration["qrs_duration_ms"].max()
                        ),
                        "personalized_pre_r_ms": half_width_ms,
                        "personalized_post_r_ms": half_width_ms,
                        "required_mean_symmetric_width_ms": required_mean_width_ms,
                        "required_p95_symmetric_width_ms": required_p95_width_ms,
                        "required_max_symmetric_width_ms": required_max_width_ms,
                    }
                )
                for row in evaluation.itertuples(index=False):
                    personalized_complete = bool(
                        float(row.onset_lead_ms) <= half_width_ms
                        and float(row.offset_lag_ms) <= half_width_ms
                    )
                    required_mean_complete = _captured(
                        row, required_mean_width_ms
                    )
                    required_p95_complete = _captured(
                        row, required_p95_width_ms
                    )
                    required_max_complete = _captured(
                        row, required_max_width_ms
                    )
                    beat_rows.append(
                        {
                            "dataset": dataset,
                            "record_id": record.record_id,
                            "lead_name": record.lead_name,
                            "calibration_beat_count": calibration_count,
                            "calibration_mean_qrs_duration_ms": personalized_width_ms,
                            "personalized_pre_r_ms": half_width_ms,
                            "personalized_post_r_ms": half_width_ms,
                            "evaluation_anchor_index": int(row.anchor_index),
                            "evaluation_r_peak_sample": int(row.r_peak_sample),
                            "evaluation_onset_lead_ms": float(row.onset_lead_ms),
                            "evaluation_offset_lag_ms": float(row.offset_lag_ms),
                            "evaluation_qrs_duration_ms": float(row.qrs_duration_ms),
                            "fixed_120ms_complete_qrs_captured": bool(
                                row.complete_qrs_captured
                            ),
                            "patient_average_complete_qrs_captured": personalized_complete,
                            "required_mean_complete_qrs_captured": required_mean_complete,
                            "required_p95_complete_qrs_captured": required_p95_complete,
                            "required_max_complete_qrs_captured": required_max_complete,
                        }
                    )
            print(
                f"completed {dataset} {record.record_id} "
                f"({record_number}/{len(record_ids)})",
                flush=True,
            )

    beats = pd.DataFrame(beat_rows)
    calibrations = pd.DataFrame(calibration_rows)
    summary = _summarize(beats, calibrations)
    beats.to_csv(output / "held_out_beat_capture.csv", index=False)
    calibrations.to_csv(output / "record_calibrations.csv", index=False)
    summary.to_csv(output / "capture_summary.csv", index=False)
    scope = {
        "purpose": "patient-calibrated symmetric Varon window capture audit",
        "fixed_control_ms": 120.0,
        "personalized_parameters": [
            "mean total manual QRS duration in first k beats",
            "mean of per-beat 2x max(onset-to-R, R-to-offset)",
            "p95 of per-beat 2x max(onset-to-R, R-to-offset)",
            "maximum per-beat 2x max(onset-to-R, R-to-offset)",
        ],
        "placement": "symmetric about reference R anchor",
        "calibration_beats": calibration_sizes,
        "test_beats": "strictly later manual QRS beats from the same record",
        "ludb_lead": args.ludb_lead,
        "qtdb_lead": args.qtdb_lead,
        "not_measured": [
            "automatic R-peak performance",
            "automatic QRS-boundary performance",
            "Varon seizure discrimination",
            "seizure sensitivity or PPV",
        ],
    }
    (output / "benchmark_scope.json").write_text(
        json.dumps(scope, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    return 0


def _summarize(
    beats: pd.DataFrame,
    calibrations: pd.DataFrame,
) -> pd.DataFrame:
    if beats.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (dataset, calibration_count), group in beats.groupby(
        ["dataset", "calibration_beat_count"], sort=True
    ):
        calibration_group = calibrations.loc[
            (calibrations["dataset"] == dataset)
            & (calibrations["calibration_beat_count"] == calibration_count)
        ]
        fixed = group["fixed_120ms_complete_qrs_captured"].astype(bool)
        row: dict[str, object] = {
            "dataset": dataset,
            "calibration_beat_count": int(calibration_count),
            "records": int(group["record_id"].nunique()),
            "held_out_beats": int(len(group)),
            "fixed_120ms_capture_fraction": float(fixed.mean()),
        }
        variants = [
            (
                "mean_total",
                "calibration_mean_qrs_duration_ms",
                "patient_average_complete_qrs_captured",
            ),
            (
                "required_mean",
                "required_mean_symmetric_width_ms",
                "required_mean_complete_qrs_captured",
            ),
            (
                "required_p95",
                "required_p95_symmetric_width_ms",
                "required_p95_complete_qrs_captured",
            ),
            (
                "required_max",
                "required_max_symmetric_width_ms",
                "required_max_complete_qrs_captured",
            ),
        ]
        for label, width_column, capture_column in variants:
            captured = group[capture_column].astype(bool)
            row[f"{label}_median_window_ms"] = float(
                calibration_group[width_column].median()
            )
            row[f"{label}_capture_fraction"] = float(captured.mean())
            row[f"{label}_change_vs_fixed"] = float(
                captured.mean() - fixed.mean()
            )
            row[f"{label}_beats_gained_vs_fixed"] = int(
                (captured & ~fixed).sum()
            )
            row[f"{label}_beats_lost_vs_fixed"] = int(
                (fixed & ~captured).sum()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _captured(row: object, symmetric_width_ms: float) -> bool:
    half_width_ms = symmetric_width_ms / 2.0
    return bool(
        float(row.onset_lead_ms) <= half_width_ms
        and float(row.offset_lag_ms) <= half_width_ms
    )


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
