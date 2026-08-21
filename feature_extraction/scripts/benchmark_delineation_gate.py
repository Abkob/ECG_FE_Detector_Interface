"""Run the lead-controlled manual ECG-delineation acceptance gate.

This benchmark isolates waveform delineation from beat detection by supplying
manual/reference QRS anchors.  It evaluates NeuroKit delineators on LUDB lead
II and on QTDB records whose manually annotated channel 0 is MLII.
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
    LANDMARK_NAMES,
    ManualDelineationRecord,
    evaluate_landmark_samples,
    load_ludb_delineation_record,
    load_qtdb_delineation_record,
    run_neurokit_delineation,
)


EVALUATED_LANDMARKS = tuple(
    name for name in LANDMARK_NAMES if name != "r_peak"
)


def _default_ludb() -> Path:
    return (
        PROJECT.parent
        / "Datasets"
        / "lobachevsky-university-electrocardiography-database-1.0.1"
    )


def _default_qtdb() -> Path:
    return PROJECT.parent / "Datasets" / "qt-database-1.0.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ludb", type=Path, default=_default_ludb())
    parser.add_argument("--qtdb", type=Path, default=_default_qtdb())
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "outputs" / "delineation_gate_matched_lead_v1",
    )
    parser.add_argument("--methods", nargs="+", default=["dwt", "cwt"])
    parser.add_argument("--ludb-lead", default="ii")
    parser.add_argument("--qtdb-lead", default="MLII")
    parser.add_argument("--tolerance-ms", type=float, default=150.0)
    parser.add_argument(
        "--ludb-records",
        nargs="*",
        default=None,
        help="Optional LUDB record stems (for example data/1 data/2).",
    )
    parser.add_argument(
        "--qtdb-records",
        nargs="*",
        default=None,
        help="Optional QTDB record stems; otherwise channel-0 lead is filtered.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ludb = args.ludb.expanduser().resolve()
    qtdb = args.qtdb.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    methods = [str(value).casefold() for value in args.methods]

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

    metric_rows: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    record_rows: list[dict[str, object]] = []

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
                qtdb,
                record_id,
                annotation_extension="q1c",
                required_channel0=args.qtdb_lead,
            ),
        ),
    ]
    for dataset_name, record_ids, loader in loaders:
        for record_number, record_id in enumerate(record_ids, start=1):
            record = loader(record_id)
            record_rows.append(_record_scope_row(record))
            for method in methods:
                try:
                    predictions = run_neurokit_delineation(
                        record,
                        method=method,
                    )
                    failure = ""
                except Exception as exc:  # benchmark must retain failed records
                    predictions = {
                        landmark: np.asarray([], dtype=np.int64)
                        for landmark in LANDMARK_NAMES
                    }
                    failure = f"{type(exc).__name__}: {exc}"
                    failure_rows.append(
                        {
                            "dataset": dataset_name,
                            "record_id": record.record_id,
                            "lead_name": record.lead_name,
                            "method": method,
                            "error": failure,
                        }
                    )
                for landmark in EVALUATED_LANDMARKS:
                    metrics, errors_ms = evaluate_landmark_samples(
                        predictions[landmark],
                        record.landmarks[landmark],
                        sampling_rate_hz=record.sampling_rate_hz,
                        tolerance_ms=args.tolerance_ms,
                    )
                    metric_rows.append(
                        {
                            "dataset": dataset_name,
                            "record_id": record.record_id,
                            "lead_name": record.lead_name,
                            "sampling_rate_hz": record.sampling_rate_hz,
                            "method": method,
                            "landmark": landmark,
                            "run_succeeded": not bool(failure),
                            "run_error": failure,
                            **metrics,
                        }
                    )
                    error_rows.extend(
                        {
                            "dataset": dataset_name,
                            "record_id": record.record_id,
                            "lead_name": record.lead_name,
                            "method": method,
                            "landmark": landmark,
                            "timing_error_ms": float(error),
                            "absolute_timing_error_ms": float(abs(error)),
                        }
                        for error in errors_ms
                    )
            print(
                f"completed {dataset_name} {record_id} "
                f"({record_number}/{len(record_ids)})",
                flush=True,
            )

    metrics = pd.DataFrame(metric_rows)
    errors = pd.DataFrame(error_rows)
    failures = pd.DataFrame(
        failure_rows,
        columns=["dataset", "record_id", "lead_name", "method", "error"],
    )
    records = pd.DataFrame(record_rows).drop_duplicates()
    pooled = _pool_metrics(metrics, errors)

    records.to_csv(output / "record_scope.csv", index=False)
    metrics.to_csv(output / "record_landmark_metrics.csv", index=False)
    errors.to_csv(output / "matched_landmark_errors.csv", index=False)
    failures.to_csv(output / "run_failures.csv", index=False)
    pooled.to_csv(output / "pooled_landmark_metrics.csv", index=False)
    scope = {
        "claim": "manual landmark delineation fidelity with reference QRS anchors",
        "not_measured": [
            "R-peak detector accuracy",
            "seizure sensitivity",
            "seizure PPV",
            "incremental value beyond RR/HRV",
        ],
        "ludb": {
            "path": str(ludb),
            "lead": args.ludb_lead,
            "records": ludb_records,
            "annotation": "lead-specific manual boundaries and peaks",
        },
        "qtdb": {
            "path": str(qtdb),
            "channel_0_required": args.qtdb_lead,
            "records": qtdb_records,
            "annotation": "q1c second-pass manual selected beats",
            "anchor_context": "full atr reference stream when available",
        },
        "methods": methods,
        "matching_tolerance_ms": float(args.tolerance_ms),
        "neurokit_check": False,
        "neurokit_check_reason": (
            "NeuroKit2 0.2.13 CWT returns unequal landmark-list lengths; "
            "the built-in check raises before those failures can be audited"
        ),
    }
    (output / "benchmark_scope.json").write_text(
        json.dumps(scope, indent=2), encoding="utf-8"
    )
    (output / "SUMMARY.md").write_text(
        _markdown_summary(pooled, failures, scope), encoding="utf-8"
    )
    print(pooled.to_string(index=False), flush=True)
    return 0


def _record_scope_row(record: ManualDelineationRecord) -> dict[str, object]:
    return {
        "dataset": record.dataset,
        "record_id": record.record_id,
        "lead_name": record.lead_name,
        "lead_index": record.lead_index,
        "sampling_rate_hz": record.sampling_rate_hz,
        "sample_count": record.samples.size,
        "delineator_anchor_count": record.anchor_samples.size,
        "manually_delineated_beat_count": record.evaluation_anchor_indices.size,
        "annotation_source": record.annotation_source,
        **{
            f"reference_{landmark}_count": record.landmarks[landmark].size
            for landmark in LANDMARK_NAMES
        },
    }


def _read_records(dataset: Path) -> list[str]:
    records_path = dataset / "RECORDS"
    if not records_path.is_file():
        raise FileNotFoundError(records_path)
    return [
        value.strip()
        for value in records_path.read_text(encoding="utf-8").splitlines()
        if value.strip()
    ]


def _qtdb_records_for_channel0(dataset: Path, lead: str) -> list[str]:
    output = []
    for record_id in _read_records(dataset):
        header = wfdb.rdheader(str(dataset / record_id))
        if str(header.sig_name[0]).casefold() == lead.casefold():
            output.append(record_id)
    return output


def _pool_metrics(metrics: pd.DataFrame, errors: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, method, landmark), group in metrics.groupby(
        ["dataset", "method", "landmark"], sort=True
    ):
        reference_count = int(group["reference_count"].sum())
        predicted_count = int(group["predicted_count"].sum())
        matched_count = int(group["matched_count"].sum())
        error_values = errors.loc[
            (errors["dataset"] == dataset)
            & (errors["method"] == method)
            & (errors["landmark"] == landmark),
            "timing_error_ms",
        ].to_numpy(dtype=float)
        absolute = np.abs(error_values)
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "landmark": landmark,
                "records": int(group["record_id"].nunique()),
                "failed_records": int((~group["run_succeeded"].astype(bool)).sum()),
                "reference_count": reference_count,
                "predicted_count": predicted_count,
                "matched_count": matched_count,
                "sensitivity": _divide(matched_count, reference_count),
                "ppv": _divide(matched_count, predicted_count),
                "f1": _divide(
                    2 * matched_count, reference_count + predicted_count
                ),
                "mean_error_ms": _mean_or_nan(error_values),
                "median_error_ms": _median_or_nan(error_values),
                "median_absolute_error_ms": _median_or_nan(absolute),
                "p95_absolute_error_ms": _percentile_or_nan(absolute, 95.0),
                "within_20_ms": _mean_or_nan(absolute <= 20.0),
                "within_40_ms": _mean_or_nan(absolute <= 40.0),
                "within_80_ms": _mean_or_nan(absolute <= 80.0),
                "seizure_accuracy_measured": False,
            }
        )
    return pd.DataFrame(rows)


def _markdown_summary(
    pooled: pd.DataFrame,
    failures: pd.DataFrame,
    scope: dict[str, object],
) -> str:
    selected = pooled[
        pooled["landmark"].isin(
            ["p_onset", "qrs_onset", "qrs_offset", "t_offset"]
        )
    ][
        [
            "dataset",
            "method",
            "landmark",
            "reference_count",
            "sensitivity",
            "ppv",
            "median_absolute_error_ms",
            "p95_absolute_error_ms",
        ]
    ]
    table_lines = [
        "| dataset | method | landmark | reference | sensitivity | PPV | median abs error ms | p95 abs error ms |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in selected.itertuples(index=False):
        table_lines.append(
            "| "
            + " | ".join(
                [
                    str(row.dataset),
                    str(row.method),
                    str(row.landmark),
                    str(int(row.reference_count)),
                    _format_float(row.sensitivity),
                    _format_float(row.ppv),
                    _format_float(row.median_absolute_error_ms),
                    _format_float(row.p95_absolute_error_ms),
                ]
            )
            + " |"
        )
    return "\n".join(
        [
            "# Lead-controlled delineation gate",
            "",
            f"Methods: {', '.join(scope['methods'])}",
            f"Matching tolerance: {scope['matching_tolerance_ms']} ms",
            f"Runtime failures: {len(failures)}",
            "",
            "This benchmark supplies reference QRS anchors. It measures",
            "delineation fidelity, not beat detection or seizure accuracy.",
            "",
            *table_lines,
            "",
        ]
    )


def _format_float(value: object) -> str:
    number = float(value)
    return f"{number:.4g}" if np.isfinite(number) else "NA"


def _divide(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else np.nan


def _mean_or_nan(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else np.nan


def _median_or_nan(values: np.ndarray) -> float:
    return float(np.median(values)) if values.size else np.nan


def _percentile_or_nan(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values.size else np.nan


if __name__ == "__main__":
    raise SystemExit(main())
