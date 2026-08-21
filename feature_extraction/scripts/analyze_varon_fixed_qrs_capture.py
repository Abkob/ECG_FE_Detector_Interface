"""Check the published fixed 120-ms Varon capture against manual QRS bounds."""

from __future__ import annotations

import argparse
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
    qrs_polarity_summary,
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
        default=PROJECT / "outputs" / "delineation_gate_matched_lead_v1",
    )
    parser.add_argument("--pre-r-ms", type=float, default=60.0)
    parser.add_argument("--post-r-ms", type=float, default=60.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ludb = args.ludb.expanduser().resolve()
    qtdb = args.qtdb.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    polarity_rows: list[dict[str, object]] = []
    for record_id in _read_records(ludb):
        record = load_ludb_delineation_record(ludb, record_id, lead="ii")
        rows.extend(
            fixed_qrs_capture_rows(
                record, pre_r_ms=args.pre_r_ms, post_r_ms=args.post_r_ms
            )
        )
        polarity_rows.append(qrs_polarity_summary(record))
    for record_id in _read_records(qtdb):
        header = wfdb.rdheader(str(qtdb / record_id))
        if str(header.sig_name[0]).casefold() != "mlii":
            continue
        record = load_qtdb_delineation_record(qtdb, record_id)
        rows.extend(
            fixed_qrs_capture_rows(
                record, pre_r_ms=args.pre_r_ms, post_r_ms=args.post_r_ms
            )
        )
    table = pd.DataFrame(rows)
    table.to_csv(output / "varon_fixed_qrs_capture.csv", index=False)
    pooled_rows = []
    for dataset, group in table.groupby("dataset", sort=True):
        pooled_rows.append(_summary_row(dataset, group))
    pooled_rows.append(_summary_row("combined", table))
    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(output / "varon_fixed_qrs_capture_summary.csv", index=False)
    polarity = pd.DataFrame(polarity_rows)
    polarity.to_csv(output / "ludb_lead_ii_polarity.csv", index=False)
    _write_polarity_stratification(output, polarity)
    print(pooled.to_string(index=False), flush=True)
    return 0


def _summary_row(dataset: str, group: pd.DataFrame) -> dict[str, object]:
    return {
        "dataset": dataset,
        "beats_with_complete_manual_qrs": int(len(group)),
        "complete_qrs_capture_fraction": float(
            group["complete_qrs_captured"].astype(bool).mean()
        ),
        "median_onset_lead_ms": float(group["onset_lead_ms"].median()),
        "p95_onset_lead_ms": float(group["onset_lead_ms"].quantile(0.95)),
        "median_offset_lag_ms": float(group["offset_lag_ms"].median()),
        "p95_offset_lag_ms": float(group["offset_lag_ms"].quantile(0.95)),
        "median_qrs_duration_ms": float(group["qrs_duration_ms"].median()),
        "p95_qrs_duration_ms": float(group["qrs_duration_ms"].quantile(0.95)),
    }


def _read_records(dataset: Path) -> list[str]:
    return [
        value.strip()
        for value in (dataset / "RECORDS").read_text(encoding="utf-8").splitlines()
        if value.strip()
    ]


def _write_polarity_stratification(
    output: Path,
    polarity: pd.DataFrame,
) -> None:
    metric_path = output / "record_landmark_metrics.csv"
    error_path = output / "matched_landmark_errors.csv"
    if not metric_path.is_file() or not error_path.is_file():
        return
    metrics = pd.read_csv(metric_path)
    errors = pd.read_csv(error_path)
    groups = polarity[["record_id", "record_polarity_group"]]
    metrics = metrics.merge(groups, on="record_id", how="inner")
    errors = errors.merge(groups, on="record_id", how="inner")
    rows = []
    for (method, landmark, polarity_group), group in metrics.groupby(
        ["method", "landmark", "record_polarity_group"], sort=True
    ):
        reference_count = int(group["reference_count"].sum())
        predicted_count = int(group["predicted_count"].sum())
        matched_count = int(group["matched_count"].sum())
        absolute = errors.loc[
            (errors["method"] == method)
            & (errors["landmark"] == landmark)
            & (errors["record_polarity_group"] == polarity_group),
            "absolute_timing_error_ms",
        ]
        rows.append(
            {
                "method": method,
                "landmark": landmark,
                "record_polarity_group": polarity_group,
                "records": int(group["record_id"].nunique()),
                "reference_count": reference_count,
                "predicted_count": predicted_count,
                "matched_count": matched_count,
                "sensitivity": matched_count / reference_count
                if reference_count
                else np.nan,
                "ppv": matched_count / predicted_count
                if predicted_count
                else np.nan,
                "median_absolute_error_ms": float(absolute.median())
                if not absolute.empty
                else np.nan,
                "p95_absolute_error_ms": float(absolute.quantile(0.95))
                if not absolute.empty
                else np.nan,
            }
        )
    pd.DataFrame(rows).to_csv(
        output / "ludb_lead_ii_polarity_stratification.csv", index=False
    )


if __name__ == "__main__":
    raise SystemExit(main())
