"""Compare independently re-extracted per-dataset matrices with the frozen matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecg_cascade.comprehensive_dataset import DATASET_KEYS, _label_columns, feature_keys  # noqa: E402


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--rebuild-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline_path = args.baseline.resolve() / "observations_with_labels.csv"
    baseline = pd.read_csv(baseline_path, low_memory=False, dtype={"row_id": "string"})
    rebuilt_frames = []
    reports = []
    for dataset in DATASET_KEYS:
        directory = args.rebuild_root.resolve() / dataset
        path = directory / "observations_with_labels.csv"
        report_path = directory / "validation_report.json"
        if not path.is_file() or not report_path.is_file():
            raise FileNotFoundError(f"Missing completed rebuild for {dataset}: {path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        reports.append(
            {
                "dataset_key": dataset,
                "validation_passed": bool(report["validation_passed"]),
                "failed_segments": int(report["failed_segments"]),
                "rows": int(report["observation_rows"]),
                "observations_sha256": file_hash(path),
            }
        )
        rebuilt_frames.append(pd.read_csv(path, low_memory=False, dtype={"row_id": "string"}))

    rebuilt = pd.concat(rebuilt_frames, ignore_index=True, sort=False)
    sort_columns = ["dataset_key", "record_id", "timestamp_track", "observation_time_s"]
    baseline = baseline.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    rebuilt = rebuilt.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    baseline_ids = set(baseline["row_id"].astype(str))
    rebuilt_ids = set(rebuilt["row_id"].astype(str))
    missing = sorted(baseline_ids - rebuilt_ids)
    unexpected = sorted(rebuilt_ids - baseline_ids)

    common = sorted(baseline_ids.intersection(rebuilt_ids))
    left = baseline.set_index("row_id").loc[common]
    right = rebuilt.set_index("row_id").loc[common]
    numeric_differences = []
    for column in feature_keys():
        a = pd.to_numeric(left[column], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(right[column], errors="coerce").to_numpy(dtype=float)
        mismatch = ~(np.isclose(a, b, rtol=1e-10, atol=1e-10, equal_nan=True))
        if mismatch.any():
            numeric_differences.append(
                {
                    "column": column,
                    "different_rows": int(mismatch.sum()),
                    "maximum_absolute_difference": float(np.nanmax(np.abs(a[mismatch] - b[mismatch]))),
                }
            )

    label_differences = []
    for column in _label_columns():
        a_raw = left[column]
        b_raw = right[column]
        a_numeric = pd.to_numeric(a_raw, errors="coerce")
        b_numeric = pd.to_numeric(b_raw, errors="coerce")
        a_non_null = a_raw.notna()
        b_non_null = b_raw.notna()
        numeric_semantics = (
            a_numeric[a_non_null].notna().all()
            and b_numeric[b_non_null].notna().all()
        )
        if numeric_semantics:
            mismatch = pd.Series(
                ~(np.isclose(
                    a_numeric.to_numpy(dtype=float),
                    b_numeric.to_numpy(dtype=float),
                    rtol=1e-10,
                    atol=1e-10,
                    equal_nan=True,
                )),
                index=left.index,
            )
        else:
            # Textual source labels and provenance enums remain byte-for-byte exact.
            a = a_raw.astype("string").fillna("<NA>")
            b = b_raw.astype("string").fillna("<NA>")
            mismatch = a.ne(b)
        if mismatch.any():
            label_differences.append({"column": column, "different_rows": int(mismatch.sum())})

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "baseline": str(baseline_path),
        "baseline_rows": int(baseline.shape[0]),
        "rebuilt_rows": int(rebuilt.shape[0]),
        "common_row_ids": len(common),
        "missing_row_ids": len(missing),
        "unexpected_row_ids": len(unexpected),
        "numeric_feature_differences": numeric_differences,
        "label_differences": label_differences,
        "per_dataset_rebuilds": reports,
        "exact_reproduction": (
            baseline.shape[0] == rebuilt.shape[0]
            and not missing
            and not unexpected
            and not numeric_differences
            and not label_differences
            and all(row["validation_passed"] and row["failed_segments"] == 0 for row in reports)
        ),
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["exact_reproduction"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
