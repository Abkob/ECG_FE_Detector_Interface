"""Refresh case tables and append independent evidence to a completed audit."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecg_cascade.comprehensive_dataset import feature_keys  # noqa: E402
from ecg_cascade.rigorous_audit import _json_safe, _record_and_case_tables  # noqa: E402


def add_descriptive_error_explanations(cases: pd.DataFrame, weak: pd.DataFrame) -> pd.DataFrame:
    """Describe which values make an error resemble the predicted class.

    This is deliberately not presented as causal attribution or SHAP. It is a
    reproducible class-median resemblance audit that works for all five model
    families.
    """

    output = weak.copy()
    output["descriptive_resemblance_json"] = "[]"
    output["missing_features_json"] = "[]"
    keys = feature_keys()
    for target_id, indices in output.groupby("target_id", sort=False).groups.items():
        reference = cases.loc[cases["target_id"].eq(target_id)]
        values = reference[keys].apply(pd.to_numeric, errors="coerce")
        medians = {
            label: values.loc[reference["y_true"].eq(label)].median(axis=0)
            for label in (0, 1)
        }
        scale = values.quantile(0.75).sub(values.quantile(0.25)).abs()
        fallback = values.std(ddof=0).replace(0.0, np.nan)
        scale = scale.replace(0.0, np.nan).fillna(fallback).replace(0.0, np.nan).fillna(1.0)
        for index in indices:
            row = output.loc[index]
            numeric = pd.to_numeric(row[keys], errors="coerce")
            true_label = int(row["y_true"])
            predicted_label = int(row["y_pred"])
            resemblance = (
                numeric.sub(medians[true_label]).abs()
                .sub(numeric.sub(medians[predicted_label]).abs())
                .div(scale)
                .dropna()
                .sort_values(ascending=False)
                .head(5)
            )
            explanation = [
                {
                    "feature": str(feature),
                    "value": float(numeric[feature]),
                    "predicted_class_resemblance": float(score),
                }
                for feature, score in resemblance.items()
            ]
            missing = [feature for feature in keys if pd.isna(numeric[feature])]
            output.at[index, "descriptive_resemblance_json"] = json.dumps(explanation)
            output.at[index, "missing_features_json"] = json.dumps(missing)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--web-json", type=Path, required=True)
    parser.add_argument("--exhaustive-plan", type=Path)
    parser.add_argument("--rebuild-comparison", type=Path)
    args = parser.parse_args()

    audit_dir = args.audit.resolve()
    payload_path = audit_dir / "audit_results.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    observations = pd.read_csv(
        args.dataset.resolve() / "observations_with_labels.csv",
        low_memory=False,
        dtype={"row_id": "string", "lineage_group_id": "string"},
    )
    selected_ids = {row["experiment_id"] for row in payload["selected"]}
    selected_frames = []
    for chunk in pd.read_csv(
        audit_dir / "all_group_oof_predictions.csv.gz",
        compression="gzip",
        chunksize=100_000,
    ):
        frame = chunk.loc[chunk["experiment_id"].isin(selected_ids)]
        if not frame.empty:
            selected_frames.append(frame)
    selected_predictions = pd.concat(selected_frames, ignore_index=True)
    cases, records, patients, weak = _record_and_case_tables(observations, selected_predictions)
    weak = add_descriptive_error_explanations(cases, weak)
    cases.to_csv(audit_dir / "selected_case_predictions.csv.gz", index=False, compression="gzip")
    records.to_csv(audit_dir / "record_metrics.csv", index=False)
    patients.to_csv(audit_dir / "patient_metrics.csv", index=False)
    weak.to_csv(audit_dir / "weak_cases_all_errors.csv.gz", index=False, compression="gzip")
    weak[
        [
            "target_id", "experiment_id", "row_id", "dataset_key", "record_id",
            "subject_id", "lineage_group_id", "observation_time_s", "error_type",
            "y_true", "y_pred", "decision_score", "descriptive_resemblance_json",
            "missing_features_json",
        ]
    ].to_csv(audit_dir / "weak_case_explanations.csv.gz", index=False, compression="gzip")
    weak_records = records.copy()
    weak_records["ranking_score"] = weak_records["balanced_accuracy"].fillna(weak_records["accuracy"])
    weak_records.sort_values(
        ["target_id", "ranking_score", "errors"],
        ascending=[True, True, False],
        kind="stable",
    ).to_csv(audit_dir / "weak_records_ranked.csv", index=False)

    payload["records"] = [_json_safe(row) for row in records.to_dict(orient="records")]
    payload["patients"] = [_json_safe(row) for row in patients.to_dict(orient="records")]
    web_weak = weak.groupby("target_id", group_keys=False, sort=False).head(100)
    web_columns = [
        "target_id", "dataset_key", "record_id", "subject_id", "lineage_group_id",
        "observation_time_s", "y_true", "y_pred", "decision_score", "error_type",
        "available_feature_count", "label_beat_symbol_original",
        "label_quality_consensus_original", "label_seizure_phase_derived",
        "label_noise_condition_original",
        "descriptive_resemblance_json",
    ]
    payload["weak_cases"] = [
        _json_safe(row) for row in web_weak[web_columns].to_dict(orient="records")
    ]

    semantics_path = audit_dir / "label_semantics_audit.json"
    if semantics_path.is_file():
        payload["label_semantics"] = json.loads(semantics_path.read_text(encoding="utf-8"))
    if args.exhaustive_plan and args.exhaustive_plan.is_file():
        payload["exhaustive_plan"] = json.loads(args.exhaustive_plan.read_text(encoding="utf-8"))
    if args.rebuild_comparison and args.rebuild_comparison.is_file():
        payload["independent_rebuild"] = json.loads(
            args.rebuild_comparison.read_text(encoding="utf-8")
        )

    encoded = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    payload_path.write_text(encoded, encoding="utf-8")
    args.web_json.resolve().write_text(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    web_asset_dir = args.web_json.resolve().parent
    for filename in (
        "selected_case_predictions.csv.gz",
        "weak_case_explanations.csv.gz",
        "record_metrics.csv",
        "patient_metrics.csv",
        "group_oof_results.csv",
        "split_manifest.csv",
    ):
        shutil.copy2(audit_dir / filename, web_asset_dir / filename)
    print(
        json.dumps(
            {
                "selected_case_rows": int(cases.shape[0]),
                "errors": int(weak.shape[0]),
                "records": int(records.shape[0]),
                "patients": int(patients.shape[0]),
                "web_json": str(args.web_json.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
