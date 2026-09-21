"""Recover a completed OOF audit from its durable prediction ledger.

The full evaluator writes every experiment's predictions before running the
post-hoc controls.  This command intentionally performs no model selection
from memory: it recalculates every metric, fold and group-bootstrap interval
from that on-disk ledger, checkpoints the results, then runs the selected
permutation controls with per-target checkpoints.
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

from ecg_cascade.comprehensive_dataset import branch_combinations, feature_keys  # noqa: E402
from ecg_cascade.model_benchmark import MODEL_SPECS, TARGET_SPECS  # noqa: E402
from ecg_cascade.rigorous_audit import (  # noqa: E402
    AUDIT_VERSION,
    _baseline_rows,
    _cluster_bootstrap_interval,
    _group_macro_metrics,
    _inventory_tables,
    _json_safe,
    _matrix_integrity,
    _permutation_check,
    _record_and_case_tables,
    _reliability_grade,
    _seed_for,
    _sha256,
    _write_markdown_report,
    grouped_fold_ids,
    metric_record,
)


def _prediction_rows(path: Path, target_id: str | None = None, ids: set[str] | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, compression="gzip", chunksize=150_000):
        keep = pd.Series(True, index=chunk.index)
        if target_id is not None:
            keep &= chunk["target_id"].eq(target_id)
        if ids is not None:
            keep &= chunk["experiment_id"].isin(ids)
        current = chunk.loc[keep]
        if not current.empty:
            frames.append(current)
    if not frames:
        raise RuntimeError(f"No matching predictions in {path}")
    return pd.concat(frames, ignore_index=True)


def _raw_report(output: Path) -> dict[str, object]:
    inventory = pd.read_csv(output / "raw_file_inventory.csv")
    duplicates = pd.read_csv(output / "raw_duplicate_files.csv")
    failed = inventory.loc[inventory["read_status"].ne("complete")]
    return {
        "files_planned": int(inventory.shape[0]),
        "files_byte_read": int(inventory["read_status"].eq("complete").sum()),
        "bytes_byte_read": int(inventory.loc[inventory["read_status"].eq("complete"), "bytes"].sum()),
        "read_errors": failed["path"].astype(str).tolist(),
        "duplicate_content_sets": int(duplicates.shape[0]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--web-json", type=Path, required=True)
    parser.add_argument("--bootstraps", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=20)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    output = args.audit.resolve()
    ledger = output / "all_group_oof_predictions.csv.gz"
    observations_path = dataset / "observations_with_labels.csv"
    observations = pd.read_csv(
        observations_path,
        low_memory=False,
        dtype={
            "row_id": "string", "dataset_key": "string", "record_id": "string",
            "subject_id": "string", "lineage_group_id": "string",
        },
    )
    combos = branch_combinations()
    combo_map = {row["combination_id"]: row for row in combos}
    integrity, duplicate_features = _matrix_integrity(observations)
    duplicate_features.to_csv(output / "duplicate_feature_vectors.csv", index=False)
    datasets, group_inventory, missingness = _inventory_tables(observations)
    datasets.to_csv(output / "dataset_inventory.csv", index=False)
    group_inventory.to_csv(output / "lineage_group_inventory.csv", index=False)
    missingness.to_csv(output / "feature_missingness_by_dataset.csv", index=False)

    tasks: dict[str, pd.DataFrame] = {}
    fold_ids_by_target: dict[str, np.ndarray] = {}
    target_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []

    for target_id, spec in TARGET_SPECS.items():
        task = observations.loc[observations[spec.key].notna()].copy().reset_index(drop=True)
        task[spec.key] = task[spec.key].astype(int)
        tasks[target_id] = task
        fold_ids, split_manifest = grouped_fold_ids(task, spec.key)
        fold_ids_by_target[target_id] = fold_ids
        split_rows.extend({"target_id": target_id, **row} for row in split_manifest)
        baseline_rows.extend(_baseline_rows(target_id, task, spec.key, fold_ids))
        y = task[spec.key].to_numpy(dtype=int)
        groups = task["lineage_group_id"].astype(str).to_numpy()
        target_rows.append({
            "id": target_id, "key": spec.key, "label": spec.label,
            "negative_label": spec.negative_label, "positive_label": spec.positive_label,
            "definition": spec.definition, "rows": int(task.shape[0]),
            "class_0": int(np.sum(y == 0)), "class_1": int(np.sum(y == 1)),
            "groups": int(np.unique(groups).size), "records": int(task["record_id"].nunique()),
            "subjects": int(task["subject_id"].nunique()), "folds": int(np.unique(fold_ids).size),
            "rows_per_group": float(task.shape[0] / np.unique(groups).size),
            "reliability": _reliability_grade(int(np.unique(groups).size)),
        })

        target_predictions = _prediction_rows(ledger, target_id=target_id)
        if target_predictions["experiment_id"].nunique() != 75:
            raise RuntimeError(f"{target_id}: expected 75 experiments in ledger")
        order = pd.Series(np.arange(task.shape[0]), index=task["row_id"].astype(str))
        complete_by_combo = {
            combo_id: int(task[list(json.loads(str(combo["feature_columns_json"])))].notna().all(axis=1).sum())
            for combo_id, combo in combo_map.items()
        }
        for experiment_id, frame in target_predictions.groupby("experiment_id", sort=False):
            frame = frame.assign(_order=frame["row_id"].astype(str).map(order)).sort_values("_order")
            if frame.shape[0] != task.shape[0] or frame["_order"].isna().any():
                raise RuntimeError(f"{experiment_id}: prediction coverage does not match matrix rows")
            if not np.array_equal(frame["oof_fold"].to_numpy(dtype=int), fold_ids):
                raise RuntimeError(f"{experiment_id}: durable fold IDs differ from reconstructed group split")
            truth = frame["y_true"].to_numpy(dtype=int)
            pred = frame["y_pred"].to_numpy(dtype=int)
            score = frame["decision_score"].to_numpy(dtype=float)
            if not np.array_equal(truth, y):
                raise RuntimeError(f"{experiment_id}: truth labels differ from source matrix")
            model_id = str(frame["model_id"].iloc[0])
            combo_id = str(frame["combination_id"].iloc[0])
            combo = combo_map[combo_id]
            low, high = _cluster_bootstrap_interval(
                truth, pred, groups, repetitions=args.bootstraps, seed=_seed_for(experiment_id)
            )
            result_rows.append({
                "experiment_id": experiment_id, "target_id": target_id,
                "target_column": spec.key, "model_id": model_id,
                "model_label": MODEL_SPECS[model_id].label, "combination_id": combo_id,
                "branches": combo["branches"], "branch_count": int(combo["branch_count"]),
                "feature_count": int(combo["feature_count"]),
                "oof_folds": int(np.unique(fold_ids).size), "rows": int(task.shape[0]),
                "groups": int(np.unique(groups).size), "records": int(task["record_id"].nunique()),
                "complete_case_rows": complete_by_combo[combo_id],
                "balanced_accuracy_group_bootstrap_ci_low": low,
                "balanced_accuracy_group_bootstrap_ci_high": high,
                **metric_record(truth, pred, score),
                **_group_macro_metrics(truth, pred, groups),
            })
            for fold in sorted(frame["oof_fold"].unique()):
                mask = frame["oof_fold"].eq(fold).to_numpy()
                fold_rows.append({
                    "experiment_id": experiment_id, "target_id": target_id,
                    "fold": int(fold), "fit_seconds": None,
                    "recovered_from_prediction_ledger": True,
                    **metric_record(truth[mask], pred[mask], score[mask]),
                })
        print(f"[recovery] {target_id}: verified 75 complete OOF experiments", flush=True)

    results = pd.DataFrame(result_rows)
    if results.shape[0] != 375:
        raise RuntimeError(f"Expected 375 recovered experiments, found {results.shape[0]}")
    results["rank_within_target"] = (
        results.groupby("target_id")["group_macro_balanced_accuracy"]
        .rank(method="min", ascending=False).astype(int)
    )
    results = results.sort_values(
        ["target_id", "rank_within_target", "balanced_accuracy_group_bootstrap_ci_low"],
        ascending=[True, True, False], kind="stable",
    ).reset_index(drop=True)
    selected = (
        results.sort_values(
            ["target_id", "group_macro_balanced_accuracy", "balanced_accuracy_group_bootstrap_ci_low", "balanced_accuracy"],
            ascending=[True, False, False, False], kind="stable",
        ).groupby("target_id", as_index=False, sort=False).head(1).copy()
    )

    # Durable checkpoint before the expensive permutation controls.
    results.to_csv(output / "group_oof_results.csv", index=False)
    selected.to_csv(output / "selected_experiments.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "fold_metrics.csv", index=False)
    pd.DataFrame(split_rows).to_csv(output / "split_manifest.csv", index=False)
    pd.DataFrame(baseline_rows).to_csv(output / "leakage_and_naive_baselines.csv", index=False)

    selected_predictions = _prediction_rows(ledger, ids=set(selected["experiment_id"]))
    cases, records, patients, weak = _record_and_case_tables(observations, selected_predictions)
    cases.to_csv(output / "selected_case_predictions.csv.gz", index=False, compression="gzip")
    records.to_csv(output / "record_metrics.csv", index=False)
    patients.to_csv(output / "patient_metrics.csv", index=False)
    weak.to_csv(output / "weak_cases_all_errors.csv.gz", index=False, compression="gzip")
    weak_records = records.copy()
    weak_records["ranking_score"] = weak_records["balanced_accuracy"].fillna(weak_records["accuracy"])
    weak_records.sort_values(
        ["target_id", "ranking_score", "errors"], ascending=[True, True, False], kind="stable"
    ).to_csv(output / "weak_records_ranked.csv", index=False)

    permutation_path = output / "permutation_checks.csv"
    done = pd.read_csv(permutation_path).to_dict(orient="records") if permutation_path.is_file() else []
    done_targets = {str(row["target_id"]) for row in done}
    permutation_rows = list(done)
    for row in selected.itertuples(index=False):
        if row.target_id in done_targets:
            continue
        task = tasks[row.target_id]
        columns = list(json.loads(str(combo_map[row.combination_id]["feature_columns_json"])))
        check = _permutation_check(
            MODEL_SPECS[row.model_id].factory(),
            task[columns].apply(pd.to_numeric, errors="coerce"),
            task[TARGET_SPECS[row.target_id].key].to_numpy(dtype=int),
            fold_ids_by_target[row.target_id], repetitions=args.permutations,
            seed=_seed_for(f"permutation|{row.experiment_id}"),
        )
        permutation_rows.append({
            "target_id": row.target_id, "experiment_id": row.experiment_id,
            "observed_group_macro_balanced_accuracy": row.group_macro_balanced_accuracy,
            "observed_row_balanced_accuracy": row.balanced_accuracy, **check,
        })
        pd.DataFrame(permutation_rows).to_csv(permutation_path, index=False)
        print(f"[recovery] {row.target_id}: {args.permutations} permutation controls complete", flush=True)

    validation = json.loads((dataset / "validation_report.json").read_text(encoding="utf-8"))
    raw_report = _raw_report(output)
    web_weak = weak.groupby("target_id", group_keys=False, sort=False).head(100)[[
        "target_id", "dataset_key", "record_id", "subject_id", "lineage_group_id",
        "observation_time_s", "y_true", "y_pred", "decision_score", "error_type",
        "available_feature_count", "label_beat_symbol_original",
        "label_quality_consensus_original", "label_seizure_phase_derived",
        "label_noise_condition_original",
    ]]
    payload = {
        "audit_version": AUDIT_VERSION,
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "verdict": "Exploratory out-of-group backtest only. Every labelled row is predicted without its lineage group in training, but no target has an untouched external clinical cohort.",
        "dataset": {
            "profile": validation.get("profile"), "rows": int(observations.shape[0]),
            "records": int(observations["record_id"].nunique()),
            "subjects": int(observations["subject_id"].nunique()),
            "lineage_groups": int(observations["lineage_group_id"].nunique()),
            "features": len(feature_keys()), "matrix_sha256": _sha256(observations_path),
            "validation_passed": bool(validation.get("validation_passed")),
        },
        "raw_inventory": raw_report, "integrity": integrity,
        "protocol": {
            "split": "deterministic stratified group out-of-fold; 5 folds when >=5 groups, otherwise leave-small-group folds",
            "unit_of_independence": "lineage_group_id (patient/source family)",
            "preprocessing": "refit inside each training fold; no test-row imputation or scaling fit",
            "ranking": "mean of per-group sensitivity and per-group specificity",
            "uncertainty": f"{args.bootstraps} bootstrap resamples of complete lineage groups",
            "permutation": f"{args.permutations} training-label permutations for each displayed selected experiment",
            "selection_warning": "The displayed winner is selected on these same OOF results and is not an unbiased final-test estimate.",
            "recovery": "Metrics were independently recalculated from the durable complete OOF prediction ledger after terminal interruption.",
        },
        "targets": [_json_safe(row) for row in target_rows],
        "models": [{
            "id": key, "label": spec.label, "family": spec.family,
            "rationale": spec.rationale, "reference_title": spec.reference_title,
            "reference_url": spec.reference_url, "configuration": dict(spec.configuration),
        } for key, spec in MODEL_SPECS.items()],
        "combinations": [{"id": row["combination_id"], "branches": row["branches"], "feature_count": int(row["feature_count"])} for row in combos],
        "results": [_json_safe(row) for row in results.to_dict(orient="records")],
        "selected": [_json_safe(row) for row in selected.to_dict(orient="records")],
        "baselines": [_json_safe(row) for row in baseline_rows],
        "permutations": [_json_safe(row) for row in permutation_rows],
        "datasets": [_json_safe(row) for row in datasets.to_dict(orient="records")],
        "records": [_json_safe(row) for row in records.to_dict(orient="records")],
        "patients": [_json_safe(row) for row in patients.to_dict(orient="records")],
        "weak_cases": [_json_safe(row) for row in web_weak.to_dict(orient="records")],
        "files": {
            "all_experiment_predictions": ledger.name,
            "selected_case_predictions": "selected_case_predictions.csv.gz",
            "weak_cases_all_errors": "weak_cases_all_errors.csv.gz",
            "record_metrics": "record_metrics.csv", "patient_metrics": "patient_metrics.csv",
            "feature_associations": "feature_target_associations.csv",
            "raw_inventory": "raw_file_inventory.csv",
        },
    }
    (output / "audit_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    args.web_json.resolve().write_text(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    _write_markdown_report(payload, output / "AUDIT_REPORT.md")
    print(json.dumps({"recovered_experiments": 375, "selected": selected["experiment_id"].tolist()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
