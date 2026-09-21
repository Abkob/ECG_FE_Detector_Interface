"""Patient/group-out audit for the comprehensive ECG feature matrix.

Unlike the original single-holdout benchmark, this module predicts every
labelled row from a model that never saw its lineage group.  It also retains
the fold, case, record, and patient evidence needed to audit failures.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .comprehensive_dataset import branch_combinations, feature_keys
from .model_benchmark import MODEL_SPECS, RANDOM_SEED, TARGET_SPECS, _decision_scores


AUDIT_VERSION = "2.0.0"
DEFAULT_BOOTSTRAPS = 1000
DEFAULT_PERMUTATIONS = 20

CASE_COLUMNS = [
    "row_id",
    "dataset_key",
    "record_id",
    "subject_id",
    "lineage_group_id",
    "segment_id",
    "source_path",
    "source_kind",
    "channel",
    "sampling_rate_hz",
    "segment_start_s",
    "segment_end_s",
    "observation_time_s",
    "timestamp_track",
    "available_feature_count",
    "missing_feature_count",
    "label_beat_symbol_original",
    "label_beat_aami_superclass_derived",
    "label_quality_consensus_original",
    "label_quality_annotator_agreement_count",
    "label_seizure_binary",
    "label_seizure_phase_derived",
    "label_seconds_from_seizure_onset",
    "label_noise_condition_original",
    "label_noise_snr_db_original",
    "label_noise_active",
    "target_artifact_source_derived",
]


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return _safe_float(value)
    if pd.isna(value):
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_record(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return explicit binary metrics without hiding undefined denominators."""

    truth = np.asarray(y_true, dtype=int)
    predicted = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if tp + fn else None
    specificity = float(tn / (tn + fp)) if tn + fp else None
    precision = float(tp / (tp + fp)) if tp + fp else None
    npv = float(tn / (tn + fn)) if tn + fn else None
    balanced = (
        float((sensitivity + specificity) / 2.0)
        if sensitivity is not None and specificity is not None
        else None
    )
    output: dict[str, Any] = {
        "rows": int(truth.size),
        "class_0": int(np.sum(truth == 0)),
        "class_1": int(np.sum(truth == 1)),
        "prevalence": float(np.mean(truth)) if truth.size else None,
        "accuracy": float(accuracy_score(truth, predicted)) if truth.size else None,
        "balanced_accuracy": balanced,
        "f1": float(f1_score(truth, predicted, zero_division=0)) if truth.size else None,
        "precision": precision,
        "npv": npv,
        "recall_sensitivity": sensitivity,
        "specificity": specificity,
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else None,
        "mcc": (
            float(matthews_corrcoef(truth, predicted))
            if truth.size and np.unique(truth).size == 2 and np.unique(predicted).size == 2
            else None
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "errors": int(fp + fn),
    }
    if score is not None:
        scores = np.asarray(score, dtype=float)
        output["roc_auc"] = (
            float(roc_auc_score(truth, scores)) if np.unique(truth).size == 2 else None
        )
        output["average_precision"] = (
            float(average_precision_score(truth, scores))
            if np.any(truth == 1)
            else None
        )
    return output


def grouped_fold_ids(frame: pd.DataFrame, target_column: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Assign deterministic stratified folds while keeping lineages indivisible."""

    groups = frame["lineage_group_id"].astype(str).to_numpy()
    y = frame[target_column].astype(int).to_numpy()
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        raise ValueError(f"{target_column} has fewer than two independent lineage groups")
    n_splits = min(5, int(unique_groups.size))
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    fold_ids = np.full(frame.shape[0], -1, dtype=np.int16)
    manifest: list[dict[str, Any]] = []
    dummy = np.zeros((frame.shape[0], 1), dtype=np.int8)
    for fold, (train_index, test_index) in enumerate(splitter.split(dummy, y, groups)):
        train_classes = np.unique(y[train_index])
        if train_classes.size != 2:
            raise ValueError(
                f"Fold {fold} for {target_column} has one training class; "
                "the target is not supportable by grouped validation"
            )
        fold_ids[test_index] = fold
        train_groups = set(groups[train_index])
        test_groups = set(groups[test_index])
        manifest.append(
            {
                "fold": fold,
                "train_rows": int(train_index.size),
                "test_rows": int(test_index.size),
                "train_groups": len(train_groups),
                "test_groups": len(test_groups),
                "group_overlap": len(train_groups.intersection(test_groups)),
                "train_class_0": int(np.sum(y[train_index] == 0)),
                "train_class_1": int(np.sum(y[train_index] == 1)),
                "test_class_0": int(np.sum(y[test_index] == 0)),
                "test_class_1": int(np.sum(y[test_index] == 1)),
                "test_group_ids_json": json.dumps(sorted(test_groups)),
            }
        )
    if np.any(fold_ids < 0):
        raise RuntimeError(f"Not every row received an out-of-fold prediction for {target_column}")
    return fold_ids, manifest


def _group_macro_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    sensitivities: list[float] = []
    specificities: list[float] = []
    accuracies: list[float] = []
    for group in np.unique(groups):
        mask = groups == group
        metrics = metric_record(y_true[mask], y_pred[mask])
        accuracies.append(float(metrics["accuracy"]))
        if metrics["recall_sensitivity"] is not None:
            sensitivities.append(float(metrics["recall_sensitivity"]))
        if metrics["specificity"] is not None:
            specificities.append(float(metrics["specificity"]))
    sensitivity = float(np.mean(sensitivities)) if sensitivities else None
    specificity = float(np.mean(specificities)) if specificities else None
    return {
        "group_macro_accuracy": float(np.mean(accuracies)) if accuracies else None,
        "group_macro_sensitivity": sensitivity,
        "group_macro_specificity": specificity,
        "group_macro_balanced_accuracy": (
            float((sensitivity + specificity) / 2.0)
            if sensitivity is not None and specificity is not None
            else None
        ),
        "groups_with_positive_cases": len(sensitivities),
        "groups_with_negative_cases": len(specificities),
    }


def _cluster_bootstrap_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> tuple[float | None, float | None]:
    """Bootstrap entire lineage groups, not correlated rows."""

    if repetitions <= 0:
        return None, None
    unique = np.unique(groups)
    group_counts: dict[str, np.ndarray] = {}
    for group in unique:
        mask = groups == group
        group_counts[str(group)] = confusion_matrix(
            y_true[mask], y_pred[mask], labels=[0, 1]
        ).ravel()
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(repetitions):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        tn = fp = fn = tp = 0
        for group in sampled:
            current = group_counts[str(group)]
            tn += int(current[0])
            fp += int(current[1])
            fn += int(current[2])
            tp += int(current[3])
        if tn + fp and tp + fn:
            estimates.append(0.5 * (tn / (tn + fp) + tp / (tp + fn)))
    if not estimates:
        return None, None
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def _seed_for(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _fit_oof(
    model: Any,
    features: pd.DataFrame,
    y: np.ndarray,
    fold_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    predictions = np.full(y.size, -1, dtype=np.int8)
    scores = np.full(y.size, np.nan, dtype=float)
    fold_metrics: list[dict[str, Any]] = []
    for fold in sorted(np.unique(fold_ids)):
        train = fold_ids != fold
        test = ~train
        fitted = clone(model)
        started = time.perf_counter()
        fitted.fit(features.loc[train], y[train])
        fit_seconds = time.perf_counter() - started
        predicted = np.asarray(fitted.predict(features.loc[test]), dtype=np.int8)
        decision = _decision_scores(fitted, features.loc[test])
        predictions[test] = predicted
        scores[test] = decision
        fold_metrics.append(
            {
                "fold": int(fold),
                "fit_seconds": float(fit_seconds),
                **metric_record(y[test], predicted, decision),
            }
        )
    if np.any(predictions < 0) or np.any(~np.isfinite(scores)):
        raise RuntimeError("OOF prediction coverage is incomplete")
    return predictions, scores, fold_metrics


def _reliability_grade(groups: int) -> str:
    if groups < 5:
        return "critical: fewer than five independent groups"
    if groups < 15:
        return "fragile: fewer than fifteen independent groups"
    if groups < 30:
        return "limited: fewer than thirty independent groups"
    return "exploratory: adequate group count, but no external clinical test set"


def _baseline_rows(
    target_id: str,
    task: pd.DataFrame,
    target_column: str,
    fold_ids: np.ndarray,
) -> list[dict[str, Any]]:
    y = task[target_column].astype(int).to_numpy()
    groups = task["lineage_group_id"].astype(str).to_numpy()
    output: list[dict[str, Any]] = []

    majority = int(np.mean(y) >= 0.5)
    majority_pred = np.full(y.size, majority, dtype=int)
    output.append({"target_id": target_id, "baseline": "global majority", **metric_record(y, majority_pred)})

    missing = task[feature_keys()].isna().astype(np.int8)
    missing_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    pred, score, _ = _fit_oof(missing_model, missing, y, fold_ids)
    output.append(
        {
            "target_id": target_id,
            "baseline": "feature missingness only",
            **metric_record(y, pred, score),
            **_group_macro_metrics(y, pred, groups),
        }
    )

    if task["dataset_key"].nunique() == 1:
        source_pred = majority_pred
        source_score = np.full(y.size, float(np.mean(y)))
    else:
        source_model = ColumnTransformer(
            [("source", OneHotEncoder(handle_unknown="ignore"), ["dataset_key"])]
        )
        source_pipeline = Pipeline(
            [
                ("source", source_model),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )
        source_pred, source_score, _ = _fit_oof(
            source_pipeline, task[["dataset_key"]], y, fold_ids
        )
    output.append(
        {
            "target_id": target_id,
            "baseline": "dataset identity only",
            **metric_record(y, source_pred, source_score),
            **_group_macro_metrics(y, source_pred, groups),
        }
    )
    return output


def _matrix_integrity(observations: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    keys = feature_keys()
    coordinates = ["dataset_key", "record_id", "timestamp_track", "observation_time_s"]
    coordinate_duplicates = observations.duplicated(coordinates, keep=False)
    feature_hash = pd.util.hash_pandas_object(observations[keys], index=False)
    hashed = pd.DataFrame(
        {
            "feature_hash": feature_hash.astype(str),
            "lineage_group_id": observations["lineage_group_id"].astype(str),
            "row_id": observations["row_id"].astype(str),
        }
    )
    duplicate_groups = (
        hashed.groupby("feature_hash", sort=False)
        .agg(rows=("row_id", "size"), groups=("lineage_group_id", "nunique"))
        .reset_index()
    )
    duplicate_groups = duplicate_groups.loc[duplicate_groups["rows"] > 1]
    cross_group = duplicate_groups.loc[duplicate_groups["groups"] > 1]
    feature_names_suspicious = [
        key
        for key in keys
        if any(token in key.casefold() for token in ("label", "target", "patient", "record", "subject", "fold"))
    ]
    report = {
        "rows": int(observations.shape[0]),
        "columns": int(observations.shape[1]),
        "row_id_unique": bool(observations["row_id"].is_unique),
        "source_coordinate_duplicate_rows": int(coordinate_duplicates.sum()),
        "exact_duplicate_feature_hash_sets": int(duplicate_groups.shape[0]),
        "exact_duplicate_feature_sets_crossing_groups": int(cross_group.shape[0]),
        "lineage_groups": int(observations["lineage_group_id"].nunique()),
        "lineage_groups_crossing_suggested_folds": int(
            (observations.groupby("lineage_group_id")["suggested_cv_fold"].nunique() > 1).sum()
        ),
        "feature_columns": len(keys),
        "feature_names_with_identifier_or_label_tokens": feature_names_suspicious,
        "feature_contract_is_numeric": bool(
            all(pd.api.types.is_numeric_dtype(observations[key]) for key in keys)
        ),
    }
    return report, duplicate_groups


def _feature_association_table(observations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target_id, spec in TARGET_SPECS.items():
        task = observations.loc[observations[spec.key].notna()]
        y_all = task[spec.key].astype(int)
        for key in feature_keys():
            known = task[key].notna()
            y = y_all.loc[known]
            if known.sum() < 20 or y.nunique() != 2 or task.loc[known, key].nunique() < 2:
                auc = None
            else:
                raw_auc = float(roc_auc_score(y, task.loc[known, key]))
                auc = max(raw_auc, 1.0 - raw_auc)
            rows.append(
                {
                    "target_id": target_id,
                    "feature_key": key,
                    "known_rows": int(known.sum()),
                    "class_separation_auc_absolute": auc,
                    "missing_rate_class_0": float(task.loc[y_all.eq(0), key].isna().mean()),
                    "missing_rate_class_1": float(task.loc[y_all.eq(1), key].isna().mean()),
                }
            )
    return pd.DataFrame(rows)


def _inventory_tables(observations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    datasets: list[dict[str, Any]] = []
    for dataset, group in observations.groupby("dataset_key", sort=True):
        row: dict[str, Any] = {
            "dataset_key": dataset,
            "rows": int(group.shape[0]),
            "records": int(group["record_id"].nunique()),
            "subjects": int(group["subject_id"].nunique()),
            "lineage_groups": int(group["lineage_group_id"].nunique()),
            "median_available_features": float(group["available_feature_count"].median()),
        }
        for target_id, spec in TARGET_SPECS.items():
            known = group[spec.key].notna()
            row[f"{target_id}_known"] = int(known.sum())
            row[f"{target_id}_positive"] = int(group.loc[known, spec.key].sum())
        datasets.append(row)

    group_rows: list[dict[str, Any]] = []
    for group_id, group in observations.groupby("lineage_group_id", sort=True):
        row = {
            "lineage_group_id": str(group_id),
            "datasets_json": json.dumps(sorted(group["dataset_key"].astype(str).unique())),
            "records_json": json.dumps(sorted(group["record_id"].astype(str).unique())),
            "subjects_json": json.dumps(sorted(group["subject_id"].astype(str).unique())),
            "rows": int(group.shape[0]),
            "suggested_cv_fold": int(group["suggested_cv_fold"].iloc[0]),
        }
        for target_id, spec in TARGET_SPECS.items():
            known = group[spec.key].notna()
            row[f"{target_id}_known"] = int(known.sum())
            row[f"{target_id}_positive"] = int(group.loc[known, spec.key].sum())
        group_rows.append(row)

    missing_rows: list[dict[str, Any]] = []
    for dataset, group in observations.groupby("dataset_key", sort=True):
        for feature in feature_keys():
            missing_rows.append(
                {
                    "dataset_key": dataset,
                    "feature_key": feature,
                    "rows": int(group.shape[0]),
                    "available": int(group[feature].notna().sum()),
                    "missing_rate": float(group[feature].isna().mean()),
                }
            )
    return pd.DataFrame(datasets), pd.DataFrame(group_rows), pd.DataFrame(missing_rows)


def _write_raw_inventory(project: Path, output: Path) -> dict[str, Any]:
    """Byte-read every local corpus file and retain its digest as evidence."""

    candidates = list((project / "Datasets").rglob("*"))
    config_path = project / "feature_extraction" / "config" / "datasets.toml"
    if config_path.is_file():
        import tomllib

        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
        candidates.extend(Path(str(item["path"])) for item in config.get("recordings", []))
    unique_paths = sorted({path.resolve() for path in candidates if path.is_file()}, key=str)
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    total_bytes = 0
    for index, path in enumerate(unique_paths, start=1):
        try:
            size = int(path.stat().st_size)
            digest = _sha256(path)
            total_bytes += size
            rows.append(
                {
                    "path": str(path),
                    "suffix": path.suffix.casefold(),
                    "bytes": size,
                    "sha256": digest,
                    "read_status": "complete",
                }
            )
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
        if index % 250 == 0:
            print(f"[raw inventory] byte-read {index}/{len(unique_paths)} files", flush=True)
    inventory = pd.DataFrame(rows)
    inventory.to_csv(output / "raw_file_inventory.csv", index=False)
    duplicates = (
        inventory.groupby(["bytes", "sha256"], sort=False)
        .agg(files=("path", "size"), paths_json=("path", lambda value: json.dumps(list(value))))
        .reset_index()
    )
    duplicates = duplicates.loc[duplicates["files"] > 1]
    duplicates.to_csv(output / "raw_duplicate_files.csv", index=False)
    return {
        "files_planned": len(unique_paths),
        "files_byte_read": int(inventory.shape[0]),
        "bytes_byte_read": total_bytes,
        "read_errors": errors,
        "duplicate_content_sets": int(duplicates.shape[0]),
    }


def _permutation_check(
    model: Any,
    features: pd.DataFrame,
    y: np.ndarray,
    fold_ids: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(repetitions):
        predictions = np.full(y.size, -1, dtype=np.int8)
        for fold in sorted(np.unique(fold_ids)):
            train = fold_ids != fold
            test = ~train
            shuffled = rng.permutation(y[train])
            fitted = clone(model)
            fitted.fit(features.loc[train], shuffled)
            predictions[test] = np.asarray(fitted.predict(features.loc[test]), dtype=np.int8)
        values.append(float(balanced_accuracy_score(y, predictions)))
    return {
        "repetitions": repetitions,
        "mean_balanced_accuracy": float(np.mean(values)),
        "minimum_balanced_accuracy": float(np.min(values)),
        "maximum_balanced_accuracy": float(np.max(values)),
        "values_json": json.dumps(values),
    }


def _record_and_case_tables(
    observations: pd.DataFrame,
    selected_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    right_columns = [
        column
        for column in CASE_COLUMNS + feature_keys()
        if column in observations
        and (column == "row_id" or column not in selected_predictions.columns)
    ]
    joined = selected_predictions.merge(
        observations[right_columns],
        on="row_id",
        how="left",
        validate="many_to_one",
    )
    joined["error"] = joined["y_true"].ne(joined["y_pred"])
    joined["error_type"] = np.select(
        [joined["y_true"].eq(0) & joined["y_pred"].eq(1), joined["y_true"].eq(1) & joined["y_pred"].eq(0)],
        ["false_positive", "false_negative"],
        default="correct",
    )
    joined["decision_margin_absolute"] = np.nan
    for _, indices in joined.groupby("experiment_id", sort=False).groups.items():
        values = joined.loc[indices, "decision_score"].astype(float)
        probability_scale = values.between(0.0, 1.0).all()
        joined.loc[indices, "decision_margin_absolute"] = (
            values.sub(0.5).abs() if probability_scale else values.abs()
        )

    record_rows: list[dict[str, Any]] = []
    patient_rows: list[dict[str, Any]] = []
    for columns, destination in (
        (["target_id", "dataset_key", "record_id", "subject_id", "lineage_group_id"], record_rows),
        (["target_id", "subject_id", "lineage_group_id"], patient_rows),
    ):
        for identity, group in joined.groupby(columns, dropna=False, sort=True):
            identity_tuple = identity if isinstance(identity, tuple) else (identity,)
            row = dict(zip(columns, identity_tuple))
            row.update(
                metric_record(
                    group["y_true"].to_numpy(dtype=int),
                    group["y_pred"].to_numpy(dtype=int),
                    group["decision_score"].to_numpy(dtype=float),
                )
            )
            row["mean_available_features"] = float(group["available_feature_count"].mean())
            destination.append(row)

    weak = joined.loc[joined["error"]].copy()
    weak = weak.sort_values(
        ["target_id", "error_type", "decision_margin_absolute"],
        ascending=[True, True, False],
        kind="stable",
    )
    return joined, pd.DataFrame(record_rows), pd.DataFrame(patient_rows), weak


def run_rigorous_audit(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    targets: Sequence[str] = tuple(TARGET_SPECS),
    models: Sequence[str] = tuple(MODEL_SPECS),
    combinations: Sequence[str] | None = None,
    bootstraps: int = DEFAULT_BOOTSTRAPS,
    permutations: int = DEFAULT_PERMUTATIONS,
    raw_inventory: bool = True,
    web_json_path: str | Path | None = None,
) -> dict[str, Any]:
    dataset = Path(dataset_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    observations_path = dataset / "observations_with_labels.csv"
    observations = pd.read_csv(
        observations_path,
        low_memory=False,
        dtype={
            "row_id": "string",
            "dataset_key": "string",
            "record_id": "string",
            "subject_id": "string",
            "lineage_group_id": "string",
        },
    )
    combo_rows = branch_combinations()
    if combinations is not None:
        requested = set(combinations)
        combo_rows = [row for row in combo_rows if row["combination_id"] in requested]

    integrity, duplicate_features = _matrix_integrity(observations)
    duplicate_features.to_csv(output / "duplicate_feature_vectors.csv", index=False)
    dataset_inventory, group_inventory, feature_missingness = _inventory_tables(observations)
    dataset_inventory.to_csv(output / "dataset_inventory.csv", index=False)
    group_inventory.to_csv(output / "lineage_group_inventory.csv", index=False)
    feature_missingness.to_csv(output / "feature_missingness_by_dataset.csv", index=False)
    feature_associations = _feature_association_table(observations)
    feature_associations.to_csv(output / "feature_target_associations.csv", index=False)

    raw_report: dict[str, Any] = {"status": "not requested"}
    if raw_inventory:
        project = Path(__file__).resolve().parents[3]
        raw_report = _write_raw_inventory(project, output)

    result_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    fold_ids_by_target: dict[str, np.ndarray] = {}
    tasks: dict[str, pd.DataFrame] = {}
    all_prediction_path = output / "all_group_oof_predictions.csv.gz"
    total_experiments = len(targets) * len(models) * len(combo_rows)
    completed = 0
    prediction_header = True
    with gzip.open(all_prediction_path, "wt", encoding="utf-8", newline="") as prediction_handle:
        for target_id in targets:
            spec = TARGET_SPECS[target_id]
            task = observations.loc[observations[spec.key].notna()].copy().reset_index(drop=True)
            task[spec.key] = task[spec.key].astype(int)
            tasks[target_id] = task
            fold_ids, fold_manifest = grouped_fold_ids(task, spec.key)
            fold_ids_by_target[target_id] = fold_ids
            for item in fold_manifest:
                split_rows.append({"target_id": target_id, **item})
            y = task[spec.key].to_numpy(dtype=int)
            groups = task["lineage_group_id"].astype(str).to_numpy()
            target_rows.append(
                {
                    "id": target_id,
                    "key": spec.key,
                    "label": spec.label,
                    "negative_label": spec.negative_label,
                    "positive_label": spec.positive_label,
                    "definition": spec.definition,
                    "rows": int(task.shape[0]),
                    "class_0": int(np.sum(y == 0)),
                    "class_1": int(np.sum(y == 1)),
                    "groups": int(np.unique(groups).size),
                    "records": int(task["record_id"].nunique()),
                    "subjects": int(task["subject_id"].nunique()),
                    "folds": int(np.unique(fold_ids).size),
                    "rows_per_group": float(task.shape[0] / np.unique(groups).size),
                    "reliability": _reliability_grade(int(np.unique(groups).size)),
                }
            )
            baseline_rows.extend(_baseline_rows(target_id, task, spec.key, fold_ids))
            for combo in combo_rows:
                columns = list(json.loads(str(combo["feature_columns_json"])))
                features = task[columns].apply(pd.to_numeric, errors="coerce")
                complete_cases = int(features.notna().all(axis=1).sum())
                for model_id in models:
                    completed += 1
                    experiment_id = f"{target_id}__{combo['combination_id']}__{model_id}"
                    pred, score, per_fold = _fit_oof(
                        MODEL_SPECS[model_id].factory(), features, y, fold_ids
                    )
                    metrics = metric_record(y, pred, score)
                    macro = _group_macro_metrics(y, pred, groups)
                    low, high = _cluster_bootstrap_interval(
                        y,
                        pred,
                        groups,
                        repetitions=bootstraps,
                        seed=_seed_for(experiment_id),
                    )
                    result_rows.append(
                        {
                            "experiment_id": experiment_id,
                            "target_id": target_id,
                            "target_column": spec.key,
                            "model_id": model_id,
                            "model_label": MODEL_SPECS[model_id].label,
                            "combination_id": combo["combination_id"],
                            "branches": combo["branches"],
                            "branch_count": int(combo["branch_count"]),
                            "feature_count": int(combo["feature_count"]),
                            "oof_folds": int(np.unique(fold_ids).size),
                            "rows": int(task.shape[0]),
                            "groups": int(np.unique(groups).size),
                            "records": int(task["record_id"].nunique()),
                            "complete_case_rows": complete_cases,
                            "balanced_accuracy_group_bootstrap_ci_low": low,
                            "balanced_accuracy_group_bootstrap_ci_high": high,
                            **metrics,
                            **macro,
                        }
                    )
                    for fold_metric in per_fold:
                        fold_rows.append({"experiment_id": experiment_id, "target_id": target_id, **fold_metric})
                    prediction_frame = pd.DataFrame(
                        {
                            "experiment_id": experiment_id,
                            "target_id": target_id,
                            "model_id": model_id,
                            "combination_id": combo["combination_id"],
                            "row_id": task["row_id"].astype(str),
                            "lineage_group_id": groups,
                            "oof_fold": fold_ids,
                            "y_true": y,
                            "y_pred": pred,
                            "decision_score": score,
                        }
                    )
                    prediction_frame.to_csv(
                        prediction_handle,
                        index=False,
                        header=prediction_header,
                    )
                    prediction_header = False
                    print(
                        f"[{completed}/{total_experiments}] {target_id} | "
                        f"{combo['combination_id']} | {model_id} | "
                        f"group-macro BA={macro['group_macro_balanced_accuracy']:.4f}",
                        flush=True,
                    )

    results = pd.DataFrame(result_rows)
    results["rank_within_target"] = (
        results.groupby("target_id")["group_macro_balanced_accuracy"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    results = results.sort_values(
        ["target_id", "rank_within_target", "balanced_accuracy_group_bootstrap_ci_low"],
        ascending=[True, True, False],
        kind="stable",
    ).reset_index(drop=True)
    selected = (
        results.sort_values(
            ["target_id", "group_macro_balanced_accuracy", "balanced_accuracy_group_bootstrap_ci_low", "balanced_accuracy"],
            ascending=[True, False, False, False],
            kind="stable",
        )
        .groupby("target_id", as_index=False, sort=False)
        .head(1)
        .copy()
    )
    # Checkpoint the expensive OOF evidence before post-hoc permutation fits.
    # A terminal interruption must not force the 375 experiment grid to rerun.
    results.to_csv(output / "group_oof_results.csv", index=False)
    selected.to_csv(output / "selected_experiments.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "fold_metrics.csv", index=False)
    pd.DataFrame(split_rows).to_csv(output / "split_manifest.csv", index=False)
    pd.DataFrame(baseline_rows).to_csv(output / "leakage_and_naive_baselines.csv", index=False)
    selected_ids = set(selected["experiment_id"])

    selected_frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(all_prediction_path, compression="gzip", chunksize=100_000):
        current = chunk.loc[chunk["experiment_id"].isin(selected_ids)]
        if not current.empty:
            selected_frames.append(current)
    selected_predictions = pd.concat(selected_frames, ignore_index=True)
    cases, record_metrics, patient_metrics, weak_cases = _record_and_case_tables(
        observations, selected_predictions
    )

    permutation_rows: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        task = tasks[row.target_id]
        columns = list(
            json.loads(
                next(
                    str(combo["feature_columns_json"])
                    for combo in combo_rows
                    if combo["combination_id"] == row.combination_id
                )
            )
        )
        check = _permutation_check(
            MODEL_SPECS[row.model_id].factory(),
            task[columns].apply(pd.to_numeric, errors="coerce"),
            task[TARGET_SPECS[row.target_id].key].to_numpy(dtype=int),
            fold_ids_by_target[row.target_id],
            repetitions=permutations,
            seed=_seed_for(f"permutation|{row.experiment_id}"),
        )
        permutation_rows.append(
            {
                "target_id": row.target_id,
                "experiment_id": row.experiment_id,
                "observed_group_macro_balanced_accuracy": row.group_macro_balanced_accuracy,
                "observed_row_balanced_accuracy": row.balanced_accuracy,
                **check,
            }
        )

    results.to_csv(output / "group_oof_results.csv", index=False)
    selected.to_csv(output / "selected_experiments.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "fold_metrics.csv", index=False)
    pd.DataFrame(split_rows).to_csv(output / "split_manifest.csv", index=False)
    pd.DataFrame(baseline_rows).to_csv(output / "leakage_and_naive_baselines.csv", index=False)
    pd.DataFrame(permutation_rows).to_csv(output / "permutation_checks.csv", index=False)
    cases.to_csv(output / "selected_case_predictions.csv.gz", index=False, compression="gzip")
    record_metrics.to_csv(output / "record_metrics.csv", index=False)
    patient_metrics.to_csv(output / "patient_metrics.csv", index=False)
    weak_cases.to_csv(output / "weak_cases_all_errors.csv.gz", index=False, compression="gzip")

    weak_records = record_metrics.copy()
    weak_records["ranking_score"] = weak_records["balanced_accuracy"].fillna(weak_records["accuracy"])
    weak_records = weak_records.sort_values(
        ["target_id", "ranking_score", "errors"],
        ascending=[True, True, False],
        kind="stable",
    )
    weak_records.to_csv(output / "weak_records_ranked.csv", index=False)

    validation = json.loads((dataset / "validation_report.json").read_text(encoding="utf-8"))
    web_weak = (
        weak_cases.groupby("target_id", group_keys=False, sort=False)
        .head(100)[
            [
                "target_id", "dataset_key", "record_id", "subject_id",
                "lineage_group_id", "observation_time_s", "y_true", "y_pred",
                "decision_score", "error_type", "available_feature_count",
                "label_beat_symbol_original", "label_quality_consensus_original",
                "label_seizure_phase_derived", "label_noise_condition_original",
            ]
        ]
    )
    payload = {
        "audit_version": AUDIT_VERSION,
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "verdict": (
            "Exploratory out-of-group backtest only. Every labelled row is predicted without "
            "its lineage group in training, but no target has an untouched external clinical cohort."
        ),
        "dataset": {
            "profile": validation.get("profile"),
            "rows": int(observations.shape[0]),
            "records": int(observations["record_id"].nunique()),
            "subjects": int(observations["subject_id"].nunique()),
            "lineage_groups": int(observations["lineage_group_id"].nunique()),
            "features": len(feature_keys()),
            "matrix_sha256": _sha256(observations_path),
            "validation_passed": bool(validation.get("validation_passed")),
        },
        "raw_inventory": raw_report,
        "integrity": integrity,
        "protocol": {
            "split": "deterministic stratified group out-of-fold; 5 folds when >=5 groups, otherwise leave-small-group folds",
            "unit_of_independence": "lineage_group_id (patient/source family)",
            "preprocessing": "refit inside each training fold; no test-row imputation or scaling fit",
            "ranking": "mean of per-group sensitivity and per-group specificity",
            "uncertainty": f"{bootstraps} bootstrap resamples of complete lineage groups",
            "permutation": f"{permutations} training-label permutations for each displayed selected experiment",
            "selection_warning": "The displayed winner is selected on these same OOF results and is not an unbiased final-test estimate.",
        },
        "targets": [_json_safe(row) for row in target_rows],
        "models": [
            {
                "id": key,
                "label": MODEL_SPECS[key].label,
                "family": MODEL_SPECS[key].family,
                "rationale": MODEL_SPECS[key].rationale,
                "reference_title": MODEL_SPECS[key].reference_title,
                "reference_url": MODEL_SPECS[key].reference_url,
                "configuration": dict(MODEL_SPECS[key].configuration),
            }
            for key in models
        ],
        "combinations": [
            {
                "id": row["combination_id"],
                "branches": row["branches"],
                "feature_count": int(row["feature_count"]),
            }
            for row in combo_rows
        ],
        "results": [_json_safe(row) for row in results.to_dict(orient="records")],
        "selected": [_json_safe(row) for row in selected.to_dict(orient="records")],
        "baselines": [_json_safe(row) for row in pd.DataFrame(baseline_rows).to_dict(orient="records")],
        "permutations": [_json_safe(row) for row in permutation_rows],
        "datasets": [_json_safe(row) for row in dataset_inventory.to_dict(orient="records")],
        "records": [_json_safe(row) for row in record_metrics.to_dict(orient="records")],
        "patients": [_json_safe(row) for row in patient_metrics.to_dict(orient="records")],
        "weak_cases": [_json_safe(row) for row in web_weak.to_dict(orient="records")],
        "files": {
            "all_experiment_predictions": all_prediction_path.name,
            "selected_case_predictions": "selected_case_predictions.csv.gz",
            "weak_cases_all_errors": "weak_cases_all_errors.csv.gz",
            "record_metrics": "record_metrics.csv",
            "patient_metrics": "patient_metrics.csv",
            "feature_associations": "feature_target_associations.csv",
            "raw_inventory": "raw_file_inventory.csv" if raw_inventory else None,
        },
    }
    manifest_path = output / "audit_results.json"
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    _write_markdown_report(payload, output / "AUDIT_REPORT.md")
    if web_json_path is not None:
        destination = Path(web_json_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
    return payload


def _write_markdown_report(payload: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# ECG branch matrix: rigorous grouped audit",
        "",
        f"Audit version: `{payload['audit_version']}`  ",
        f"Matrix SHA-256: `{payload['dataset']['matrix_sha256']}`",
        "",
        "## Verdict",
        "",
        str(payload["verdict"]),
        "",
        "The winner for each target was selected on the same out-of-fold matrix. It is shown for failure analysis, not as an unbiased final-test claim.",
        "",
        "## Target evidence",
        "",
        "| Target | Rows | Groups | Reliability | Selected model / branches | Group-macro BA | Row BA | Errors |",
        "|---|---:|---:|---|---|---:|---:|---:|",
    ]
    selected = {row["target_id"]: row for row in payload["selected"]}
    for target in payload["targets"]:
        row = selected[target["id"]]
        lines.append(
            f"| {target['label']} | {target['rows']:,} | {target['groups']} | {target['reliability']} | "
            f"{row['model_label']} / {row['combination_id']} | "
            f"{row['group_macro_balanced_accuracy']:.3f} | {row['balanced_accuracy']:.3f} | {row['errors']:,} |"
        )
    lines.extend(
        [
            "",
            "## Metric calculations",
            "",
            "- Sensitivity = TP / (TP + FN).",
            "- Specificity = TN / (TN + FP).",
            "- Balanced accuracy = (sensitivity + specificity) / 2.",
            "- Precision = TP / (TP + FP).",
            "- Accuracy = (TP + TN) / all rows; it can be misleading under imbalance.",
            "- Group-macro sensitivity/specificity average each rate across independent lineage groups before combining them.",
            "- Confidence limits resample whole lineage groups, never individual correlated beats.",
            "",
            "## Evidence files",
            "",
        ]
    )
    for label, filename in payload["files"].items():
        if filename:
            lines.append(f"- `{filename}` — {label.replace('_', ' ')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run patient/group-out ECG matrix audit.")
    default_dataset = Path(__file__).resolve().parents[2] / "outputs" / "comprehensive_branch_matrix_v1"
    parser.add_argument("--dataset", type=Path, default=default_dataset)
    parser.add_argument("--output", type=Path, default=default_dataset / "rigorous_audit_v2")
    parser.add_argument("--targets", nargs="+", choices=tuple(TARGET_SPECS), default=list(TARGET_SPECS))
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_SPECS), default=list(MODEL_SPECS))
    parser.add_argument("--combinations", nargs="+", default=None)
    parser.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--skip-raw-inventory", action="store_true")
    parser.add_argument("--web-json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_rigorous_audit(
        args.dataset,
        args.output,
        targets=args.targets,
        models=args.models,
        combinations=args.combinations,
        bootstraps=args.bootstraps,
        permutations=args.permutations,
        raw_inventory=not args.skip_raw_inventory,
        web_json_path=args.web_json,
    )
    print(json.dumps({"audit_version": payload["audit_version"], "output": str(args.output), "experiments": len(payload["results"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
