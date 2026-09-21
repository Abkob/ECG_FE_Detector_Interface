"""Grouped binary-classifier benchmark for every ECG branch combination.

This module consumes the comprehensive dataset artifact.  It never derives
features from labels and never random-splits rows: a complete lineage group is
held out according to the dataset's stable fold assignment.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .comprehensive_dataset import _apply_binary_targets, branch_combinations, feature_keys


BENCHMARK_VERSION = "1.0.0"
RANDOM_SEED = 20260903


@dataclass(frozen=True)
class TargetSpec:
    key: str
    label: str
    negative_label: str
    positive_label: str
    definition: str
    strata_column: str | None = None


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    family: str
    rationale: str
    reference_title: str
    reference_url: str
    configuration: Mapping[str, Any]
    factory: Callable[[], BaseEstimator]


TARGET_SPECS: dict[str, TargetSpec] = {
    "artifact": TargetSpec(
        key="target_artifact_binary",
        label="Artifact / degraded signal",
        negative_label="clean / inactive (0)",
        positive_label="degraded / active (1)",
        definition=(
            "BUT-QDB pure consensus class 1=0 and class 2/3=1, combined with "
            "NSTDB official noise-inactive=0 and noise-active=1."
        ),
        strata_column="target_artifact_source_derived",
    ),
    "seizure": TargetSpec(
        key="target_seizure_binary",
        label="Ictal seizure interval",
        negative_label="non-ictal (0)",
        positive_label="ictal (1)",
        definition="1 only inside a configured original expert seizure interval; 0 outside.",
    ),
    "abnormal_beat": TargetSpec(
        key="target_abnormal_beat_binary",
        label="Non-normal beat",
        negative_label="AAMI N (0)",
        positive_label="AAMI S/V/F/Q (1)",
        definition="0 for AAMI N and 1 for S, V, F, or Q; exact WFDB symbols remain preserved.",
    ),
    "noise_active": TargetSpec(
        key="target_noise_active_binary",
        label="NSTDB noise active",
        negative_label="inactive (0)",
        positive_label="active (1)",
        definition="Official NSTDB alternating electrode-motion noise schedule.",
    ),
    "signal_unusable": TargetSpec(
        key="target_signal_unusable_binary",
        label="Strict unusable signal",
        negative_label="BUT-QDB class 1 (0)",
        positive_label="BUT-QDB class 3 (1)",
        definition="Pure BUT-QDB consensus class 1 versus class 3; class 2 is excluded.",
    ),
}


def _imputer() -> SimpleImputer:
    return SimpleImputer(
        strategy="median",
        add_indicator=True,
        keep_empty_features=True,
    )


MODEL_SPECS: dict[str, ModelSpec] = {
    "logistic_regression": ModelSpec(
        key="logistic_regression",
        label="Regularized logistic regression",
        family="linear probability baseline",
        rationale="Interpretable linear baseline with balanced class weights.",
        reference_title="scikit-learn LogisticRegression",
        reference_url="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html",
        configuration={
            "l1_ratio": 0.0,
            "C": 1.0,
            "solver": "liblinear",
            "class_weight": "balanced",
            "max_iter": 2000,
        },
        factory=lambda: Pipeline(
            [
                ("imputer", _imputer()),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        l1_ratio=0.0,
                        C=1.0,
                        solver="liblinear",
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
    ),
    "linear_svm": ModelSpec(
        key="linear_svm",
        label="Linear support-vector machine",
        family="maximum-margin linear",
        rationale="Strong high-dimensional margin baseline with balanced class weights.",
        reference_title="Cortes & Vapnik, Support-vector networks (1995)",
        reference_url="https://doi.org/10.1007/BF00994018",
        configuration={
            "C": 1.0,
            "loss": "squared_hinge",
            "class_weight": "balanced",
            "max_iter": 10000,
        },
        factory=lambda: Pipeline(
            [
                ("imputer", _imputer()),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LinearSVC(
                        C=1.0,
                        loss="squared_hinge",
                        class_weight="balanced",
                        max_iter=10000,
                        dual="auto",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
    ),
    "random_forest": ModelSpec(
        key="random_forest",
        label="Random forest",
        family="bagged randomized trees",
        rationale="Nonlinear interaction model robust to scaling and mixed feature behavior.",
        reference_title="Breiman, Random Forests (2001)",
        reference_url="https://doi.org/10.1023/A:1010933404324",
        configuration={
            "n_estimators": 120,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
        },
        factory=lambda: Pipeline(
            [
                ("imputer", _imputer()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=120,
                        min_samples_leaf=2,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
    ),
    "extra_trees": ModelSpec(
        key="extra_trees",
        label="Extremely randomized trees",
        family="fully randomized tree ensemble",
        rationale="Tests whether stronger split randomization improves generalization and speed.",
        reference_title="Geurts, Ernst & Wehenkel, Extremely randomized trees (2006)",
        reference_url="https://doi.org/10.1007/s10994-006-6226-1",
        configuration={
            "n_estimators": 120,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "class_weight": "balanced",
        },
        factory=lambda: Pipeline(
            [
                ("imputer", _imputer()),
                (
                    "classifier",
                    ExtraTreesClassifier(
                        n_estimators=120,
                        min_samples_leaf=2,
                        max_features="sqrt",
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
    ),
    "hist_gradient_boosting": ModelSpec(
        key="hist_gradient_boosting",
        label="Histogram gradient boosting",
        family="boosted decision trees",
        rationale="Efficient nonlinear boosting for larger tabular matrices with native NaN routing.",
        reference_title="Friedman, Greedy function approximation (2001)",
        reference_url="https://doi.org/10.1214/aos/1013203451",
        configuration={
            "learning_rate": 0.06,
            "max_iter": 160,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "l2_regularization": 1.0,
            "class_weight": "balanced",
        },
        factory=lambda: HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=160,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            class_weight="balanced",
            early_stopping=False,
            random_state=RANDOM_SEED,
        ),
    ),
}


def choose_group_holdout(
    frame: pd.DataFrame,
    target_column: str,
    *,
    strata_column: str | None = None,
) -> int:
    """Choose one stable fold with both classes in train and test."""

    candidates: list[tuple[float, int, int]] = []
    total = int(frame.shape[0])
    required_strata = (
        set(frame[strata_column].dropna().astype(str).unique())
        if strata_column is not None
        else set()
    )
    for fold in sorted(int(value) for value in frame["suggested_cv_fold"].unique()):
        test = frame.loc[frame["suggested_cv_fold"].eq(fold), target_column].astype(int)
        train = frame.loc[frame["suggested_cv_fold"].ne(fold), target_column].astype(int)
        if set(test.unique()) != {0, 1} or set(train.unique()) != {0, 1}:
            continue
        if required_strata:
            test_strata = set(
                frame.loc[
                    frame["suggested_cv_fold"].eq(fold), strata_column
                ].dropna().astype(str)
            )
            train_strata = set(
                frame.loc[
                    frame["suggested_cv_fold"].ne(fold), strata_column
                ].dropna().astype(str)
            )
            if test_strata != required_strata or train_strata != required_strata:
                continue
        test_counts = test.value_counts()
        minority = int(min(test_counts.get(0, 0), test_counts.get(1, 0)))
        test_fraction = len(test) / total
        # Prefer a useful number of minority examples, then a conventional ~20% holdout.
        score = float(minority) - abs(test_fraction - 0.2)
        candidates.append((score, -fold, fold))
    if not candidates:
        raise ValueError(
            f"No suggested_cv_fold leaves both classes in train and test for {target_column}"
        )
    return max(candidates)[2]


def _decision_scores(model: BaseEstimator, features: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        values = np.asarray(model.predict_proba(features), dtype=float)
        return values[:, 1]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(features), dtype=float).reshape(-1)
    return np.asarray(model.predict(features), dtype=float).reshape(-1)


def _metric_record(y_true: np.ndarray, y_pred: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if tn + fp else None
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, score)),
        "average_precision": float(average_precision_score(y_true, score)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": specificity,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def run_benchmark(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    targets: Sequence[str] = tuple(TARGET_SPECS),
    models: Sequence[str] = tuple(MODEL_SPECS),
    combinations: Sequence[str] | None = None,
    keep_predictions: bool = True,
    web_json_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run all requested model/target/branch experiments and persist results."""

    dataset = Path(dataset_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    requested_targets = tuple(dict.fromkeys(targets))
    requested_models = tuple(dict.fromkeys(models))
    unknown_targets = set(requested_targets).difference(TARGET_SPECS)
    unknown_models = set(requested_models).difference(MODEL_SPECS)
    if unknown_targets:
        raise ValueError(f"Unknown benchmark targets: {sorted(unknown_targets)}")
    if unknown_models:
        raise ValueError(f"Unknown benchmark models: {sorted(unknown_models)}")

    observations_path = dataset / "observations_with_labels.csv"
    observations = pd.read_csv(
        observations_path,
        low_memory=False,
        dtype={
            "row_id": "string",
            "dataset_key": "string",
            "record_id": "string",
            "lineage_group_id": "string",
        },
    )
    if any(spec.key not in observations for spec in TARGET_SPECS.values()):
        observations = _apply_binary_targets(observations)
    combo_rows = branch_combinations()
    if combinations is not None:
        requested_combinations = set(combinations)
        known_combinations = {row["combination_id"] for row in combo_rows}
        unknown = requested_combinations.difference(known_combinations)
        if unknown:
            raise ValueError(f"Unknown branch combinations: {sorted(unknown)}")
        combo_rows = [
            row for row in combo_rows if row["combination_id"] in requested_combinations
        ]

    result_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    split_rows: list[dict[str, Any]] = []
    target_summaries: list[dict[str, Any]] = []
    experiment_count = len(requested_targets) * len(combo_rows) * len(requested_models)
    experiment_index = 0

    for target_id in requested_targets:
        target_spec = TARGET_SPECS[target_id]
        task = observations.loc[
            observations[target_spec.key].notna()
        ].copy()
        task[target_spec.key] = task[target_spec.key].astype(int)
        test_fold = choose_group_holdout(
            task,
            target_spec.key,
            strata_column=target_spec.strata_column,
        )
        train_mask = task["suggested_cv_fold"].ne(test_fold)
        test_mask = ~train_mask
        train_groups = set(task.loc[train_mask, "lineage_group_id"].astype(str))
        test_groups = set(task.loc[test_mask, "lineage_group_id"].astype(str))
        if train_groups.intersection(test_groups):
            raise RuntimeError(f"Lineage leakage detected for target {target_id}")
        target_summaries.append(
            {
                "id": target_id,
                "key": target_spec.key,
                "label": target_spec.label,
                "negative_label": target_spec.negative_label,
                "positive_label": target_spec.positive_label,
                "definition": target_spec.definition,
                "rows": int(task.shape[0]),
                "class_0": int(task[target_spec.key].eq(0).sum()),
                "class_1": int(task[target_spec.key].eq(1).sum()),
                "train_rows": int(train_mask.sum()),
                "test_rows": int(test_mask.sum()),
                "train_groups": len(train_groups),
                "test_groups": len(test_groups),
                "test_fold": test_fold,
            }
        )
        for group_id, group in task.groupby("lineage_group_id", sort=True):
            split_rows.append(
                {
                    "target_id": target_id,
                    "target_column": target_spec.key,
                    "lineage_group_id": str(group_id),
                    "suggested_cv_fold": int(group["suggested_cv_fold"].iloc[0]),
                    "split": "test" if int(group["suggested_cv_fold"].iloc[0]) == test_fold else "train",
                    "rows": int(group.shape[0]),
                    "class_0": int(group[target_spec.key].eq(0).sum()),
                    "class_1": int(group[target_spec.key].eq(1).sum()),
                }
            )

        y_train = task.loc[train_mask, target_spec.key].to_numpy(dtype=int)
        y_test = task.loc[test_mask, target_spec.key].to_numpy(dtype=int)
        for combo in combo_rows:
            columns = list(json.loads(str(combo["feature_columns_json"])))
            features = task[columns].apply(pd.to_numeric, errors="coerce")
            x_train = features.loc[train_mask]
            x_test = features.loc[test_mask]
            available_counts = features.notna().sum(axis=1)
            for model_id in requested_models:
                experiment_index += 1
                model_spec = MODEL_SPECS[model_id]
                model = model_spec.factory()
                started = time.perf_counter()
                model.fit(x_train, y_train)
                fit_seconds = time.perf_counter() - started
                prediction_started = time.perf_counter()
                y_pred = np.asarray(model.predict(x_test), dtype=int)
                scores = _decision_scores(model, x_test)
                predict_seconds = time.perf_counter() - prediction_started
                metrics = _metric_record(y_test, y_pred, scores)
                result_rows.append(
                    {
                        "experiment_id": f"{target_id}__{combo['combination_id']}__{model_id}",
                        "target_id": target_id,
                        "target_column": target_spec.key,
                        "model_id": model_id,
                        "model_label": model_spec.label,
                        "combination_id": combo["combination_id"],
                        "branches": combo["branches"],
                        "branch_count": int(combo["branch_count"]),
                        "feature_count": int(combo["feature_count"]),
                        "train_rows": int(train_mask.sum()),
                        "test_rows": int(test_mask.sum()),
                        "train_groups": len(train_groups),
                        "test_groups": len(test_groups),
                        "test_fold": test_fold,
                        "class_1_prevalence_train": float(y_train.mean()),
                        "class_1_prevalence_test": float(y_test.mean()),
                        "complete_case_rows": int(available_counts.eq(len(columns)).sum()),
                        "median_available_features": float(available_counts.median()),
                        "fit_seconds": float(fit_seconds),
                        "predict_seconds": float(predict_seconds),
                        **metrics,
                    }
                )
                if keep_predictions:
                    prediction_frames.append(
                        pd.DataFrame(
                            {
                                "experiment_id": result_rows[-1]["experiment_id"],
                                "target_id": target_id,
                                "target_column": target_spec.key,
                                "model_id": model_id,
                                "combination_id": combo["combination_id"],
                                "row_id": task.loc[test_mask, "row_id"].astype(str).to_numpy(),
                                "lineage_group_id": task.loc[
                                    test_mask, "lineage_group_id"
                                ].astype(str).to_numpy(),
                                "y_true": y_test,
                                "y_pred": y_pred,
                                "decision_score": scores,
                            }
                        )
                    )
                print(
                    f"[{experiment_index}/{experiment_count}] {target_id} | "
                    f"{combo['combination_id']} | {model_id} | "
                    f"balanced_accuracy={metrics['balanced_accuracy']:.4f}",
                    flush=True,
                )

    results = pd.DataFrame(result_rows)
    results["rank_within_target_combination"] = (
        results.groupby(["target_id", "combination_id"])["balanced_accuracy"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    results = results.sort_values(
        ["target_id", "branch_count", "combination_id", "rank_within_target_combination"],
        kind="stable",
    ).reset_index(drop=True)
    leaderboard = results.loc[results["rank_within_target_combination"].eq(1)].copy()
    leaderboard = leaderboard.sort_values(
        ["target_id", "balanced_accuracy", "roc_auc"],
        ascending=[True, False, False],
        kind="stable",
    )

    results_path = output / "benchmark_results.csv"
    leaderboard_path = output / "leaderboard.csv"
    splits_path = output / "split_manifest.csv"
    predictions_path = output / "holdout_predictions.csv.gz"
    results.to_csv(results_path, index=False)
    leaderboard.to_csv(leaderboard_path, index=False)
    pd.DataFrame(split_rows).to_csv(splits_path, index=False)
    if keep_predictions:
        predictions = pd.concat(prediction_frames, ignore_index=True)
        with gzip.open(predictions_path, "wt", encoding="utf-8", newline="") as handle:
            predictions.to_csv(handle, index=False)

    validation_report = json.loads(
        (dataset / "validation_report.json").read_text(encoding="utf-8")
    )
    compact_results = [
        _json_safe(record)
        for record in results.to_dict(orient="records")
    ]
    payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "dataset": {
            "profile": validation_report.get("profile", "validation"),
            "rows": int(observations.shape[0]),
            "feature_columns": len(feature_keys()),
            "label_columns": int(validation_report["label_columns"]),
            "branch_combinations": len(combo_rows),
            "datasets": [
                {
                    "key": str(key),
                    "rows": int(group.shape[0]),
                    "records": int(group["record_id"].nunique()),
                }
                for key, group in observations.groupby("dataset_key", sort=True)
            ],
            "observations_sha256": _sha256(observations_path),
            "validation_passed": bool(validation_report["validation_passed"]),
        },
        "protocol": {
            "split": "one deterministic lineage-grouped holdout per target",
            "selection": (
                "test suggested_cv_fold maximizes minority-class test examples while "
                "requiring both classes in train and test; composite targets also "
                "retain every contributing label source on both sides"
            ),
            "preprocessing": (
                "training-fold median imputation plus missing indicators for linear and "
                "randomized-tree models; native NaN routing for histogram boosting"
            ),
            "ranking_metric": "balanced_accuracy",
            "random_seed": RANDOM_SEED,
            "warning": (
                "Exploratory engineering benchmark only; no independent clinical test set, "
                "patient-level prospective validation, or diagnostic claim."
            ),
        },
        "targets": target_summaries,
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
            for key in requested_models
        ],
        "combinations": [
            {
                "id": row["combination_id"],
                "branches": row["branches"],
                "branch_count": int(row["branch_count"]),
                "feature_count": int(row["feature_count"]),
            }
            for row in combo_rows
        ],
        "results": compact_results,
        "files": {
            "results": results_path.name,
            "leaderboard": leaderboard_path.name,
            "splits": splits_path.name,
            "predictions": predictions_path.name if keep_predictions else None,
        },
    }
    manifest_path = output / "benchmark_manifest.json"
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    if web_json_path is not None:
        destination = Path(web_json_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if pd.isna(value):
        return None
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark five classifiers on all 15 ECG branch combinations."
    )
    default_dataset = (
        Path(__file__).resolve().parents[2]
        / "outputs"
        / "comprehensive_branch_matrix_v1"
    )
    parser.add_argument("--dataset", type=Path, default=default_dataset)
    parser.add_argument(
        "--output", type=Path, default=default_dataset / "model_benchmark"
    )
    parser.add_argument("--targets", nargs="+", choices=tuple(TARGET_SPECS), default=list(TARGET_SPECS))
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_SPECS), default=list(MODEL_SPECS))
    parser.add_argument("--combinations", nargs="+", default=None)
    parser.add_argument("--no-predictions", action="store_true")
    parser.add_argument("--web-json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_benchmark(
        args.dataset,
        args.output,
        targets=args.targets,
        models=args.models,
        combinations=args.combinations,
        keep_predictions=not args.no_predictions,
        web_json_path=args.web_json,
    )
    print(
        json.dumps(
            {
                "benchmark_version": payload["benchmark_version"],
                "experiments": len(payload["results"]),
                "targets": len(payload["targets"]),
                "models": len(payload["models"]),
                "combinations": len(payload["combinations"]),
                "output": str(Path(args.output).expanduser().resolve()),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
