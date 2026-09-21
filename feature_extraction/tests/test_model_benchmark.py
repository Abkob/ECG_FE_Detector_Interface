import pandas as pd

from ecg_cascade.model_benchmark import (
    MODEL_SPECS,
    TARGET_SPECS,
    choose_group_holdout,
)


def test_benchmark_freezes_five_models_and_five_binary_targets():
    assert list(MODEL_SPECS) == [
        "logistic_regression",
        "linear_svm",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
    ]
    assert len(TARGET_SPECS) == 5
    assert all(spec.key.endswith("_binary") for spec in TARGET_SPECS.values())


def test_holdout_selection_never_splits_a_lineage_group():
    frame = pd.DataFrame(
        {
            "lineage_group_id": ["a", "a", "b", "b", "c", "c"],
            "suggested_cv_fold": [0, 0, 1, 1, 2, 2],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )

    test_fold = choose_group_holdout(frame, "target")
    train_groups = set(frame.loc[frame.suggested_cv_fold.ne(test_fold), "lineage_group_id"])
    test_groups = set(frame.loc[frame.suggested_cv_fold.eq(test_fold), "lineage_group_id"])

    assert train_groups.isdisjoint(test_groups)
    assert set(frame.loc[frame.suggested_cv_fold.eq(test_fold), "target"]) == {0, 1}
