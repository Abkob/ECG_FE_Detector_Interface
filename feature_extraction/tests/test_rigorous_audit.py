import numpy as np
import pandas as pd

from ecg_cascade.rigorous_audit import grouped_fold_ids, metric_record


def test_grouped_folds_predict_each_row_once_without_group_overlap():
    frame = pd.DataFrame(
        {
            "lineage_group_id": np.repeat(["a", "b", "c", "d", "e"], 4),
            "target": np.tile([0, 0, 1, 1], 5),
        }
    )
    folds, manifest = grouped_fold_ids(frame, "target")

    assert np.all(folds >= 0)
    assert len(np.unique(folds)) == 5
    assert all(row["group_overlap"] == 0 for row in manifest)
    assert frame.assign(fold=folds).groupby("lineage_group_id")["fold"].nunique().eq(1).all()


def test_metric_record_keeps_confusion_counts_and_undefined_rates_explicit():
    result = metric_record(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))

    assert (result["tn"], result["fp"], result["fn"], result["tp"]) == (1, 1, 1, 1)
    assert result["balanced_accuracy"] == 0.5
    assert result["errors"] == 2

    no_positive = metric_record(np.array([0, 0]), np.array([0, 1]))
    assert no_positive["recall_sensitivity"] is None
    assert no_positive["balanced_accuracy"] is None
