import json

import numpy as np
import pandas as pd

from ecg_cascade.comprehensive_dataset import (
    AAMI_BEAT_CLASS,
    _apply_binary_targets,
    _beat_labels_at_time,
    _complete_record_windows,
    _interval_at_time,
    _label_columns,
    _label_dictionary,
    _merge_ranges,
    _seizure_labels_at_time,
    branch_combinations,
    branch_feature_map,
    feature_keys,
    stable_group_fold,
)


def test_all_52_features_are_assigned_once_to_15_branch_combinations():
    keys = feature_keys()
    mapping = branch_feature_map()

    assert len(keys) == 52
    assert set(mapping) == {"B1", "B2", "B3", "B4"}
    assert sorted(key for values in mapping.values() for key in values) == sorted(keys)
    assert len(branch_combinations()) == 15
    assert branch_combinations()[-1]["feature_count"] == 52


def test_exact_beat_symbol_is_preserved_beside_derived_aami_class():
    labels = _beat_labels_at_time(
        np.asarray([100, 200, 300, 400, 500]),
        np.asarray(["N", "N", "A", "V", "F"], dtype=object),
        time_s=0.5,
        sampling_rate_hz=1000.0,
    )

    assert labels["label_beat_symbol_original"] == "F"
    assert labels["label_beat_aami_superclass_derived"] == AAMI_BEAT_CLASS["F"]
    assert json.loads(labels["label_beat_context5_original_json"]) == [
        "N",
        "N",
        "A",
        "V",
        "F",
    ]


def test_quality_interval_lookup_respects_inclusive_source_endpoint():
    frame = pd.DataFrame(
        {
            "start_s": [0.0, 10.0],
            "end_s": [10.0, 20.0],
            "label_original": ["1", "2"],
        }
    )

    assert _interval_at_time(frame, 5.0)["label_original"] == "1"
    assert _interval_at_time(frame, 15.0)["label_original"] == "2"
    assert _interval_at_time(frame, 21.0) is None


def test_seizure_original_interval_and_derived_phase_stay_separate():
    intervals = ((100.0, 120.0),)

    ictal = _seizure_labels_at_time(intervals, 110.0)
    preictal = _seizure_labels_at_time(intervals, 90.0)

    assert ictal["label_seizure_binary"] == "ictal"
    assert json.loads(ictal["label_seizure_interval_original_json"]) == [100.0, 120.0]
    assert preictal["label_seizure_binary"] == "non_ictal"
    assert preictal["label_seizure_phase_derived"] == "preictal_5min"


def test_ranges_merge_without_losing_selection_reasons():
    merged = _merge_ranges(
        [(0.0, 180.0, "class 1"), (100.0, 280.0, "class 2"), (500.0, 600.0, "class 3")]
    )

    assert merged == [
        (0.0, 280.0, ("class 1", "class 2")),
        (500.0, 600.0, ("class 3",)),
    ]


def test_group_fold_is_deterministic_and_group_based():
    first = stable_group_fold("mitdb:118")
    second = stable_group_fold("mitdb:118")

    assert first == second
    assert 0 <= first < 5


def test_prsa_context_with_fewer_than_two_peaks_is_explicitly_empty():
    from ecg_cascade.prsa_bprsa import extract_prsa_bprsa_features

    result = extract_prsa_bprsa_features(
        np.zeros(1000, dtype=float),
        np.array([500], dtype=int),
        100.0,
    )

    assert result.features.empty
    assert result.parameters["insufficient_peak_context"] is True


def test_label_dictionary_documents_every_label_side_column():
    dictionary = _label_dictionary()

    assert dictionary["label_column"].tolist() == _label_columns()
    assert dictionary["label_column"].is_unique


def test_binary_targets_preserve_unknowns_and_source_semantics():
    frame = pd.DataFrame(
        {
            "label_quality_trailing_10s_pure": [True, True, True, np.nan, np.nan],
            "label_quality_consensus_original": ["1", "2", "3", None, None],
            "label_noise_active": [None, None, None, False, True],
            "label_seizure_binary": [None, "ictal", "non_ictal", None, None],
            "label_beat_aami_superclass_derived": ["N", "V", None, None, None],
        }
    )

    result = _apply_binary_targets(frame)

    assert result["target_artifact_binary"].tolist() == [0, 1, 1, 0, 1]
    assert result["target_signal_unusable_binary"].iloc[0] == 0
    assert pd.isna(result["target_signal_unusable_binary"].iloc[1])
    assert result["target_signal_unusable_binary"].iloc[2] == 1
    assert result["target_seizure_binary"].tolist()[1:3] == [1, 0]
    assert result["target_abnormal_beat_binary"].tolist()[:2] == [0, 1]
    assert result["target_noise_active_binary"].tolist()[3:] == [0, 1]


def test_exhaustive_windows_cover_the_record_without_a_short_tail():
    windows = _complete_record_windows(505.0, 180.0)

    assert windows == [(0.0, 180.0), (180.0, 360.0), (360.0, 505.0)]
    assert windows[0][0] == 0.0
    assert windows[-1][1] == 505.0
    assert all(left[1] == right[0] for left, right in zip(windows, windows[1:]))
