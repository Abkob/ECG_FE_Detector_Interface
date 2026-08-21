import numpy as np
import pandas as pd

from ecg_cascade.fiducial_fusion import fuse_unsw_neurokit_zhai
from ecg_cascade.interval_consistent_fusion import (
    build_interval_consistent_from_associations,
    fuse_rr_intervals_consistently,
)


def test_interval_consistent_rule_reproduces_user_transition_example():
    fs = 1000.0
    unsw = np.array([1015, 1815, 2615])
    neurokit = np.array([1000, 1800])
    empty = np.array([], dtype=int)
    old = fuse_unsw_neurokit_zhai(
        unsw,
        neurokit,
        empty,
        sampling_rate_hz=fs,
        association_tolerance_ms=50.0,
    )
    new = fuse_rr_intervals_consistently(
        unsw,
        neurokit,
        empty,
        sampling_rate_hz=fs,
        association_tolerance_ms=50.0,
    )
    assert old.rr_intervals["rr_ms"].tolist() == [800.0, 815.0]
    assert new.rr_intervals["rr_ms"].tolist() == [800.0, 800.0]
    assert new.rr_intervals["rr_source"].tolist() == ["neurokit", "unsw"]
    assert new.rr_intervals["same_source_endpoints"].all()


def test_zhai_is_used_only_when_both_boundaries_exist():
    events = pd.DataFrame(
        {
            "event_index": [0, 1, 2],
            "unsw_sample": [100, 200, 300],
            "neurokit_unique_sample": pd.Series([pd.NA, pd.NA, pd.NA], dtype="Int64"),
            "zhai_unique_sample": pd.Series([95, 195, pd.NA], dtype="Int64"),
        }
    )
    result = build_interval_consistent_from_associations(
        events,
        sampling_rate_hz=1000.0,
        association_tolerance_ms=50.0,
    )
    assert result.rr_intervals["rr_source"].tolist() == ["zhai", "unsw"]
    assert result.rr_intervals["rr_ms"].tolist() == [100.0, 100.0]
    assert result.summary()["mixed_endpoint_interval_count"] == 0


def test_component_ablation_can_disable_neurokit_or_zhai():
    events = pd.DataFrame(
        {
            "event_index": [0, 1],
            "unsw_sample": [100, 200],
            "neurokit_unique_sample": pd.Series([98, 198], dtype="Int64"),
            "zhai_unique_sample": pd.Series([96, 196], dtype="Int64"),
        }
    )
    result = build_interval_consistent_from_associations(
        events,
        sampling_rate_hz=1000.0,
        association_tolerance_ms=50.0,
        enable_neurokit=False,
        enable_zhai=True,
    )
    assert result.rr_intervals.loc[0, "rr_source"] == "zhai"
