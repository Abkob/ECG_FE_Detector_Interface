import numpy as np

from ecg_cascade.fiducial_fusion import fuse_unsw_neurokit_zhai


def test_fusion_uses_neurokit_then_zhai_then_unsw_without_changing_beat_count():
    result = fuse_unsw_neurokit_zhai(
        np.array([100, 200, 300, 400]),
        np.array([102, 198]),
        np.array([101, 201, 302]),
        sampling_rate_hz=1000.0,
        association_tolerance_ms=5.0,
        zhai_correlation_values=np.array([0.9, 0.8, -0.95]),
    )

    assert result.selected_samples.tolist() == [102, 198, 302, 400]
    assert result.events["selected_source"].tolist() == [
        "neurokit",
        "neurokit",
        "zhai",
        "unsw",
    ]
    assert result.selected_samples.size == 4
    assert result.events.loc[2, "zhai_absolute_correlation"] == 0.95


def test_ambiguous_neurokit_is_rejected_and_unique_zhai_is_used():
    result = fuse_unsw_neurokit_zhai(
        np.array([100, 300]),
        np.array([98, 102, 301]),
        np.array([101, 299]),
        sampling_rate_hz=1000.0,
        association_tolerance_ms=5.0,
    )

    assert result.selected_samples.tolist() == [101, 301]
    assert result.events["selected_source"].tolist() == ["zhai", "neurokit"]
    assert result.events["neurokit_candidates_near_unsw"].tolist() == [2, 1]


def test_candidate_near_two_rapid_primary_events_is_not_reused():
    result = fuse_unsw_neurokit_zhai(
        np.array([100, 180]),
        np.array([140]),
        np.array([], dtype=int),
        sampling_rate_hz=1000.0,
        association_tolerance_ms=50.0,
    )

    assert result.selected_samples.tolist() == [100, 180]
    assert result.events["selected_source"].tolist() == ["unsw", "unsw"]


def test_fusion_exports_source_transition_instead_of_hiding_it():
    result = fuse_unsw_neurokit_zhai(
        np.array([100, 200, 300]),
        np.array([101, 201]),
        np.array([], dtype=int),
        sampling_rate_hz=1000.0,
        association_tolerance_ms=5.0,
    )

    assert result.rr_intervals["source_transition"].tolist() == [False, True]
    assert result.rr_intervals["same_independent_source"].tolist() == [True, False]
