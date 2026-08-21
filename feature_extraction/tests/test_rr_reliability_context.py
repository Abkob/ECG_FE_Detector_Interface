import numpy as np

from ecg_cascade.rr_reliability_context import (
    build_unsw_neurokit_rr_context,
    detector_agreement_fraction,
)


def test_context_architecture_never_replaces_unsw_timestamps():
    unsw = np.array([100, 200, 300])
    neurokit = np.array([102, 198, 301])
    result = build_unsw_neurokit_rr_context(
        unsw,
        neurokit,
        sampling_rate_hz=1000.0,
        support_tolerance_ms=5.0,
        agreement_window_s=1.0,
    )

    assert result.unsw_samples.tolist() == unsw.tolist()
    assert result.rr_intervals["rr_samples"].tolist() == [100, 100]
    assert result.events["unsw_sample"].tolist() == unsw.tolist()
    assert result.events["unsw_supported"].tolist() == [True, True, True]
    assert result.rr_intervals["rr_supported"].tolist() == [True, True]


def test_extra_neurokit_event_inside_interval_makes_rr_unsupported():
    result = build_unsw_neurokit_rr_context(
        np.array([100, 200, 300]),
        np.array([101, 199, 250, 301]),
        sampling_rate_hz=1000.0,
        support_tolerance_ms=5.0,
        agreement_window_s=1.0,
    )

    assert result.events["unsw_supported"].tolist() == [True, True, True]
    assert result.rr_intervals["rr_supported"].tolist() == [True, False]
    assert result.rr_intervals["additional_neurokit_event"].tolist() == [False, True]


def test_agreement_fraction_uses_published_two_count_form():
    score = detector_agreement_fraction(
        np.array([100, 200, 300]),
        np.array([101, 199, 250, 301]),
        tolerance_samples=5,
    )
    assert np.isclose(score, 6 / 7)


def test_multiple_neurokit_events_near_one_unsw_event_are_ambiguous():
    result = build_unsw_neurokit_rr_context(
        np.array([100, 300]),
        np.array([98, 102, 301]),
        sampling_rate_hz=1000.0,
        support_tolerance_ms=5.0,
        agreement_window_s=1.0,
    )

    assert result.events["neurokit_match_count"].tolist() == [2, 1]
    assert result.events["unsw_supported"].tolist() == [False, True]
    assert not bool(result.rr_intervals.iloc[0]["rr_supported"])
