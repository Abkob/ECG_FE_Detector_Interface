import neurokit2 as nk
import numpy as np

from ecg_cascade import RRHRVConfig, run_rr_hrv_array
from ecg_cascade.peaks import _adaptive_energy_peakdetect
from ecg_cascade.reliability import (
    assess_rr_reliability,
    collapse_close_detections,
)
from ecg_cascade.rr_hrv import extract_rr_hrv_windows


def test_250_ms_inclusive_and_strict_rules_differ_at_exact_boundary():
    fs = 360.0
    detection = np.zeros(300)
    detection[[100, 190]] = 10.0  # exactly 90 samples = 250 ms

    inclusive = _adaptive_energy_peakdetect(
        detection,
        fs,
        minimum_delay_ms=250.0,
        minimum_delay_inclusive=True,
        missed_peak_minimum_ms=250.0,
    )
    strict = _adaptive_energy_peakdetect(
        detection,
        fs,
        minimum_delay_ms=250.0,
        minimum_delay_inclusive=False,
        missed_peak_minimum_ms=250.0,
    )

    assert inclusive.tolist() == [100, 190]
    assert strict.tolist() == [100]


def test_close_detection_rule_keeps_higher_oriented_amplitude():
    signal = np.zeros(1000)
    signal[100] = 1.0
    signal[130] = 3.0
    signal[500] = 2.0
    collapsed = collapse_close_detections(
        np.array([100, 130, 500]),
        signal,
        sampling_rate_hz=250,
        exclusion_ms=150,
    )
    assert collapsed.tolist() == [130, 500]


def test_rr_is_supported_only_with_two_supported_boundaries_and_no_extra_secondary():
    fs = 1000.0
    signal = np.zeros(4000)
    primary = np.array([1000, 1800, 2600])
    signal[primary] = 1.0
    secondary = np.array([1005, 1795, 2200, 2605])

    result = assess_rr_reliability(
        primary,
        secondary,
        signal,
        sampling_rate_hz=fs,
        support_tolerance_ms=50,
        close_detection_exclusion_ms=150,
        r_fiducial_refinement_ms=50,
    )

    assert result.primary_events["qrs_supported"].tolist() == [True, True, True]
    assert result.rr_intervals["rr_supported"].tolist() == [True, False]
    assert result.rr_intervals["secondary_events_in_expanded_rr"].tolist() == [2, 3]


def test_hrv_values_are_retained_but_reliability_is_separate():
    fs = 1000.0
    rr = 800.0 + 20.0 * np.sin(np.linspace(0, 8 * np.pi, 120))
    end_time = np.cumsum(rr) / 1000.0
    import pandas as pd

    intervals = pd.DataFrame(
        {
            "start_time_s": np.concatenate([[0.0], end_time[:-1]]),
            "end_time_s": end_time,
            "rr_ms": rr,
            "heart_rate_bpm": 60000.0 / rr,
            "rr_supported": True,
        }
    )
    intervals.loc[50, "rr_supported"] = False

    features = extract_rr_hrv_windows(intervals, window_size=100)

    assert features.loc[99, "feature_defined"]
    assert not features.loc[99, "feature_reliable"]
    assert np.isclose(features.loc[99, "rr_reliability_coverage"], 0.99)
    assert np.isfinite(features.loc[99, "j1_csi_x_slope"])


def test_complete_pipeline_runs_unsw_neurokit_and_named_ablations():
    fs = 250
    ecg = nk.ecg_simulate(
        duration=30,
        sampling_rate=fs,
        heart_rate=70,
        noise=0.01,
        random_state=42,
    )
    result = run_rr_hrv_array(
        ecg,
        fs,
        config=RRHRVConfig(compute_inverted_context=False),
    )

    assert result.selected.primary.method == "khamis2016_unsw"
    assert result.selected.secondary_experimental.parameters["minimum_delay_ms"] == 250.0
    assert result.selected.secondary_experimental.parameters["comparison"] == ">="
    assert result.selected.secondary_baseline.parameters["minimum_delay_ms"] == 300.0
    assert "rr_supported_nk300_baseline" in result.selected.features
    assert "rr_supported_support150_ablation" in result.selected.features
    assert "r_fiducial_physiozoo_positive_sample" in result.selected.primary_events
    assert "r_fiducial_physiozoo_negative_sample" in result.selected.primary_events
    assert "r_fiducial_physiozoo_rqrs_adapted_sample" in result.selected.primary_events
    assert "r_fiducial_rdeco_positive_backward_sample" in result.selected.primary_events
    assert (
        result.selected.fiducial_candidates.physiozoo_rqrs_selected_sign
        in {-1, 1}
    )
    assert not result.polarity_context["automatic_routing_applied"].any()
