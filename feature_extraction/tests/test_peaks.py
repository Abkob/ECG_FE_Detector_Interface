import numpy as np

from ecg_cascade.peaks import compare_peak_sequences
from ecg_cascade.validation import peak_metrics


def test_peak_agreement_matches_within_75_ms_at_256_hz():
    primary = np.array([100, 356, 612, 868])
    comparator = np.array([105, 350, 620, 1100])

    result = compare_peak_sequences(
        primary,
        comparator,
        sampling_rate_hz=256,
        tolerance_ms=75,
    )

    assert result.matched_primary_samples.tolist() == [100, 356, 612]
    assert result.unmatched_primary_samples.tolist() == [868]
    assert result.unmatched_comparator_samples.tolist() == [1100]
    assert result.agreement_f1 == 0.75
    assert result.unmatched_primary_fraction == 0.25
    assert result.unmatched_comparator_fraction == 0.25


def test_empty_peak_sequences_are_explicitly_undefined():
    result = compare_peak_sequences(
        np.array([], dtype=int),
        np.array([], dtype=int),
        sampling_rate_hz=256,
    )
    assert np.isnan(result.agreement_f1)
    assert np.isnan(result.median_timing_difference_ms)


def test_peak_metrics_report_full_matched_timing_distribution():
    reference = np.array([1000, 2000, 3000, 4000])
    detected = np.array([990, 2000, 3020, 4040, 5000])
    result = peak_metrics(
        detected,
        reference,
        sampling_rate_hz=1000.0,
        tolerance_ms=75.0,
    )

    errors = np.array([-10.0, 0.0, 20.0, 40.0])
    assert result["tp"] == 4
    assert result["fp"] == 1
    assert result["fn"] == 0
    assert np.isclose(result["mean_absolute_timing_ms"], np.mean(np.abs(errors)))
    assert np.isclose(result["median_absolute_timing_ms"], 15.0)
    assert np.isclose(result["maximum_absolute_timing_ms"], 40.0)
    assert np.isclose(result["mean_signed_timing_ms"], np.mean(errors))
    assert np.isclose(
        result["signed_timing_standard_deviation_ms"], np.std(errors, ddof=0)
    )
