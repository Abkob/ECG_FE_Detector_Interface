import numpy as np

from ecg_cascade.hrv_fidelity import (
    lin_concordance_correlation,
    symmetric_absolute_percentage_error,
    threshold_agreement_counts,
)


def test_lin_ccc_is_one_only_for_identical_nonconstant_values():
    truth = np.array([1.0, 2.0, 4.0, 8.0])
    assert np.isclose(lin_concordance_correlation(truth, truth), 1.0)
    assert lin_concordance_correlation(truth, truth + 2.0) < 1.0


def test_symmetric_percentage_error_handles_joint_zero():
    truth = np.array([0.0, 1.0])
    estimate = np.array([0.0, 2.0])
    assert np.isclose(
        symmetric_absolute_percentage_error(truth, estimate),
        1.0 / 3.0,
    )


def test_threshold_agreement_uses_same_absolute_threshold():
    result = threshold_agreement_counts(
        np.array([1.0, 4.0, 6.0, 9.0]),
        np.array([2.0, 7.0, 3.0, 10.0]),
        threshold=5.0,
    )
    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 1
    assert result["tn"] == 1
    assert np.isclose(result["f1"], 0.5)
