"""Agreement measures for expert-versus-detector HRV feature audits."""

from __future__ import annotations

import numpy as np


def lin_concordance_correlation(
    reference: np.ndarray,
    estimate: np.ndarray,
) -> float:
    """Return Lin's concordance correlation coefficient.

    Unlike ordinary correlation, CCC penalizes a systematic location or scale
    difference from the identity line. Non-finite pairs are omitted.
    """

    truth, measured = _finite_pairs(reference, estimate)
    if truth.size < 2:
        return float("nan")
    truth_mean = float(np.mean(truth))
    measured_mean = float(np.mean(measured))
    truth_variance = float(np.var(truth, ddof=0))
    measured_variance = float(np.var(measured, ddof=0))
    covariance = float(
        np.mean((truth - truth_mean) * (measured - measured_mean))
    )
    denominator = (
        truth_variance
        + measured_variance
        + (truth_mean - measured_mean) ** 2
    )
    if denominator == 0:
        return 1.0 if np.array_equal(truth, measured) else float("nan")
    return 2.0 * covariance / denominator


def symmetric_absolute_percentage_error(
    reference: np.ndarray,
    estimate: np.ndarray,
) -> float:
    """Return mean symmetric absolute percentage error as a fraction.

    Pairs for which both values are zero contribute zero. The returned value
    lies from zero to two and is not a probability.
    """

    truth, measured = _finite_pairs(reference, estimate)
    if truth.size == 0:
        return float("nan")
    numerator = 2.0 * np.abs(measured - truth)
    denominator = np.abs(truth) + np.abs(measured)
    terms = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    return float(np.mean(terms))


def threshold_agreement_counts(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float | int]:
    """Compare whether paired measurements exceed the same fixed threshold."""

    truth, measured = _finite_pairs(reference, estimate)
    truth_positive = truth > threshold
    measured_positive = measured > threshold
    tp = int(np.sum(truth_positive & measured_positive))
    fp = int(np.sum(~truth_positive & measured_positive))
    fn = int(np.sum(truth_positive & ~measured_positive))
    tn = int(np.sum(~truth_positive & ~measured_positive))
    sensitivity = _divide(tp, tp + fn)
    ppv = _divide(tp, tp + fp)
    specificity = _divide(tn, tn + fp)
    f1 = _divide(2 * tp, 2 * tp + fp + fn)
    return {
        "n": int(truth.size),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "sensitivity": sensitivity,
        "ppv": ppv,
        "specificity": specificity,
        "f1": f1,
    }


def _finite_pairs(
    reference: np.ndarray,
    estimate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(reference, dtype=np.float64)
    measured = np.asarray(estimate, dtype=np.float64)
    if truth.shape != measured.shape:
        raise ValueError("reference and estimate must have identical shapes")
    finite = np.isfinite(truth) & np.isfinite(measured)
    return truth[finite], measured[finite]


def _divide(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")
