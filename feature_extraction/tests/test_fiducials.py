import numpy as np

from ecg_cascade.fiducials import (
    physiozoo_qrs_adjust,
    physiozoo_rqrs_polarity_adjust_from_candidates,
    rdeco_final_localization,
)


def test_physiozoo_qrs_adjust_uses_explicit_positive_or_negative_sign():
    fs = 1000.0
    ecg = np.zeros(500)
    ecg[190] = -4.0
    ecg[205] = 2.0
    candidates = np.array([200])

    positive = physiozoo_qrs_adjust(
        candidates, ecg, sampling_rate_hz=fs, input_sign=1, tolerance_s=0.05
    )
    negative = physiozoo_qrs_adjust(
        candidates, ecg, sampling_rate_hz=fs, input_sign=-1, tolerance_s=0.05
    )

    assert positive.tolist() == [205]
    assert negative.tolist() == [190]


def test_physiozoo_qrs_adjust_search_is_inclusive():
    fs = 1000.0
    ecg = np.zeros(500)
    ecg[250] = 5.0
    adjusted = physiozoo_qrs_adjust(
        np.array([200]),
        ecg,
        sampling_rate_hz=fs,
        input_sign=1,
        tolerance_s=0.05,
    )
    assert adjusted.tolist() == [250]


def test_rqrs_rule_selects_one_global_polarity():
    fs = 1000.0
    ecg = np.zeros(600)
    candidates = np.array([100, 300, 500])
    ecg[candidates] = -0.2
    ecg[candidates + 10] = -3.0
    ecg[candidates + 20] = 0.5

    adjusted, sign, delta_positive, delta_negative = (
        physiozoo_rqrs_polarity_adjust_from_candidates(
            candidates, ecg, sampling_rate_hz=fs, forward_window_s=0.05
        )
    )

    assert sign == -1
    assert adjusted.tolist() == [110, 310, 510]
    assert delta_negative > delta_positive


def test_rdeco_localization_is_positive_after_optional_global_inversion():
    fs = 1000.0
    ecg = np.zeros(500)
    ecg[180] = -5.0
    ecg[195] = 2.0
    candidates = np.array([200])

    positive = rdeco_final_localization(
        candidates, ecg, sampling_rate_hz=fs, inverted=False
    )
    negative = rdeco_final_localization(
        candidates, ecg, sampling_rate_hz=fs, inverted=True
    )

    assert positive.tolist() == [195]
    assert negative.tolist() == [180]
