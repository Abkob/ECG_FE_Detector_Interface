import numpy as np

from ecg_cascade.delineation_validation import (
    ManualDelineationRecord,
    associate_landmarks_with_anchors,
    evaluate_landmark_samples,
    fixed_qrs_capture_rows,
    match_anchor_indices,
    parse_ludb_annotations,
    parse_qtdb_annotations,
    qrs_polarity_summary,
)


def test_ludb_triplets_preserve_missing_leading_onset():
    samples = [10, 20, 30, 40, 50, 60, 70, 80]
    symbols = ["t", ")", "(", "p", ")", "(", "N", ")"]

    landmarks = parse_ludb_annotations(samples, symbols)

    assert landmarks["t_onset"].size == 0
    assert landmarks["t_peak"].tolist() == [10]
    assert landmarks["t_offset"].tolist() == [20]
    assert landmarks["p_onset"].tolist() == [30]
    assert landmarks["p_peak"].tolist() == [40]
    assert landmarks["qrs_onset"].tolist() == [60]
    assert landmarks["r_peak"].tolist() == [70]


def test_qtdb_num_fields_distinguish_wave_boundaries():
    samples = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
    symbols = ["(", "p", ")", "(", "N", ")", "(", "t", ")", "u", ")"]
    nums = [0, 0, 0, 1, 7, 1, 2, 0, 2, 0, 3]

    landmarks = parse_qtdb_annotations(samples, symbols, nums)

    assert landmarks["p_onset"].tolist() == [10]
    assert landmarks["qrs_onset"].tolist() == [40]
    assert landmarks["r_peak"].tolist() == [50]
    assert landmarks["qrs_offset"].tolist() == [60]
    assert landmarks["t_onset"].tolist() == [70]
    assert landmarks["t_peak"].tolist() == [80]
    assert landmarks["t_offset"].tolist() == [90]


def test_selected_manual_anchors_map_bijectively_to_full_stream():
    full = np.asarray([100, 200, 300, 400])
    selected = np.asarray([102, 398])

    indices = match_anchor_indices(
        full,
        selected,
        sampling_rate_hz=1000.0,
        tolerance_ms=5.0,
    )

    assert indices.tolist() == [0, 3]


def test_landmark_metrics_report_signed_and_absolute_errors():
    reference = np.asarray([100, 200, 300])
    predicted = np.asarray([102, 197, 400])

    metrics, errors = evaluate_landmark_samples(
        predicted,
        reference,
        sampling_rate_hz=1000.0,
        tolerance_ms=10.0,
    )

    assert metrics["reference_count"] == 3
    assert metrics["predicted_count"] == 3
    assert metrics["matched_count"] == 2
    assert np.isclose(metrics["sensitivity"], 2 / 3)
    assert errors.tolist() == [2.0, -3.0]
    assert metrics["median_absolute_error_ms"] == 2.5


def test_phase_association_keeps_p_before_and_t_after_same_anchor():
    anchors = np.asarray([1000, 2000, 3000])

    p_indices = associate_landmarks_with_anchors(
        np.asarray([900, 1900, 2900]),
        anchors,
        landmark="p_peak",
        sampling_rate_hz=1000.0,
    )
    t_indices = associate_landmarks_with_anchors(
        np.asarray([1200, 2200, 3200]),
        anchors,
        landmark="t_peak",
        sampling_rate_hz=1000.0,
    )

    assert p_indices.tolist() == [0, 1, 2]
    assert t_indices.tolist() == [0, 1, 2]


def test_fixed_qrs_capture_requires_both_manual_boundaries_inside_window():
    record = ManualDelineationRecord(
        dataset="fixture",
        record_id="one",
        record_path="fixture",
        lead_name="ii",
        lead_index=0,
        sampling_rate_hz=1000.0,
        samples=np.zeros(3000),
        landmarks={
            "p_onset": np.asarray([], dtype=int),
            "p_peak": np.asarray([], dtype=int),
            "p_offset": np.asarray([], dtype=int),
            "qrs_onset": np.asarray([950, 1930]),
            "r_peak": np.asarray([1000, 2000]),
            "qrs_offset": np.asarray([1050, 2050]),
            "t_onset": np.asarray([], dtype=int),
            "t_peak": np.asarray([], dtype=int),
            "t_offset": np.asarray([], dtype=int),
        },
        anchor_samples=np.asarray([1000, 2000]),
        evaluation_anchor_indices=np.asarray([0, 1]),
        annotation_source="fixture",
    )

    rows = fixed_qrs_capture_rows(record, pre_r_ms=60.0, post_r_ms=60.0)

    assert [row["complete_qrs_captured"] for row in rows] == [True, False]


def test_qrs_polarity_summary_preserves_negative_complexes():
    landmarks = {
        "p_onset": np.asarray([], dtype=int),
        "p_peak": np.asarray([], dtype=int),
        "p_offset": np.asarray([], dtype=int),
        "qrs_onset": np.asarray([950]),
        "r_peak": np.asarray([1000]),
        "qrs_offset": np.asarray([1050]),
        "t_onset": np.asarray([], dtype=int),
        "t_peak": np.asarray([], dtype=int),
        "t_offset": np.asarray([], dtype=int),
    }
    signal = np.zeros(2000)
    signal[950:1051] = -2.0
    record = ManualDelineationRecord(
        dataset="fixture",
        record_id="negative",
        record_path="fixture",
        lead_name="ii",
        lead_index=0,
        sampling_rate_hz=1000.0,
        samples=signal,
        landmarks=landmarks,
        anchor_samples=np.asarray([1000]),
        evaluation_anchor_indices=np.asarray([0]),
        annotation_source="fixture",
    )

    summary = qrs_polarity_summary(record)

    assert summary["negative_dominant_qrs_fraction"] == 1.0
    assert summary["record_polarity_group"] == "negative_or_mixed"
