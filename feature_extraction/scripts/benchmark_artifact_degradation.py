"""Controlled HRV and morphology degradation benchmark on MIT--BIH NSTDB.

The benchmark uses the clean MIT--BIH records 118 and 119 and their NSTDB
electrode-motion-noise copies at 24, 18, 12, 6, 0, and -6 dB.  Only feature
windows fully contained in the official two-minute noisy intervals are scored.

This is a measurement-fidelity benchmark.  It does not evaluate seizure
detection or train an artifact classifier.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecg_cascade.hrv_fidelity import (  # noqa: E402
    lin_concordance_correlation,
    symmetric_absolute_percentage_error,
)
from ecg_cascade.morphology import extract_varon_morphology  # noqa: E402
from ecg_cascade.morphology_validation import (  # noqa: E402
    audit_morphology_timestamp_fidelity,
)
from ecg_cascade.peaks import (  # noqa: E402
    compare_peak_sequences,
    detect_neurokit_gradient,
    detect_unsw,
)
from ecg_cascade.rr_hrv import extract_rr_hrv_features  # noqa: E402
from ecg_cascade.validation import match_peaks_to_reference  # noqa: E402
from ecg_cascade.wfdb_io import load_wfdb_beat_annotations  # noqa: E402
from ecg_cascade.zhai import detect_zhai_template  # noqa: E402


SOURCES = ("118", "119")
SNR_SUFFIXES: tuple[tuple[int, str], ...] = (
    (24, "e24"),
    (18, "e18"),
    (12, "e12"),
    (6, "e06"),
    (0, "e00"),
    (-6, "e_6"),
)
CONDITIONS: tuple[tuple[str, float | None, str | None], ...] = (
    ("clean", None, None),
    *((f"{snr:+d} dB", float(snr), suffix) for snr, suffix in SNR_SUFFIXES),
)
TRACKS = ("unsw", "neurokit", "zhai")
MORPH_TRACKS = ("expert_anchor_oracle", *TRACKS)
HRV_FEATURES = {
    "sd1_raw_ms": "SD1 raw",
    "sd2_raw_ms": "SD2 raw",
    "csi100": "CSI100",
    "sd1_filtered_ms": "SD1 filtered",
    "sd2_filtered_ms": "SD2 filtered",
    "modcsi100_filtered_ms": "ModCSI100 filtered",
    "slope100_bpm_per_s": "HR slope100",
    "j1_csi_x_slope": "J1 = CSI x slope",
    "j2_modcsi_filtered_x_slope": "J2 = ModCSI x slope",
}
THRESHOLD_FEATURES = ("j1_csi_x_slope", "j2_modcsi_filtered_x_slope")
NOISE_START_S = 5.0 * 60.0
NOISE_DURATION_S = 2.0 * 60.0
NOISE_CYCLE_S = 4.0 * 60.0
MATCH_TOLERANCE_MS = 75.0
HRV_WINDOW_RR = 100
MEDIAN_WIDTH_RR = 7
MORPH_PRE_MS = 60.0
MORPH_POST_MS = 60.0


def _default_mitdb() -> Path:
    return (
        PROJECT.parent
        / "Datasets"
        / "mit-bih-arrhythmia-database-1.0"
        / "mit-bih-arrhythmia-database-1.0.0"
    )


def _default_nstdb() -> Path:
    return (
        PROJECT.parent
        / "Datasets"
        / "mit-bih-noise-stress-test-database-1.0"
        / "mit-bih-noise-stress-test-database-1.0.0"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mitdb", type=Path, default=_default_mitdb())
    parser.add_argument("--nstdb", type=Path, default=_default_nstdb())
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "outputs" / "artifact_degradation_nstdb_v1",
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        choices=SOURCES,
        default=list(SOURCES),
        help="Clean source records to evaluate (default: 118 and 119).",
    )
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=None,
        help="Optional labels such as clean, '+24 dB', '0 dB', or '-6 dB'.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mitdb = args.mitdb.expanduser().resolve()
    nstdb = args.nstdb.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = tuple(str(value) for value in args.sources)
    conditions = _select_conditions(args.conditions)

    interval_rows: list[dict[str, object]] = []
    hrv_record_rows: list[dict[str, object]] = []
    hrv_feature_rows: list[dict[str, object]] = []
    hrv_pair_frames: list[pd.DataFrame] = []
    hrv_reference_frames: list[pd.DataFrame] = []
    hrv_algorithm_frames: list[pd.DataFrame] = []
    morphology_record_rows: list[dict[str, object]] = []
    morphology_window_frames: list[pd.DataFrame] = []
    detector_failures: list[dict[str, object]] = []
    input_alignment_rows: list[dict[str, object]] = []
    annotation_provenance_rows: list[dict[str, object]] = []

    source_context: dict[str, dict[str, object]] = {}
    for source in sources:
        clean_base = mitdb / source
        clean_ecg, fs, lead, duration_s = _load_signal(clean_base)
        reference, symbols = load_wfdb_beat_annotations(clean_base)
        intervals = _noise_intervals(duration_s)
        if not intervals:
            raise ValueError(f"{source}: no official noisy intervals fit record")
        for interval_index, (start_s, end_s) in enumerate(intervals):
            interval_rows.append(
                {
                    "source_record": source,
                    "interval_index": interval_index,
                    "start_s": start_s,
                    "end_s": end_s,
                    "duration_s": end_s - start_s,
                }
            )
        clean_reference_morphology = extract_varon_morphology(
            clean_ecg,
            reference,
            fs,
            anchor_track="clean_expert_atr",
            pre_r_ms=MORPH_PRE_MS,
            post_r_ms=MORPH_POST_MS,
        )
        clean_reference_hrv = _rr_features(reference, fs)
        source_context[source] = {
            "clean_ecg": clean_ecg,
            "fs": fs,
            "lead": lead,
            "duration_s": duration_s,
            "reference": reference,
            "symbols": symbols,
            "intervals": intervals,
            "clean_reference_morphology": clean_reference_morphology,
            "clean_reference_hrv": clean_reference_hrv,
        }

    # Thresholds are fixed once from expert/reference HRV in the same clock
    # intervals and then reused at every SNR.
    thresholds_by_source: dict[str, dict[str, float]] = {}
    for source, context in source_context.items():
        reference_windows = _eligible_rr_windows(
            context["reference"],  # type: ignore[arg-type]
            context["clean_reference_hrv"],  # type: ignore[arg-type]
            context["intervals"],  # type: ignore[arg-type]
            float(context["fs"]),
            owner="reference",
        )
        thresholds_by_source[source] = {
            feature: float(reference_windows[feature].quantile(0.95))
            for feature in THRESHOLD_FEATURES
        }

    for source in sources:
        context = source_context[source]
        clean_ecg = np.asarray(context["clean_ecg"], dtype=np.float64)
        fs = float(context["fs"])
        lead = str(context["lead"])
        duration_s = float(context["duration_s"])
        reference = np.asarray(context["reference"], dtype=np.int64)
        symbols = np.asarray(context["symbols"], dtype=object)
        intervals = list(context["intervals"])  # type: ignore[arg-type]
        clean_reference_hrv = context["clean_reference_hrv"]
        clean_reference_morphology = context["clean_reference_morphology"]

        for condition_order, (condition, snr_db, suffix) in enumerate(conditions):
            if suffix is None:
                record_id = source
                signal = clean_ecg
                input_alignment_rows.append(
                    {
                        "source_record": source,
                        "record_id": record_id,
                        "condition": condition,
                        "snr_db": snr_db,
                        "learning_period_s": NOISE_START_S,
                        "constant_offset_added_mv": 0.0,
                        "learning_period_median_absolute_difference_before_mv": 0.0,
                        "learning_period_median_absolute_difference_after_mv": 0.0,
                        "learning_period_max_absolute_difference_after_mv": 0.0,
                    }
                )
                annotation_provenance_rows.append(
                    {
                        "source_record": source,
                        "record_id": record_id,
                        "condition": condition,
                        "snr_db": snr_db,
                        "clean_annotation_count": int(reference.size),
                        "record_annotation_count": int(reference.size),
                        "exact_sample_array_match": True,
                        "exact_symbol_array_match": True,
                        "matches_within_75_ms": int(reference.size),
                        "median_absolute_timestamp_difference_ms": 0.0,
                        "maximum_absolute_timestamp_difference_ms": 0.0,
                        "reference_used": "current clean MIT-BIH atr",
                    }
                )
            else:
                record_id = f"{source}{suffix}"
                noisy_base = nstdb / record_id
                signal, noisy_fs, noisy_lead, noisy_duration_s = _load_signal(noisy_base)
                if not np.isclose(noisy_fs, fs):
                    raise ValueError(f"{record_id}: sampling rate differs from clean")
                if signal.size != clean_ecg.size:
                    raise ValueError(f"{record_id}: sample count differs from clean")
                if noisy_lead != lead:
                    raise ValueError(f"{record_id}: channel-0 lead differs from clean")
                if not np.isclose(noisy_duration_s, duration_s):
                    raise ValueError(f"{record_id}: duration differs from clean")
                noisy_reference, noisy_symbols = load_wfdb_beat_annotations(noisy_base)
                annotation_matches = match_peaks_to_reference(
                    noisy_reference,
                    reference,
                    sampling_rate_hz=fs,
                    tolerance_ms=MATCH_TOLERANCE_MS,
                )
                annotation_timing_ms = 1000.0 * (
                    noisy_reference[annotation_matches.detected_indices]
                    - reference[annotation_matches.reference_indices]
                ) / fs
                annotation_provenance_rows.append(
                    {
                        "source_record": source,
                        "record_id": record_id,
                        "condition": condition,
                        "snr_db": snr_db,
                        "clean_annotation_count": int(reference.size),
                        "record_annotation_count": int(noisy_reference.size),
                        "exact_sample_array_match": bool(
                            np.array_equal(noisy_reference, reference)
                        ),
                        "exact_symbol_array_match": bool(
                            np.array_equal(noisy_symbols, symbols)
                        ),
                        "matches_within_75_ms": int(
                            annotation_matches.detected_indices.size
                        ),
                        "median_absolute_timestamp_difference_ms": (
                            float(np.median(np.abs(annotation_timing_ms)))
                            if annotation_timing_ms.size
                            else np.nan
                        ),
                        "maximum_absolute_timestamp_difference_ms": (
                            float(np.max(np.abs(annotation_timing_ms)))
                            if annotation_timing_ms.size
                            else np.nan
                        ),
                        "reference_used": "current clean MIT-BIH atr",
                    }
                )
                # NSTDB changes ADC zero/format relative to the source records.
                # Remove only the one constant representation offset estimated
                # from the official initial five-minute noise-free learning
                # period.  Time-varying baseline displacement in the noisy
                # intervals remains part of the artifact being tested.
                learning_samples = min(signal.size, int(np.rint(NOISE_START_S * fs)))
                learning_difference = clean_ecg[:learning_samples] - signal[:learning_samples]
                offset_correction = float(np.median(learning_difference))
                before_mae = float(np.median(np.abs(learning_difference)))
                signal = signal + offset_correction
                after_difference = clean_ecg[:learning_samples] - signal[:learning_samples]
                input_alignment_rows.append(
                    {
                        "source_record": source,
                        "record_id": record_id,
                        "condition": condition,
                        "snr_db": snr_db,
                        "learning_period_s": NOISE_START_S,
                        "constant_offset_added_mv": offset_correction,
                        "learning_period_median_absolute_difference_before_mv": before_mae,
                        "learning_period_median_absolute_difference_after_mv": float(
                            np.median(np.abs(after_difference))
                        ),
                        "learning_period_max_absolute_difference_after_mv": float(
                            np.max(np.abs(after_difference))
                        ),
                    }
                )

            detections: dict[str, np.ndarray] = {}
            for track in TRACKS:
                try:
                    detections[track] = _detect_track(track, signal, fs)
                except Exception as exc:  # preserve a failed condition as evidence
                    detections[track] = np.array([], dtype=np.int64)
                    detector_failures.append(
                        {
                            "source_record": source,
                            "record_id": record_id,
                            "condition": condition,
                            "snr_db": snr_db,
                            "track": track,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )

            # HRV branch: current primary/owner is unchanged UNSW timestamps.
            unsw = detections["unsw"]
            hrv_bundle = _audit_hrv_condition(
                source=source,
                record_id=record_id,
                condition=condition,
                condition_order=condition_order,
                snr_db=snr_db,
                candidates=unsw,
                reference=reference,
                reference_features=clean_reference_hrv,  # type: ignore[arg-type]
                intervals=intervals,
                fs=fs,
                thresholds=thresholds_by_source[source],
            )
            hrv_record_rows.append(hrv_bundle["summary"])
            hrv_feature_rows.extend(hrv_bundle["features"])
            hrv_pair_frames.append(hrv_bundle["pairs"])
            hrv_reference_frames.append(hrv_bundle["reference_windows"])
            hrv_algorithm_frames.append(hrv_bundle["algorithm_windows"])

            # Morphology: oracle anchors isolate direct waveform corruption;
            # detector tracks add missed/extra/timing degradation.
            for track in MORPH_TRACKS:
                candidates = reference if track == "expert_anchor_oracle" else detections[track]
                morph_bundle = _audit_morphology_condition(
                    source=source,
                    record_id=record_id,
                    condition=condition,
                    condition_order=condition_order,
                    snr_db=snr_db,
                    track=track,
                    signal=signal,
                    candidates=candidates,
                    reference=reference,
                    symbols=symbols,
                    clean_reference=clean_reference_morphology,
                    intervals=intervals,
                    fs=fs,
                )
                morphology_record_rows.append(morph_bundle["summary"])
                morphology_window_frames.append(morph_bundle["windows"])

            print(f"completed {record_id} ({condition})", flush=True)

    intervals_table = pd.DataFrame(interval_rows)
    hrv_records = pd.DataFrame(hrv_record_rows)
    hrv_features = pd.DataFrame(hrv_feature_rows)
    hrv_pairs = pd.concat(hrv_pair_frames, ignore_index=True)
    hrv_reference_windows = pd.concat(hrv_reference_frames, ignore_index=True)
    hrv_algorithm_windows = pd.concat(hrv_algorithm_frames, ignore_index=True)
    morphology_records = pd.DataFrame(morphology_record_rows)
    morphology_windows = pd.concat(morphology_window_frames, ignore_index=True)
    failures = pd.DataFrame(
        detector_failures,
        columns=[
            "source_record",
            "record_id",
            "condition",
            "snr_db",
            "track",
            "error_type",
            "error",
        ],
    )
    input_alignment = pd.DataFrame(input_alignment_rows)
    annotation_provenance = pd.DataFrame(annotation_provenance_rows)

    hrv_pooled, hrv_pooled_features = _pool_hrv(
        hrv_records,
        hrv_pairs,
        hrv_reference_windows,
        hrv_algorithm_windows,
        thresholds_by_source,
    )
    morphology_pooled = _pool_morphology(morphology_records, morphology_windows)
    hrv_pooled = _add_clean_deltas(
        hrv_pooled,
        group_columns=[],
        metrics=(
            "beat_f1",
            "hrv_window_coverage",
            "j1_ccc",
            "j2_ccc",
            "j1_effective_q95_f1",
            "j2_effective_q95_f1",
        ),
    )
    morphology_pooled = _add_clean_deltas(
        morphology_pooled,
        group_columns=["candidate_track"],
        metrics=(
            "beat_f1",
            "five_beat_window_coverage",
            "median_raw_eigenvalue_relative_l1_error",
            "median_normalized_spectrum_l1_error",
            "lambda1_ccc",
        ),
    )

    intervals_table.to_csv(output / "artifact_intervals.csv", index=False)
    input_alignment.to_csv(output / "input_alignment.csv", index=False)
    annotation_provenance.to_csv(output / "annotation_provenance.csv", index=False)
    hrv_records.to_csv(output / "hrv_record_summary.csv", index=False)
    hrv_features.to_csv(output / "hrv_record_feature_agreement.csv", index=False)
    hrv_pooled.to_csv(output / "hrv_pooled_summary.csv", index=False)
    hrv_pooled_features.to_csv(output / "hrv_pooled_feature_agreement.csv", index=False)
    hrv_pairs.to_csv(output / "hrv_window_pairs.csv.gz", index=False, compression="gzip")
    morphology_records.to_csv(output / "morphology_record_summary.csv", index=False)
    morphology_pooled.to_csv(output / "morphology_pooled_summary.csv", index=False)
    morphology_windows.to_csv(
        output / "morphology_window_fidelity.csv.gz", index=False, compression="gzip"
    )
    failures.to_csv(output / "detector_failures.csv", index=False)

    scope = {
        "benchmark": "controlled HRV and morphology artifact degradation",
        "mitdb": str(mitdb),
        "nstdb": str(nstdb),
        "dataset_sources": [
            "https://physionet.org/content/nstdb/1.0.0/",
            "https://physionet.org/content/mitdb/1.0.0/",
            "https://physionet.org/physiotools/wag/evnode14.htm",
        ],
        "source_records": list(sources),
        "conditions": [
            {"label": label, "snr_db": snr, "suffix": suffix}
            for label, snr, suffix in conditions
        ],
        "artifact_type": "electrode motion artifact (NSTDB em)",
        "channel_index": 0,
        "lead_by_source": {
            source: str(source_context[source]["lead"]) for source in sources
        },
        "official_protocol": (
            "first 5 minutes clean, then 2 minutes noisy alternating with "
            "2 minutes clean; only fully contained noisy windows were scored"
        ),
        "input_alignment": (
            "one constant per-record physical offset estimated from the official "
            "initial five-minute noise-free learning period; time-varying artifact "
            "and baseline displacement were not removed"
        ),
        "matching_tolerance_ms": MATCH_TOLERANCE_MS,
        "annotation_reference_policy": (
            "current clean MIT-BIH atr annotation used at every SNR; the NSTDB "
            "119 copies contain older timestamp placements, documented in "
            "annotation_provenance.csv"
        ),
        "hrv": {
            "timestamp_owner": "unchanged UNSW",
            "window_rr": HRV_WINDOW_RR,
            "median_filter_rr": MEDIAN_WIDTH_RR,
            "reference": "expert atr timestamps on clean source records",
            "threshold_audit": (
                "per-source expert 95th percentile fixed across all SNRs; "
                "descriptive measurement stress test only"
            ),
        },
        "morphology": {
            "method": "Varon 2015 five-beat uncentered Q @ Q.T eigenvalues",
            "capture_ms": [MORPH_PRE_MS, MORPH_POST_MS],
            "reference": "clean waveform at expert atr timestamps",
            "expert_anchor_oracle": (
                "isolates direct waveform corruption without detector error"
            ),
        },
        "not_measured": [
            "seizure sensitivity or specificity",
            "clinical alarm validity",
            "artifact classification accuracy",
            "patient generalization beyond records 118 and 119",
            "baseline-wander or muscle-artifact degradation",
        ],
        "detector_failure_count": int(failures.shape[0]),
    }
    (output / "benchmark_scope.json").write_text(
        json.dumps(scope, indent=2), encoding="utf-8"
    )
    _plot_hrv(hrv_pooled, output / "hrv_artifact_degradation.png")
    _plot_morphology(morphology_pooled, output / "morphology_artifact_degradation.png")
    report = _render_report(
        scope,
        hrv_pooled,
        morphology_pooled,
        hrv_records,
        morphology_records,
        failures,
    )
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    print("\nHRV POOLED\n" + hrv_pooled.to_string(index=False), flush=True)
    print("\nMORPHOLOGY POOLED\n" + morphology_pooled.to_string(index=False), flush=True)
    return 0


def _select_conditions(requested: list[str] | None) -> tuple[tuple[str, float | None, str | None], ...]:
    if not requested:
        return CONDITIONS
    normalized = {value.strip().lower() for value in requested}
    selected = tuple(
        item
        for item in CONDITIONS
        if item[0].lower() in normalized
        or (item[1] is not None and str(int(item[1])) in normalized)
    )
    if not selected:
        raise ValueError(f"No conditions matched {requested!r}")
    return selected


def _load_signal(base: Path) -> tuple[np.ndarray, float, str, float]:
    header = wfdb.rdheader(str(base))
    record = wfdb.rdrecord(str(base), channels=[0], physical=True)
    if record.p_signal is None:
        raise ValueError(f"{base}: no physical signal")
    signal = np.asarray(record.p_signal[:, 0], dtype=np.float64)
    if not np.all(np.isfinite(signal)):
        raise ValueError(f"{base}: non-finite ECG samples")
    fs = float(header.fs)
    return signal, fs, str(header.sig_name[0]), float(signal.size / fs)


def _noise_intervals(duration_s: float) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    start = NOISE_START_S
    while start < duration_s:
        end = min(start + NOISE_DURATION_S, duration_s)
        if end > start:
            intervals.append((start, end))
        start += NOISE_CYCLE_S
    return intervals


def _detect_track(track: str, signal: np.ndarray, fs: float) -> np.ndarray:
    if track == "unsw":
        return detect_unsw(signal, fs, orientation="original").peak_samples
    if track == "neurokit":
        return detect_neurokit_gradient(
            signal,
            fs,
            orientation="original",
            minimum_delay_ms=300.0,
            minimum_delay_inclusive=False,
        ).peak_samples
    if track == "zhai":
        return detect_zhai_template(signal, fs, orientation="original").peak_samples
    raise ValueError(track)


def _rr_features(peaks: np.ndarray, fs: float) -> pd.DataFrame:
    peaks = np.asarray(peaks, dtype=np.int64)
    if peaks.size < 2:
        return pd.DataFrame()
    return extract_rr_hrv_features(
        peaks,
        sampling_rate_hz=fs,
        segment_start_s=0.0,
        agreement=compare_peak_sequences(peaks, peaks, sampling_rate_hz=fs),
        window_size=HRV_WINDOW_RR,
        median_width=MEDIAN_WIDTH_RR,
    )


def _interval_index_for_window(
    start_sample: int,
    end_sample: int,
    intervals: list[tuple[float, float]],
    fs: float,
    *,
    margin_start_samples: int = 0,
    margin_end_samples: int = 0,
) -> int | None:
    for index, (start_s, end_s) in enumerate(intervals):
        lower = int(np.ceil(start_s * fs)) + margin_start_samples
        upper = int(np.floor(end_s * fs)) - margin_end_samples
        if start_sample >= lower and end_sample < upper:
            return index
    return None


def _segment_event_metrics(
    candidates: np.ndarray,
    reference: np.ndarray,
    intervals: list[tuple[float, float]],
    fs: float,
) -> dict[str, object]:
    candidate_count = reference_count = tp = 0
    timing_errors: list[np.ndarray] = []
    for start_s, end_s in intervals:
        start_sample = int(np.ceil(start_s * fs))
        end_sample = int(np.floor(end_s * fs))
        detected = candidates[(candidates >= start_sample) & (candidates < end_sample)]
        expert = reference[(reference >= start_sample) & (reference < end_sample)]
        matches = match_peaks_to_reference(
            detected,
            expert,
            sampling_rate_hz=fs,
            tolerance_ms=MATCH_TOLERANCE_MS,
        )
        candidate_count += int(detected.size)
        reference_count += int(expert.size)
        tp += int(matches.detected_indices.size)
        if matches.detected_indices.size:
            timing_errors.append(
                1000.0
                * (
                    detected[matches.detected_indices]
                    - expert[matches.reference_indices]
                )
                / fs
            )
    fp = candidate_count - tp
    fn = reference_count - tp
    errors = np.concatenate(timing_errors) if timing_errors else np.array([], dtype=float)
    return {
        "reference_beat_count": reference_count,
        "candidate_beat_count": candidate_count,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "beat_sensitivity": _divide(tp, tp + fn),
        "beat_ppv": _divide(tp, tp + fp),
        "beat_f1": _divide(2 * tp, 2 * tp + fp + fn),
        "median_absolute_timing_error_ms": (
            float(np.median(np.abs(errors))) if errors.size else np.nan
        ),
    }


def _eligible_rr_windows(
    peaks: np.ndarray,
    features: pd.DataFrame,
    intervals: list[tuple[float, float]],
    fs: float,
    *,
    owner: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if features.empty:
        return pd.DataFrame(columns=["peak_index", "interval_index", *HRV_FEATURES])
    for peak_index in range(HRV_WINDOW_RR, peaks.size):
        feature_row = features.iloc[peak_index - 1]
        if not bool(feature_row["feature_defined"]):
            continue
        interval_index = _interval_index_for_window(
            int(peaks[peak_index - HRV_WINDOW_RR]),
            int(peaks[peak_index]),
            intervals,
            fs,
        )
        if interval_index is None:
            continue
        row: dict[str, object] = {
            "peak_index": peak_index,
            "interval_index": interval_index,
            "anchor_sample": int(peaks[peak_index]),
            "anchor_time_s": float(peaks[peak_index] / fs),
            "window_start_sample": int(peaks[peak_index - HRV_WINDOW_RR]),
            "owner": owner,
        }
        for feature in HRV_FEATURES:
            row[feature] = float(feature_row[feature])
        rows.append(row)
    return pd.DataFrame(rows)


def _audit_hrv_condition(
    *,
    source: str,
    record_id: str,
    condition: str,
    condition_order: int,
    snr_db: float | None,
    candidates: np.ndarray,
    reference: np.ndarray,
    reference_features: pd.DataFrame,
    intervals: list[tuple[float, float]],
    fs: float,
    thresholds: dict[str, float],
) -> dict[str, object]:
    candidates = np.asarray(candidates, dtype=np.int64)
    candidate_features = _rr_features(candidates, fs)
    reference_windows = _eligible_rr_windows(
        reference, reference_features, intervals, fs, owner="reference"
    )
    algorithm_windows = _eligible_rr_windows(
        candidates, candidate_features, intervals, fs, owner="algorithm"
    )
    matches = match_peaks_to_reference(
        candidates,
        reference,
        sampling_rate_hz=fs,
        tolerance_ms=MATCH_TOLERANCE_MS,
    )
    reference_index_by_candidate = {
        int(candidate_index): int(reference_index)
        for candidate_index, reference_index in zip(
            matches.detected_indices, matches.reference_indices, strict=True
        )
    }
    reference_rows = {
        int(row.peak_index): row for row in reference_windows.itertuples(index=False)
    }
    pair_rows: list[dict[str, object]] = []
    for algorithm_row in algorithm_windows.itertuples(index=False):
        reference_index = reference_index_by_candidate.get(int(algorithm_row.peak_index))
        reference_row = reference_rows.get(reference_index) if reference_index is not None else None
        if reference_row is None or int(reference_row.interval_index) != int(
            algorithm_row.interval_index
        ):
            continue
        row: dict[str, object] = {
            "source_record": source,
            "record_id": record_id,
            "condition": condition,
            "condition_order": condition_order,
            "snr_db": snr_db,
            "algorithm_peak_index": int(algorithm_row.peak_index),
            "reference_peak_index": int(reference_row.peak_index),
            "interval_index": int(reference_row.interval_index),
            "anchor_time_s": float(reference_row.anchor_time_s),
            "anchor_timing_error_ms": 1000.0
            * (
                int(algorithm_row.anchor_sample) - int(reference_row.anchor_sample)
            )
            / fs,
        }
        for feature in HRV_FEATURES:
            row[f"reference__{feature}"] = float(getattr(reference_row, feature))
            row[f"algorithm__{feature}"] = float(getattr(algorithm_row, feature))
        pair_rows.append(row)
    pairs = pd.DataFrame(pair_rows)

    prefix = {
        "source_record": source,
        "record_id": record_id,
        "condition": condition,
        "condition_order": condition_order,
        "snr_db": snr_db,
    }
    reference_windows = _with_prefix(reference_windows, prefix)
    algorithm_windows = _with_prefix(algorithm_windows, prefix)
    events = _segment_event_metrics(candidates, reference, intervals, fs)
    feature_rows: list[dict[str, object]] = []
    for feature, label in HRV_FEATURES.items():
        if pairs.empty:
            reference_values = candidate_values = np.array([], dtype=float)
        else:
            reference_values = pairs[f"reference__{feature}"].to_numpy(float)
            candidate_values = pairs[f"algorithm__{feature}"].to_numpy(float)
        feature_rows.append(
            {
                **prefix,
                "feature": feature,
                "feature_label": label,
                "paired_windows": int(reference_values.size),
                "ccc": lin_concordance_correlation(reference_values, candidate_values),
                "smape": symmetric_absolute_percentage_error(
                    reference_values, candidate_values
                ),
                "median_absolute_error": (
                    float(np.median(np.abs(candidate_values - reference_values)))
                    if reference_values.size
                    else np.nan
                ),
            }
        )
    feature_lookup = {row["feature"]: row for row in feature_rows}
    threshold_metrics = _effective_threshold_metrics(
        reference_windows,
        algorithm_windows,
        matches,
        thresholds,
    )
    summary = {
        **prefix,
        **events,
        "reference_hrv_window_count": int(reference_windows.shape[0]),
        "paired_hrv_window_count": int(pairs.shape[0]),
        "hrv_window_coverage": _divide(
            int(pairs.shape[0]), int(reference_windows.shape[0])
        ),
        "j1_ccc": feature_lookup["j1_csi_x_slope"]["ccc"],
        "j2_ccc": feature_lookup["j2_modcsi_filtered_x_slope"]["ccc"],
        "j1_smape": feature_lookup["j1_csi_x_slope"]["smape"],
        "j2_smape": feature_lookup["j2_modcsi_filtered_x_slope"]["smape"],
        **threshold_metrics,
    }
    return {
        "summary": summary,
        "features": feature_rows,
        "pairs": pairs,
        "reference_windows": reference_windows,
        "algorithm_windows": algorithm_windows,
    }


def _effective_threshold_metrics(
    reference_windows: pd.DataFrame,
    algorithm_windows: pd.DataFrame,
    matches: object,
    thresholds: dict[str, float],
) -> dict[str, object]:
    candidate_to_reference = {
        int(candidate_index): int(reference_index)
        for candidate_index, reference_index in zip(
            matches.detected_indices, matches.reference_indices, strict=True  # type: ignore[attr-defined]
        )
    }
    reference_by_index = {
        int(row.peak_index): row for row in reference_windows.itertuples(index=False)
    }
    payload: dict[str, object] = {}
    for feature in THRESHOLD_FEATURES:
        threshold = float(thresholds[feature])
        reference_high = {
            int(row.peak_index)
            for row in reference_windows.itertuples(index=False)
            if float(getattr(row, feature)) >= threshold
        }
        candidate_high_rows = [
            row
            for row in algorithm_windows.itertuples(index=False)
            if float(getattr(row, feature)) >= threshold
        ]
        matched_reference_high: set[int] = set()
        fp = 0
        for row in candidate_high_rows:
            reference_index = candidate_to_reference.get(int(row.peak_index))
            reference_row = reference_by_index.get(reference_index) if reference_index is not None else None
            if (
                reference_row is not None
                and int(reference_row.interval_index) == int(row.interval_index)
                and reference_index in reference_high
            ):
                matched_reference_high.add(int(reference_index))
            else:
                fp += 1
        tp = len(matched_reference_high)
        fn = len(reference_high) - tp
        stem = "j1" if feature == "j1_csi_x_slope" else "j2"
        payload.update(
            {
                f"{stem}_q95_threshold": threshold,
                f"{stem}_effective_q95_tp": tp,
                f"{stem}_effective_q95_fp": fp,
                f"{stem}_effective_q95_fn": fn,
                f"{stem}_effective_q95_sensitivity": _divide(tp, tp + fn),
                f"{stem}_effective_q95_ppv": _divide(tp, tp + fp),
                f"{stem}_effective_q95_f1": _divide(2 * tp, 2 * tp + fp + fn),
            }
        )
    return payload


def _audit_morphology_condition(
    *,
    source: str,
    record_id: str,
    condition: str,
    condition_order: int,
    snr_db: float | None,
    track: str,
    signal: np.ndarray,
    candidates: np.ndarray,
    reference: np.ndarray,
    symbols: np.ndarray,
    clean_reference: object,
    intervals: list[tuple[float, float]],
    fs: float,
) -> dict[str, object]:
    audit = audit_morphology_timestamp_fidelity(
        signal,
        reference,
        symbols,
        candidates,
        fs,
        candidate_track=track,
        tolerance_ms=MATCH_TOLERANCE_MS,
        pre_r_ms=MORPH_PRE_MS,
        post_r_ms=MORPH_POST_MS,
    )
    clean_features = clean_reference.features  # type: ignore[attr-defined]
    clean_by_end_index = {
        int(row.end_beat_index): row for row in clean_features.itertuples(index=False)
    }
    pre_samples = int(clean_reference.parameters["pre_r_samples"])  # type: ignore[attr-defined]
    post_samples = int(clean_reference.parameters["post_r_samples"])  # type: ignore[attr-defined]
    rows: list[dict[str, object]] = []
    for window in audit.windows.itertuples(index=False):
        interval_index = _interval_index_for_window(
            int(window.reference_start_sample),
            int(window.reference_end_sample),
            intervals,
            fs,
            margin_start_samples=pre_samples,
            margin_end_samples=post_samples,
        )
        if interval_index is None:
            continue
        end_reference_index = int(window.reference_window_index) + 4
        clean_row = clean_by_end_index[end_reference_index]
        comparable = bool(window.morphology_comparable)
        clean_values = np.asarray(
            [getattr(clean_row, f"lambda{i}") for i in range(1, 6)], dtype=float
        )
        candidate_values = np.asarray(
            [getattr(window, f"candidate_lambda{i}") for i in range(1, 6)],
            dtype=float,
        )
        clean_total = float(np.sum(clean_values))
        candidate_total = float(np.sum(candidate_values)) if comparable else np.nan
        relative_l1 = (
            float(np.sum(np.abs(candidate_values - clean_values)) / clean_total)
            if comparable and clean_total > 0
            else np.nan
        )
        clean_spectrum = clean_values / clean_total if clean_total > 0 else np.full(5, np.nan)
        candidate_spectrum = (
            candidate_values / candidate_total
            if comparable and candidate_total > 0
            else np.full(5, np.nan)
        )
        row: dict[str, object] = {
            "source_record": source,
            "record_id": record_id,
            "condition": condition,
            "condition_order": condition_order,
            "snr_db": snr_db,
            "candidate_track": track,
            "interval_index": interval_index,
            "reference_window_index": int(window.reference_window_index),
            "reference_end_time_s": float(window.reference_end_time_s),
            "morphology_comparable": comparable,
            "unavailable_reason": str(window.unavailable_reason),
            "raw_eigenvalue_relative_l1_error": relative_l1,
            "normalized_spectrum_l1_error": (
                float(np.sum(np.abs(candidate_spectrum - clean_spectrum)))
                if comparable
                else np.nan
            ),
            "eigenvalue_total_energy_ratio": (
                candidate_total / clean_total
                if comparable and clean_total > 0
                else np.nan
            ),
        }
        for component in range(1, 6):
            row[f"clean_lambda{component}"] = float(clean_values[component - 1])
            row[f"candidate_lambda{component}"] = float(candidate_values[component - 1])
        rows.append(row)
    windows = pd.DataFrame(rows)
    comparable = (
        windows["morphology_comparable"].to_numpy(bool)
        if not windows.empty
        else np.array([], dtype=bool)
    )
    events = _segment_event_metrics(candidates, reference, intervals, fs)
    summary: dict[str, object] = {
        "source_record": source,
        "record_id": record_id,
        "condition": condition,
        "condition_order": condition_order,
        "snr_db": snr_db,
        "candidate_track": track,
        **events,
        "reference_five_beat_window_count": int(windows.shape[0]),
        "comparable_five_beat_window_count": int(np.sum(comparable)),
        "five_beat_window_coverage": _divide(int(np.sum(comparable)), int(windows.shape[0])),
        "median_raw_eigenvalue_relative_l1_error": (
            float(windows.loc[comparable, "raw_eigenvalue_relative_l1_error"].median())
            if np.any(comparable)
            else np.nan
        ),
        "median_normalized_spectrum_l1_error": (
            float(windows.loc[comparable, "normalized_spectrum_l1_error"].median())
            if np.any(comparable)
            else np.nan
        ),
        "median_total_energy_ratio": (
            float(windows.loc[comparable, "eigenvalue_total_energy_ratio"].median())
            if np.any(comparable)
            else np.nan
        ),
    }
    for component in range(1, 6):
        if np.any(comparable):
            clean_values = windows.loc[comparable, f"clean_lambda{component}"].to_numpy(float)
            candidate_values = windows.loc[
                comparable, f"candidate_lambda{component}"
            ].to_numpy(float)
        else:
            clean_values = candidate_values = np.array([], dtype=float)
        summary[f"lambda{component}_ccc"] = lin_concordance_correlation(
            clean_values, candidate_values
        )
        summary[f"lambda{component}_smape"] = symmetric_absolute_percentage_error(
            clean_values, candidate_values
        )
    return {"summary": summary, "windows": windows}


def _pool_hrv(
    records: pd.DataFrame,
    pairs: pd.DataFrame,
    reference_windows: pd.DataFrame,
    algorithm_windows: pd.DataFrame,
    thresholds_by_source: dict[str, dict[str, float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    for keys, record_group in records.groupby(
        ["condition_order", "condition", "snr_db"], dropna=False, sort=True
    ):
        order, condition, snr_db = keys
        selected_pairs = pairs[pairs["condition_order"] == order]
        selected_reference = reference_windows[
            reference_windows["condition_order"] == order
        ]
        selected_algorithm = algorithm_windows[
            algorithm_windows["condition_order"] == order
        ]
        tp = int(record_group["tp"].sum())
        fp = int(record_group["fp"].sum())
        fn = int(record_group["fn"].sum())
        pooled_features: dict[str, dict[str, object]] = {}
        for feature, label in HRV_FEATURES.items():
            reference_values = selected_pairs[f"reference__{feature}"].to_numpy(float)
            candidate_values = selected_pairs[f"algorithm__{feature}"].to_numpy(float)
            row = {
                "condition_order": int(order),
                "condition": condition,
                "snr_db": snr_db,
                "feature": feature,
                "feature_label": label,
                "paired_windows": int(reference_values.size),
                "ccc": lin_concordance_correlation(reference_values, candidate_values),
                "smape": symmetric_absolute_percentage_error(
                    reference_values, candidate_values
                ),
                "median_absolute_error": (
                    float(np.median(np.abs(candidate_values - reference_values)))
                    if reference_values.size
                    else np.nan
                ),
            }
            feature_rows.append(row)
            pooled_features[feature] = row
        threshold_payload = _pool_effective_threshold_metrics(
            selected_reference,
            selected_algorithm,
            thresholds_by_source,
        )
        summary_rows.append(
            {
                "condition_order": int(order),
                "condition": condition,
                "snr_db": snr_db,
                "source_record_count": int(record_group.shape[0]),
                "reference_beat_count": int(record_group["reference_beat_count"].sum()),
                "candidate_beat_count": int(record_group["candidate_beat_count"].sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "beat_sensitivity": _divide(tp, tp + fn),
                "beat_ppv": _divide(tp, tp + fp),
                "beat_f1": _divide(2 * tp, 2 * tp + fp + fn),
                "median_record_absolute_timing_error_ms": float(
                    record_group["median_absolute_timing_error_ms"].median()
                ),
                "reference_hrv_window_count": int(
                    record_group["reference_hrv_window_count"].sum()
                ),
                "paired_hrv_window_count": int(record_group["paired_hrv_window_count"].sum()),
                "hrv_window_coverage": _divide(
                    int(record_group["paired_hrv_window_count"].sum()),
                    int(record_group["reference_hrv_window_count"].sum()),
                ),
                "j1_ccc": pooled_features["j1_csi_x_slope"]["ccc"],
                "j2_ccc": pooled_features["j2_modcsi_filtered_x_slope"]["ccc"],
                "j1_smape": pooled_features["j1_csi_x_slope"]["smape"],
                "j2_smape": pooled_features["j2_modcsi_filtered_x_slope"]["smape"],
                **threshold_payload,
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(feature_rows)


def _pool_effective_threshold_metrics(
    reference_windows: pd.DataFrame,
    algorithm_windows: pd.DataFrame,
    thresholds_by_source: dict[str, dict[str, float]],
) -> dict[str, object]:
    payload: dict[str, object] = {}
    # Peak indices are local to a source.  Reconstruct the candidate/reference
    # event association from anchor time with the same 75-ms one-to-one rule.
    for feature in THRESHOLD_FEATURES:
        tp = fp = fn = 0
        for source, reference_group in reference_windows.groupby("source_record", sort=True):
            algorithm_group = algorithm_windows[
                algorithm_windows["source_record"] == source
            ]
            threshold = thresholds_by_source[str(source)][feature]
            reference_high = reference_group[
                reference_group[feature].to_numpy(float) >= threshold
            ]
            algorithm_high = algorithm_group[
                algorithm_group[feature].to_numpy(float) >= threshold
            ]
            reference_times = reference_high["anchor_sample"].to_numpy(np.int64)
            algorithm_times = algorithm_high["anchor_sample"].to_numpy(np.int64)
            # Both records use 360 Hz; infer it from the 75-ms integer tolerance.
            matches = match_peaks_to_reference(
                algorithm_times,
                reference_times,
                sampling_rate_hz=360.0,
                tolerance_ms=MATCH_TOLERANCE_MS,
            )
            matched = int(matches.detected_indices.size)
            tp += matched
            fp += int(algorithm_times.size - matched)
            fn += int(reference_times.size - matched)
        stem = "j1" if feature == "j1_csi_x_slope" else "j2"
        payload.update(
            {
                f"{stem}_effective_q95_tp": tp,
                f"{stem}_effective_q95_fp": fp,
                f"{stem}_effective_q95_fn": fn,
                f"{stem}_effective_q95_sensitivity": _divide(tp, tp + fn),
                f"{stem}_effective_q95_ppv": _divide(tp, tp + fp),
                f"{stem}_effective_q95_f1": _divide(2 * tp, 2 * tp + fp + fn),
            }
        )
    return payload


def _pool_morphology(records: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = ["condition_order", "condition", "snr_db", "candidate_track"]
    for keys, record_group in records.groupby(group_columns, dropna=False, sort=True):
        order, condition, snr_db, track = keys
        selected = windows[
            (windows["condition_order"] == order)
            & (windows["candidate_track"] == track)
        ]
        comparable = selected["morphology_comparable"].to_numpy(bool)
        tp = int(record_group["tp"].sum())
        fp = int(record_group["fp"].sum())
        fn = int(record_group["fn"].sum())
        row: dict[str, object] = {
            "condition_order": int(order),
            "condition": condition,
            "snr_db": snr_db,
            "candidate_track": track,
            "source_record_count": int(record_group.shape[0]),
            "reference_beat_count": int(record_group["reference_beat_count"].sum()),
            "candidate_beat_count": int(record_group["candidate_beat_count"].sum()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "beat_sensitivity": _divide(tp, tp + fn),
            "beat_ppv": _divide(tp, tp + fp),
            "beat_f1": _divide(2 * tp, 2 * tp + fp + fn),
            "reference_five_beat_window_count": int(selected.shape[0]),
            "comparable_five_beat_window_count": int(np.sum(comparable)),
            "five_beat_window_coverage": _divide(int(np.sum(comparable)), int(selected.shape[0])),
            "median_raw_eigenvalue_relative_l1_error": (
                float(selected.loc[comparable, "raw_eigenvalue_relative_l1_error"].median())
                if np.any(comparable)
                else np.nan
            ),
            "median_normalized_spectrum_l1_error": (
                float(selected.loc[comparable, "normalized_spectrum_l1_error"].median())
                if np.any(comparable)
                else np.nan
            ),
            "median_total_energy_ratio": (
                float(selected.loc[comparable, "eigenvalue_total_energy_ratio"].median())
                if np.any(comparable)
                else np.nan
            ),
        }
        for component in range(1, 6):
            if np.any(comparable):
                clean_values = selected.loc[
                    comparable, f"clean_lambda{component}"
                ].to_numpy(float)
                candidate_values = selected.loc[
                    comparable, f"candidate_lambda{component}"
                ].to_numpy(float)
            else:
                clean_values = candidate_values = np.array([], dtype=float)
            row[f"lambda{component}_ccc"] = lin_concordance_correlation(
                clean_values, candidate_values
            )
            row[f"lambda{component}_smape"] = symmetric_absolute_percentage_error(
                clean_values, candidate_values
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _add_clean_deltas(
    table: pd.DataFrame,
    *,
    group_columns: list[str],
    metrics: Iterable[str],
) -> pd.DataFrame:
    result = table.copy()
    groups: Iterable[tuple[object, pd.DataFrame]]
    if group_columns:
        group_key: str | list[str] = (
            group_columns[0] if len(group_columns) == 1 else group_columns
        )
        groups = result.groupby(group_key, dropna=False, sort=False)
    else:
        groups = [("all", result)]
    for _, group in groups:
        clean = group[group["condition"] == "clean"]
        if clean.empty:
            continue
        clean_row = clean.iloc[0]
        for metric in metrics:
            result.loc[group.index, f"delta_vs_clean__{metric}"] = (
                group[metric].to_numpy(float) - float(clean_row[metric])
            )
    return result


def _with_prefix(frame: pd.DataFrame, prefix: dict[str, object]) -> pd.DataFrame:
    result = frame.copy()
    for column, value in reversed(list(prefix.items())):
        result.insert(0, column, value)
    return result


def _plot_hrv(table: pd.DataFrame, path: Path) -> None:
    ordered = table.sort_values("condition_order")
    labels = ordered["condition"].astype(str).tolist()
    x = np.arange(len(labels))
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes[0, 0].plot(x, ordered["beat_f1"], marker="o")
    axes[0, 0].set_title("UNSW beat F1 in artifact intervals")
    axes[0, 1].plot(x, ordered["hrv_window_coverage"], marker="o")
    axes[0, 1].set_title("Strict 100-RR window coverage")
    axes[1, 0].plot(x, ordered["j1_ccc"], marker="o", label="J1")
    axes[1, 0].plot(x, ordered["j2_ccc"], marker="o", label="J2")
    axes[1, 0].set_title("Feature CCC versus expert timestamps")
    axes[1, 0].legend()
    axes[1, 1].plot(x, ordered["j1_effective_q95_f1"], marker="o", label="J1")
    axes[1, 1].plot(x, ordered["j2_effective_q95_f1"], marker="o", label="J2")
    axes[1, 1].set_title("Effective fixed-95th-percentile F1")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.set_xticks(x, labels, rotation=25)
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.25)
    axes[1, 0].set_ylim(-0.10, 1.02)
    figure.suptitle("HRV branch degradation under electrode-motion artifact")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_morphology(table: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    labels = (
        table.sort_values("condition_order")["condition"].drop_duplicates().tolist()
    )
    x = np.arange(len(labels))
    for track in MORPH_TRACKS:
        selected = table[table["candidate_track"] == track].sort_values(
            "condition_order"
        )
        label = track.replace("expert_anchor_oracle", "oracle anchors")
        axes[0, 0].plot(x, selected["beat_f1"], marker="o", label=label)
        axes[0, 1].plot(
            x, selected["five_beat_window_coverage"], marker="o", label=label
        )
        axes[1, 0].plot(
            x,
            selected["median_raw_eigenvalue_relative_l1_error"],
            marker="o",
            label=label,
        )
        axes[1, 1].plot(x, selected["lambda1_ccc"], marker="o", label=label)
    axes[0, 0].set_title("Beat F1")
    axes[0, 1].set_title("Strict five-beat coverage")
    axes[1, 0].set_title("Median raw eigenspectrum relative L1 error")
    axes[1, 1].set_title("Lambda1 CCC versus clean expert reference")
    for axis in (axes[0, 0], axes[0, 1], axes[1, 1]):
        axis.set_ylim(-0.02, 1.02)
    for axis in axes.flat:
        axis.set_xticks(x, labels, rotation=25)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    figure.suptitle("Morphology branch degradation under electrode-motion artifact")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _render_report(
    scope: dict[str, object],
    hrv: pd.DataFrame,
    morphology: pd.DataFrame,
    hrv_records: pd.DataFrame,
    morphology_records: pd.DataFrame,
    failures: pd.DataFrame,
) -> str:
    hrv_ordered = hrv.sort_values("condition_order")
    hrv_lines = [
        "| Condition | Beat F1 | 100-RR coverage | J1 CCC | J2 CCC | J1 q95 F1 | J2 q95 F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in hrv_ordered.itertuples(index=False):
        hrv_lines.append(
            f"| {row.condition} | {_f(row.beat_f1)} | {_f(row.hrv_window_coverage)} | "
            f"{_f(row.j1_ccc)} | {_f(row.j2_ccc)} | "
            f"{_f(row.j1_effective_q95_f1)} | {_f(row.j2_effective_q95_f1)} |"
        )
    morph_lines = [
        "| Track | Condition | Beat F1 | Five-beat coverage | Raw eigenvalue rel. L1 | Normalized-spectrum L1 | Lambda1 CCC | Energy ratio |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in morphology.sort_values(
        ["candidate_track", "condition_order"]
    ).itertuples(index=False):
        morph_lines.append(
            f"| {row.candidate_track} | {row.condition} | {_f(row.beat_f1)} | "
            f"{_f(row.five_beat_window_coverage)} | "
            f"{_f(row.median_raw_eigenvalue_relative_l1_error)} | "
            f"{_f(row.median_normalized_spectrum_l1_error)} | "
            f"{_f(row.lambda1_ccc)} | {_f(row.median_total_energy_ratio)} |"
        )
    hrv_record_lines = [
        "| Source | Condition | Beat F1 | 100-RR coverage | J1 CCC | J2 CCC | J1 q95 F1 | J2 q95 F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in hrv_records.sort_values(
        ["source_record", "condition_order"]
    ).itertuples(index=False):
        hrv_record_lines.append(
            f"| {row.source_record} | {row.condition} | {_f(row.beat_f1)} | "
            f"{_f(row.hrv_window_coverage)} | {_f(row.j1_ccc)} | {_f(row.j2_ccc)} | "
            f"{_f(row.j1_effective_q95_f1)} | {_f(row.j2_effective_q95_f1)} |"
        )
    morphology_record_lines = [
        "| Source | Track | Condition | Beat F1 | Five-beat coverage | Raw eigenvalue rel. L1 | Normalized-spectrum L1 | Lambda1 CCC | Energy ratio |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in morphology_records.sort_values(
        ["source_record", "candidate_track", "condition_order"]
    ).itertuples(index=False):
        morphology_record_lines.append(
            f"| {row.source_record} | {row.candidate_track} | {row.condition} | "
            f"{_f(row.beat_f1)} | {_f(row.five_beat_window_coverage)} | "
            f"{_f(row.median_raw_eigenvalue_relative_l1_error)} | "
            f"{_f(row.median_normalized_spectrum_l1_error)} | "
            f"{_f(row.lambda1_ccc)} | {_f(row.median_total_energy_ratio)} |"
        )

    clean_hrv = hrv_ordered[hrv_ordered["condition"] == "clean"].iloc[0]
    worst_hrv = hrv_ordered.iloc[-1]
    oracle = morphology[morphology["candidate_track"] == "expert_anchor_oracle"].sort_values(
        "condition_order"
    )
    clean_oracle = oracle[oracle["condition"] == "clean"].iloc[0]
    worst_oracle = oracle.iloc[-1]

    return "\n".join(
        [
            "# Controlled artifact-degradation benchmark: HRV and morphology",
            "",
            "## Bottom line",
            "",
            (
                "This rerun shows degradation against a clean, beat-annotated reference; "
                "it is not a summary of the earlier mixed-condition MIT--BIH audit. "
                f"At {worst_hrv.condition}, the HRV branch's beat F1 changed from {_f(clean_hrv.beat_f1)} "
                f"to {_f(worst_hrv.beat_f1)}, strict 100-RR coverage from "
                f"{_f(clean_hrv.hrv_window_coverage)} to {_f(worst_hrv.hrv_window_coverage)}, "
                f"and J1/J2 CCC to {_f(worst_hrv.j1_ccc)}/{_f(worst_hrv.j2_ccc)}."
            ),
            "",
            (
                "For morphology, the expert-anchor oracle isolates waveform corruption. "
                f"Its median raw-eigenvalue relative L1 error rose from "
                f"{_f(clean_oracle.median_raw_eigenvalue_relative_l1_error)} clean to "
                f"{_f(worst_oracle.median_raw_eigenvalue_relative_l1_error)} at {worst_oracle.condition}, "
                "before any detector errors were added."
            ),
            "",
            "## Test design",
            "",
            "- Data: MIT--BIH clean records 118 and 119 plus the 12 NSTDB electrode-motion copies.",
            "- SNRs: clean control, +24, +18, +12, +6, 0, and -6 dB.",
            "- Scoring intervals: official 2-minute noisy intervals after the initial 5-minute clean period, alternating with 2-minute clean intervals.",
            "- Boundary rule: an HRV or morphology window counted only if its entire supporting window lay inside one noisy interval.",
            "- Input alignment: one constant physical offset was estimated from each record's official initial five-minute noise-free learning period. This removes the NSTDB/source ADC-zero representation shift; it does not remove time-varying electrode-motion artifact.",
            "- Reference: expert `atr` beat timestamps; morphology additionally uses the clean source waveform as the feature reference.",
            "- Annotation version policy: the current clean MIT--BIH `atr` annotation was used at every SNR. NSTDB's copied record-119 annotation has older timestamp placements; the differences are preserved in `annotation_provenance.csv`.",
            f"- Event matching tolerance: {MATCH_TOLERANCE_MS:.0f} ms.",
            "- HRV: unchanged UNSW timestamps, 100 RR intervals, causal seven-RR median filter, J1 and J2 outputs.",
            "- Morphology: five consecutive 120-ms QRS captures and uncentered `Q @ Q.T` eigenvalues.",
            "- The 95th-percentile HRV thresholds were derived once per source from expert windows and frozen across SNRs. They are stress-test thresholds, not seizure thresholds.",
            "",
            "## HRV results",
            "",
            *hrv_lines,
            "",
            "The threshold F1 is effective rather than conditional: missing expert-high windows and unmatched algorithm-high windows count against it.",
            "",
            "## Morphology results",
            "",
            *morph_lines,
            "",
            "The oracle-anchor row measures direct signal corruption. UNSW, NeuroKit, and Zhai rows add detector misses, extras, and timing shifts; their error values are conditional on strict five-beat availability, so coverage must be read beside error.",
            "",
            "## Per-source HRV results",
            "",
            *hrv_record_lines,
            "",
            "## Per-source morphology results",
            "",
            *morphology_record_lines,
            "",
            "## Failure interpretation",
            "",
            "- A high CCC with low coverage is not robust performance; it describes only the surviving comparable windows.",
            "- A morphology energy ratio far from 1 means the unnormalized Varon eigenvalues changed scale under artifact even when expert anchors were supplied.",
            "- Electrode-motion artifact can mimic ectopic activity, so beat F1, HRV preservation, and waveform-feature preservation degrade through different mechanisms.",
            "- The two source records are a controlled engineering stress test, not a population estimate; uncertainty intervals across patients are not identifiable from n=2.",
            "",
            "## Scope limits",
            "",
            "This benchmark does not measure seizure accuracy, clinical alarm validity, artifact classification, baseline-wander tolerance, muscle-artifact tolerance, or generalization beyond records 118 and 119. Adjacent HRV and morphology windows overlap heavily and are not independent observations.",
            "",
            f"Detector execution failures recorded: {int(failures.shape[0])}.",
            "",
            "## Reproducible artifacts",
            "",
            "- `benchmark_scope.json`: exact dataset paths, conditions, rules, and non-claims.",
            "- `input_alignment.csv`: the pre-noise constant offset correction and residual learning-period differences.",
            "- `annotation_provenance.csv`: clean/current versus NSTDB-copied annotation agreement and the reference version used.",
            "- `hrv_record_summary.csv` and `hrv_pooled_summary.csv`: event, coverage, feature, and fixed-threshold results.",
            "- `hrv_pooled_feature_agreement.csv`: all HRV feature agreement metrics by SNR.",
            "- `hrv_window_pairs.csv.gz`: paired expert/algorithm HRV windows.",
            "- `morphology_record_summary.csv` and `morphology_pooled_summary.csv`: oracle and detector-track results.",
            "- `morphology_window_fidelity.csv.gz`: per-window clean-reference comparisons.",
            "- `hrv_artifact_degradation.png` and `morphology_artifact_degradation.png`: degradation curves.",
            "- `detector_failures.csv`: any detector exceptions preserved rather than silently dropped.",
            "",
            "## Dataset provenance",
            "",
            "NSTDB states that calibrated electrode-motion noise was added to clean records 118 and 119 after an initial five-minute clean period, in alternating two-minute noisy and clean intervals. Because the ECG time axis is preserved, the current clean-source reference annotations can be reused at every SNR; the older record-119 placements shipped with NSTDB are documented separately.",
            "",
            "- [MIT--BIH Noise Stress Test Database v1.0.0](https://physionet.org/content/nstdb/1.0.0/)",
            "- [MIT--BIH Arrhythmia Database v1.0.0](https://physionet.org/content/mitdb/1.0.0/)",
            "- [Official WFDB noise-stress protocol](https://physionet.org/physiotools/wag/evnode14.htm)",
            "",
        ]
    )


def _divide(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else np.nan


def _f(value: object) -> str:
    number = float(value)
    return "NA" if not np.isfinite(number) else f"{number:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
