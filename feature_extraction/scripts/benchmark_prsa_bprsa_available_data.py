"""Benchmark the implemented PRSA/BPRSA extractor on available local data.

MIT-BIH supplies expert beat anchors but no seizure labels, so its endpoint is
feature availability and lead/polarity sensitivity.  The local CHB04_28 EDF
contains one ECG lead and two seizure intervals; its endpoint is descriptive
within-patient feature separation only, never clinical accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import wfdb


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecg_cascade.config import NeuroKitZhaiConfig  # noqa: E402
from ecg_cascade.edf import load_ecg_segment  # noqa: E402
from ecg_cascade.neurokit_zhai import run_neurokit_zhai_array  # noqa: E402
from ecg_cascade.prsa_bprsa import (  # noqa: E402
    calculate_prsa_bprsa_feature_arrays,
    extract_prsa_bprsa_features,
)
from ecg_cascade.wfdb_io import load_wfdb_beat_annotations  # noqa: E402


FEATURES = (
    "mean_rr80_ms",
    "sdnn80_ms",
    "prsa_s_rr_ms_per_sample",
    "prsa_delta_rr_ms_per_sample",
    "bprsa_s_r_ms_per_sample",
    "bprsa_delta_r_ms_per_sample",
)
LEAD_POLARITY_RECORDS = ("108", "113", "207", "222", "231")
CHB_SEIZURES = ((1, 1679.0, 1781.0), (2, 3782.0, 3898.0))


def _default_mit() -> Path:
    return (
        PROJECT.parent
        / "Datasets"
        / "mit-bih-arrhythmia-database-1.0"
        / "mit-bih-arrhythmia-database-1.0.0"
    )


def _default_chb() -> Path:
    return PROJECT.parent / "Datasets" / "chb04_28" / "chb04_28.edf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mit-dataset", type=Path, default=_default_mit())
    parser.add_argument("--chb-edf", type=Path, default=_default_chb())
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "outputs" / "prsa_bprsa_available_data_v1",
    )
    parser.add_argument("--records", nargs="*", default=None)
    parser.add_argument("--skip-chb", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    mit_dataset = args.mit_dataset.expanduser().resolve()
    records = (
        [str(value) for value in args.records]
        if args.records
        else sorted(path.stem for path in mit_dataset.glob("*.hea") if path.stem.isdigit())
    )
    if not records:
        raise FileNotFoundError(f"No numeric MIT-BIH records found in {mit_dataset}")

    started = time.perf_counter()
    mit_record_summary, mit_windows, sensitivity = _benchmark_mit(
        mit_dataset, records
    )
    mit_record_summary.to_csv(output / "mit_record_summary.csv", index=False)
    mit_windows.to_csv(
        output / "mit_expert_anchor_features.csv.gz",
        index=False,
        compression="gzip",
    )
    sensitivity.to_csv(output / "mit_lead_polarity_sensitivity.csv", index=False)
    pooled = _mit_pooled_summary(mit_record_summary, mit_windows)
    (output / "mit_pooled_summary.json").write_text(
        json.dumps(pooled, indent=2), encoding="utf-8"
    )

    chb_summary: list[dict[str, object]] = []
    chb_support: list[dict[str, object]] = []
    if not args.skip_chb:
        chb_edf = args.chb_edf.expanduser().resolve()
        if chb_edf.is_file():
            chb_rows, chb_summary, chb_support = _benchmark_chb(chb_edf)
            chb_rows.to_csv(output / "chb04_28_two_seizure_features.csv", index=False)
            pd.DataFrame(chb_summary).to_csv(
                output / "chb04_28_exploratory_separation.csv", index=False
            )
            pd.DataFrame(chb_support).to_csv(
                output / "chb04_28_support_by_phase.csv", index=False
            )

    metadata = {
        "implementation": "Varon-2015 deterministic 80-beat PRSA/BPRSA features",
        "window": "80 RR-defined beats, step 1, overlap 79",
        "resampling": "4 Hz scipy CubicSpline (not-a-knot)",
        "prsa": "RR driver, RR target, driver[i] < driver[i-1] anchors",
        "bprsa": "R-peak-amplitude driver, RR target, same anchor rule",
        "curve": "L=20 samples = +/-5 seconds, 41 samples including anchor",
        "local_slope": "(curve[+1] - curve[-1]) / 2",
        "long_range_slope": "(curve[+L] - curve[-L]) / (2L+1)",
        "classifier_applied": False,
        "calibration_applied": False,
        "mit_dataset": str(mit_dataset),
        "mit_records": records,
        "mit_ground_truth": "expert atr beat timestamps; no seizure labels",
        "chb_edf": str(args.chb_edf.expanduser().resolve()),
        "chb_ground_truth": "two seizure intervals from chb04-summary.txt; no manual ECG landmarks",
        "chb_seizure_intervals_s": [list(value) for value in CHB_SEIZURES],
        "chb_claim": (
            "descriptive, overlapping-window, single-patient/two-seizure feature "
            "separation only; not sensitivity, PPV, specificity, false-alarm rate, "
            "or generalization"
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "benchmark_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (output / "SUMMARY.md").write_text(
        _summary_markdown(pooled, sensitivity, chb_summary, chb_support),
        encoding="utf-8",
    )
    print(json.dumps(pooled, indent=2), flush=True)
    if chb_summary:
        print(pd.DataFrame(chb_summary).to_string(index=False), flush=True)
    return 0


def _benchmark_mit(
    dataset: Path, records: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, object]] = []
    feature_frames: list[pd.DataFrame] = []
    sensitivity_rows: list[dict[str, object]] = []
    for record_id in records:
        base = dataset / record_id
        header = wfdb.rdheader(str(base))
        record = wfdb.rdrecord(str(base), physical=True)
        if record.p_signal is None:
            raise ValueError(f"{record_id}: no physical signals returned")
        signal = np.asarray(record.p_signal, dtype=np.float64)
        peaks, _ = load_wfdb_beat_annotations(base)
        rr = np.diff(peaks).astype(np.float64) * 1000.0 / float(header.fs)
        times = peaks[1:].astype(np.float64) / float(header.fs)
        primary_ramp = signal[peaks[1:], 0]
        result = calculate_prsa_bprsa_feature_arrays(
            rr, primary_ramp, end_time_s=times
        )
        frame = result.features.loc[result.features["window_beat_count"] == 80].copy()
        frame.insert(0, "record_id", record_id)
        frame.insert(1, "lead_name", str(header.sig_name[0]))
        feature_frames.append(
            frame[
                [
                    "record_id",
                    "lead_name",
                    "end_time_s",
                    "window_start_time_s",
                    "window_end_time_s",
                    "window_elapsed_s",
                    "prsa_anchor_count",
                    "bprsa_anchor_count",
                    "resampled_rr_nonpositive_count",
                    "rr_cubic_range_overshoot_fraction",
                    "r_amplitude_cubic_range_overshoot_fraction",
                    "numerical_quality_pass",
                    "feature_defined",
                    *FEATURES,
                ]
            ]
        )
        summaries.append(
            {
                "record_id": record_id,
                "lead_name": str(header.sig_name[0]),
                "expert_beat_count": int(peaks.size),
                "eligible_80_beat_windows": int(frame.shape[0]),
                "defined_windows": int(frame["feature_defined"].sum()),
                "defined_coverage": float(frame["feature_defined"].mean()),
                "numerical_quality_pass_windows": int(
                    frame["numerical_quality_pass"].sum()
                ),
                "numerical_quality_pass_coverage": float(
                    frame["numerical_quality_pass"].mean()
                ),
                "median_prsa_anchor_count": float(frame["prsa_anchor_count"].median()),
                "minimum_prsa_anchor_count": int(frame["prsa_anchor_count"].min()),
                "median_bprsa_anchor_count": float(frame["bprsa_anchor_count"].median()),
                "minimum_bprsa_anchor_count": int(frame["bprsa_anchor_count"].min()),
            }
        )

        if record_id in LEAD_POLARITY_RECORDS:
            inverted = calculate_prsa_bprsa_feature_arrays(
                rr, -primary_ramp, end_time_s=times
            )
            sensitivity_rows.extend(
                _compare_tracks(
                    record_id,
                    str(header.sig_name[0]),
                    "global_polarity_inversion",
                    result.features,
                    inverted.features,
                )
            )
            if signal.shape[1] > 1:
                secondary = calculate_prsa_bprsa_feature_arrays(
                    rr, signal[peaks[1:], 1], end_time_s=times
                )
                sensitivity_rows.extend(
                    _compare_tracks(
                        record_id,
                        f"{header.sig_name[0]}_vs_{header.sig_name[1]}",
                        "lead_change",
                        result.features,
                        secondary.features,
                    )
                )
        print(f"MIT completed {record_id}", flush=True)
    return (
        pd.DataFrame(summaries),
        pd.concat(feature_frames, ignore_index=True),
        pd.DataFrame(sensitivity_rows),
    )


def _compare_tracks(
    record_id: str,
    lead_pair: str,
    comparison: str,
    first: pd.DataFrame,
    second: pd.DataFrame,
) -> list[dict[str, object]]:
    usable = first["feature_defined"].to_numpy(bool) & second[
        "feature_defined"
    ].to_numpy(bool)
    rows: list[dict[str, object]] = []
    for feature in FEATURES:
        x = first.loc[usable, feature].to_numpy(float)
        y = second.loc[usable, feature].to_numpy(float)
        correlation = (
            float(np.corrcoef(x, y)[0, 1])
            if x.size > 1 and np.std(x) > 0 and np.std(y) > 0
            else np.nan
        )
        rows.append(
            {
                "record_id": record_id,
                "lead_pair": lead_pair,
                "comparison": comparison,
                "feature": feature,
                "paired_defined_windows": int(x.size),
                "pearson_r": correlation,
                "median_absolute_difference": (
                    float(np.median(np.abs(x - y))) if x.size else np.nan
                ),
                "maximum_absolute_difference": (
                    float(np.max(np.abs(x - y))) if x.size else np.nan
                ),
            }
        )
    return rows


def _mit_pooled_summary(
    records: pd.DataFrame, features: pd.DataFrame
) -> dict[str, object]:
    eligible = int(records["eligible_80_beat_windows"].sum())
    defined = int(records["defined_windows"].sum())
    return {
        "record_count": int(records.shape[0]),
        "expert_beat_count": int(records["expert_beat_count"].sum()),
        "eligible_80_beat_windows": eligible,
        "defined_windows": defined,
        "defined_coverage": float(defined / eligible) if eligible else np.nan,
        "numerical_quality_pass_windows": int(
            features["numerical_quality_pass"].sum()
        ),
        "numerical_quality_pass_coverage": float(
            features["numerical_quality_pass"].mean()
        ),
        "windows_with_nonpositive_cubic_rr": int(
            (features["resampled_rr_nonpositive_count"] > 0).sum()
        ),
        "median_rr_cubic_range_overshoot_fraction": float(
            features["rr_cubic_range_overshoot_fraction"].median()
        ),
        "records_with_complete_defined_coverage": int(
            np.isclose(records["defined_coverage"], 1.0).sum()
        ),
        "median_prsa_anchor_count_per_window": float(
            features["prsa_anchor_count"].median()
        ),
        "p05_prsa_anchor_count_per_window": float(
            features["prsa_anchor_count"].quantile(0.05)
        ),
        "median_bprsa_anchor_count_per_window": float(
            features["bprsa_anchor_count"].median()
        ),
        "p05_bprsa_anchor_count_per_window": float(
            features["bprsa_anchor_count"].quantile(0.05)
        ),
        "seizure_accuracy_measured": False,
    }


def _benchmark_chb(
    edf_path: Path,
) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    rows: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    support_summaries: list[dict[str, object]] = []
    config = NeuroKitZhaiConfig(compute_inverted_context=False)
    for seizure_id, onset_s, offset_s in CHB_SEIZURES:
        start_s = max(0.0, onset_s - 180.0)
        stop_s = offset_s + 120.0
        segment = load_ecg_segment(
            edf_path,
            channel_label="ECG",
            start_s=start_s,
            duration_s=stop_s - start_s,
        )
        detector = run_neurokit_zhai_array(
            segment.samples,
            segment.channel.sampling_rate_hz,
            segment_start_s=segment.start_s,
            config=config,
            source_metadata={"benchmark": "PRSA/BPRSA CHB04_28"},
        )
        track = detector.selected.neurokit
        peaks = track.events["primary_timestamp_sample"].to_numpy(np.int64)
        result = extract_prsa_bprsa_features(
            segment.samples,
            peaks,
            segment.channel.sampling_rate_hz,
            segment_start_s=segment.start_s,
            orientation=detector.selected_orientation,
            rr_supported=track.rr_intervals["rr_supported"].to_numpy(bool),
            r_peak_supported=track.events["qrs_supported"].to_numpy(bool),
            lead_name="ECG",
            anchor_track="neurokit",
        )
        frame = result.features.loc[result.features["feature_defined"]].copy()
        frame.insert(0, "seizure_id", seizure_id)
        frame["seizure_onset_s"] = onset_s
        frame["seizure_offset_s"] = offset_s
        frame["phase"] = np.select(
            [
                frame["end_time_s"] < onset_s - 60.0,
                frame["end_time_s"] < onset_s,
                frame["end_time_s"] <= offset_s,
            ],
            ["baseline", "preictal_60s", "ictal"],
            default="postictal",
        )
        rows.append(frame)
        for phase, phase_frame in frame.groupby("phase"):
            support_summaries.append(
                {
                    "seizure_id": seizure_id,
                    "phase": phase,
                    "defined_window_count": int(phase_frame.shape[0]),
                    "strict_reliable_window_count": int(
                        phase_frame["feature_reliable"].sum()
                    ),
                    "strict_reliable_fraction": float(
                        phase_frame["feature_reliable"].mean()
                    ),
                    "numerical_quality_pass_fraction": float(
                        phase_frame["numerical_quality_pass"].mean()
                    ),
                    "median_rr_detector_support_coverage": float(
                        phase_frame["rr_reliability_coverage"].median()
                    ),
                    "median_r_amplitude_support_coverage": float(
                        phase_frame["r_amplitude_reliability_coverage"].median()
                    ),
                }
            )
        for feature in FEATURES:
            baseline = frame.loc[frame["phase"] == "baseline", feature].to_numpy(float)
            ictal = frame.loc[frame["phase"] == "ictal", feature].to_numpy(float)
            reliable_baseline = frame.loc[
                (frame["phase"] == "baseline") & frame["feature_reliable"], feature
            ].to_numpy(float)
            reliable_ictal = frame.loc[
                (frame["phase"] == "ictal") & frame["feature_reliable"], feature
            ].to_numpy(float)
            auc = _auc(ictal, baseline)
            reliable_auc = _auc(reliable_ictal, reliable_baseline)
            median = float(np.median(baseline)) if baseline.size else np.nan
            mad = (
                float(np.median(np.abs(baseline - median)))
                if baseline.size
                else np.nan
            )
            scale = 1.4826 * mad
            ictal_abs_z = (
                np.abs((ictal - median) / scale)
                if ictal.size and np.isfinite(scale) and scale > 0
                else np.array([], dtype=float)
            )
            summaries.append(
                {
                    "seizure_id": seizure_id,
                    "feature": feature,
                    "baseline_window_count": int(baseline.size),
                    "ictal_window_count": int(ictal.size),
                    "baseline_median": median,
                    "ictal_median": float(np.median(ictal)) if ictal.size else np.nan,
                    "signed_median_change": (
                        float(np.median(ictal) - median)
                        if baseline.size and ictal.size
                        else np.nan
                    ),
                    "direction_free_window_auc": (
                        float(max(auc, 1.0 - auc)) if np.isfinite(auc) else np.nan
                    ),
                    "window_auc_support_status": (
                        "all mathematically defined windows, including detector-disagreement windows"
                    ),
                    "strict_reliable_baseline_window_count": int(
                        reliable_baseline.size
                    ),
                    "strict_reliable_ictal_window_count": int(reliable_ictal.size),
                    "direction_free_strict_reliable_window_auc": (
                        float(max(reliable_auc, 1.0 - reliable_auc))
                        if np.isfinite(reliable_auc)
                        else np.nan
                    ),
                    "maximum_ictal_absolute_robust_z": (
                        float(np.max(ictal_abs_z)) if ictal_abs_z.size else np.nan
                    ),
                    "ictal_fraction_absolute_robust_z_gt_3": (
                        float(np.mean(ictal_abs_z > 3.0)) if ictal_abs_z.size else np.nan
                    ),
                    "overlapping_windows_independent": False,
                    "classifier_applied": False,
                }
            )
        print(f"CHB04_28 completed seizure {seizure_id}", flush=True)
    return pd.concat(rows, ignore_index=True), summaries, support_summaries


def _auc(positive: np.ndarray, negative: np.ndarray) -> float:
    if positive.size == 0 or negative.size == 0:
        return np.nan
    combined = np.concatenate([positive, negative])
    ranks = rankdata(combined, method="average")
    positive_rank_sum = float(np.sum(ranks[: positive.size]))
    return float(
        (positive_rank_sum - positive.size * (positive.size + 1) / 2)
        / (positive.size * negative.size)
    )


def _summary_markdown(
    pooled: dict[str, object],
    sensitivity: pd.DataFrame,
    chb: list[dict[str, object]],
    chb_support: list[dict[str, object]],
) -> str:
    lines = [
        "# PRSA/BPRSA available-data benchmark",
        "",
        "## MIT-BIH expert-anchor robustness",
        "",
        f"- Records: {pooled['record_count']}",
        f"- Expert beats: {pooled['expert_beat_count']}",
        f"- Eligible 80-beat windows: {pooled['eligible_80_beat_windows']}",
        f"- Mathematically defined windows: {pooled['defined_windows']} ({100 * float(pooled['defined_coverage']):.3f}%)",
        f"- Positive-resampled-RR numerical gate: {pooled['numerical_quality_pass_windows']} ({100 * float(pooled['numerical_quality_pass_coverage']):.3f}%)",
        f"- Windows containing nonpositive cubic-spline RR samples: {pooled['windows_with_nonpositive_cubic_rr']}",
        f"- Median usable PRSA/BPRSA anchors: {pooled['median_prsa_anchor_count_per_window']:.1f} / {pooled['median_bprsa_anchor_count_per_window']:.1f}",
        "- This is not seizure accuracy: MIT-BIH has no seizure labels.",
        "",
        "## Lead and polarity audit",
        "",
    ]
    if sensitivity.empty:
        lines.append("No lead/polarity comparison records were selected.")
    else:
        for comparison, group in sensitivity.groupby("comparison"):
            bprsa = group[group["feature"].str.startswith("bprsa")]
            prsa = group[group["feature"].str.startswith("prsa")]
            lines.append(
                f"- {comparison}: median PRSA Pearson r={prsa['pearson_r'].median():.4f}; median BPRSA Pearson r={bprsa['pearson_r'].median():.4f}."
            )
    lines.extend(
        [
            "",
            "## CHB04_28 exploratory seizure-interval comparison",
            "",
            "Single patient, two seizures, overlapping windows, automatic R peaks. Direction-free AUC is descriptive window separation—not a validated detector result.",
            "",
        ]
    )
    if chb:
        table = pd.DataFrame(chb)
        support = pd.DataFrame(chb_support)
        for seizure_id, group in table.groupby("seizure_id"):
            best = group.sort_values("direction_free_window_auc", ascending=False).iloc[0]
            ictal_support = support[
                (support["seizure_id"] == seizure_id)
                & (support["phase"] == "ictal")
            ].iloc[0]
            lines.append(
                f"- Seizure {seizure_id}: strongest unfiltered descriptive feature was `{best['feature']}` with direction-free window AUC {best['direction_free_window_auc']:.3f}; strict reliable ictal windows were {int(ictal_support['strict_reliable_window_count'])}/{int(ictal_support['defined_window_count'])}."
            )
        if int(support.loc[support["phase"] == "ictal", "strict_reliable_window_count"].sum()) == 0:
            lines.append(
                "- No ictal window passed strict detector-support reliability. The unfiltered AUC values therefore cannot be interpreted as physiological seizure discrimination."
            )
    else:
        lines.append("CHB04_28 was not run or unavailable.")
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "No KSC or other classifier was fitted. No sensitivity, PPV, specificity, false-alarm rate, or patient-level generalization was measured.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
