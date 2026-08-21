"""Measure PRSA/BPRSA fidelity under calibrated electrode-motion noise.

Clean MIT-BIH records 118/119 and their Noise Stress Test Database copies
share the same time axis.  Expert clean-source R annotations are used for both
signals so the benchmark isolates waveform-noise effects from detector error.
RR-only PRSA is therefore an exact negative control; BPRSA is the
polarity-preserving R-amplitude context under test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import wfdb


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecg_cascade.prsa_bprsa import calculate_prsa_bprsa_feature_arrays  # noqa: E402
from ecg_cascade.wfdb_io import load_wfdb_beat_annotations  # noqa: E402


SOURCES = ("118", "119")
SNR_SUFFIXES = ((24, "e24"), (18, "e18"), (12, "e12"), (6, "e06"), (0, "e00"), (-6, "e_6"))
FEATURES = (
    "mean_rr80_ms",
    "sdnn80_ms",
    "prsa_s_rr_ms_per_sample",
    "prsa_delta_rr_ms_per_sample",
    "bprsa_s_r_ms_per_sample",
    "bprsa_delta_r_ms_per_sample",
)
RR_ONLY_FEATURES = FEATURES[:4]
BPRSA_FEATURES = FEATURES[4:]
NOISE_START_S = 5.0 * 60.0
NOISE_DURATION_S = 2.0 * 60.0
NOISE_CYCLE_S = 4.0 * 60.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mitdb",
        type=Path,
        default=(
            PROJECT.parent
            / "Datasets"
            / "mit-bih-arrhythmia-database-1.0"
            / "mit-bih-arrhythmia-database-1.0.0"
        ),
    )
    parser.add_argument(
        "--nstdb",
        type=Path,
        default=(
            PROJECT.parent
            / "Datasets"
            / "mit-bih-noise-stress-test-database-1.0"
            / "mit-bih-noise-stress-test-database-1.0.0"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "outputs" / "prsa_bprsa_noise_fidelity_20260819",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mitdb = args.mitdb.expanduser().resolve()
    nstdb = args.nstdb.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    detail_rows: list[dict[str, object]] = []

    for source_id in SOURCES:
        clean, fs, clean_lead = _load_signal(mitdb / source_id)
        peaks, _ = load_wfdb_beat_annotations(mitdb / source_id)
        rr = np.diff(peaks).astype(float) * 1000.0 / fs
        times = peaks[1:].astype(float) / fs
        clean_result = calculate_prsa_bprsa_feature_arrays(
            rr,
            clean[peaks[1:]],
            end_time_s=times,
        ).features
        intervals = _noise_intervals(clean.size / fs)

        for snr_db, suffix in SNR_SUFFIXES:
            noisy, noisy_fs, noisy_lead = _load_signal(nstdb / f"{source_id}{suffix}")
            if noisy_fs != fs or noisy.shape != clean.shape:
                raise ValueError(
                    f"{source_id}{suffix}: clean/noisy time axes do not match"
                )
            noisy_result = calculate_prsa_bprsa_feature_arrays(
                rr,
                noisy[peaks[1:]],
                end_time_s=times,
            ).features
            eligible = np.zeros(times.size, dtype=bool)
            window_start = clean_result["window_start_time_s"].to_numpy(float)
            window_end = clean_result["window_end_time_s"].to_numpy(float)
            for start_s, end_s in intervals:
                eligible |= (window_start >= start_s) & (window_end <= end_s)
            paired_defined = (
                eligible
                & clean_result["feature_defined"].to_numpy(bool)
                & noisy_result["feature_defined"].to_numpy(bool)
            )
            for feature in FEATURES:
                clean_values = clean_result.loc[paired_defined, feature].to_numpy(float)
                noisy_values = noisy_result.loc[paired_defined, feature].to_numpy(float)
                delta = noisy_values - clean_values
                scale = float(np.std(clean_values, ddof=0)) if clean_values.size else np.nan
                detail_rows.append(
                    {
                        "source_record": source_id,
                        "clean_lead": clean_lead,
                        "noisy_lead": noisy_lead,
                        "snr_db": snr_db,
                        "feature": feature,
                        "feature_family": "RR-only PRSA" if feature in RR_ONLY_FEATURES else "R-amplitude BPRSA",
                        "paired_window_count": int(clean_values.size),
                        "pearson_r": _correlation(clean_values, noisy_values),
                        "median_absolute_error": float(np.median(np.abs(delta))) if delta.size else np.nan,
                        "rmse": float(np.sqrt(np.mean(delta**2))) if delta.size else np.nan,
                        "normalized_rmse_by_clean_sd": (
                            float(np.sqrt(np.mean(delta**2)) / scale)
                            if delta.size and np.isfinite(scale) and scale > 0
                            else np.nan
                        ),
                        "exact_match_fraction": float(np.mean(delta == 0.0)) if delta.size else np.nan,
                        "sign_agreement_fraction": float(
                            np.mean(np.signbit(clean_values) == np.signbit(noisy_values))
                        ) if delta.size else np.nan,
                        "clean_numerical_quality_pass_fraction": float(
                            clean_result.loc[eligible, "numerical_quality_pass"].mean()
                        ),
                        "noisy_numerical_quality_pass_fraction": float(
                            noisy_result.loc[eligible, "numerical_quality_pass"].mean()
                        ),
                        "expert_r_anchors_used_for_both": True,
                    }
                )
            print(f"completed {source_id} at {snr_db:+d} dB", flush=True)

    detail = pd.DataFrame(detail_rows)
    detail.to_csv(output / "record_feature_fidelity.csv", index=False)
    pooled = (
        detail.groupby(["snr_db", "feature", "feature_family"], as_index=False)
        .agg(
            record_count=("source_record", "nunique"),
            paired_window_count=("paired_window_count", "sum"),
            median_record_pearson_r=("pearson_r", "median"),
            median_record_normalized_rmse=("normalized_rmse_by_clean_sd", "median"),
            median_record_exact_match_fraction=("exact_match_fraction", "median"),
            median_record_sign_agreement_fraction=("sign_agreement_fraction", "median"),
        )
        .sort_values(["feature_family", "feature", "snr_db"], ascending=[True, True, False])
    )
    pooled.to_csv(output / "pooled_fidelity_by_snr.csv", index=False)
    rr = detail[detail["feature_family"] == "RR-only PRSA"]
    bprsa = detail[detail["feature_family"] == "R-amplitude BPRSA"]
    summary = {
        "source_records": list(SOURCES),
        "snr_db": [value for value, _ in SNR_SUFFIXES],
        "record_feature_comparisons": int(detail.shape[0]),
        "rr_only_exact_match_fraction": float(
            np.average(rr["exact_match_fraction"], weights=rr["paired_window_count"])
        ),
        "rr_only_maximum_median_absolute_error": float(rr["median_absolute_error"].max()),
        "bprsa_median_record_correlation_by_snr": {
            str(int(snr)): float(group["pearson_r"].median())
            for snr, group in bprsa.groupby("snr_db")
        },
        "bprsa_median_record_sign_agreement_by_snr": {
            str(int(snr)): float(group["sign_agreement_fraction"].median())
            for snr, group in bprsa.groupby("snr_db")
        },
        "numerical_quality_gate_detects_amplitude_noise": False,
        "reason": (
            "the current numerical gate checks cubic RR positivity; with shared "
            "expert R timestamps it is identical for clean and noisy ECG"
        ),
        "seizure_accuracy_measured": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "SUMMARY.md").write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def _load_signal(base: Path) -> tuple[np.ndarray, float, str]:
    header = wfdb.rdheader(str(base))
    record = wfdb.rdrecord(str(base), physical=True)
    if record.p_signal is None:
        raise ValueError(f"{base}: no physical signal")
    return (
        np.asarray(record.p_signal[:, 0], dtype=float),
        float(header.fs),
        str(header.sig_name[0]),
    )


def _noise_intervals(duration_s: float) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    start = NOISE_START_S
    while start + NOISE_DURATION_S <= duration_s + 1e-9:
        intervals.append((start, start + NOISE_DURATION_S))
        start += NOISE_CYCLE_S
    return intervals


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    if first.size < 2 or np.std(first) == 0 or np.std(second) == 0:
        return np.nan
    return float(np.corrcoef(first, second)[0, 1])


def _markdown(summary: dict[str, object]) -> str:
    correlations = summary["bprsa_median_record_correlation_by_snr"]
    signs = summary["bprsa_median_record_sign_agreement_by_snr"]
    lines = [
        "# PRSA/BPRSA electrode-motion noise fidelity",
        "",
        "Expert R timestamps were held fixed between clean MIT-BIH records 118/119 and their calibrated NSTDB electrode-motion-noise copies.",
        "",
        f"- RR-only feature exact-match fraction: {summary['rr_only_exact_match_fraction']:.6f}",
        f"- Maximum RR-only median absolute error: {summary['rr_only_maximum_median_absolute_error']:.6g}",
        "- BPRSA median record correlation / sign agreement:",
    ]
    for snr in (24, 18, 12, 6, 0, -6):
        lines.append(f"  - {snr:+d} dB: r={correlations[str(snr)]:.4f}, sign agreement={signs[str(snr)]:.4f}")
    lines.extend(
        [
            "- The RR-positivity numerical gate cannot detect pure amplitude corruption when R timestamps are fixed.",
            "- Therefore BPRSA needs the separate general ECG artifact/lead-stability context before model eligibility.",
            "- This is measurement fidelity under artifact, not seizure accuracy.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
