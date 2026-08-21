"""Audit causal PRSA/BPRSA attachment to the fusion matrix.

The input is the expert-anchor MIT-BIH feature table produced by
``benchmark_prsa_bprsa_available_data.py``.  This script deliberately tests
matrix mechanics, not seizure discrimination: it inserts exact and between-
measurement probe times, independently finds the latest permissible context
row, and checks every attached paper feature and gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecg_cascade.fusion import build_fusion_ready_table  # noqa: E402


FEATURES = (
    "mean_rr80_ms",
    "sdnn80_ms",
    "prsa_s_rr_ms_per_sample",
    "prsa_delta_rr_ms_per_sample",
    "bprsa_s_r_ms_per_sample",
    "bprsa_delta_r_ms_per_sample",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=Path,
        default=(
            PROJECT
            / "outputs"
            / "prsa_bprsa_context_audit_20260819"
            / "mit_expert_anchor_features.csv.gz"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "outputs" / "prsa_bprsa_context_matrix_audit_20260819",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_path = args.features.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(source_path)
    required = {"record_id", "end_time_s", "feature_defined", "numerical_quality_pass", *FEATURES}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Input feature table is missing columns: {sorted(missing)}")

    record_rows: list[dict[str, object]] = []
    for record_id, record in source.groupby("record_id", sort=True):
        record = record.sort_values("end_time_s", kind="stable").reset_index(drop=True)
        # Expert R anchors have complete timestamp support.  The remaining
        # reliability gate is the positive-resampled-RR numerical check.
        record["feature_reliable"] = record["numerical_quality_pass"].astype(bool)
        times = record["end_time_s"].to_numpy(dtype=float)
        midpoints = (times[:-1] + times[1:]) / 2.0
        probe_times = np.unique(np.concatenate(([times[0] - 1e-6], times, midpoints)))
        morphology = pd.DataFrame(
            {
                "window_end_time_s": probe_times,
                "published_varon_core_defined": True,
                "support_context_pass": True,
            }
        )
        hrv = pd.DataFrame(
            {
                "end_time_s": probe_times,
                "feature_defined": True,
                "feature_reliable": True,
            }
        )
        without_context = build_fusion_ready_table(
            hrv,
            morphology,
            anchor_track=f"mit_expert_{record_id}",
        )
        joined = build_fusion_ready_table(
            hrv,
            morphology,
            anchor_track=f"mit_expert_{record_id}",
            prsa_bprsa_context=record,
        )

        expected_indices = np.searchsorted(times, probe_times, side="right") - 1
        available = expected_indices >= 0
        attached_times = joined["prsa_context_measurement_time_s"].to_numpy(float)
        expected_times = np.full(probe_times.size, np.nan)
        expected_times[available] = times[expected_indices[available]]
        time_mismatches = int(
            np.sum(~np.isclose(attached_times, expected_times, equal_nan=True, atol=1e-12))
        )
        value_mismatches = 0
        for feature in FEATURES:
            actual = joined[f"prsa_context_{feature}"].to_numpy(float)
            expected = np.full(probe_times.size, np.nan)
            expected[available] = record[feature].to_numpy(float)[
                expected_indices[available]
            ]
            value_mismatches += int(
                np.sum(~np.isclose(actual, expected, equal_nan=True, rtol=1e-12, atol=1e-12))
            )

        expected_usable = np.zeros(probe_times.size, dtype=bool)
        expected_usable[available] = record["numerical_quality_pass"].to_numpy(bool)[
            expected_indices[available]
        ]
        actual_usable = joined["prsa_context_usable"].to_numpy(bool)
        record_rows.append(
            {
                "record_id": str(record_id),
                "source_context_rows": int(record.shape[0]),
                "matrix_probe_rows": int(joined.shape[0]),
                "before_warmup_probe_rows": int((~available).sum()),
                "time_mismatch_count": time_mismatches,
                "six_value_mismatch_count": value_mismatches,
                "future_leak_count": int(
                    np.sum(attached_times[available] > probe_times[available] + 1e-12)
                ),
                "negative_context_age_count": int(
                    np.sum(joined.loc[available, "prsa_context_age_s"].to_numpy(float) < -1e-12)
                ),
                "availability_flag_mismatch_count": int(
                    np.sum(joined["prsa_context_available"].to_numpy(bool) != available)
                ),
                "usable_flag_mismatch_count": int(np.sum(actual_usable != expected_usable)),
                "core_definition_gate_change_count": int(
                    np.sum(
                        joined["fusion_measurements_defined"].to_numpy(bool)
                        != without_context["fusion_measurements_defined"].to_numpy(bool)
                    )
                ),
                "core_context_gate_change_count": int(
                    np.sum(
                        joined["fusion_context_pass"].to_numpy(bool)
                        != without_context["fusion_context_pass"].to_numpy(bool)
                    )
                ),
                "context_marked_as_core_count": int(
                    joined["prsa_context_affects_core_fusion_gate"].sum()
                ),
                "signal_quality_incorrectly_marked_attached_count": int(
                    joined["prsa_context_signal_quality_attached"].sum()
                ),
                "context_incorrectly_marked_model_eligible_count": int(
                    joined["prsa_context_model_eligible"].sum()
                ),
            }
        )

    records = pd.DataFrame(record_rows)
    records.to_csv(output / "per_record_matrix_audit.csv", index=False)
    count_columns = [
        column
        for column in records.columns
        if column.endswith("_count") or column.endswith("_rows")
    ]
    totals = {column: int(records[column].sum()) for column in count_columns}
    failure_columns = [
        column
        for column in count_columns
        if column not in {"source_context_rows", "matrix_probe_rows", "before_warmup_probe_rows"}
    ]
    summary = {
        "input": str(source_path),
        "record_count": int(records.shape[0]),
        **totals,
        "all_causal_attachment_checks_passed": bool(
            all(int(records[column].sum()) == 0 for column in failure_columns)
        ),
        "seizure_accuracy_measured": False,
        "claim": "matrix causality and gating correctness only",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "SUMMARY.md").write_text(
        "\n".join(
            [
                "# PRSA/BPRSA context-matrix audit",
                "",
                f"- MIT-BIH records: {summary['record_count']}",
                f"- Source context rows: {summary['source_context_rows']}",
                f"- Exact/between-time matrix probes: {summary['matrix_probe_rows']}",
                f"- Time mismatches: {summary['time_mismatch_count']}",
                f"- Six-value mismatches: {summary['six_value_mismatch_count']}",
                f"- Future leaks: {summary['future_leak_count']}",
                f"- Usability-flag mismatches: {summary['usable_flag_mismatch_count']}",
                f"- Core definition/context gate changes: {summary['core_definition_gate_change_count']} / {summary['core_context_gate_change_count']}",
                f"- Incorrect signal-quality-attached/model-eligible flags: {summary['signal_quality_incorrectly_marked_attached_count']} / {summary['context_incorrectly_marked_model_eligible_count']}",
                f"- All causal attachment checks passed: {summary['all_causal_attachment_checks_passed']}",
                "- This validates matrix mechanics, not seizure discrimination.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["all_causal_attachment_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
