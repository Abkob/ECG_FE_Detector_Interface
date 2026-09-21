"""Independently recalculate binary mappings and timing guards from preserved labels."""

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

from ecg_cascade.comprehensive_dataset import AAMI_BEAT_CLASS, _nstdb_noise_state  # noqa: E402


def comparable(series: pd.Series) -> pd.Series:
    return series.astype("Int8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = pd.read_csv(
        args.matrix,
        low_memory=False,
        dtype={
            "label_quality_annotator_1_original": "string",
            "label_quality_annotator_2_original": "string",
            "label_quality_annotator_3_original": "string",
            "label_quality_consensus_original": "string",
        },
    )
    checks: list[dict[str, object]] = []

    def add(name: str, eligible: pd.Series, mismatch: pd.Series, detail: str) -> None:
        checks.append(
            {
                "check": name,
                "eligible_rows": int(eligible.sum()),
                "mismatches": int((eligible & mismatch).sum()),
                "passed": int((eligible & mismatch).sum()) == 0,
                "detail": detail,
            }
        )

    symbols = data["label_beat_symbol_original"].astype("string")
    expected_aami = symbols.map(AAMI_BEAT_CLASS).astype("string")
    aami_known = symbols.isin(AAMI_BEAT_CLASS)
    add(
        "original beat symbol to AAMI superclass",
        aami_known,
        expected_aami.ne(data["label_beat_aami_superclass_derived"].astype("string")),
        "Exact WFDB symbols are mapped only through the frozen AAMI dictionary.",
    )
    expected_abnormal = expected_aami.ne("N").astype("Int8").where(aami_known)
    add(
        "AAMI superclass to abnormal-beat binary",
        aami_known,
        comparable(data["target_abnormal_beat_binary"]).ne(expected_abnormal),
        "N=0; S/V/F/Q=1; unknown remains null.",
    )

    quality = data["label_quality_consensus_original"].astype("string")
    pure = data["label_quality_trailing_10s_pure"].eq(True)
    quality_known = pure & quality.isin(["1", "2", "3"])
    expected_quality_artifact = quality.isin(["2", "3"]).astype("Int8")
    expected_noise = data["label_noise_active"].astype("boolean").astype("Int8")
    noise_known = data["label_noise_active"].notna()
    expected_artifact = pd.Series(pd.NA, index=data.index, dtype="Int8")
    expected_artifact.loc[quality_known] = expected_quality_artifact.loc[quality_known]
    expected_artifact.loc[noise_known & ~quality_known] = expected_noise.loc[noise_known & ~quality_known]
    artifact_known = quality_known | noise_known
    add(
        "composite artifact mapping",
        artifact_known,
        comparable(data["target_artifact_binary"]).ne(expected_artifact),
        "Pure BUT-QDB 1=0, 2/3=1; otherwise known NSTDB inactive=0, active=1.",
    )
    strict = pure & quality.isin(["1", "3"])
    expected_unusable = quality.eq("3").astype("Int8")
    add(
        "strict unusable mapping",
        strict,
        comparable(data["target_signal_unusable_binary"]).ne(expected_unusable),
        "Pure BUT-QDB class 1=0 and class 3=1; class 2 is excluded.",
    )

    add(
        "NSTDB source flag to binary",
        noise_known,
        comparable(data["target_noise_active_binary"]).ne(expected_noise),
        "The preserved boolean source schedule is converted to 0/1 without inversion.",
    )
    schedule_rows = data["dataset_key"].eq("nstdb")
    recalculated_schedule = data.loc[schedule_rows, "observation_time_s"].map(
        lambda value: int(_nstdb_noise_state(float(value))[0])
    ).astype("Int8")
    schedule_mismatch = pd.Series(False, index=data.index)
    schedule_mismatch.loc[schedule_rows] = comparable(
        data.loc[schedule_rows, "target_noise_active_binary"]
    ).ne(recalculated_schedule)
    add(
        "NSTDB official alternating schedule",
        schedule_rows,
        schedule_mismatch,
        "Time is recomputed against the official two-minute-on/two-minute-off schedule after minute five.",
    )

    seizure_known = data["label_seizure_binary"].notna()
    expected_seizure = data["label_seizure_binary"].eq("ictal").astype("Int8")
    add(
        "seizure interval label to binary",
        seizure_known,
        comparable(data["target_seizure_binary"]).ne(expected_seizure),
        "Only the exact ictal source state maps to 1.",
    )

    quality_timing = data["label_quality_interval_start_s"].notna()
    timing_mismatch = ~(
        data["observation_time_s"].ge(data["label_quality_interval_start_s"])
        & data["observation_time_s"].le(data["label_quality_interval_end_s"])
    )
    add(
        "quality interval containment",
        quality_timing,
        timing_mismatch,
        "Every attached quality class must contain the observation time.",
    )
    pure_timing = pure & quality_timing
    pure_mismatch = data["observation_time_s"].sub(10.0).lt(data["label_quality_interval_start_s"])
    add(
        "quality trailing-window purity",
        pure_timing,
        pure_mismatch,
        "A pure label requires the full causal ten-second feature window inside one class interval.",
    )

    beat_matched = data["label_beat_match_within_75ms"].eq(True)
    beat_tolerance_mismatch = data["label_beat_match_offset_ms"].abs().gt(75.0 + 1e-9)
    add(
        "beat-label 75 ms attachment tolerance",
        beat_matched,
        beat_tolerance_mismatch,
        "A matched beat label may not exceed the frozen absolute timing tolerance.",
    )

    frame = pd.DataFrame(checks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    summary = {
        "checks": int(frame.shape[0]),
        "passed": int(frame["passed"].sum()),
        "mismatches": int(frame["mismatches"].sum()),
        "all_passed": bool(frame["passed"].all()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(frame.to_string(index=False))
    print(json.dumps(summary, indent=2))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
