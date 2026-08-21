"""Run the deterministic patient-template morphology branch on one WFDB lead."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecg_cascade.patient_template_morphology import (  # noqa: E402
    extract_patient_template_morphology,
)
from ecg_cascade.wfdb_io import (  # noqa: E402
    load_wfdb_beat_annotations,
    load_wfdb_segment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--record",
        type=Path,
        required=True,
        help="WFDB record path without .hea/.dat suffixes.",
    )
    parser.add_argument("--channel", default="0")
    parser.add_argument("--annotation-extension", default="atr")
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--calibration-beats", type=int, default=20)
    parser.add_argument("--patient-id", required=True)
    parser.add_argument("--lead-name", default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    channel: int | str = (
        int(args.channel) if str(args.channel).isdigit() else str(args.channel)
    )
    segment = load_wfdb_segment(
        args.record,
        channel=channel,
        start_s=args.start_s,
        duration_s=args.duration_s,
    )
    end_sample = segment.start_sample + segment.samples.size
    peaks, symbols = load_wfdb_beat_annotations(
        args.record,
        extension=args.annotation_extension,
        start_sample=segment.start_sample,
        end_sample=end_sample,
    )
    lead_name = args.lead_name or segment.channel_name
    result = extract_patient_template_morphology(
        segment.samples,
        peaks,
        segment.sampling_rate_hz,
        anchor_track=f"expert_{args.annotation_extension}",
        patient_id=args.patient_id,
        lead_name=lead_name,
        calibration_beats=args.calibration_beats,
        segment_start_s=segment.start_s,
    )
    symbol_by_sample = {
        int(sample): str(symbol) for sample, symbol in zip(peaks, symbols)
    }
    features = result.features.copy()
    features["reference_symbol"] = features[
        "anchor_sample_in_segment"
    ].map(symbol_by_sample)

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    features.to_csv(output / "patient_template_features.csv", index=False)
    np.save(output / "whole_beat_waveforms.npy", result.beat_waveforms)
    np.save(output / "normalized_whole_beat_waveforms.npy", result.normalized_waveforms)
    np.save(output / "patient_raw_template.npy", result.raw_template)
    np.save(output / "patient_normalized_template.npy", result.normalized_template)
    np.save(output / "patient_raw_template_bank.npy", result.raw_template_bank)
    np.save(
        output / "patient_normalized_template_bank.npy",
        result.normalized_template_bank,
    )
    np.save(output / "template_member_counts.npy", result.template_member_counts)
    scope = {
        "record": str(Path(args.record).expanduser().resolve()),
        "channel_index": segment.channel_index,
        "lead_name": lead_name,
        "sampling_rate_hz": segment.sampling_rate_hz,
        "start_s": segment.start_s,
        "duration_s": segment.duration_s,
        "annotation_extension": args.annotation_extension,
        "reference_anchor_count": int(peaks.size),
        "summary": result.summary(),
        "reference_symbols_are_context_not_training_targets": True,
    }
    (output / "run_scope.json").write_text(
        json.dumps(scope, indent=2), encoding="utf-8"
    )
    print(json.dumps(result.summary(), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
