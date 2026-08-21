"""Run the standalone conduction timing prototype on one WFDB lead."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecg_cascade.conduction_timing import extract_conduction_timing  # noqa: E402
from ecg_cascade.prominence_morphology import extract_prominence_morphology  # noqa: E402
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
    parser.add_argument("--patient-id", required=True)
    parser.add_argument("--lead-name", default=None)
    parser.add_argument("--variability-window-beats", type=int, default=30)
    parser.add_argument("--minimum-valid-fraction", type=float, default=0.8)
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
    prominence = extract_prominence_morphology(
        segment.samples,
        peaks,
        segment.sampling_rate_hz,
        anchor_track=f"expert_{args.annotation_extension}",
        patient_id=args.patient_id,
        lead_name=lead_name,
        segment_start_s=segment.start_s,
    )
    conduction = extract_conduction_timing(
        prominence.features,
        segment.sampling_rate_hz,
        variability_window_beats=args.variability_window_beats,
        minimum_valid_fraction=args.minimum_valid_fraction,
        landmark_method=str(prominence.parameters["method"]),
    )

    symbol_by_sample = {
        int(sample): str(symbol) for sample, symbol in zip(peaks, symbols)
    }
    beat_features = conduction.beat_features.copy()
    beat_features["reference_symbol"] = beat_features[
        "r_peak_sample_in_segment"
    ].map(symbol_by_sample)

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    prominence.features.to_csv(output / "pqrst_landmarks_and_morphology.csv", index=False)
    beat_features.to_csv(output / "conduction_beat_features.csv", index=False)
    conduction.rolling_features.to_csv(
        output / "conduction_rolling_features.csv", index=False
    )
    scope = {
        "record": str(Path(args.record).expanduser().resolve()),
        "channel_index": segment.channel_index,
        "lead_name": lead_name,
        "sampling_rate_hz": segment.sampling_rate_hz,
        "start_s": segment.start_s,
        "duration_s": segment.duration_s,
        "annotation_extension": args.annotation_extension,
        "reference_r_anchor_count": int(peaks.size),
        "reference_symbols_are_context_not_training_targets": True,
        "pqrst_summary": prominence.summary(),
        "conduction_summary": conduction.summary(),
    }
    (output / "run_scope.json").write_text(
        json.dumps(scope, indent=2), encoding="utf-8"
    )
    print(json.dumps(conduction.summary(), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
