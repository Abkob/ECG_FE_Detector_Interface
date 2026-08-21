"""Windows-friendly command line for the complete RR--HRV architecture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RRHRVConfig
from .edf import inspect_edf
from .neurokit_zhai import (
    run_neurokit_zhai_edf,
    run_neurokit_zhai_wfdb,
    save_neurokit_zhai_outputs,
)
from .config import NeuroKitZhaiConfig
from .outputs import save_rr_hrv_outputs
from .pipeline import run_rr_hrv_edf, run_rr_hrv_wfdb
from .plotting import (
    save_complete_rr_hrv_diagnostic,
    save_neurokit_zhai_diagnostic,
    save_neurokit_zhai_morphology_diagnostic,
)
from .validation import (
    beat_symbol_sensitivity,
    evaluate_branch_against_reference,
)
from .wfdb_io import load_wfdb_beat_annotations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ecg-features")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-edf", help="List EDF channels")
    inspect_parser.add_argument("--edf", required=True, type=Path)

    edf_parser = subparsers.add_parser(
        "run-rr-hrv",
        help="Run the complete UNSW + NeuroKit RR--HRV architecture on EDF",
    )
    edf_parser.add_argument("--edf", required=True, type=Path)
    edf_parser.add_argument("--channel")
    edf_parser.add_argument("--start-s", type=float, default=0.0)
    edf_parser.add_argument("--duration-s", type=float, default=600.0)
    edf_parser.add_argument("--output-dir", required=True, type=Path)
    _add_architecture_arguments(edf_parser)

    wfdb_parser = subparsers.add_parser(
        "run-rr-hrv-wfdb",
        help="Run the complete RR--HRV architecture on one WFDB record",
    )
    wfdb_parser.add_argument("--record", required=True, type=Path)
    wfdb_parser.add_argument("--channel", default="0")
    wfdb_parser.add_argument("--start-s", type=float, default=0.0)
    wfdb_parser.add_argument("--duration-s", type=float)
    wfdb_parser.add_argument("--output-dir", required=True, type=Path)
    _add_architecture_arguments(wfdb_parser)

    benchmark_parser = subparsers.add_parser(
        "benchmark-mitdb",
        help="Benchmark the architecture against MIT--BIH expert beat labels",
    )
    benchmark_parser.add_argument("--database-dir", required=True, type=Path)
    benchmark_parser.add_argument(
        "--records",
        nargs="+",
        default=["108", "113", "207", "222", "231"],
    )
    benchmark_parser.add_argument("--channel", default="0")
    benchmark_parser.add_argument("--output-dir", required=True, type=Path)
    benchmark_parser.add_argument("--save-record-plots", action="store_true")
    _add_architecture_arguments(benchmark_parser)

    nz_edf_parser = subparsers.add_parser(
        "run-neurokit-zhai-edf",
        help=(
            "Run independent NeuroKit and Zhai RR--HRV plus Varon morphology "
            "tracks on EDF"
        ),
    )
    nz_edf_parser.add_argument("--edf", required=True, type=Path)
    nz_edf_parser.add_argument("--channel")
    nz_edf_parser.add_argument("--start-s", type=float, default=0.0)
    nz_edf_parser.add_argument("--duration-s", type=float, default=600.0)
    nz_edf_parser.add_argument("--output-dir", required=True, type=Path)
    _add_neurokit_zhai_arguments(nz_edf_parser)

    nz_wfdb_parser = subparsers.add_parser(
        "run-neurokit-zhai-wfdb",
        help=(
            "Run independent NeuroKit and Zhai RR--HRV plus Varon morphology "
            "tracks on WFDB"
        ),
    )
    nz_wfdb_parser.add_argument("--record", required=True, type=Path)
    nz_wfdb_parser.add_argument("--channel", default="0")
    nz_wfdb_parser.add_argument("--start-s", type=float, default=0.0)
    nz_wfdb_parser.add_argument("--duration-s", type=float)
    nz_wfdb_parser.add_argument("--output-dir", required=True, type=Path)
    _add_neurokit_zhai_arguments(nz_wfdb_parser)
    return parser


def _add_architecture_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--orientation", choices=["original", "inverted"], default="original"
    )
    parser.add_argument(
        "--no-inverted-context",
        action="store_true",
        help="Skip the non-routing dual-polarity context run",
    )
    parser.add_argument(
        "--neurokit-min-delay-ms",
        type=float,
        default=250.0,
        help="Experimental secondary-detector delay; 300-ms baseline is still run",
    )
    parser.add_argument(
        "--strict-min-delay",
        action="store_true",
        help="Use > instead of >= for the experimental minimum-delay rule",
    )
    parser.add_argument(
        "--support-tolerance-ms",
        type=float,
        default=50.0,
        help="Current Ho same-QRS support tolerance; 150-ms ablation is still run",
    )
    parser.add_argument("--hrv-window-rr", type=int, default=100)


def _add_neurokit_zhai_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--orientation", choices=["original", "inverted"], default="original"
    )
    parser.add_argument(
        "--with-inverted-context",
        action="store_true",
        help="Run the opposite polarity as non-routing diagnostic context",
    )
    parser.add_argument(
        "--neurokit-min-delay-ms",
        type=float,
        default=300.0,
        help="NeuroKit minimum separation; the unmodified default is 300 ms",
    )
    parser.add_argument(
        "--inclusive-min-delay",
        action="store_true",
        help="Use >= instead of NeuroKit's unmodified strict > comparison",
    )
    parser.add_argument("--support-tolerance-ms", type=float, default=50.0)
    parser.add_argument("--audit-tolerance-ms", type=float, default=75.0)
    parser.add_argument("--hrv-window-rr", type=int, default=100)


def _config_from_args(args: argparse.Namespace) -> RRHRVConfig:
    return RRHRVConfig(
        processing_orientation=args.orientation,
        compute_inverted_context=not args.no_inverted_context,
        neurokit_experimental_min_delay_ms=args.neurokit_min_delay_ms,
        min_delay_inclusive=not args.strict_min_delay,
        support_tolerance_ms=args.support_tolerance_ms,
        hrv_window_rr_intervals=args.hrv_window_rr,
    )


def _neurokit_zhai_config_from_args(args: argparse.Namespace) -> NeuroKitZhaiConfig:
    return NeuroKitZhaiConfig(
        processing_orientation=args.orientation,
        compute_inverted_context=args.with_inverted_context,
        neurokit_minimum_delay_ms=args.neurokit_min_delay_ms,
        neurokit_minimum_delay_inclusive=args.inclusive_min_delay,
        support_tolerance_ms=args.support_tolerance_ms,
        audit_match_tolerance_ms=args.audit_tolerance_ms,
        hrv_window_rr_intervals=args.hrv_window_rr,
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "inspect-edf":
        print(json.dumps(inspect_edf(args.edf).to_dict(), indent=2))
        return
    if args.command == "benchmark-mitdb":
        _benchmark_mitdb(args)
        return
    if args.command in {"run-neurokit-zhai-edf", "run-neurokit-zhai-wfdb"}:
        _run_neurokit_zhai_command(args)
        return

    config = _config_from_args(args)
    if args.command == "run-rr-hrv":
        segment, result = run_rr_hrv_edf(
            args.edf,
            channel_label=args.channel,
            start_s=args.start_s,
            duration_s=args.duration_s,
            config=config,
        )
        signal = segment.samples
        title = f"{Path(segment.path).name} | {segment.channel.label}"
    else:
        segment, result = run_rr_hrv_wfdb(
            args.record,
            channel=_parse_wfdb_channel(args.channel),
            start_s=args.start_s,
            duration_s=args.duration_s,
            config=config,
        )
        signal = segment.samples
        title = f"{Path(segment.record_path).name} | {segment.channel_name}"
    paths = save_rr_hrv_outputs(result, args.output_dir)
    plot_path = Path(args.output_dir).resolve() / "rr_hrv_complete_diagnostic.png"
    save_complete_rr_hrv_diagnostic(
        plot_path,
        ecg=signal,
        result=result,
        title=title,
    )
    summary = result.summary()
    summary["output_files"] = {**paths, "diagnostic_plot": str(plot_path)}
    print(json.dumps(summary, indent=2, default=_json_default))


def _run_neurokit_zhai_command(args: argparse.Namespace) -> None:
    config = _neurokit_zhai_config_from_args(args)
    if args.command == "run-neurokit-zhai-edf":
        segment, result = run_neurokit_zhai_edf(
            args.edf,
            channel_label=args.channel,
            start_s=args.start_s,
            duration_s=args.duration_s,
            config=config,
        )
        signal = segment.samples
        title = f"{Path(segment.path).name} | {segment.channel.label}"
    else:
        segment, result = run_neurokit_zhai_wfdb(
            args.record,
            channel=_parse_wfdb_channel(args.channel),
            start_s=args.start_s,
            duration_s=args.duration_s,
            config=config,
        )
        signal = segment.samples
        title = f"{Path(segment.record_path).name} | {segment.channel_name}"
    paths = save_neurokit_zhai_outputs(result, args.output_dir)
    plot_path = Path(args.output_dir).resolve() / "neurokit_zhai_diagnostic.png"
    save_neurokit_zhai_diagnostic(
        plot_path,
        ecg=signal,
        result=result,
        title=title,
    )
    morphology_plot_path = (
        Path(args.output_dir).resolve()
        / "neurokit_zhai_rr_hrv_morphology_diagnostic.png"
    )
    save_neurokit_zhai_morphology_diagnostic(
        morphology_plot_path,
        ecg=signal,
        result=result,
        title=title,
    )
    summary = result.summary()
    summary["output_files"] = {
        **paths,
        "diagnostic_plot": str(plot_path),
        "morphology_diagnostic_plot": str(morphology_plot_path),
    }
    print(json.dumps(summary, indent=2, default=_json_default))


def _benchmark_mitdb(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _config_from_args(args)
    channel = _parse_wfdb_channel(args.channel)
    all_detector_metrics: list[pd.DataFrame] = []
    all_symbol_metrics: list[pd.DataFrame] = []
    rr_metrics: list[dict[str, object]] = []

    for record_name in args.records:
        record_path = args.database_dir.resolve() / str(record_name)
        segment, result = run_rr_hrv_wfdb(
            record_path,
            channel=channel,
            config=config,
        )
        reference, symbols = load_wfdb_beat_annotations(
            record_path,
            start_sample=segment.start_sample,
            end_sample=segment.start_sample + segment.samples.size,
        )
        detector_metrics, rr_result = evaluate_branch_against_reference(
            result,
            reference,
            sampling_rate_hz=segment.sampling_rate_hz,
            tolerance_ms=config.audit_match_tolerance_ms,
        )
        detector_metrics.insert(0, "record", str(record_name))
        all_detector_metrics.append(detector_metrics)
        symbol_metrics = beat_symbol_sensitivity(
            result.selected.reliability.primary_refined_samples,
            reference,
            symbols,
            sampling_rate_hz=segment.sampling_rate_hz,
            tolerance_ms=config.audit_match_tolerance_ms,
        )
        symbol_metrics.insert(0, "record", str(record_name))
        all_symbol_metrics.append(symbol_metrics)
        rr_metrics.append(
            {
                "record": str(record_name),
                "orientation": result.selected_orientation,
                **rr_result,
                "rr_coverage_nk300": result.selected.baseline_reliability.rr_coverage,
                "rr_coverage_support150_ablation": (
                    result.selected.legacy_tolerance_reliability.rr_coverage
                ),
            }
        )
        if args.save_record_plots:
            save_complete_rr_hrv_diagnostic(
                output_dir / f"record_{record_name}_diagnostic.png",
                ecg=segment.samples,
                result=result,
                title=f"MIT--BIH {record_name} | {segment.channel_name}",
            )

    detector_frame = pd.concat(all_detector_metrics, ignore_index=True)
    symbol_frame = pd.concat(all_symbol_metrics, ignore_index=True)
    rr_frame = pd.DataFrame(rr_metrics)
    detector_frame.to_csv(output_dir / "detector_metrics.csv", index=False)
    symbol_frame.to_csv(output_dir / "unsw_symbol_sensitivity.csv", index=False)
    rr_frame.to_csv(output_dir / "rr_reliability_metrics.csv", index=False)
    (output_dir / "benchmark_config.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "records": [str(record) for record in args.records],
                "output_dir": str(output_dir),
                "detector_metrics": detector_frame.to_dict(orient="records"),
                "rr_metrics": rr_frame.to_dict(orient="records"),
            },
            indent=2,
            default=_json_default,
        )
    )


def _parse_wfdb_channel(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


if __name__ == "__main__":
    main()
