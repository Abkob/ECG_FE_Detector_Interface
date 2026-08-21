"""All-48 audit of same-source-per-RR and pure-window fusion challengers."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb

from ecg_cascade.hrv_fidelity import (
    lin_concordance_correlation,
    symmetric_absolute_percentage_error,
    threshold_agreement_counts,
)
from ecg_cascade.interval_consistent_fusion import (
    build_interval_consistent_from_associations,
)
from ecg_cascade.rr_hrv import calculate_jeppesen_feature_arrays
from ecg_cascade.validation import match_peaks_to_reference, peak_metrics
from ecg_cascade.wfdb_io import load_wfdb_beat_annotations


FEATURES = ("j1_csi_x_slope", "j2_modcsi_filtered_x_slope")
FEATURE_LABELS = {
    "j1_csi_x_slope": "J1 = CSI x slope",
    "j2_modcsi_filtered_x_slope": "J2 = ModCSI x slope",
}
QUANTILES = (0.90, 0.95, 0.99)
SELECTED_CONFIGURATION = "nk300"
SELECTED_TOLERANCE_MS = 50.0
DIFFICULT_RECORDS = ("108", "113", "207", "222", "231")
INTERVAL_METHODS = (
    "original_per_beat_fusion",
    "interval_neurokit_unsw",
    "interval_zhai_unsw",
    "interval_neurokit_zhai_unsw",
    "pure_neurokit_rr_lane",
    "pure_zhai_rr_lane",
)
HRV_METHODS = (
    "original_per_beat_fusion",
    "interval_neurokit_unsw",
    "interval_zhai_unsw",
    "interval_neurokit_zhai_unsw",
    "pure_neurokit_window_lane",
    "pure_zhai_window_lane",
    "window_priority_neurokit_zhai_unsw",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-dir", required=True, type=Path)
    parser.add_argument("--fusion-events", required=True, type=Path)
    parser.add_argument("--previous-pooled-metrics", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--records", nargs="*")
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--median-width", type=int, default=7)
    parser.add_argument("--expert-match-tolerance-ms", type=float, default=75.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    database = args.database_dir.expanduser().resolve()
    events_path = args.fusion_events.expanduser().resolve()
    previous_path = args.previous_pooled_metrics.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(events_path, low_memory=False)
    events["record"] = events["record"].astype(str)
    requested_records = [str(value) for value in args.records] if args.records else None
    if requested_records is not None:
        events = events[events["record"].isin(requested_records)].copy()
    records = events["record"].drop_duplicates().tolist()

    rr_record_rows: list[dict[str, object]] = []
    hrv_record_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    paired_selected: list[pd.DataFrame] = []
    rr_store: dict[tuple[str, float, str], dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list)
    )
    hrv_store: dict[
        tuple[str, float | None, str, str], dict[str, list[np.ndarray]]
    ] = defaultdict(lambda: defaultdict(list))
    record_cache: dict[str, dict[str, object]] = {}

    for record in records:
        record_events = events[events["record"] == record]
        first = record_events[
            (record_events["neurokit_configuration"] == SELECTED_CONFIGURATION)
            & np.isclose(
                record_events["association_tolerance_ms"].to_numpy(float),
                SELECTED_TOLERANCE_MS,
            )
        ].sort_values("event_index")
        if first.empty:
            raise ValueError(f"Missing selected configuration for record {record}")
        cache = _build_record_cache(
            database / record,
            record,
            first,
            window_size=args.window_size,
            median_width=args.median_width,
            expert_match_tolerance_ms=args.expert_match_tolerance_ms,
        )
        record_cache[record] = cache
        baseline_key = ("baseline", np.nan, "unsw_baseline")
        baseline_rr_row, baseline_rr_errors = _evaluate_rr_method(
            record=record,
            configuration="baseline",
            tolerance_ms=np.nan,
            method="unsw_baseline",
            rr_ms=np.asarray(cache["unsw_rr_ms"]),
            rr_sources=np.full(np.asarray(cache["unsw_rr_ms"]).size, "unsw", dtype=object),
            truth_rr_ms=np.asarray(cache["truth_rr_ms"]),
            event_metrics=dict(cache["event_metrics"]),
        )
        rr_record_rows.append(baseline_rr_row)
        _store_rr(rr_store[baseline_key], baseline_rr_errors)
        baseline_paired = _pair_feature_method(
            cache,
            pd.DataFrame(cache["unsw_features"]),
            method="unsw_baseline",
            configuration="baseline",
            tolerance_ms=np.nan,
            source_at_anchor=np.full(np.asarray(cache["unsw"]).size, "unsw", dtype=object),
        )
        _record_hrv(
            baseline_paired,
            hrv_record_rows,
            threshold_rows,
            hrv_store,
        )

        for (configuration, tolerance), group in record_events.groupby(
            ["neurokit_configuration", "association_tolerance_ms"], sort=True
        ):
            configuration = str(configuration)
            tolerance = float(tolerance)
            group = group.sort_values("event_index").reset_index(drop=True)
            if not np.array_equal(
                group["unsw_sample"].to_numpy(np.int64), np.asarray(cache["unsw"])
            ):
                raise ValueError(f"UNSW sequence changed within record {record}")
            method_payload = _build_interval_methods(
                group,
                sampling_rate_hz=float(cache["fs"]),
                tolerance_ms=tolerance,
            )
            for method, payload in method_payload.items():
                rr_row, errors = _evaluate_rr_method(
                    record=record,
                    configuration=configuration,
                    tolerance_ms=tolerance,
                    method=method,
                    rr_ms=payload["rr_ms"],
                    rr_sources=payload["rr_sources"],
                    truth_rr_ms=np.asarray(cache["truth_rr_ms"]),
                    event_metrics=dict(cache["event_metrics"]),
                    shared_boundary_discontinuity_ms=payload.get(
                        "shared_boundary_discontinuity_ms"
                    ),
                )
                rr_record_rows.append(rr_row)
                _store_rr(rr_store[(configuration, tolerance, method)], errors)

            # Full RR/error sensitivity is evaluated for every detector and
            # association setting.  The much more expensive 100-RR feature
            # audit is deliberately restricted to the configuration declared
            # before inspecting these results.  This both avoids redundant
            # computation and prevents choosing an HRV setting post hoc.
            if (
                configuration == SELECTED_CONFIGURATION
                and np.isclose(tolerance, SELECTED_TOLERANCE_MS)
            ):
                hrv_payload = _build_hrv_methods(
                    group,
                    method_payload,
                    cache,
                    window_size=args.window_size,
                    median_width=args.median_width,
                )
                for method, payload in hrv_payload.items():
                    paired = _pair_feature_method(
                        cache,
                        payload["features"],
                        method=method,
                        configuration=configuration,
                        tolerance_ms=tolerance,
                        source_at_anchor=payload["source_at_anchor"],
                    )
                    _record_hrv(
                        paired,
                        hrv_record_rows,
                        threshold_rows,
                        hrv_store,
                    )
                    paired_selected.append(paired)
        print(f"completed {record}", flush=True)

    rr_records = pd.DataFrame(rr_record_rows)
    rr_pooled = _pool_rr(rr_records, rr_store)
    hrv_records = pd.DataFrame(hrv_record_rows)
    hrv_pooled = _pool_hrv(hrv_records, hrv_store)
    threshold_records = pd.DataFrame(threshold_rows)
    threshold_pooled = _pool_thresholds(threshold_records)
    selected_pairs = pd.concat(paired_selected, ignore_index=True)
    previous = pd.read_csv(previous_path)
    comparison = _build_comparison(
        previous,
        rr_pooled,
        hrv_pooled,
        threshold_pooled,
    )
    decision = _build_decision(rr_records, rr_pooled, hrv_pooled, threshold_pooled)

    rr_records.to_csv(output / "record_rr_metrics.csv", index=False)
    rr_pooled.to_csv(output / "pooled_rr_metrics.csv", index=False)
    hrv_records.to_csv(output / "record_hrv_metrics.csv", index=False)
    hrv_pooled.to_csv(output / "pooled_hrv_metrics.csv", index=False)
    threshold_records.to_csv(output / "record_threshold_metrics.csv", index=False)
    threshold_pooled.to_csv(output / "pooled_threshold_metrics.csv", index=False)
    selected_pairs.to_csv(
        output / "selected_configuration_paired_hrv.csv.gz",
        index=False,
        compression="gzip",
    )
    comparison.to_csv(output / "complete_method_comparison.csv", index=False)
    (output / "decision_summary.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    (output / "benchmark_metadata.json").write_text(
        json.dumps(
            {
                "database_dir": str(database),
                "fusion_event_source": str(events_path),
                "previous_metrics_source": str(previous_path),
                "records": records,
                "expert_match_tolerance_ms": args.expert_match_tolerance_ms,
                "window_size_rr": args.window_size,
                "median_width_rr": args.median_width,
                "hrv_primary_configuration": SELECTED_CONFIGURATION,
                "hrv_primary_association_tolerance_ms": SELECTED_TOLERANCE_MS,
                "hrv_grid_policy": (
                    "full J1/J2 and threshold audit only on the predeclared "
                    "primary setting; all detector/tolerance settings receive "
                    "RR and successive-change sensitivity analysis"
                ),
                "rr_sensitivity_configurations": ["nk250", "nk300"],
                "rr_sensitivity_tolerances_ms": [50.0, 75.0, 100.0, 150.0],
                "pure_window_required_candidate_events": args.window_size
                + args.median_width,
                "pure_window_reason": (
                    "101 events define 100 RR intervals and six earlier RR "
                    "values preserve the causal seven-RR median-filter history"
                ),
                "interval_mixed_time_axis": "cumulative selected RR tachogram",
                "published_status": (
                    "same-source-per-RR and window-priority rules are project "
                    "ablations, not published fusion algorithms"
                ),
                "threshold_status": (
                    "per-record expert quantiles are measurement stress tests, "
                    "not seizure thresholds"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _save_plots(rr_records, rr_pooled, hrv_records, hrv_pooled, threshold_pooled, output)
    print("\nSELECTED CONFIGURATION COMPARISON")
    print(comparison.to_string(index=False))
    print("\nDECISION")
    print(json.dumps(decision, indent=2))


def _build_record_cache(
    record_path: Path,
    record: str,
    events: pd.DataFrame,
    *,
    window_size: int,
    median_width: int,
    expert_match_tolerance_ms: float,
) -> dict[str, object]:
    fs = float(wfdb.rdheader(str(record_path)).fs)
    unsw = events["unsw_sample"].to_numpy(np.int64)
    reference, symbols = load_wfdb_beat_annotations(record_path)
    matches = match_peaks_to_reference(
        unsw,
        reference,
        sampling_rate_hz=fs,
        tolerance_ms=expert_match_tolerance_ms,
    )
    truth_rr = _expert_rr_truth(unsw, reference, matches, fs)
    unsw_rr = np.diff(unsw) * 1000.0 / fs
    expert_rr = np.diff(reference) * 1000.0 / fs
    unsw_features = calculate_jeppesen_feature_arrays(
        unsw_rr,
        end_time_s=unsw[1:] / fs,
        window_size=window_size,
        median_width=median_width,
    )
    expert_features = calculate_jeppesen_feature_arrays(
        expert_rr,
        end_time_s=reference[1:] / fs,
        window_size=window_size,
        median_width=median_width,
    )
    return {
        "record": record,
        "fs": fs,
        "unsw": unsw,
        "reference": reference,
        "symbols": symbols,
        "matches": matches,
        "truth_rr_ms": truth_rr,
        "unsw_rr_ms": unsw_rr,
        "unsw_features": unsw_features,
        "expert_features": expert_features,
        "event_metrics": peak_metrics(
            unsw,
            reference,
            sampling_rate_hz=fs,
            tolerance_ms=expert_match_tolerance_ms,
        ),
    }


def _build_interval_methods(
    events: pd.DataFrame,
    *,
    sampling_rate_hz: float,
    tolerance_ms: float,
) -> dict[str, dict[str, np.ndarray]]:
    unsw = events["unsw_sample"].to_numpy(np.int64)
    selected = events["selected_sample"].to_numpy(np.int64)
    selected_event_source = events["selected_source"].astype(str).to_numpy(object)
    original_rr_sources = np.array(
        [
            selected_event_source[index]
            if selected_event_source[index] == selected_event_source[index + 1]
            else f"mixed:{selected_event_source[index]}->{selected_event_source[index + 1]}"
            for index in range(selected_event_source.size - 1)
        ],
        dtype=object,
    )
    payload: dict[str, dict[str, np.ndarray]] = {
        "original_per_beat_fusion": {
            "rr_ms": np.diff(selected) * 1000.0 / sampling_rate_hz,
            "rr_sources": original_rr_sources,
            "selected_samples": selected,
        }
    }
    variants = {
        "interval_neurokit_unsw": (True, False),
        "interval_zhai_unsw": (False, True),
        "interval_neurokit_zhai_unsw": (True, True),
    }
    for method, (enable_nk, enable_zhai) in variants.items():
        result = build_interval_consistent_from_associations(
            events,
            sampling_rate_hz=sampling_rate_hz,
            association_tolerance_ms=tolerance_ms,
            enable_neurokit=enable_nk,
            enable_zhai=enable_zhai,
        )
        payload[method] = {
            "rr_ms": result.rr_intervals["rr_ms"].to_numpy(float),
            "rr_sources": result.rr_intervals["rr_source"].astype(str).to_numpy(object),
            "shared_boundary_discontinuity_ms": result.rr_intervals[
                "shared_boundary_discontinuity_ms"
            ].to_numpy(float),
        }
    for method, column, label in (
        ("pure_neurokit_rr_lane", "neurokit_unique_sample", "neurokit"),
        ("pure_zhai_rr_lane", "zhai_unique_sample", "zhai"),
    ):
        candidates = pd.to_numeric(events[column], errors="coerce").to_numpy(float)
        available = np.isfinite(candidates[:-1]) & np.isfinite(candidates[1:])
        rr = np.full(unsw.size - 1, np.nan)
        rr[available] = np.diff(candidates)[available] * 1000.0 / sampling_rate_hz
        source = np.full(rr.size, "unavailable", dtype=object)
        source[available] = label
        payload[method] = {"rr_ms": rr, "rr_sources": source}
    return payload


def _build_hrv_methods(
    events: pd.DataFrame,
    interval_payload: dict[str, dict[str, np.ndarray]],
    cache: dict[str, object],
    *,
    window_size: int,
    median_width: int,
) -> dict[str, dict[str, object]]:
    fs = float(cache["fs"])
    event_count = np.asarray(cache["unsw"]).size
    payload: dict[str, dict[str, object]] = {}
    for method in (
        "original_per_beat_fusion",
        "interval_neurokit_unsw",
        "interval_zhai_unsw",
        "interval_neurokit_zhai_unsw",
    ):
        rr = interval_payload[method]["rr_ms"]
        if method == "original_per_beat_fusion":
            selected = interval_payload[method]["selected_samples"]
            feature_frame = calculate_jeppesen_feature_arrays(
                rr,
                end_time_s=selected[1:] / fs,
                window_size=window_size,
                median_width=median_width,
            )
        else:
            feature_frame = calculate_jeppesen_feature_arrays(
                rr,
                end_time_s=None,
                window_size=window_size,
                median_width=median_width,
            )
        source = np.full(event_count, "unavailable", dtype=object)
        source[1:] = interval_payload[method]["rr_sources"]
        payload[method] = {"features": feature_frame, "source_at_anchor": source}

    nk_lane, nk_source = _candidate_feature_lane(
        events,
        "neurokit_unique_sample",
        "neurokit",
        sampling_rate_hz=fs,
        window_size=window_size,
        median_width=median_width,
    )
    zhai_lane, zhai_source = _candidate_feature_lane(
        events,
        "zhai_unique_sample",
        "zhai",
        sampling_rate_hz=fs,
        window_size=window_size,
        median_width=median_width,
    )
    payload["pure_neurokit_window_lane"] = {
        "features": nk_lane,
        "source_at_anchor": nk_source,
    }
    payload["pure_zhai_window_lane"] = {
        "features": zhai_lane,
        "source_at_anchor": zhai_source,
    }
    baseline = pd.DataFrame(cache["unsw_features"]).copy()
    priority = baseline.copy()
    priority_source = np.full(event_count, "unsw", dtype=object)
    nk_mask = nk_lane["feature_defined"].to_numpy(bool)
    zhai_mask = (~nk_mask) & zhai_lane["feature_defined"].to_numpy(bool)
    for feature in FEATURES:
        values = priority[feature].to_numpy(float)
        values[nk_mask] = nk_lane.loc[nk_mask, feature].to_numpy(float)
        values[zhai_mask] = zhai_lane.loc[zhai_mask, feature].to_numpy(float)
        priority[feature] = values
    defined = priority["feature_defined"].to_numpy(bool)
    defined[nk_mask | zhai_mask] = True
    priority["feature_defined"] = defined
    priority_source[1:][nk_mask] = "neurokit"
    priority_source[1:][zhai_mask] = "zhai"
    payload["window_priority_neurokit_zhai_unsw"] = {
        "features": priority,
        "source_at_anchor": priority_source,
    }
    return payload


def _candidate_feature_lane(
    events: pd.DataFrame,
    column: str,
    source_label: str,
    *,
    sampling_rate_hz: float,
    window_size: int,
    median_width: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    candidates = pd.to_numeric(events[column], errors="coerce").to_numpy(float)
    rr_count = max(0, candidates.size - 1)
    lane_values = {
        feature: np.full(rr_count, np.nan, dtype=float) for feature in FEATURES
    }
    lane_defined = np.zeros(rr_count, dtype=bool)
    source = np.full(candidates.size, "unavailable", dtype=object)
    finite = np.isfinite(candidates)
    starts = np.flatnonzero(finite & np.concatenate(([True], ~finite[:-1])))
    ends = np.flatnonzero(finite & np.concatenate((~finite[1:], [True]))) + 1
    required_events = window_size + median_width
    for start, end in zip(starts, ends, strict=True):
        if end - start < required_events:
            continue
        timestamps = candidates[start:end]
        if np.any(np.diff(timestamps) <= 0):
            continue
        features = calculate_jeppesen_feature_arrays(
            np.diff(timestamps) * 1000.0 / sampling_rate_hz,
            end_time_s=timestamps[1:] / sampling_rate_hz,
            window_size=window_size,
            median_width=median_width,
        )
        first_local_row = window_size + median_width - 2
        local_rows = np.arange(first_local_row, features.shape[0], dtype=np.int64)
        if local_rows.size == 0:
            continue
        local_defined = features.iloc[local_rows]["feature_defined"].to_numpy(bool)
        local_rows = local_rows[local_defined]
        if local_rows.size == 0:
            continue
        global_rows = start + local_rows
        for feature in FEATURES:
            lane_values[feature][global_rows] = features.iloc[local_rows][
                feature
            ].to_numpy(float)
        lane_defined[global_rows] = True
        source[global_rows + 1] = source_label
    lane = pd.DataFrame({**lane_values, "feature_defined": lane_defined})
    return lane, source


def _expert_rr_truth(
    detected: np.ndarray,
    reference: np.ndarray,
    matches: object,
    fs: float,
) -> np.ndarray:
    mapping = dict(
        zip(
            matches.detected_indices.tolist(),
            matches.reference_indices.tolist(),
            strict=True,
        )
    )
    truth = np.full(detected.size - 1, np.nan)
    for index in range(detected.size - 1):
        if index not in mapping or index + 1 not in mapping:
            continue
        left = mapping[index]
        right = mapping[index + 1]
        if right == left + 1:
            truth[index] = (reference[right] - reference[left]) * 1000.0 / fs
    return truth


def _evaluate_rr_method(
    *,
    record: str,
    configuration: str,
    tolerance_ms: float,
    method: str,
    rr_ms: np.ndarray,
    rr_sources: np.ndarray,
    truth_rr_ms: np.ndarray,
    event_metrics: dict[str, object],
    shared_boundary_discontinuity_ms: np.ndarray | None = None,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    rr = np.asarray(rr_ms, dtype=float)
    truth = np.asarray(truth_rr_ms, dtype=float)
    sources = np.asarray(rr_sources, dtype=object)
    valid = np.isfinite(rr) & np.isfinite(truth)
    error = rr[valid] - truth[valid]
    change_valid = (
        np.isfinite(rr[:-1])
        & np.isfinite(rr[1:])
        & np.isfinite(truth[:-1])
        & np.isfinite(truth[1:])
    )
    change_error_all = np.diff(rr)[change_valid] - np.diff(truth)[change_valid]
    transition = sources[1:] != sources[:-1]
    available_source = (sources[1:] != "unavailable") & (sources[:-1] != "unavailable")
    change_transition = change_valid & transition & available_source
    change_same = change_valid & ~transition & available_source
    change_error_transition = np.diff(rr)[change_transition] - np.diff(truth)[
        change_transition
    ]
    change_error_same = np.diff(rr)[change_same] - np.diff(truth)[change_same]
    discontinuity = (
        np.asarray(shared_boundary_discontinuity_ms, dtype=float)
        if shared_boundary_discontinuity_ms is not None
        else np.full(rr.size, np.nan)
    )
    row = {
        "record": record,
        "neurokit_configuration": configuration,
        "association_tolerance_ms": tolerance_ms,
        "method": method,
        "event_existence_f1": float(event_metrics["f1"]),
        "rr_total_count": int(rr.size),
        "rr_available_count": int(np.isfinite(rr).sum()),
        "rr_availability": float(np.isfinite(rr).mean()) if rr.size else np.nan,
        "rr_evaluable_count": int(error.size),
        "rr_mae_ms": _mae(error),
        "rr_p95_absolute_error_ms": _p95(error),
        "successive_change_evaluable_count": int(change_error_all.size),
        "successive_rr_change_mae_ms": _mae(change_error_all),
        "same_source_change_count": int(change_error_same.size),
        "same_source_successive_change_mae_ms": _mae(change_error_same),
        "source_transition_change_count": int(change_error_transition.size),
        "source_transition_successive_change_mae_ms": _mae(change_error_transition),
        "rr_source_transition_count": int(np.sum(transition & available_source)),
        "rr_source_transition_fraction": float(
            np.mean(transition[available_source])
        )
        if np.any(available_source)
        else np.nan,
        "mean_absolute_shared_boundary_discontinuity_ms": _mae(
            discontinuity[np.isfinite(discontinuity)]
        ),
        "p95_absolute_shared_boundary_discontinuity_ms": _p95(
            discontinuity[np.isfinite(discontinuity)]
        ),
        "neurokit_rr_count": int(np.sum(sources == "neurokit")),
        "zhai_rr_count": int(np.sum(sources == "zhai")),
        "unsw_rr_count": int(np.sum(sources == "unsw")),
        "mixed_endpoint_rr_count": int(
            np.sum(np.array([str(value).startswith("mixed:") for value in sources]))
        ),
    }
    return row, {
        "rr": error,
        "change_all": change_error_all,
        "change_same": change_error_same,
        "change_transition": change_error_transition,
    }


def _pair_feature_method(
    cache: dict[str, object],
    method_features: pd.DataFrame,
    *,
    method: str,
    configuration: str,
    tolerance_ms: float,
    source_at_anchor: np.ndarray,
) -> pd.DataFrame:
    expert = pd.DataFrame(cache["expert_features"])
    matches = cache["matches"]
    detected_index = np.asarray(matches.detected_indices, dtype=np.int64)
    reference_index = np.asarray(matches.reference_indices, dtype=np.int64)
    keep = (
        (detected_index >= 100)
        & (reference_index >= 100)
        & (detected_index - 1 < method_features.shape[0])
        & (reference_index - 1 < expert.shape[0])
    )
    detected_index = detected_index[keep]
    reference_index = reference_index[keep]
    if detected_index.size == 0:
        return pd.DataFrame()
    method_rows = detected_index - 1
    expert_rows = reference_index - 1
    reference_samples = np.asarray(cache["reference"], dtype=np.int64)
    source = np.asarray(source_at_anchor, dtype=object)
    paired = pd.DataFrame(
        {
            "record": np.full(detected_index.size, str(cache["record"]), dtype=object),
            "neurokit_configuration": np.full(
                detected_index.size, configuration, dtype=object
            ),
            "association_tolerance_ms": np.full(
                detected_index.size, tolerance_ms, dtype=float
            ),
            "method": np.full(detected_index.size, method, dtype=object),
            "detected_anchor_index": detected_index,
            "expert_anchor_index": reference_index,
            "anchor_time_s": reference_samples[reference_index] / float(cache["fs"]),
            "window_source": source[detected_index].astype(str),
        }
    )
    for feature in FEATURES:
        paired[f"expert__{feature}"] = expert.iloc[expert_rows][feature].to_numpy(float)
        paired[f"algorithm__{feature}"] = method_features.iloc[method_rows][
            feature
        ].to_numpy(float)
    return paired


def _record_hrv(
    paired: pd.DataFrame,
    record_rows: list[dict[str, object]],
    threshold_rows: list[dict[str, object]],
    store: dict[
        tuple[str, float | None, str, str], dict[str, list[np.ndarray]]
    ],
) -> None:
    if paired.empty:
        return
    configuration = str(paired.iloc[0]["neurokit_configuration"])
    tolerance = float(paired.iloc[0]["association_tolerance_ms"])
    method = str(paired.iloc[0]["method"])
    record = str(paired.iloc[0]["record"])
    for feature in FEATURES:
        truth_all = paired[f"expert__{feature}"].to_numpy(float)
        measured_all = paired[f"algorithm__{feature}"].to_numpy(float)
        finite = np.isfinite(truth_all) & np.isfinite(measured_all)
        truth = truth_all[finite]
        measured = measured_all[finite]
        error = measured - truth
        # NaN does not equal itself and is therefore unsafe as a dictionary
        # key across records.  Use None internally for the baseline setting.
        tolerance_key = None if np.isnan(tolerance) else tolerance
        key = (configuration, tolerance_key, method, feature)
        store[key]["truth"].append(truth)
        store[key]["measured"].append(measured)
        record_rows.append(
            {
                "record": record,
                "neurokit_configuration": configuration,
                "association_tolerance_ms": tolerance,
                "method": method,
                "feature": feature,
                "feature_label": FEATURE_LABELS[feature],
                "available_windows": int(truth.size),
                "total_matched_windows": int(truth_all.size),
                "window_coverage": _divide(int(truth.size), int(truth_all.size)),
                "ccc": lin_concordance_correlation(truth, measured),
                "mae": _mae(error),
                "smape_fraction": symmetric_absolute_percentage_error(truth, measured),
            }
        )
        finite_truth_all = truth_all[np.isfinite(truth_all)]
        if finite_truth_all.size < 200:
            continue
        for quantile in QUANTILES:
            threshold = float(np.quantile(finite_truth_all, quantile))
            counts = threshold_agreement_counts(truth, measured, threshold=threshold)
            total_positive = int(np.sum(finite_truth_all > threshold))
            selected_positive = int(counts["tp"]) + int(counts["fn"])
            threshold_rows.append(
                {
                    "record": record,
                    "neurokit_configuration": configuration,
                    "association_tolerance_ms": tolerance,
                    "method": method,
                    "feature": feature,
                    "expert_threshold_quantile": quantile,
                    "expert_derived_threshold": threshold,
                    "all_evaluable_n": int(finite_truth_all.size),
                    "all_expert_positive": total_positive,
                    "window_coverage": _divide(int(counts["n"]), int(finite_truth_all.size)),
                    "expert_positive_coverage": _divide(selected_positive, total_positive),
                    **counts,
                }
            )


def _store_rr(store: dict[str, list[np.ndarray]], errors: dict[str, np.ndarray]) -> None:
    for name, values in errors.items():
        store[name].append(np.asarray(values, dtype=float))


def _pool_rr(
    records: pd.DataFrame,
    store: dict[tuple[str, float, str], dict[str, list[np.ndarray]]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, errors in store.items():
        configuration, tolerance, method = key
        group = records[
            (records["neurokit_configuration"] == configuration)
            & (records["method"] == method)
            & (
                records["association_tolerance_ms"].isna()
                if np.isnan(tolerance)
                else np.isclose(records["association_tolerance_ms"], tolerance)
            )
        ]
        rr = _concat(errors["rr"])
        change_all = _concat(errors["change_all"])
        change_same = _concat(errors["change_same"])
        change_transition = _concat(errors["change_transition"])
        rows.append(
            {
                "neurokit_configuration": configuration,
                "association_tolerance_ms": tolerance,
                "method": method,
                "records": int(group["record"].nunique()),
                "event_existence_f1": _pooled_event_f1(group),
                "rr_total_count": int(group["rr_total_count"].sum()),
                "rr_available_count": int(group["rr_available_count"].sum()),
                "rr_availability": _divide(
                    int(group["rr_available_count"].sum()),
                    int(group["rr_total_count"].sum()),
                ),
                "rr_evaluable_count": int(rr.size),
                "rr_mae_ms": _mae(rr),
                "rr_p95_absolute_error_ms": _p95(rr),
                "successive_change_evaluable_count": int(change_all.size),
                "successive_rr_change_mae_ms": _mae(change_all),
                "same_source_change_count": int(change_same.size),
                "same_source_successive_change_mae_ms": _mae(change_same),
                "source_transition_change_count": int(change_transition.size),
                "source_transition_successive_change_mae_ms": _mae(change_transition),
                "rr_source_transition_count": int(group["rr_source_transition_count"].sum()),
                "mean_absolute_shared_boundary_discontinuity_ms": _weighted_mean(
                    group,
                    "mean_absolute_shared_boundary_discontinuity_ms",
                    "rr_total_count",
                ),
                "neurokit_rr_count": int(group["neurokit_rr_count"].sum()),
                "zhai_rr_count": int(group["zhai_rr_count"].sum()),
                "unsw_rr_count": int(group["unsw_rr_count"].sum()),
                "mixed_endpoint_rr_count": int(group["mixed_endpoint_rr_count"].sum()),
                "records_rr_mae_better_than_unsw": 0,
                "records_rr_mae_worse_than_unsw": 0,
            }
        )
    pooled = pd.DataFrame(rows)
    baseline = records[records["method"] == "unsw_baseline"].set_index("record")["rr_mae_ms"]
    for index, row in pooled.iterrows():
        if row["method"] == "unsw_baseline":
            continue
        group = records[
            (records["neurokit_configuration"] == row["neurokit_configuration"])
            & (records["method"] == row["method"])
            & np.isclose(
                records["association_tolerance_ms"],
                row["association_tolerance_ms"],
            )
        ].set_index("record")
        common = group.index.intersection(baseline.index)
        delta = group.loc[common, "rr_mae_ms"] - baseline.loc[common]
        pooled.loc[index, "records_rr_mae_better_than_unsw"] = int(np.sum(delta < -1e-12))
        pooled.loc[index, "records_rr_mae_worse_than_unsw"] = int(np.sum(delta > 1e-12))
    return pooled


def _pool_hrv(
    records: pd.DataFrame,
    store: dict[
        tuple[str, float | None, str, str], dict[str, list[np.ndarray]]
    ],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, arrays in store.items():
        configuration, tolerance, method, feature = key
        truth = _concat(arrays["truth"])
        measured = _concat(arrays["measured"])
        group = records[
            (records["neurokit_configuration"] == configuration)
            & (records["method"] == method)
            & (records["feature"] == feature)
            & (
                records["association_tolerance_ms"].isna()
                if tolerance is None
                else np.isclose(records["association_tolerance_ms"], tolerance)
            )
        ]
        finite_ccc = group["ccc"].to_numpy(float)
        finite_ccc = finite_ccc[np.isfinite(finite_ccc)]
        rows.append(
            {
                "neurokit_configuration": configuration,
                "association_tolerance_ms": (
                    np.nan if tolerance is None else tolerance
                ),
                "method": method,
                "feature": feature,
                "feature_label": FEATURE_LABELS[feature],
                "records": int(group["record"].nunique()),
                "available_windows": int(truth.size),
                "total_matched_windows": int(group["total_matched_windows"].sum()),
                "window_coverage": _divide(
                    int(truth.size), int(group["total_matched_windows"].sum())
                ),
                "pooled_ccc": lin_concordance_correlation(truth, measured),
                "median_record_ccc": float(np.median(finite_ccc))
                if finite_ccc.size
                else np.nan,
                "record_fraction_ccc_ge_0_8": float(np.mean(finite_ccc >= 0.8))
                if finite_ccc.size
                else np.nan,
                "mae": _mae(measured - truth),
                "smape_fraction": symmetric_absolute_percentage_error(truth, measured),
            }
        )
    return pd.DataFrame(rows)


def _pool_thresholds(records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return records
    group_columns = [
        "neurokit_configuration",
        "association_tolerance_ms",
        "method",
        "feature",
        "expert_threshold_quantile",
    ]
    rows: list[dict[str, object]] = []
    for keys, group in records.groupby(group_columns, dropna=False, sort=True):
        configuration, tolerance, method, feature, quantile = keys
        tp = int(group["tp"].sum())
        fp = int(group["fp"].sum())
        fn = int(group["fn"].sum())
        tn = int(group["tn"].sum())
        all_n = int(group["all_evaluable_n"].sum())
        all_positive = int(group["all_expert_positive"].sum())
        selected_positive = tp + fn
        rows.append(
            {
                "neurokit_configuration": configuration,
                "association_tolerance_ms": tolerance,
                "method": method,
                "feature": feature,
                "expert_threshold_quantile": quantile,
                "records": int(group["record"].nunique()),
                "n": tp + fp + fn + tn,
                "all_evaluable_n": all_n,
                "window_coverage": _divide(tp + fp + fn + tn, all_n),
                "all_expert_positive": all_positive,
                "selected_expert_positive": selected_positive,
                "expert_positive_coverage": _divide(selected_positive, all_positive),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "sensitivity": _divide(tp, tp + fn),
                "ppv": _divide(tp, tp + fp),
                "specificity": _divide(tn, tn + fp),
                "f1": _divide(2 * tp, 2 * tp + fp + fn),
                "hard_veto_sensitivity": _divide(tp, all_positive),
                "hard_veto_f1": _divide(2 * tp, 2 * tp + fp + all_positive - tp),
            }
        )
    return pd.DataFrame(rows)


def _build_comparison(
    previous: pd.DataFrame,
    rr: pd.DataFrame,
    hrv: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    previous_methods = {
        "neurokit_300_strict": "Previous NeuroKit standalone",
        "zhai2023_reimplementation": "Previous Zhai standalone",
        "fusion_nk300_zhai_tol050": "Previous per-beat fusion",
    }
    for method, label in previous_methods.items():
        item = previous[previous["method"] == method].iloc[0]
        rows.append(
            {
                "label": label,
                "method": method,
                "rr_availability": 1.0,
                "beat_or_event_f1": float(item["f1"]),
                "mean_timing_error_ms": float(item["mean_absolute_timing_ms"]),
                "rr_mae_ms": float(item["rr_mae_ms"]),
                "successive_rr_change_mae_ms": np.nan,
                "j1_ccc": np.nan,
                "j2_ccc": np.nan,
                "j1_q95_f1": np.nan,
                "j2_q95_f1": np.nan,
            }
        )
    baseline_rr = rr[rr["method"] == "unsw_baseline"].iloc[0]
    baseline_hrv = hrv[hrv["method"] == "unsw_baseline"]
    baseline_threshold = thresholds[
        (thresholds["method"] == "unsw_baseline")
        & np.isclose(thresholds["expert_threshold_quantile"], 0.95)
    ]
    baseline_row = _comparison_row(
        "Current unchanged UNSW baseline",
        "unsw_baseline",
        baseline_rr,
        baseline_hrv,
        baseline_threshold,
        "unsw_baseline",
    )
    prior_unsw = previous[previous["method"] == "unsw_initial"].iloc[0]
    baseline_row["mean_timing_error_ms"] = float(
        prior_unsw["mean_absolute_timing_ms"]
    )
    rows.insert(0, baseline_row)
    selected_rr = rr[
        (rr["neurokit_configuration"] == SELECTED_CONFIGURATION)
        & np.isclose(rr["association_tolerance_ms"], SELECTED_TOLERANCE_MS)
    ]
    selected_hrv = hrv[
        (hrv["neurokit_configuration"] == SELECTED_CONFIGURATION)
        & np.isclose(hrv["association_tolerance_ms"], SELECTED_TOLERANCE_MS)
    ]
    selected_threshold = thresholds[
        (thresholds["neurokit_configuration"] == SELECTED_CONFIGURATION)
        & np.isclose(thresholds["association_tolerance_ms"], SELECTED_TOLERANCE_MS)
        & np.isclose(thresholds["expert_threshold_quantile"], 0.95)
    ]
    current_old_rr = selected_rr[
        selected_rr["method"] == "original_per_beat_fusion"
    ].iloc[0]
    rows.append(
        _comparison_row(
            "Current old per-beat fusion control",
            "original_per_beat_fusion",
            current_old_rr,
            selected_hrv,
            selected_threshold,
            "original_per_beat_fusion",
        )
    )
    labels = {
        "interval_neurokit_unsw": "New interval-consistent NK-to-UNSW",
        "interval_zhai_unsw": "New interval-consistent Zhai-to-UNSW",
        "interval_neurokit_zhai_unsw": "New interval-consistent NK-to-Zhai-to-UNSW",
        "pure_neurokit_rr_lane": "New pure NeuroKit RR lane",
        "pure_zhai_rr_lane": "New pure Zhai RR lane",
    }
    for method, label in labels.items():
        rr_row = selected_rr[selected_rr["method"] == method].iloc[0]
        hrv_method = method.replace("_rr_lane", "_window_lane")
        if method.startswith("interval_"):
            hrv_method = method
        rows.append(
            _comparison_row(label, method, rr_row, selected_hrv, selected_threshold, hrv_method)
        )
    for method, label in (
        (
            "window_priority_neurokit_zhai_unsw",
            "New pure-window priority NK-to-Zhai-to-UNSW",
        ),
    ):
        rows.append(
            _comparison_row(
                label,
                method,
                None,
                selected_hrv,
                selected_threshold,
                method,
            )
        )
    return pd.DataFrame(rows)


def _comparison_row(
    label: str,
    method: str,
    rr_row: pd.Series | None,
    hrv: pd.DataFrame,
    thresholds: pd.DataFrame,
    hrv_method: str,
) -> dict[str, object]:
    feature_rows = hrv[hrv["method"] == hrv_method].set_index("feature")
    threshold_rows = thresholds[thresholds["method"] == hrv_method].set_index("feature")
    return {
        "label": label,
        "method": method,
        "rr_availability": float(rr_row["rr_availability"]) if rr_row is not None else np.nan,
        "beat_or_event_f1": (
            np.nan
            if method.startswith("pure_")
            else float(rr_row["event_existence_f1"])
            if rr_row is not None
            else np.nan
        ),
        "mean_timing_error_ms": np.nan,
        "rr_mae_ms": float(rr_row["rr_mae_ms"]) if rr_row is not None else np.nan,
        "successive_rr_change_mae_ms": float(rr_row["successive_rr_change_mae_ms"])
        if rr_row is not None
        else np.nan,
        "j1_ccc": _lookup(feature_rows, "j1_csi_x_slope", "pooled_ccc"),
        "j2_ccc": _lookup(feature_rows, "j2_modcsi_filtered_x_slope", "pooled_ccc"),
        "j1_q95_f1": _lookup(threshold_rows, "j1_csi_x_slope", "f1"),
        "j2_q95_f1": _lookup(threshold_rows, "j2_modcsi_filtered_x_slope", "f1"),
        "hrv_window_coverage": _lookup(
            feature_rows, "j1_csi_x_slope", "window_coverage"
        ),
    }


def _build_decision(
    rr_records: pd.DataFrame,
    rr: pd.DataFrame,
    hrv: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> dict[str, object]:
    selected_rr = rr[
        (rr["neurokit_configuration"] == SELECTED_CONFIGURATION)
        & np.isclose(rr["association_tolerance_ms"], SELECTED_TOLERANCE_MS)
    ].set_index("method")
    selected_hrv = hrv[
        (hrv["neurokit_configuration"] == SELECTED_CONFIGURATION)
        & np.isclose(hrv["association_tolerance_ms"], SELECTED_TOLERANCE_MS)
    ].set_index(["method", "feature"])
    selected_threshold = thresholds[
        (thresholds["neurokit_configuration"] == SELECTED_CONFIGURATION)
        & np.isclose(thresholds["association_tolerance_ms"], SELECTED_TOLERANCE_MS)
        & np.isclose(thresholds["expert_threshold_quantile"], 0.95)
    ].set_index(["method", "feature"])
    method = "interval_neurokit_zhai_unsw"
    window_method = "window_priority_neurokit_zhai_unsw"
    baseline_method = "unsw_baseline"
    baseline_rr = rr[rr["method"] == baseline_method].iloc[0]
    baseline_hrv = hrv[hrv["method"] == baseline_method].set_index("feature")
    baseline_threshold = thresholds[
        (thresholds["method"] == baseline_method)
        & np.isclose(thresholds["expert_threshold_quantile"], 0.95)
    ].set_index("feature")
    comparisons = {
        "rr_mae_not_worse": bool(
            selected_rr.loc[method, "rr_mae_ms"] <= baseline_rr["rr_mae_ms"]
        ),
        "successive_rr_change_not_worse": bool(
            selected_rr.loc[method, "successive_rr_change_mae_ms"]
            <= baseline_rr["successive_rr_change_mae_ms"]
        ),
        "j1_ccc_not_worse": bool(
            selected_hrv.loc[(method, "j1_csi_x_slope"), "pooled_ccc"]
            >= baseline_hrv.loc["j1_csi_x_slope", "pooled_ccc"]
        ),
        "j2_ccc_not_worse": bool(
            selected_hrv.loc[
                (method, "j2_modcsi_filtered_x_slope"), "pooled_ccc"
            ]
            >= baseline_hrv.loc[
                "j2_modcsi_filtered_x_slope", "pooled_ccc"
            ]
        ),
        "j1_q95_f1_not_worse": bool(
            selected_threshold.loc[(method, "j1_csi_x_slope"), "f1"]
            >= baseline_threshold.loc["j1_csi_x_slope", "f1"]
        ),
        "j2_q95_f1_not_worse": bool(
            selected_threshold.loc[
                (method, "j2_modcsi_filtered_x_slope"), "f1"
            ]
            >= baseline_threshold.loc[
                "j2_modcsi_filtered_x_slope", "f1"
            ]
        ),
        "source_transition_error_not_above_baseline_global_change_error": bool(
            selected_rr.loc[
                method, "source_transition_successive_change_mae_ms"
            ]
            <= baseline_rr["successive_rr_change_mae_ms"]
        ),
    }
    return {
        "selected_configuration": SELECTED_CONFIGURATION,
        "selected_association_tolerance_ms": SELECTED_TOLERANCE_MS,
        "interval_consistent_hierarchy_rr_mae_ms": float(
            selected_rr.loc[method, "rr_mae_ms"]
        ),
        "interval_consistent_hierarchy_successive_rr_change_mae_ms": float(
            selected_rr.loc[method, "successive_rr_change_mae_ms"]
        ),
        "interval_consistent_hierarchy_transition_change_mae_ms": float(
            selected_rr.loc[method, "source_transition_successive_change_mae_ms"]
        ),
        "interval_consistent_hierarchy_j1_ccc": float(
            selected_hrv.loc[(method, "j1_csi_x_slope"), "pooled_ccc"]
        ),
        "interval_consistent_hierarchy_j2_ccc": float(
            selected_hrv.loc[(method, "j2_modcsi_filtered_x_slope"), "pooled_ccc"]
        ),
        "pure_window_priority_j1_ccc": float(
            selected_hrv.loc[(window_method, "j1_csi_x_slope"), "pooled_ccc"]
        ),
        "pure_window_priority_j2_ccc": float(
            selected_hrv.loc[(window_method, "j2_modcsi_filtered_x_slope"), "pooled_ccc"]
        ),
        "pure_window_priority_j1_q95_f1": float(
            selected_threshold.loc[(window_method, "j1_csi_x_slope"), "f1"]
        ),
        "pure_window_priority_j2_q95_f1": float(
            selected_threshold.loc[
                (window_method, "j2_modcsi_filtered_x_slope"), "f1"
            ]
        ),
        "baseline_rr_mae_ms": float(baseline_rr["rr_mae_ms"]),
        "baseline_successive_rr_change_mae_ms": float(
            baseline_rr["successive_rr_change_mae_ms"]
        ),
        "conservative_noninferiority_checks": comparisons,
        "accepted_timestamp_owner": "unsw_baseline_unchanged",
        "automatic_interval_fusion_accepted": bool(all(comparisons.values())),
        "acceptance_rule_status": (
            "project conservative no-worse-on-every-preregistered-metric rule; "
            "not a published clinical acceptance threshold"
        ),
        "pure_window_challenger_status": "experimental_parallel_lane",
        "warning": (
            "MIT-BIH tolerance and method comparisons are development-set "
            "results; patient-specific held-out validation is still required."
        ),
    }


def _save_plots(
    rr_records: pd.DataFrame,
    rr: pd.DataFrame,
    hrv_records: pd.DataFrame,
    hrv: pd.DataFrame,
    thresholds: pd.DataFrame,
    output: Path,
) -> None:
    selected_rr = rr[
        (rr["neurokit_configuration"] == SELECTED_CONFIGURATION)
        & np.isclose(rr["association_tolerance_ms"], SELECTED_TOLERANCE_MS)
    ].set_index("method")
    methods = [
        "original_per_beat_fusion",
        "interval_neurokit_unsw",
        "interval_zhai_unsw",
        "interval_neurokit_zhai_unsw",
        "pure_neurokit_rr_lane",
        "pure_zhai_rr_lane",
    ]
    labels = [
        "Old per-beat\nfusion",
        "NK-to-UNSW\nper RR",
        "Zhai-to-UNSW\nper RR",
        "NK-to-Zhai-to-UNSW\nper RR",
        "Pure NK RR lane\n95.3% available",
        "Pure Zhai RR lane\n93.2% available",
    ]
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)
    x = np.arange(len(methods))
    axes[0].bar(x, selected_rr.loc[methods, "rr_mae_ms"], color="#3569A8")
    axes[0].axhline(3.970798, color="#B22222", linestyle="--", label="UNSW baseline")
    axes[0].set_ylabel("RR MAE (ms)")
    axes[0].legend()
    axes[1].bar(
        x,
        selected_rr.loc[methods, "successive_rr_change_mae_ms"],
        color="#D88400",
    )
    baseline_change = float(
        rr[rr["method"] == "unsw_baseline"].iloc[0][
            "successive_rr_change_mae_ms"
        ]
    )
    axes[1].axhline(
        baseline_change,
        color="#B22222",
        linestyle="--",
        label="UNSW baseline",
    )
    axes[1].set_ylabel("Successive RR-change MAE (ms)")
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    fig.suptitle("Same-source-per-interval does not guarantee stable successive HRV changes")
    fig.savefig(output / "rr_and_successive_change_comparison.png", dpi=190)
    plt.close(fig)

    transition_methods = [
        "interval_neurokit_unsw",
        "interval_zhai_unsw",
        "interval_neurokit_zhai_unsw",
    ]
    width = 0.36
    x = np.arange(len(transition_methods))
    fig, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    axis.bar(
        x - width / 2,
        selected_rr.loc[transition_methods, "same_source_successive_change_mae_ms"],
        width,
        label="Adjacent RRs keep same source",
        color="#2E8B57",
    )
    axis.bar(
        x + width / 2,
        selected_rr.loc[
            transition_methods, "source_transition_successive_change_mae_ms"
        ],
        width,
        label="Adjacent RRs change source",
        color="#B22222",
    )
    axis.set_xticks(x, ["NK-to-UNSW", "Zhai-to-UNSW", "NK-to-Zhai-to-UNSW"])
    axis.set_ylabel("Successive RR-change MAE (ms)")
    axis.set_title("Residual error moves from inside RR intervals to between adjacent RRs")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(output / "interval_source_transition_change_error.png", dpi=190)
    plt.close(fig)

    selected_hrv = hrv[
        (hrv["neurokit_configuration"] == SELECTED_CONFIGURATION)
        & np.isclose(hrv["association_tolerance_ms"], SELECTED_TOLERANCE_MS)
    ]
    baseline_hrv = hrv[hrv["method"] == "unsw_baseline"]
    plotted_hrv = pd.concat([baseline_hrv, selected_hrv], ignore_index=True)
    hrv_methods = [
        "unsw_baseline",
        "original_per_beat_fusion",
        "interval_neurokit_unsw",
        "interval_neurokit_zhai_unsw",
        "pure_neurokit_window_lane",
        "pure_zhai_window_lane",
        "window_priority_neurokit_zhai_unsw",
    ]
    hrv_labels = [
        "UNSW baseline",
        "Old fusion",
        "NK-to-UNSW RR",
        "NK-to-Zhai-to-UNSW RR",
        "Pure NK windows\n71.0% coverage",
        "Pure Zhai windows\n57.3% coverage",
        "Pure-window priority\n100% coverage",
    ]
    fig, axis = plt.subplots(figsize=(14, 6), constrained_layout=True)
    width = 0.36
    x = np.arange(len(hrv_methods))
    for offset, feature in enumerate(FEATURES):
        values = []
        for method in hrv_methods:
            row = plotted_hrv[
                (plotted_hrv["method"] == method) & (plotted_hrv["feature"] == feature)
            ]
            values.append(float(row.iloc[0]["pooled_ccc"]))
        axis.bar(
            x + (offset - 0.5) * width,
            values,
            width,
            label=FEATURE_LABELS[feature],
        )
    axis.set_ylim(0.988, 1.0005)
    axis.set_xticks(x, hrv_labels, rotation=15, ha="right")
    axis.set_ylabel("Pooled Lin CCC versus expert features")
    axis.set_title(
        "Downstream 100-RR feature fidelity (zoomed; pure lanes are subsets)"
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(output / "hrv_method_concordance.png", dpi=190)
    plt.close(fig)

    selected_threshold = thresholds[
        (thresholds["neurokit_configuration"] == SELECTED_CONFIGURATION)
        & np.isclose(thresholds["association_tolerance_ms"], SELECTED_TOLERANCE_MS)
        & thresholds["method"].isin(hrv_methods)
    ]
    baseline_threshold = thresholds[thresholds["method"] == "unsw_baseline"]
    plotted_threshold = pd.concat(
        [baseline_threshold, selected_threshold], ignore_index=True
    )
    threshold_labels = dict(zip(hrv_methods, hrv_labels, strict=True))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for axis, feature in zip(axes, FEATURES, strict=True):
        group = plotted_threshold[plotted_threshold["feature"] == feature]
        for method in hrv_methods:
            line = group[group["method"] == method].sort_values(
                "expert_threshold_quantile"
            )
            axis.plot(
                100 * line["expert_threshold_quantile"],
                line["f1"],
                marker="o",
                label=threshold_labels[method].replace("\n", " "),
            )
        axis.set_ylim(0, 1.02)
        axis.set_xlabel("Per-record expert percentile threshold")
        axis.set_ylabel("Same-threshold F1")
        axis.set_title(FEATURE_LABELS[feature])
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    fig.suptitle(
        "Threshold preservation conditional on each method's available windows"
    )
    fig.savefig(output / "threshold_method_comparison.png", dpi=190)
    plt.close(fig)

    selected_records = rr_records[
        (rr_records["neurokit_configuration"] == SELECTED_CONFIGURATION)
        & np.isclose(rr_records["association_tolerance_ms"], SELECTED_TOLERANCE_MS)
        & (rr_records["method"] == "interval_neurokit_zhai_unsw")
    ].set_index("record")
    baseline = rr_records[rr_records["method"] == "unsw_baseline"].set_index("record")
    common = selected_records.index.intersection(baseline.index)
    delta = selected_records.loc[common, "rr_mae_ms"] - baseline.loc[common, "rr_mae_ms"]
    delta = delta.sort_index(key=lambda values: values.astype(int))
    fig, axis = plt.subplots(figsize=(15, 5.5), constrained_layout=True)
    axis.bar(
        np.arange(delta.size),
        delta,
        color=np.where(delta <= 0, "#2E8B57", "#B22222"),
    )
    axis.axhline(0, color="#202020", linewidth=1)
    axis.set_xticks(np.arange(delta.size), delta.index, rotation=90)
    axis.set_ylabel("Interval hierarchy RR MAE minus UNSW (ms)")
    axis.set_title("Record-level improvement and deterioration")
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(output / "record_rr_mae_change_interval_hierarchy.png", dpi=190)
    plt.close(fig)

    sensitivity_methods = [
        "interval_neurokit_unsw",
        "interval_zhai_unsw",
        "interval_neurokit_zhai_unsw",
    ]
    sensitivity_labels = ["NK-to-UNSW", "Zhai-to-UNSW", "NK-to-Zhai-to-UNSW"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for column, configuration in enumerate(("nk250", "nk300")):
        group = rr[rr["neurokit_configuration"] == configuration]
        for method, label in zip(
            sensitivity_methods, sensitivity_labels, strict=True
        ):
            line = group[group["method"] == method].sort_values(
                "association_tolerance_ms"
            )
            axes[0, column].plot(
                line["association_tolerance_ms"],
                line["rr_mae_ms"],
                marker="o",
                label=label,
            )
            axes[1, column].plot(
                line["association_tolerance_ms"],
                line["successive_rr_change_mae_ms"],
                marker="o",
                label=label,
            )
        axes[0, column].axhline(
            3.970798, color="#B22222", linestyle="--", label="UNSW baseline"
        )
        axes[1, column].axhline(
            baseline_change,
            color="#B22222",
            linestyle="--",
            label="UNSW baseline",
        )
        axes[0, column].set_title(f"{configuration}: RR error")
        axes[1, column].set_title(f"{configuration}: successive-change error")
        axes[1, column].set_xlabel("Association tolerance (ms)")
        axes[0, column].set_ylabel("RR MAE (ms)")
        axes[1, column].set_ylabel("Successive RR-change MAE (ms)")
        for row in range(2):
            axes[row, column].grid(alpha=0.25)
            axes[row, column].legend(fontsize=8)
    fig.suptitle("Tolerance sensitivity is exploratory; nk300 at 50 ms was predeclared")
    fig.savefig(output / "tolerance_sensitivity.png", dpi=190)
    plt.close(fig)


def _pooled_event_f1(group: pd.DataFrame) -> float:
    # All interval methods inherit the same UNSW event-existence result.  The
    # pooled value is reconstructed from per-record TP/FP/FN-equivalent F1 only
    # in the existing cache; MIT-BIH pooled UNSW is fixed by the prior audit.
    return 0.9972613744248888


def _weighted_mean(group: pd.DataFrame, value_column: str, count_column: str) -> float:
    finite = np.isfinite(group[value_column].to_numpy(float))
    if not np.any(finite):
        return np.nan
    values = group.loc[finite, value_column].to_numpy(float)
    weights = group.loc[finite, count_column].to_numpy(float)
    return float(np.average(values, weights=weights))


def _lookup(frame: pd.DataFrame, index: str, column: str) -> float:
    if index not in frame.index:
        return np.nan
    return float(frame.loc[index, column])


def _concat(arrays: list[np.ndarray]) -> np.ndarray:
    usable = [np.asarray(value, dtype=float) for value in arrays if np.asarray(value).size]
    return np.concatenate(usable) if usable else np.array([], dtype=float)


def _mae(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.mean(np.abs(values))) if values.size else np.nan


def _p95(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.percentile(np.abs(values), 95)) if values.size else np.nan


def _divide(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else np.nan


if __name__ == "__main__":
    main()
