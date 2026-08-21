"""Stable output files for inspecting and testing the RR--HRV branch."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .pipeline import RRHRVBranchResult


def save_rr_hrv_outputs(
    result: RRHRVBranchResult,
    output_dir: str | Path,
) -> dict[str, str]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = result.selected

    paths = {
        "rr_hrv_features": output_dir / "rr_hrv_features.csv",
        "rr_intervals": output_dir / "rr_intervals.csv",
        "primary_qrs_events": output_dir / "primary_qrs_events.csv",
        "polarity_context": output_dir / "polarity_context.csv",
        "detector_events": output_dir / "detector_events.csv",
        "metadata": output_dir / "metadata.json",
    }
    selected.features.to_csv(paths["rr_hrv_features"], index=False)
    selected.reliability.rr_intervals.to_csv(paths["rr_intervals"], index=False)
    selected.primary_events.to_csv(paths["primary_qrs_events"], index=False)
    result.polarity_context.to_csv(paths["polarity_context"], index=False)
    _detector_event_table(result).to_csv(paths["detector_events"], index=False)
    paths["metadata"].write_text(
        json.dumps(result.summary(), indent=2, default=_json_default),
        encoding="utf-8",
    )
    return {name: str(path) for name, path in paths.items()}


def _detector_event_table(result: RRHRVBranchResult) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    fs = float(result.metadata["sampling_rate_hz"])
    start_s = float(result.metadata["segment_start_s"])
    for orientation, branch in result.orientations.items():
        detector_specs = [
            ("primary", branch.primary, None),
            ("secondary_experimental", branch.secondary_experimental, None),
            ("secondary_baseline", branch.secondary_baseline, None),
        ]
        detector_specs.extend(
            ("pan_context", detector, delay)
            for delay, detector in branch.pan_context.items()
        )
        for role, detector, delay in detector_specs:
            for sample in detector.peak_samples:
                rows.append(
                    {
                        "orientation": orientation,
                        "role": role,
                        "method": detector.method,
                        "minimum_delay_ms": delay
                        if delay is not None
                        else detector.parameters.get("minimum_delay_ms", np.nan),
                        "sample_in_segment": int(sample),
                        "time_s": start_s + int(sample) / fs,
                    }
                )
    return pd.DataFrame(rows)


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
