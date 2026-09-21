"""Build a label-preserving, multi-dataset ECG branch-validation matrix.

The output deliberately separates numeric model inputs from labels and
provenance.  Different source databases provide different kinds of truth:
beat symbols, manual P/QRS/T landmarks, signal-quality classes, controlled
noise levels, cardiac diagnoses, or seizure intervals.  Those truths must not
be collapsed into one ambiguous target column.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
import hashlib
import itertools
import json
from pathlib import Path
import re
import tomllib
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import wfdb

from .clinical_demo_data import (
    feature_definitions,
    inspect_signal_source,
    load_signal_segment,
    run_demo_analysis,
)
from .delineation_validation import (
    LANDMARK_NAMES,
    ManualDelineationRecord,
    associate_landmarks_with_anchors,
    load_ludb_delineation_record,
    load_qtdb_delineation_record,
)
from .wfdb_io import load_wfdb_beat_annotations


BUILDER_VERSION = "1.0.0"
DEFAULT_PROFILE = "validation"
SUPPORTED_PROFILES = ("validation", "exhaustive")
DEFAULT_SEGMENT_DURATION_S = 180.0
BEAT_MATCH_TOLERANCE_MS = 75.0
QUALITY_LOOKBACK_S = 10.0
SEIZURE_CONTEXT_S = 300.0

# Signal-only detector-support audit of LUDB 1.0.1.  Lead II yields fewer than
# enough usable NeuroKit anchors in these records, while lead I yields 11, 8,
# and 9 anchors, respectively.  The corresponding manual annotations are loaded from the
# same selected lead; no diagnosis or delineation label is used for routing.
LUDB_SIGNAL_ONLY_LEAD_OVERRIDES = {
    "data/45": "i",
    "data/74": "i",
    "data/90": "i",
}

DATASET_KEYS = (
    "mitdb",
    "nstdb",
    "ludb",
    "qtdb",
    "butqdb",
    "seizure_edf",
)

AAMI_BEAT_CLASS = {
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",
    "n": "N",
    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",
    "V": "V",
    "E": "V",
    "r": "V",
    "B": "V",
    "F": "F",
    "/": "Q",
    "f": "Q",
    "Q": "Q",
    "?": "Q",
}

QUALITY_CLASS_TEXT = {
    "0": "not annotated",
    "1": "all significant waveforms visible; onsets and offsets reliably detectable",
    "2": "significant points unreliable; QRS detection remains reliable",
    "3": "QRS detection unreliable; unsuitable for further analysis",
}

BRANCH_NAMES = {
    "B1": "Peak detection, RR, HRV, and PRSA/BPRSA context",
    "B2": "Patient morphology, template morphology, and P-QRS-T morphology",
    "B3": "Conduction and repolarization information",
    "B4": "Signal-quality context",
}


@dataclass(frozen=True)
class DatasetRoots:
    """Resolved local inputs used by the validation profile."""

    project: Path
    mitdb: Path
    nstdb: Path
    ludb: Path
    qtdb: Path
    butqdb: Path
    butqdb_metadata: Path
    seizure_config: Path


@dataclass(frozen=True)
class SegmentSpec:
    """One bounded, reproducible extraction request."""

    dataset_key: str
    dataset_name: str
    record_id: str
    subject_id: str
    source_path: str
    channel: str
    start_s: float
    duration_s: float
    calibration_beats: int
    selection_reason: str
    lineage_group_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def end_s(self) -> float:
        return float(self.start_s + self.duration_s)

    @property
    def segment_id(self) -> str:
        raw = (
            f"{BUILDER_VERSION}|{self.dataset_key}|{self.record_id}|"
            f"{self.channel}|{self.start_s:.6f}|{self.duration_s:.6f}"
        )
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
        return f"{self.dataset_key}_{_slug(self.record_id)}_{digest}"


@dataclass
class LabelContext:
    """Labels and reference measurements for one source recording."""

    beat_samples: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=np.int64)
    )
    beat_symbols: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=object)
    )
    quality_intervals: dict[str, pd.DataFrame] = field(default_factory=dict)
    seizure_intervals: tuple[tuple[float, float], ...] = ()
    record_comments: tuple[str, ...] = ()
    diagnoses: tuple[str, ...] = ()
    rhythm: str | None = None
    manual_truth: pd.DataFrame = field(default_factory=pd.DataFrame)
    noise_condition: str | None = None
    noise_snr_db: float | None = None
    source_age: str | None = None
    source_sex: str | None = None


def default_dataset_roots(project: str | Path | None = None) -> DatasetRoots:
    project_path = (
        Path(project).expanduser().resolve()
        if project is not None
        else Path(__file__).resolve().parents[3]
    )
    datasets = project_path / "Datasets"
    return DatasetRoots(
        project=project_path,
        mitdb=(
            datasets
            / "mit-bih-arrhythmia-database-1.0"
            / "mit-bih-arrhythmia-database-1.0.0"
        ),
        nstdb=(
            datasets
            / "mit-bih-noise-stress-test-database-1.0"
            / "mit-bih-noise-stress-test-database-1.0.0"
        ),
        ludb=datasets / "lobachevsky-university-electrocardiography-database-1.0.1",
        qtdb=datasets / "qt-database-1.0.0",
        butqdb=(
            datasets
            / "brno-university-of-technology-ecg-quality-database-but-qdb-1.0.0"
        ),
        butqdb_metadata=datasets / "butqdb-1.0.0",
        seizure_config=project_path / "feature_extraction" / "config" / "datasets.toml",
    )


def feature_keys() -> list[str]:
    """Return the frozen 52-column model-input contract in display order."""

    keys = [str(item["key"]) for item in feature_definitions()]
    if len(keys) != 52 or len(set(keys)) != 52:
        raise RuntimeError(
            f"Expected 52 unique feature columns from clinical_demo_data, got {len(keys)}"
        )
    return keys


def branch_feature_map() -> dict[str, list[str]]:
    """Map the maintained 52-column schema to the four research branches."""

    output = {key: [] for key in BRANCH_NAMES}
    for key in feature_keys():
        if key.startswith(("hrv_", "prsa_", "bprsa_")):
            output["B1"].append(key)
        elif key.startswith(("cal_varon_", "template_", "pqrst_")):
            output["B2"].append(key)
        elif key.startswith("conduction_"):
            output["B3"].append(key)
        elif key.startswith("quality_"):
            output["B4"].append(key)
        else:
            raise RuntimeError(f"Feature {key!r} is not assigned to a branch")
    assigned = [value for values in output.values() for value in values]
    if assigned != [
        key for branch in ("B2", "B1", "B4", "B3") for key in output[branch]
    ]:
        # The feature-definition display order intentionally differs from B1-B4 order.
        if set(assigned) != set(feature_keys()):
            raise RuntimeError("Branch feature map does not cover the 52-column contract")
    return output


def branch_combinations() -> list[dict[str, Any]]:
    """Enumerate all 15 non-empty B1-B4 feature-set ablations."""

    mapping = branch_feature_map()
    rows: list[dict[str, Any]] = []
    branch_order = tuple(BRANCH_NAMES)
    for size in range(1, len(branch_order) + 1):
        for selected in itertools.combinations(branch_order, size):
            columns = [column for branch in selected for column in mapping[branch]]
            rows.append(
                {
                    "combination_id": "+".join(selected),
                    "branch_count": len(selected),
                    "branches": "+".join(selected),
                    "branch_names_json": json.dumps(
                        [BRANCH_NAMES[branch] for branch in selected], ensure_ascii=False
                    ),
                    "feature_count": len(columns),
                    "feature_columns_json": json.dumps(columns),
                }
            )
    if len(rows) != 15:
        raise RuntimeError("Expected 15 non-empty combinations of four branches")
    return rows


def plan_validation_segments(
    roots: DatasetRoots,
    *,
    datasets: Sequence[str] = DATASET_KEYS,
    segment_duration_s: float = DEFAULT_SEGMENT_DURATION_S,
    max_records_per_dataset: int | None = None,
) -> list[SegmentSpec]:
    """Plan a bounded, label-aware validation sample from every dataset family.

    The validation profile covers every LUDB record, every branch-qualified
    QTDB MLII record, all 48 MIT-BIH records, all 12 current NSTDB stress
    records, every BUT-QDB record/class combination, and every configured
    seizure interval plus an inter-event control.  It avoids processing the
    unannotated majority of the 24-hour BUT-QDB signals.
    """

    requested = tuple(dict.fromkeys(str(value) for value in datasets))
    unknown = set(requested).difference(DATASET_KEYS)
    if unknown:
        raise ValueError(f"Unknown dataset keys: {sorted(unknown)}")
    if segment_duration_s < 120.0:
        raise ValueError("segment_duration_s must be at least 120 seconds")

    planners = {
        "mitdb": lambda: _plan_mitdb(roots, segment_duration_s),
        "nstdb": lambda: _plan_nstdb(roots, segment_duration_s),
        "ludb": lambda: _plan_ludb(roots),
        "qtdb": lambda: _plan_qtdb(roots, segment_duration_s),
        "butqdb": lambda: _plan_butqdb(roots, segment_duration_s),
        "seizure_edf": lambda: _plan_seizure_edf(roots, segment_duration_s),
    }
    planned: list[SegmentSpec] = []
    for dataset_key in requested:
        current = planners[dataset_key]()
        if max_records_per_dataset is not None:
            record_ids = list(dict.fromkeys(spec.record_id for spec in current))[
                : int(max_records_per_dataset)
            ]
            keep = set(record_ids)
            current = [spec for spec in current if spec.record_id in keep]
        planned.extend(current)
    return sorted(
        planned,
        key=lambda item: (item.dataset_key, item.record_id, item.start_s),
    )


def plan_exhaustive_segments(
    roots: DatasetRoots,
    *,
    datasets: Sequence[str] = DATASET_KEYS,
    segment_duration_s: float = DEFAULT_SEGMENT_DURATION_S,
    max_records_per_dataset: int | None = None,
) -> list[SegmentSpec]:
    """Plan gap-free, non-overlapping windows across every supported record.

    The exhaustive profile is intentionally much larger than the validation
    profile.  It covers every sample in each included recording (merging a
    final short remainder into the preceding window), while retaining the same
    record/patient lineage groups and source-label loaders.
    """

    requested = tuple(dict.fromkeys(str(value) for value in datasets))
    unknown = set(requested).difference(DATASET_KEYS)
    if unknown:
        raise ValueError(f"Unknown dataset keys: {sorted(unknown)}")
    if segment_duration_s < 120.0:
        raise ValueError("segment_duration_s must be at least 120 seconds")

    prototype_planners = {
        "mitdb": lambda: _plan_mitdb(roots, segment_duration_s),
        "nstdb": lambda: _plan_nstdb(roots, segment_duration_s),
        "ludb": lambda: _plan_ludb(roots),
        "qtdb": lambda: _plan_qtdb_exhaustive(roots),
        "butqdb": lambda: _plan_butqdb(roots, segment_duration_s),
        "seizure_edf": lambda: _plan_seizure_edf(roots, segment_duration_s),
    }
    planned: list[SegmentSpec] = []
    for dataset_key in requested:
        by_record: dict[str, SegmentSpec] = {}
        for prototype in prototype_planners[dataset_key]():
            by_record.setdefault(prototype.record_id, prototype)
        prototypes = list(by_record.values())
        if max_records_per_dataset is not None:
            prototypes = prototypes[: int(max_records_per_dataset)]
        for prototype in prototypes:
            source = inspect_signal_source(prototype.source_path)
            record_duration = float(source.duration_s or 0.0)
            if record_duration < 3.0:
                continue
            for start_s, end_s in _complete_record_windows(
                record_duration, segment_duration_s
            ):
                planned.append(
                    replace(
                        prototype,
                        start_s=start_s,
                        duration_s=end_s - start_s,
                        selection_reason=(
                            "exhaustive full-record coverage; gap-free "
                            "non-overlapping window"
                        ),
                    )
                )
    return sorted(
        planned,
        key=lambda item: (item.dataset_key, item.record_id, item.start_s),
    )


def _complete_record_windows(
    record_duration_s: float,
    window_duration_s: float,
) -> list[tuple[float, float]]:
    """Cover a recording without gaps and avoid an unusably short tail."""

    if record_duration_s <= window_duration_s:
        return [(0.0, record_duration_s)]
    boundaries = list(np.arange(0.0, record_duration_s, window_duration_s))
    windows = [
        (float(start), float(min(record_duration_s, start + window_duration_s)))
        for start in boundaries
    ]
    if len(windows) > 1 and windows[-1][1] - windows[-1][0] < 120.0:
        windows[-2] = (windows[-2][0], record_duration_s)
        windows.pop()
    return windows


def build_comprehensive_dataset(
    output_dir: str | Path,
    *,
    roots: DatasetRoots | None = None,
    datasets: Sequence[str] = DATASET_KEYS,
    tracks: Sequence[str] = ("neurokit",),
    segment_duration_s: float = DEFAULT_SEGMENT_DURATION_S,
    max_records_per_dataset: int | None = None,
    resume: bool = True,
    profile: str = DEFAULT_PROFILE,
) -> dict[str, Any]:
    """Execute one profile and write separated feature/label views."""

    roots = roots or default_dataset_roots()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache_dir = output / "_segment_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    selected_tracks = tuple(dict.fromkeys(str(value).casefold() for value in tracks))
    if not selected_tracks or set(selected_tracks).difference({"neurokit", "zhai"}):
        raise ValueError("tracks must contain neurokit and/or zhai")

    normalized_profile = str(profile).casefold()
    if normalized_profile not in SUPPORTED_PROFILES:
        raise ValueError(f"profile must be one of {SUPPORTED_PROFILES}")
    planner = (
        plan_exhaustive_segments
        if normalized_profile == "exhaustive"
        else plan_validation_segments
    )
    specs = planner(
        roots,
        datasets=datasets,
        segment_duration_s=segment_duration_s,
        max_records_per_dataset=max_records_per_dataset,
    )
    label_cache: dict[tuple[str, str], LabelContext] = {}
    frames: list[pd.DataFrame] = []
    execution_rows: list[dict[str, Any]] = []
    total = len(specs) * len(selected_tracks)
    completed = 0
    for spec in specs:
        context_key = (spec.dataset_key, spec.record_id)
        if context_key not in label_cache:
            label_cache[context_key] = _load_label_context(spec, roots)
        context = label_cache[context_key]
        for track in selected_tracks:
            completed += 1
            cache_path = cache_dir / f"{spec.segment_id}_{track}.csv"
            cache_meta = cache_path.with_suffix(".json")
            try:
                if resume and _valid_cache(cache_path, cache_meta):
                    frame = pd.read_csv(
                        cache_path,
                        low_memory=False,
                        dtype={
                            "row_id": "string",
                            "segment_id": "string",
                            "dataset_key": "string",
                            "dataset_name": "string",
                            "record_id": "string",
                            "subject_id": "string",
                            "lineage_group_id": "string",
                            "source_path": "string",
                            "source_kind": "string",
                            "channel": "string",
                            "input_unit": "string",
                            "timestamp_track": "string",
                            "selection_reason": "string",
                        },
                    )
                    status = "cached"
                else:
                    frame = _run_segment(spec, track, context)
                    frame.to_csv(cache_path, index=False)
                    cache_meta.write_text(
                        json.dumps(
                            {
                                "builder_version": BUILDER_VERSION,
                                "segment_id": spec.segment_id,
                                "track": track,
                                "rows": int(frame.shape[0]),
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    status = "executed"
                frames.append(frame)
                error = ""
                row_count = int(frame.shape[0])
            except Exception as exc:  # retain every failure in the manifest
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
                row_count = 0
            execution_rows.append(
                {
                    **_segment_manifest_row(spec),
                    "timestamp_track": track,
                    "execution_status": status,
                    "output_rows": row_count,
                    "error": error,
                }
            )
            print(
                f"[{completed}/{total}] {status}: {spec.dataset_key}/"
                f"{spec.record_id} {spec.start_s:.1f}-{spec.end_s:.1f}s {track} "
                f"rows={row_count}",
                flush=True,
            )

    observations = (
        pd.concat(frames, ignore_index=True, sort=False)
        if frames
        else _empty_observations()
    )
    if not observations.empty:
        observations = observations.sort_values(
            ["dataset_key", "record_id", "timestamp_track", "observation_time_s"],
            kind="stable",
        ).reset_index(drop=True)
        for column in (
            "label_quality_annotator_1_original",
            "label_quality_annotator_2_original",
            "label_quality_annotator_3_original",
            "label_quality_consensus_original",
        ):
            observations[column] = observations[column].map(_canonical_label_string)
        observations = _apply_binary_targets(observations)
        duplicate_ids = observations["row_id"].duplicated(keep=False)
        if duplicate_ids.any():
            duplicated = observations.loc[duplicate_ids, "row_id"].tolist()[:5]
            raise RuntimeError(f"Duplicate row_id values detected: {duplicated}")

    execution = pd.DataFrame(execution_rows)
    artifacts = _write_dataset_artifacts(
        observations, execution, output, profile=normalized_profile
    )
    report = _validate_outputs(
        observations, execution, artifacts, profile=normalized_profile
    )
    (output / "validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "README.md").write_text(
        _dataset_readme(report), encoding="utf-8"
    )
    return report


def _plan_mitdb(roots: DatasetRoots, duration_s: float) -> list[SegmentSpec]:
    records = _read_records(roots.mitdb)
    annotations: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    counts: Counter[str] = Counter()
    for record_id in records:
        samples, symbols = load_wfdb_beat_annotations(roots.mitdb / record_id)
        annotations[record_id] = (samples, symbols)
        counts.update(str(value) for value in symbols.tolist())
    specs: list[SegmentSpec] = []
    for record_id in records:
        base = roots.mitdb / record_id
        header = wfdb.rdheader(str(base))
        record_duration = float(header.sig_len / header.fs)
        samples, symbols = annotations[record_id]
        start_s = _best_label_diversity_start(
            samples,
            symbols,
            sampling_rate_hz=float(header.fs),
            record_duration_s=record_duration,
            segment_duration_s=duration_s,
            global_counts=counts,
        )
        specs.append(
            SegmentSpec(
                dataset_key="mitdb",
                dataset_name="MIT-BIH Arrhythmia Database",
                record_id=record_id,
                subject_id=f"mitdb_{record_id}",
                source_path=str(base.with_suffix(".hea")),
                channel=str(header.sig_name[0]),
                start_s=start_s,
                duration_s=min(duration_s, record_duration - start_s),
                calibration_beats=20,
                selection_reason="record coverage; label-diversity weighted beat-symbol window",
                lineage_group_id=f"mitdb:{record_id}",
                metadata={"annotation_extension": "atr"},
            )
        )
    return specs


def _plan_nstdb(roots: DatasetRoots, duration_s: float) -> list[SegmentSpec]:
    specs: list[SegmentSpec] = []
    pattern = re.compile(r"^(118|119)e(24|18|12|06|00|_6)$")
    for header_path in sorted(roots.nstdb.glob("*.hea")):
        record_id = header_path.stem
        match = pattern.match(record_id)
        if match is None:
            continue
        header = wfdb.rdheader(str(header_path.with_suffix("")))
        record_duration = float(header.sig_len / header.fs)
        actual_duration = min(duration_s, record_duration)
        # The first official noisy interval is 300-420 s; include 30 seconds
        # of causal context on each side when the default 180-s window is used.
        start_s = max(0.0, 300.0 - max(0.0, (actual_duration - 120.0) / 2.0))
        start_s = min(start_s, max(0.0, record_duration - actual_duration))
        source_record = match.group(1)
        snr_token = match.group(2)
        snr_db = -6 if snr_token == "_6" else int(snr_token)
        specs.append(
            SegmentSpec(
                dataset_key="nstdb",
                dataset_name="MIT-BIH Noise Stress Test Database",
                record_id=record_id,
                subject_id=f"mitdb_{source_record}",
                source_path=str(header_path),
                channel=str(header.sig_name[0]),
                start_s=start_s,
                duration_s=actual_duration,
                calibration_beats=20,
                selection_reason="first official electrode-motion interval plus causal context",
                lineage_group_id=f"mitdb:{source_record}",
                metadata={
                    "source_record": source_record,
                    "noise_type_original": "em",
                    "noise_condition_original": f"{snr_db:+d} dB",
                    "snr_db_original": snr_db,
                },
            )
        )
    return specs


def _plan_ludb(roots: DatasetRoots) -> list[SegmentSpec]:
    specs: list[SegmentSpec] = []
    for record_id in _read_records(roots.ludb):
        base = roots.ludb / record_id
        header = wfdb.rdheader(str(base))
        preferred_matches = [
            index
            for index, name in enumerate(header.sig_name)
            if str(name).casefold() == "ii"
        ]
        if len(preferred_matches) != 1:
            continue
        selected_lead = LUDB_SIGNAL_ONLY_LEAD_OVERRIDES.get(record_id, "ii")
        selected_matches = [
            index
            for index, name in enumerate(header.sig_name)
            if str(name).casefold() == selected_lead
        ]
        if len(selected_matches) != 1:
            raise RuntimeError(
                f"LUDB {record_id} does not contain selected lead {selected_lead!r}"
            )
        record_duration = float(header.sig_len / header.fs)
        used_fallback = selected_lead != "ii"
        specs.append(
            SegmentSpec(
                dataset_key="ludb",
                dataset_name="Lobachevsky University ECG Database",
                record_id=record_id,
                subject_id=f"ludb_{Path(record_id).name}",
                source_path=str(base.with_suffix(".hea")),
                channel=str(header.sig_name[selected_matches[0]]),
                start_s=0.0,
                duration_s=record_duration,
                calibration_beats=5,
                selection_reason=(
                    "all 200 records; complete 10-second manual lead; "
                    "lead-I signal-only detector-support fallback"
                    if used_fallback
                    else "all 200 records; complete 10-second manual lead-II record"
                ),
                lineage_group_id=f"ludb:{Path(record_id).name}",
                metadata={
                    "annotation_extension": selected_lead,
                    "preferred_lead": "ii",
                    "signal_only_lead_fallback": used_fallback,
                },
            )
        )
    return specs


def _plan_qtdb(roots: DatasetRoots, duration_s: float) -> list[SegmentSpec]:
    specs: list[SegmentSpec] = []
    for record_id in _read_records(roots.qtdb):
        base = roots.qtdb / record_id
        if not base.with_suffix(".q1c").is_file():
            continue
        header = wfdb.rdheader(str(base))
        if str(header.sig_name[0]).casefold() != "mlii":
            continue
        record_duration = float(header.sig_len / header.fs)
        annotation = wfdb.rdann(str(base), "q1c")
        center_s = (
            float(np.median(annotation.sample) / header.fs)
            if len(annotation.sample)
            else record_duration / 2.0
        )
        actual_duration = min(duration_s, record_duration)
        start_s = min(
            max(0.0, center_s - actual_duration / 2.0),
            max(0.0, record_duration - actual_duration),
        )
        comments = tuple(str(value) for value in (header.comments or []))
        source = _source_record_from_comments(comments)
        lineage = f"qtdb:{record_id}"
        if source and re.fullmatch(r"\d+", source):
            lineage = f"mitdb:{source}"
        specs.append(
            SegmentSpec(
                dataset_key="qtdb",
                dataset_name="QT Database",
                record_id=record_id,
                subject_id=f"qtdb_{record_id}",
                source_path=str(base.with_suffix(".hea")),
                channel=str(header.sig_name[0]),
                start_s=start_s,
                duration_s=actual_duration,
                calibration_beats=20,
                selection_reason="branch-qualified channel-0 MLII record centered on q1c manual beats",
                lineage_group_id=lineage,
                metadata={
                    "annotation_extension": "q1c",
                    "source_record_original": source or "",
                },
            )
        )
    return specs


def _plan_qtdb_exhaustive(roots: DatasetRoots) -> list[SegmentSpec]:
    """Create one full-record prototype for every QTDB record with q1c truth."""

    specs: list[SegmentSpec] = []
    for record_id in _read_records(roots.qtdb):
        base = roots.qtdb / record_id
        if not base.with_suffix(".q1c").is_file():
            continue
        header = wfdb.rdheader(str(base))
        if not header.sig_name:
            continue
        record_duration = float(header.sig_len / header.fs)
        comments = tuple(str(value) for value in (header.comments or []))
        source = _source_record_from_comments(comments)
        lineage = f"qtdb:{record_id}"
        if source and re.fullmatch(r"\d+", source):
            lineage = f"mitdb:{source}"
        specs.append(
            SegmentSpec(
                dataset_key="qtdb",
                dataset_name="QT Database",
                record_id=record_id,
                subject_id=f"qtdb_{record_id}",
                source_path=str(base.with_suffix(".hea")),
                channel=str(header.sig_name[0]),
                start_s=0.0,
                duration_s=record_duration,
                calibration_beats=20,
                selection_reason=(
                    "exhaustive QTDB record with q1c manual delineation; "
                    "channel 0 retained exactly"
                ),
                lineage_group_id=lineage,
                metadata={
                    "annotation_extension": "q1c",
                    "source_record_original": source or "",
                },
            )
        )
    return specs


def _plan_butqdb(roots: DatasetRoots, duration_s: float) -> list[SegmentSpec]:
    specs: list[SegmentSpec] = []
    for record_dir in sorted(path for path in roots.butqdb.iterdir() if path.is_dir()):
        record_id = record_dir.name
        header_path = record_dir / f"{record_id}_ECG.hea"
        annotation_path = record_dir / f"{record_id}_ANN.csv"
        if not header_path.is_file() or not annotation_path.is_file():
            continue
        header = wfdb.rdheader(str(header_path.with_suffix("")))
        record_duration = float(header.sig_len / header.fs)
        consensus = _read_butqdb_annotations(annotation_path)["consensus"]
        candidates: list[tuple[float, float, str]] = []
        for quality_class in ("1", "2", "3"):
            class_rows = consensus.loc[
                consensus["label_original"] == quality_class
            ].copy()
            if class_rows.empty:
                continue
            longest = class_rows.assign(
                length=class_rows["end_s"] - class_rows["start_s"]
            ).sort_values("length", ascending=False, kind="stable").iloc[0]
            actual_duration = min(duration_s, record_duration)
            center_s = 0.5 * (float(longest.start_s) + float(longest.end_s))
            start_s = min(
                max(0.0, center_s - actual_duration / 2.0),
                max(0.0, record_duration - actual_duration),
            )
            candidates.append((start_s, start_s + actual_duration, quality_class))
        for start_s, end_s, reasons in _merge_ranges(candidates):
            specs.append(
                SegmentSpec(
                    dataset_key="butqdb",
                    dataset_name="Brno University of Technology ECG Quality Database",
                    record_id=record_id,
                    subject_id=f"butqdb_{record_id[:3]}",
                    source_path=str(header_path),
                    channel=str(header.sig_name[0]),
                    start_s=start_s,
                    duration_s=end_s - start_s,
                    calibration_beats=20,
                    selection_reason=(
                        "longest consensus interval sample for official quality class(es) "
                        + ",".join(reasons)
                    ),
                    lineage_group_id=f"butqdb:{record_id[:3]}",
                    metadata={
                        "annotation_path": str(annotation_path),
                        "planned_quality_classes": list(reasons),
                    },
                )
            )
    return specs


def _plan_seizure_edf(roots: DatasetRoots, duration_s: float) -> list[SegmentSpec]:
    config = tomllib.loads(roots.seizure_config.read_text(encoding="utf-8"))
    specs: list[SegmentSpec] = []
    for item in config.get("recordings", []):
        path = Path(str(item["path"])).expanduser().resolve()
        if not path.is_file():
            continue
        source = inspect_signal_source(path)
        record_id = str(item["id"])
        patient = str(item["patient"])
        dataset_name = (
            "CHB-MIT Scalp EEG Database ECG channel"
            if record_id.startswith("chb")
            else "Siena Scalp EEG Database ECG channel"
        )
        dataset_prefix = "chb_mit" if record_id.startswith("chb") else "siena"
        intervals = tuple(
            (float(pair[0]), float(pair[1])) for pair in item.get("seizures_s", [])
        )
        record_duration = float(source.duration_s or 0.0)
        candidates: list[tuple[float, float, str]] = []
        for event_index, (onset, offset) in enumerate(intervals, start=1):
            start_s = max(0.0, onset - SEIZURE_CONTEXT_S)
            end_s = min(record_duration, offset + SEIZURE_CONTEXT_S)
            candidates.append((start_s, end_s, f"seizure_event_{event_index}"))
        control_duration = min(duration_s, record_duration)
        if control_duration >= 3.0:
            control_start = _control_start(record_duration, intervals, control_duration)
            candidates.append(
                (control_start, control_start + control_duration, "inter-event control")
            )
        for start_s, end_s, reasons in _merge_ranges(candidates):
            specs.append(
                SegmentSpec(
                    dataset_key="seizure_edf",
                    dataset_name=dataset_name,
                    record_id=record_id,
                    subject_id=patient,
                    source_path=str(path),
                    channel=str(item["ecg_channel"]),
                    start_s=start_s,
                    duration_s=end_s - start_s,
                    calibration_beats=20,
                    selection_reason="; ".join(reasons),
                    lineage_group_id=f"{dataset_prefix}:{patient}",
                    metadata={
                        "seizure_intervals_s": [list(pair) for pair in intervals],
                        "expected_sampling_rate_hz": float(
                            item["expected_sampling_rate_hz"]
                        ),
                    },
                )
            )
    return specs


def _run_segment(
    spec: SegmentSpec,
    timestamp_track: str,
    context: LabelContext,
) -> pd.DataFrame:
    source = inspect_signal_source(spec.source_path)
    segment = load_signal_segment(
        source,
        channel=spec.channel,
        start_s=spec.start_s,
        duration_s=spec.duration_s,
    )
    payload = run_demo_analysis(
        segment,
        timestamp_track=timestamp_track,
        calibration_beats=spec.calibration_beats,
    )
    return _flatten_payload(payload, spec, context)


def _flatten_payload(
    payload: Mapping[str, Any],
    spec: SegmentSpec,
    context: LabelContext,
) -> pd.DataFrame:
    keys = feature_keys()
    branch_map = branch_feature_map()
    rows: list[dict[str, Any]] = []
    for fusion_row in payload.get("fusion", []):
        time_s = float(fusion_row["time_s"])
        values = fusion_row.get("values", {})
        raw_id = (
            f"{BUILDER_VERSION}|{spec.segment_id}|{payload['track']['key']}|{time_s:.9f}"
        )
        row_id = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:24]
        row: dict[str, Any] = {
            "row_id": row_id,
            "segment_id": spec.segment_id,
            "dataset_key": spec.dataset_key,
            "dataset_name": spec.dataset_name,
            "record_id": spec.record_id,
            "subject_id": spec.subject_id,
            "lineage_group_id": spec.lineage_group_id,
            "suggested_cv_fold": stable_group_fold(spec.lineage_group_id),
            "source_path": spec.source_path,
            "source_kind": payload["source"]["kind"],
            "channel": payload["source"]["channel"],
            "sampling_rate_hz": float(payload["source"]["sampling_rate_hz"]),
            "input_unit": payload["source"]["unit"],
            "segment_start_s": float(payload["source"]["start_s"]),
            "segment_end_s": float(
                payload["source"]["start_s"] + payload["source"]["duration_s"]
            ),
            "observation_time_s": time_s,
            "timestamp_track": payload["track"]["key"],
            "calibration_beats_requested": spec.calibration_beats,
            "selection_reason": spec.selection_reason,
            "measurements_defined": bool(fusion_row.get("measurements_defined", False)),
            "core_context_pass": bool(fusion_row.get("context_pass", False)),
            "morphology_defined": bool(fusion_row.get("morphology_defined", False)),
            "hrv_defined": bool(fusion_row.get("hrv_defined", False)),
            "hrv_reliable": bool(fusion_row.get("hrv_reliable", False)),
            "prsa_bprsa_defined": bool(
                fusion_row.get("prsa_bprsa_defined", False)
            ),
            "prsa_bprsa_reliable": bool(
                fusion_row.get("prsa_bprsa_reliable", False)
            ),
            "signal_quality_context_available": bool(
                fusion_row.get("signal_quality_context_available", False)
            ),
            "conduction_information_available": bool(
                fusion_row.get("conduction_information_available", False)
            ),
            "artifact_label_produced_by_pipeline": bool(
                fusion_row.get("signal_quality_artifact_label_produced", False)
            ),
            "conduction_final_matrix_eligible": bool(
                fusion_row.get("conduction_final_matrix_eligible", False)
            ),
        }
        for key in keys:
            row[key] = _finite_or_nan(values.get(key))
        for branch, columns in branch_map.items():
            available = sum(pd.notna(row[column]) for column in columns)
            row[f"{branch.lower()}_available_feature_count"] = int(available)
            row[f"{branch.lower()}_feature_count"] = len(columns)
            row[f"{branch.lower()}_all_features_available"] = bool(
                available == len(columns)
            )
        available_count = sum(pd.notna(row[key]) for key in keys)
        row["available_feature_count"] = int(available_count)
        row["missing_feature_count"] = int(len(keys) - available_count)
        row.update(_labels_at_time(context, spec, time_s, row["sampling_rate_hz"]))
        rows.append(row)
    return pd.DataFrame(rows, columns=_observation_columns())


def _load_label_context(spec: SegmentSpec, roots: DatasetRoots) -> LabelContext:
    context = LabelContext()
    path = Path(spec.source_path)
    record_base = path.with_suffix("")
    if record_base.with_suffix(".hea").is_file():
        header = wfdb.rdheader(str(record_base))
        context.record_comments = tuple(str(value) for value in (header.comments or []))
        context.source_age, context.source_sex = _age_sex_from_comments(
            context.record_comments
        )
        atr_path = record_base.with_suffix(".atr")
        if atr_path.is_file():
            context.beat_samples, context.beat_symbols = load_wfdb_beat_annotations(
                record_base
            )

    if spec.dataset_key == "butqdb":
        annotation_path = Path(str(spec.metadata["annotation_path"]))
        context.quality_intervals = _read_butqdb_annotations(annotation_path)
        demographics = _butqdb_demographics(roots.butqdb_metadata)
        demo = demographics.get(spec.record_id, {})
        context.source_age = str(demo.get("Age", "")) or None
        context.source_sex = str(demo.get("Gender", "")) or None
    elif spec.dataset_key == "seizure_edf":
        context.seizure_intervals = tuple(
            (float(pair[0]), float(pair[1]))
            for pair in spec.metadata.get("seizure_intervals_s", [])
        )
    elif spec.dataset_key == "nstdb":
        context.noise_condition = str(spec.metadata["noise_condition_original"])
        context.noise_snr_db = float(spec.metadata["snr_db_original"])
    elif spec.dataset_key == "ludb":
        manual = load_ludb_delineation_record(
            roots.ludb, spec.record_id, lead=spec.channel
        )
        context.manual_truth = _manual_truth_table(manual)
        context.diagnoses, context.rhythm = _ludb_diagnoses(context.record_comments)
    elif spec.dataset_key == "qtdb":
        manual = load_qtdb_delineation_record(
            roots.qtdb,
            spec.record_id,
            annotation_extension="q1c",
            required_channel0=spec.channel,
        )
        context.manual_truth = _manual_truth_table(manual)
    return context


def _labels_at_time(
    context: LabelContext,
    spec: SegmentSpec,
    time_s: float,
    sampling_rate_hz: float,
) -> dict[str, Any]:
    labels = {column: None for column in _label_columns()}
    labels["label_record_comments_original_json"] = (
        json.dumps(context.record_comments, ensure_ascii=False)
        if context.record_comments
        else None
    )
    labels["label_ludb_diagnoses_original_json"] = (
        json.dumps(context.diagnoses, ensure_ascii=False) if context.diagnoses else None
    )
    labels["label_ludb_rhythm_original"] = context.rhythm
    labels["source_age_original"] = context.source_age
    labels["source_sex_original"] = context.source_sex

    labels.update(
        _beat_labels_at_time(
            context.beat_samples,
            context.beat_symbols,
            time_s,
            sampling_rate_hz,
        )
    )
    labels.update(_manual_labels_at_time(context.manual_truth, time_s))
    if context.quality_intervals:
        annotator_values: list[str | None] = []
        for key in ("annotator_1", "annotator_2", "annotator_3", "consensus"):
            match = _interval_at_time(context.quality_intervals[key], time_s)
            value = str(match["label_original"]) if match is not None else None
            labels[f"label_quality_{key}_original"] = value
            annotator_values.append(value)
            if key == "consensus" and match is not None:
                labels["label_quality_consensus_text"] = QUALITY_CLASS_TEXT.get(value)
                labels["label_quality_interval_start_s"] = float(match["start_s"])
                labels["label_quality_interval_end_s"] = float(match["end_s"])
                labels["label_quality_trailing_10s_pure"] = bool(
                    time_s - QUALITY_LOOKBACK_S >= float(match["start_s"])
                    and time_s <= float(match["end_s"])
                )
        consensus = annotator_values[-1]
        labels["label_quality_annotator_agreement_count"] = (
            sum(value == consensus for value in annotator_values[:3] if value is not None)
            if consensus is not None
            else None
        )

    if context.seizure_intervals:
        labels.update(_seizure_labels_at_time(context.seizure_intervals, time_s))
    if spec.dataset_key == "nstdb":
        noise_active, interval_index = _nstdb_noise_state(time_s)
        labels["label_noise_type_original"] = "em"
        labels["label_noise_condition_original"] = context.noise_condition
        labels["label_noise_snr_db_original"] = context.noise_snr_db
        labels["label_noise_active"] = noise_active
        labels["label_noise_interval_index"] = interval_index

    label_values = [
        labels.get("label_beat_symbol_original"),
        labels.get("label_quality_consensus_original"),
        labels.get("label_seizure_binary"),
        labels.get("label_ludb_diagnoses_original_json"),
        labels.get("label_manual_annotation_source_original"),
        labels.get("label_noise_condition_original"),
    ]
    labels["any_reference_label_available"] = any(
        value is not None and value != "" for value in label_values
    )
    return labels


def _beat_labels_at_time(
    samples: np.ndarray,
    symbols: np.ndarray,
    time_s: float,
    sampling_rate_hz: float,
) -> dict[str, Any]:
    output = {
        "label_beat_symbol_original": None,
        "label_beat_aami_superclass_derived": None,
        "label_beat_reference_sample": None,
        "label_beat_match_offset_ms": None,
        "label_beat_match_within_75ms": False,
        "label_beat_context5_original_json": None,
        "label_beat_context5_aami_derived_json": None,
        "label_beat_context5_homogeneous": None,
        "label_beat_context5_contains_non_n": None,
    }
    if samples.size == 0:
        return output
    target_sample = int(np.rint(time_s * sampling_rate_hz))
    index = _nearest_index(samples, target_sample)
    offset_ms = 1000.0 * (target_sample - int(samples[index])) / sampling_rate_hz
    output["label_beat_match_offset_ms"] = float(offset_ms)
    output["label_beat_match_within_75ms"] = bool(
        abs(offset_ms) <= BEAT_MATCH_TOLERANCE_MS
    )
    if not output["label_beat_match_within_75ms"]:
        return output
    symbol = str(symbols[index])
    first = max(0, index - 4)
    context = [str(value) for value in symbols[first : index + 1].tolist()]
    aami = [AAMI_BEAT_CLASS.get(value) for value in context]
    output.update(
        {
            "label_beat_symbol_original": symbol,
            "label_beat_aami_superclass_derived": AAMI_BEAT_CLASS.get(symbol),
            "label_beat_reference_sample": int(samples[index]),
            "label_beat_context5_original_json": json.dumps(context),
            "label_beat_context5_aami_derived_json": json.dumps(aami),
            "label_beat_context5_homogeneous": bool(
                len(context) == 5 and len(set(context)) == 1
            ),
            "label_beat_context5_contains_non_n": bool(
                any(value != "N" for value in context)
            ),
        }
    )
    return output


def _manual_truth_table(record: ManualDelineationRecord) -> pd.DataFrame:
    by_landmark: dict[str, dict[int, int]] = {}
    for landmark in LANDMARK_NAMES:
        values = record.landmarks[landmark]
        indices = associate_landmarks_with_anchors(
            values,
            record.anchor_samples,
            landmark=landmark,
            sampling_rate_hz=record.sampling_rate_hz,
        )
        by_landmark[landmark] = {
            int(anchor_index): int(sample)
            for sample, anchor_index in zip(values.tolist(), indices.tolist(), strict=True)
            if anchor_index >= 0
        }
    rows: list[dict[str, Any]] = []
    for anchor_index in record.evaluation_anchor_indices.tolist():
        anchor = int(record.anchor_samples[anchor_index])
        row: dict[str, Any] = {
            "anchor_index": int(anchor_index),
            "anchor_sample": anchor,
            "anchor_time_s": float(anchor / record.sampling_rate_hz),
            "annotation_source": record.annotation_source,
        }
        for landmark in LANDMARK_NAMES:
            row[f"{landmark}_sample"] = by_landmark[landmark].get(anchor_index)
        row["r_peak_sample"] = anchor
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.attrs["sampling_rate_hz"] = float(record.sampling_rate_hz)
    return frame


def _manual_labels_at_time(frame: pd.DataFrame, time_s: float) -> dict[str, Any]:
    output = {
        "label_manual_annotation_source_original": None,
        "label_manual_reference_match_offset_ms": None,
        "label_manual_reference_anchor_sample": None,
        "label_manual_p_onset_sample": None,
        "label_manual_qrs_onset_sample": None,
        "label_manual_r_peak_sample": None,
        "label_manual_qrs_offset_sample": None,
        "label_manual_t_peak_sample": None,
        "label_manual_t_offset_sample": None,
        "label_manual_pr_interval_ms": None,
        "label_manual_qrs_duration_ms": None,
        "label_manual_qt_interval_ms": None,
        "label_manual_jt_interval_ms": None,
        "label_manual_t_peak_to_end_ms": None,
    }
    if frame.empty:
        return output
    times = frame["anchor_time_s"].to_numpy(dtype=float)
    index = _nearest_index(times, time_s)
    row = frame.iloc[index]
    offset_ms = 1000.0 * (time_s - float(row.anchor_time_s))
    if abs(offset_ms) > BEAT_MATCH_TOLERANCE_MS:
        output["label_manual_reference_match_offset_ms"] = float(offset_ms)
        return output
    output.update(
        {
            "label_manual_annotation_source_original": str(row.annotation_source),
            "label_manual_reference_match_offset_ms": float(offset_ms),
            "label_manual_reference_anchor_sample": int(row.anchor_sample),
        }
    )
    sample_map = {
        "p_onset": "label_manual_p_onset_sample",
        "qrs_onset": "label_manual_qrs_onset_sample",
        "r_peak": "label_manual_r_peak_sample",
        "qrs_offset": "label_manual_qrs_offset_sample",
        "t_peak": "label_manual_t_peak_sample",
        "t_offset": "label_manual_t_offset_sample",
    }
    for source, target in sample_map.items():
        value = row.get(f"{source}_sample")
        output[target] = int(value) if pd.notna(value) else None
    fs = _sampling_rate_from_manual_row(frame)
    p_onset = output["label_manual_p_onset_sample"]
    qrs_onset = output["label_manual_qrs_onset_sample"]
    qrs_offset = output["label_manual_qrs_offset_sample"]
    t_peak = output["label_manual_t_peak_sample"]
    t_offset = output["label_manual_t_offset_sample"]
    if fs is not None:
        output["label_manual_pr_interval_ms"] = _sample_interval_ms(
            p_onset, qrs_onset, fs
        )
        output["label_manual_qrs_duration_ms"] = _sample_interval_ms(
            qrs_onset, qrs_offset, fs
        )
        output["label_manual_qt_interval_ms"] = _sample_interval_ms(
            qrs_onset, t_offset, fs
        )
        output["label_manual_jt_interval_ms"] = _sample_interval_ms(
            qrs_offset, t_offset, fs
        )
        output["label_manual_t_peak_to_end_ms"] = _sample_interval_ms(
            t_peak, t_offset, fs
        )
    return output


def _sampling_rate_from_manual_row(frame: pd.DataFrame) -> float | None:
    value = frame.attrs.get("sampling_rate_hz")
    return float(value) if value is not None else None


def _seizure_labels_at_time(
    intervals: Sequence[tuple[float, float]], time_s: float
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "label_seizure_binary": "non_ictal",
        "label_seizure_phase_derived": "interictal",
        "label_seizure_interval_id": None,
        "label_seizure_interval_original_json": None,
        "label_seconds_from_seizure_onset": None,
        "label_seconds_from_seizure_offset": None,
        "label_seizure_phase_rule": (
            "ictal=inside configured interval; pre/post=causal +/-300 s; otherwise interictal"
        ),
    }
    best_distance = np.inf
    for index, (onset, offset) in enumerate(intervals, start=1):
        if onset <= time_s <= offset:
            output.update(
                {
                    "label_seizure_binary": "ictal",
                    "label_seizure_phase_derived": "ictal",
                    "label_seizure_interval_id": index,
                    "label_seizure_interval_original_json": json.dumps(
                        [onset, offset]
                    ),
                    "label_seconds_from_seizure_onset": float(time_s - onset),
                    "label_seconds_from_seizure_offset": float(time_s - offset),
                }
            )
            return output
        distance = min(abs(time_s - onset), abs(time_s - offset))
        if distance < best_distance:
            best_distance = distance
            output["label_seizure_interval_id"] = index
            output["label_seizure_interval_original_json"] = json.dumps(
                [onset, offset]
            )
            output["label_seconds_from_seizure_onset"] = float(time_s - onset)
            output["label_seconds_from_seizure_offset"] = float(time_s - offset)
            if onset - SEIZURE_CONTEXT_S <= time_s < onset:
                output["label_seizure_phase_derived"] = "preictal_5min"
            elif offset < time_s <= offset + SEIZURE_CONTEXT_S:
                output["label_seizure_phase_derived"] = "postictal_5min"
            else:
                output["label_seizure_phase_derived"] = "interictal"
    return output


def _apply_binary_targets(observations: pd.DataFrame) -> pd.DataFrame:
    """Add explicit 0/1 research targets without replacing source labels."""

    frame = observations.copy()
    binary_columns = [
        "target_artifact_binary",
        "target_seizure_binary",
        "target_abnormal_beat_binary",
        "target_noise_active_binary",
        "target_signal_unusable_binary",
    ]
    for column in binary_columns:
        frame[column] = pd.Series(pd.NA, index=frame.index, dtype="Int8")
    frame["target_artifact_source_derived"] = pd.Series(
        pd.NA, index=frame.index, dtype="string"
    )

    pure_quality = frame["label_quality_trailing_10s_pure"].eq(True)
    quality = frame["label_quality_consensus_original"].astype("string")
    quality_known = pure_quality & quality.isin(["1", "2", "3"])
    frame.loc[quality_known, "target_artifact_binary"] = (
        quality.loc[quality_known].isin(["2", "3"]).astype("int8")
    )
    frame.loc[quality_known, "target_artifact_source_derived"] = (
        "BUT-QDB consensus: class 1=0; class 2/3=1"
    )

    noise_known = frame["label_noise_active"].notna()
    frame.loc[noise_known, "target_noise_active_binary"] = (
        frame.loc[noise_known, "label_noise_active"].astype(bool).astype("int8")
    )
    artifact_noise = noise_known & frame["target_artifact_binary"].isna()
    frame.loc[artifact_noise, "target_artifact_binary"] = frame.loc[
        artifact_noise, "target_noise_active_binary"
    ]
    frame.loc[artifact_noise, "target_artifact_source_derived"] = (
        "NSTDB official schedule: noise inactive=0; active=1"
    )

    seizure_known = frame["label_seizure_binary"].notna()
    frame.loc[seizure_known, "target_seizure_binary"] = (
        frame.loc[seizure_known, "label_seizure_binary"]
        .eq("ictal")
        .astype("int8")
    )

    beat_known = frame["label_beat_aami_superclass_derived"].notna()
    frame.loc[beat_known, "target_abnormal_beat_binary"] = (
        frame.loc[beat_known, "label_beat_aami_superclass_derived"]
        .ne("N")
        .astype("int8")
    )

    strict_quality = pure_quality & quality.isin(["1", "3"])
    frame.loc[strict_quality, "target_signal_unusable_binary"] = (
        quality.loc[strict_quality].eq("3").astype("int8")
    )
    return frame


def stable_group_fold(group_id: str, *, folds: int = 5) -> int:
    """Assign a stable group-wise fold without using any labels."""

    if folds < 2:
        raise ValueError("folds must be at least 2")
    digest = hashlib.sha256(group_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % folds


def _write_dataset_artifacts(
    observations: pd.DataFrame,
    execution: pd.DataFrame,
    output: Path,
    *,
    profile: str,
) -> dict[str, Path]:
    keys = feature_keys()
    provenance = [column for column in _provenance_columns() if column in observations]
    labels = [column for column in _label_columns() if column in observations]
    qc = [column for column in _quality_control_columns() if column in observations]
    artifacts = {
        "observations": output / "observations_with_labels.csv",
        "features": output / "X_features.csv",
        "labels": output / "y_labels.csv",
        "provenance": output / "row_provenance_and_qc.csv",
        "features_dictionary": output / "feature_dictionary.csv",
        "labels_dictionary": output / "label_dictionary.csv",
        "branch_combinations": output / "branch_combinations.csv",
        "execution_manifest": output / "segment_execution_manifest.csv",
    }
    observations.to_csv(artifacts["observations"], index=False)
    observations[["row_id", *keys]].to_csv(artifacts["features"], index=False)
    observations[["row_id", *labels]].to_csv(artifacts["labels"], index=False)
    observations[["row_id", *provenance, *qc]].to_csv(
        artifacts["provenance"], index=False
    )
    _feature_dictionary().to_csv(artifacts["features_dictionary"], index=False)
    _label_dictionary().to_csv(artifacts["labels_dictionary"], index=False)
    pd.DataFrame(branch_combinations()).to_csv(
        artifacts["branch_combinations"], index=False
    )
    execution.to_csv(artifacts["execution_manifest"], index=False)

    tasks_dir = output / "model_tasks"
    tasks_dir.mkdir(exist_ok=True)
    task_specs = {
        "beat_symbol": {
            "filter": observations["label_beat_symbol_original"].notna(),
            "targets": [
                "label_beat_symbol_original",
                "label_beat_aami_superclass_derived",
                "label_beat_context5_original_json",
            ],
        },
        "beat_aami": {
            "filter": observations["label_beat_aami_superclass_derived"].notna(),
            "targets": ["label_beat_aami_superclass_derived"],
        },
        "signal_quality": {
            "filter": observations["label_quality_consensus_original"].isin(
                ["1", "2", "3"]
            )
            & observations["label_quality_trailing_10s_pure"].eq(True),
            "targets": [
                "label_quality_consensus_original",
                "label_quality_consensus_text",
            ],
        },
        "noise_snr": {
            "filter": observations["label_noise_active"].eq(True),
            "targets": [
                "label_noise_condition_original",
                "label_noise_snr_db_original",
            ],
        },
        "seizure": {
            "filter": observations["label_seizure_binary"].notna(),
            "targets": [
                "label_seizure_binary",
                "label_seizure_phase_derived",
                "label_seconds_from_seizure_onset",
            ],
        },
        "ludb_diagnosis_multilabel": {
            "filter": observations["label_ludb_diagnoses_original_json"].notna(),
            "targets": [
                "label_ludb_diagnoses_original_json",
                "label_ludb_rhythm_original",
            ],
        },
        "manual_interval_regression": {
            "filter": observations["label_manual_annotation_source_original"].notna(),
            "targets": [
                "label_manual_pr_interval_ms",
                "label_manual_qrs_duration_ms",
                "label_manual_qt_interval_ms",
                "label_manual_jt_interval_ms",
                "label_manual_t_peak_to_end_ms",
            ],
        },
        "artifact_binary": {
            "filter": observations["target_artifact_binary"].notna(),
            "targets": [
                "target_artifact_binary",
                "target_artifact_source_derived",
            ],
        },
        "seizure_binary": {
            "filter": observations["target_seizure_binary"].notna(),
            "targets": ["target_seizure_binary"],
        },
        "abnormal_beat_binary": {
            "filter": observations["target_abnormal_beat_binary"].notna(),
            "targets": ["target_abnormal_beat_binary"],
        },
        "noise_active_binary": {
            "filter": observations["target_noise_active_binary"].notna(),
            "targets": ["target_noise_active_binary"],
        },
        "signal_unusable_binary": {
            "filter": observations["target_signal_unusable_binary"].notna(),
            "targets": ["target_signal_unusable_binary"],
        },
    }
    task_manifest: dict[str, Any] = {}
    for task, definition in task_specs.items():
        path = tasks_dir / f"{task}.csv"
        columns = [
            "row_id",
            "lineage_group_id",
            "suggested_cv_fold",
            *keys,
            *definition["targets"],
        ]
        selected = observations.loc[definition["filter"], columns]
        selected.to_csv(path, index=False)
        task_manifest[task] = {
            "path": str(path.relative_to(output)).replace("\\", "/"),
            "rows": int(selected.shape[0]),
            "targets": definition["targets"],
            "group_column": "lineage_group_id",
            "fold_column": "suggested_cv_fold",
        }
    model_manifest = {
        "builder_version": BUILDER_VERSION,
        "profile": profile,
        "feature_file": "X_features.csv",
        "label_file": "y_labels.csv",
        "join_key": "row_id",
        "feature_columns": keys,
        "branch_feature_sets": branch_combinations(),
        "tasks": task_manifest,
        "warning": (
            "Never random-split rows. Split by lineage_group_id so repeated windows, "
            "noise copies, same-subject recordings, and derived QTDB excerpts stay together."
        ),
    }
    model_manifest_path = output / "model_manifest.json"
    model_manifest_path.write_text(
        json.dumps(model_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    artifacts["model_manifest"] = model_manifest_path
    return artifacts


def _validate_outputs(
    observations: pd.DataFrame,
    execution: pd.DataFrame,
    artifacts: Mapping[str, Path],
    *,
    profile: str,
) -> dict[str, Any]:
    keys = feature_keys()
    problems: list[str] = []
    if observations["row_id"].duplicated().any():
        problems.append("row_id is not unique")
    if any(column not in observations for column in keys):
        problems.append("one or more feature columns are missing")
    numeric = observations[keys].apply(pd.to_numeric, errors="coerce")
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        problems.append("feature matrix contains infinite values")
    feature_file_columns = pd.read_csv(artifacts["features"], nrows=0).columns.tolist()
    if feature_file_columns != ["row_id", *keys]:
        problems.append("X_features.csv contains labels/provenance or wrong feature order")
    label_file_columns = pd.read_csv(artifacts["labels"], nrows=0).columns.tolist()
    if set(keys).intersection(label_file_columns):
        problems.append("y_labels.csv contains model feature columns")
    if len(branch_combinations()) != 15:
        problems.append("branch combination manifest does not contain 15 ablations")

    label_counts = {
        "beat_symbol_original": _value_counts(
            observations["label_beat_symbol_original"]
        ),
        "beat_aami_derived": _value_counts(
            observations["label_beat_aami_superclass_derived"]
        ),
        "quality_consensus_original": _value_counts(
            observations["label_quality_consensus_original"]
        ),
        "seizure_binary": _value_counts(observations["label_seizure_binary"]),
        "seizure_phase_derived": _value_counts(
            observations["label_seizure_phase_derived"]
        ),
        "noise_condition_original": _value_counts(
            observations["label_noise_condition_original"]
        ),
        "target_artifact_binary": _value_counts(
            observations["target_artifact_binary"]
        ),
        "target_seizure_binary": _value_counts(
            observations["target_seizure_binary"]
        ),
        "target_abnormal_beat_binary": _value_counts(
            observations["target_abnormal_beat_binary"]
        ),
        "target_noise_active_binary": _value_counts(
            observations["target_noise_active_binary"]
        ),
        "target_signal_unusable_binary": _value_counts(
            observations["target_signal_unusable_binary"]
        ),
    }
    datasets_present = sorted(observations["dataset_key"].dropna().unique().tolist())
    planned = sorted(execution["dataset_key"].dropna().unique().tolist())
    failed = execution.loc[execution["execution_status"] == "failed"]
    planned_records = {
        (str(dataset_key), str(record_id))
        for dataset_key, record_id in execution[["dataset_key", "record_id"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    successful_records = {
        (str(dataset_key), str(record_id))
        for dataset_key, record_id in observations[["dataset_key", "record_id"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    missing_records = sorted(planned_records.difference(successful_records))
    if failed.shape[0]:
        problems.append(f"{int(failed.shape[0])} planned segment(s) failed extraction")
    if missing_records:
        preview = ", ".join(
            f"{dataset}/{record}" for dataset, record in missing_records[:10]
        )
        problems.append(
            f"{len(missing_records)} planned record(s) produced no observations: {preview}"
        )
    lineage_fold_counts = observations.groupby("lineage_group_id")[
        "suggested_cv_fold"
    ].nunique()
    crossing_lineages = lineage_fold_counts.loc[lineage_fold_counts > 1].index.tolist()
    if crossing_lineages:
        problems.append("one or more lineage groups cross suggested CV folds")
    return {
        "builder_version": BUILDER_VERSION,
        "profile": profile,
        "validation_passed": not problems,
        "problems": problems,
        "observation_rows": int(observations.shape[0]),
        "feature_columns": len(keys),
        "label_columns": len(_label_columns()),
        "branch_combinations": len(branch_combinations()),
        "planned_segments": int(execution.shape[0]),
        "failed_segments": int(failed.shape[0]),
        "planned_records": len(planned_records),
        "missing_records": [
            {"dataset_key": dataset_key, "record_id": record_id}
            for dataset_key, record_id in missing_records
        ],
        "lineage_groups_crossing_folds": len(crossing_lineages),
        "datasets_planned": planned,
        "datasets_present": datasets_present,
        "records_present_by_dataset": {
            str(key): int(group["record_id"].nunique())
            for key, group in observations.groupby("dataset_key", sort=True)
        },
        "rows_by_dataset": {
            str(key): int(value)
            for key, value in observations["dataset_key"].value_counts().sort_index().items()
        },
        "label_counts": label_counts,
        "available_feature_count": {
            "minimum": int(observations["available_feature_count"].min())
            if not observations.empty
            else 0,
            "median": float(observations["available_feature_count"].median())
            if not observations.empty
            else 0.0,
            "maximum": int(observations["available_feature_count"].max())
            if not observations.empty
            else 0,
        },
        "artifacts": {
            key: str(path) for key, path in artifacts.items()
        },
    }


def _feature_dictionary() -> pd.DataFrame:
    branch_map = branch_feature_map()
    branch_by_key = {
        key: branch for branch, columns in branch_map.items() for key in columns
    }
    rows = []
    for position, definition in enumerate(feature_definitions(), start=1):
        key = str(definition["key"])
        rows.append(
            {
                "position": position,
                "feature_key": key,
                "display_label": definition["label"],
                "branch_code": branch_by_key[key],
                "branch_name": BRANCH_NAMES[branch_by_key[key]],
                "feature_family": definition["branch"],
                "unit": definition["unit"],
                "role": definition["role"],
                "description": definition["meaning"],
                "model_input": True,
            }
        )
    return pd.DataFrame(rows)


def _label_dictionary() -> pd.DataFrame:
    definitions = {
        "label_beat_symbol_original": ("original", "WFDB atr", "Exact expert beat symbol; never overwritten."),
        "label_beat_aami_superclass_derived": ("derived", "AAMI grouping", "Separate N/S/V/F/Q superclass derived from the exact symbol."),
        "label_beat_reference_sample": ("original reference", "WFDB atr", "Expert annotation sample matched to the pipeline row anchor."),
        "label_beat_match_offset_ms": ("derived alignment QC", "row anchor vs WFDB atr", "Signed pipeline-anchor minus expert-sample offset in milliseconds."),
        "label_beat_match_within_75ms": ("derived alignment QC", "75 ms tolerance", "True when the nearest expert beat is within the frozen matching tolerance."),
        "label_beat_context5_original_json": ("original sequence", "WFDB atr", "Up to five exact expert symbols ending at the matched row anchor."),
        "label_beat_context5_aami_derived_json": ("derived sequence", "AAMI grouping", "AAMI superclass sequence corresponding to the five exact symbols."),
        "label_beat_context5_homogeneous": ("derived context", "five-beat sequence", "True when all available exact symbols in the context are identical."),
        "label_beat_context5_contains_non_n": ("derived context", "five-beat sequence", "True when the context contains an AAMI class other than N."),
        "label_quality_annotator_1_original": ("original", "BUT-QDB ANN columns 1-3", "Annotator 1 class at the row time."),
        "label_quality_annotator_2_original": ("original", "BUT-QDB ANN columns 4-6", "Annotator 2 class at the row time."),
        "label_quality_annotator_3_original": ("original", "BUT-QDB ANN columns 7-9", "Annotator 3 class at the row time."),
        "label_quality_consensus_original": ("original", "BUT-QDB ANN columns 10-12", "Consensus class 1/2/3; class 0 means not annotated."),
        "label_quality_consensus_text": ("source-defined display", "BUT-QDB class definitions", "Human-readable meaning of the exact consensus class."),
        "label_quality_interval_start_s": ("original boundary", "BUT-QDB consensus interval", "Start time of the consensus interval containing the row."),
        "label_quality_interval_end_s": ("original boundary", "BUT-QDB consensus interval", "End time of the consensus interval containing the row."),
        "label_quality_trailing_10s_pure": ("derived alignment QC", "consensus interval + 10 s lookback", "True when the complete causal quality window lies inside one consensus interval."),
        "label_quality_annotator_agreement_count": ("derived agreement", "three BUT-QDB annotators", "Number of individual annotators agreeing with consensus at the row time."),
        "label_seizure_binary": ("derived", "interval membership", "ictal only inside the original interval; otherwise non_ictal."),
        "label_seizure_phase_derived": ("derived", "interval +/- 300 s", "Exploratory ictal/preictal/postictal/interictal phase."),
        "label_seizure_interval_id": ("derived identifier", "datasets.toml interval order", "Stable identifier of the nearest or containing configured seizure interval."),
        "label_seizure_interval_original_json": ("original interval", "datasets.toml", "Configured expert seizure onset/offset pair in seconds."),
        "label_seconds_from_seizure_onset": ("derived distance", "configured seizure onset", "Signed seconds from the applicable seizure onset."),
        "label_seconds_from_seizure_offset": ("derived distance", "configured seizure offset", "Signed seconds from the applicable seizure offset."),
        "label_seizure_phase_rule": ("derived provenance", "frozen phase mapping", "Text form of the rule used to derive the exploratory phase."),
        "label_noise_type_original": ("original experimental condition", "NSTDB generator", "Exact noise type token; em denotes electrode-motion noise."),
        "label_noise_condition_original": ("original experimental condition", "NSTDB record name", "Official electrode-motion SNR condition."),
        "label_noise_snr_db_original": ("original experimental value", "NSTDB record name", "Numeric signal-to-noise ratio in decibels."),
        "label_noise_active": ("derived", "NSTDB official schedule", "True only in alternating official two-minute noisy intervals."),
        "label_noise_interval_index": ("derived schedule index", "NSTDB official schedule", "Index of the active official noisy interval, when applicable."),
        "label_ludb_diagnoses_original_json": ("original", "LUDB WFDB header comments", "Ordered diagnosis strings with source wording and punctuation preserved."),
        "label_ludb_rhythm_original": ("original", "LUDB WFDB header comments", "Exact Rhythm diagnosis line."),
        "label_record_comments_original_json": ("original", "WFDB header comments", "Ordered source record comments without normalization."),
        "label_manual_annotation_source_original": ("original reference", "LUDB lead annotation or QTDB q1c", "Manual landmark source attached to the matched anchor."),
        "label_manual_reference_match_offset_ms": ("derived alignment QC", "row anchor vs manual anchor", "Signed row-anchor minus manual-reference offset in milliseconds."),
        "label_manual_reference_anchor_sample": ("original reference", "manual delineation", "Manual reference anchor sample paired to the row."),
        "label_manual_p_onset_sample": ("original landmark", "manual delineation", "Manual P-wave onset sample."),
        "label_manual_qrs_onset_sample": ("original landmark", "manual delineation", "Manual QRS onset sample."),
        "label_manual_r_peak_sample": ("original landmark", "manual delineation", "Manual R-peak sample."),
        "label_manual_qrs_offset_sample": ("original landmark", "manual delineation", "Manual QRS offset sample."),
        "label_manual_t_peak_sample": ("original landmark", "manual delineation", "Manual T-wave peak sample."),
        "label_manual_t_offset_sample": ("original landmark", "manual delineation", "Manual T-wave offset sample."),
        "label_manual_pr_interval_ms": ("derived reference measurement", "manual landmarks", "P onset to QRS onset; label-side regression truth, never a model input."),
        "label_manual_qrs_duration_ms": ("derived reference measurement", "manual landmarks", "QRS onset to QRS offset."),
        "label_manual_qt_interval_ms": ("derived reference measurement", "manual landmarks", "QRS onset to T offset."),
        "label_manual_jt_interval_ms": ("derived reference measurement", "manual landmarks", "QRS offset to T offset."),
        "label_manual_t_peak_to_end_ms": ("derived reference measurement", "manual landmarks", "T peak to T offset."),
        "source_age_original": ("original metadata", "source demographics/header", "Source age value as published, without normalization."),
        "source_sex_original": ("original metadata", "source demographics/header", "Source sex value as published, without normalization."),
        "any_reference_label_available": ("derived availability QC", "all label namespaces", "True when at least one primary reference label is available for the row."),
        "target_artifact_binary": ("derived binary target", "BUT-QDB consensus or NSTDB schedule", "Integer 0/1 artifact target. BUT-QDB uses pure class 1=0 and pure class 2/3=1; NSTDB uses noise inactive=0 and active=1."),
        "target_artifact_source_derived": ("derived target provenance", "binary artifact mapping", "Exact mapping source used to create target_artifact_binary for this row."),
        "target_seizure_binary": ("derived binary target", "configured seizure intervals", "Integer 1 inside an original seizure interval and 0 outside it."),
        "target_abnormal_beat_binary": ("derived binary target", "AAMI beat superclass", "Integer 0 for AAMI N and 1 for AAMI S/V/F/Q; the exact WFDB symbol remains separate."),
        "target_noise_active_binary": ("derived binary target", "NSTDB official schedule", "Integer 1 during an official noisy interval and 0 during the clean interval."),
        "target_signal_unusable_binary": ("derived binary target", "BUT-QDB strict consensus", "Integer 0 for pure consensus class 1 and 1 for pure consensus class 3; class 2 is deliberately excluded."),
    }
    missing = set(_label_columns()).difference(definitions)
    extra = set(definitions).difference(_label_columns())
    if missing or extra:
        raise RuntimeError(
            f"Label dictionary mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    rows = [
        (column, *definitions[column])
        for column in _label_columns()
    ]
    return pd.DataFrame(
        rows, columns=["label_column", "status", "source", "definition"]
    )


def _dataset_readme(report: Mapping[str, Any]) -> str:
    return f"""# Comprehensive ECG branch-validation dataset

This dataset contains **{report['observation_rows']:,} observations**, the maintained
**52 numeric feature columns**, and all **15 non-empty B1-B4 feature-set
combinations**.  It is a research/engineering validation artifact, not a
clinically validated seizure detector.

## Use the files safely

- `X_features.csv` is the label-free model input matrix.  It contains only
  `row_id` plus the 52 numeric features.
- `y_labels.csv` contains source labels and derived label views keyed by
  `row_id`.  Original labels always have `_original` in their names; derived
  labels are explicitly named `_derived` or document their mapping rule.
- `observations_with_labels.csv` is the audit join of features, provenance,
  quality-control flags, and labels.  Do not pass all of its numeric columns to
  a model because doing so would leak record identity, time, manual references,
  and labels.
- `model_tasks/` provides task-specific joined views.  The target and grouping
  columns are named in `model_manifest.json`.
- `branch_combinations.csv` lists the exact feature columns for B1, B2, B3,
  B4, and every combination.
- `feature_dictionary.csv` and `label_dictionary.csv` define every feature and
  label family.

## Split rule

Never random-split rows.  Split on `lineage_group_id` (or use
`suggested_cv_fold`).  This keeps overlapping windows, same-patient recordings,
NSTDB copies of MIT-BIH 118/119, and QTDB excerpts derived from another source
record in the same fold.

## Label semantics

The databases do not share one universal target.  MIT-BIH supplies beat
symbols; BUT-QDB supplies three-expert quality classes and consensus; NSTDB
supplies controlled noise/SNR; LUDB supplies diagnoses and manual landmarks;
QTDB supplies manual waveform boundaries; CHB-MIT/Siena configurations supply
seizure intervals.  Train and score these as separate tasks.  Use clustering
on `X_features.csv`, then join labels only for post-hoc interpretation.

## Normalized binary targets

Five nullable integer targets are provided for direct binary experiments:
`target_artifact_binary`, `target_seizure_binary`,
`target_abnormal_beat_binary`, `target_noise_active_binary`, and
`target_signal_unusable_binary`.  Their exact mappings are recorded in
`label_dictionary.csv`.  A blank value means that source database does not
provide the required truth; it never means class 0.

## Missingness

Missing values are intentional.  Ten-second LUDB records cannot define a
100-RR HRV window, manual landmarks do not exist in seizure EDFs, and current
conduction measurements remain information-only.  No missing value is filled
with zero.

Validation status: **{'PASS' if report['validation_passed'] else 'FAIL'}**.
Failed extraction segments: **{report['failed_segments']}**.  See
`validation_report.json` and `segment_execution_manifest.csv` for details.
"""


def _observation_columns() -> list[str]:
    return [
        *_provenance_columns(),
        *_quality_control_columns(),
        *feature_keys(),
        *_label_columns(),
    ]


def _provenance_columns() -> list[str]:
    return [
        "row_id",
        "segment_id",
        "dataset_key",
        "dataset_name",
        "record_id",
        "subject_id",
        "lineage_group_id",
        "suggested_cv_fold",
        "source_path",
        "source_kind",
        "channel",
        "sampling_rate_hz",
        "input_unit",
        "segment_start_s",
        "segment_end_s",
        "observation_time_s",
        "timestamp_track",
        "calibration_beats_requested",
        "selection_reason",
    ]


def _quality_control_columns() -> list[str]:
    columns = [
        "measurements_defined",
        "core_context_pass",
        "morphology_defined",
        "hrv_defined",
        "hrv_reliable",
        "prsa_bprsa_defined",
        "prsa_bprsa_reliable",
        "signal_quality_context_available",
        "conduction_information_available",
        "artifact_label_produced_by_pipeline",
        "conduction_final_matrix_eligible",
    ]
    for branch in BRANCH_NAMES:
        lower = branch.lower()
        columns.extend(
            [
                f"{lower}_available_feature_count",
                f"{lower}_feature_count",
                f"{lower}_all_features_available",
            ]
        )
    columns.extend(["available_feature_count", "missing_feature_count"])
    return columns


def _label_columns() -> list[str]:
    return [
        "label_beat_symbol_original",
        "label_beat_aami_superclass_derived",
        "label_beat_reference_sample",
        "label_beat_match_offset_ms",
        "label_beat_match_within_75ms",
        "label_beat_context5_original_json",
        "label_beat_context5_aami_derived_json",
        "label_beat_context5_homogeneous",
        "label_beat_context5_contains_non_n",
        "label_quality_annotator_1_original",
        "label_quality_annotator_2_original",
        "label_quality_annotator_3_original",
        "label_quality_consensus_original",
        "label_quality_consensus_text",
        "label_quality_interval_start_s",
        "label_quality_interval_end_s",
        "label_quality_trailing_10s_pure",
        "label_quality_annotator_agreement_count",
        "label_seizure_binary",
        "label_seizure_phase_derived",
        "label_seizure_interval_id",
        "label_seizure_interval_original_json",
        "label_seconds_from_seizure_onset",
        "label_seconds_from_seizure_offset",
        "label_seizure_phase_rule",
        "label_noise_type_original",
        "label_noise_condition_original",
        "label_noise_snr_db_original",
        "label_noise_active",
        "label_noise_interval_index",
        "label_ludb_diagnoses_original_json",
        "label_ludb_rhythm_original",
        "label_record_comments_original_json",
        "label_manual_annotation_source_original",
        "label_manual_reference_match_offset_ms",
        "label_manual_reference_anchor_sample",
        "label_manual_p_onset_sample",
        "label_manual_qrs_onset_sample",
        "label_manual_r_peak_sample",
        "label_manual_qrs_offset_sample",
        "label_manual_t_peak_sample",
        "label_manual_t_offset_sample",
        "label_manual_pr_interval_ms",
        "label_manual_qrs_duration_ms",
        "label_manual_qt_interval_ms",
        "label_manual_jt_interval_ms",
        "label_manual_t_peak_to_end_ms",
        "source_age_original",
        "source_sex_original",
        "any_reference_label_available",
        "target_artifact_binary",
        "target_artifact_source_derived",
        "target_seizure_binary",
        "target_abnormal_beat_binary",
        "target_noise_active_binary",
        "target_signal_unusable_binary",
    ]


def _segment_manifest_row(spec: SegmentSpec) -> dict[str, Any]:
    record = asdict(spec)
    record["segment_id"] = spec.segment_id
    record["end_s"] = spec.end_s
    record["metadata_json"] = json.dumps(record.pop("metadata"), ensure_ascii=False)
    return record


def _empty_observations() -> pd.DataFrame:
    return pd.DataFrame(columns=_observation_columns())


def _read_records(dataset: Path) -> list[str]:
    path = dataset / "RECORDS"
    if not path.is_file():
        raise FileNotFoundError(path)
    return [
        value.strip()
        for value in path.read_text(encoding="utf-8").splitlines()
        if value.strip()
    ]


def _best_label_diversity_start(
    samples: np.ndarray,
    symbols: np.ndarray,
    *,
    sampling_rate_hz: float,
    record_duration_s: float,
    segment_duration_s: float,
    global_counts: Mapping[str, int],
) -> float:
    if record_duration_s <= segment_duration_s:
        return 0.0
    bin_count = max(1, int(np.ceil(record_duration_s / segment_duration_s)))
    scores = np.zeros(bin_count, dtype=float)
    unique_by_bin: list[set[str]] = [set() for _ in range(bin_count)]
    for sample, raw_symbol in zip(samples.tolist(), symbols.tolist(), strict=True):
        time_s = float(sample / sampling_rate_hz)
        index = min(bin_count - 1, int(time_s // segment_duration_s))
        symbol = str(raw_symbol)
        scores[index] += 1.0 / np.sqrt(max(1, int(global_counts.get(symbol, 1))))
        unique_by_bin[index].add(symbol)
    for index, symbols_in_bin in enumerate(unique_by_bin):
        scores[index] += 2.0 * sum(
            1.0 / np.sqrt(max(1, int(global_counts.get(symbol, 1))))
            for symbol in symbols_in_bin
        )
    midpoint = record_duration_s / 2.0
    ranking = sorted(
        range(bin_count),
        key=lambda index: (
            -scores[index],
            abs((index + 0.5) * segment_duration_s - midpoint),
            index,
        ),
    )
    return min(
        ranking[0] * segment_duration_s,
        max(0.0, record_duration_s - segment_duration_s),
    )


def _read_butqdb_annotations(path: Path) -> dict[str, pd.DataFrame]:
    raw = pd.read_csv(path, header=None)
    names = ("annotator_1", "annotator_2", "annotator_3", "consensus")
    output: dict[str, pd.DataFrame] = {}
    for index, name in enumerate(names):
        block = raw.iloc[:, index * 3 : index * 3 + 3].copy()
        block.columns = ["start_sample_1based", "end_sample_inclusive", "label"]
        block = block.dropna(how="any")
        if block.empty:
            output[name] = pd.DataFrame(
                columns=[
                    "start_sample_1based",
                    "end_sample_inclusive",
                    "label_original",
                    "start_s",
                    "end_s",
                ]
            )
            continue
        block["start_sample_1based"] = block["start_sample_1based"].astype(np.int64)
        block["end_sample_inclusive"] = block["end_sample_inclusive"].astype(
            np.int64
        )
        block["label_original"] = block["label"].astype(np.int64).astype(str)
        block["start_s"] = (block["start_sample_1based"] - 1) / 1000.0
        block["end_s"] = block["end_sample_inclusive"] / 1000.0
        output[name] = block.drop(columns=["label"]).sort_values(
            "start_s", kind="stable"
        ).reset_index(drop=True)
    return output


def _interval_at_time(frame: pd.DataFrame, time_s: float) -> Mapping[str, Any] | None:
    if frame.empty:
        return None
    starts = frame["start_s"].to_numpy(dtype=float)
    index = int(np.searchsorted(starts, time_s, side="right") - 1)
    if index < 0:
        return None
    row = frame.iloc[index]
    if time_s > float(row.end_s):
        return None
    return row


def _butqdb_demographics(root: Path) -> dict[str, dict[str, str]]:
    path = root / "subject-info.csv"
    if not path.is_file():
        return {}
    frame = pd.read_csv(path, sep=";", dtype=str)
    return {
        str(row["ID"]): {str(key): str(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    }


def _ludb_diagnoses(comments: Sequence[str]) -> tuple[tuple[str, ...], str | None]:
    diagnoses: list[str] = []
    in_diagnoses = False
    rhythm = None
    for raw in comments:
        value = str(raw).strip()
        if value.casefold() == "<diagnoses>:":
            in_diagnoses = True
            continue
        if in_diagnoses and value:
            diagnoses.append(value)
            if value.casefold().startswith("rhythm:"):
                rhythm = value
    return tuple(diagnoses), rhythm


def _age_sex_from_comments(comments: Sequence[str]) -> tuple[str | None, str | None]:
    age = None
    sex = None
    for raw in comments:
        value = str(raw).strip()
        age_match = re.search(r"(?:<age>|Age)\s*:\s*([^\s]+)", value, re.I)
        sex_match = re.search(r"(?:<sex>|Sex)\s*:\s*([^\s]+)", value, re.I)
        if age_match:
            age = age_match.group(1)
        if sex_match:
            sex = sex_match.group(1)
        compact = re.match(r"^(\d+)\s+([MF])(?:\s|$)", value)
        if compact:
            age = age or compact.group(1)
            sex = sex or compact.group(2)
    return age, sex


def _source_record_from_comments(comments: Sequence[str]) -> str | None:
    for value in comments:
        match = re.search(r"Produced by xform from record\s+([^,\s]+)", value, re.I)
        if match:
            return match.group(1)
    return None


def _merge_ranges(
    ranges: Sequence[tuple[float, float, str]],
) -> list[tuple[float, float, tuple[str, ...]]]:
    if not ranges:
        return []
    ordered = sorted(ranges, key=lambda item: (item[0], item[1], item[2]))
    merged: list[list[Any]] = []
    for start, end, reason in ordered:
        if not merged or start > merged[-1][1]:
            merged.append([float(start), float(end), [str(reason)]])
        else:
            merged[-1][1] = max(float(end), float(merged[-1][1]))
            merged[-1][2].append(str(reason))
    return [
        (float(start), float(end), tuple(dict.fromkeys(reasons)))
        for start, end, reasons in merged
    ]


def _control_start(
    record_duration_s: float,
    intervals: Sequence[tuple[float, float]],
    duration_s: float,
) -> float:
    exclusions = sorted(
        (
            max(0.0, onset - SEIZURE_CONTEXT_S),
            min(record_duration_s, offset + SEIZURE_CONTEXT_S),
        )
        for onset, offset in intervals
    )
    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in exclusions:
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < record_duration_s:
        gaps.append((cursor, record_duration_s))
    eligible = [gap for gap in gaps if gap[1] - gap[0] >= duration_s]
    if not eligible:
        return 0.0
    start, end = max(eligible, key=lambda gap: gap[1] - gap[0])
    return float(start + 0.5 * ((end - start) - duration_s))


def _nstdb_noise_state(time_s: float) -> tuple[bool, int | None]:
    start = 300.0
    index = 0
    while start <= time_s:
        if start <= time_s < start + 120.0:
            return True, index
        start += 240.0
        index += 1
    return False, None


def _nearest_index(values: np.ndarray, target: float | int) -> int:
    insertion = int(np.searchsorted(values, target))
    choices = [
        index for index in (insertion - 1, insertion) if 0 <= index < values.size
    ]
    if not choices:
        raise ValueError("Cannot find a nearest value in an empty array")
    return min(choices, key=lambda index: abs(float(values[index]) - float(target)))


def _sample_interval_ms(
    first: int | None, second: int | None, sampling_rate_hz: float
) -> float | None:
    if first is None or second is None:
        return None
    return float(1000.0 * (second - first) / sampling_rate_hz)


def _finite_or_nan(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def _canonical_label_string(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value)
    return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text


def _value_counts(series: pd.Series) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in series.dropna().astype(str).value_counts().sort_index().items()
    }


def _valid_cache(csv_path: Path, metadata_path: Path) -> bool:
    if not csv_path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return metadata.get("builder_version") == BUILDER_VERSION


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_") or "record"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the label-preserving comprehensive ECG branch matrix."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            Path(__file__).resolve().parents[2]
            / "outputs"
            / "comprehensive_branch_matrix_v1"
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_KEYS,
        default=list(DATASET_KEYS),
    )
    parser.add_argument(
        "--tracks",
        nargs="+",
        choices=("neurokit", "zhai"),
        default=["neurokit"],
    )
    parser.add_argument(
        "--profile",
        choices=SUPPORTED_PROFILES,
        default=DEFAULT_PROFILE,
        help=(
            "validation selects label-rich windows; exhaustive covers every sample "
            "of every supported record with resumable windows"
        ),
    )
    parser.add_argument(
        "--segment-duration-s", type=float, default=DEFAULT_SEGMENT_DURATION_S
    )
    parser.add_argument("--max-records-per-dataset", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="write the exact segment plan and summary without extracting features",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.plan_only:
        roots = default_dataset_roots()
        planner = (
            plan_exhaustive_segments
            if args.profile == "exhaustive"
            else plan_validation_segments
        )
        specs = planner(
            roots,
            datasets=args.datasets,
            segment_duration_s=args.segment_duration_s,
            max_records_per_dataset=args.max_records_per_dataset,
        )
        output = args.output.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        plan = pd.DataFrame(_segment_manifest_row(spec) for spec in specs)
        plan_path = output / "segment_plan.csv"
        plan.to_csv(plan_path, index=False)
        summary = {
            "builder_version": BUILDER_VERSION,
            "profile": args.profile,
            "planned_segments": len(specs),
            "planned_track_executions": len(specs) * len(args.tracks),
            "signal_hours": float(sum(spec.duration_s for spec in specs) / 3600.0),
            "records_by_dataset": {
                key: len({spec.record_id for spec in specs if spec.dataset_key == key})
                for key in args.datasets
            },
            "segments_by_dataset": {
                key: sum(spec.dataset_key == key for spec in specs)
                for key in args.datasets
            },
            "plan_file": str(plan_path),
        }
        (output / "plan_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        return 0
    report = build_comprehensive_dataset(
        args.output,
        datasets=args.datasets,
        tracks=args.tracks,
        segment_duration_s=args.segment_duration_s,
        max_records_per_dataset=args.max_records_per_dataset,
        resume=not args.no_resume,
        profile=args.profile,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 0 if report["validation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
