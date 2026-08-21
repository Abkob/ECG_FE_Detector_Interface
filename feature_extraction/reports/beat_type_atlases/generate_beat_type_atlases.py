"""Build five evidence-backed MIT-BIH beat-type atlases.

The reports keep three claims separate:
1. the WFDB annotation is an expert reference beat label;
2. the RR-HRV front end detects a QRS/R event;
3. the current architecture does not classify the beat label.

Every displayed target example is chosen by a deterministic chronological-
middle rule, not by detector success.  Independent comparison strips are
drawn from another published MIT-BIH record carrying the same expert symbol.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb

from ecg_cascade.peaks import detect_neurokit_gradient, detect_unsw
from ecg_cascade.reliability import ReliabilityResult, assess_rr_reliability


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
DATABASE = (
    ROOT
    / "Datasets"
    / "mit-bih-arrhythmia-database-1.0"
    / "mit-bih-arrhythmia-database-1.0.0"
)
AUDIT = ROOT / "output" / "rr_hrv_architecture" / "mitdb_all48_v1"
SYMBOL_CSV = AUDIT / "unsw_symbol_sensitivity.csv"
DETECTOR_CSV = AUDIT / "detector_metrics.csv"

BEAT_SYMBOLS = {
    "N", "L", "R", "B", "A", "a", "J", "S", "V", "r", "F", "e", "j", "n", "E", "/", "f", "Q", "?",
}

RECORDS = {
    "108": {
        "subject": "87-year-old woman",
        "leads": "MLII and V1",
        "symbols": ["N", "A", "V", "F", "j"],
        "notes": (
            "The official record notes describe borderline first-degree AV block, sinus "
            "arrhythmia, multiform premature ventricular complexes, and considerable noise "
            "and baseline shift in the lower V1 channel. The figures use the upper MLII lead."
        ),
    },
    "113": {
        "subject": "24-year-old woman",
        "leads": "MLII and V1",
        "symbols": ["N", "a"],
        "notes": (
            "The official notes describe predominantly normal sinus rhythm with rate "
            "variation possibly caused by a wandering atrial pacemaker, and no special "
            "noise warning. The figures use the upper MLII lead."
        ),
    },
    "207": {
        "subject": "89-year-old woman",
        "leads": "MLII and V1",
        "symbols": ["L", "R", "V", "E", "A"],
        "notes": (
            "The official notes describe normal sinus rhythm with first-degree AV block, "
            "predominant LBBB, intermittent RBBB, multiform PVCs, idioventricular rhythm "
            "after ventricular flutter, and termination during supraventricular "
            "tachyarrhythmia. The database calls this record extremely difficult. The "
            "figures use the upper MLII lead."
        ),
    },
    "222": {
        "subject": "84-year-old woman",
        "leads": "MLII and V1",
        "symbols": ["N", "A", "J", "j"],
        "notes": (
            "The official notes describe paroxysmal atrial flutter/fibrillation, usually "
            "followed by nodal escape beats, plus intervals of high-frequency noise/artifact "
            "in both channels. The figures use the upper MLII lead."
        ),
    },
    "231": {
        "subject": "72-year-old woman",
        "leads": "MLII and V1",
        "symbols": ["N", "R", "A", "V"],
        "notes": (
            "The official notes describe periods of 2:1 AV block, Mobitz II block, "
            "rate-related RBBB, and a probable ventricular couplet. The figures use the "
            "upper MLII lead."
        ),
    },
}

RECORD_OBSERVATIONS = {
    "108": [
        "The selected refined output matched 1,733/1,739 expert N beats. The other four labels all have small denominators, so their 100\\% point estimates must not be treated as equally strong evidence.",
        "For the displayed F beat, inverted-path refinement reached the expert sample while the selected positive-amplitude path was 33.3~ms early. This is direct evidence that polarity can alter the fiducial without implying that inversion is globally preferable.",
        "The official noise warning concerns the lower V1 lead; it cannot automatically explain an error measured on the displayed upper MLII lead.",
    ],
    "113": [
        "The selected output matched all six expert a beats, but the 95\\% Wilson interval remains 60.97--100.00\\%. Six observations do not establish robust aberrated-atrial-beat generalization.",
        "The official record has no special noise warning. Successful QRS localization here therefore tests clean morphology more than severe artifact tolerance.",
        "Both displayed selected refined timestamps are exact at the 360-Hz sample grid, while the inverted path is later. This supports the selected polarity for these examples only, not an automatic global routing rule.",
    ],
    "207": [
        "The V class is the principal label-specific failure: 96/105 were matched (91.43\\%), and the deterministic middle V was 77.8~ms late, just outside the predefined 75-ms tolerance. It is a genuine expert V beat, not an artifact label.",
        "All 1,457 L beats were matched, yet the displayed L fiducial moved 52.8~ms from the expert and lacked exact secondary support. Passing the broad audit tolerance therefore does not guarantee a precise R fiducial.",
        "Positive-amplitude refinement changed full-record $F_1$ from 0.96239 to 0.95979 (FP 140 to 145; FN 5 to 10). Refinement was not uniformly beneficial on this morphology-rich record.",
        "The displayed A beat was detected within 75~ms but lacked exact secondary support. Two-detector disagreement is therefore reliability context, not permission to veto a genuine beat.",
    ],
    "222": [
        "Uppercase J and lowercase j must remain separate: the record has 1 J but 212 j annotations. Combining them would erase the premature-versus-escape distinction and inflate the apparent J evidence.",
        "The selected output matched 2,054/2,062 N beats and every requested non-N beat, but the official record contains both arrhythmia and intervals of noise. The expert symbol table does not by itself identify which detector errors were caused by noise.",
        "Full-record $F_1$ decreased from 0.99819 to 0.99778 after positive-amplitude refinement, so even a high-performing detector can be slightly worsened by an added fiducial step.",
    ],
    "231": [
        "All requested expert beats were matched, but A has only 1 observation and V only 2. Their 100\\% results are case descriptions, not stable class-wide evidence.",
        "The refined detector still produced 11 false-positive events despite zero false negatives. Label-wise sensitivity cannot show those errors, which is why full-record PPV/F1 remains necessary.",
        "The dominant R labels reflect rate-related RBBB documented by the database. This is a genuine conduction pattern and must not be converted into an artifact label when morphology changes.",
    ],
}

REFERENCE_RECORD = {
    "N": "100",
    "A": "100",
    "a": "201",
    "V": "102",
    "F": "208",
    "j": "124",
    "J": "124",
    "L": "109",
    "R": "118",
    "E": "210",
}


TYPE_INFO = {
    "N": {
        "title": "normal beat",
        "definition": "Expert reference label for a normal beat.",
        "mechanism": (
            "A sinus impulse normally depolarizes the atria first, producing the P wave, "
            "then reaches the ventricles through the AV node and His--Purkinje system, "
            "producing the QRS; ventricular repolarization produces the T wave "
            "\\cite{kligfield2007standardization,surawicz2009conduction}. The database "
            "symbol describes the expert beat class. It does not require the largest QRS "
            "deflection to point upward in every lead."
        ),
        "reading": (
            "Polarity and amplitude depend on the projection of the cardiac electrical "
            "vector onto the selected lead. A negative or biphasic complex may therefore "
            "still carry an expert normal-beat label. The comparison strip demonstrates "
            "allowed appearance variation; it is not a universal template."
        ),
        "caution": (
            "A single-lead waveform cannot establish that the entire 12-lead ECG is normal, "
            "and QRS detection cannot establish the N class."
        ),
    },
    "A": {
        "title": "atrial premature beat",
        "definition": "Expert reference label for an atrial premature beat.",
        "mechanism": (
            "An ectopic atrial focus fires before the next expected sinus impulse. The early "
            "P wave can differ from the sinus P wave or overlap the preceding T wave. If the "
            "impulse reaches a recovered His--Purkinje system, the QRS is often narrow and "
            "similar to neighboring conducted beats \\cite{guichard2022pac,samesima2022ecg}."
        ),
        "reading": (
            "Prematurity and atrial activity are central to this label; an isolated QRS shape "
            "is insufficient. If a bundle branch remains refractory, the same supraventricular "
            "impulse can conduct aberrantly and produce a wider or altered QRS."
        ),
        "caution": (
            "The detector locates ventricular activation. It does not inspect the P-wave "
            "origin or prove that an early beat is atrial."
        ),
    },
    "a": {
        "title": "aberrated atrial premature beat",
        "definition": "Expert reference label for an aberrated atrial premature beat.",
        "mechanism": (
            "The impulse is premature and atrial, but part of the ventricular conduction "
            "system is still refractory. Ventricular activation therefore follows a "
            "bundle-branch or fascicular-block pattern, widening or changing the QRS even "
            "though the origin is supraventricular \\cite{samesima2022ecg,escudero2021wide}."
        ),
        "reading": (
            "This label illustrates why `wide' does not automatically mean ventricular. "
            "The timing of the early atrial impulse, any altered P wave, and the relationship "
            "to neighboring cycles help distinguish aberrancy from a PVC."
        ),
        "caution": (
            "There are only six such annotations in record 113. Detection of their QRS "
            "complexes is not evidence that the architecture can distinguish a from V."
        ),
    },
    "V": {
        "title": "premature ventricular contraction",
        "definition": "Expert reference label for a premature ventricular contraction.",
        "mechanism": (
            "A PVC starts in ventricular tissue rather than arriving through the usual "
            "atrial/AV conduction sequence. Activation spreads more slowly and asymmetrically "
            "through myocardium, commonly creating a broad or unusual QRS with secondary "
            "ST--T discordance \\cite{prisco2023pvc,surawicz2009conduction}."
        ),
        "reading": (
            "PVC morphology varies with ventricular origin and lead direction. A broad QRS "
            "supports abnormal activation but is not perfectly specific because an atrial or "
            "junctional beat may also be wide when conducted aberrantly."
        ),
        "caution": (
            "The expert V symbol supplies the class. A QRS detector can find the event while "
            "remaining incapable of deciding whether its origin was ventricular."
        ),
    },
    "F": {
        "title": "fusion of ventricular and normal activation",
        "definition": "Expert reference label for fusion of ventricular and normal activation.",
        "mechanism": (
            "A conducted supraventricular wavefront and a ventricular wavefront activate the "
            "ventricles at nearly the same time. The QRS is consequently intermediate between "
            "a fully conducted complex and a ventricular complex, with its exact shape set by "
            "the relative timing and sites of activation \\cite{whitaker2023vt}."
        ),
        "reading": (
            "Fusion has no fixed template. Width, notching, amplitude, and polarity change as "
            "the two wavefronts contribute different amounts. Neighboring conducted and "
            "ventricular beats are necessary context."
        ),
        "caution": (
            "Record 108 contains only two F labels. A two-beat denominator cannot support a "
            "stable performance claim, and QRS timing does not classify fusion."
        ),
    },
    "j": {
        "title": "nodal/junctional escape beat",
        "definition": "Expert reference label for a nodal or junctional escape beat.",
        "mechanism": (
            "An escape beat is delayed: a subsidiary junctional pacemaker fires because the "
            "expected dominant impulse did not activate the ventricles in time. Because the "
            "focus is near the His--Purkinje system, the QRS is often narrow; a normal P wave "
            "may be absent and retrograde atrial activity may appear around the QRS "
            "\\cite{samesima2022ecg}."
        ),
        "reading": (
            "Lowercase j is an escape event, so the preceding pause is essential. Its QRS can "
            "look nearly normal when ventricular conduction remains intact."
        ),
        "caution": (
            "Lowercase j and uppercase J are different WFDB classes. Case-sensitive handling "
            "is mandatory in code and filenames."
        ),
    },
    "J": {
        "title": "nodal/junctional premature beat",
        "definition": "Expert reference label for a nodal or junctional premature beat.",
        "mechanism": (
            "A premature junctional focus fires early, before the next expected sinus beat. "
            "Ventricular activation often enters the His--Purkinje system and may remain "
            "narrow, while atrial activation can be retrograde or hidden within the QRS "
            "\\cite{samesima2022ecg}."
        ),
        "reading": (
            "Uppercase J is premature, whereas lowercase j is delayed escape. Their isolated "
            "QRS complexes can look similar; sequence timing and atrial activity distinguish "
            "the mechanisms."
        ),
        "caution": (
            "Record 222 contains one J annotation. That one event is a case study, not a "
            "junctional-premature sensitivity estimate."
        ),
    },
    "L": {
        "title": "left bundle-branch-block beat",
        "definition": "Expert reference label for a beat with left bundle-branch-block morphology.",
        "mechanism": (
            "With LBBB, left-ventricular activation is delayed and proceeds through an altered "
            "sequence after right-sided activation. QRS duration and morphology therefore "
            "change, with criteria defined from multiple leads in clinical ECG standards "
            "\\cite{surawicz2009conduction}."
        ),
        "reading": (
            "The MIT--BIH expert L symbol is available to this audit, but the displayed MLII "
            "lead alone cannot independently reproduce the full diagnostic 12-lead LBBB "
            "criteria. Within record 207, broad morphology and polarity changes challenge "
            "fixed-shape detectors."
        ),
        "caution": (
            "LBBB is a conduction pattern, not automatically a separate ectopic beat or "
            "physical artifact. A detector disagreement must not be relabeled as noise."
        ),
    },
    "R": {
        "title": "right bundle-branch-block beat",
        "definition": "Expert reference label for a beat with right bundle-branch-block morphology.",
        "mechanism": (
            "With RBBB, right-ventricular activation is delayed after left-sided activation. "
            "The resulting QRS is prolonged and has lead-dependent terminal changes; formal "
            "diagnosis uses a multi-lead pattern \\cite{surawicz2009conduction}."
        ),
        "reading": (
            "The waveform can differ substantially from both N and L beats in the same "
            "record. Record 231 is specifically described as having rate-related RBBB, so the "
            "conduction pattern may appear or disappear with cycle length."
        ),
        "caution": (
            "The expert R annotation is the reference label. A single MLII strip shows the "
            "detector challenge but is not a standalone clinical RBBB diagnosis."
        ),
    },
    "E": {
        "title": "ventricular escape beat",
        "definition": "Expert reference label for a ventricular escape beat.",
        "mechanism": (
            "A ventricular escape focus fires late when higher pacemakers or conduction fail "
            "to activate the ventricles. Because the impulse begins in ventricular tissue, "
            "the QRS is commonly broad and unusual; unlike a PVC, its defining timing is late "
            "rather than premature \\cite{samesima2022ecg,surawicz2009conduction}."
        ),
        "reading": (
            "E and V may both have ventricular-looking QRS complexes. The preceding pause and "
            "rhythm sequence distinguish escape from premature activation, so isolated-shape "
            "comparison is insufficient."
        ),
        "caution": (
            "QRS detection does not establish whether a ventricular complex was premature or "
            "an escape response. RR context is relevant, but a classifier is still separate."
        ),
    },
}


@dataclass
class RecordRun:
    signal: np.ndarray
    fs: float
    lead_name: str
    units: str
    expert_samples: np.ndarray
    expert_symbols: np.ndarray
    original: ReliabilityResult
    inverted: ReliabilityResult


def load_record(record: str) -> tuple[wfdb.Record, wfdb.Annotation]:
    base = str(DATABASE / record)
    return wfdb.rdrecord(base), wfdb.rdann(base, "atr")


def expert_beats(annotation: wfdb.Annotation) -> tuple[np.ndarray, np.ndarray]:
    samples: list[int] = []
    symbols: list[str] = []
    for sample, symbol in zip(annotation.sample, annotation.symbol, strict=True):
        if symbol in BEAT_SYMBOLS:
            samples.append(int(sample))
            symbols.append(str(symbol))
    return np.asarray(samples, dtype=np.int64), np.asarray(symbols, dtype=object)


def run_architecture(record: str) -> RecordRun:
    rec, ann = load_record(record)
    signal = np.asarray(rec.p_signal[:, 0], dtype=np.float64)
    fs = float(rec.fs)
    expert_samples, expert_symbols = expert_beats(ann)

    primary = detect_unsw(signal, fs, orientation="original")
    secondary = detect_neurokit_gradient(
        signal,
        fs,
        orientation="original",
        minimum_delay_ms=250.0,
        minimum_delay_inclusive=True,
    )
    original = assess_rr_reliability(
        primary.peak_samples,
        secondary.peak_samples,
        primary.analysis_signal,
        sampling_rate_hz=fs,
        close_detection_exclusion_ms=150.0,
        support_tolerance_ms=50.0,
        r_fiducial_refinement_ms=50.0,
    )

    primary_i = detect_unsw(signal, fs, orientation="inverted")
    secondary_i = detect_neurokit_gradient(
        signal,
        fs,
        orientation="inverted",
        minimum_delay_ms=250.0,
        minimum_delay_inclusive=True,
    )
    inverted = assess_rr_reliability(
        primary_i.peak_samples,
        secondary_i.peak_samples,
        primary_i.analysis_signal,
        sampling_rate_hz=fs,
        close_detection_exclusion_ms=150.0,
        support_tolerance_ms=50.0,
        r_fiducial_refinement_ms=50.0,
    )
    return RecordRun(
        signal=signal,
        fs=fs,
        lead_name=str(rec.sig_name[0]),
        units=str(rec.units[0]),
        expert_samples=expert_samples,
        expert_symbols=expert_symbols,
        original=original,
        inverted=inverted,
    )


def middle_occurrence(samples: np.ndarray, symbols: np.ndarray, symbol: str, length: int, fs: float) -> int:
    candidates = samples[symbols == symbol]
    margin = int(round(1.5 * fs))
    interior = candidates[(candidates > margin) & (candidates < length - margin)]
    if interior.size:
        candidates = interior
    if not candidates.size:
        raise RuntimeError(f"No {symbol!r} annotations found")
    return int(candidates[(candidates.size - 1) // 2])


def nearest_event(result: ReliabilityResult, expert_sample: int, fs: float) -> dict[str, object]:
    events = result.primary_events
    refined = events["r_fiducial_sample"].to_numpy(dtype=int)
    if not refined.size:
        return {
            "candidate": None,
            "refined": None,
            "initial_error_ms": np.nan,
            "refined_error_ms": np.nan,
            "support": False,
            "detected": False,
        }
    index = int(np.argmin(np.abs(refined - expert_sample)))
    row = events.iloc[index]
    candidate = int(row["primary_candidate_sample"])
    fiducial = int(row["r_fiducial_sample"])
    return {
        "candidate": candidate,
        "refined": fiducial,
        "initial_error_ms": (candidate - expert_sample) * 1000.0 / fs,
        "refined_error_ms": (fiducial - expert_sample) * 1000.0 / fs,
        "support": bool(row["qrs_supported"]),
        "detected": abs(fiducial - expert_sample) <= int(round(0.075 * fs)),
    }


def safe_symbol(symbol: str) -> str:
    if symbol == "J":
        return "uppercase_J_premature"
    if symbol == "j":
        return "lowercase_j_escape"
    return symbol


def y_at(signal: np.ndarray, sample: int | None) -> float:
    if sample is None:
        return float("nan")
    return float(signal[int(np.clip(sample, 0, signal.size - 1))])


def plot_comparison(record: str, symbol: str, run: RecordRun, target_sample: int) -> Path:
    ref_record = REFERENCE_RECORD[symbol]
    ref_rec, ref_ann = load_record(ref_record)
    ref_signal = np.asarray(ref_rec.p_signal[:, 0], dtype=float)
    ref_samples, ref_symbols = expert_beats(ref_ann)
    ref_sample = middle_occurrence(ref_samples, ref_symbols, symbol, ref_signal.size, float(ref_rec.fs))

    original = nearest_event(run.original, target_sample, run.fs)
    inverted = nearest_event(run.inverted, target_sample, run.fs)
    path = FIGURES / f"record_{record}_{safe_symbol(symbol)}_comparison.png"

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.25), constrained_layout=True)
    panels = [
        (axes[0], run.signal, run.fs, target_sample, run.expert_samples, run.expert_symbols,
         f"Target: record {record}, {run.lead_name}"),
        (axes[1], ref_signal, float(ref_rec.fs), ref_sample, ref_samples, ref_symbols,
         f"Independent reference: record {ref_record}, {ref_rec.sig_name[0]}"),
    ]
    for panel_index, (ax, signal, fs, center, samples, symbols, title) in enumerate(panels):
        half = int(round(1.25 * fs))
        start = max(0, center - half)
        stop = min(signal.size, center + half)
        time = (np.arange(start, stop) - center) / fs
        ax.plot(time, signal[start:stop], color="#274C77", lw=1.15, label="Raw ECG")
        mask = (samples >= start) & (samples < stop)
        for sample, nearby_symbol in zip(samples[mask], symbols[mask], strict=True):
            is_center = int(sample) == center
            ax.scatter(
                [(sample - center) / fs],
                [signal[sample]],
                marker="D",
                s=64 if is_center else 33,
                facecolors="white",
                edgecolors="#B22222" if is_center else "#555555",
                linewidths=1.5 if is_center else 0.8,
                zorder=7,
            )
            ax.annotate(
                str(nearby_symbol),
                ((sample - center) / fs, signal[sample]),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold" if is_center else "normal",
                color="#B22222" if is_center else "#444444",
            )
        if panel_index == 0:
            ax.scatter(
                [(original["candidate"] - center) / fs],
                [y_at(signal, original["candidate"])],
                marker="^",
                s=62,
                color="#2A9D8F",
                zorder=8,
                label="UNSW initial QRS event",
            )
            ax.scatter(
                [(original["refined"] - center) / fs],
                [y_at(signal, original["refined"])],
                marker="x",
                s=76,
                linewidths=2.0,
                color="#E76F51",
                zorder=9,
                label="Selected original refined R",
            )
            ax.scatter(
                [(inverted["refined"] - center) / fs],
                [y_at(signal, inverted["refined"])],
                marker="P",
                s=66,
                color="#6A4C93",
                zorder=8,
                label="Inverted-path refined context",
            )
            secondary = run.original.secondary_collapsed_samples
            sec = secondary[(secondary >= start) & (secondary < stop)]
            low, high = np.nanpercentile(signal[start:stop], [1, 99])
            tick_y = low - 0.08 * max(high - low, 1e-6)
            ax.scatter(
                (sec - center) / fs,
                np.full(sec.size, tick_y),
                marker="|",
                s=130,
                linewidths=1.5,
                color="#111111",
                label="NeuroKit-gradient context",
            )
            ax.legend(loc="upper right", fontsize=7.2, framealpha=0.9)
        ax.axvline(0, color="#B22222", ls="--", lw=0.9, alpha=0.55)
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Time relative to selected expert annotation (s)")
        ax.grid(alpha=0.20)
    axes[0].set_ylabel(f"Amplitude ({run.units})")
    axes[1].set_ylabel(f"Amplitude ({ref_rec.units[0]})")
    fig.suptitle(
        f"Expert symbol {symbol}: target detector audit versus another published MIT-BIH example",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.savefig(path, dpi=230)
    plt.close(fig)
    return path


def wilson_interval(matched: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    p = matched / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def plot_symbol_summary(record: str, rows: pd.DataFrame) -> Path:
    order = RECORDS[record]["symbols"]
    rows = rows.set_index("symbol").loc[order].reset_index()
    matched = rows["matched_count"].to_numpy(dtype=int)
    total = rows["reference_count"].to_numpy(dtype=int)
    sensitivity = matched / total
    cis = np.asarray([wilson_interval(int(k), int(n)) for k, n in zip(matched, total, strict=True)])
    # Floating-point roundoff can make the p=1 upper distance a tiny negative.
    lower = np.maximum(0.0, sensitivity - cis[:, 0])
    upper = np.maximum(0.0, cis[:, 1] - sensitivity)

    fig, ax = plt.subplots(figsize=(9.4, max(3.0, 0.62 * len(order) + 1.5)), constrained_layout=True)
    y = np.arange(len(order))
    ax.errorbar(
        sensitivity,
        y,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        color="#163A5F",
        ecolor="#E76F51",
        capsize=4,
        lw=1.5,
    )
    for idx, (score, k, n) in enumerate(zip(sensitivity, matched, total, strict=True)):
        ax.text(max(0.01, score - 0.02), idx - 0.22, f"{k}/{n}", ha="right", va="center", fontsize=8.5)
    ax.set_yticks(y, [f"{s}: {TYPE_INFO[s]['title']}" for s in order])
    ax.set_xlim(0, 1.035)
    ax.set_xlabel("QRS-detection sensitivity with 95% Wilson interval")
    ax.set_title(f"Record {record}: selected refined R timestamps versus expert beat annotations")
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()
    path = FIGURES / f"record_{record}_symbol_sensitivity.png"
    fig.savefig(path, dpi=230)
    plt.close(fig)
    return path


def format_error(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    sign = "+" if value > 0 else ""
    return f"${sign}{value:.1f}$"


def yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def selected_sentence(symbol: str, original: dict[str, object], inverted: dict[str, object], matched: int, total: int) -> str:
    status = "was" if original["detected"] else "was not"
    return (
        f"For the deterministic middle {symbol} example, the UNSW initial event was "
        f"{format_error(float(original['initial_error_ms']))}~ms from the expert annotation; "
        f"the selected original-polarity refined timestamp was "
        f"{format_error(float(original['refined_error_ms']))}~ms away and {status} within the "
        f"predefined $\\pm75$-ms audit tolerance. Exact NeuroKit-gradient support within "
        f"$\\pm50$~ms of the primary QRS event was {yes_no(bool(original['support'])).lower()}. "
        f"The inverted-context refined timestamp was "
        f"{format_error(float(inverted['refined_error_ms']))}~ms away. Across the full record, "
        f"the selected refined output matched {matched} of {total} expert {symbol} annotations. "
        "This is QRS localization, not beat-type classification."
    )


def tex_table_inventory(record: str, rows: pd.DataFrame) -> str:
    rows = rows.set_index("symbol")
    lines = []
    for symbol in RECORDS[record]["symbols"]:
        row = rows.loc[symbol]
        matched = int(row["matched_count"])
        total = int(row["reference_count"])
        low, high = wilson_interval(matched, total)
        lines.append(
            f"\\texttt{{{symbol}}} & {TYPE_INFO[symbol]['definition']} & {total:,} & "
            f"{matched:,}/{total:,} & {100*matched/total:.2f}\\% & "
            f"{100*low:.2f}--{100*high:.2f}\\% \\\\"
        )
    return "\n".join(lines)


def tex_selected_table(record: str, selected: dict[str, dict[str, object]], run: RecordRun) -> str:
    lines = []
    for symbol in RECORDS[record]["symbols"]:
        item = selected[symbol]
        original = item["original"]
        inverted = item["inverted"]
        sample = int(item["sample"])
        lines.append(
            f"\\texttt{{{symbol}}} & {sample:,} & {sample/run.fs:.3f} & "
            f"{format_error(float(original['initial_error_ms']))} & "
            f"{format_error(float(original['refined_error_ms']))} & "
            f"{yes_no(bool(original['support']))} & "
            f"{format_error(float(inverted['refined_error_ms']))} & "
            f"{yes_no(bool(original['detected']))} & Not produced \\\\"
        )
    return "\n".join(lines)


def make_tex(record: str, run: RecordRun, symbol_rows: pd.DataFrame, detector_rows: pd.DataFrame,
             selected: dict[str, dict[str, object]]) -> str:
    config = RECORDS[record]
    refined_row = detector_rows[
        (detector_rows["record"].astype(str) == record)
        & (detector_rows["detector"] == "unsw_refined_r")
    ].iloc[0]
    initial_row = detector_rows[
        (detector_rows["record"].astype(str) == record)
        & (detector_rows["detector"] == "unsw_initial")
    ].iloc[0]
    record_symbol_rows = symbol_rows[symbol_rows["record"].astype(str) == record].copy()
    inventory = tex_table_inventory(record, record_symbol_rows)
    selected_table = tex_selected_table(record, selected, run)
    observation_items = "\n".join(f"\\item {item}" for item in RECORD_OBSERVATIONS[record])

    label_sections = []
    indexed = record_symbol_rows.set_index("symbol")
    for symbol in config["symbols"]:
        info = TYPE_INFO[symbol]
        item = selected[symbol]
        row = indexed.loc[symbol]
        matched = int(row["matched_count"])
        total = int(row["reference_count"])
        sample = int(item["sample"])
        ref_record = REFERENCE_RECORD[symbol]
        caution = info["caution"]
        if total < 20:
            caution += (
                f" The local denominator is only {total}; always report the raw {matched}/{total} "
                "rather than presenting the percentage alone."
            )
        label_sections.append(
            dedent(
                rf"""
                \clearpage
                \section{{\texttt{{{symbol}}}: {info['title']}}}

                \begin{{figure}}[H]
                \centering
                \includegraphics[width=\textwidth]{{figures/record_{record}_{safe_symbol(symbol)}_comparison.png}}
                \caption{{Left: the chronologically middle expert \texttt{{{symbol}}} occurrence in record {record}
                (sample {sample:,}, {sample/run.fs:.3f}~s), with current detector outputs. Right: an independent
                published MIT--BIH record {ref_record} occurrence carrying the same expert symbol. Neighboring
                reference symbols are shown to preserve rhythm context \cite{{moody2001mitbih,physionetmitdb}}.}}
                \end{{figure}}

                \textbf{{Official symbol meaning.}} {info['definition']}

                \textbf{{Why this type occurs and what tends to shape the waveform.}} {info['mechanism']}

                \textbf{{How to read the two strips.}} {info['reading']} The right-hand strip is not a
                synthetic textbook ideal and was not selected as a detector success; it is another expert-labeled
                observation from the published database. Similar labels can legitimately have different waveforms.

                \auditbox{{{selected_sentence(symbol, item['original'], item['inverted'], matched, total)}}}

                \cautionbox{{{caution}}}
                """
            ).strip()
        )

    counts = ", ".join(
        f"\\texttt{{{symbol}}} ($n={int(indexed.loc[symbol, 'reference_count']):,}$)"
        for symbol in config["symbols"]
    )
    return dedent(
        rf"""
        \documentclass[11pt]{{article}}
        \usepackage[margin=0.72in]{{geometry}}
        \usepackage[T1]{{fontenc}}
        \usepackage[utf8]{{inputenc}}
        \usepackage{{lmodern}}
        \usepackage{{microtype}}
        \usepackage{{graphicx}}
        \usepackage{{booktabs}}
        \usepackage{{tabularx}}
        \usepackage{{array}}
        \usepackage{{xcolor}}
        \usepackage{{float}}
        \usepackage{{caption}}
        \usepackage{{enumitem}}
        \usepackage{{fancyhdr}}
        \usepackage[numbers,sort&compress]{{natbib}}
        \usepackage[colorlinks=true,linkcolor=blue!55!black,citecolor=blue!55!black,urlcolor=blue!55!black]{{hyperref}}

        \definecolor{{navy}}{{HTML}}{{163A5F}}
        \definecolor{{pale}}{{HTML}}{{EEF4F9}}
        \definecolor{{warning}}{{HTML}}{{FFF3CD}}
        \definecolor{{rulegray}}{{HTML}}{{B8C2CC}}
        \newcommand{{\auditbox}}[1]{{%
          \par\smallskip\noindent
          \fcolorbox{{navy}}{{pale}}{{\parbox{{0.955\linewidth}}{{\textbf{{Detector audit.}} #1}}}}%
          \par\smallskip
        }}
        \newcommand{{\cautionbox}}[1]{{%
          \par\smallskip\noindent
          \fcolorbox{{rulegray}}{{warning}}{{\parbox{{0.955\linewidth}}{{\textbf{{Critical limitation.}} #1}}}}%
          \par\smallskip
        }}

        \pagestyle{{fancy}}
        \fancyhf{{}}
        \lhead{{MIT--BIH record {record} beat-type atlas}}
        \rhead{{RR--HRV front end}}
        \cfoot{{\thepage}}
        \setlength{{\headheight}}{{14pt}}
        \setlength{{\parindent}}{{0pt}}
        \setlength{{\parskip}}{{4.5pt}}
        \setlist[itemize]{{leftmargin=1.2em,itemsep=2pt,topsep=3pt}}
        \captionsetup{{font=small,labelfont=bf}}

        \title{{\textbf{{MIT--BIH Record {record}: Detailed Beat-Type Atlas}}\\[0.35em]
        \large Expert labels, physiological interpretation, and current QRS-detection audit}}
        \author{{ECG-only seizure detector project}}
        \date{{4 August 2026}}

        \begin{{document}}
        \maketitle

        \begin{{center}}
        \fcolorbox{{navy}}{{pale}}{{\parbox{{0.91\textwidth}}{{
        \textbf{{The central distinction.}} The symbols in this report are expert WFDB beat annotations.
        The current RR--HRV front end detects QRS/R events and measures detector agreement; it does
        \emph{{not}} predict these beat classes. Therefore ``QRS detected'' is never rewritten as
        ``beat type classified correctly,'' and detector disagreement is never automatically called artifact.
        }}}}
        \end{{center}}

        \section{{Record, data, and selection rule}}

        The MIT--BIH Arrhythmia Database contains 48 two-lead ambulatory ECG recordings with expert-reviewed
        beat annotations \cite{{goldberger2000physiobank,moody2001mitbih,physionetmitdb}}. Record {record} is
        a 30-minute, 360-Hz recording from a {config['subject']} with {config['leads']}. {config['notes']}

        The beat labels audited here are {counts}. For each label, the displayed target is the chronologically
        middle occurrence after excluding only the 1.5-second file edges. This deterministic rule does not
        select the best-looking or best-detected beat. The comparison comes from another MIT--BIH record with
        the same expert symbol, so the report shows real published variability rather than an invented ideal.

        \subsection{{Current detector path and marker meanings}}

        The primary event detector is the Khamis/UNSW QRS detector; the NeuroKit gradient detector supplies
        secondary context \cite{{khamis2016telehealth,makowski2021neurokit,kristof2024qrs}}. The selected
        output is the original-polarity UNSW event refined to the highest positive amplitude within $\pm50$~ms.
        The inverted-polarity result is shown as context only and cannot silently replace the selected timestamp.
        A primary event is called supported when exactly one secondary event lies within $\pm50$~ms. Expert
        matching uses a separate one-to-one $\pm75$-ms audit tolerance.

        In each target panel: the red-edged diamond is the selected expert annotation; gray diamonds and labels
        are neighboring expert beats; the teal triangle is the initial UNSW event; the orange cross is the
        selected refined R timestamp; the purple plus-shaped marker is inverted-path context; and black baseline
        ticks are NeuroKit-gradient context. None of those detector markers is a beat-class prediction.

        \cautionbox{{The $\pm75$-ms audit tolerance defines this project's matching experiment; it is not an
        industry declaration that any timestamp inside the window is an exact R-wave fiducial. A wide tolerance
        can hide timing jitter that matters to RR and HRV. Exact raw timing errors are therefore retained.}}

        \clearpage
        \section{{Full-record QRS-detection audit by expert label}}

        \begin{{table}}[H]
        \centering
        \caption{{Selected refined R timestamps versus all expert annotations of each requested label.}}
        \small
        \begin{{tabularx}}{{\textwidth}}{{lXrrrr}}
        \toprule
        Symbol & Official meaning & Expert $n$ & Matched/$n$ & Sensitivity & 95\% Wilson interval \\
        \midrule
        {inventory}
        \bottomrule
        \end{{tabularx}}
        \end{{table}}

        \begin{{figure}}[H]
        \centering
        \includegraphics[width=0.93\textwidth]{{figures/record_{record}_symbol_sensitivity.png}}
        \caption{{The interval width, not just the point estimate, exposes how little rare labels establish.
        Intervals use the Wilson score method \cite{{wilson1927probable}}.}}
        \end{{figure}}

        The full-record initial UNSW result was TP={int(initial_row['tp']):,}, FP={int(initial_row['fp']):,},
        FN={int(initial_row['fn']):,}, and $F_1={float(initial_row['f1']):.5f}$. After the current positive-amplitude
        fiducial refinement it was TP={int(refined_row['tp']):,}, FP={int(refined_row['fp']):,},
        FN={int(refined_row['fn']):,}, and $F_1={float(refined_row['f1']):.5f}$. Per-label sensitivity allocates
        matched expert beats to their symbols; false-positive detector events have no expert symbol and therefore
        cannot be represented by sensitivity alone. This table is not a beat-classification confusion matrix.

        \subsection{{Exact selected-example audit}}
        \begin{{table}}[H]
        \centering
        \caption{{Timing results for the deterministic displayed examples. Errors are detector minus expert.}}
        \resizebox{{\textwidth}}{{!}}{{%
        \begin{{tabular}}{{lrrrrrrrrl}}
        \toprule
        Expert & Sample & Time (s) & Initial error (ms) & Selected R error (ms) & Supported? & Inverted R error (ms) & QRS detected? & Beat class output \\
        \midrule
        {selected_table}
        \bottomrule
        \end{{tabular}}}}
        \end{{table}}

        \subsection{{Record-specific critical reading}}
        \begin{{itemize}}
        {observation_items}
        \end{{itemize}}

        {chr(10).join(label_sections)}

        \clearpage
        \section{{Conclusions for record {record}}}
        \begin{{enumerate}}
        \item This report validates QRS-event localization against expert timestamps. It does not validate
        normal, atrial, ventricular, conduction-pattern, or junctional classification;
        seizure detection; or artifact detection.
        \item The full raw denominators and Wilson intervals must accompany percentages. A 100\% result with
        one or two beats remains weak evidence.
        \item The independent comparison strips show that one expert label can have multiple lead-dependent
        shapes. Template resemblance alone is not ground truth.
        \item Original and inverted refinements are both displayed because morphology and polarity can move the
        chosen positive maximum. The inverted path remains context until a routing rule is prospectively specified
        and independently validated.
        \item Detector support is a reliability measurement, not a veto and not an artifact label. Expert
        annotations remain the evaluation reference.
        \end{{enumerate}}

        \bibliographystyle{{unsrtnat}}
        \bibliography{{MITBIH_Beat_Type_Atlases}}
        \end{{document}}
        """
    ).strip() + "\n"


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    symbol_rows = pd.read_csv(SYMBOL_CSV, dtype={"record": str, "symbol": str})
    detector_rows = pd.read_csv(DETECTOR_CSV, dtype={"record": str, "detector": str})
    selected_rows: list[dict[str, object]] = []

    for record, config in RECORDS.items():
        print(f"Running record {record}...", flush=True)
        run = run_architecture(record)
        selected: dict[str, dict[str, object]] = {}
        for symbol in config["symbols"]:
            sample = middle_occurrence(
                run.expert_samples,
                run.expert_symbols,
                symbol,
                run.signal.size,
                run.fs,
            )
            original = nearest_event(run.original, sample, run.fs)
            inverted = nearest_event(run.inverted, sample, run.fs)
            selected[symbol] = {"sample": sample, "original": original, "inverted": inverted}
            plot_comparison(record, symbol, run, sample)
            selected_rows.append(
                {
                    "record": record,
                    "symbol": symbol,
                    "expert_sample": sample,
                    "expert_time_s": sample / run.fs,
                    **{f"original_{key}": value for key, value in original.items()},
                    **{f"inverted_{key}": value for key, value in inverted.items()},
                }
            )
        record_symbols = symbol_rows[symbol_rows["record"] == record]
        plot_symbol_summary(record, record_symbols)
        tex = make_tex(record, run, symbol_rows, detector_rows, selected)
        (HERE / f"MITBIH_Record_{record}_Detailed_Beat_Type_Atlas.tex").write_text(tex, encoding="utf-8")

    pd.DataFrame(selected_rows).to_csv(HERE / "selected_example_audit.csv", index=False)
    print("Finished generating figures, TeX files, and selected_example_audit.csv", flush=True)


if __name__ == "__main__":
    main()
