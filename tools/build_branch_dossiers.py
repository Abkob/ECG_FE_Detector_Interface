"""Build exhaustive, current-state ECGdetector branch dossiers.

The short branch records are useful entry points, but they are not adequate
history/audit records.  This builder creates a detailed current supplement for
each branch, inventories every mapped Python symbol and test, records every
saved result directory and branch-tagged evidence item, and then appends the
selected project-authored legacy reports and case studies verbatim.

Run from the repository root after ``build_documentation_bundle.py`` has
refreshed the source maps.  Compile the generated TeX from the repository root
so the repo-relative PDF and test-log paths remain reproducible.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import re
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - page counts are informative only
    PdfReader = None


ROOT = Path(__file__).resolve().parents[1]
FE = ROOT / "feature_extraction"
OUTPUT = ROOT / "output"
QA = OUTPUT / "qa"
TEST_QA = QA / "branch_tests"
VENV_PYTHON = FE / ".venv" / "Scripts" / "python.exe"


BRANCHES = {
    "B1_peak_rr_hrv": {
        "code": "B1",
        "title": "Peak Detection, RR, HRV, and PRSA/BPRSA",
        "tex": "B1_Peak_RR_HRV_Technical_Record.tex",
        "output_keywords": ("rr", "hrv", "neurokit", "zhai", "fusion", "prsa", "method_selection"),
        "docs": tuple(f"feature_extraction/docs/{n}" for n in [
            "01_SYSTEM_ARCHITECTURE.md", "02_EDF_AND_DATA_LAYER.md", "03_RPEAK_DETECTION_AND_AGREEMENT.md",
            "04_RR_HRV_MATHEMATICS_AND_CODE.md", "05_CLI_AND_OUTPUT_ARTIFACTS.md", "06_TESTING_AND_VALIDATION.md",
            "09_COMPLETE_RR_HRV_ARCHITECTURE.md", "10_NEUROKIT_ZHAI_TWO_TRACK_ARCHITECTURE.md",
            "11_DUAL_TRACK_RR_HRV_VARON_MORPHOLOGY.md", "12_ALL48_TRUE_RPEAK_DISTANCE_AUDIT.md",
            "13_UNSW_NEUROKIT_ZHAI_FUSION_ABLATION.md", "14_UNSW_NEUROKIT_CONTEXT_BENCHMARK.md",
            "15_EXPERT_UNSW_HRV_FIDELITY.md", "16_INTERVAL_CONSISTENT_FUSION_AUDIT.md",
            "22_PRSA_BPRSA_CONTEXT_VALIDATION.md",
        ]),
        "legacy": (
            ("feature_extraction/reports/RR_HRV_Cascade_RPeak_Audit_PreIntervalFusion.pdf", "Original exhaustive RR/HRV and detector audit before interval-consistent fusion."),
            ("feature_extraction/reports/RR_HRV_Cascade_Current_Method_Addendum.pdf", "Current-method correction and post-audit addendum."),
            ("feature_extraction/reports/algorithm_explainers/UNSW_Khamis_QRS_Detector_Algorithm_Explainer.pdf", "Literal UNSW/Khamis detector implementation and counterexamples."),
            ("feature_extraction/reports/algorithm_explainers/NeuroKit_Gradient_RPeak_Detector_Algorithm_Explainer.pdf", "Literal NeuroKit gradient-detector implementation and failures."),
            ("feature_extraction/reports/MITBIH_Record_108_RPeak_RR_Reliability_Case_Analysis.pdf", "Record 108 detector/RR reliability case study."),
            ("feature_extraction/reports/MITBIH_Record_113_TWave_Double_Detection_Case_Analysis.pdf", "Record 113 T-wave double-detection case study."),
            ("feature_extraction/reports/MITBIH_Record_207_Morphology_Polarity_Case_Analysis.pdf", "Record 207 polarity and timestamp-failure case study."),
            ("feature_extraction/reports/MITBIH_Record_222_Arrhythmia_Noise_Case_Analysis.pdf", "Record 222 arrhythmia/noise case study."),
            ("feature_extraction/reports/MITBIH_Record_231_Conduction_False_Detection_Case_Analysis.pdf", "Record 231 false-detection/conduction case study."),
        ),
    },
    "B2_morphology": {
        "code": "B2",
        "title": "Beat Morphology",
        "tex": "B2_Morphology_Technical_Record.tex",
        "output_keywords": ("morphology", "varon", "delineation", "fiducial", "patient_template", "prsa", "ui_validation"),
        "docs": tuple(f"feature_extraction/docs/{n}" for n in [
            "06_TESTING_AND_VALIDATION.md", "11_DUAL_TRACK_RR_HRV_VARON_MORPHOLOGY.md",
            "17_MORPHOLOGY_FIDELITY_AND_REVIEWER.md", "18_DELINEATION_GATE_MATCHED_LEAD_V1.md",
            "19_PERSONALIZED_MORPHOLOGY_BRANCH_V1.md", "22_PRSA_BPRSA_CONTEXT_VALIDATION.md",
        ]),
        "legacy": (
            ("feature_extraction/docs/latex_morphology_branch/morphology_branch_complete_technical_record.pdf", "Complete morphology branch technical record, retained intact as the primary historical record."),
            ("feature_extraction/reports/Beat_Types_from_Record_108_Analyzed.pdf", "Detailed record-108 beat-type morphology study."),
            *tuple((f"feature_extraction/reports/beat_type_atlases/MITBIH_Record_{record}_Detailed_Beat_Type_Atlas.pdf", f"Record {record} detailed morphology and beat-type atlas.") for record in (108, 113, 207, 222, 231)),
            *tuple((f"feature_extraction/reports/record_case_reports/MITBIH_Record_{record}_Case_Analysis.pdf", f"Record {record} cross-detector morphology case study.") for record in (108, 113, 207, 222, 231)),
        ),
    },
    "B3_conduction_repolarization": {
        "code": "B3",
        "title": "Conduction and Repolarization",
        "tex": "B3_Conduction_Repolarization_Technical_Record.tex",
        "output_keywords": ("conduction", "delineation", "prominence"),
        "docs": tuple(f"feature_extraction/docs/{n}" for n in [
            "18_DELINEATION_GATE_MATCHED_LEAD_V1.md", "19_PERSONALIZED_MORPHOLOGY_BRANCH_V1.md",
            "20_CONDUCTION_TIMING_BRANCH_V1.md", "21_CONDUCTION_INFORMATION_BRANCH_V2.md",
            "21_SIGNAL_QUALITY_BRANCH_V0.md",
        ]),
        "legacy": (
            ("output/evidence/artifacts/patient_specific_conduction_repolarization_review.pdf", "Patient-specific conduction/repolarization literature and architecture review."),
            ("feature_extraction/reports/MITBIH_Record_113_TWave_Double_Detection_Case_Analysis.pdf", "Record 113 T-wave boundary/double-detection case study."),
            ("feature_extraction/reports/MITBIH_Record_231_Conduction_False_Detection_Case_Analysis.pdf", "Record 231 conduction false-detection case study."),
            ("feature_extraction/reports/record_case_reports/MITBIH_Record_113_Case_Analysis.pdf", "Record 113 full cascade case analysis."),
            ("feature_extraction/reports/record_case_reports/MITBIH_Record_207_Case_Analysis.pdf", "Record 207 full cascade case analysis."),
            ("feature_extraction/reports/record_case_reports/MITBIH_Record_231_Case_Analysis.pdf", "Record 231 full cascade case analysis."),
            *tuple((f"feature_extraction/reports/beat_type_atlases/MITBIH_Record_{record}_Detailed_Beat_Type_Atlas.pdf", f"Record {record} beat-type context relevant to landmark and interval failure.") for record in (113, 207, 231)),
        ),
    },
    "B4_signal_quality": {
        "code": "B4",
        "title": "Signal Quality",
        "tex": "B4_Signal_Quality_Technical_Record.tex",
        "output_keywords": ("signal_quality", "artifact_degradation", "nstdb", "ui_validation"),
        "docs": tuple(f"feature_extraction/docs/{n}" for n in [
            "03_RPEAK_DETECTION_AND_AGREEMENT.md", "06_TESTING_AND_VALIDATION.md",
            "17_MORPHOLOGY_FIDELITY_AND_REVIEWER.md", "21_SIGNAL_QUALITY_BRANCH_V0.md",
            "22_PRSA_BPRSA_CONTEXT_VALIDATION.md",
        ]),
        "legacy": (
            ("output/evidence/artifacts/ECG_Signal_Quality_Branch_Architecture_and_Evidence_Foundation_v2.pdf", "Signal-quality branch architecture and evidence foundation."),
            ("feature_extraction/reports/Signal_Quality_as_Context_Evidence_Review.pdf", "Current signal-quality-as-context decision review."),
            ("output/evidence/artifacts/ECG_Wavelet_Artifact_Literature_Review.pdf", "Wavelet/artifact literature review and held design alternatives."),
            ("feature_extraction/reports/MITBIH_Record_108_RPeak_RR_Reliability_Case_Analysis.pdf", "Record 108 reliability and quality-context case study."),
            ("feature_extraction/reports/MITBIH_Record_207_Morphology_Polarity_Case_Analysis.pdf", "Record 207 polarity/context case study."),
            ("feature_extraction/reports/MITBIH_Record_222_Arrhythmia_Noise_Case_Analysis.pdf", "Record 222 arrhythmia/noise case study."),
        ),
    },
}


ARCHITECTURE = {
    "B1": [
        ("Data and channel contract", "implemented", "EDF/WFDB signal, fs, physical units, patient/record/lead and interval provenance", "Validated samples plus explicit channel metadata", "Missing/ambiguous channel or nonmonotone time is rejected."),
        ("UNSW/Khamis primary lane", "implemented control", "Single-lead ECG and frozen detector configuration", "Primary QRS/R samples with diagnostic intermediates", "High pooled beat F1 does not make every timestamp exact."),
        ("NeuroKit and Zhai lanes", "implemented comparators", "Same samples and independent configurations", "Candidate events, offsets, associations, Zhai correlation", "Misses/extras and morphology dependence remain visible."),
        ("RR construction", "implemented", "One declared timestamp owner and monotone consecutive anchors", "RR ms, HR bpm, endpoint provenance and support", "Mixed-source endpoints are never silently accepted."),
        ("Jeppesen HRV", "implemented", "Causal seven-RR median; 100-RR trailing history", "J1/J2 and continuous HRV arrays at ending R time", "Warm-up remains missing; finite value is separate from reliability."),
        ("PRSA/BPRSA", "implemented context", "80 RR intervals plus ECG R amplitudes for BPRSA", "Six measurements, curves, anchors, definition/reliability/age flags", "No signal-quality attachment or model eligibility is implied."),
        ("Interval-consistent fusion", "experimental held", "Uniquely associated pure candidate intervals", "Audited challenger RR/HRV lanes", "Catastrophic record failures and transition error prevent promotion."),
        ("Seizure decision", "not implemented", "Would require patient-disjoint continuous labeled ECG", "None", "Proof runs and overlapping-window AUC are not seizure validation."),
    ],
    "B2": [
        ("Anchor and crop contract", "implemented", "ECG, fs, ordered R anchors, lead and detector provenance", "Fixed or calibrated beat crops with boundary flags", "Incomplete edge crops and ambiguous anchors remain unavailable."),
        ("Varon-2015 control", "implemented control", "Five consecutive fixed 120-ms R-centered crops", "Uncentered Gram matrix and five raw eigenvalues", "Faithful replication is not complete-QRS capture or seizure accuracy."),
        ("Detector-track fidelity", "implemented audit", "Expert/UNSW/NeuroKit/Zhai five-beat sequences", "Coverage, timing and eigenvalue-fidelity tables", "Conditional low error cannot hide misses/extras."),
        ("Personalized symmetric/asymmetric crops", "experimental", "Earlier same-record manual QRS triplets", "Calibrated pre/post samples and held-out capture", "Calibration size and tail contamination prevent production promotion."),
        ("Whole-cycle template bank", "experimental", "Patient/lead calibration cycles", "Correlation, residuals, beat change, constrained derivative-DTW", "No seizure discrimination or prospective generalization."),
        ("Prominence P/QRS/T lane", "implemented experimental", "ECG and R anchors", "Landmarks, intervals, amplitudes, areas, order/missingness flags", "Endpoint tails exceed expected variability in some strata."),
        ("Reviewer", "implemented development tool", "WFDB/EDF record and optional expert annotations", "Interactive tracks, TP/FP/FN and morphology panels", "Display is not a validated clinical user interface."),
    ],
    "B3": [
        ("Landmark input gate", "experimental prerequisite", "Aligned P/Q/R/S/T peaks and boundaries with lead/provenance", "Per-landmark availability and physiological order", "Automatic endpoint accuracy gate is not passed."),
        ("Timing v1", "implemented standalone", "Finalized landmarks and preceding RR", "PR, QRS, QT, JT, QTc and 30-beat variability", "Not streaming-validated and not called by pipeline.py."),
        ("Diab dynamics", "implemented information", "Current and previous P/Q/R/S/T peaks", "Four within-beat and five interbeat timing features", "No aggregation is claimed to reproduce the published classifier."),
        ("Five-beat summaries", "implemented information", "Five eligible beat measurements", "Robust/ordinary spread, change and nonoverlapping-window shift", "No universal bad-change threshold."),
        ("Nine-beat QT context", "implemented information", "Nine paired QT and preceding RR values", "Mean QT/RR and named QT corrections", "Physiology context only; not a seizure detector."),
        ("External quality eligibility", "implemented interface", "Optional caller-owned Boolean eligibility", "Excluded-from-summary but visible beat rows", "B3 never invents or reinterprets artifact status."),
        ("Pipeline/model integration", "not implemented", "Would require passed landmark and patient validation gates", "None", "All final-matrix/probability/diagnostic flags remain false."),
    ],
    "B4": [
        ("Window contract", "implemented", "ECG window, fs, uV conversion and exact support interval", "One deterministic measurement row plus provenance", "Nonfinite/short input is explicit, not silently repaired."),
        ("Unconditional SQIs", "implemented", "Finite waveform and physical scale", "Finite/flat/amplitude/spectral/moment measurements", "Device-transfer choices are recorded."),
        ("Conditional SQIs", "implemented", "Optional rails, paired detectors, cross-lead peaks, QRS onsets, groups, mains", "Rail/agreement/template/baseline/mains/residual fields and availability reasons", "Prerequisites are never synthesized."),
        ("Trailing execution", "implemented causal wrapper", "Recording plus trailing window endpoints", "Rows timestamped at exclusive support end", "Future samples/events are not read."),
        ("NSTDB degradation", "experimental validation", "Clean/noisy paired records across SNR", "Feature-versus-SNR behavior", "Controlled degradation is not artifact-classifier performance."),
        ("Artifact classifier/calibration", "proposed", "Would require manually labeled artifact types and patient-wise splits", "No current probability or deletion policy", "No threshold is learned from NSTDB and transferred clinically."),
    ],
}


CONTRACTS = {
    "B1": [
        ("Peak tracks", "ecg[float], fs, detector configuration", "integer sample indices; times; per-event diagnostics", "Sorted unique in-range events; segment offset and orientation retained."),
        ("Reliability", "primary and secondary events; tolerance; window support", "one-to-one matches, offsets, qSQI/bSQI context, RR support and reasons", "Agreement is context, not truth or artifact probability."),
        ("RR/HR", "one monotone anchor sequence and fs", "RR ms and HR bpm aligned to later anchor", "Both endpoint sources recorded; nonpositive intervals rejected."),
        ("Jeppesen arrays", "RR ms, optional end times, window=100, median width=7", "mean/std/RMSSD-like values, SD1/SD2, CSI/ModCSI, J1/J2 and slope", "Rows before required history are NaN with coverage metadata."),
        ("PRSA/BPRSA", "RR, R amplitudes, 80-interval history and reliability context", "curves, SDNN80, slopes, anchors, usable/reliable flags", "BPRSA requires stable lead/polarity and separate quality context."),
        ("Persisted outputs", "record/lead/provenance plus branch arrays", "CSV/JSON/plots and clinical-demo context tables", "Schemas preserve missingness, age and nonprediction flags."),
    ],
    "B2": [
        ("Fixed Varon", "ECG, fs, consecutive R anchors, 120-ms nominal width", "five crops, Gram matrix and ordered raw eigenvalues", "Five complete consecutive crops required; uncentered Gram is intentional."),
        ("Calibrated crop", "earlier manual onset/R/offset triplets", "symmetric or separate pre/post sample counts and fallback reason", "Evaluation must use later beats from same record/lead."),
        ("Whole-cycle template", "calibration cycles and later complete cycles", "template identity, correlation, raw/normalized residual, change and DTW", "No cross-patient template reuse without explicit provenance."),
        ("Prominence morphology", "ECG, fs and R anchors", "P/Q/R/S/T peaks/boundaries, intervals, amplitudes, area and flags", "Missing or nonphysiological order stays explicit."),
        ("Fidelity audit", "candidate and expert event streams plus expert symbols", "beat F1/timing, strict five-beat coverage and feature errors", "Only one-to-one consecutive five-beat windows are comparable."),
        ("Reviewer", "record, optional atr/q1c/manual annotations", "interactive visual audit and region summaries", "UI output is evidence tooling, not a clinical decision."),
    ],
    "B3": [
        ("Timing beats", "ordered finalized landmarks, RR and fs", "PR/QRS/QT/JT/TpTe and named QTc columns", "Each endpoint has availability; missing never becomes zero."),
        ("Timing rolling", "30 finalized beat rows", "SD/RMSSD/STV/QTVI-style context", "Thirty-beat history and zero-variance rules are explicit."),
        ("Information beats", "current/previous P/Q/R/S/T peaks", "nine raw Diab timing dynamics plus interval measurements", "No classifier or threshold column is emitted."),
        ("Windows5 long", "five eligible beats and preceding nonoverlapping window", "mean/median/SD/MAD/IQR/range/RMSSD/slope/shifts/effect size", "Availability fraction and support state accompany every row."),
        ("QT context9", "nine paired QT/RR values", "mean QT/RR and Bazett/Hodges/Fridericia/Framingham context", "All nine pairs required."),
        ("External eligibility", "optional Boolean from B4", "summary inclusion mask and preserved raw beats", "Absence is recorded as no artifact gate applied."),
    ],
    "B4": [
        ("Core window", "samples, fs, uv_per_input_unit", "finite, flat, amplitude, spectral and moment SQIs", "Physical scaling is mandatory for amplitude-valued fields."),
        ("Rails", "both ADC rails", "rail fraction and longest dilated run", "Unavailable without both rails."),
        ("Detector context", "two event tracks and declared tolerances", "bSQI/qSQI/RR-support and offsets", "Detector agreement is not calibrated correctness."),
        ("Cross-lead", "same-detector events on two leads", "max Jaccard and unmatched fractions", "Unavailable for single-lead input."),
        ("Template/baseline/mains", "R peaks, QRS onsets, mains frequency and complete beats", "template correlation and fitted baseline/mains RMS", "Conditional fields expose exact missing prerequisites."),
        ("Residual/Menon", "externally supplied morphology groups or converged representative beat", "dominant-group residual and Fourier score", "Comparator outputs are not hard vetoes."),
    ],
}


ITERATIONS = {
    "B1": [
        ("UNSW-only primary control", "kept", "Pooled MIT-BIH TP/FP/FN 109244/350/250 and F1 0.997261; full coverage.", "Matched timing MAE 8.705 ms and record-specific morphology/timing failures.", "Retained as the safest single timestamp owner."),
        ("NeuroKit and Zhai standalone tracks", "kept as comparators", "NeuroKit matched timing median 0 ms; Zhai F1 0.98564 with fewer false positives than NeuroKit.", "Both have materially lower event coverage and severe record counterexamples.", "Useful independent evidence, not universal owners."),
        ("Per-beat NeuroKit-then-Zhai substitution", "rejected", "Reduced pooled RR MAE from 3.971 to 3.382 ms.", "Source-transition RR MAE reached 23.535 ms; F1 fell and record 200 deteriorated.", "Mixed endpoint ownership is unsafe for HRV."),
        ("UNSW timestamps plus agreement context", "implemented baseline", "At 50 ms, supported coverage 93.79% and supported RR MAE 3.819 ms.", "The published 150-ms tolerance selected slightly worse supported intervals.", "Context retained; no hard gate or probability."),
        ("Interval-consistent fusion", "held experimental", "Combined interval RR MAE 3.182 ms; zero mixed-endpoint intervals.", "Transitions still have large error; Zhai adds 0.0029-ms benefit and worsens J2 threshold behavior; severe record tails remain.", "Offline challenger only."),
        ("Expert-versus-UNSW HRV fidelity", "qualified pass", "Pooled CCC is high and q95 threshold F1 is about 0.86.", "Records 107/104/114/231 have poor threshold preservation; agreement hard veto loses expert-high windows.", "Requires patient-specific untouched test periods."),
        ("PRSA/BPRSA equations and causal matrix", "software pass", "105654 rows; 211308 probes; zero value/time/flag mismatches, future leaks, negative ages or core-gate changes.", "No clinical truth value or seizure endpoint exists.", "Retain as context with full provenance."),
        ("PRSA/BPRSA noise and rhythm audit", "qualified/negative", "RR-only PRSA was exactly invariant with fixed R timestamps.", "BPRSA correlation fell 0.8100 to 0.0315 from +24 to -6 dB; all 946 cubic failures were in non-normal windows.", "BPRSA needs B4 quality plus lead/polarity/rhythm context."),
        ("CHB04_28 seizure proof runs", "clinically unevaluable", "Unfiltered overlapping-window AUC reached 0.991/0.886 for selected features.", "Zero strict reliable ictal windows (0/118 and 0/146), one patient and overlapping windows.", "No seizure-accuracy claim."),
    ],
    "B2": [
        ("Fixed 120-ms Varon replication", "kept control", "Exact published five-beat uncentered Gram/eigenvalue lane is implemented and tested.", "Only 71.5% of 2279 manual-QRS beats are fully captured.", "Faithful control, not universal complete-QRS representation."),
        ("Three detector-track fidelity audit", "qualified pass", "UNSW strict five-beat coverage 0.98708; NeuroKit/Zhai conditional raw errors 0.00707/0.00825.", "NeuroKit/Zhai coverage 0.93424/0.92258 and record failures; UNSW conditional error larger.", "Expose the coverage-versus-fidelity trade-off."),
        ("NeuroKit CWT delineation", "rejected", "Conditionally precise returned boundaries.", "Unequal arrays make check=True fail; QRS onset/offset sensitivity as low as 24.9/34.6% on QTDB.", "Not integrated."),
        ("NeuroKit DWT delineation", "held baseline", "Higher coverage and small median errors for several landmarks.", "QRS-onset tails up to 124/128 ms; polarity subgroup sensitivity fell to 75.8%; T endpoints incomplete.", "Does not pass complete-QRS/QT gate."),
        ("Patient-average total-width crop", "negative result", "Exact proposal was implemented reproducibly.", "Held-out capture only about 16-28%, far below fixed control because R is not centered.", "Must not replace the control."),
        ("Symmetric required-width p95", "experimental", "Five-beat calibration capture rose to 83.1% LUDB and 79.1% QTDB.", "Three beats were insufficient; maximum windows had 184-232-ms P95 tails.", "p95 retained as challenger; maximum is sensitivity analysis."),
        ("Separate asymmetric p95 crop", "preferred experimental", "Independent pre/post quantiles correct the centered-window design error and execute end to end.", "Still calibration-limited and not seizure-tested.", "Keep separate from fixed replication lane."),
        ("Whole-cycle template and prominence lanes", "implemented experimental", "Emit interpretable cycle similarity, residual, DTW and P/QRS/T measurements.", "Landmark tails can exceed physiological beat-to-beat variation; no patient-generalization evidence.", "Remain measurement challengers."),
        ("Interactive reviewer and case atlases", "implemented evidence tool", "Makes false positives, misses, polarity, beat symbols and morphology changes inspectable.", "Not a validated clinical UI and cannot supply missing ground truth.", "Retain for manual error audits."),
    ],
    "B3": [
        ("DWT/CWT landmark gate", "failed acceptance", "Some landmarks have good medians and DWT provides useful baseline coverage.", "Boundary sensitivity/tails and CWT missingness prevent complete-QRS/QT use.", "No automatic delineator promoted."),
        ("Prominence landmark challenger", "implemented experimental", "LUDB medians: P onset 8 ms, QRS onset 8 ms, QRS offset 14 ms, T offset 18 ms.", "QTDB P/T tails reach 68.2/108 ms and can dominate true variability.", "Requires interval-level paired manual validation."),
        ("Timing v1", "software pass", "Hand equations, missingness, 30-beat history and causality tests pass; MIT100 smoke produced 74 beat rows.", "No manual interval truth in the smoke run; final QT SD30 43 ms is uninterpretable.", "Standalone only, not pipeline integrated."),
        ("Diab nine timing dynamics", "implemented information", "Follows explicit four within-beat plus five interbeat equations.", "Does not reproduce the paper's classifier and paper feature count is internally inconsistent.", "Raw ordered dynamics are retained."),
        ("Five-beat robust summaries", "implemented information", "Produces auditable mean/median/spread/slope and nonoverlapping shifts.", "A five-beat change is confounded by rate, ectopy, lead, quality and landmark error.", "No bad-change flag."),
        ("Nine-beat QT/RR context", "implemented information", "Implements published nine-beat averaging and named QT corrections.", "Not a seizure detector and not a short-window QTV replacement.", "Context only."),
        ("External quality eligibility", "implemented interface", "Consumes caller-owned eligibility without deleting raw beat rows.", "No accepted B4 classifier currently supplies a final production gate.", "No gate is recorded when absent."),
        ("MIT100 v2 smoke", "execution pass", "74 beats, 70 complete five-beat endpoints, 73 beats with all Diab dynamics.", "No manual P/Q/S/T truth, seizure labels, classifier or final-matrix integration.", "Availability diagnostic only."),
    ],
    "B4": [
        ("Paper-named deterministic SQIs", "implemented", "Time, spectral, statistical, detector, template, baseline, mains and residual equations are unit-tested.", "Some source papers do not freeze all software transfer choices.", "Emit measurements and provenance, not labels."),
        ("Conditional prerequisite contract", "implemented", "Rails/detector pairs/leads/onsets/groups are consumed when supplied.", "Unavailable prerequisites cannot be inferred safely.", "Every conditional output has availability and reason."),
        ("NSTDB controlled degradation", "qualified pass", "baSQI rho 0.975; qSQI 0.829; bSQI 0.789 with improving SNR; amplitude/HF behave directionally.", "Menon score rho -0.127 and no manually labeled artifact classes.", "Useful stress test, not classifier validation."),
        ("Detector agreement as quality", "context only", "Agreement enriches some RR subsets and is reproducible.", "150-ms support can select worse intervals; agreement misses amplitude corruption and genuine arrhythmia.", "Never call qSQI an artifact probability."),
        ("Wavelet energy/entropy candidates", "held", "Literature supports multiple potentially useful representations.", "Definitions, scaling, levels and signal domains are not interchangeable or frozen.", "Not emitted in v0."),
        ("Amplitude-sensitive BPRSA interaction", "negative/required context", "B4 can supply independent amplitude/noise context.", "BPRSA numerical reliability alone stays true while waveform fidelity collapses under noise.", "Model eligibility must combine explicit B4 context later."),
        ("Artifact classifier and calibration", "not implemented", "Architecture and candidate evidence are documented.", "No patient-wise labeled BUT-QDB experiment, calibration, probability or deletion policy.", "Future work only."),
    ],
}


KEY_RESULTS = {
    "B1": [
        ("MIT-BIH UNSW control", "109494 expert beats; F1 0.997261; RR MAE 3.971 ms", "Strong event control with nontrivial timestamp and record tails."),
        ("NeuroKit/Zhai standalone", "F1 0.98033/0.98564; matched RR MAE 3.31/3.83 ms", "Lower conditional timing error does not compensate for missing/excess events globally."),
        ("Morphology-sensitive records", "108, 113, 200, 207, 222, 231 and 233 expose distinct failures", "Pooled means cannot select one universal track."),
        ("PRSA matrix audit", "105654 feature rows; 211308 probes; zero causal/value/gate errors", "Software join is validated; clinical usefulness is not."),
        ("Noise fidelity", "BPRSA correlation 0.8100 at +24 dB to 0.0315 at -6 dB", "Separate signal-quality and lead/polarity context are mandatory."),
    ],
    "B2": [
        ("All-48 fidelity", "UNSW/NeuroKit/Zhai strict coverage 0.98708/0.93424/0.92258", "Timestamp choice trades continuity against conditional morphology fidelity."),
        ("Fixed crop capture", "1629 of 2279 complete QRS beats, 71.5%", "The published fixed crop remains a replication control only."),
        ("Matched-lead gate", "Neither DWT nor CWT passes complete QRS acceptance", "No automatic boundary method enters the production morphology control."),
        ("Personal calibration", "p95 symmetric five-beat capture 83.1% LUDB, 79.1% QTDB", "Improvement is experimental and calibration-limited."),
        ("PRSA/BPRSA adjacency", "RR-only PRSA stable; amplitude-driven BPRSA lead/noise sensitive", "Autonomic lane is context beside morphology, not part of its core gate."),
    ],
    "B3": [
        ("Prominence benchmark", "LUDB medians 8/8/14/18 ms for P/QRSon/QRSoff/Toff", "Error tails and QTDB degradation prevent variability promotion."),
        ("Timing v1 smoke", "74 beats; QT range 286.1-502.8 ms; final QT SD30 43.0 ms", "Without manual truth the spread cannot be interpreted physiologically."),
        ("Information v2 smoke", "74 beats; 70 five-beat endpoints; 73 complete Diab rows", "Software availability only."),
        ("Integration flags", "Final matrix, seizure probability, clinical interpretation and abnormality label all false", "Non-integration is an enforced contract, not missing documentation."),
    ],
    "B4": [
        ("NSTDB monotonicity", "baSQI/qSQI/bSQI rho 0.975/0.829/0.789 with improving SNR", "Several measurements react usefully to controlled noise."),
        ("Weak comparator", "Menon Fourier score rho -0.127", "Do not use as a hard veto."),
        ("Scale-sensitive measures", "Amplitude and HF RMS rho -0.961/-0.848 with improving SNR", "Physical-unit and device provenance are essential."),
        ("Classification status", "No trained artifact classifier, calibrated probability or clinical threshold", "B4 currently supplies context measurements only."),
    ],
}


def esc(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
        "|": r"\textbar{}", "<": r"\textless{}", ">": r"\textgreater{}",
        "±": r"+/-", "–": "--", "—": "---", "≤": r"$\leq$", "≥": r"$\geq$",
        "µ": r"$\mu$", "→": r"$\rightarrow$", "×": r"$\times$",
    }
    return "".join(replacements.get(char, char) for char in text)


def ptex(path: str | Path) -> str:
    value = Path(path).as_posix() if isinstance(path, Path) else str(path).replace("\\", "/")
    return rf"\path{{{value}}}"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def source_map(branch: str) -> list[dict[str, str]]:
    path = OUTPUT / "branches" / branch / "source_map.csv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def all_first_party_python() -> list[str]:
    files: list[str] = []
    for base in (FE / "src", FE / "scripts", FE / "tests"):
        for path in base.rglob("*.py"):
            if "__pycache__" not in path.parts:
                files.append(path.relative_to(ROOT).as_posix())
    return sorted(files)


def symbol_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for row in rows:
        if row["role"] != "production":
            continue
        path = ROOT / row["relative_path"]
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(text)
        except Exception as exc:
            result.append({"path": row["relative_path"], "symbol": "<parse error>", "signature": "", "description": str(exc), "line": 0})
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                result.append({
                    "path": row["relative_path"], "symbol": node.name,
                    "signature": f"{node.name}({ast.unparse(node.args)})",
                    "description": (ast.get_docstring(node) or "No function docstring; consult implementation and tests.").splitlines()[0],
                    "line": node.lineno,
                })
            elif isinstance(node, ast.ClassDef):
                result.append({"path": row["relative_path"], "symbol": node.name, "signature": f"class {node.name}", "description": (ast.get_docstring(node) or "Class definition.").splitlines()[0], "line": node.lineno})
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        result.append({
                            "path": row["relative_path"], "symbol": f"{node.name}.{child.name}",
                            "signature": f"{node.name}.{child.name}({ast.unparse(child.args)})",
                            "description": (ast.get_docstring(child) or "No method docstring; consult implementation and tests.").splitlines()[0],
                            "line": child.lineno,
                        })
        if not any(item["path"] == row["relative_path"] for item in result):
            result.append({"path": row["relative_path"], "symbol": "<module constants/import surface>", "signature": "", "description": "No top-level function or class definitions.", "line": 1})
    return result


def test_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in rows:
        if row["role"] != "test" or row["relative_path"] in seen:
            continue
        seen.add(row["relative_path"])
        path = ROOT / row["relative_path"]
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                result.append({"path": row["relative_path"], "node_id": f"{path.name}::{node.name}", "line": node.lineno, "purpose": (ast.get_docstring(node) or node.name.replace("test_", "").replace("_", " ")).splitlines()[0]})
            elif isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name.startswith("test_"):
                        result.append({"path": row["relative_path"], "node_id": f"{path.name}::{node.name}::{child.name}", "line": child.lineno, "purpose": (ast.get_docstring(child) or child.name.replace("test_", "").replace("_", " ")).splitlines()[0]})
    return result


def run_tests(branch: str, rows: list[dict[str, str]]) -> dict[str, object]:
    TEST_QA.mkdir(parents=True, exist_ok=True)
    tests = sorted({str((ROOT / row["relative_path"]).relative_to(FE)) for row in rows if row["role"] == "test"})
    command = [str(VENV_PYTHON), "-m", "pytest", "-vv", "--tb=short", *tests]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=FE, capture_output=True, text=True, encoding="utf-8", errors="replace")
    elapsed = time.perf_counter() - started
    log_path = TEST_QA / f"{BRANCHES[branch]['code']}_pytest_verbose.txt"
    log_path.write_text(completed.stdout + ("\nSTDERR\n" + completed.stderr if completed.stderr else ""), encoding="utf-8")
    passed_match = re.search(r"(\d+) passed", completed.stdout)
    failed_match = re.search(r"(\d+) failed", completed.stdout)
    return {
        "branch": BRANCHES[branch]["code"], "command": command, "test_files": tests,
        "exit_code": completed.returncode, "elapsed_seconds": round(elapsed, 3),
        "passed": int(passed_match.group(1)) if passed_match else 0,
        "failed": int(failed_match.group(1)) if failed_match else 0,
        "log_path": log_path.relative_to(ROOT).as_posix(),
    }


def run_full_tests() -> dict[str, object]:
    command = [str(VENV_PYTHON), "-m", "pytest", "-vv", "--tb=short"]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=FE, capture_output=True, text=True, encoding="utf-8", errors="replace")
    elapsed = time.perf_counter() - started
    path = TEST_QA / "full_suite_pytest_verbose.txt"
    path.write_text(completed.stdout + ("\nSTDERR\n" + completed.stderr if completed.stderr else ""), encoding="utf-8")
    passed_match = re.search(r"(\d+) passed", completed.stdout)
    failed_match = re.search(r"(\d+) failed", completed.stdout)
    return {"command": command, "exit_code": completed.returncode, "elapsed_seconds": round(elapsed, 3), "passed": int(passed_match.group(1)) if passed_match else 0, "failed": int(failed_match.group(1)) if failed_match else 0, "log_path": path.relative_to(ROOT).as_posix()}


def result_directories(keywords: tuple[str, ...]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    output_root = FE / "outputs"
    for directory in sorted((p for p in output_root.iterdir() if p.is_dir()), key=lambda p: p.name.lower()):
        if not any(keyword in directory.name.lower() for keyword in keywords):
            continue
        files = [p for p in directory.rglob("*") if p.is_file()]
        key_files = [p.name for p in files if p.suffix.lower() in {".csv", ".json", ".png", ".pdf"}][:12]
        rows.append({
            "directory": directory.relative_to(ROOT).as_posix(), "file_count": len(files),
            "bytes": sum(p.stat().st_size for p in files),
            "last_modified": datetime.fromtimestamp(max((p.stat().st_mtime for p in files), default=directory.stat().st_mtime)).isoformat(timespec="seconds"),
            "key_files": "; ".join(key_files),
        })
    return rows


def result_file_rows(directories: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for entry in directories:
        directory = ROOT / str(entry["directory"])
        for path in sorted(p for p in directory.rglob("*") if p.is_file()):
            rows.append({
                "directory": entry["directory"], "relative_file": path.relative_to(directory).as_posix(),
                "extension": path.suffix.lower(), "bytes": path.stat().st_size,
                "last_modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                "sha256": sha256(path),
            })
    return rows


def evidence_rows(code: str) -> list[dict[str, object]]:
    docs_path = OUTPUT / "evidence" / "indexes" / "document_manifest.csv"
    branch_path = OUTPUT / "evidence" / "indexes" / "branch_evidence_map.csv"
    with docs_path.open(newline="", encoding="utf-8-sig") as handle:
        docs = {row["sha256"]: row for row in csv.DictReader(handle)}
    with branch_path.open(newline="", encoding="utf-8-sig") as handle:
        branches = list(csv.DictReader(handle))
    result = []
    for row in branches:
        if code not in row.get("branch_tags", "").split("|"):
            continue
        doc = docs.get(row["sha256"])
        if not doc:
            continue
        result.append({
            "sha256": row["sha256"], "category": doc["category"], "page_count": doc["page_count"],
            "source_status": doc["source_status"], "canonical_output_path": doc["canonical_output_path"],
            "branch_tags": row["branch_tags"], "mapping_reason": row["mapping_reason"],
        })
    return sorted(result, key=lambda item: (str(item["category"]), str(item["canonical_output_path"])))


def doc_rows(paths: tuple[str, ...], legacy: tuple[tuple[str, str], ...]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for raw, role in [*((p, "maintained design/history record") for p in paths), *legacy]:
        path = ROOT / raw
        rows.append({
            "path": raw, "exists": path.exists(), "role": role,
            "bytes": path.stat().st_size if path.exists() else 0,
            "last_modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds") if path.exists() else "",
            "sha256": sha256(path) if path.exists() else "",
        })
    return rows


def page_count(path: Path) -> int:
    if PdfReader is None:
        return 0
    try:
        return len(PdfReader(str(path), strict=False).pages)
    except Exception:
        return 0


def longtable(headers: list[str], widths: list[float], rows: list[list[str]], caption: str = "") -> str:
    columns = "".join(rf">{{\raggedright\arraybackslash}}p{{{width:.3f}\linewidth}}" for width in widths)
    head = " & ".join(rf"\textbf{{{esc(value)}}}" for value in headers) + r" \\ \midrule"
    body = "\n".join(" & ".join(values) + r" \\" for values in rows)
    label = rf"\caption{{{esc(caption)}}}\\" if caption else ""
    return rf"""\begin{{longtable}}{{{columns}}}
{label}\toprule
{head}
\endfirsthead
\toprule
{head}
\endhead
{body}
\bottomrule
\end{{longtable}}
"""


def build_tex(branch: str, spec: dict[str, object], rows: list[dict[str, str]], symbols: list[dict[str, object]], tests: list[dict[str, object]], test_report: dict[str, object], full_report: dict[str, object], results: list[dict[str, object]], evidence: list[dict[str, object]], docs: list[dict[str, object]]) -> str:
    code = str(spec["code"])
    inventory_table = longtable(
        ["Role", "Maintained path", "Shared", "LOC", "SHA-256 prefix"], [0.09, 0.54, 0.07, 0.07, 0.13],
        [[esc(row["role"]), ptex(row["relative_path"]), esc(row["shared"]), esc(sum(1 for _ in (ROOT / row["relative_path"]).open(encoding="utf-8", errors="replace"))), esc(sha256(ROOT / row["relative_path"])[:16])] for row in rows],
        "Complete branch source inventory. Shared files appear in every branch that consumes them.",
    )
    arch_table = longtable(
        ["Stage", "Status", "Exact input", "Output", "Failure/non-claim"], [0.13, 0.11, 0.24, 0.22, 0.22],
        [[esc(cell) for cell in item] for item in ARCHITECTURE[code]], "Operational architecture and branch boundary.",
    )
    contract_table = longtable(
        ["Interface", "Required input", "Returned/persisted output", "Availability and semantics"], [0.14, 0.25, 0.27, 0.26],
        [[esc(cell) for cell in item] for item in CONTRACTS[code]], "Exact branch input/output contracts.",
    )
    result_table = longtable(
        ["Evidence", "Observed result", "What it supports"], [0.19, 0.36, 0.37],
        [[esc(cell) for cell in item] for item in KEY_RESULTS[code]], "Current quantitative evidence boundary.",
    )
    iteration_table = longtable(
        ["Iteration", "Disposition", "What worked", "What failed", "Decision/why"], [0.14, 0.10, 0.24, 0.24, 0.20],
        [[esc(cell) for cell in item] for item in ITERATIONS[code]], "Chronological design decisions, including negative results.",
    )
    symbol_table = longtable(
        ["Module:line", "Top-level symbol/signature", "Contract note"], [0.30, 0.35, 0.27],
        [[ptex(f"{item['path']}:{item['line']}"), ptex(item["signature"] or item["symbol"]), esc(item["description"])] for item in symbols],
        "Every top-level production function/class and class method in the branch map.",
    )
    test_table = longtable(
        ["Test source:line", "Test definition", "Purpose"], [0.34, 0.29, 0.29],
        [[ptex(f"{item['path']}:{item['line']}"), ptex(item["node_id"]), esc(item["purpose"])] for item in tests],
        "Every discovered branch test definition. Parametrized invocations appear in the verbatim pytest log.",
    )
    result_dir_table = longtable(
        ["Saved output directory", "Files", "Last modified", "Representative artifacts"], [0.36, 0.07, 0.16, 0.33],
        [[ptex(item["directory"]), esc(item["file_count"]), esc(item["last_modified"]), esc(item["key_files"] or "No CSV/JSON/PNG/PDF file") ] for item in results],
        "All branch-matched saved result directories; the companion CSV inventories every file and hash.",
    )
    docs_table = longtable(
        ["Historical/design source", "Role", "Modified", "SHA-256 prefix"], [0.43, 0.27, 0.16, 0.10],
        [[ptex(item["path"]), esc(item["role"]), esc(item["last_modified"]), esc(str(item["sha256"])[:12])] for item in docs],
        "Folder-by-folder maintained documentation, selected reports and case studies.",
    )
    evidence_counts = Counter(str(item["category"]) for item in evidence)
    evidence_table = longtable(
        ["Category/pages", "Canonical evidence path", "Tags/reason"], [0.14, 0.53, 0.25],
        [[esc(f"{item['category']} / {item['page_count']}"), ptex(item["canonical_output_path"]), esc(f"{item['branch_tags']}: {item['mapping_reason']}")] for item in evidence],
        "Every canonical PDF content group tagged to this branch. External papers are referenced, not republished.",
    )
    legacy_intro_rows = []
    legacy_includes = []
    for index, (raw, reason) in enumerate(spec["legacy"], start=1):
        path = ROOT / str(raw)
        if not path.exists():
            legacy_intro_rows.append([esc(index), ptex(str(raw)), esc("MISSING - not included"), esc(reason)])
            continue
        pages = page_count(path)
        legacy_intro_rows.append([esc(index), ptex(str(raw)), esc(f"{pages} pages; {sha256(path)[:16]}"), esc(reason)])
        legacy_includes.append(rf"\includepdf[pages=-,pagecommand={{\thispagestyle{{plain}}}}]{{{path.resolve().as_posix()}}}")
    legacy_table = longtable(["No.", "Appended project-authored record", "Identity", "Why selected"], [0.04, 0.45, 0.15, 0.28], legacy_intro_rows, "Legacy reports and case studies appended intact after the current dossier.")
    category_summary = ", ".join(f"{key}: {value}" for key, value in sorted(evidence_counts.items()))
    body = rf"""
\section{{Purpose, completeness rule, and status}}
This is the exhaustive {esc(code)} dossier, not the earlier short summary. It reconciles current code and tests with maintained Markdown design records, every branch-matched saved result directory, the canonical evidence manifest, and selected project-authored legacy reports/case studies. The appendices preserve the old records verbatim; the front matter states the current decision when an old report has been superseded.

\textbf{{Clinical boundary.}} This research implementation does not produce a validated seizure probability, diagnosis, treatment recommendation, artifact probability, or clinical abnormality label. ``Passed'' below means software/equation/declared benchmark execution, never clinical validation unless explicitly stated.

\begin{{itemize}}
\item Source-map rows: {len(rows)}; production symbols/methods: {len(symbols)}; test definitions: {len(tests)}.
\item Branch pytest run: {test_report['passed']} passed, {test_report['failed']} failed, exit {test_report['exit_code']}, {test_report['elapsed_seconds']} seconds.
\item Full repository run: {full_report['passed']} passed, {full_report['failed']} failed, exit {full_report['exit_code']}, {full_report['elapsed_seconds']} seconds.
\item Saved result directories: {len(results)}; canonical branch-tagged PDF content groups: {len(evidence)} ({esc(category_summary)}).
\item Every statement about an old iteration is tied to a maintained design file, saved output directory, test, or appended project report. Where a log was not preserved, the dossier says so rather than inventing a result.
\end{{itemize}}

\section{{Exact operational architecture}}
{arch_table}

\section{{Exact input/output contracts}}
{contract_table}

\section{{Current quantitative result boundary}}
{result_table}

\section{{Iteration history: what worked, what failed, and why}}
{iteration_table}

\section{{Complete current source inventory}}
{inventory_table}

\section{{Function, class, and method inventory}}
The signatures below are parsed directly from the current source. They inventory the callable surface without duplicating implementation bodies; the maintained Python files remain the source of truth.

{symbol_table}

\section{{All mapped branch tests}}
{test_table}

\subsection{{Exact branch pytest transcript}}
The following is the fresh verbose execution of every test file in the branch source map, including parametrized node IDs and final status.
\VerbatimInput[fontsize=\scriptsize,breaklines=true,breakanywhere=true]{{{(ROOT / str(test_report['log_path'])).resolve().as_posix()}}}

\subsection{{Full repository pytest transcript}}
The complete suite is included because shared modules can change branch behavior outside a narrow file selection.
\VerbatimInput[fontsize=\scriptsize,breaklines=true,breakanywhere=true]{{{(ROOT / str(full_report['log_path'])).resolve().as_posix()}}}

\section{{Saved benchmark and proof-run inventory}}
{result_dir_table}
The file-level ledger is {ptex(f"output/branches/{branch}/result_inventory.csv")}. It records every saved file, byte count, modification time and SHA-256 hash. Presence proves preservation/execution evidence only; it does not upgrade the validation status.

\section{{Folder-by-folder design/report inventory}}
{docs_table}

\section{{All branch-tagged PDF evidence}}
{evidence_table}
The detailed mapping is also saved as {ptex(f"output/branches/{branch}/branch_evidence_ledger.csv")}. Literature PDFs are not synthesized or embedded; only project-authored reports/case studies selected below are appended.

\section{{Legacy reports and case studies included verbatim}}
{legacy_table}
\clearpage
\appendix
\section{{Verbatim project-authored historical records}}
Each following document keeps its original title pages, tables, figures, citations, caveats and iteration-specific conclusions. Later front-matter decisions take precedence where an older report describes a superseded architecture.
{chr(10).join(legacy_includes)}
"""
    return rf"""% Compile from the ECGdetector repository root.
\documentclass[10pt]{{article}}
\usepackage[margin=0.62in]{{geometry}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern,microtype}}
\usepackage{{amsmath,amssymb,booktabs,longtable,array,tabularx}}
\usepackage[dvipsnames,table]{{xcolor}}
\usepackage{{enumitem,fancyhdr,lastpage,pdfpages,fvextra}}
\usepackage[hidelinks]{{hyperref}}
\hypersetup{{pdftitle={{{esc(code)} Exhaustive Technical Dossier - {esc(spec['title'])}}}}}
\Urlmuskip=0mu plus 1mu
\emergencystretch=2em
\setlist{{nosep,leftmargin=*}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0.45em}}
\setlength{{\LTpre}}{{0.3em}}
\setlength{{\LTpost}}{{0.7em}}
\pagestyle{{fancy}}
\fancyhf{{}}
\lhead{{ECGdetector {esc(code)} exhaustive dossier}}
\rhead{{v0.4.0 current plus preserved iterations}}
\cfoot{{\thepage/\pageref{{LastPage}}}}
\title{{{esc(code)} Exhaustive Technical Dossier\\\large {esc(spec['title'])}}}
\author{{ECGdetector documentation and evidence bundle}}
\date{{20 August 2026}}
\begin{{document}}
\sloppy
\maketitle
\begin{{center}}
\fbox{{\parbox{{0.94\linewidth}}{{Engineering/research record only. Current behavior, negative results, missing validation and superseded iterations are deliberately separated.}}}}
\end{{center}}
\tableofcontents
\clearpage
{body}
\end{{document}}
"""


def main() -> int:
    TEST_QA.mkdir(parents=True, exist_ok=True)
    maps = {branch: source_map(branch) for branch in BRANCHES}
    mapped = {row["relative_path"] for rows in maps.values() for row in rows}
    all_code = set(all_first_party_python())
    coverage = {
        "first_party_python_files": len(all_code), "mapped_unique_files": len(mapped),
        "unmapped": sorted(all_code - mapped), "mapped_missing": sorted(path for path in mapped if not (ROOT / path).exists()),
    }
    (QA / "branch_source_coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    if coverage["unmapped"] or coverage["mapped_missing"]:
        raise RuntimeError(f"Branch source coverage is incomplete: {coverage}")

    full_report = run_full_tests()
    reports: list[dict[str, object]] = []
    for branch, spec in BRANCHES.items():
        directory = OUTPUT / "branches" / branch
        rows = maps[branch]
        symbols = symbol_rows(rows)
        tests = test_rows(rows)
        branch_report = run_tests(branch, rows)
        results = result_directories(spec["output_keywords"])
        evidence = evidence_rows(str(spec["code"]))
        docs = doc_rows(spec["docs"], spec["legacy"])
        file_rows = result_file_rows(results)
        write_csv(directory / "result_inventory.csv", file_rows)
        write_csv(directory / "branch_evidence_ledger.csv", evidence)
        write_csv(directory / "historical_document_ledger.csv", docs)
        write_csv(directory / "symbol_inventory.csv", symbols)
        write_csv(directory / "test_inventory.csv", tests)
        tex = build_tex(branch, spec, rows, symbols, tests, branch_report, full_report, results, evidence, docs)
        (directory / str(spec["tex"])).write_text(tex, encoding="utf-8")
        reports.append({
            "branch": spec["code"], "tex": (directory / str(spec["tex"])).relative_to(ROOT).as_posix(),
            "source_rows": len(rows), "symbols": len(symbols), "test_definitions": len(tests),
            "branch_test": branch_report, "result_directories": len(results), "result_files": len(file_rows),
            "evidence_groups": len(evidence), "legacy_pdfs": len(spec["legacy"]),
        })
    report = {"status": "passed" if full_report["exit_code"] == 0 and all(item["branch_test"]["exit_code"] == 0 for item in reports) else "failed", "coverage": coverage, "full_test": full_report, "branches": reports}
    (QA / "branch_dossier_build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
