"""Generate the clean ECGdetector v0.4.0 documentation sources and notebooks.

This script intentionally does not compile LaTeX or copy evidence PDFs.  Those
steps are separate so build logs and validation results remain auditable.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import tomllib
from collections import Counter
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
SRC = ROOT / "feature_extraction" / "src" / "ecg_cascade"
SCRIPTS = ROOT / "feature_extraction" / "scripts"
TESTS = ROOT / "feature_extraction" / "tests"


BRANCHES = {
    "B1_peak_rr_hrv": {
        "code": "B1",
        "title": "Peak Detection, RR, and HRV",
        "tex": "B1_Peak_RR_HRV_Technical_Record.tex",
        "pdf": "B1_Peak_RR_HRV_Technical_Record.pdf",
        "notebook": "B1_Peak_RR_HRV_Walkthrough.ipynb",
        "production": [
            "__init__.py", "config.py", "edf.py", "wfdb_io.py", "peaks.py", "neurokit_zhai.py",
            "zhai.py", "fusion.py", "interval_consistent_fusion.py", "reliability.py",
            "rr_reliability_context.py", "rr_hrv.py", "hrv_fidelity.py",
            "prsa_bprsa.py", "pipeline.py", "outputs.py", "plotting.py", "cli.py",
            "clinical_demo_data.py", "clinical_demo_web.py", "rpeak_viewer.py", "validation.py",
        ],
        "scripts": [
            "analyze_rr_hrv_method_selection.py", "benchmark_unsw_neurokit_zhai_fusion.py",
            "benchmark_unsw_neurokit_context.py", "benchmark_neurokit_zhai.py",
            "benchmark_interval_consistent_fusion.py", "benchmark_expert_unsw_hrv_fidelity.py",
            "benchmark_prsa_bprsa_available_data.py", "plot_previous_vs_complete_rr_hrv.py",
            "plot_neurokit_pan_gate_failures.py", "audit_prsa_bprsa_context_matrix.py",
            "benchmark_prsa_bprsa_noise_fidelity.py", "audit_prsa_bprsa_arrhythmia_context.py",
        ],
        "tests": [
            "test_peaks.py", "test_zhai.py", "test_interval_consistent_fusion.py",
            "test_rr_reliability_context.py", "test_rr_hrv.py", "test_complete_rr_hrv.py",
            "test_hrv_fidelity.py", "test_prsa_bprsa.py", "test_clinical_demo_data.py",
            "test_rpeak_viewer.py",
        ],
    },
    "B2_morphology": {
        "code": "B2",
        "title": "Beat Morphology",
        "tex": "B2_Morphology_Technical_Record.tex",
        "pdf": "B2_Morphology_Technical_Record.pdf",
        "notebook": "B2_Morphology_Walkthrough.ipynb",
        "production": [
            "__init__.py", "morphology.py", "patient_template_morphology.py", "prominence_morphology.py",
            "fiducials.py", "fiducial_fusion.py", "morphology_validation.py",
            "delineation_validation.py", "peaks.py", "plotting.py", "wfdb_io.py",
            "prsa_bprsa.py", "clinical_demo_data.py", "clinical_demo_web.py", "rpeak_viewer.py",
        ],
        "scripts": [
            "analyze_varon_fixed_qrs_capture.py", "audit_fiducial_morphology.py",
            "run_patient_template_morphology.py", "benchmark_patient_average_varon_capture.py",
            "benchmark_patient_asymmetric_varon_capture.py", "benchmark_morphology_fidelity.py",
            "benchmark_delineation_gate.py", "benchmark_fiducial_adjustments.py",
            "plot_bbb_fiducial_cases.py", "audit_prsa_bprsa_context_matrix.py",
            "benchmark_prsa_bprsa_noise_fidelity.py", "audit_prsa_bprsa_arrhythmia_context.py",
        ],
        "tests": [
            "test_morphology_validation.py", "test_morphology_fusion.py",
            "test_patient_template_morphology.py", "test_prominence_morphology.py",
            "test_fiducials.py", "test_fiducial_fusion.py", "test_delineation_validation.py",
            "test_prsa_bprsa.py", "test_clinical_demo_data.py", "test_rpeak_viewer.py",
        ],
    },
    "B3_conduction_repolarization": {
        "code": "B3",
        "title": "Conduction and Repolarization",
        "tex": "B3_Conduction_Repolarization_Technical_Record.tex",
        "pdf": "B3_Conduction_Repolarization_Technical_Record.pdf",
        "notebook": "B3_Conduction_Repolarization_Walkthrough.ipynb",
        "production": [
            "__init__.py", "conduction_timing.py", "conduction_information.py", "fiducials.py",
            "prominence_morphology.py", "delineation_validation.py", "signal_quality.py",
            "clinical_demo_data.py", "clinical_demo_web.py",
        ],
        "scripts": ["run_conduction_timing.py", "run_conduction_information.py", "benchmark_delineation_gate.py"],
        "tests": ["test_conduction_timing.py", "test_conduction_information.py", "test_delineation_validation.py", "test_clinical_demo_data.py"],
    },
    "B4_signal_quality": {
        "code": "B4",
        "title": "Signal Quality",
        "tex": "B4_Signal_Quality_Technical_Record.tex",
        "pdf": "B4_Signal_Quality_Technical_Record.pdf",
        "notebook": "B4_Signal_Quality_Walkthrough.ipynb",
        "production": ["__init__.py", "signal_quality.py", "peaks.py", "reliability.py", "wfdb_io.py", "plotting.py", "clinical_demo_data.py", "clinical_demo_web.py"],
        "scripts": ["validate_signal_quality_nstdb.py", "benchmark_artifact_degradation.py"],
        "tests": ["test_signal_quality.py", "test_rr_reliability_context.py", "test_clinical_demo_data.py"],
    },
}


def tex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def tex_document(title: str, subtitle: str, body: str) -> str:
    return rf"""\documentclass[11pt]{{article}}
\usepackage[margin=0.72in]{{geometry}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern,microtype}}
\usepackage{{amsmath,amssymb,booktabs,longtable,array,tabularx}}
\usepackage[dvipsnames,table]{{xcolor}}
\usepackage{{enumitem}}
\usepackage[hidelinks]{{hyperref}}
\definecolor{{implemented}}{{HTML}}{{DDEEDD}}
\definecolor{{experimental}}{{HTML}}{{FFF0C7}}
\definecolor{{held}}{{HTML}}{{E8E8E8}}
\definecolor{{proposed}}{{HTML}}{{DDE8F8}}
\setlist{{nosep,leftmargin=*}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0.55em}}
\newcommand{{\status}}[1]{{\textsf{{\textbf{{#1}}}}}}
\newcommand{{\code}}[1]{{\texttt{{#1}}}}
\title{{{title}\\\large {subtitle}}}
\author{{ECGdetector v0.4.0 documentation bundle}}
\date{{19 August 2026}}
\begin{{document}}
\maketitle
\begin{{center}}
\fbox{{\parbox{{0.94\linewidth}}{{This is an engineering and research record, not a clinical device specification. Missing measurements remain missing; no branch in v0.4.0 produces a validated seizure probability, diagnosis, or treatment recommendation.}}}}
\end{{center}}
\tableofcontents
\newpage
{body}
\end{{document}}
"""


ARCHITECTURE_BODY = r"""
\section{Purpose and correction of the legacy architecture}
The archived architecture predates the conduction/repolarization branch and does not match the live detector roles. This record replaces it as the current architectural source for v0.4.0. The live system is a measurement cascade with shared data and peak layers feeding four independent coding branches. It is not a four-vote seizure classifier.

\section{Status vocabulary and non-claims}
\begin{longtable}{p{0.18\linewidth}p{0.74\linewidth}}
\toprule Status & Operational meaning \\ \midrule
\rowcolor{implemented}\status{Implemented} & Maintained production module, stable public call surface, and direct automated tests.\\
\rowcolor{experimental}\status{Experimental} & Executable research path or benchmark whose outputs are not accepted as a production decision rule.\\
\rowcolor{held}\status{Held} & Preserved result or rejected alternative kept for traceability and comparison.\\
\rowcolor{proposed}\status{Proposed} & Future design without completed code and validation.\\ \bottomrule
\end{longtable}
The architecture does not claim prospective seizure sensitivity, specificity, false-alarm rate, patient generalization, artifact-class probability, or clinical abnormality detection.

\section{Operational topology}
\begin{enumerate}
\item \textbf{Data contract.} EDF or WFDB input, explicit channel/lead, physical-unit metadata, sampling rate, requested interval, and provenance identifiers.
\item \textbf{Shared peak layer.} The UNSW/Pan--Tompkins-derived primary control and independent NeuroKit/Zhai comparator tracks remain distinct. Agreement and offsets are context, not permission to silently rewrite one source with another.
\item \textbf{B1: peak/RR/HRV.} Constructs source-consistent RR tachograms and causal HRV feature sequences.
\item \textbf{B2: morphology.} Extracts fixed Varon QRS crops, patient-calibrated challengers, full-cycle templates, and landmark-based descriptors.
\item \textbf{B3: conduction/repolarization.} Consumes supplied aligned P/QRS/T landmarks and emits timing measurements and information-only temporal summaries.
\item \textbf{B4: signal quality.} Emits deterministic signal-quality indices with explicit prerequisite availability.
\item \textbf{Future fusion.} A proposed temporal model may consume validated branch measurements, availability masks, quality context, and provenance. No such fusion is implemented.
\end{enumerate}

\section{Four-branch status matrix}
\begin{longtable}{p{0.09\linewidth}p{0.28\linewidth}p{0.23\linewidth}p{0.30\linewidth}}
\toprule Branch & Implemented & Experimental or held & Explicitly absent \\ \midrule
B1 & Peak tracks, match context, RR construction, Jeppesen HRV arrays, fidelity audit utilities, PRSA/BPRSA measurements. & Interval-consistent same-source fusion; seizure-EDF descriptive proof runs. & Validated seizure classifier and prospective patient evaluation.\\
B2 & Fixed 120-ms Varon five-beat uncentered Gram eigenvalues; template and prominence measurement paths. & Personalized symmetric/asymmetric crops; whole-cycle templates; matched-lead delineation challengers. & Morphology seizure probability and accepted landmark production gate.\\
B3 & Timing v1; information v2; Diab-aligned nine peak dynamics; five-beat summaries; nine-beat QT context; external eligibility input. & MIT-BIH record-100 smoke outputs and manual error studies. & Pipeline integration, final feature-matrix eligibility, diagnostic thresholds, seizure label.\\
B4 & Deterministic time, spectral, detector-agreement, template, baseline, mains, residual, and saturation indices. & NSTDB degradation validation. & Artifact classifier, probability calibration, quality-dependent deletion policy.\\ \bottomrule
\end{longtable}

\section{Shared data, missingness, and provenance contracts}
Every measurement row must identify the patient/record, lead/channel, source interval, sampling rate, anchor track, algorithm and parameters, and feature schema. Conditional inputs such as detector pairs, ADC rails, QRS onsets, cross-lead peaks, or group labels are never invented. Each unavailable value is represented by a missing value plus an availability flag and reason. Zero is reserved for a measured zero.

Branch outputs retain unsupported or rejected beats for audit. A quality branch may expose eligibility, but B3 only consumes a caller-supplied Boolean eligibility column and does not reinterpret it. Cross-branch joins require an explicit event key and compatible provenance; time proximity alone is insufficient.

\section{Temporal semantics}
RR intervals end at the later R peak. Jeppesen windows are causal trailing windows. Varon morphology uses five consecutive anchored complexes. B3 timing v1 uses trailing 30-beat variability windows; information v2 uses causal trailing five-beat summaries, non-overlapping previous-window comparisons, and a separate nine-beat QT/RR context. B4 window functions report the exact support interval. Future temporal fusion must preserve those support endpoints to prevent look-ahead leakage.

\section{Validation layers}
Unit tests verify equations, schemas, boundary conditions, and explicit non-claims. Dataset benchmarks answer narrower questions: MIT-BIH Arrhythmia tests beat timing but has no seizure labels; LUDB/QTDB test delineation and capture; NSTDB tests controlled artifact degradation; CHB-MIT/Siena entries support descriptive EDF proof runs. None alone establishes a clinical detector.

\section{Public API stability}
The cleanup changes documentation and evidence organization only. The maintained production-module tree remains the single source of truth. The notebooks import those modules; they do not contain forked implementations. No public \code{ecg\_cascade} API is renamed or changed by this bundle.

\section{Future temporal fusion gate}
A fusion model may be considered only after frozen, patient-separated evaluation protocols exist for each branch; B3 landmark error tails are controlled; B4 thresholds/classifier are calibrated without contaminating measurement semantics; and availability/provenance masks are carried into model training. Until then branch values are observational evidence, not votes.
"""


BRANCH_BODIES = {
    "B1_peak_rr_hrv": r"""
\section{Scope and live role}
B1 owns beat-anchor generation, source-consistent RR construction, and continuous HRV measurements. The primary UNSW/Pan--Tompkins-derived control, NeuroKit comparator, and Zhai correlation track remain inspectable. Comparator evidence describes reliability; it is not a license to mix interval endpoints from unrelated sources.

\section{Peak tracks and reliability context}
Candidate indices are integer samples with explicit sampling rate, orientation, and segment offset. Agreement uses a declared tolerance and a one-to-one matching policy. Match count, signed/absolute offset, detector support, and Zhai correlation remain adjacent to the anchor. Reliability does not delete a peak or impute an alternative timestamp.

\section{RR construction}
For monotonically increasing peak samples $r_i$ at sampling rate $f_s$,
\[RR_i=1000\frac{r_i-r_{i-1}}{f_s}\quad\text{ms},\qquad HR_i=\frac{60000}{RR_i}\quad\text{bpm}.\]
Every RR interval inherits both endpoint provenance. Interval-consistent fusion permits a replacement only as an explicitly audited same-source sequence. The held naive fusion alternatives remain evidence of why per-event mixing can distort variability.

\section{Jeppesen HRV implementation}
The production function applies a causal seven-sample median to RR, evaluates trailing 100-interval windows, and returns continuous statistics. The record includes mean/standard deviation, RMSSD-like successive variation, Poincar\'e axes, and the absolute least-squares HR slope. For successive pairs $x_1=RR_1\ldots RR_{n-1}$ and $x_2=RR_2\ldots RR_n$,
\[SD1=\operatorname{sd}\left(\frac{x_2-x_1}{\sqrt 2}\right),\qquad SD2=\operatorname{sd}\left(\frac{x_2+x_1}{\sqrt 2}\right).\]
The causal timestamp is the ending R peak. Startup rows are missing until their stated history exists.

\section{Fidelity and proof-run evidence}
The completed 48-record MIT-BIH audit used 109,494 expert beats. Saved reports place the UNSW control near F1 0.997 and the independent NeuroKit/Zhai tracks lower under their frozen configurations. These are beat-detection results, not seizure performance. HRV fidelity compares expert-anchored and detector-anchored sequences and reports change preservation rather than declaring interchangeable timestamps.

CHB04\_28 proof runs include 256-Hz quiet and seizure-containing intervals. The seizure-containing recording exposes severe contamination and detector disagreement. PRSA/BPRSA outputs over the two labeled intervals are descriptive, overlapping-window, single-patient observations only. Configured Siena PN06/PN12 records remain local proof-run entries with header-selected \code{EKG EKG}; channel semantics still require documentation confirmation.

A fresh 19-August audit attached all-48 expert-anchor PRSA/BPRSA rows to exact and midpoint probes. Across 105,654 feature rows and 211,308 matrix probes it reported zero timestamp/value/flag mismatches, zero future leaks, zero negative context ages, and zero changes to either core gate. Controlled NSTDB electrode-motion comparisons held expert R timestamps fixed: RR-only values were invariant, while median BPRSA correlation fell from 0.8100 at +24 dB to 0.0315 at -6 dB and sign agreement from 0.8836 to 0.5678. Of 105,654 windows, 73.54\% contained a non-normal expert beat and all 946 nonpositive-spline failures occurred in that stratum. Therefore detector/numerical reliability is not artifact freedom; BPRSA still requires separately attached signal-quality, lead/polarity, and rhythm context. The fresh CHB and MIT tables exactly reproduced the earlier parsed outputs.

\section{Failure modes and safeguards}
Duplicate/nonmonotonic peaks, wrong sampling rate, polarity error, cross-source interval mixing, ectopy, saturation, missing history, and artifact-driven false peaks can all dominate HRV. Safeguards are validation errors, availability/context columns, immutable source labels, and B4 context. No hard quality deletion is hidden in B1.

\section{Current decision}
\status{Implemented}: measurements, peak contexts, and tests. \status{Experimental}: fusion challengers and seizure-EDF descriptive comparisons. \status{Held}: naive event-wise fusion and unsupported threshold claims. \status{Proposed}: patient-separated prospective seizure modeling with temporal fusion.
""",
    "B2_morphology": r"""
\section{Scope and measurement lanes}
B2 measures beat shape around an explicit anchor. It does not decide that a waveform is seizure, artifact, PVC, or bundle-branch block. Four lanes are kept distinct: fixed Varon core, patient-calibrated crops, whole-cycle templates, and landmark/prominence descriptors.

\section{Fixed Varon core}
For each R anchor the implemented control extracts a nominal symmetric 120-ms waveform. Five consecutive rows form $Q$. The paper-traceable core computes the uncentered Gram matrix
\[G=QQ^\mathsf{T}\]
and returns its five eigenvalues in descending order. No undocumented per-beat centering or amplitude normalization is introduced. Partial boundary crops remain present but cannot produce a defined five-beat core.

\section{Patient-specific challengers}
Median/mean duration crops were tested and often under-captured long-tail QRS complexes. Symmetric and separate-side p95 challengers improve some coverage but trade added context against incomplete calibration support. The strongest saved LUDB examples improve over the sampled fixed crop, yet support varies sharply by calibration size and dataset. These paths are experimental; the fixed core remains the reference control.

\section{Whole-cycle and prominence lanes}
Patient template morphology captures wider beat context for visual and similarity analysis. Prominence-based P/QRS/T delineation yields amplitudes, durations, and boundary context without replacing the fixed Varon result. These outputs are measurement candidates and must retain method, lead, polarity, and boundary provenance.

\section{Validation evidence}
On the all-48 MIT-BIH fidelity audit, saved continuity results are approximately 98.708\% for expert-to-UNSW anchors, 93.424\% for NeuroKit, and 92.258\% for Zhai. MIT-BIH beat labels do not provide manual beatwise QRS boundaries. Manual matched-lead capture is therefore evaluated on LUDB lead II and the qualifying QTDB MLII subset. Fixed nominal 120-ms complete-QRS capture is about 73.7\% on LUDB and 62.8\% on QTDB.

The prominence matched-lead gate improves several medians, but error tails remain material: saved LUDB p95 errors include roughly 24 ms for QRS onset, 44 ms for QRS offset, and 52 ms for T offset. These tails are especially consequential for B3 variability.

\section{Failure modes}
Anchor displacement, polarity, clipping, ectopy, pacing, bundle-branch morphology, sampling parity, truncated edge beats, and lead mismatch can alter eigenvalues. Crop capture and detector support are reported rather than silently filtering beats. Manual boundary datasets are not treated as seizure datasets.

\section{Current decision}
\status{Implemented}: fixed Varon extraction and reproducible feature tables. \status{Experimental}: personalized crops, whole-cycle templates, and prominence landmarks. \status{Held}: poorly supported mean/median personalized policies and any implied classifier. \status{Proposed}: frozen patient-level calibration plus seizure-labeled external validation.
""",
    "B3_conduction_repolarization": r"""
\section{Scope and non-integration status}
B3 consumes already aligned P, Q, R, S, and T landmarks. It does not detect beats, delineate waves, diagnose conduction disease, or enter the v0.4.0 final feature matrix. The branch is intentionally information-only because upstream landmark error may exceed the physiological changes of interest.

\section{Timing v1}
Beatwise intervals include P duration, PR interval/segment, QRS duration, QT, JT, and T-peak-to-end. QT correction is explicit:
\[QTc_B=\frac{QT}{\sqrt{RR_s}},\qquad QTc_F=\frac{QT}{\sqrt[3]{RR_s}},\qquad QTc_{Fram}=QT+154(1-RR_s).\]
QTVI variants name their heart-rate or RR denominator. STV30 requires exactly 30 finite consecutive intervals and equals $\sum|D_{i+1}-D_i|/(30\sqrt2)$. Timing windows trail the current beat and never read later rows.

\section{Information v2}
The v2 layer reuses frozen v1 interval equations, then adds paper-aligned Diab dynamics: P--R, Q--R, S--R, T--R within-beat timing and P--P, Q--Q, R--R, S--S, T--T interbeat timing. The ordered raw dynamics are retained; they are not falsely described as five-beat averages.

For engineering display, each causal five-beat endpoint reports mean, median, sample SD, raw MAD, IQR, range, adjacent-pair RMSSD, robust Theil--Sen trend, late-versus-early shift, and current-minus-prior median. A non-overlapping previous-window comparison appears only after two complete windows. A separate nine-beat context averages QT and preceding RR before applying each QT correction, matching the stated Brotherstone-style context without reproducing a seizure detector.

\section{External quality eligibility}
An optional caller-supplied Boolean column can mark beats eligible for summaries. Ineligible beats remain in the beat table. B3 never creates, thresholds, calibrates, or reinterprets artifact quality. Without the column, only finite-measurement support applies.

\section{Validation and limits}
Unit tests hand-check interval equations, causality, Diab dynamics, robust summaries, non-overlapping comparison, nine-beat QT context, and external eligibility. A MIT-BIH record-100 smoke run proves execution but not landmark truth: \code{.atr} annotations are beat/R references, not manual P/Q/S/T boundaries. Matched-lead LUDB/QTDB audits show useful medians with substantial P/T/QRS endpoint tails. Consequently thresholds, abnormality labels, clinical interpretation, and seizure probability remain disabled.

\section{Current decision}
\status{Implemented}: timing v1 and information v2 standalone APIs/tests. \status{Experimental}: saved smoke runs and observational temporal summaries. \status{Held}: any feature-matrix integration based on current endpoint error. \status{Proposed}: validated landmark gate, patient/lead baselines, rate-aware norms, and patient-separated seizure evaluation.
""",
    "B4_signal_quality": r"""
\section{Scope}
B4 is an implemented deterministic measurement layer. It exposes signal-quality indices (SQIs) and their prerequisites. It is not the proposed artifact classifier and does not emit a quality probability or hard keep/drop decision.

\section{Implemented SQI families}
Unconditional features include finite fraction, population skewness/kurtosis, amplitude summaries, flatline measures, Clifford band-power ratios, and Langley-style saturation when physical scaling is known. Conditional families include ADC rail occupancy, Li bSQI Jaccard and Zhao qSQI Dice detector agreement, Ho endpoint-supported RR context, Orphanidou template correlation, cross-lead context, Galeotti baseline/mains/residual estimates, and Menon fifth-order Fourier-template score.

Detector-agreement quantities stay distinct. For primary and secondary matched event sets, the Jaccard and Dice forms are
\[bSQI=\frac{|A\cap B|}{|A\cup B|},\qquad qSQI=\frac{2|A\cap B|}{|A|+|B|}.\]
Ho support operates on RR endpoints and therefore is not numerically interchangeable with either event-set score.

\section{Missingness contract}
If ADC rails, a comparator detector, QRS onsets, other-lead peaks, or beat-group labels are absent, the dependent value is missing, \code{available=false}, and a reason is recorded. The branch never imputes zero. Units are explicit through \code{uv\_per\_input\_unit}; rail and saturation claims are unavailable without required metadata.

\section{NSTDB validation}
The saved v0 benchmark compares clean MIT-BIH 118/119 segments with electrode-motion Noise Stress Test Database copies across six SNR conditions per record. It verifies deterministic response and provenance, not classifier sensitivity or a universal threshold. BUT-QDB or comparable broader labeled quality validation remains future work.

\section{Proposed classifier architecture}
The earlier architecture proposed artifact classes, calibration, and branch-aware decisions. Those components are retained only as future design. A valid classifier would require frozen labels, patient/record-separated calibration, prespecified operating points, out-of-domain evaluation, probability calibration, and a policy specifying whether quality is context, eligibility, or exclusion. None is silently represented by an SQI.

\section{Failure modes}
Wrong physical units, unknown ADC rails, detector dependence, paced/ectopic beats, narrowband mains, short windows, nonstationarity, and clean-but-unusual morphology can all confound SQIs. Each score must be interpreted with prerequisites and source metadata. B4 context may accompany B1--B3, but measurement deletion requires an explicit downstream policy.

\section{Current decision}
\status{Implemented}: deterministic SQIs, availability reasons, and tests. \status{Experimental}: NSTDB degradation study. \status{Held}: uncalibrated thresholds and label names. \status{Proposed}: artifact classifier, calibration, and cross-dataset operating policy.
""",
}


DATASET_BODY = r"""
\section{Scope and provenance rule}
This record replaces the legacy two-branch dataset summary. It covers the five local dataset collections, the configured CHB/Siena proof recordings, annotations, executed outputs, and branch-appropriate claims. The 588 MB raw collection remains under \code{Datasets/}; it is indexed, not copied into this bundle.

\section{Dataset-to-branch use}
\small
\begin{longtable}{p{0.17\linewidth}p{0.115\linewidth}p{0.115\linewidth}p{0.115\linewidth}p{0.115\linewidth}p{0.12\linewidth}}
\toprule Dataset & B1 & B2 & B3 & B4 & Limits \\ \midrule
CHB-MIT \code{chb04\_28} & EDF proof runs; labeled seizure intervals & waveform inspection only & not landmark truth & noisy-signal context & one patient/two intervals; no manual ECG landmarks\\
MIT-BIH Arrhythmia & 48-record beat and HRV fidelity & anchor continuity and cases & execution smoke only & clean source records & no seizure labels; beat annotations are not P/Q/S/T boundaries\\
MIT-BIH Noise Stress & detector and interval stress & not primary & not primary & controlled artifact degradation & synthetic noise mixtures; not universal quality labels\\
LUDB & limited anchor context & manual lead-II QRS/P/T gate & interval landmark gate & possible future clean-quality context & short records; not seizure labeled\\
QTDB & limited anchor context & matched MLII manual boundary subset & repolarization landmark gate & possible future context & only qualifying manually delineated leads/records\\
Configured Siena PN06/PN12 & local EDF proof configuration & not validated & not validated & possible future artifact study & paths may be external; \code{EKG EKG} mapping requires confirmation\\ \bottomrule
\end{longtable}
\normalsize

\section{Annotations and sampling}
The generated \path{dataset_file_manifest.csv}, \path{wfdb_header_inventory.csv}, and \path{configured_recordings.csv} enumerate files, header sampling rates, signal labels, annotation extensions, and configured seizure intervals. Sampling and lead claims come from headers/configuration rather than filename inference. CHB04\_28 is configured at 256 Hz with ECG; Siena PN06/PN12 at 512 Hz with \code{EKG EKG}. LUDB commonly supplies 12 leads at 500 Hz; MIT-BIH Arrhythmia is commonly two-channel at 360 Hz; QTDB records vary by source but the matched project subset is explicitly lead controlled.

\section{Executed evidence}
Saved outputs include the all-48 MIT-BIH peak, fusion, morphology, and HRV fidelity audits; LUDB/QTDB delineation and capture audits; NSTDB artifact degradation; conduction record-100 smoke runs; and CHB04\_28 PRSA/BPRSA descriptions. The evidence manifest links each content hash to all former paths and branch tags.

\section{Unsupported claims}
No local collection supports a claim that the complete four-branch architecture detects seizures prospectively. MIT-BIH/NSTDB do not contain seizure labels, LUDB/QTDB are landmark resources, and the local CHB/Siena proof scope is insufficient for generalization. Dataset overlap, repeated records at multiple SNRs, and parameter development on benchmark sets must be respected in future splits.
"""


def write_source_maps() -> None:
    for branch, spec in BRANCHES.items():
        directory = OUTPUT / "branches" / branch
        directory.mkdir(parents=True, exist_ok=True)
        rows = []
        for role, base, names in (
            ("production", SRC, spec["production"]),
            ("script", SCRIPTS, spec["scripts"]),
            ("test", TESTS, spec["tests"]),
        ):
            for name in names:
                path = base / name
                rows.append({
                    "branch": spec["code"], "role": role,
                    "relative_path": path.relative_to(ROOT).as_posix(),
                    "exists": path.exists(),
                    "shared": name in {"peaks.py", "reliability.py", "signal_quality.py", "wfdb_io.py", "plotting.py"},
                    "source_of_truth": role == "production",
                })
        with (directory / "source_map.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        (directory / "evidence_index.md").write_text(
            f"# {spec['code']} evidence index\n\n"
            "This branch uses content-addressed evidence. See `../../evidence/indexes/branch_evidence_map.csv` "
            "for all PDFs and LaTeX sources tagged to this branch. Cross-branch documents are stored once.\n\n"
            "`source_map.csv` is the authoritative inventory of maintained production, script, and test Python files.\n",
            encoding="utf-8",
        )


def markdown_cell(text: str):
    return nbformat.v4.new_markdown_cell(text)


def code_cell(code: str):
    return nbformat.v4.new_code_cell(code)


COMMON_SETUP = """from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "feature_extraction" / "src").exists())
sys.path.insert(0, str(ROOT / "feature_extraction" / "src"))
print(f"Repository root: {ROOT}")
"""


NOTEBOOK_CODE = {
    "B1_peak_rr_hrv": [
        ("Create an auditable synthetic tachogram", """from ecg_cascade.rr_hrv import calculate_jeppesen_feature_arrays
RUN = ROOT / "output" / "notebook_runs" / "B1_peak_rr_hrv"; RUN.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(41)
beat = np.arange(180)
rr_ms = 800 + 35*np.sin(2*np.pi*beat/30) + rng.normal(0, 6, beat.size)
features = calculate_jeppesen_feature_arrays(rr_ms, window_size=100, median_width=7)
features.to_csv(RUN / "jeppesen_feature_arrays.csv", index=False)
features.tail(5)
"""),
        ("Plot inputs and a defined trailing feature", """fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(rr_ms); axes[0].set_ylabel("RR (ms)"); axes[0].set_title("Synthetic source-consistent RR tachogram")
numeric = [c for c in features.columns if c not in {"window_index"} and pd.api.types.is_numeric_dtype(features[c])]
chosen = next(c for c in numeric if features[c].notna().sum() > 5)
axes[1].plot(features[chosen]); axes[1].set_ylabel(chosen); axes[1].set_xlabel("RR endpoint")
fig.tight_layout(); fig.savefig(RUN / "b1_diagnostic.png", dpi=160); plt.show()
assert (RUN / "jeppesen_feature_arrays.csv").exists() and (RUN / "b1_diagnostic.png").exists()
"""),
    ],
    "B2_morphology": [
        ("Build a deterministic ECG-like waveform and call the production Varon extractor", """from ecg_cascade.morphology import extract_varon_morphology
RUN = ROOT / "output" / "notebook_runs" / "B2_morphology"; RUN.mkdir(parents=True, exist_ok=True)
fs = 250.0; samples = np.arange(int(12*fs)); peaks = np.arange(int(fs), samples.size-int(fs), int(fs))
ecg = 0.02*np.sin(2*np.pi*0.4*samples/fs)
for i, peak in enumerate(peaks):
    ecg += (1+0.08*np.sin(i/2))*np.exp(-0.5*((samples-peak)/5.0)**2)
result = extract_varon_morphology(ecg, peaks, fs, anchor_track="synthetic_control")
result.beat_context.to_csv(RUN / "beat_context.csv", index=False)
result.features.to_csv(RUN / "varon_features.csv", index=False)
np.save(RUN / "beat_waveforms.npy", result.beat_waveforms)
result.summary()
"""),
        ("Inspect aligned waveforms and eigenvalues", """fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(result.beat_waveforms[:6].T, alpha=.7); axes[0].set_title("First six 120-ms captures"); axes[0].set_xlabel("sample")
for c in [f"lambda{i}" for i in range(1,6)]: axes[1].plot(result.features[c], label=c)
axes[1].set_title("Five-beat Gram eigenvalues"); axes[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(RUN / "b2_diagnostic.png", dpi=160); plt.show()
assert result.features["published_varon_core_defined"].all()
"""),
    ],
    "B3_conduction_repolarization": [
        ("Construct supplied P/QRS/T landmarks with known intervals", """from ecg_cascade.conduction_timing import extract_conduction_timing
from ecg_cascade.conduction_information import extract_conduction_information
RUN = ROOT / "output" / "notebook_runs" / "B3_conduction_repolarization"; RUN.mkdir(parents=True, exist_ok=True)
fs = 1000.0; n = 36; r = 1000 + np.arange(n)*1000; qrs_on = r-70
pqrst = pd.DataFrame({
 "beat_index":np.arange(n), "patient_id":"synthetic", "lead_name":"II", "anchor_track":"known",
 "r_peak_sample_in_segment":r, "r_peak_time_s":r/fs,
 "p_onset_sample_in_segment":r-220, "p_peak_sample_in_segment":r-180, "p_offset_sample_in_segment":r-130,
 "q_peak_sample_in_segment":r-40, "qrs_onset_sample_in_segment":qrs_on,
 "s_peak_sample_in_segment":r+40, "qrs_offset_sample_in_segment":qrs_on+120+6*np.sin(np.arange(n)/4),
 "t_peak_sample_in_segment":r+250, "t_offset_sample_in_segment":r+400+8*np.sin(np.arange(n)/6),
 "quality_eligible":True})
pqrst.loc[12, "quality_eligible"] = False
timing = extract_conduction_timing(pqrst, fs)
information = extract_conduction_information(pqrst, fs, eligibility_column="quality_eligible")
timing.beat_features.to_csv(RUN / "timing_v1_beats.csv", index=False)
timing.rolling_features.to_csv(RUN / "timing_v1_rolling.csv", index=False)
information.information_windows.to_csv(RUN / "information_v2_windows.csv", index=False)
information.qt_nine_beat_context.to_csv(RUN / "qt_nine_beat_context.csv", index=False)
information.summary()
"""),
        ("Plot measured intervals; do not interpret them clinically", """fig, ax = plt.subplots(figsize=(10,4))
for c in ["pr_interval_ms","qrs_duration_ms","qt_interval_ms"]: ax.plot(timing.beat_features[c], label=c)
ax.axvline(12, color="gray", ls="--", label="externally ineligible beat"); ax.legend(); ax.set_xlabel("beat"); ax.set_ylabel("ms")
fig.tight_layout(); fig.savefig(RUN / "b3_diagnostic.png", dpi=160); plt.show()
assert not information.summary()["final_feature_matrix_eligible"]
"""),
    ],
    "B4_signal_quality": [
        ("Compare clean and contaminated signals with the production deterministic SQIs", """from ecg_cascade.signal_quality import extract_signal_quality_window
RUN = ROOT / "output" / "notebook_runs" / "B4_signal_quality"; RUN.mkdir(parents=True, exist_ok=True)
fs = 250; t = np.arange(10*fs)/fs; peaks = np.arange(fs, len(t), fs)
clean = 0.02*np.sin(2*np.pi*.5*t)
for peak in peaks: clean += np.exp(-0.5*((np.arange(len(t))-peak)/5.0)**2)
rng = np.random.default_rng(42); noisy = clean + .30*rng.normal(size=clean.size) + .25*np.sin(2*np.pi*50*t)
clean_result = extract_signal_quality_window(clean, fs, uv_per_input_unit=1000.0, primary_peak_samples=peaks, secondary_peak_samples=peaks+2)
noisy_result = extract_signal_quality_window(noisy, fs, uv_per_input_unit=1000.0, primary_peak_samples=peaks, secondary_peak_samples=peaks+2)
records = pd.DataFrame([{"condition":"clean", **clean_result.to_record()}, {"condition":"noisy", **noisy_result.to_record()}])
records.to_csv(RUN / "signal_quality_records.csv", index=False)
records[["condition","finite_fraction","psqi_clifford","bassqi_clifford","bsqi_li2008_jaccard_150ms"]]
"""),
        ("Plot the signals and verify prerequisite-aware missingness", """fig, axes = plt.subplots(2,1,figsize=(10,5),sharex=True)
axes[0].plot(t,clean); axes[0].set_title("Clean synthetic input"); axes[1].plot(t,noisy); axes[1].set_title("Contaminated input"); axes[1].set_xlabel("seconds")
fig.tight_layout(); fig.savefig(RUN / "b4_diagnostic.png", dpi=160); plt.show()
assert clean_result.to_record()["rail_fraction__available"] is False
assert (RUN / "signal_quality_records.csv").exists()
"""),
    ],
}


def write_notebooks() -> None:
    for branch, spec in BRANCHES.items():
        cells = [
            markdown_cell(f"# {spec['code']} — {spec['title']}\n\nExecutable, heavily annotated walkthrough. Production modules are imported as the single source of truth; this notebook does not reimplement branch algorithms."),
            markdown_cell("## 1. Reproducible setup\n\nFind the repository root without depending on a hidden working directory, add the maintained `src` tree, and create only branch-specific demonstration artifacts."),
            code_cell(COMMON_SETUP),
            markdown_cell("## 2. Complete branch source inventory\n\nThe source map names every production module, research script, and test assigned to this branch. Shared modules are explicitly flagged rather than copied."),
            code_cell(f"source_map = pd.read_csv(ROOT / 'output' / 'branches' / '{branch}' / 'source_map.csv')\nassert source_map['exists'].all(), source_map.loc[~source_map['exists']]\nsource_map"),
            markdown_cell("## 3. Inputs and contracts\n\nThe example below uses deterministic local synthetic data so clean-kernel execution does not depend on downloads. It demonstrates data shape, units, intermediate tables, outputs, and missingness; dataset benchmark claims remain in the technical record."),
        ]
        for heading, code in NOTEBOOK_CODE[branch]:
            cells.extend([markdown_cell(f"### {heading}"), code_cell(code)])
        cells.extend([
            markdown_cell("## 4. Failure modes and evidence status\n\nCommon hazards include wrong sampling rate or units, anchor/lead mismatch, insufficient causal history, edge truncation, non-finite inputs, and treating an observational measurement as a classifier. The assertions above verify expected files and branch-specific non-claims."),
            code_cell("produced = sorted(str(p.relative_to(ROOT)) for p in RUN.glob('*'))\nassert produced\nproduced"),
        ])
        nb = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":sys.version.split()[0]}})
        nbformat.write(nb, OUTPUT / "branches" / branch / spec["notebook"])


def parse_wfdb_header(path: Path) -> dict[str, object] | None:
    try:
        lines = path.read_text(encoding="latin-1").splitlines()
        if not lines: return None
        first = lines[0].split()
        n_sig = int(first[1]); fs_token = first[2].split("/")[0]
        fs = float(fs_token.strip("()")); n_samples = first[3] if len(first) > 3 else ""
        leads = [line.split()[-1] for line in lines[1:1+n_sig] if line.split()]
        return {"relative_path":path.relative_to(ROOT).as_posix(),"record":first[0],"signal_count":n_sig,"sampling_rate_hz":fs,"sample_count":n_samples,"lead_labels":"|".join(leads)}
    except Exception:
        return None


def write_dataset_indexes() -> None:
    index_dir = OUTPUT / "evidence" / "indexes"; index_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = ROOT / "Datasets"
    file_rows = []
    for path in sorted(dataset_root.rglob("*")):
        if path.is_file():
            rel = path.relative_to(ROOT).as_posix(); parts = path.relative_to(dataset_root).parts
            file_rows.append({"dataset_collection":parts[0] if parts else "","relative_path":rel,"extension":path.suffix.lower(),"size_bytes":path.stat().st_size})
    with (index_dir / "dataset_file_manifest.csv").open("w", newline="", encoding="utf-8") as h:
        w=csv.DictWriter(h, fieldnames=list(file_rows[0])); w.writeheader(); w.writerows(file_rows)

    summaries=[]
    for name in sorted({r["dataset_collection"] for r in file_rows}):
        rows=[r for r in file_rows if r["dataset_collection"]==name]; ext=Counter(r["extension"] or "[none]" for r in rows)
        summaries.append({"dataset_collection":name,"file_count":len(rows),"size_bytes":sum(r["size_bytes"] for r in rows),"header_files":ext[".hea"],"edf_files":ext[".edf"],"annotation_files":sum(v for k,v in ext.items() if k in {".atr",".qrs",".pu",".man",".ari",".ecg",".st",".t0",".t1",".16a"}),"extension_counts":json.dumps(dict(sorted(ext.items())),sort_keys=True)})
    with (index_dir / "dataset_summary.csv").open("w", newline="", encoding="utf-8") as h:
        w=csv.DictWriter(h, fieldnames=list(summaries[0])); w.writeheader(); w.writerows(summaries)

    headers=[row for p in dataset_root.rglob("*.hea") if (row:=parse_wfdb_header(p))]
    with (index_dir / "wfdb_header_inventory.csv").open("w", newline="", encoding="utf-8") as h:
        w=csv.DictWriter(h, fieldnames=list(headers[0])); w.writeheader(); w.writerows(headers)

    config_path=ROOT/"feature_extraction"/"config"/"datasets.toml"
    config=tomllib.loads(config_path.read_text(encoding="utf-8")); configured=[]
    for rec in config.get("recordings",[]):
        configured.append({"id":rec.get("id"),"patient":rec.get("patient"),"configured_path":rec.get("path"),"local_path_exists":Path(rec.get("path","")).exists(),"ecg_channel":rec.get("ecg_channel"),"expected_sampling_rate_hz":rec.get("expected_sampling_rate_hz"),"seizure_intervals_s":json.dumps(rec.get("seizures_s",[]))})
    with (index_dir / "configured_recordings.csv").open("w", newline="", encoding="utf-8") as h:
        w=csv.DictWriter(h, fieldnames=list(configured[0])); w.writeheader(); w.writerows(configured)


def write_tex_sources() -> None:
    (OUTPUT/"architecture"/"Architecture.tex").write_text(tex_document("ECGdetector Architecture", "Four operational coding branches and future temporal fusion", ARCHITECTURE_BODY), encoding="utf-8")
    for branch,spec in BRANCHES.items():
        source_table="\\section{Branch source inventory}\\small\\begin{longtable}{p{0.14\\linewidth}p{0.76\\linewidth}}\\toprule Role & Maintained path \\\\ \\midrule\n"
        for role,base,names in (("Production",SRC,spec["production"]),("Script",SCRIPTS,spec["scripts"]),("Test",TESTS,spec["tests"])):
            for name in names: source_table += f"{role} & \\path{{{(base/name).relative_to(ROOT).as_posix()}}} \\\\\n"
        source_table += "\\bottomrule\\end{longtable}\\normalsize\n"
        body=BRANCH_BODIES[branch]+source_table+r"""
\section{Reproduction and acceptance}
Run the branch notebook from a clean kernel. It imports the production modules, writes only to its named \code{output/notebook\_runs} directory, and asserts the expected artifacts/non-claims. The full test suite, LaTeX build status, PDF validation, and checksums are recorded in \code{output/qa} and \code{output/evidence/indexes}.
"""
        (OUTPUT/"branches"/branch/spec["tex"]).write_text(tex_document(f"{spec['code']} Technical Record",spec["title"],body),encoding="utf-8")
    dataset_dir=OUTPUT/"evidence"/"datasets"; dataset_dir.mkdir(parents=True,exist_ok=True)
    (dataset_dir/"Datasets_and_Branch_Use_Technical_Record.tex").write_text(tex_document("Datasets and Branch Use","B1--B4 technical record",DATASET_BODY),encoding="utf-8")


def main() -> None:
    for path in [OUTPUT/"architecture",*(OUTPUT/"branches"/b for b in BRANCHES),OUTPUT/"evidence"/"datasets",OUTPUT/"evidence"/"indexes",OUTPUT/"qa"]: path.mkdir(parents=True,exist_ok=True)
    write_source_maps(); write_notebooks(); write_dataset_indexes(); write_tex_sources()
    (OUTPUT/"qa"/"bundle_generation.json").write_text(json.dumps({"status":"success","branch_count":4,"latex_source_count":6,"notebook_count":4},indent=2),encoding="utf-8")
    print(json.dumps({"status":"success","latex_sources":6,"notebooks":4,"dataset_files":sum(1 for p in (ROOT/"Datasets").rglob("*") if p.is_file())},indent=2))


if __name__ == "__main__":
    main()
