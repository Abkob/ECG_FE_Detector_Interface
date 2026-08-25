# ECG feature-extraction documentation

This directory is the technical documentation for the code in
`feature_extraction/src/ecg_cascade`. It is intentionally more detailed than
the package-level `README.md`. The package README answers "how do I run it?";
these documents answer "what does every step mean, what evidence supports it,
what can fail, and what has not yet been validated?"

## Current software boundary

The implemented package now contains morphology plus two separately callable
autonomic architectures and one rejected experimental fusion ablation:

- the earlier UNSW-primary architecture retained as a control;
- the new independent NeuroKit-versus-Zhai architecture, which calculates two
  complete RR/HRV and Varon-morphology tracks without selecting or fusing
  their timestamps;
- the deterministic Varon-2015 PRSA/BPRSA autonomic feature lane, without the
  paper's kernel-spectral-clustering seizure classifier;
- an UNSW-existence/NeuroKit-then-Zhai timestamp-substitution experiment,
  retained for audit but not accepted as the production timestamp owner.

Across those paths, the package contains:

1. EDF and WFDB loading with explicit channel selection.
2. The author-provided Python UNSW/Khamis detector as primary timestamp owner.
3. NeuroKit secondary detection using the requested 250-ms inclusive experiment
   and the unmodified 300-ms strict baseline.
4. The current Ho two-detector RR-reliability method at +/-50 ms, with the
   earlier +/-150-ms support assumption retained as an ablation.
5. Pan-Tompkins at 300 and 250 ms as context only.
6. Original and inverted signal contexts without automatic polarity routing.
7. Per-QRS support, per-RR reliability, timing offsets, missingness, and
   coverage.
8. Continuous 100-RR CSI/modified-CSI/slope measurements described in the
   Jeppesen phase-3 seizure-detection study.
9. CSV, JSON, annotation-based MIT-BIH metrics, and diagnostic plots.
10. Eighty-five unit tests plus real-data execution checks.
11. A paper-traceable Zhai 2023 reimplementation with exposed filtered signal,
    QRS envelope, dynamic threshold, windows, template, correlation series,
    and correlation-localized timestamps.
12. A full 48-record NeuroKit--Zhai MIT--BIH audit at 75 ms, 25 ms, and one
    sample.
13. Separate NeuroKit- and Zhai-aligned five-QRS Varon eigenvalue sequences.
14. Same-track fusion-ready measurement tables with reliability kept separate.
15. Beat-aligned waveform arrays and detector-association tables for the next
    record-level morphology audit.
16. A complete 48-record UNSW--NeuroKit--Zhai timestamp-fusion ablation,
    including component ablations and source-transition RR errors.
17. A complete 48-record unchanged-UNSW/NeuroKit-context benchmark with
    tolerance ablations, per-RR support, causal qSQI, and difficult-record
    counterexamples.
18. A complete expert-versus-UNSW 100-RR HRV fidelity and simulated
    patient-specific threshold-crossing audit.
19. A complete same-source-per-RR and pure-window fusion audit across all 48
    MIT--BIH records, including transition error, long-run HRV concordance,
    high-threshold preservation, coverage, and tolerance sensitivity.
20. A Varon-2015 120-ms, five-beat morphology implementation plus a strict
    all-48 expert-timestamp feature-fidelity audit for UNSW, NeuroKit, and Zhai.
21. A multi-branch interactive reviewer with record selection, detector-track
    toggles, expert-label highlighting, TP/FP/FN overlays, morphology traces,
    marker hover, region summaries, and keyboard zoom/navigation.
22. A lead-controlled manual delineation gate on all 200 LUDB lead-II records
    and 12 QTDB MLII records, with DWT/CWT landmark coverage, timing errors,
    polarity strata, and fixed-Varon-window QRS capture analysis.
23. A held-out patient-average-width Varon experiment showing that symmetric
    mean-duration personalization reduces complete-QRS capture and must remain
    a negative control rather than replacing the fixed 120-ms branch.
    A corrected one-number calibration using p95 of
    `2*max(onset-to-R, R-to-offset)` improves held-out capture when enough
    patient calibration beats are available.
24. A deterministic patient/lead whole-cycle template bank with raw and
    normalized residuals, beat-to-beat change, and constrained derivative-DTW.
25. A full matched-lead prominence delineation gate plus a separately callable
    interpretable P/QRS/T feature extractor with missingness and ordering flags.
26. A deterministic 80-beat PRSA/BPRSA implementation with 4-Hz resampling,
    acceleration-period anchors, published short- and long-term curve
    measurements, numerical-quality flags, a full 48-record expert-anchor
    MIT--BIH audit, and a rejected low-support CHB04_28 seizure stress test.
27. A controlled clean-through-minus-6-dB MIT--BIH Noise Stress Test Database benchmark
    for the HRV and morphology branches, restricted to fully artifact-contained
    windows and separating direct waveform corruption from detector failure.
28. A standalone conduction/repolarization information branch containing the
    exact nine Diab PQRST peak-timing dynamics, long-format robust five-beat
    descriptions, and Brotherstone-style nine-beat QT context, explicitly
    excluded from the final feature matrix and from clinical/seizure labels.
28. A paper-named signal-quality v0 measurement branch with raw masks,
    spectral/moment SQIs, detector agreement, template consistency, conditional
    Galeotti noise decomposition, Menon comparison, explicit missingness, and
    causal trailing 10-second orchestration.

The package does **not** currently contain:

- a trained seizure classifier;
- a calibrated seizure probability;
- the Jeppesen alarm threshold;
- a validated primary R-peak detector for the eventual seizure patient;
- a signal-quality classifier;
- full QRS onset/Q/R-prime/S/offset delineation;
- clinically validated automatic P/QRS/T features on seizure recordings;
- a trained or clinically validated multibranch fusion model. The browser demo
  contains only a deterministic same-timestamp 31-feature table join.

## Documentation map

| Document | Purpose |
|---|---|
| [01 System architecture](01_SYSTEM_ARCHITECTURE.md) | End-to-end execution flow, module boundaries, time bases, and design contracts. |
| [02 EDF and data layer](02_EDF_AND_DATA_LAYER.md) | Every EDF class and function, channel selection, physical units, segment semantics, and dataset manifest. |
| [03 R-peak detection and agreement](03_RPEAK_DETECTION_AND_AGREEMENT.md) | Both detector pipelines, the matching algorithm, equations, interpretation, and observed failure modes. |
| [04 RR/HRV mathematics and code](04_RR_HRV_MATHEMATICS_AND_CODE.md) | Detailed derivation of every RR/HRV feature and a complete CSV-column dictionary. |
| [05 CLI and output artifacts](05_CLI_AND_OUTPUT_ARTIFACTS.md) | Installation, commands, arguments, output files, JSON schemas, and diagnostic plots. |
| [06 Testing and validation](06_TESTING_AND_VALIDATION.md) | What each automated test proves, what it does not prove, and the required manual/clinical validation protocol. |
| [07 Literature traceability](07_LITERATURE_TRACEABILITY.md) | Paper-to-code mapping, evidence strength, deviations, ambiguities, and alternatives. |
| [08 Developer extension guide](08_DEVELOPER_EXTENSION_GUIDE.md) | How to add signal-quality, morphology, and conduction branches without breaking auditability. |
| [09 Complete RR-HRV architecture](09_COMPLETE_RR_HRV_ARCHITECTURE.md) | Current detector roles, 250/300-ms and 50/150-ms ablations, polarity handling, output schemas, real MIT-BIH findings, and remaining validation gates. |
| [10 NeuroKit--Zhai two-track architecture](10_NEUROKIT_ZHAI_TWO_TRACK_ARCHITECTURE.md) | New no-fusion architecture, complete Zhai equations/code mapping, implementation ambiguities, commands, and all-48-record audit. |
| [11 Dual-track RR/HRV and Varon morphology](11_DUAL_TRACK_RR_HRV_VARON_MORPHOLOGY.md) | Implemented five-QRS branch, context fields, fusion-ready tables, output artifacts, and the next error audit. |
| [12 All-48 true-R-peak distance audit](12_ALL48_TRUE_RPEAK_DISTANCE_AUDIT.md) | Expert-annotation timing distances, FP/FN context, all-record outputs, and critical interpretation. |
| [13 UNSW--NeuroKit--Zhai fusion ablation](13_UNSW_NEUROKIT_ZHAI_FUSION_ABLATION.md) | Exact substitution rule, component ablations, all-48 results, source-transition failure, and rejection decision. |
| [14 UNSW-owned RR with NeuroKit context](14_UNSW_NEUROKIT_CONTEXT_BENCHMARK.md) | All-48 test of the proposed no-substitution architecture, support-tolerance ablation, qSQI limitations, and baseline decision. |
| [15 Expert-versus-UNSW HRV fidelity](15_EXPERT_UNSW_HRV_FIDELITY.md) | All-48 long-run J1/J2 concordance, high-threshold preservation, normal/abnormal-beat stratification, and the patient-validation gate. |
| [16 Interval-consistent fusion audit](16_INTERVAL_CONSISTENT_FUSION_AUDIT.md) | All-48 same-source-per-RR, pure-window, source-transition, long-run HRV, threshold, coverage, and tolerance comparison. |
| [17 Morphology fidelity and reviewer](17_MORPHOLOGY_FIDELITY_AND_REVIEWER.md) | Strict all-48 Varon-feature fidelity audit, detector tradeoffs, counterexamples, reviewer controls, and the patient-validation gate. |
| [18 Matched-lead delineation gate](18_DELINEATION_GATE_MATCHED_LEAD_V1.md) | Manual LUDB/QTDB P/QRS/T validation, DWT/CWT results, polarity effects, fixed 120-ms capture coverage, rejection decision, and next challengers. |
| [19 Personalized morphology branch](19_PERSONALIZED_MORPHOLOGY_BRANCH_V1.md) | Patient-average Varon negative result, whole-cycle template bank, prominence gate, interpretable features, code, outputs, and remaining seizure-validation boundary. |
| [20 Conduction timing branch](20_CONDUCTION_TIMING_BRANCH_V1.md) | Standalone PR/QRS/QT/JT and beat-to-beat timing prototype, literature traceability, causal summaries, equations, and the interval-validation gate. |
| [21 Conduction information branch](21_CONDUCTION_INFORMATION_BRANCH_V2.md) | Evidence-narrowed information-only output: exact raw PQRST peak dynamics, robust five-beat descriptions, nine-beat QT context, quality input contract, and no bad-change label. |
| [21 Signal-quality branch v0](21_SIGNAL_QUALITY_BRANCH_V0.md) | Implemented feature schema, paper/project boundaries, exact tolerance names, synthetic tests, NSTDB behavior, and the remaining BUT-QDB label gate. |
| [22 PRSA/BPRSA context validation](22_PRSA_BPRSA_CONTEXT_VALIDATION.md) | Equation and causal-matrix correctness, all-48 expert-anchor reproducibility, lead/polarity sensitivity, controlled electrode-motion noise, arrhythmia confounding, CHB seizure reliability, and the remaining clinical-validation gate. |
| Controlled artifact-degradation benchmark (`../outputs/artifact_degradation_nstdb_v1/REPORT.md`, generated locally) | Clean, +24 to -6 dB electrode-motion results for UNSW-owned HRV and oracle/UNSW/NeuroKit/Zhai morphology, with pooled and per-source tables. |
| Signal-quality NSTDB v0 diagnostic (`../outputs/signal_quality_nstdb_v0/REPORT.md`, generated locally) | Feature calculability and rank response across clean 118/119 and controlled +24 to -6 dB electrode-motion windows. |
| [Complete morphology technical record](latex_morphology_branch/morphology_branch_complete_technical_record.tex) | Auditable morphology and PRSA/BPRSA specifications, every implemented and rejected trial, exact results, sources, and remaining validation limits. |

## Evidence-status vocabulary

The documents use four status labels:

- **Published specification**: explicitly described in a cited paper.
- **Implementation decision**: required to make the software executable but
  not uniquely specified by the paper.
- **Software-verified**: covered by an automated test or successful execution.
- **Physiologically unvalidated**: the code runs, but accuracy against manual
  ECG annotations or clinical endpoints has not been established.

These labels must not be collapsed. A feature can be published and
software-verified while still being physiologically invalid on a particular
recording because its R peaks were wrong.

## Most important current conclusion

The baseline keeps every UNSW timestamp unchanged and uses NeuroKit only as
reliability context. Direct expert-versus-UNSW 100-RR testing produced pooled
CCC 0.990 for J1 and 0.994 for J2, but simulated 99th-percentile threshold F1
fell to 0.707 and 0.750 and several records failed badly. Requiring complete
detector agreement increased conditional concordance but discarded roughly
half of expert-high windows at the 95th percentile. Therefore, unchanged UNSW
remains the timestamp owner and agreement remains context rather than a veto.
The branch is ready for patient-specific validation, not for a claimed seizure
threshold.

The interval-consistent fusion audit subsequently reduced pooled RR MAE to
3.182 ms, but retained 23.155-ms successive-change error at source transitions
and worsened q95 J2 threshold F1. Pure candidate windows were highly concordant
but covered only 71% (NeuroKit) or 57% (Zhai) of windows. Therefore, these
methods remain parallel research challengers; they do not replace unchanged
UNSW as the conservative timestamp owner.

The detailed evidence and next gate are recorded in
[`../INITIAL_AUDIT.md`](../INITIAL_AUDIT.md).
