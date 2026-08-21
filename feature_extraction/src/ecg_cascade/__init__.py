"""Auditable ECG feature-extraction branches."""

from .config import NeuroKitZhaiConfig, RRHRVConfig
from .hrv_fidelity import (
    lin_concordance_correlation,
    symmetric_absolute_percentage_error,
    threshold_agreement_counts,
)
from .neurokit_zhai import (
    NeuroKitZhaiResult,
    run_neurokit_zhai_array,
    run_neurokit_zhai_edf,
    run_neurokit_zhai_wfdb,
    save_neurokit_zhai_outputs,
)
from .pipeline import (
    RRHRVBranchResult,
    run_rr_hrv_array,
    run_rr_hrv_edf,
    run_rr_hrv_segment,
    run_rr_hrv_wfdb,
)
from .morphology import (
    VaronMorphologyResult,
    calibrate_patient_symmetric_varon_width,
    extract_patient_average_varon_morphology,
    extract_patient_required_width_varon_morphology,
    extract_varon_morphology,
)
from .morphology_validation import (
    MorphologyFidelityAudit,
    audit_morphology_timestamp_fidelity,
    build_reference_context,
)
from .patient_template_morphology import (
    PatientTemplateMorphologyResult,
    derivative_dtw_distance,
    extract_patient_template_morphology,
)
from .prominence_morphology import (
    ProminenceMorphologyResult,
    build_pqrst_feature_table,
    extract_prominence_morphology,
)
from .fusion import build_detector_association_table, build_fusion_ready_table
from .fiducial_fusion import FiducialFusionResult, fuse_unsw_neurokit_zhai
from .rr_reliability_context import (
    RRContextResult,
    build_unsw_neurokit_rr_context,
    detector_agreement_fraction,
)
from .rr_hrv import calculate_jeppesen_feature_arrays
from .prsa_bprsa import (
    PRSABPRSAResult,
    calculate_prsa_bprsa_feature_arrays,
    extract_prsa_bprsa_features,
)
from .interval_consistent_fusion import (
    IntervalConsistentFusionResult,
    build_interval_consistent_from_associations,
    fuse_rr_intervals_consistently,
)
from .conduction_timing import (
    ConductionTimingResult,
    extract_conduction_timing,
    qt_correct_bazett,
    qt_correct_framingham,
    qt_correct_fridericia,
    qt_variability_index_hr,
    qt_variability_index_rr,
    short_term_variability_30,
)
from .conduction_information import (
    ConductionInformationResult,
    extract_conduction_information,
)
from .signal_quality import (
    GaleottiBaselineResult,
    GaleottiPowerlineResult,
    GaleottiResidualResult,
    HoRRSupport,
    MenonTemplateResult,
    SignalQualitySeriesResult,
    SignalQualityWindowResult,
    bsqi_li2008_jaccard,
    clifford_baseline_sqi,
    clifford_qrs_power_sqi,
    extract_signal_quality_window,
    extract_trailing_signal_quality_windows,
    galeotti_baseline_wander,
    galeotti_powerline,
    galeotti_residual_noise,
    learn_menon_fourier_template,
    menon_fourier_score,
    pearson_kurtosis,
    population_skewness,
    qsqi_zhao2018_dice,
    rr_support_ho2024,
    template_correlation_orphanidou,
)

__all__ = [
    "RRHRVBranchResult",
    "RRHRVConfig",
    "lin_concordance_correlation",
    "symmetric_absolute_percentage_error",
    "threshold_agreement_counts",
    "NeuroKitZhaiConfig",
    "NeuroKitZhaiResult",
    "VaronMorphologyResult",
    "calibrate_patient_symmetric_varon_width",
    "extract_patient_average_varon_morphology",
    "extract_patient_required_width_varon_morphology",
    "extract_varon_morphology",
    "MorphologyFidelityAudit",
    "audit_morphology_timestamp_fidelity",
    "build_reference_context",
    "PatientTemplateMorphologyResult",
    "derivative_dtw_distance",
    "extract_patient_template_morphology",
    "ProminenceMorphologyResult",
    "build_pqrst_feature_table",
    "extract_prominence_morphology",
    "build_detector_association_table",
    "build_fusion_ready_table",
    "FiducialFusionResult",
    "fuse_unsw_neurokit_zhai",
    "RRContextResult",
    "build_unsw_neurokit_rr_context",
    "detector_agreement_fraction",
    "calculate_jeppesen_feature_arrays",
    "PRSABPRSAResult",
    "calculate_prsa_bprsa_feature_arrays",
    "extract_prsa_bprsa_features",
    "IntervalConsistentFusionResult",
    "build_interval_consistent_from_associations",
    "fuse_rr_intervals_consistently",
    "ConductionTimingResult",
    "extract_conduction_timing",
    "qt_correct_bazett",
    "qt_correct_framingham",
    "qt_correct_fridericia",
    "qt_variability_index_hr",
    "qt_variability_index_rr",
    "short_term_variability_30",
    "ConductionInformationResult",
    "extract_conduction_information",
    "GaleottiBaselineResult",
    "GaleottiPowerlineResult",
    "GaleottiResidualResult",
    "HoRRSupport",
    "MenonTemplateResult",
    "SignalQualitySeriesResult",
    "SignalQualityWindowResult",
    "bsqi_li2008_jaccard",
    "clifford_baseline_sqi",
    "clifford_qrs_power_sqi",
    "extract_signal_quality_window",
    "extract_trailing_signal_quality_windows",
    "galeotti_baseline_wander",
    "galeotti_powerline",
    "galeotti_residual_noise",
    "learn_menon_fourier_template",
    "menon_fourier_score",
    "pearson_kurtosis",
    "population_skewness",
    "qsqi_zhao2018_dice",
    "rr_support_ho2024",
    "template_correlation_orphanidou",
    "run_rr_hrv_array",
    "run_rr_hrv_edf",
    "run_rr_hrv_segment",
    "run_rr_hrv_wfdb",
    "run_neurokit_zhai_array",
    "run_neurokit_zhai_edf",
    "run_neurokit_zhai_wfdb",
    "save_neurokit_zhai_outputs",
]
