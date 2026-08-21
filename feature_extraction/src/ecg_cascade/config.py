"""Configuration for the evidence-traceable RR--HRV branch."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class NeuroKitZhaiConfig:
    """Configuration for the independent NeuroKit-versus-Zhai architecture.

    Neither track is declared the RR owner.  The architecture calculates both
    RR/HRV series and leaves ownership to annotation-based validation.
    """

    processing_orientation: str = "original"
    compute_inverted_context: bool = False
    neurokit_minimum_delay_ms: float = 300.0
    neurokit_minimum_delay_inclusive: bool = False
    support_tolerance_ms: float = 50.0
    audit_match_tolerance_ms: float = 75.0
    hrv_window_rr_intervals: int = 100
    causal_rr_median_width: int = 7
    reliable_hrv_coverage: float = 1.0
    # The primary implementation follows the later peer-reviewed Varon 2015
    # study: a symmetric 120-ms capture.  The 2013 80-ms version remains an
    # explicit ablation by passing 40 ms on each side.
    varon_pre_r_ms: float = 60.0
    varon_post_r_ms: float = 60.0
    varon_stack_beats: int = 5
    reliable_morphology_support_coverage: float = 1.0

    def __post_init__(self) -> None:
        if self.processing_orientation not in {"original", "inverted"}:
            raise ValueError("processing_orientation must be 'original' or 'inverted'")
        for name in [
            "neurokit_minimum_delay_ms",
            "support_tolerance_ms",
            "audit_match_tolerance_ms",
        ]:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.hrv_window_rr_intervals < 3:
            raise ValueError("hrv_window_rr_intervals must be at least 3")
        if self.causal_rr_median_width <= 0:
            raise ValueError("causal_rr_median_width must be positive")
        if not 0 <= self.reliable_hrv_coverage <= 1:
            raise ValueError("reliable_hrv_coverage must be between 0 and 1")
        if self.varon_pre_r_ms <= 0 or self.varon_post_r_ms <= 0:
            raise ValueError("Varon QRS capture durations must be positive")
        if self.varon_stack_beats != 5:
            raise ValueError(
                "The paper-traceable Varon core requires varon_stack_beats=5"
            )
        if not 0 <= self.reliable_morphology_support_coverage <= 1:
            raise ValueError(
                "reliable_morphology_support_coverage must be between 0 and 1"
            )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.update(
            {
                "timestamp_owner": "unselected_pending_annotation_validation",
                "raw_amplitude_argmax_refinement": False,
                "automatic_detector_fusion": False,
                "automatic_polarity_routing": False,
                "morphology_core": (
                    "Varon_2015_five_QRS_symmetric_120ms_Gram_eigenvalues"
                ),
                "delineation_lane": "not_run_separate_future_integration",
                "fusion_status": "measurements_joined_no_classifier",
                "zhai_status": (
                    "independent_paper_reimplementation; no official author "
                    "code located"
                ),
            }
        )
        return payload


@dataclass(frozen=True)
class RRHRVConfig:
    """Parameters which materially change RR/HRV extraction.

    The defaults implement the current architecture, while keeping every
    experimental choice visible in metadata.  In particular, 250 ms is the
    requested experimental NeuroKit delay; the unmodified 300-ms result is
    always computed as a baseline.
    """

    primary_detector: str = "unsw"
    secondary_detector: str = "neurokit"
    processing_orientation: str = "original"
    compute_inverted_context: bool = True
    neurokit_experimental_min_delay_ms: float = 250.0
    neurokit_baseline_min_delay_ms: float = 300.0
    min_delay_inclusive: bool = True
    close_detection_exclusion_ms: float = 150.0
    support_tolerance_ms: float = 50.0
    legacy_support_tolerance_ms: float = 150.0
    r_fiducial_refinement_ms: float = 50.0
    audit_match_tolerance_ms: float = 75.0
    hrv_window_rr_intervals: int = 100
    causal_rr_median_width: int = 7
    reliable_hrv_coverage: float = 1.0
    pan_context_min_delays_ms: tuple[float, ...] = (300.0, 250.0)

    def __post_init__(self) -> None:
        if self.primary_detector != "unsw":
            raise ValueError("The evidence-backed primary detector is 'unsw'")
        if self.secondary_detector != "neurokit":
            raise ValueError("The evidence-backed secondary detector is 'neurokit'")
        if self.processing_orientation not in {"original", "inverted"}:
            raise ValueError("processing_orientation must be 'original' or 'inverted'")
        for name in [
            "neurokit_experimental_min_delay_ms",
            "neurokit_baseline_min_delay_ms",
            "close_detection_exclusion_ms",
            "support_tolerance_ms",
            "legacy_support_tolerance_ms",
            "r_fiducial_refinement_ms",
            "audit_match_tolerance_ms",
        ]:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.hrv_window_rr_intervals < 3:
            raise ValueError("hrv_window_rr_intervals must be at least 3")
        if self.causal_rr_median_width <= 0:
            raise ValueError("causal_rr_median_width must be positive")
        if not 0 <= self.reliable_hrv_coverage <= 1:
            raise ValueError("reliable_hrv_coverage must be between 0 and 1")
        if not self.pan_context_min_delays_ms:
            raise ValueError("At least one Pan--Tompkins context delay is required")
        if any(delay <= 0 for delay in self.pan_context_min_delays_ms):
            raise ValueError("Pan--Tompkins context delays must be positive")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["pan_context_min_delays_ms"] = list(self.pan_context_min_delays_ms)
        payload["neurokit_250_status"] = "experimental_requested_configuration"
        payload["neurokit_300_status"] = "unmodified_software_baseline"
        payload["support_tolerance_status"] = (
            "50_ms_strict_project_ablation; 150_ms_is_the_Li_bSQI_and_"
            "Ho2024_RR_endpoint_replication_tolerance; legacy_field_name_"
            "retained_only_for_output_schema_compatibility"
        )
        payload["polarity_routing_status"] = (
            "dual-orientation and PhysioZoo/R-DECO fiducial candidates only; "
            "no automatic switching"
        )
        payload["physiozoo_qrs_adjust_status"] = (
            "implemented named ablation; caller-supplied sign; not RR owner"
        )
        payload["rdeco_status"] = (
            "final localization ablation plus planned human GUI correction; "
            "not a full automatic R-DECO replication"
        )
        return payload
