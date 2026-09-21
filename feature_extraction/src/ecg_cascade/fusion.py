"""Non-destructive detector association and fusion-ready measurement tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .peaks import PeakAgreement


def build_detector_association_table(
    agreement: PeakAgreement,
    *,
    sampling_rate_hz: float,
    segment_start_s: float,
    zhai_events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return matched and unmatched detector events without choosing a winner.

    Rows are one-to-one temporal associations produced by
    :func:`compare_peak_sequences`.  They are audit relationships, not expert
    labels and not artifact decisions.
    """

    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")
    correlation_by_sample: dict[int, float] = {}
    if zhai_events is not None and not zhai_events.empty:
        required = {"primary_timestamp_sample", "zhai_absolute_correlation"}
        if required.issubset(zhai_events.columns):
            correlation_by_sample = {
                int(sample): float(correlation)
                for sample, correlation in zip(
                    zhai_events["primary_timestamp_sample"],
                    zhai_events["zhai_absolute_correlation"],
                    strict=True,
                )
                if np.isfinite(correlation)
            }

    rows: list[dict[str, object]] = []
    for neurokit_sample, zhai_sample in zip(
        agreement.matched_primary_samples,
        agreement.matched_comparator_samples,
        strict=True,
    ):
        neurokit_sample = int(neurokit_sample)
        zhai_sample = int(zhai_sample)
        rows.append(
            _association_row(
                status="matched",
                neurokit_sample=neurokit_sample,
                zhai_sample=zhai_sample,
                sampling_rate_hz=sampling_rate_hz,
                segment_start_s=segment_start_s,
                zhai_absolute_correlation=correlation_by_sample.get(
                    zhai_sample, np.nan
                ),
            )
        )
    for neurokit_sample in agreement.unmatched_primary_samples:
        rows.append(
            _association_row(
                status="neurokit_only",
                neurokit_sample=int(neurokit_sample),
                zhai_sample=None,
                sampling_rate_hz=sampling_rate_hz,
                segment_start_s=segment_start_s,
                zhai_absolute_correlation=np.nan,
            )
        )
    for zhai_sample in agreement.unmatched_comparator_samples:
        zhai_sample = int(zhai_sample)
        rows.append(
            _association_row(
                status="zhai_only",
                neurokit_sample=None,
                zhai_sample=zhai_sample,
                sampling_rate_hz=sampling_rate_hz,
                segment_start_s=segment_start_s,
                zhai_absolute_correlation=correlation_by_sample.get(
                    zhai_sample, np.nan
                ),
            )
        )
    frame = pd.DataFrame(rows, columns=_ASSOCIATION_COLUMNS)
    if frame.empty:
        frame.insert(0, "association_id", pd.Series(dtype=int))
        return frame
    frame = frame.sort_values("association_time_s", kind="stable").reset_index(
        drop=True
    )
    frame.insert(0, "association_id", np.arange(frame.shape[0], dtype=int))
    frame["neurokit_sample_in_segment"] = frame[
        "neurokit_sample_in_segment"
    ].astype("Int64")
    frame["zhai_sample_in_segment"] = frame["zhai_sample_in_segment"].astype(
        "Int64"
    )
    return frame


def build_fusion_ready_table(
    hrv_features: pd.DataFrame,
    morphology_features: pd.DataFrame,
    *,
    anchor_track: str,
    prsa_bprsa_context: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join same-track measurements and optional PRSA/BPRSA context.

    This is a causal, last-observation measurement join.  It is not Varon's
    incompletely specified normalization/resampling procedure, a trained
    fusion model, or a seizure probability.  Every column is prefixed so the
    slow HRV and fast morphology timescales remain visible.  PRSA/BPRSA is
    attached as autonomic/cardiorespiratory context only: its availability or
    reliability never changes the core morphology--HRV definition or context
    gates.
    """

    required_morphology = {
        "window_end_time_s",
        "published_varon_core_defined",
        "support_context_pass",
    }
    missing_morphology = required_morphology.difference(
        morphology_features.columns
    )
    if missing_morphology:
        raise ValueError(
            "Morphology table is missing required columns: "
            f"{sorted(missing_morphology)}"
        )
    required_hrv = {"end_time_s", "feature_defined", "feature_reliable"}
    missing_hrv = required_hrv.difference(hrv_features.columns)
    if missing_hrv:
        raise ValueError(
            f"HRV table is missing required columns: {sorted(missing_hrv)}"
        )

    morphology = morphology_features.copy()
    morphology = morphology.rename(
        columns={column: f"morph_{column}" for column in morphology.columns}
    )
    morphology["fusion_time_s"] = morphology["morph_window_end_time_s"]
    morphology["fusion_time_s"] = pd.to_numeric(
        morphology["fusion_time_s"], errors="coerce"
    ).astype(float)
    morphology = morphology.sort_values("fusion_time_s", kind="stable")

    hrv = hrv_features.copy()
    hrv = hrv.rename(columns={column: f"hrv_{column}" for column in hrv.columns})
    hrv["hrv_measurement_time_s"] = hrv["hrv_end_time_s"]
    hrv["hrv_measurement_time_s"] = pd.to_numeric(
        hrv["hrv_measurement_time_s"], errors="coerce"
    ).astype(float)
    hrv = hrv.sort_values("hrv_measurement_time_s", kind="stable")

    if hrv.empty:
        joined = morphology.copy()
        joined["hrv_measurement_time_s"] = np.nan
        joined["hrv_feature_defined"] = False
        joined["hrv_feature_reliable"] = False
    else:
        joined = pd.merge_asof(
            morphology,
            hrv,
            left_on="fusion_time_s",
            right_on="hrv_measurement_time_s",
            direction="backward",
            allow_exact_matches=True,
        )

    joined = _attach_prsa_bprsa_context(joined, prsa_bprsa_context)

    joined.insert(0, "anchor_track", anchor_track)
    joined.insert(
        1,
        "timestamp_owner_status",
        "candidate_track_pending_annotation_validation",
    )
    joined["hrv_age_s"] = (
        joined["fusion_time_s"] - joined["hrv_measurement_time_s"]
    )
    morph_defined = joined["morph_published_varon_core_defined"].fillna(
        False
    ).astype(bool)
    hrv_defined = joined["hrv_feature_defined"].fillna(False).astype(bool)
    morph_context = joined["morph_support_context_pass"].fillna(False).astype(
        bool
    )
    hrv_context = joined["hrv_feature_reliable"].fillna(False).astype(bool)
    joined["fusion_measurements_defined"] = morph_defined & hrv_defined
    joined["fusion_context_pass"] = morph_context & hrv_context
    joined["seizure_probability_produced"] = False
    joined["artifact_label_produced"] = False
    joined["fusion_operation"] = (
        "same_track_causal_asof_join_core_measurements_plus_optional_context"
    )
    joined["published_varon_5s_reproduction"] = False
    joined["published_varon_5s_reproduction_note"] = (
        "not_performed_normalization_formula_under_specified"
    )
    return joined.reset_index(drop=True)


def _attach_prsa_bprsa_context(
    fusion_rows: pd.DataFrame,
    context: pd.DataFrame | None,
) -> pd.DataFrame:
    """Causally attach the latest 80-beat PRSA/BPRSA context row."""

    joined = fusion_rows.copy()
    canonical_values = (
        "mean_rr80_ms",
        "sdnn80_ms",
        "prsa_s_rr_ms_per_sample",
        "prsa_delta_rr_ms_per_sample",
        "bprsa_s_r_ms_per_sample",
        "bprsa_delta_r_ms_per_sample",
    )
    if context is None or context.empty:
        joined["prsa_context_measurement_time_s"] = np.nan
        for column in canonical_values:
            joined[f"prsa_context_{column}"] = np.nan
        joined["prsa_context_feature_defined"] = False
        joined["prsa_context_feature_reliable"] = False
    else:
        required = {"end_time_s", "feature_defined", "feature_reliable"}
        missing = required.difference(context.columns)
        if missing:
            raise ValueError(
                "PRSA/BPRSA context table is missing required columns: "
                f"{sorted(missing)}"
            )
        missing_values = set(canonical_values).difference(context.columns)
        if missing_values:
            raise ValueError(
                "PRSA/BPRSA context table is missing the six paper-derived "
                f"values: {sorted(missing_values)}"
            )
        source = context.copy()
        source = source.rename(
            columns={column: f"prsa_context_{column}" for column in source.columns}
        )
        source["prsa_context_measurement_time_s"] = source[
            "prsa_context_end_time_s"
        ]
        source = source.sort_values("prsa_context_measurement_time_s", kind="stable")
        joined = pd.merge_asof(
            joined.sort_values("fusion_time_s", kind="stable"),
            source,
            left_on="fusion_time_s",
            right_on="prsa_context_measurement_time_s",
            direction="backward",
            allow_exact_matches=True,
        )

    joined["prsa_context_age_s"] = (
        joined["fusion_time_s"] - joined["prsa_context_measurement_time_s"]
    )
    joined["prsa_context_available"] = joined[
        "prsa_context_measurement_time_s"
    ].notna()
    joined["prsa_context_defined"] = joined[
        "prsa_context_feature_defined"
    ].eq(True)
    joined["prsa_context_reliable"] = joined[
        "prsa_context_feature_reliable"
    ].eq(True)
    joined["prsa_context_computationally_usable"] = (
        joined["prsa_context_defined"] & joined["prsa_context_reliable"]
    )
    joined["prsa_context_usable"] = joined[
        "prsa_context_computationally_usable"
    ]
    joined["prsa_context_signal_quality_attached"] = False
    joined["prsa_context_model_eligible"] = False
    joined["prsa_context_role"] = "autonomic_cardiorespiratory_context_only"
    joined["prsa_context_affects_core_fusion_gate"] = False
    return joined


def _association_row(
    *,
    status: str,
    neurokit_sample: int | None,
    zhai_sample: int | None,
    sampling_rate_hz: float,
    segment_start_s: float,
    zhai_absolute_correlation: float,
) -> dict[str, object]:
    samples = [
        sample for sample in [neurokit_sample, zhai_sample] if sample is not None
    ]
    association_sample = float(np.mean(samples))
    offset_ms = (
        (zhai_sample - neurokit_sample) * 1000.0 / sampling_rate_hz
        if neurokit_sample is not None and zhai_sample is not None
        else np.nan
    )
    return {
        "agreement_status": status,
        "association_time_s": segment_start_s
        + association_sample / sampling_rate_hz,
        "neurokit_sample_in_segment": (
            neurokit_sample if neurokit_sample is not None else pd.NA
        ),
        "neurokit_time_s": (
            segment_start_s + neurokit_sample / sampling_rate_hz
            if neurokit_sample is not None
            else np.nan
        ),
        "zhai_sample_in_segment": zhai_sample if zhai_sample is not None else pd.NA,
        "zhai_time_s": (
            segment_start_s + zhai_sample / sampling_rate_hz
            if zhai_sample is not None
            else np.nan
        ),
        "zhai_minus_neurokit_ms": offset_ms,
        "absolute_detector_offset_ms": abs(offset_ms),
        "zhai_absolute_correlation": zhai_absolute_correlation,
        "expert_label_available": False,
        "artifact_label_produced": False,
        "timestamp_selected": False,
    }


_ASSOCIATION_COLUMNS = [
    "agreement_status",
    "association_time_s",
    "neurokit_sample_in_segment",
    "neurokit_time_s",
    "zhai_sample_in_segment",
    "zhai_time_s",
    "zhai_minus_neurokit_ms",
    "absolute_detector_offset_ms",
    "zhai_absolute_correlation",
    "expert_label_available",
    "artifact_label_produced",
    "timestamp_selected",
]
