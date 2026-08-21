# Signal-quality measurement branch v0

## Outcome and scope

The implemented branch calculates a paper-named vector from each ECG window
and carries a boolean availability field and machine-readable reason beside
every value.  It does not label a window clean/artifact, learn a threshold, or
produce a probability.  Those steps require independent artifact annotations,
patient-wise partitions, and calibration data.

The public entry points are:

- `extract_signal_quality_window`: one already selected window;
- `extract_trailing_signal_quality_windows`: trailing windows timestamped at
  their exclusive end sample, with recording-relative events sliced and
  shifted without reading future samples;
- small equation functions for direct unit tests and ablations.

Amplitude-valued calls require `uv_per_input_unit`.  This prevents a WFDB mV
signal, an EDF µV signal, and ADC counts from silently sharing one numerical
scale.

## v0 feature schema

| Output | Implemented definition and availability |
|---|---|
| `finite_fraction` | Finite samples divided by all window samples. |
| `rail_fraction`, `longest_rail_run_s` | Samples within 1% of supplied ADC rails, with Redmond's ±1-s mask dilation. Unavailable without both rails. |
| `flat_fraction`, `longest_flat_run_s` | Runs constant within the explicit tolerance for at least 1 s. Default tolerance is exact equality in the supplied representation. |
| `absmax_uv`, `peak_to_peak_uv`, `rms_uv` | Continuous amplitude descriptors in µV. |
| `saturation_fraction_langley2011`, `longest_saturation_run_s_langley2011` | Named comparator for absolute amplitude above 2 mV continuously for more than 200 ms. It is not promoted to a universal limited-lead rule. |
| `hf_rms_uv` | RMS of the Redmond-style 50-Hz-notch, fifth-order elliptic >40-Hz, zero-phase, 50-ms Hamming-smoothed envelope. |
| `hf_contaminated_fraction` | Fraction above a caller-supplied device threshold. Unavailable by default because Redmond's published threshold was ADC/device tuned. |
| `bassqi_clifford` | `1 - P(0–1 Hz)/P(0–40 Hz)`. |
| `psqi_clifford` | `P(5–15 Hz)/P(5–40 Hz)`. |
| `sdr_li2008`, `ssqi_li2008_binary` | Separate `P(5–14 Hz)/P(5–50 Hz)` sensitivity analysis and Li's published 0.5–0.8 flag. |
| `skewness` | Biased population third standardized moment. |
| `pearson_kurtosis`, `ksqi_li2008_binary` | Biased population fourth standardized moment and the separately named `>5` rule. |
| `bsqi_li2008_jaccard_150ms` | `M/(N1+N2-M)` with one-to-one ±150-ms matching. |
| `qsqi_zhao2018_dice_75ms` | `2M/(N1+N2)`; 75 ms is the named Kristof comparator setting, not a tolerance frozen by Zhao. |
| `rr_support_ho2024_150ms` | Fraction of primary RR intervals whose two endpoint QRS events each have exactly one secondary event within ±150 ms. No extra within-interval event-count rule. |
| `isqi_li2008_max_jaccard_150ms` | Maximum same-detector cross-lead Jaccard agreement plus unmatched fractions for the selected lead pair. |
| `template_corr_orphanidou` | Mean Pearson correlation between complete median-RR-wide beats and their mean template. |
| `baseline_rms_galeotti` | RMS of a cubic spline through pre-QRS-onset isoelectric anchors (20 ms at 50 Hz, 16 ms at 60 Hz). Requires QRS onsets and mains-region metadata. |
| `mains_rms_galeotti` | RMS of beat-specific sine/cosine fits at the supplied 50/60-Hz frequency after baseline removal. |
| `residual_rms_galeotti` | Mean dominant-group beat-to-median-template RMS after baseline and mains removal. Requires external morphology-group labels. |
| `dominant_beat_excluded_fraction` | Complete beats outside the supplied dominant group; no beat is deleted from another branch. |
| `menon_fourier_score` | Integrated squared ECG/template cross-correlation divided by QRS count, using a fifth-order Fourier beat approximation after ≤5% sequential coefficient change. |

Raw-ECG wavelet energy/entropy and HRV wavelet entropy are not emitted.  The
source publications do not justify one interchangeable definition.  The
Moeyersons `a_Q` classifier score is also absent until a model is trained; a
native score will not be called a probability unless separately calibrated.

## Paper-exact parts versus frozen transfer choices

The band equations, statistical moments, agreement equations, 150-ms Li/Ho
tolerances, Ho endpoint rule, isoelectric durations, mains regression equation,
and residual RMS equation are directly testable implementations of the audited
definitions.

Several source reports do not freeze every software detail needed for identical
numeric reproduction.  v0 records the following transfer choices instead of
hiding them:

- Welch PSD: periodic Hann windows, 4-s segments, 50% overlap, constant
  detrending;
- high-frequency proxy: notch Q 30, elliptic ripple 0.5 dB, stop attenuation
  40 dB;
- Galeotti spline: natural cubic boundary conditions, evaluated over the full
  window;
- Galeotti morphology grouping: supplied by the caller because the paper does
  not freeze a transferable grouping threshold;
- Menon representative beat: running mean of complete median-RR-centred beats;
  the fifth-order coefficient convergence and published score construction are
  retained, but author-code equivalence still requires a direct comparator.

These fields may be used as explicit ablations.  They must not be described as
bit-for-bit author-code replications.

## Automated validation

`tests/test_signal_quality.py` includes hand-computed population moments,
known-frequency spectral signals, exact flat/rail durations, distinct bSQI and
qSQI equations, the Ho endpoint counterexample with an extra secondary event
inside an RR interval, identical-beat template correlation, controlled
baseline/mains recovery, externally grouped residual RMS, Menon convergence,
missing-prerequisite schemas, a full conditional-feature construction, and a
future-data causality test.

The local diagnostic script is:

```powershell
& '.venv/Scripts/python.exe' 'scripts/validate_signal_quality_nstdb.py'
```

It evaluated two clean MIT-BIH windows and 12 NSTDB electrode-motion windows
(records 118/119, +24 through -6 dB).  Across the 12 noisy windows, Spearman
correlations with improving SNR were:

| Feature | Spearman rho |
|---|---:|
| `bassqi_clifford` | 0.975 |
| `qsqi_zhao2018_dice_75ms` | 0.829 |
| `bsqi_li2008_jaccard_150ms` | 0.789 |
| `rr_support_ho2024_150ms` | 0.590 |
| `template_corr_orphanidou` | 0.523 |
| `menon_fourier_score` | -0.127 |

Amplitude and high-frequency RMS increased as SNR worsened (rho -0.961 and
-0.848 with improving SNR).  This is useful controlled-degradation behavior,
not classifier performance.  The nearly non-monotonic Menon result supports
keeping it as a comparator rather than a hard veto.

## Remaining BUT-QDB gate

BUT-QDB is still required to assess manually labelled artifact windows and
artifact types.  The first classifier experiment must use patient-wise splits,
fit all normalization and imputation only on training patients, compare a
small regularized model against single-feature baselines, report precision-
recall metrics, and preserve clean-seizure versus seizure-plus-artifact strata.
No threshold learned from NSTDB SNR is transferred as a clinical cutoff.
