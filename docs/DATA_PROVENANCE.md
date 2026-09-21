# Provenance of the hosted audit results

Checked on 21 September 2026. This describes the existing audit exports under
`feature_extraction/src/ecg_cascade/web_demo`, not arbitrary future uploads.

The record/subject identifiers in these exports are public benchmark identifiers.
The exports contain extracted measurements, source labels, model predictions,
and aggregate error metrics. The `patient_metrics.csv` filename refers to grouped
benchmark evaluation; it does not imply a private clinical patient registry.

| Export dataset key | Public source | Records represented in the selected prediction export |
|---|---|---|
| `mitdb` | [MIT-BIH Arrhythmia Database 1.0.0](https://physionet.org/content/mitdb/1.0.0/) | All 48 standard record IDs. |
| `nstdb` | [MIT-BIH Noise Stress Test Database 1.0.0](https://physionet.org/content/nstdb/1.0.0/) | Twelve noisy derivatives of records 118 and 119. |
| `butqdb` | [Brno University of Technology ECG Quality Database 1.0.0](https://physionet.org/content/butqdb/1.0.0/) | Eighteen public recordings; six-digit IDs such as 100001 and 126001. |
| `qtdb` | [QT Database 1.0.0](https://www.physionet.org/content/qtdb/1.0.0/) | Twelve public `sel*` records: 100, 103, 116, 117, 123, 213, 221, 223, 230, 231, 232 and 233. |
| `seizure_edf` | [CHB-MIT Scalp EEG Database 1.0.0](https://physionet.org/content/chbmit/1.0.0/chb04/) | `chb04_28`. |
| `seizure_edf` | [Siena Scalp EEG Database 1.0.0](https://physionet.org/content/siena-scalp-eeg/1.0.0/) | Five PN06 and three PN12 files; local IDs use lowercase and underscores. The [PN06](https://physionet.org/content/siena-scalp-eeg/1.0.0/PN06/) and [PN12](https://physionet.org/content/siena-scalp-eeg/1.0.0/PN12/) directories are public. |

The CHB-MIT documentation states that identifying information was replaced with
surrogate information. Siena documents de-identified dates and public subject
codes. See the linked original dataset pages for authorship, citations, versions,
licenses and access terms. The project-specific extraction, grouping, label
mapping and inference steps are documented in
[`23_RIGOROUS_CLASSIFICATION_AUDIT.md`](../feature_extraction/docs/23_RIGOROUS_CLASSIFICATION_AUDIT.md).

The selected prediction export contains 46,038 rows: 13,347 BUT-QDB,
7,852 NSTDB, 11,346 seizure-EDF, 10,867 MIT-BIH and 2,626 QTDB rows. The weak-case
export contains 7,162 rows from the same dataset namespaces. These are derived
research results, not independent clinical validation or calibrated diagnoses.

`source_path` records the original workstation's location of a downloaded public
recording. It is provenance metadata, not a web URL or a credential. Raw corpora,
private upload storage, database credentials, and environment files remain outside
this commit. The small public MIT-BIH record 100 was already tracked for the demo.

The configured publication destination is the existing public repository
[`Abkob/ECG_FE_Detector_Interface`](https://github.com/Abkob/ECG_FE_Detector_Interface),
verified through GitHub repository metadata. The user explicitly requested
committing and pushing the current project and established the standing rule
to push after future commits.
