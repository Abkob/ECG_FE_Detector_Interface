# ECG cascade reports

This folder is the curated PDF-report collection for the ECG feature-extraction cascade. The PDFs are copied from their existing build or review locations so that the original LaTeX, DOCX, figures, and build paths continue to work.

| Report | Pages | Role in the project | Original source |
|---|---:|---|---|
| `Architecture.pdf` | 37 | Current full multi-branch architecture working document | `output/pdf/Architecture.pdf` |
| `Architecture_Evidence_Audit.pdf` | 29 | Critical evidence and alternatives audit of the proposed branches | `output/pdf/Architecture_Evidence_Audit.pdf` |
| `RR_HRV_Cascade_RPeak_Audit.pdf` | 56 | Consolidated RR-HRV audit: original detector/failure-case report plus all-48 interval-fusion, downstream HRV, coverage, transition-error, and current implementation results | `RR_HRV_Cascade_Current_Method_Addendum.tex` appended to the preserved pre-interval-fusion edition |
| `Signal_Quality_as_Context_Evidence_Review.pdf` | 6 | Evidence review for using signal quality as contextual evidence in fusion | `_qa_signal_quality_review_v8/Signal_Quality_as_Context_Evidence_Review.pdf` |
| `ECG_Only_Seizure_Detector_Integrated_Research_and_Implementation_Plan.pdf` | 21 | Consolidated multi-branch research, implementation, evaluation, and ablation plan | `tmp/docx_review/integrated_plan/integrated_plan.pdf` |
| `ecg_prearchitecture_draft.pdf` | 30 | Pre-architecture scientific rationale and problem definition | `output/pdf/ecg_prearchitecture_draft.pdf` |
| `ECG_Seizure_Merged_Research_Strategy.pdf` | 12 | Merged research strategy and specific-aims narrative | `output/pdf/ECG_Seizure_Merged_Research_Strategy.pdf` |
| `ECG_Internship_Progress_Report_Week5.pdf` | 6 | Project progress report covering the cascade branches and implementation direction | `output/pdf/ECG_Internship_Progress_Report_Week5.pdf` |

## Current classification-audit deliverables

These final PDFs accompany the current grouped classification audit. Their
editable TeX, figure builders, and required figures are kept in the same folders.
The reports retain their original analysis dates; copying them into Git does
not rerun the experiments or turn cross-validation into an external test.

| Deliverable | PDF | Editable source |
|---|---|---|
| Clinician grouped-audit report | [PDF](clinician_grouped_audit_4page/main.pdf) | [TeX](clinician_grouped_audit_4page/main.tex) |
| Explanation and improvement script | [PDF](ecg_audit_explanation_script/main.pdf) | [TeX](ecg_audit_explanation_script/main.tex) |
| PI feature-matrix briefing | [PDF](pi_seizure_feature_matrix_4page/main.pdf) | [TeX](pi_seizure_feature_matrix_4page/main.tex) |

The [dataset audit workbook](../../outputs/01a066b5-0c49-7dd2-b1f2-2496eef277a7/comprehensive_ecg_branch_dataset_audit.xlsx)
is retained as the original session deliverable. The
[rigorous-audit technical note](../docs/23_RIGOROUS_CLASSIFICATION_AUDIT.md)
documents the current method and limitations.

## Scope rule

Included files are project-authored reports about the ECG cascade, its feature-extraction branches, or their validation. Published literature PDFs remain in the literature/reference folders. Duplicate LaTeX builds, temporary QA revisions, source-input PDFs, and plot-only PDF exports are not duplicated here.
