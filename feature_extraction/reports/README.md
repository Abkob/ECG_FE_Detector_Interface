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

## Scope rule

Included files are project-authored reports about the ECG cascade, its feature-extraction branches, or their validation. Published literature PDFs remain in the literature/reference folders. Duplicate LaTeX builds, temporary QA revisions, source-input PDFs, and plot-only PDF exports are not duplicated here.
