# ECGdetector

ECGdetector is a research implementation organized around four coding branches:

1. **B1 — Peak detection, RR, and HRV**
2. **B2 — Beat morphology**
3. **B3 — Conduction and repolarization**
4. **B4 — Signal quality**

The maintained Python package and tests live under `feature_extraction/`. Raw local corpora remain under `Datasets/`. The dated pre-cleanup repository state is preserved under `archive/pre_cleanup_2026-08-19/`; no source was permanently deleted during the cleanup.

The reproducible documentation bundle is under `output/`. Start with `output/architecture/Architecture.pdf`, then use the exhaustive branch dossiers (B1 147 pages, B2 148 pages, B3 101 pages, B4 81 pages) and their executable walkthrough notebooks. Together they map all 77 first-party Python files and include complete symbol, test, saved-result, historical-report, case-study, and evidence ledgers. Evidence indexes distinguish project artifacts, dataset records, case studies, literature, unresolved material, and quarantined invalid files.

## Web demonstration

The clinician-facing research walkthrough is deployed at
[ecg-cascade-demo.vercel.app](https://ecg-cascade-demo.vercel.app). Vercel
builds the static interface and Python API from this repository. Neon stores
the latest compressed analysis for each browser session. Hosted ECG files are
uploaded through one-hour signed URLs to a private Vercel Blob store, read only
inside the analysis function, and deleted after successful extraction. Neon
does not store raw ECG uploads.

Raw datasets, local credentials, generated outputs, rendered reports, and
historical/reference collections are excluded by `.gitignore`. The only raw
recording files tracked are MIT-BIH record 100 (`100.hea` and `100.dat`), which
are required by the prepared public demonstration.

## Reproduce

From PowerShell at the repository root:

```powershell
feature_extraction\.venv\Scripts\python.exe tools\build_documentation_bundle.py
feature_extraction\.venv\Scripts\python.exe tools\build_branch_dossiers.py
python tools\audit_evidence.py
feature_extraction\.venv\Scripts\python.exe tools\compile_latex_bundle.py --include-canonical-evidence
feature_extraction\.venv\Scripts\python.exe tools\execute_notebooks.py
feature_extraction\.venv\Scripts\python.exe -m pytest -q feature_extraction\tests
python tools\render_pdf_qa.py
feature_extraction\.venv\Scripts\python.exe tools\verify_bundle.py
```

Run these commands from the repository root. The audit and render steps require Python with `pypdf`/Pillow and the detected Poppler utilities. LaTeX and PDF QA commands, statuses, and outputs are recorded in `output/qa/`.

## Status vocabulary

- **Implemented:** present in maintained production modules and exercised by tests.
- **Experimental:** implemented as an analysis path but not accepted as a production decision rule.
- **Held:** preserved as evidence but explicitly excluded from the live design.
- **Proposed:** architecture or research work that has not been implemented or validated.
