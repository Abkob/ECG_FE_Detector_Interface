# Standalone ECG preprocessing branch notebooks

These four notebooks are independent, detailed, executable views of the
current package implementation:

1. `01_RR_HRV_Standalone_Detailed.ipynb`
2. `02_Morphology_Standalone_Detailed.ipynb`
3. `03_Signal_Quality_Artifact_Context_Standalone_Detailed.ipynb`
4. `04_Conduction_Standalone_Detailed.ipynb`

Each notebook loads its own signal, freezes its own configuration, runs the
real branch functions, displays intermediate and final tables, produces
diagnostic plots, exports CSV/JSON results below
`outputs/standalone_notebooks`, executes explicit acceptance assertions, and
embeds the complete source text plus SHA-256 hashes of its implementation
modules.

Every executable cell is immediately preceded by a **Code walkthrough** that
documents:

- why the cell exists;
- each logical code chunk in execution order;
- what the chunk does and why it is required;
- visible evidence of a correct run; and
- likely causes to investigate if that evidence is absent.

Each branch also contains one evidence-backed teaching case with an annotated
successful-run figure and a detailed reading guide:

| Branch | Case | Teaching purpose |
|---|---|---|
| RR/HRV | MIT–BIH record 113, 893–901 s | Reproduces post-QRS NeuroKit extra events and shows why detector disagreement is not automatically artifact. |
| Morphology | MIT–BIH record 207 reviewer/audit | Shows changing morphology/polarity, timestamp error, and the five-beat coverage–fidelity tradeoff. |
| Signal quality | NSTDB records 118/119, +24 to −6 dB | Shows how named SQIs respond to controlled electrode-motion noise without fitting an artifact label. |
| Conduction | MIT–BIH record 231, 461.8–470.3 s | Shows smaller intervening deflections being mistaken for ventricular events and why ventricular anchors must be validated. |

The case narratives link the repository's prior PDF reports, implementation
audits, and the papers that justify or constrain each method. Case-study
assertions make the expected pattern fail visibly if dependencies or behavior
change.

The notebooks use MIT–BIH record 100/MLII as the executable example. Change
the configuration cell near the top of a notebook to use a different verified
WFDB input. EDF users must inspect the header and explicitly select an ECG
channel and physical-unit scale.

Regenerate and execute the complete suite from the repository root:

```powershell
feature_extraction\.venv\Scripts\python.exe tools\build_standalone_branch_notebooks.py
feature_extraction\.venv\Scripts\python.exe tools\run_standalone_branch_notebooks.py
```

The signal-quality notebook returns measurements and availability reasons; it
does not train an artifact classifier. None of the four notebooks produces a
seizure prediction or a clinical decision.
