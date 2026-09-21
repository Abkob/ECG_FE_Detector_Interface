# 2026-09-21 — Commit, push, and standing push policy

## Purpose and result

The user requested that all current pending project work be committed and
pushed, and instructed that every future project commit must be followed by
a push. The remote literature-review commit `281610d` was fast-forwarded into
the local branch, then all previously uncommitted work was restored cleanly.
The current audit/demo implementation, documents, and project workflow are
prepared together for publication to `origin/main`.

## Files changed in this session

| Action | Source path | Change and reason |
|---|---|---|
| Edited | `AGENTS.md` | Require a normal push and remote verification after every project commit, with standing user authorization. |
| Edited | `docs/PROJECT_PROTOCOL.md` | Version 1.1; document the standing push rule, reconciliation, and accurate failure reporting. |
| Edited | `tools/sync_y_ecg_backup.py` | Include research PDFs under root docs, including the remote literature review; align generated backup instructions with the push policy. |
| Edited | `tests/test_y_ecg_backup.py` | Verify inclusion of root-level research PDFs. |
| Edited | `.gitignore` | Keep the three current audit PDFs and their six required figures trackable. Existing secret/cache/raw-data exclusions remain. |
| Edited | `README.md` | Explain retained audit deliverables and the standing push rule. Preserve both the incoming literature-review section and the pending audit/demo descriptions. |
| Added | `docs/DATA_PROVENANCE.md` | Identify the public datasets, record namespaces and exact existing GitHub destination for the audit exports. |
| Edited | `feature_extraction/reports/README.md` | Link the three final audit reports, their editable sources, the workbook, and the technical audit note. |
| Added | `feature_extraction/reports/clinician_grouped_audit_4page/main.pdf` | Copy the existing final clinician report from `output/pdf/ecg_branch_matrix_clinician_report.pdf` beside its source. |
| Added | `feature_extraction/reports/ecg_audit_explanation_script/main.pdf` | Copy the existing final explanation report from `output/pdf/ecg_audit_explanation_and_improvement_script.pdf` beside its source. |
| Added | `docs/chat_updates/2026-09-21_202719_commit_push_and_standing_policy.md` | This session's handoff and verification record. |

The upstream commit also supplied
`docs/research/ecg-change-detection-during-eeg-literature-review.pdf` and changes
to `.gitignore`, `README.md`, `feature_extraction/README.md`, and
`feature_extraction/docs/README.md`. These retain their existing upstream
authorship. The prior local classification/audit modules, scripts, tests,
web pages and runtime data, report sources/figures, spreadsheet, dependency
updates, fusion/PRSA changes and backup tooling are included in the requested
commit, not attributed to new implementation in this session. Git's commit
inventory and the generated backup delta list every exact included path.

## Decisions and status

- Keep the existing repository and `main` branch; integrate the remote update
  without rewriting history or discarding local edits.
- Scope standing authorization to normal pushes of project commits to the
  configured upstream. Do not force-push or create/change remote repositories
  under that authorization.
- Retain the current final report PDFs alongside their editable sources and
  figures. The broader reference collection stays in the organized local backup.
- Do not rerun research experiments or claim updated scientific results merely
  because existing artifacts are committed. The PDFs are byte-identical copies
  of the existing deliverables.

## Verification

- Python project suite: **131 passed**, using
  `feature_extraction/.venv/Scripts/python.exe -m pytest -q feature_extraction/tests --basetemp=tmp/pytest_push_verified`.
- Backup behavior suite: **7 passed**, including retention of the incoming
  literature-review PDF, safe repeat refreshes, and conflict protection.
- Node upload/cleanup suite: **3 passed** with
  `node --test tests/node/blob.test.js`.
- `node --check` passed for `audit.js` and `models.js`.
- `node tools/build_vercel_static.mjs` succeeded, including the required audit
  pages and runtime downloads.
- The first Python/Node attempts encountered sandbox filesystem/process
  restrictions. Scoped elevated reruns passed; no test assertions were relaxed.
- Both newly copied final PDFs match their original SHA-256 hashes.
- An automatic approval review initially rejected publication because the
  repository visibility and patient/record-level export provenance had not been
  established. Read-only GitHub metadata confirmed the existing destination is
  public. Export inspection and the original PhysioNet pages identified the
  public benchmark record namespaces; `docs/DATA_PROVENANCE.md` records this
  evidence for the reviewed publication retry. No dataset payload was changed.
- Final closeout checks cover the staged diff, backup hashes, commit, push,
  and equality of the local and remote branch tips. Git and the final response
  identify the resulting commit; this note avoids a self-referential commit hash.

## Remaining work

No new scientific validation is claimed. Future completed commits must be
pushed automatically under the standing instruction. Any failure to push must
be reported as pending, with the local commit preserved for retry.

## Backup scope

Refresh Y-ECG from the final working files, including the integrated remote PDF
and updated protocol. The static `public/` build, pytest temporary directories,
logs, environments, credentials and raw corpora remain excluded. No secrets
or local environment files belong in the commit. The temporary Git stash used
for integration is retained until the restored work has been safely published.
