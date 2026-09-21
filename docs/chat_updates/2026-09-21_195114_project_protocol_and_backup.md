# 2026-09-21 — Project protocol and organized Y-ECG backup

## Purpose and result

Establish a shared end-of-chat protocol and create an organized local backup
for eventual GitHub use. The user confirmed that ECGdetector remains the working
project and Y-ECG is the backup. The backup groups the four research branches,
classification audit, web demo, data/validation, strategy, and references, with
one complete runnable code tree for shared imports.

## Files changed in this session

| Action | Source path | Change and reason |
|---|---|---|
| Added | `AGENTS.md` | Persistent mandatory closeout instructions for new project chats. |
| Added | `docs/PROJECT_PROTOCOL.md` | Version 1.0 of the full working agreement, folder design, inclusion rules, and session-note template. |
| Added | `tools/sync_y_ecg_backup.py` | Repeatable local selection, deduplication, synchronization, per-idea indexes, exact change records and SHA-256 verification. |
| Added | `tests/test_y_ecg_backup.py` | Behavioral checks for selection, no-op refreshes, changes/removals, conflicts, provenance and path safety. |
| Edited | `README.md` | Link the working project to the protocol and backup workflow; preserve the existing scientific and demo content. |
| Added | `docs/chat_updates/2026-09-21_195114_project_protocol_and_backup.md` | This session's authored handoff. |

The generated backup includes `README.md`, `PROTOCOL.md`, `AGENTS.md`,
`.gitignore`, `BACKUP_MANIFEST.json`, `CHANGELOG.md`, ten idea READMEs/update
indexes, and the selected copied files. Its automatic delta gives every exact
destination and working source. Files imported at baseline were not all edited
in this session.

## Decisions and status

- Keep one repository-shaped backup, with no nested Git repositories.
- Keep code, tests, configs/dependencies, PDFs, editable sources, useful
  notebooks, spreadsheets/slides and required assets. Identical document files
  share one stored copy and retain source aliases in the inventory.
- Preserve original file contents except that backup notebook execution output
  is cleared. The working notebooks remain untouched.
- Keep the current web application's small JSON/CSV/compressed runtime assets
  and its existing public MIT-BIH demonstration record. These are necessary
  inputs, not disposable console output.
- Exclude caches, installed libraries, credentials, raw corpora, previews,
  temporary text/logs, compile sidecars and bulk run outputs. Keep unique
  historical documents already present in the curated evidence collection.
- Do not regenerate scientific reports or change model conclusions during this
  organizational task. Existing PDFs retain their original state and dates.
- Protect manually edited backup files and all unmanaged files. Synchronization
  is local; GitHub creation, commit and push are separate actions.

## Verification

- `python -m unittest discover -s tests -p test_y_ecg_backup.py -v`: seven
  behavioral tests passed during implementation. An initial attempt used the
  Windows temporary directory and was blocked by its permissions; test fixtures
  now use a validated directory under the project's ignored `tmp/`.
- Dry-run selection checks identify retained documents inside ignored output
  folders and preserve the current demo's required runtime assets.
- A complete staged copy passed SHA-256 verification. Navigation checks found
  no broken links in the generated main/idea READMEs. A tracked-file coverage
  check identified a nested ignore file and a curated example-audit CSV; both
  were added to the selection rules so they are retained.
- `git diff --check` passed; Git reported only the existing LF/CRLF conversion
  notices. The backup retains 252 distinct PDFs and 22 Office deliverables.
- Each successful refresh verifies every destination payload by SHA-256 before
  saving its manifest. The final `--verify` also checks the working sources.
- ECG scientific tests and experiment reruns are not required by this change:
  no algorithm, model, API or UI implementation was edited in this session.

## Remaining work

Refine the wording or folder organization together if the user wants changes
to this first protocol version. No scientific PDF is claimed to have been
rebuilt. No GitHub repository has been created and nothing has been pushed.

## Backup scope

The working tree already contained research/audit/demo changes when this chat
began, including modifications to both READMEs, the package configuration,
fusion/PRSA modules, web pages/styles, requirements, the static builder and
deployment configuration, plus new audit modules, reports, tests and assets.
Those changes are preserved and copied as current state; their implementation
is not attributed to this chat. Temporary preview/test files under `tmp/` and
Python caches are excluded. Desktop writes use scoped filesystem escalation
where required by the active sandbox.
