# ECG project protocol

Version 1.1 — 21 September 2026. Adds the user's standing instruction to push
after every project commit. We can continue refining this protocol together.

## 1. Working location and backup

- **Working project:** `C:\Users\Salam\Documents\ECGdetector`.
- **Organized backup:** `C:\Users\Salam\Desktop\evo-Sep&Oct26\Y-ECG`.
- ECGdetector is the source of truth. The backup reflects the actual working
  files, including relevant work that has not yet been committed to Git.
- Y-ECG is one repository-shaped collection with a folder for each idea. Do not
  create nested `.git` repositories. Shared runnable code keeps its original
  paths in `00_project/code`; focused code copies accompany the individual ideas.

## 2. At the start of a chat

Read this protocol, the latest relevant session note, and the backup changelog.
Inspect the working tree before editing. Identify pre-existing changes so that
they are preserved and are not attributed to the current chat. If the preceding
backup was interrupted, finish or report that closeout before claiming it is up
to date. Keep track of files actually changed throughout the session.

## 3. What belongs in the backup

Keep maintained source code, launchers, tests, dependency locks, configuration,
useful notebooks, technical Markdown, final PDFs, editable TeX/BibTeX/Word
sources, important Excel/PowerPoint deliverables, and assets needed by those
documents or the application. Keep concise final experiment reports when they
explain a scientific decision. Preserve rejected research ideas with their
status when they explain why the current method was chosen.

Do not copy installed libraries, environments, caches, `.git`, credentials,
bulk raw recordings, per-run prediction dumps, terminal transcripts, temporary
text, render previews, compile sidecars, or transient intermediate products.
The existing tiny public demonstration record is a documented exception to
the raw-data rule. Runtime JSON/CSV assets and `requirements.txt` are essential
inputs, not clutter. Identical document files are stored once and cross-linked;
unique historical evidence is labelled separately from current documents.

Do not blindly exclude an `output` folder: it may contain the only final PDF or
spreadsheet. Conversely, a `.pdf` suffix alone does not make a temporary render
important. The selection rules live in `tools/sync_y_ecg_backup.py`; update them
when a new idea, file type, report location, or required dependency is added.
The machine-readable inventory records the selected source paths, destinations,
hashes, duplicate aliases, and explicitly excluded candidates.

## 4. Folder design

| Folder | Purpose |
|---|---|
| `00_project` | Complete runnable source layout, architecture, shared tools and project instructions. |
| `01_rpeak_rr_hrv` | Peak detection, RR/HRV, reliability, and PRSA/BPRSA. |
| `02_morphology` | Beat morphology, templates, fiducials, delineation and morphology case studies. |
| `03_conduction_repolarization` | PQRST timing, conduction and repolarization information. |
| `04_signal_quality` | Signal quality, artifacts, noise and contextual reliability. |
| `05_classification_audit` | Comprehensive matrix, model comparisons, rigorous grouped audit and briefings. |
| `06_web_demo` | Browser application, API, hosting configuration and UI tests. |
| `07_data_and_validation` | Dataset documentation, configuration and shared validation context. |
| `08_research_strategy` | Aims, plans, progress documents and research presentations. |
| `09_reference_library` | Shared literature and evidence whose ownership spans ideas. |

Each folder has a concise README with purpose, status boundaries, file links,
dependencies, and a link to its updates. Use `code`, `documents`, `notebooks`,
`evidence`, and `historical` only where content exists. Preserve meaningful
source subdirectories. Do not split tightly coupled Python packages merely to
make the folder tree look neat. Idea code copies share the runnable project's
dependencies and are not advertised as independently installable packages.

## 5. Before finishing each work session

1. Finish the authorized work and appropriate verification.
2. Update technical explanations affected by the change. Do not rewrite reports
   or rebuild all PDFs just to change a date. If code changed and a PDF was not
   rebuilt, record that the PDF still represents its previous state.
3. Create `docs/chat_updates/YYYY-MM-DD_HHMMSS_topic.md` using the template below.
   Every created, edited, renamed or deleted file must be accounted for. Record
   excluded generated files by path/pattern and explain why they are not copied.
4. Run the refresh and independent verification from the working project:

   ```powershell
   python tools/sync_y_ecg_backup.py --note-file docs/chat_updates/YYYY-MM-DD_HHMMSS_topic.md
   python tools/sync_y_ecg_backup.py --verify
   ```

5. Review the generated change record: exact added/changed/removed backup paths,
   original source paths, checksums and affected idea folders. The first run is
   an import baseline, not a claim that all imported files were edited today.
6. After every commit of project work, push to the configured upstream. This is
   already authorized by the user; do not ask again. If the remote has moved,
   fetch and reconcile both sides while preserving existing work, then push
   normally. Never force-push. Verify that the remote branch contains the commit.
7. Give a concise final response with the result, the backup/update links, the
   verification outcome, and unresolved work. If the copy failed, say it failed.

A discussion-only chat needs a note only when it establishes a substantive
decision, finding, or next step. No-op refreshes must not create repetitive
changelog entries. The closeout is performed before the assistant's final
response, not by an app-close hook or unattended background process.

## 6. Session note template

```markdown
# YYYY-MM-DD — short topic

## Purpose and result
What was requested and what is now different.

## Files changed in this session
| Action | Source path | Change and reason |
|---|---|---|
| Added / Edited / Renamed / Deleted | exact relative path | concise description |

## Decisions and status
What was accepted, rejected, superseded, or left experimental, and why.

## Verification
Commands/checks actually run, their results, and tests not run with a reason.

## Remaining work
Specific next steps or 'None for this request'. Mention any stale PDF.

## Backup scope
Important exclusions, pre-existing changes, and any copy/permission blocker.
The generated backup change record supplies the complete file inventory.
```

## 7. Safe refresh and GitHub preparation

Routine local synchronization is already authorized. The refresh preflights
the destination, verifies hashes, and stops if a managed file was manually
edited. Only unchanged files recorded in the preceding backup manifest may be
replaced or removed. Unmanaged files and `.git` are preserved. No source file
is moved or deleted by the backup tool. Never substitute a destructive mirror.

The backup includes an appropriate `.gitignore` that permits the documents we
intend to retain. Keep one copy of shared evidence and link to it. The local
backup does not grant redistribution rights to third-party literature; preserve
existing attribution and review such material before making a public repository.
The user has explicitly authorized committing and pushing the current pending
project work, and has given standing authorization to push after every future
project commit. Scope each commit to the authorized work, preserve unrelated
changes, and exclude credentials and disposable outputs. Do not leave a successful
commit only on this computer when a normal push is possible. If pushing fails,
report the commit and the actual blocker without claiming synchronization.
Creating a new remote, changing visibility, or rewriting published history
remains a separate user action.

`AGENTS.md` makes this protocol available to new project tasks using Codex's
[project instruction discovery](https://learn.chatgpt.com/docs/agent-configuration/agents-md).
Existing chats may need to read the new file explicitly. This file is the
maintained protocol; the root `AGENTS.md` contains the short mandatory checklist.
