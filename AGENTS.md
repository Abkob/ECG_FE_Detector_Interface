# ECGdetector project working agreement

The working project is `C:\Users\Salam\Documents\ECGdetector`.
The organized, GitHub-ready backup is
`C:\Users\Salam\Desktop\evo-Sep&Oct26\Y-ECG`.
Keep doing implementation work in ECGdetector. Do not move the working project.

## Required session closeout

Read `docs/PROJECT_PROTOCOL.md` at the start of each project task and follow it.
Before the final response for a completed work session:

1. Record the actual work in `docs/chat_updates/YYYY-MM-DD_HHMMSS_topic.md`:
   purpose, every file created/edited/renamed/deleted and why, decisions,
   checks with actual results, limitations, and concrete next steps.
   Distinguish this session's edits from changes already present on arrival.
2. Update relevant technical documentation when behavior or conclusions change.
   Preserve the distinction between implemented, experimental, rejected, and
   clinically unvalidated work. Never invent test results or refresh dates.
3. Refresh the backup using
   `python tools/sync_y_ecg_backup.py --note-file docs/chat_updates/<session>.md`.
   Then run `python tools/sync_y_ecg_backup.py --verify`.
4. After every project commit, push it to the configured upstream before
   reporting completion. The user has given standing authorization for these
   pushes; do not ask again. Fetch and reconcile remote changes while preserving
   local work when needed. Never force-push or overwrite remote history.
5. Report the backup location, the useful changes, verification, and any real
   blocker in the final answer. Never claim synchronization succeeded if it failed.

The user has authorized routine local backup refreshes; do not repeatedly ask
for that permission. Respect filesystem permissions and request a scoped tool
escalation if the desktop destination is outside the sandbox.
The user's standing instruction is: always push after committing project work.
If a push fails, report the local commit and the blocker, and keep it available
for retry; do not claim GitHub is current. Creating new remote repositories,
changing visibility, or rewriting published history requires a separate request.

Keep code, tests, configuration, dependency files, useful notebooks, final PDFs,
editable document sources, important spreadsheets/slides, and required assets.
Exclude disposable output dumps, logs, temporary text files, environments,
caches, credentials, and bulk raw datasets. Do not exclude essential files
merely because they end in `.txt`, `.json`, or `.csv`.

Keep each research idea navigable in its own backup folder, with code,
documents, and update links. Preserve the runnable project layout under
`00_project/code`; the idea-level code copies are for focused browsing.
Never overwrite a manually changed backup file or delete an unmanaged file.

For a discussion-only session, record only a substantive decision or next step.
If nothing changed, verify the existing backup without manufacturing a new log.
This closeout runs before a final response; it is not a background task triggered
by closing the app or abandoning a chat. Resume an interrupted closeout next time.
