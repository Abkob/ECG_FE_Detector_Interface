"""Curate, synchronize and verify the Y-ECG backup using only the standard library.

Run from ECGdetector. --dry-run performs no writes; --verify checks every managed
file. A normal refresh requires a human-readable session note. No Git or network
mutation is performed. Paths from manifests are validated before use.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import quote

DEFAULT_DESTINATION = Path(r"C:\Users\Salam\Desktop\evo-Sep&Oct26\Y-ECG")
MANIFEST = "BACKUP_MANIFEST.json"
IDEAS = {
    "00_project": ("Project and architecture", "Runnable source layout, shared tools, architecture, and working agreement."),
    "01_rpeak_rr_hrv": ("R-peak detection, RR and HRV", "Detector tracks, RR reliability, HRV, and PRSA/BPRSA. Research implementation; patient validation remains separate."),
    "02_morphology": ("Beat morphology", "Varon features, patient templates, fiducials, delineation, and morphology case studies."),
    "03_conduction_repolarization": ("Conduction and repolarization", "PQRST peak dynamics and interval context. Information features do not establish clinical labels."),
    "04_signal_quality": ("Signal quality", "Noise, artifacts, signal-quality measurements, and their use as context."),
    "05_classification_audit": ("Classification and comprehensive audit", "Feature matrix, model comparisons, grouped validation, and briefings. Cross-validation is not an untouched external test."),
    "06_web_demo": ("Web demonstration", "Browser interface, Python API, deployment configuration, and application tests."),
    "07_data_and_validation": ("Data and validation", "Dataset configuration, acquisition helpers, documentation, and shared validation context."),
    "08_research_strategy": ("Research strategy", "Aims, research plans, progress records, and presentations; these may describe proposed work."),
    "09_reference_library": ("Shared references", "Published literature and shared evidence. Retained evidence is not an endorsement or a redistribution license."),
}
BRANCH_FOLDERS = dict(zip(("B1", "B2", "B3", "B4"), list(IDEAS)[1:5]))
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache",
             ".mypy_cache", ".ruff_cache", ".vercel", ".vercel_python_packages", ".codex",
             ".agents", "tmp", "temp", "build", "dist", "_build", ".ipynb_checkpoints",
             ".codex_pdf_preview_context", "archive", "Datasets", "public", "quarantine",
             "rendered", "rendered_final", "renders"}
DOCS = {".pdf", ".docx", ".pptx", ".xlsx", ".tex", ".bib", ".md"}
BINARY_DOCS = {".pdf", ".docx", ".pptx", ".xlsx"}
CODE = {".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".html", ".css",
        ".ps1", ".bat", ".sh", ".toml", ".yaml", ".yml", ".json", ".ipynb",
        ".svg", ".png", ".jpg", ".jpeg", ".ico", ".txt", ".csv", ".gz"}
ASSETS = {".png", ".svg", ".jpg", ".jpeg", ".eps"}
ROOT_FILES = {"README.md", "AGENTS.md", "requirements.txt", "package.json", "package-lock.json",
              "vercel.json", ".python-version", ".vercelignore", ".gitignore", ".env.example"}
CURATED_TABLES = {"feature_extraction/reports/beat_type_atlases/selected_example_audit.csv"}
DEMO_FILES = ["Datasets/mit-bih-arrhythmia-database-1.0/mit-bih-arrhythmia-database-1.0.0/100." + e
              for e in ("dat", "hea")]
IGNORE_TEXT = """# Generated Y-ECG backup: final documents are intentionally trackable.
.git/
.venv/
venv/
node_modules/
__pycache__/
*.py[cod]
.pytest_cache/
.vercel/
.vercel_python_packages/
.env
.env.*
!.env.example
*.pem
*.key
*.p12
*.pfx
~$*
*.log
*.aux
*.blg
*.fdb_latexmk
*.fls
*.out
*.synctex.gz
*.toc
*.xdv
tmp/
temp/
public/
"""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    rel = PurePosixPath(relative)
    if not relative or rel.is_absolute() or ".." in rel.parts or "\\" in relative or ":" in relative:
        raise ValueError(f"Unsafe relative path: {relative!r}")
    path = root.joinpath(*rel.parts)
    if not path.resolve().is_relative_to(root.resolve()) or path == root:
        raise ValueError(f"Path escapes target root: {relative}")
    return path


def walk(root: Path):
    for base, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not (Path(base) / d).is_symlink()
                         and not (Path(base) / d).is_junction())
        relbase = Path(base).relative_to(root).as_posix()
        if relbase in {"output/qa", "output/notebook_runs", "output/evidence/indexes/extracted_text"}:
            dirs[:] = []
            continue
        for name in sorted(names):
            path = Path(base) / name
            if path.is_symlink() or name.startswith("~$") or name.startswith(".env") and name != ".env.example":
                continue
            if path.suffix.lower() in {".pem", ".key", ".p12", ".pfx"}:
                continue
            if name.endswith(".synctex.gz"):
                continue
            yield path.relative_to(root).as_posix()


def idea_for(rel: str) -> str:
    name = rel.lower()
    if any(x in name for x in ("comprehensive", "rigorous", "model_benchmark", "audit_label", "independent_rebuild",
                               "finalize_audit", "clinician_grouped", "ecg_audit_explanation", "pi_seizure_feature",
                               "ecg_branch_matrix", "ecg_feature_matrix", "classification")):
        return "05_classification_audit"
    if any(x in name for x in ("web_demo", "clinical_demo", "rpeak_viewer", "vercel", "blob.test", "api/", "package.json", "package-lock")):
        return "06_web_demo"
    if any(x in name for x in ("b4_", "signal_quality", "artifact_degradation", "wavelet_artifact", "noise_measurement", "sqi", "butqdb")):
        return "04_signal_quality"
    if any(x in name for x in ("b3_", "conduction", "repolarization")):
        return "03_conduction_repolarization"
    if any(x in name for x in ("b2_", "morphology", "varon", "fiducial", "delineation", "beat_type", "patient_template", "prominence", "case_analysis", "case_reports")):
        return "02_morphology"
    if any(x in name for x in ("b1_", "rpeak", "r_peak", "r-peak", "rr_hrv", "hrv", "prsa", "neurokit", "zhai", "rr_reliability", "fusion", "peaks.py", "reliability.py", "method_selection", "khamis")):
        return "01_rpeak_rr_hrv"
    if any(x in name for x in ("dataset", "edf", "wfdb", "validation", "proof_of_concept")):
        return "07_data_and_validation"
    if name.startswith("docs/research/"):
        return "09_reference_library"
    if "reference_collections" in name and PurePosixPath(rel).suffix.lower() != ".pptx":
        return "09_reference_library"
    if any(x in name for x in ("aim", "strategy", "plan", "progress", "internship", "prearchitecture", "week", "draft", "meeting")):
        return "08_research_strategy"
    if "reference_collections" in name or "/evidence/" in name and "architecture" not in name:
        return "09_reference_library"
    return "00_project"


def is_runtime(rel: str) -> bool:
    p = PurePosixPath(rel)
    if len(p.parts) == 1:
        return p.name in ROOT_FILES or p.suffix in {".ps1", ".bat"}
    if p.parts[0] in {"api", "tools", "tests", "docs"}:
        return p.suffix.lower() in CODE | {".md"}
    if p.parts[0] == "feature_extraction":
        if p.parts[1] in {"docs", "reports"}:
            return p.suffix.lower() in {".py", ".md"}
        return (len(p.parts) == 2 and p.suffix in {".md", ".toml", ".bat"}) or (
            p.parts[1] in {"src", "scripts", "tests", "config", "notebooks", "tools", "docs", "reports"}
            and p.suffix.lower() in CODE | {".md", ".tex", ".bib"})
    return False


def document_target(rel: str, idea: str) -> str:
    p = PurePosixPath(rel)
    replacements = {
        "feature_extraction/reports/": "documents/reports/",
        "feature_extraction/docs/": "documents/technical_notes/",
        "output/architecture/": "documents/architecture/",
        "output/pdf/": "documents/briefings/",
        "source_materials/project_reports/": "documents/research/",
        "source_materials/aims/": "documents/aims/",
        "source_materials/reference_collections/": "documents/reference_collections/",
        "output/evidence/": "historical/evidence/",
    }
    if rel.startswith("output/branches/"):
        return f"{idea}/documents/technical_record/" + "/".join(p.parts[3:])
    if rel.startswith("feature_extraction/outputs/"):
        return f"{idea}/evidence/feature_runs/" + "/".join(p.parts[2:])
    if rel.startswith("outputs/"):
        return f"{idea}/evidence/session_exports/" + "/".join(p.parts[1:])
    for start, end in replacements.items():
        if rel.startswith(start):
            return f"{idea}/{end}{rel[len(start):]}"
    return f"{idea}/documents/{rel}"


def rows(path: Path):
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    return []


def make_plan(root: Path):
    entries, aliases, exclusions = {}, [], []
    binary_seen = {}
    evidence_tags = {}
    mapped_code = {}
    for row in rows(root / "output/evidence/indexes/branch_evidence_map.csv"):
        evidence_tags[row["sha256"]] = [BRANCH_FOLDERS[x] for x in row["branch_tags"].split("|") if x in BRANCH_FOLDERS]
    for path in sorted((root / "output/branches").glob("*/source_map.csv")):
        for row in rows(path):
            if row["branch"] in BRANCH_FOLDERS:
                mapped_code.setdefault(row["relative_path"], set()).add(BRANCH_FOLDERS[row["branch"]])
    invalid = {r["sha256"] for r in rows(root / "output/evidence/indexes/document_manifest.csv")
               if r["pdfinfo_status"] != "valid"}

    def add(rel, dest, *, deduplicate=False):
        path = safe_path(root, rel)
        data = path.read_bytes()
        source_hash = digest(data)
        if path.suffix.lower() == ".pdf" and (source_hash in invalid or not data.startswith(b"%PDF-")):
            exclusions.append({"source": rel, "reason": "invalid or quarantined PDF"})
            return
        if deduplicate and source_hash in binary_seen:
            aliases.append({"source": rel, "destination": binary_seen[source_hash], "source_sha256": source_hash,
                            "related_ideas": sorted(set([idea_for(rel)] + evidence_tags.get(source_hash, [])))})
            return
        transform = None
        if path.suffix.lower() == ".ipynb":
            notebook = json.loads(data)
            for cell in notebook.get("cells", []):
                if cell.get("cell_type") == "code":
                    cell["outputs"] = []
                    cell["execution_count"] = None
            notebook.get("metadata", {}).pop("widgets", None)
            data = (json.dumps(notebook, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
            transform = "notebook execution outputs cleared; source unchanged"
        record = {"source": rel, "source_sha256": source_hash, "sha256": digest(data), "bytes": len(data),
                  "source_modified_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                  "related_ideas": sorted(set([dest.split("/")[0]] + evidence_tags.get(source_hash, [])))}
        if transform:
            record["transform"] = transform
        if dest in entries and entries[dest][0]["sha256"] != record["sha256"]:
            raise ValueError(f"Destination collision: {dest}")
        entries[dest] = (record, data)
        if deduplicate:
            binary_seen[source_hash] = dest

    def priority(rel):
        return (0 if rel.startswith(("feature_extraction/", "output/branches/", "output/architecture/", "output/pdf/"))
                else 1 if rel.startswith("source_materials/") else 2, rel)

    for rel in sorted(walk(root), key=priority):
        p = PurePosixPath(rel)
        ext = p.suffix.lower()
        idea = idea_for(rel)
        runtime = is_runtime(rel)
        if rel == "feature_extraction/.gitignore":
            add(rel, "00_project/settings/feature_extraction.gitignore")
        if runtime:
            if rel == ".gitignore":
                add(rel, "00_project/settings/original.gitignore")
            else:
                add(rel, f"00_project/code/{rel}")
            if ext in CODE and ext not in ASSETS and not rel.startswith("docs/"):
                for folder in sorted(mapped_code.get(rel, {idea}) - {"00_project"}):
                    section = "notebooks" if ext == ".ipynb" else "code"
                    add(rel, f"{folder}/{section}/{rel}")
        doc_location = rel.startswith(("feature_extraction/docs/", "feature_extraction/reports/",
                                      "output/architecture/", "output/branches/", "output/pdf/",
                                      "source_materials/", "output/evidence/"))
        selected = doc_location and ext in DOCS or rel in CURATED_TABLES
        if rel.startswith("docs/") and ext in BINARY_DOCS | {".tex", ".bib"}:
            selected = True
        if rel.startswith("output/evidence/indexes/"):
            selected = False
        if p.name.lower() in {"test_known_good.pdf", "normalized.pdf", "report-retry.pdf"}:
            selected = False
        if rel.startswith(("feature_extraction/outputs/", "outputs/")):
            selected = ext in BINARY_DOCS or (ext == ".md" and p.name.upper() in {"REPORT.MD", "README.MD"}
                                             and "smoke" not in rel.lower())
        if selected:
            add(rel, document_target(rel, idea), deduplicate=ext in BINARY_DOCS)
        elif doc_location and ext in ASSETS and "/evidence/" not in rel and "/reference_collections/" not in rel:
            add(rel, document_target(rel, idea))
        elif ext in BINARY_DOCS and not runtime:
            exclusions.append({"source": rel, "reason": "temporary or unselected document location"})
    for rel in DEMO_FILES:
        if safe_path(root, rel).exists():
            add(rel, f"00_project/code/{rel}")
    # Small curated indexes support provenance; raw inventories and extracted text do not.
    for name in ("document_manifest.csv", "branch_evidence_map.csv", "latex_manifest.csv", "dataset_summary.csv"):
        rel = "output/evidence/indexes/" + name
        if safe_path(root, rel).exists():
            add(rel, "09_reference_library/provenance/" + name)
    protocol = "docs/PROJECT_PROTOCOL.md"
    if safe_path(root, protocol).exists():
        add(protocol, "PROTOCOL.md")
    return entries, aliases, exclusions


def link(path: str, label: str | None = None) -> str:
    return f"[{label or PurePosixPath(path).name}]({quote(path, safe='/._-')})"


def generated_files(entries, aliases):
    files = {".gitignore": IGNORE_TEXT,
             "AGENTS.md": "# Y-ECG backup\n\nRead `PROTOCOL.md` and `README.md`. This folder is an organized backup.\n"
             "Implementation work continues in `C:\\Users\\Salam\\Documents\\ECGdetector`.\n"
             "Update the working source, then use its `tools/sync_y_ecg_backup.py`.\n"
             "Do not edit generated copies here. Follow the standing commit-and-push instruction "
             "in the working project's AGENTS.md.\n"}
    overview = ["# Y-ECG", "", "An organized backup of the ECGdetector research project.", "",
                "**Working source:** `C:\\Users\\Salam\\Documents\\ECGdetector`.",
                "**Start here:** [protocol](PROTOCOL.md), [session changes](CHANGELOG.md), and [verified inventory](BACKUP_MANIFEST.json).", "",
                "| Folder | Contents |", "|---|---|"]
    for folder, (title, purpose) in IDEAS.items():
        overview.append(f"| {link(folder + '/README.md', title)} | {purpose} |")
        own = sorted(p for p in entries if p.startswith(folder + "/"))
        shared = sorted({p for p, (record, _) in entries.items() if folder in record["related_ideas"] and p not in own}
                        | {r["destination"] for r in aliases if folder in r["related_ideas"] and r["destination"] not in own})
        text = [f"# {title}", "", purpose, "", "[Project index](../README.md) · [Updates](UPDATES.md)", "",
                "## Use and status", "",
                "Edit the working project, then refresh the backup. Run application code from "
                "`../00_project/code` with its recorded dependencies and locally configured data paths. "
                "The focused code copies in this folder are browsing extracts, not independent packages.", "",
                "Documents retain their original content and dates. `historical/` preserves older evidence; "
                "a new backup date does not mean those reports were scientifically updated or rebuilt. "
                "Historical TeX may retain original build paths; the inventory identifies its source.", "",
                "## Files", ""]
        if folder == "00_project":
            text += ["The `code/` tree preserves the maintained project layout, tests, configuration, "
                     "runtime web assets and the small public demonstration record. Bulk datasets and "
                     "local credentials must be supplied separately. Install the Python package with "
                     "`python -m pip install -e './feature_extraction[test]'` from `code/`. "
                     "Notebook execution outputs are cleared in the backup.", ""]
        # Grouped file lists provide a usable per-idea inventory without a giant root README.
        for section in sorted({p.split("/")[1] for p in own}):
            members = [p for p in own if p.split("/")[1] == section]
            text += [f"### {section} ({len(members)})", ""]
            if folder == "00_project" and section == "code":
                text += ["Full layout: " + ", ".join(link("code/" + part, part) for part in
                         sorted({p.split('/')[2] for p in members})), ""]
            else:
                text += [f"- {link(p[len(folder)+1:], p[len(folder)+1:])}" for p in members] + [""]
        if shared:
            text += ["## Related files stored in other folders", "",
                     "Shared evidence is linked instead of storing identical PDF copies.", ""]
            text += [f"- {link('../' + p, p)}" for p in shared]
        files[f"{folder}/README.md"] = "\n".join(text) + "\n"
    overview += ["", "## What is retained", "",
                 "Code, tests, dependency files, configuration, useful notebooks, final PDFs, editable "
                 "document sources, important spreadsheets/slides, required figures and small runtime assets. "
                 "One complete runnable code tree is retained alongside focused idea copies; identical "
                 "document payloads are stored once. Every source-to-copy mapping and duplicate alias "
                 "is recorded in `BACKUP_MANIFEST.json`.", "",
                 "## What stays local", "",
                 "Raw corpora (except the existing small public demo record), installed libraries, caches, "
                 "credentials, Git internals, temporary logs/text, QA renders, bulk per-case outputs, and "
                 "the wholesale pre-cleanup archive. Unique historical documents already curated into "
                 "the evidence collection are retained. Runtime JSON/CSV files and requirements.txt are kept.", "",
                 "## Refresh and verify", "", "From the working ECGdetector project:", "", "```powershell",
                 "python tools/sync_y_ecg_backup.py --note-file docs/chat_updates/<session>.md",
                 "python tools/sync_y_ecg_backup.py --verify", "```", "",
                 "The tool never modifies the working source, never pushes to GitHub, preserves unmanaged "
                 "files, and refuses to overwrite local edits in the backup. This is a local backup collection; "
                 "third-party reference material retains its original rights and attribution."]
    files["README.md"] = "\n".join(overview) + "\n"
    return {p: text.encode("utf-8") for p, text in files.items()}


def load_manifest(destination):
    path = destination / MANIFEST
    if not path.exists():
        return {"files": {}, "sessions": []}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != 1 or not isinstance(manifest.get("files"), dict):
        raise ValueError("Unrecognized backup manifest; refusing to overwrite it")
    for rel in manifest["files"]:
        safe_path(destination, rel)
    return manifest


def verify(destination, manifest):
    problems = []
    for rel, record in manifest["files"].items():
        path = safe_path(destination, rel)
        if not path.is_file() or file_hash(path) != record["sha256"]:
            problems.append(rel)
    if problems:
        raise ValueError("Backup verification failed: " + ", ".join(problems[:12]))
    return len(manifest["files"])


def verify_sources(root, manifest):
    sources = {r["source"]: r["source_sha256"] for r in manifest["files"].values() if "source" in r}
    sources.update({r["source"]: r["source_sha256"] for r in manifest.get("document_aliases", [])})
    changed = []
    for rel, expected in sources.items():
        path = safe_path(root, rel)
        if not path.is_file() or file_hash(path) != expected:
            changed.append(rel)
    if changed:
        raise ValueError("Working sources changed since backup: " + ", ".join(changed[:12]))
    return len(sources)


def preflight(destination, previous, planned):
    conflicts = []
    for rel in sorted(set(previous) | set(planned)):
        path = safe_path(destination, rel)
        if not path.exists():
            continue
        if not path.is_file():
            conflicts.append(rel)
            continue
        actual = file_hash(path)
        acceptable = {r["sha256"] for r in (previous.get(rel), planned.get(rel)) if r}
        if actual not in acceptable:
            conflicts.append(rel)
    if conflicts:
        raise ValueError("Refusing to overwrite/remove edited or unmanaged files: " + ", ".join(conflicts[:12]))


def write_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive temporary creation avoids overwriting unrelated files on a retry.
    import tempfile
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".y-ecg-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def sync(root, destination, note, *, dry_run=False):
    root, destination = root.resolve(), destination.resolve()
    if destination == root or root.is_relative_to(destination):
        raise ValueError("Destination cannot be the working source or its ancestor")
    if destination.is_relative_to(root) and not destination.is_relative_to(root / "tmp"):
        raise ValueError("In-project previews must be under tmp to avoid recursive backup")
    old = load_manifest(destination)
    entries, aliases, exclusions = make_plan(root)
    generated = generated_files(entries, aliases)
    records = {p: r for p, (r, _) in entries.items()}
    payloads = {p: data for p, (_, data) in entries.items()}
    for p, data in generated.items():
        records[p] = {"sha256": digest(data), "bytes": len(data), "generated": True}
        payloads[p] = data
    previous_content = {p: r for p, r in old["files"].items() if not r.get("session_record")}
    inventory_changed = (
        {p: r.get("source_sha256") for p, r in records.items()} !=
        {p: r.get("source_sha256") for p, r in previous_content.items()}
        or aliases != old.get("document_aliases", [])
        or exclusions != old.get("excluded_candidates", [])
    )
    added = sorted(set(records) - set(previous_content))
    changed = sorted(p for p in set(records) & set(previous_content) if records[p]["sha256"] != previous_content[p]["sha256"])
    removed = sorted(set(previous_content) - set(records))
    summary = {"destination": str(destination), "copied_files": len(entries), "document_aliases": len(aliases),
               "size_mib": round(sum(r["bytes"] for r in records.values()) / 1024**2, 2),
               "added": len(added), "changed": len(changed), "removed": len(removed),
               "largest_file_mib": round(max((r["bytes"] for r in records.values()), default=0) / 1024**2, 2),
               "by_folder": dict(sorted(Counter(p.split('/')[0] for p in entries).items()))}
    if dry_run:
        preflight(destination, old["files"], records)
        return summary
    if not note or not note.strip():
        raise ValueError("A nonempty --note-file is required for a refresh")
    note_hash = digest(note.encode("utf-8"))
    if not (added or changed or removed or inventory_changed) and old.get("last_note_sha256") == note_hash:
        verify(destination, old)
        return {**summary, "status": "unchanged; verified", "verified_files": len(old["files"])}
    now = datetime.now().astimezone()
    session = now.strftime("%Y-%m-%d_%H%M%S_%f")
    sessions = old.get("sessions", []) + [session]
    affected = sorted({p.split('/')[0] for p in added + changed + removed if p.split('/')[0] in IDEAS})
    lines = [f"# Backup update — {now.isoformat(timespec='seconds')}", "", note.strip(), "",
             "## Automatic backup delta", "",
             ("Initial import baseline: these files already existed in the working project; "
              "importing them does not attribute their authorship to this session." if not old["files"] else
              "Content changes since the previous successful backup; source paths identify the working originals."), "",
             f"Added: {len(added)}. Changed: {len(changed)}. Removed: {len(removed)}.", "",
             "| Action | Backup path | Working source |", "|---|---|---|"]
    for action, paths in (("Added", added), ("Changed", changed), ("Removed", removed)):
        for p in paths:
            rec = records.get(p, previous_content.get(p, {}))
            lines.append(f"| {action} | `{p}` | `{rec.get('source', '(generated index)')}` |")
    lines += ["", "## Verification performed by refresh", "",
              "All managed payloads were checked against their SHA-256 hashes before the successful "
              "manifest was saved. This verifies the copy, not clinical accuracy or re-execution of experiments."]
    session_data = "\n".join(lines).encode("utf-8") + b"\n"
    # Retain immutable session history; generated indexes are replaced on refresh.
    for p, record in old["files"].items():
        if record.get("session_record") and p.startswith("changes/"):
            records[p] = record
    new_logs = {f"changes/{session}.md": session_data}
    new_logs["CHANGELOG.md"] = ("# Backup changes\n\nNewest first. Each entry contains the session note and exact file delta.\n\n"
                                   + "\n".join(f"- {link('changes/' + s + '.md', s)}" for s in reversed(sessions)) + "\n").encode("utf-8")
    for folder in IDEAS:
        path = f"{folder}/UPDATES.md"
        if path in old["files"] and safe_path(destination, path).exists():
            if file_hash(safe_path(destination, path)) != old["files"][path]["sha256"]:
                raise ValueError(f"Manually edited backup update index: {path}")
        prior = safe_path(destination, path).read_text(encoding="utf-8") if path in old["files"] and safe_path(destination, path).exists() else f"# {IDEAS[folder][0]} updates\n\n"
        if folder in affected or not old["files"] or not (added or changed or removed):
            prior += f"- {link('../changes/' + session + '.md', session)}\n"
        new_logs[path] = prior.encode("utf-8")
    for p, data in new_logs.items():
        records[p] = {"sha256": digest(data), "bytes": len(data), "session_record": True}
        payloads[p] = data
    preflight(destination, old["files"], records)
    # Check inputs again before mutating the destination, catching concurrent edits.
    verify_sources(root, {"files": records, "document_aliases": aliases})
    for rel, data in payloads.items():
        target = safe_path(destination, rel)
        if not target.exists() or file_hash(target) != digest(data):
            write_atomic(target, data)
    for rel in removed:
        target = safe_path(destination, rel)
        if target.exists():
            if file_hash(target) != old["files"][rel]["sha256"]:
                raise ValueError(f"Backup changed during refresh; refusing removal: {rel}")
            target.unlink()  # Validated managed file only; never recursive removal.
    manifest = {"schema": 1, "source_root": str(root), "updated_at": now.isoformat(timespec="seconds"),
                "last_note_sha256": note_hash, "sessions": sessions, "files": dict(sorted(records.items())),
                "document_aliases": aliases, "excluded_candidates": exclusions,
                "exclusion_policy": ["environments/caches/secrets/Git internals", "bulk raw datasets except public demo 100",
                                     "temporary logs/text, QA renders and LaTeX sidecars", "bulk generated outputs",
                                     "wholesale pre-cleanup archive; curated historical evidence retained"]}
    count = verify(destination, manifest)
    write_atomic(destination / MANIFEST, (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return {**summary, "status": "synchronized and verified", "verified_files": count, "update": f"changes/{session}.md"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--note-file", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.verify:
        manifest = load_manifest(args.destination)
        if not manifest["files"]:
            parser.error("No completed backup manifest found")
        result = {"status": "verified", "files": verify(args.destination, manifest),
                  "source_files": verify_sources(root, manifest), "updated_at": manifest["updated_at"]}
    else:
        note = args.note_file.read_text(encoding="utf-8") if args.note_file else None
        result = sync(root, args.destination, note, dry_run=args.dry_run)
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
