"""Compile canonical ECGdetector LaTeX sources with isolated sidecars."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
COMPILER = Path(r"C:\Users\Salam\.codex\plugins\cache\openai-bundled\latex\0.2.6\scripts\compile_latex.py")


NEW_SOURCES = [
    OUTPUT / "architecture" / "Architecture.tex",
    OUTPUT / "branches" / "B1_peak_rr_hrv" / "B1_Peak_RR_HRV_Technical_Record.tex",
    OUTPUT / "branches" / "B2_morphology" / "B2_Morphology_Technical_Record.tex",
    OUTPUT / "branches" / "B3_conduction_repolarization" / "B3_Conduction_Repolarization_Technical_Record.tex",
    OUTPUT / "branches" / "B4_signal_quality" / "B4_Signal_Quality_Technical_Record.tex",
    OUTPUT / "evidence" / "datasets" / "Datasets_and_Branch_Use_Technical_Record.tex",
]


def canonical_sources() -> list[tuple[Path, Path]]:
    manifest = OUTPUT / "evidence" / "indexes" / "latex_manifest.csv"
    if not manifest.exists():
        return []
    sources: list[tuple[Path, Path]] = []
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("standalone", "").lower() == "true" and row.get("canonical_output_path"):
                canonical = ROOT / Path(row["canonical_output_path"])
                source = ROOT / Path(row.get("compilation_source_path") or row["canonical_output_path"])
                if source.exists() and canonical.exists():
                    sources.append((source, canonical))
    return sources


def compile_one(source: Path, canonical_tex: Path) -> dict[str, object]:
    key = hashlib.sha256(str(source).encode()).hexdigest()[:10]
    build_dir = OUTPUT / "qa" / "latex_build" / f"{source.stem}_{key}"
    build_dir.mkdir(parents=True, exist_ok=True)
    compile_source = source
    maintained_morphology_dir = ROOT / "feature_extraction" / "docs" / "latex_morphology_branch"
    if source.name == "morphology_branch_complete_technical_record.tex" and source.parent.resolve() != maintained_morphology_dir.resolve():
        # Archived and canonical evidence copies retain paths relative to the
        # maintained docs tree. Recreate that layout in isolated QA staging.
        staging_root = OUTPUT / "qa" / "latex_staging" / key / "feature_extraction"
        staging_docs = staging_root / "docs" / "latex_morphology_branch"
        staging_figure = staging_root / "outputs" / "morphology_fidelity_all48_v1" / "morphology_fidelity_summary.png"
        staging_docs.mkdir(parents=True, exist_ok=True)
        staging_figure.parent.mkdir(parents=True, exist_ok=True)
        compile_source = staging_docs / source.name
        shutil.copy2(source, compile_source)
        shutil.copy2(ROOT / "feature_extraction" / "outputs" / "morphology_fidelity_all48_v1" / "morphology_fidelity_summary.png", staging_figure)
    # Legacy standalone sources use relative graphics/BibTeX paths that were
    # valid in their original project roots.  Preserve those search roots
    # while still keeping all sidecars in the isolated QA build directory.
    search_roots = [
        compile_source.parent,
        compile_source.parent.parent,
        ROOT / "feature_extraction" / "docs" / "latex_morphology_branch",
        ROOT / "archive" / "pre_cleanup_2026-08-19" / "output_legacy" / "latex",
        ROOT / "source_materials" / "reference_collections" / "ECG_detector" / "Resources",
    ]
    environment = os.environ.copy()
    environment["TEXINPUTS"] = os.pathsep.join(str(path) for path in search_roots if path.exists()) + os.pathsep
    environment["BIBINPUTS"] = environment["TEXINPUTS"]
    legacy_bib = ROOT / "archive" / "pre_cleanup_2026-08-19" / "output_legacy" / "latex" / "references.bib"
    if legacy_bib.exists():
        shutil.copy2(legacy_bib, build_dir.parent / "references.bib")
    command = [sys.executable, str(COMPILER), str(compile_source), "--compiler", "texlive", "--engine", "pdflatex", "--output-directory", str(build_dir), "--json"]
    completed = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, encoding="utf-8", errors="replace")
    raw = completed.stdout.strip()
    try:
        details = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        details = {"raw_stdout": raw[-12000:]}
    (build_dir / "compile_result.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    log_text = "\n".join([str(details.get("log", "")), completed.stderr])
    built = build_dir / f"{source.stem}.pdf"
    destination = canonical_tex.with_suffix(".pdf")
    if completed.returncode == 0 and built.exists():
        shutil.copy2(built, destination)
    overfull = re.findall(r"Overfull \\hbox \(([^)]+)\)", log_text)
    undefined = bool(re.search(r"undefined references|Citation .* undefined", log_text, re.I))
    return {
        "source": source.relative_to(ROOT).as_posix(),
        "canonical_tex": canonical_tex.relative_to(ROOT).as_posix(),
        "destination_pdf": destination.relative_to(ROOT).as_posix(),
        "exit_code": completed.returncode,
        "pdf_created": destination.exists() and destination.stat().st_size > 0,
        "overfull_hbox_count": len(overfull),
        "largest_overfull": overfull[0] if overfull else "",
        "undefined_reference_warning": undefined,
        "build_directory": build_dir.relative_to(ROOT).as_posix(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-canonical-evidence", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--only-canonical",
        help="Compile only the canonical path named here (repository-relative).",
    )
    args = parser.parse_args()
    sources: list[tuple[Path, Path]] = [(path, path) for path in NEW_SOURCES]
    if args.include_canonical_evidence:
        sources.extend(canonical_sources())
    if args.only_canonical:
        requested = Path(args.only_canonical).as_posix().lstrip("./")
        sources = [
            (source, canonical)
            for source, canonical in sources
            if canonical.relative_to(ROOT).as_posix() == requested
        ]
    report_path = OUTPUT / "qa" / "latex_build_report.json"
    if args.retry_failed and report_path.exists():
        previous_report = json.loads(report_path.read_text(encoding="utf-8"))
        failed_canonical = {row.get("canonical_tex") or row.get("source") for row in previous_report.get("sources", []) if row.get("exit_code") or not row.get("pdf_created")}
        sources = [(source, canonical) for source, canonical in sources if canonical.relative_to(ROOT).as_posix() in failed_canonical]
    unique: list[tuple[Path, Path]] = []
    seen = set()
    for source, canonical in sources:
        resolved = canonical.resolve()
        if resolved not in seen and source.exists() and canonical.exists():
            unique.append((source, canonical)); seen.add(resolved)
    results = [compile_one(source, canonical) for source, canonical in unique]
    previous = []
    if report_path.exists():
        try: previous = json.loads(report_path.read_text(encoding="utf-8")).get("sources", [])
        except Exception: previous = []
    # A live source path can legitimately change content and therefore map to a
    # second canonical evidence copy.  Keying by canonical path preserves both.
    merged = {(row.get("canonical_tex") or row["source"]): row for row in [*previous, *results]}
    report = {"status":"success" if all(r["exit_code"] == 0 and r["pdf_created"] for r in results) else "failure", "sources":list(merged.values())}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status":report["status"],"compiled":len(results),"failed":[r["source"] for r in results if r["exit_code"] or not r["pdf_created"]],"overfull":sum(r["overfull_hbox_count"] for r in results)}, indent=2))
    return 0 if report["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
