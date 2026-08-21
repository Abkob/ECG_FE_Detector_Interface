"""Content-addressed PDF/LaTeX audit and evidence-library builder."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
EVIDENCE = OUTPUT / "evidence"
INDEX = EVIDENCE / "indexes"
QA = OUTPUT / "qa"
BASELINE = ROOT / "archive" / "pre_cleanup_2026-08-19" / "inventory_before_cleanup.csv"


def excluded(path: Path) -> bool:
    parts = {part.lower() for part in path.relative_to(ROOT).parts}
    # The evidence audit is intentionally broader than the branch-code map:
    # even third-party PDFs inside the local environment remain traceable.
    rel = path.relative_to(ROOT).as_posix().lower()
    return ".git" in parts or "__pycache__" in parts or "archive/pre_cleanup_2026-08-19/audit_attempt_" in rel


def poppler_executable(name: str) -> str | None:
    bundled = Path.home() / ".cache" / "codex-runtimes" / "codex-primary-runtime" / "dependencies" / "native" / "poppler" / "Library" / "bin" / f"{name}.exe"
    if bundled.exists():
        return str(bundled)
    discovered = shutil.which(name)
    return discovered


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(stem: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return value[:150] or "document"


def rank(path: Path) -> tuple[int, int, str]:
    rel = path.relative_to(ROOT).as_posix().lower()
    if rel.startswith("output/architecture/") or rel.startswith("output/branches/"):
        priority = 0
    elif rel.startswith("output/evidence/") and "/indexes/" not in rel:
        priority = 1
    elif rel.startswith("feature_extraction/"):
        priority = 10
    elif rel.startswith("source_materials/"):
        priority = 20
    elif rel.startswith("archive/pre_cleanup_2026-08-19/output_legacy/organized_references/"):
        priority = 25
    elif rel.startswith("archive/"):
        priority = 40
    elif rel.startswith("output/qa/"):
        priority = 90
    else:
        priority = 30
    return priority, len(rel), rel


def classify(path: Path, text: str = "") -> str:
    rel = path.relative_to(ROOT).as_posix().lower()
    name = path.name.lower()
    joined = f"{rel} {text[:12000].lower()}"
    if path.stat().st_size == 0:
        return "quarantine"
    if any(token in joined for token in ["beat_type_atlas", "record_case", "case_analysis", "mitbih_record", "mit-bih_record", "case_study"]):
        return "case_studies"
    if any(token in joined for token in ["datasets_used", "datasets_and_branch", "dataset_technical", "edf_selection", "dataset record"]):
        return "datasets"
    if any(token in rel for token in ["annotatedpapers", "annotated_papers", "/literature/", "papers/"]):
        return "literature"
    if any(token in joined for token in [
        "architecture", "technical_record", "technical record", "algorithm_explainer",
        "audit", "review", "implementation_plan", "research_and_implementation",
        "progress_report", "internship", "wavelet_artifact", "strategy", "project report",
        "proof_of_concept", "proof of concept", "current_status", "current status",
    ]):
        return "artifacts"
    if rel.startswith("source_materials/reference_collections/") and not any(token in joined for token in ["technical record", "architecture", "project"]):
        return "literature"
    return "unresolved"


def branch_tags(path: Path, text: str = "") -> tuple[str, str]:
    blob = f"{path.as_posix()} {text[:30000]}".lower()
    tags=[]; reasons=[]
    rules={
        "B1": ["rr", "hrv", "r-peak", "rpeak", "peak detection", "jeppesen", "prsa", "bprsa", "zhai", "pan-tompkins"],
        "B2": ["morphology", "varon", "beat template", "qrs capture", "fiducial", "delineation", "beat atlas"],
        "B3": ["conduction", "repolarization", "qt interval", "qtvi", "diab", "pqrst", "qtc"],
        "B4": ["signal quality", "artifact", "noise stress", "sqi", "baseline wander", "powerline", "saturation"],
    }
    for branch,words in rules.items():
        matched=next((w for w in words if w in blob),None)
        if matched: tags.append(branch); reasons.append(f"{branch}:{matched}")
    return "|".join(tags), ";".join(reasons) if reasons else "no branch keyword match"


def unique_destination(category: str, stem: str, extension: str, digest: str) -> Path:
    directory = EVIDENCE / category; directory.mkdir(parents=True, exist_ok=True)
    preferred = directory / f"{safe_name(stem)}{extension}"
    if not preferred.exists() or sha256(preferred) == digest:
        return preferred
    return directory / f"{safe_name(stem)}__{digest[:10]}{extension}"


def validate_pdf(path: Path, digest: str, category_hint: str) -> dict[str, object]:
    result={"pdfinfo_status":"not_run","page_count":"","extraction_status":"not_run","text_character_count":0,"text_page_count":0,"visual_review_required":False,"validation_error":""}
    if path.stat().st_size == 0:
        result.update({"pdfinfo_status":"invalid","extraction_status":"invalid","visual_review_required":True,"validation_error":"zero_byte_pdf"}); return result
    pdfinfo=poppler_executable("pdfinfo")
    if pdfinfo:
        proc=subprocess.run([pdfinfo,str(path)],capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=45)
        if proc.returncode==0:
            result["pdfinfo_status"]="valid"
            match=re.search(r"^Pages:\s+(\d+)",proc.stdout,re.M)
            if match: result["page_count"]=int(match.group(1))
        else:
            result.update({"pdfinfo_status":"invalid","visual_review_required":True,"validation_error":proc.stderr.strip()[:500]})
    else: result["pdfinfo_status"]="unavailable"
    extracted=[]
    try:
        reader=PdfReader(str(path),strict=False)
        if result["page_count"]=="": result["page_count"]=len(reader.pages)
        for page in reader.pages:
            value=page.extract_text() or ""; extracted.append(value)
        chars=sum(len(x.strip()) for x in extracted); text_pages=sum(bool(x.strip()) for x in extracted)
        result.update({"extraction_status":"success" if chars else "no_text","text_character_count":chars,"text_page_count":text_pages})
        page_count=int(result["page_count"] or 0)
        result["visual_review_required"]=bool(chars < max(100, page_count*60) or text_pages < max(1,page_count//2))
        if category_hint != "literature" and chars:
            text_dir=INDEX/"extracted_text"; text_dir.mkdir(parents=True,exist_ok=True)
            (text_dir/f"{digest}.txt").write_text("\n\n\f\n\n".join(extracted),encoding="utf-8",errors="replace")
    except Exception as exc:
        result.update({"extraction_status":"failed","visual_review_required":True,"validation_error":f"{result['validation_error']} | {type(exc).__name__}: {exc}".strip(" |")} )
    return result


def render_text_poor(path: Path, digest: str) -> str:
    renderer=poppler_executable("pdftoppm")
    if not renderer or path.stat().st_size==0: return ""
    preview_dir=QA/"text_poor_previews"; preview_dir.mkdir(parents=True,exist_ok=True)
    target=preview_dir/digest
    proc=subprocess.run([renderer,"-f","1","-singlefile","-png","-scale-to","1200",str(path),str(target)],capture_output=True,text=True,timeout=60)
    output=target.with_suffix(".png")
    return output.relative_to(ROOT).as_posix() if proc.returncode==0 and output.exists() else ""


def read_baseline() -> list[dict[str,str]]:
    if not BASELINE.exists(): return []
    with BASELINE.open(newline="",encoding="utf-8-sig") as h:
        return list(csv.DictReader(h))


def write_csv(path: Path, rows: list[dict[str,object]], fields: list[str] | None=None) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    if fields is None: fields=list(rows[0]) if rows else []
    with path.open("w",newline="",encoding="utf-8") as h:
        writer=csv.DictWriter(h,fieldnames=fields,extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    for name in ["artifacts","datasets","case_studies","literature","unresolved","quarantine","indexes"]: (EVIDENCE/name).mkdir(parents=True,exist_ok=True)
    baseline=read_baseline()
    current_files=[p for p in ROOT.rglob("*") if p.is_file() and not excluded(p) and p.suffix.lower() in {".pdf",".tex"}]
    groups: dict[str,list[Path]]=defaultdict(list)
    for path in current_files: groups[sha256(path)].append(path)

    baseline_by_hash: dict[str,list[str]]=defaultdict(list)
    for row in baseline:
        if row.get("extension") in {".pdf",".tex"} and row.get("sha256"):
            baseline_by_hash[row["sha256"].lower()].append(row["relative_path"])

    tex_rows=[]; tex_by_original_stem={}; tex_by_stem_global={}; tex_canonical_by_hash={}
    for digest,paths in sorted(groups.items()):
        tex_paths=[p for p in paths if p.suffix.lower()==".tex"]
        if not tex_paths: continue
        origins=[p for p in tex_paths if not p.relative_to(ROOT).as_posix().lower().startswith(("output/evidence/","output/qa/"))]
        source=min(origins or tex_paths,key=rank); content=source.read_text(encoding="utf-8",errors="replace")
        category=classify(source,content); tags,reason=branch_tags(source,content)
        existing=[p for p in tex_paths if p.relative_to(ROOT).as_posix().startswith(("output/architecture/","output/branches/","output/evidence/"))]
        if existing:
            destination=min(existing,key=rank)
        else:
            destination=unique_destination(category,source.stem,".tex",digest); shutil.copy2(source,destination)
        standalone=bool(re.search(r"\\documentclass(?:\[.*?\])?\{",content))
        row={"sha256":digest,"canonical_output_path":destination.relative_to(ROOT).as_posix(),"compilation_source_path":source.relative_to(ROOT).as_posix(),"category":category,"standalone":standalone,"branch_tags":tags,"tag_reason":reason,"current_path_count":len(tex_paths),"pre_cleanup_path_count":len([p for p in baseline_by_hash.get(digest,[]) if p.lower().endswith(".tex")]),"source_size_bytes":source.stat().st_size}
        tex_rows.append(row); tex_canonical_by_hash[digest]=destination
        for p in tex_paths:
            tex_by_original_stem[(p.parent.resolve(),p.stem.lower())]=destination
            tex_by_stem_global.setdefault(p.stem.lower(),destination)

    document_rows=[]; validation_rows=[]; duplicate_rows=[]; branch_rows=[]
    for digest,paths in sorted(groups.items()):
        pdf_paths=[p for p in paths if p.suffix.lower()==".pdf"]
        if not pdf_paths: continue
        origins=[p for p in pdf_paths if not p.relative_to(ROOT).as_posix().lower().startswith(("output/evidence/","output/qa/"))]
        source=min(origins or pdf_paths,key=rank)
        quick_text=""
        try:
            reader=PdfReader(str(source),strict=False)
            quick_text="\n".join((page.extract_text() or "") for page in reader.pages[:5])
        except Exception: pass
        category=classify(source,quick_text)
        validation=validate_pdf(source,digest,category)
        if validation["pdfinfo_status"]=="invalid" or validation["extraction_status"] in {"invalid","failed"}: category="quarantine"
        sibling_tex=tex_by_original_stem.get((source.parent.resolve(),source.stem.lower())) or tex_by_stem_global.get(source.stem.lower())
        existing=[p for p in pdf_paths if p.relative_to(ROOT).as_posix().startswith(("output/architecture/","output/branches/","output/evidence/"))]
        if existing:
            destination=min(existing,key=rank)
        elif sibling_tex is not None and sibling_tex.parent.name==category:
            destination=sibling_tex.with_suffix(".pdf")
            if destination.exists() and sha256(destination)!=digest: destination=unique_destination(category,source.stem,".pdf",digest)
            if not destination.exists(): shutil.copy2(source,destination)
        else:
            destination=unique_destination(category,source.stem,".pdf",digest); shutil.copy2(source,destination)
        tags,reason=branch_tags(source,quick_text)
        preview=render_text_poor(source,digest) if validation["visual_review_required"] and category!="quarantine" else ""
        has_source=sibling_tex is not None
        row={"sha256":digest,"canonical_output_path":destination.relative_to(ROOT).as_posix(),"category":category,"source_status":"available" if has_source else "source_unavailable","paired_latex_path":sibling_tex.relative_to(ROOT).as_posix() if has_source else "","size_bytes":source.stat().st_size,"page_count":validation["page_count"],"pdfinfo_status":validation["pdfinfo_status"],"extraction_status":validation["extraction_status"],"text_character_count":validation["text_character_count"],"text_page_count":validation["text_page_count"],"visual_review_required":validation["visual_review_required"],"preview_path":preview,"branch_tags":tags,"tag_reason":reason,"current_path_count":len(pdf_paths),"pre_cleanup_path_count":len([p for p in baseline_by_hash.get(digest,[]) if p.lower().endswith(".pdf")])}
        document_rows.append(row)
        validation_rows.append({"sha256":digest,"canonical_output_path":row["canonical_output_path"],"size_bytes":row["size_bytes"],"page_count":row["page_count"],"pdfinfo_status":row["pdfinfo_status"],"extraction_status":row["extraction_status"],"text_character_count":row["text_character_count"],"text_page_count":row["text_page_count"],"visual_review_required":row["visual_review_required"],"preview_path":preview,"validation_error":validation["validation_error"]})
        branch_rows.append({"sha256":digest,"canonical_output_path":row["canonical_output_path"],"category":category,"branch_tags":tags,"mapping_reason":reason})

    canonical_lookup={row["sha256"]:row["canonical_output_path"] for row in [*document_rows,*tex_rows]}
    for digest,paths in sorted(groups.items()):
        relevant=[p for p in paths if p.suffix.lower() in {".pdf",".tex"}]
        for path in relevant:
            duplicate_rows.append({"sha256":digest,"path":path.relative_to(ROOT).as_posix(),"path_state":"current","canonical_output_path":canonical_lookup.get(digest,"")})
        for old in sorted(set(baseline_by_hash.get(digest,[]))):
            duplicate_rows.append({"sha256":digest,"path":old,"path_state":"pre_cleanup_original","canonical_output_path":canonical_lookup.get(digest,"")})

    # A project-authored source changed concurrently during this build.  Keep
    # the pre-cleanup hash/path trace even when that exact byte sequence is no
    # longer present, without pretending there is a canonical current copy.
    baseline_hash_only=sorted(set(baseline_by_hash)-set(groups))
    for digest in baseline_hash_only:
        for old in sorted(set(baseline_by_hash[digest])):
            duplicate_rows.append({"sha256":digest,"path":old,"path_state":"pre_cleanup_original_hash_only_source_changed_concurrently","canonical_output_path":""})

    write_csv(INDEX/"document_manifest.csv",document_rows)
    write_csv(INDEX/"latex_manifest.csv",tex_rows)
    write_csv(INDEX/"duplicate_map.csv",duplicate_rows)
    write_csv(INDEX/"branch_evidence_map.csv",branch_rows)
    write_csv(QA/"pdf_validation.csv",validation_rows)
    canonical_paths=sorted({ROOT/row["canonical_output_path"] for row in [*document_rows,*tex_rows] if row.get("canonical_output_path")})
    with (INDEX/"checksums.sha256").open("w",encoding="utf-8",newline="\n") as h:
        for path in canonical_paths:
            if path.exists(): h.write(f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n")
    summary={
        "baseline":{"pdf_paths":sum(r.get("extension")==".pdf" for r in baseline),"latex_paths":sum(r.get("extension")==".tex" for r in baseline),"unique_pdf_hashes":len({r.get("sha256") for r in baseline if r.get("extension")==".pdf" and r.get("sha256")}),"unique_latex_hashes":len({r.get("sha256") for r in baseline if r.get("extension")==".tex" and r.get("sha256")})},
        "current":{"pdf_paths":sum(p.suffix.lower()==".pdf" for p in current_files),"latex_paths":sum(p.suffix.lower()==".tex" for p in current_files),"unique_pdf_hashes":len(document_rows),"unique_latex_hashes":len(tex_rows)},
        "categories":dict(sorted(__import__("collections").Counter(r["category"] for r in document_rows).items())),
        "invalid_or_zero_byte":sum(r["category"]=="quarantine" for r in document_rows),
        "source_unavailable":sum(r["source_status"]=="source_unavailable" for r in document_rows),
        "text_poor_visual_review":sum(bool(r["visual_review_required"]) for r in document_rows),
        "baseline_hash_only":len(baseline_hash_only),
    }
    (INDEX/"audit_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
