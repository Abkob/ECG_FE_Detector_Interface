"""Final mechanical acceptance checks for the ECGdetector bundle."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/"output"
INDEX=OUTPUT/"evidence"/"indexes"
BRANCHES={
    "B1_peak_rr_hrv":"B1_Peak_RR_HRV_Technical_Record",
    "B2_morphology":"B2_Morphology_Technical_Record",
    "B3_conduction_repolarization":"B3_Conduction_Repolarization_Technical_Record",
    "B4_signal_quality":"B4_Signal_Quality_Technical_Record",
}


def load_csv(path: Path):
    with path.open(newline="",encoding="utf-8-sig") as h: return list(csv.DictReader(h))


def digest(path: Path) -> str:
    value=hashlib.sha256()
    with path.open("rb") as h:
        for chunk in iter(lambda:h.read(1024*1024),b""): value.update(chunk)
    return value.hexdigest()


def main() -> int:
    checks=[]
    def add(name: str, passed: bool, detail: object): checks.append({"name":name,"passed":bool(passed),"detail":detail})

    documents=load_csv(INDEX/"document_manifest.csv"); latex=load_csv(INDEX/"latex_manifest.csv"); duplicates=load_csv(INDEX/"duplicate_map.csv")
    add("document_manifest_nonempty",len(documents)>0,len(documents))
    add("latex_manifest_unique_sources",len(latex)>=32,len(latex))
    add("quarantine_count",sum(r["category"]=="quarantine" for r in documents)==4,sum(r["category"]=="quarantine" for r in documents))
    non_quarantine=[r for r in documents if r["category"]!="quarantine"]
    add("all_non_quarantine_pdfs_validate",all(r["pdfinfo_status"]=="valid" for r in non_quarantine),[r["canonical_output_path"] for r in non_quarantine if r["pdfinfo_status"]!="valid"])

    branch_detail={}
    for branch,stem in BRANCHES.items():
        directory=OUTPUT/"branches"/branch
        counts={"tex":len(list(directory.glob("*.tex"))),"pdf":len(list(directory.glob("*.pdf"))),"ipynb":len(list(directory.glob("*.ipynb"))),"source_map":(directory/"source_map.csv").exists(),"evidence_index":(directory/"evidence_index.md").exists()}
        branch_detail[branch]=counts
    add("exact_branch_deliverables",all(v=={"tex":1,"pdf":1,"ipynb":1,"source_map":True,"evidence_index":True} for v in branch_detail.values()),branch_detail)

    latex_report=json.loads((OUTPUT/"qa"/"latex_build_report.json").read_text(encoding="utf-8"))
    standalone_count=sum(r["standalone"].lower()=="true" for r in latex)
    add("all_standalone_latex_compiled",len(latex_report["sources"])==standalone_count and all(r["exit_code"]==0 and r["pdf_created"] for r in latex_report["sources"]),{"expected":standalone_count,"sources":len(latex_report["sources"]),"failures":[r["source"] for r in latex_report["sources"] if r["exit_code"] or not r["pdf_created"]]})
    new_sources=[r for r in latex_report["sources"] if r.get("canonical_tex","").startswith(("output/architecture/","output/branches/","output/evidence/datasets/Datasets_and_Branch_Use"))]
    add("new_latex_has_no_layout_or_reference_warnings",len(new_sources)==6 and all(r["overfull_hbox_count"]==0 and not r["undefined_reference_warning"] for r in new_sources),new_sources)

    notebooks=json.loads((OUTPUT/"qa"/"notebook_execution_report.json").read_text(encoding="utf-8"))
    add("notebooks_clean_kernel_pass",notebooks["status"]=="passed" and len(notebooks["notebooks"])==4 and all(not r["missing_artifacts"] for r in notebooks["notebooks"]),notebooks["notebooks"])
    dossier=json.loads((OUTPUT/"qa"/"branch_dossier_build_report.json").read_text(encoding="utf-8"))
    dossier_pages={
        row["pdf"].split("/")[-2]:row["page_count"]
        for row in json.loads((OUTPUT/"qa"/"pdf_render_report.json").read_text(encoding="utf-8"))["documents"]
        if row["pdf"].startswith("output/branches/")
    }
    add(
        "exhaustive_branch_dossiers",
        dossier["status"]=="passed"
        and dossier["coverage"]["first_party_python_files"]==dossier["coverage"]["mapped_unique_files"]
        and not dossier["coverage"]["unmapped"]
        and not dossier["coverage"]["mapped_missing"]
        and dossier["full_test"]["exit_code"]==0
        and all(row["branch_test"]["exit_code"]==0 for row in dossier["branches"])
        and len(dossier_pages)==4
        and min(dossier_pages.values())>=75,
        {"coverage":dossier["coverage"],"full_test":dossier["full_test"]["passed"],"branch_pages":dossier_pages},
    )
    test_text=(OUTPUT/"qa"/"test_suite.txt").read_text(encoding="utf-8",errors="replace")
    match=re.search(r"(\d+) passed",test_text)
    test_count=int(match.group(1)) if match else 0
    add("test_baseline_met",test_count>=102,test_count)

    render=json.loads((OUTPUT/"qa"/"pdf_render_report.json").read_text(encoding="utf-8"))
    visual=json.loads((OUTPUT/"qa"/"visual_inspection_report.json").read_text(encoding="utf-8"))
    add("compiled_pdf_render_and_visual_qa",render["status"]=="passed" and render["pdf_count"]==len(latex_report["sources"]) and render["total_pages"]>0 and render["visual_inspection_status"].startswith("passed") and visual["status"]=="passed",{"expected_pdfs":len(latex_report["sources"]),"pdfs":render["pdf_count"],"pages":render["total_pages"],"visual":visual["status"]})

    checksum_errors=[]; checksum_count=0
    for line in (INDEX/"checksums.sha256").read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        expected,relative=line.split("  ",1); path=ROOT/relative; checksum_count+=1
        if not path.exists() or digest(path)!=expected: checksum_errors.append(relative)
    add("canonical_checksums_verify",not checksum_errors,{"checked":checksum_count,"errors":checksum_errors})

    baseline=load_csv(ROOT/"archive"/"pre_cleanup_2026-08-19"/"inventory_before_cleanup.csv")
    baseline_hashes={r["sha256"].lower() for r in baseline if r["extension"] in {".pdf",".tex"} and r["sha256"]}
    traced_hashes={r["sha256"] for r in duplicates if r["path_state"].startswith("pre_cleanup_original")}
    add("all_baseline_hashes_traceable",baseline_hashes<=traced_hashes,{"baseline_hashes":len(baseline_hashes),"traced_hashes":len(traced_hashes),"missing":sorted(baseline_hashes-traced_hashes)})
    dataset_rows=load_csv(INDEX/"dataset_file_manifest.csv")
    missing_dataset_paths=[row["relative_path"] for row in dataset_rows if not (ROOT/row["relative_path"]).exists()]
    add("dataset_file_inventory_complete",bool(dataset_rows) and not missing_dataset_paths,{"files":len(dataset_rows),"missing":missing_dataset_paths})

    report={"status":"passed" if all(c["passed"] for c in checks) else "failed","checks":checks}
    (OUTPUT/"qa"/"acceptance_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"status":report["status"],"checks":len(checks),"failed":[c["name"] for c in checks if not c["passed"]]},indent=2))
    return 0 if report["status"]=="passed" else 1


if __name__=="__main__": raise SystemExit(main())
