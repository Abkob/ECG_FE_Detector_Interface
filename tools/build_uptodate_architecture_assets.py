"""Prepare the local AIM appendix and de-duplicated report bibliography."""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AIM_SOURCE = Path(r"C:\Users\Salam\Downloads\ECG_Seizure (5).pdf")
AIM_DESTINATION = ROOT / "source_materials" / "aims" / "ECG_Seizure_AIM_2026-08-20.pdf"
EXPECTED_AIM_SHA256 = "cdccc4af3538bb5116d04fece1cc372e788cadc7c6e0be9396c381de573e2f7e"
BIB_DESTINATION = ROOT / "output" / "architecture" / "UptoDate_Architecture.bib"
BIB_SOURCES = (
    ROOT / "source_materials" / "reference_collections" / "ECG_detector" / "Resources" / "references.bib",
    ROOT / "feature_extraction" / "reports" / "record_case_reports" / "MITBIH_Record_Case_Reports.bib",
    ROOT / "feature_extraction" / "reports" / "beat_type_atlases" / "MITBIH_Beat_Type_Atlases.bib",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bib_entries(text: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    position = 0
    while True:
        match = re.search(r"(?m)^@[^\n{]+\{\s*([^,\s]+)\s*,", text[position:])
        if match is None:
            break
        start = position + match.start()
        cursor = position + match.end()
        depth = text[start:cursor].count("{") - text[start:cursor].count("}")
        while cursor < len(text) and depth > 0:
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
            cursor += 1
        entries.append((match.group(1), text[start:cursor].strip()))
        position = cursor
    return entries


def main() -> None:
    if not AIM_SOURCE.exists():
        raise FileNotFoundError(AIM_SOURCE)
    source_hash = sha256(AIM_SOURCE)
    if source_hash != EXPECTED_AIM_SHA256:
        raise RuntimeError(f"AIM source hash changed: {source_hash}")
    AIM_DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(AIM_SOURCE, AIM_DESTINATION)
    if sha256(AIM_DESTINATION) != EXPECTED_AIM_SHA256:
        raise RuntimeError("Copied AIM appendix does not match the source hash")

    merged: dict[str, str] = {}
    for source in BIB_SOURCES:
        for key, entry in bib_entries(source.read_text(encoding="utf-8")):
            merged.setdefault(key.casefold(), entry)
    BIB_DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    BIB_DESTINATION.write_text(
        "% De-duplicated from the project reference libraries.\n\n"
        + "\n\n".join(merged[key] for key in sorted(merged))
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"AIM {AIM_DESTINATION.relative_to(ROOT)} {sha256(AIM_DESTINATION)}")
    print(f"BIB {BIB_DESTINATION.relative_to(ROOT)} entries={len(merged)}")


if __name__ == "__main__":
    main()
