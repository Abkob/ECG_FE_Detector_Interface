"""Render every compiled canonical PDF and build page-labelled contact sheets."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
REPORT = OUTPUT / "qa" / "latex_build_report.json"
RENDER_ROOT = OUTPUT / "qa" / "renders"
PDFTOPPM = Path.home() / ".cache" / "codex-runtimes" / "codex-primary-runtime" / "dependencies" / "native" / "poppler" / "Library" / "bin" / "pdftoppm.exe"


def border_ink_fraction(image: Image.Image) -> float:
    gray=image.convert("L"); w,h=gray.size; band=max(2,min(w,h)//200)
    pixels=[]
    for crop in [(0,0,w,band),(0,h-band,w,h),(0,0,band,h),(w-band,0,w,h)]: pixels.extend(gray.crop(crop).getdata())
    return sum(value<235 for value in pixels)/max(1,len(pixels))


def contact_sheets(pages: list[Path], directory: Path) -> list[Path]:
    outputs=[]; font=ImageFont.load_default(); per_sheet=12; columns=3; thumb_w=330; thumb_h=430; label_h=24; gap=14
    for sheet_index in range(0,len(pages),per_sheet):
        batch=pages[sheet_index:sheet_index+per_sheet]; rows=(len(batch)+columns-1)//columns
        canvas=Image.new("RGB",(columns*(thumb_w+gap)+gap,rows*(thumb_h+label_h+gap)+gap),"#d9dde3")
        draw=ImageDraw.Draw(canvas)
        for slot,path in enumerate(batch):
            page=Image.open(path).convert("RGB"); page.thumbnail((thumb_w,thumb_h))
            x=gap+(slot%columns)*(thumb_w+gap)+(thumb_w-page.width)//2
            y=gap+(slot//columns)*(thumb_h+label_h+gap)
            canvas.paste(page,(x,y)); draw.text((gap+(slot%columns)*(thumb_w+gap),y+thumb_h+4),path.stem,fill="black",font=font)
        target=directory/f"contact_{sheet_index//per_sheet+1:02d}.png"; canvas.save(target); outputs.append(target)
    return outputs


def main() -> int:
    report=json.loads(REPORT.read_text(encoding="utf-8")); rows=[]
    destinations=[]
    for item in report.get("sources",[]):
        path=ROOT/item["destination_pdf"]
        if path.exists() and path not in destinations: destinations.append(path)
    for pdf in destinations:
        key=hashlib.sha256(str(pdf).encode()).hexdigest()[:8]
        directory=RENDER_ROOT/f"{pdf.stem}_{key}"
        # A document can be rebuilt with fewer or differently numbered pages while
        # retaining the same destination path.  Reusing its old render directory
        # would mix stale PNGs into the new page count and contact sheets.
        if directory.exists(): shutil.rmtree(directory)
        directory.mkdir(parents=True,exist_ok=True)
        prefix=directory/"page"
        completed=subprocess.run([str(PDFTOPPM),"-png","-r","100",str(pdf),str(prefix)],capture_output=True,text=True,timeout=300)
        pages=sorted(directory.glob("page-*.png"),key=lambda path:int(path.stem.rsplit("-",1)[1]))
        border=[]
        for page in pages:
            with Image.open(page) as image: border.append(border_ink_fraction(image))
        contacts=contact_sheets(pages,directory) if pages else []
        rows.append({"pdf":pdf.relative_to(ROOT).as_posix(),"status":"passed" if completed.returncode==0 and pages else "failed","page_count":len(pages),"max_border_ink_fraction":max(border,default=0.0),"border_review_pages":[i+1 for i,v in enumerate(border) if v>0.01],"contact_sheets":[p.relative_to(ROOT).as_posix() for p in contacts],"stderr":completed.stderr[-1000:]})
    result={"status":"passed" if all(r["status"]=="passed" for r in rows) else "failed","pdf_count":len(rows),"total_pages":sum(r["page_count"] for r in rows),"documents":rows,"visual_inspection_status":"pending_manual_contact_sheet_review"}
    (OUTPUT/"qa"/"pdf_render_report.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"status":result["status"],"pdf_count":result["pdf_count"],"total_pages":result["total_pages"],"contact_sheets":sum(len(r["contact_sheets"]) for r in rows),"border_review_documents":sum(bool(r["border_review_pages"]) for r in rows)},indent=2))
    return 0 if result["status"]=="passed" else 1


if __name__=="__main__":
    raise SystemExit(main())
