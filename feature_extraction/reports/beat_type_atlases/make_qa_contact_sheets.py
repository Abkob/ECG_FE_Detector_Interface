"""Create readable 2x2 contact sheets from Poppler-rendered report pages."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent / "rendered_final"


def main() -> None:
    for record_dir in sorted(path for path in ROOT.iterdir() if path.is_dir()):
        pages = sorted(record_dir.glob("page-*.png"))
        for start in range(0, len(pages), 4):
            chosen = pages[start : start + 4]
            opened = [Image.open(path).convert("RGB") for path in chosen]
            thumb_width = 1000
            thumbs = []
            for image in opened:
                height = round(image.height * thumb_width / image.width)
                thumbs.append(image.resize((thumb_width, height), Image.Resampling.LANCZOS))
            cell_height = max(image.height for image in thumbs) + 38
            sheet = Image.new("RGB", (2 * thumb_width, 2 * cell_height), "white")
            draw = ImageDraw.Draw(sheet)
            for index, (image, path) in enumerate(zip(thumbs, chosen, strict=True)):
                x = (index % 2) * thumb_width
                y = (index // 2) * cell_height + 30
                sheet.paste(image, (x, y))
                draw.text((x + 8, 7 + (index // 2) * cell_height), path.stem, fill="black")
            output = record_dir / f"contact-{start + 1:02d}-{start + len(chosen):02d}.png"
            sheet.save(output, optimize=True)


if __name__ == "__main__":
    main()
