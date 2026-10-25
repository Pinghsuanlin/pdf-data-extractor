"""Render page 1 of a native and a heavy-scan benchmark document to PNG for the README.

Run ``make benchmark`` first, then ``make samples``. Writes ``docs/images/sample_native.png`` and
``docs/images/sample_scan_heavy.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pdfplumber

PDFS = Path("eval/synthetic/pdfs")
OUT = Path("docs/images")


def first(pattern: str) -> Path:
    matches = sorted(PDFS.glob(pattern))
    if not matches:
        sys.exit(f"No PDF matches {PDFS / pattern}. Run `make benchmark` first.")
    return matches[0]


def render(pdf: Path, target: Path, dpi: int = 80) -> None:
    with pdfplumber.open(str(pdf)) as doc:
        page = doc.pages[0]
        image = page.to_image(resolution=dpi).original.convert("L")
        # Keep the top half: that is where the content sits on these one-page summaries.
        image = image.crop((0, 0, image.width, int(image.height * 0.45)))
        image.save(target, optimize=True)
    print(f"wrote {target}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    render(first("synth_*_kv_table.pdf"), OUT / "sample_native.png")
    render(first("synth_*_kv_table_scan_heavy.pdf"), OUT / "sample_scan_heavy.png")
