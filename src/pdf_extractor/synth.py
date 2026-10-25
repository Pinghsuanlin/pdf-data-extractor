"""Synthetic benchmark: funding-application PDFs with known answers.

Why synthetic: the real FHFC cost data is published mostly as Excel pro formas, so a hand-labelled
PDF set is slow to build and small. A generator gives a labelled set of any size in seconds, with
the hard cases chosen on purpose rather than left to chance. It does **not** replace a check on
real documents; the README reports the two separately.

Every document gets a gold file in the same format as hand labels (``eval/gold``) plus ``tags``
used by the evaluation to break results down:

* ``layout``: ``form`` (label: value lines), ``kv_table`` (two-column table), ``header_table``
  (one-row table with field headers), ``narrative`` (values inside sentences) or ``multipage``
  (fields spread over three pages among boilerplate).
* ``scanned``: the pages are rendered to images and degraded (skew, noise, blur, JPEG), so the PDF
  has no text layer at all. ``scan_quality`` is ``light`` or ``heavy``.
* ``traps``: deliberate difficulties, e.g. ``per_unit_distractor`` ('Total Development Cost per
  Unit' next to the real total), ``unseen_labels`` (synonyms the rules do not know),
  ``short_money`` ('$14.5M'), ``no_funding_amount`` (optional fields absent, gold ``null``) and
  ``leverage_inconsistent`` (the stated leverage disagrees with cost / funding; the gold value is
  what the document states, and a good system should flag it for review).

All names and numbers are invented, and every page carries a 'synthetic' footer.
"""

from __future__ import annotations

import io
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

LAYOUTS = ("form", "kv_table", "header_table", "narrative", "multipage")

_NAME_A = ("Harbor", "Sunrise", "Cypress", "Magnolia", "Palm", "Bayview", "Coral", "Riverside",
           "Oak", "Heron", "Pelican", "Seagrape", "Lakeshore", "Mangrove", "Citrus", "Willow")
_NAME_B = ("View", "Point", "Gardens", "Crossing", "Commons", "Terrace", "Landing", "Pines",
           "Village", "Place", "Square", "Ridge")
_SUFFIX = ("Housing, LP", "Apartments, LLC", "Partners, Ltd.", "Development, LP", "Homes, LLC")
_PROGRAMS = ("SAIL", "HOME-ARP", "ELI Loan", "Viability Loan", "NHTF")

# Label variants. "known" ones are in schemas.FIELD_SPECS; "unseen" ones are realistic synonyms the
# rules were never told about, so only a model that reads for meaning can catch them.
LABELS: dict[str, dict[str, tuple[str, ...]]] = {
    "project_id": {"known": ("Application Number", "Application ID", "Project Number"),
                   "unseen": ("Submission Reference", "File No.")},
    "applicant_name": {"known": ("Name of Applicant", "Applicant Name", "Applicant Entity"),
                       "unseen": ("Borrowing Entity", "Sponsor Entity")},
    "total_dev_cost": {"known": ("Total Development Cost", "Total Project Cost"),
                       "unseen": ("Total Development Budget", "All-In Project Budget")},
    "hard_construction_cost": {"known": ("Hard Construction Cost", "Total Hard Costs"),
                               "unseen": ("Construction Contract Amount", "Building Hard Costs")},
    "funding_source": {"known": ("Funding Source", "Funding Program"),
                       "unseen": ("Programme Applied Under", "Source of Agency Funds")},
    "funding_amount": {"known": ("Requested Amount", "Funding Amount", "Loan Amount"),
                       "unseen": ("Agency Request", "Corporation Funding Requested")},
    "leverage_ratio": {"known": ("Leverage Ratio",), "unseen": ("Leveraging Multiple",)},
}

_BOILERPLATE = (
    "The Applicant certifies that the information in this Application is true and correct and "
    "acknowledges that the Corporation may verify any statement made herein.",
    "All costs are stated in US dollars and reflect the Development Cost Pro Forma submitted with "
    "this Application. Contingency is included in the hard cost line where applicable.",
    "This page intentionally summarises information provided elsewhere in the Application; in the "
    "event of a conflict the Development Cost Pro Forma governs.",
    "Site control, zoning and environmental documentation are provided as separate exhibits and "
    "are not reproduced in this summary.",
)


@dataclass
class SynthDoc:
    """One generated document: its gold values, how they are written, and its tags."""

    doc_id: str
    layout: str
    gold: dict[str, Any]
    shown: dict[str, str]
    labels: dict[str, str]
    traps: list[str] = field(default_factory=list)
    scanned: bool = False
    scan_quality: Optional[str] = None
    per_unit: Optional[float] = None


# --------------------------------------------------------------------------- values


def _money_text(value: float, short: bool) -> str:
    if short:
        return f"${value / 1e6:.1f}M" if value % 100_000 == 0 else f"${value / 1e6:.2f}M"
    return f"${value:,.0f}"


def make_doc(index: int, rng: random.Random, layout: Optional[str] = None) -> SynthDoc:
    """Draw one document's values, labels and traps (rendering happens separately)."""
    layout = layout or LAYOUTS[index % len(LAYOUTS)]
    year = rng.choice((2023, 2024, 2025))
    project_id = f"{year}-{rng.randint(100, 399):03d}{rng.choice(('C', 'BS', 'S', 'H'))}"
    applicant = f"{rng.choice(_NAME_A)} {rng.choice(_NAME_B)} {rng.choice(_SUFFIX)}"
    if rng.random() < 0.5:
        funding_source = f"RFA {year}-{rng.randint(101, 312)}"
    else:
        funding_source = rng.choice(_PROGRAMS)
    tdc = rng.randint(60, 450) * 100_000.0
    hard = round(tdc * rng.uniform(0.55, 0.75), -3)
    leverage_target = rng.uniform(1.2, 3.5)
    funding = round(tdc / leverage_target, -4)
    leverage = round(tdc / funding, 2)

    traps: list[str] = []
    if rng.random() < 0.35:
        traps.append("per_unit_distractor")
    if rng.random() < 0.30:
        traps.append("unseen_labels")
    if rng.random() < 0.30:
        traps.append("short_money")
    # The two rarer traps follow a fixed schedule so every benchmark of 20+ documents has several.
    if index % 7 == 3:
        traps.append("no_funding_amount")
    elif index % 9 == 5:
        traps.append("leverage_inconsistent")

    gold: dict[str, Any] = {
        "project_id": project_id,
        "applicant_name": applicant,
        "total_dev_cost": tdc,
        "hard_construction_cost": hard,
        "funding_source": funding_source,
        "funding_amount": funding,
        "leverage_ratio": leverage,
    }
    if "no_funding_amount" in traps:
        gold["funding_amount"] = None
        gold["leverage_ratio"] = None
    if "leverage_inconsistent" in traps:
        gold["leverage_ratio"] = round(leverage + rng.choice((-1, 1)) * rng.uniform(0.3, 0.8), 2)

    short = "short_money" in traps
    shown: dict[str, str] = {}
    for name, value in gold.items():
        if value is None:
            continue
        if name in {"total_dev_cost", "hard_construction_cost", "funding_amount"}:
            # Short form only on the large totals; it is what busy summaries actually do.
            shown[name] = _money_text(value, short and name != "hard_construction_cost")
            if shown[name].endswith("M"):
                # Keep gold equal to what the page says, to the cent.
                gold[name] = round(float(shown[name].strip("$M")) * 1e6, 2)
        elif name == "leverage_ratio":
            shown[name] = f"{value:.2f}"
        else:
            shown[name] = str(value)

    unseen = "unseen_labels" in traps
    labels: dict[str, str] = {}
    for name, variants in LABELS.items():
        # With the trap on, about half the fields use a synonym the rules do not know.
        pool = variants["unseen"] if unseen and rng.random() < 0.5 else variants["known"]
        labels[name] = rng.choice(pool)

    units = rng.randint(40, 180)
    return SynthDoc(
        doc_id=f"synth_{index:03d}_{layout}",
        layout=layout,
        gold=gold,
        shown=shown,
        labels=labels,
        traps=traps,
        per_unit=round(tdc / units, -2) if "per_unit_distractor" in traps else None,
    )


# --------------------------------------------------------------------------- rendering


def _styles() -> dict[str, Any]:
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet

    base = getSampleStyleSheet()
    return {
        "title": base["Title"],
        "h2": base["Heading2"],
        "body": ParagraphStyle("body", parent=base["Normal"], fontSize=10, leading=14, spaceAfter=6),
        "line": ParagraphStyle("line", parent=base["Normal"], fontSize=10.5, leading=16),
        "cell": ParagraphStyle("cell", parent=base["Normal"], fontSize=7.5, leading=9),
    }


def _footer(canvas: Any, doc: Any) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.drawString(54, 30, "SYNTHETIC BENCHMARK DOCUMENT - invented data, not a real application")
    canvas.drawRightString(558, 30, f"Page {doc.page}")
    canvas.restoreState()


def _kv_lines(d: SynthDoc, names: tuple[str, ...], st: dict[str, Any]) -> list[Any]:
    from reportlab.platypus import Paragraph

    return [Paragraph(f"{d.labels[n]}: {d.shown[n]}", st["line"]) for n in names if n in d.shown]


def _grid_table(rows: list[list[Any]], widths: Optional[list[float]] = None) -> Any:
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    table = Table(rows, colWidths=widths)
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEEEEE")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return table


def _distractor(d: SynthDoc) -> Optional[str]:
    if d.per_unit is None:
        return None
    return f"Total Development Cost per Unit: ${d.per_unit:,.0f}"


def _story(d: SynthDoc, rng: random.Random) -> list[Any]:
    from reportlab.platypus import PageBreak, Paragraph, Spacer

    st = _styles()
    story: list[Any] = [Paragraph("Funding Application Summary", st["title"])]
    all_fields = tuple(LABELS)

    if d.layout == "form":
        story.append(Paragraph(rng.choice(_BOILERPLATE), st["body"]))
        story += _kv_lines(d, all_fields[:2], st)
        story.append(Spacer(1, 8))
        story += _kv_lines(d, all_fields[2:4], st)
        if (text := _distractor(d)) is not None:
            story.append(Paragraph(text, st["line"]))
        story.append(Spacer(1, 8))
        story += _kv_lines(d, all_fields[4:], st)
        story.append(Paragraph(rng.choice(_BOILERPLATE), st["body"]))

    elif d.layout == "kv_table":
        story.append(Paragraph(rng.choice(_BOILERPLATE), st["body"]))
        rows = [["Item", "Value"]] + [[d.labels[n], d.shown[n]] for n in all_fields if n in d.shown]
        if d.per_unit is not None:
            rows.insert(4, ["Total Development Cost per Unit", f"${d.per_unit:,.0f}"])
        story.append(_grid_table(rows, [230, 200]))
        story.append(Spacer(1, 10))
        story.append(Paragraph(rng.choice(_BOILERPLATE), st["body"]))

    elif d.layout == "header_table":
        names = [n for n in all_fields if n in d.shown]
        header = [Paragraph(d.labels[n], st["cell"]) for n in names]
        values = [Paragraph(d.shown[n], st["cell"]) for n in names]
        story.append(Paragraph(rng.choice(_BOILERPLATE), st["body"]))
        story.append(_grid_table([header, values]))
        if (text := _distractor(d)) is not None:
            story.append(Spacer(1, 8))
            story.append(Paragraph(text, st["line"]))

    elif d.layout == "narrative":
        s = d.shown
        parts = [
            f"Application {s['project_id']} was submitted by {s['applicant_name']} under "
            f"{s['funding_source']}.",
            f"The development has a total development cost of {s['total_dev_cost']}, of which hard "
            f"construction accounts for {s['hard_construction_cost']}.",
        ]
        if d.per_unit is not None:
            parts.append(f"This equates to roughly ${d.per_unit:,.0f} per unit.")
        if "funding_amount" in s:
            parts.append(f"The applicant requests {s['funding_amount']} from the Corporation, "
                         f"a leverage of {s['leverage_ratio']}.")
        else:
            parts.append("The funding request will be confirmed at credit underwriting.")
        story.append(Paragraph(" ".join(parts), st["body"]))
        story.append(Paragraph(rng.choice(_BOILERPLATE), st["body"]))

    else:  # multipage
        story.append(Paragraph("Section 1. Applicant", st["h2"]))
        story += _kv_lines(d, ("project_id", "applicant_name"), st)
        story += [Paragraph(t, st["body"]) for t in rng.sample(_BOILERPLATE, 2)]
        story.append(PageBreak())
        story.append(Paragraph("Section 2. Development Budget", st["h2"]))
        rows = [["Cost category", "Amount"],
                [d.labels["hard_construction_cost"], d.shown["hard_construction_cost"]],
                ["General Contractor Fee", f"${d.gold['hard_construction_cost'] * 0.12:,.0f}"],
                ["Soft Costs", f"${d.gold['total_dev_cost'] * 0.18:,.0f}"],
                [d.labels["total_dev_cost"], d.shown["total_dev_cost"]]]
        if d.per_unit is not None:
            rows.append(["Total Development Cost per Unit", f"${d.per_unit:,.0f}"])
        story.append(_grid_table(rows, [250, 180]))
        story.append(Paragraph(rng.choice(_BOILERPLATE), st["body"]))
        story.append(PageBreak())
        story.append(Paragraph("Section 3. Financing", st["h2"]))
        story += _kv_lines(d, ("funding_source", "funding_amount", "leverage_ratio"), st)
        story.append(Paragraph(rng.choice(_BOILERPLATE), st["body"]))
    return story


def render_native(d: SynthDoc, path: Path, rng: random.Random) -> Path:
    """Write the document as a normal PDF with a text layer."""
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate

    SimpleDocTemplate(str(path), pagesize=letter, title=d.doc_id, author="synthetic generator",
                      topMargin=54, bottomMargin=54, invariant=1).build(
        _story(d, rng), onFirstPage=_footer, onLaterPages=_footer)
    return path


def degrade_page(image: Any, quality: str, rng: random.Random) -> Any:
    """Make a clean page render look like a photocopy or a phone scan."""
    from PIL import Image, ImageFilter

    heavy = quality == "heavy"
    img = image.convert("L")
    angle = rng.uniform(0.6, 1.8) if heavy else rng.uniform(0.2, 0.7)
    img = img.rotate(rng.choice((-1, 1)) * angle, resample=Image.BICUBIC, expand=False, fillcolor=255)
    img = img.filter(ImageFilter.GaussianBlur(radius=1.1 if heavy else 0.5))
    # Speckle plus a grey cast. The noise comes from the seeded generator (not PIL's own random
    # source), so the same seed always gives byte-identical scans.
    noise = Image.frombytes("L", img.size, rng.randbytes(img.size[0] * img.size[1]))
    img = Image.blend(img, noise, 0.10 if heavy else 0.05)
    img = img.point(lambda p: int(25 + p * (0.82 if heavy else 0.9)))
    if heavy:  # lower effective resolution, then back up, like a cheap scanner
        w, h = img.size
        img = img.resize((int(w * 0.6), int(h * 0.6)), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=35 if heavy else 60)
    buf.seek(0)
    return Image.open(buf).convert("L")


def render_scanned(native_pdf: Path, path: Path, quality: str, rng: random.Random, dpi: int = 150) -> Path:
    """Rasterise a native PDF, degrade every page and save an image-only PDF (no text layer)."""
    import pdfplumber

    pages = []
    with pdfplumber.open(str(native_pdf)) as pdf:
        for page in pdf.pages:
            pages.append(degrade_page(page.to_image(resolution=dpi).original, quality, rng))
    fixed = time.gmtime(1_767_225_600)  # 2026-01-01; fixed dates keep the file byte-identical for a given seed
    pages[0].save(str(path), "PDF", resolution=float(dpi), save_all=True, append_images=pages[1:],
                  creationDate=fixed, modDate=fixed)
    return path


# --------------------------------------------------------------------------- generator


def gold_record(d: SynthDoc, source_file: str) -> dict[str, Any]:
    """Gold file in the same format as hand labels, plus ``tags`` for segment breakdowns."""
    return {
        "source_file": source_file,
        "labelled_by": "synthetic generator (values known by construction)",
        "labelled_on": "generated; see manifest.json for the seed",
        "fields": d.gold,
        "tags": {
            "layout": d.layout,
            "scanned": d.scanned,
            "scan_quality": d.scan_quality,
            "traps": d.traps,
        },
    }


def generate(
    out_dir: str | Path,
    n: int = 40,
    seed: int = 7,
    scanned_share: float = 0.3,
) -> list[SynthDoc]:
    """Generate ``n`` PDFs into ``out_dir/pdfs`` and gold files into ``out_dir/gold``.

    The same seed always gives the same values, layouts, traps and scan choices. Scanned documents
    are spread evenly over layouts; half are ``light`` scans and half ``heavy``.
    """
    rng = random.Random(seed)
    out = Path(out_dir)
    pdf_dir, gold_dir = out / "pdfs", out / "gold"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    gold_dir.mkdir(parents=True, exist_ok=True)
    for old in [*pdf_dir.glob("synth_*.pdf"), *gold_dir.glob("synth_*.json")]:
        old.unlink()

    n_scanned = round(n * scanned_share)
    # Every k-th document is scanned so scans cover all layouts rather than clustering.
    scanned_idx = set(range(1, n, max(1, n // max(1, n_scanned)))[:n_scanned]) if n_scanned else set()
    docs: list[SynthDoc] = []
    for i in range(n):
        d = make_doc(i, rng)
        native = pdf_dir / f"{d.doc_id}.pdf"
        render_native(d, native, rng)
        if i in scanned_idx:
            d.scanned = True
            d.scan_quality = "heavy" if len([x for x in docs if x.scanned]) % 2 else "light"
            d.doc_id = f"{d.doc_id}_scan_{d.scan_quality}"
            scanned = pdf_dir / f"{d.doc_id}.pdf"
            render_scanned(native, scanned, d.scan_quality, rng)
            native.unlink()
            final = scanned
        else:
            final = native
        (gold_dir / f"{d.doc_id}.json").write_text(
            json.dumps(gold_record(d, final.name), indent=2), encoding="utf-8")
        docs.append(d)
    manifest = {
        "seed": seed,
        "n": n,
        "scanned_share": scanned_share,
        "documents": [{"doc": d.doc_id, "layout": d.layout, "scanned": d.scanned,
                       "scan_quality": d.scan_quality, "traps": d.traps} for d in docs],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return docs
