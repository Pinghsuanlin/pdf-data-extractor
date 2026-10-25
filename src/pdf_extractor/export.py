"""Export results to CSV / Excel, with an audit log and a review queue.

Three tables come out of every run:

* ``records``: one row per document. Numeric columns stay numeric (empty when missing) so the
  file is analysis-ready; a ``flagged_fields`` column lists what needs review. ``display_frame``
  gives the human-facing version with ``[Missing]`` / ``[Flagged]`` markers, as in the v1 table.
* ``audit_log``: one row per field with the chosen value, method, page, bounding box, raw text,
  evidence, confidence and reasons, plus how many candidates were considered.
* ``review_queue``: the low-confidence and missing required fields, lowest confidence first.
"""

from __future__ import annotations

import io
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from pdf_extractor.confidence import review_queue
from pdf_extractor.schemas import (
    FIELD_SPECS,
    FLAGGED_MARKER,
    MISSING_MARKER,
    OUTPUT_FIELDS,
    DocumentResult,
    ExtractedValue,
)


def records_frame(results: Iterable[DocumentResult]) -> pd.DataFrame:
    """One analysis-ready row per document."""
    rows = []
    for doc in results:
        row: dict[str, Any] = {"source_file": doc.source_file}
        row.update(doc.values())
        row["data_confidence"] = round(doc.record_confidence, 3)
        row["data_confidence_label"] = doc.data_confidence_label
        row["flagged_fields"] = ";".join(
            name for name, r in doc.fields.items() if r.status == "review"
        )
        row["missing_fields"] = ";".join(
            name for name, r in doc.fields.items() if r.status == "missing"
        )
        rows.append(row)
    columns = [
        "source_file",
        *OUTPUT_FIELDS,
        "data_confidence",
        "data_confidence_label",
        "flagged_fields",
        "missing_fields",
    ]
    return pd.DataFrame(rows, columns=columns)


def _display_value(kind: str, value: Any) -> str:
    """Format a value for people: dollars with separators, ratios to two decimals."""
    if kind == "money":
        return f"${value:,.0f}"
    if kind == "ratio":
        return f"{value:.2f}"
    return str(value)


def display_frame(results: Iterable[DocumentResult]) -> pd.DataFrame:
    """Human-facing table in the style of the v1 target schema, with markers for gaps."""
    rows = []
    for doc in results:
        row: dict[str, Any] = {"source_file": doc.source_file}
        for name, res in doc.fields.items():
            if res.status == "missing":
                row[name] = MISSING_MARKER
            elif res.status == "review":
                row[name] = f"{FLAGGED_MARKER} {_display_value(FIELD_SPECS[name].kind, res.value)}"
            else:
                row[name] = _display_value(FIELD_SPECS[name].kind, res.value)
        row["data_confidence"] = doc.data_confidence_label
        rows.append(row)
    return pd.DataFrame(rows)


def _bbox_columns(value: ExtractedValue | None) -> dict[str, Any]:
    """Split a bbox into four columns (empty when unknown)."""
    if value is None or value.bbox is None:
        return {"x0": None, "top": None, "x1": None, "bottom": None}
    x0, top, x1, bottom = value.bbox
    return {"x0": round(x0, 1), "top": round(top, 1), "x1": round(x1, 1), "bottom": round(bottom, 1)}


def audit_frame(results: Iterable[DocumentResult]) -> pd.DataFrame:
    """One row per document and field: where the value came from and why it was scored so."""
    rows = []
    for doc in results:
        for name, res in doc.fields.items():
            chosen = res.chosen
            rows.append(
                {
                    "source_file": doc.source_file,
                    "field": name,
                    "value": res.value,
                    "status": res.status,
                    "confidence": round(res.confidence, 3),
                    "reasons": ";".join(res.reasons),
                    "method": chosen.method if chosen else None,
                    "page": chosen.page if chosen else None,
                    **_bbox_columns(chosen),
                    "raw_text": chosen.raw if chosen else None,
                    "evidence": chosen.evidence if chosen else None,
                    "n_candidates": len(res.candidates),
                    "previous_value": res.previous_value,
                }
            )
    return pd.DataFrame(rows)


def review_frame(results: Iterable[DocumentResult]) -> pd.DataFrame:
    """The human-review queue as a table."""
    rows = []
    for doc, res in review_queue(list(results)):
        rows.append(
            {
                "source_file": doc.source_file,
                "field": res.field,
                "value": res.value,
                "status": res.status,
                "confidence": round(res.confidence, 3),
                "reasons": ";".join(res.reasons),
                "page": res.chosen.page if res.chosen else None,
            }
        )
    return pd.DataFrame(
        rows, columns=["source_file", "field", "value", "status", "confidence", "reasons", "page"]
    )


def apply_human_edits(doc: DocumentResult, edits: dict[str, Any]) -> DocumentResult:
    """Apply reviewer decisions in place and return the document.

    Each edit sets the value, marks it as human-verified with confidence 1.0 and keeps the
    machine value in ``previous_value`` so the audit log shows what changed. Passing the current
    value accepts it unchanged.
    """
    for name, new_value in edits.items():
        if name not in doc.fields:
            raise KeyError(f"Unknown field '{name}'.")
        res = doc.fields[name]
        res.previous_value = res.value
        res.value = new_value
        res.confidence = 1.0
        res.status = "ok" if new_value is not None else "missing"
        res.reasons = [*res.reasons, "human_verified"]
        res.chosen = ExtractedValue(
            field=name,
            value=new_value,
            raw=str(new_value),
            method="human",
            page=res.chosen.page if res.chosen else None,
            bbox=res.chosen.bbox if res.chosen else None,
            evidence=res.chosen.evidence if res.chosen else None,
        )
    return doc


def to_excel_bytes(results: list[DocumentResult]) -> bytes:
    """Workbook with records, review_queue and audit_log sheets, as bytes (for downloads)."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        records_frame(results).to_excel(writer, sheet_name="records", index=False)
        review_frame(results).to_excel(writer, sheet_name="review_queue", index=False)
        audit_frame(results).to_excel(writer, sheet_name="audit_log", index=False)
    return buffer.getvalue()


def export_results(
    results: list[DocumentResult], out_dir: str | Path, formats: Iterable[str] = ("csv", "xlsx")
) -> list[Path]:
    """Write records, review queue and audit log to ``out_dir``. Returns the written paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    formats = set(formats)
    if "csv" in formats:
        for name, frame in (
            ("records", records_frame(results)),
            ("review_queue", review_frame(results)),
            ("audit_log", audit_frame(results)),
        ):
            path = out / f"{name}.csv"
            frame.to_csv(path, index=False)
            written.append(path)
    if "xlsx" in formats:
        path = out / "extraction.xlsx"
        path.write_bytes(to_excel_bytes(results))
        written.append(path)
    return written
