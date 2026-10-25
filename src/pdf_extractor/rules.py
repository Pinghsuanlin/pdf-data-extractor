"""Rule-based extraction: value normalisers, regex label matching and table header matching.

Rules run first because they are cheap, deterministic and explainable. Each match carries its
page and bounding box so the audit log can point at where a value came from. The label lists live
in ``schemas.FIELD_SPECS``; tune them against real documents before trusting the output.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any, Optional

from pdf_extractor.parse import ParsedDocument, ParsedPage, ParsedTable, locate_snippet
from pdf_extractor.schemas import FIELD_SPECS, ExtractedValue, FieldKind, FieldSpec

_SCALE_WORDS = {
    "thousand": 1e3,
    "k": 1e3,
    "million": 1e6,
    "mm": 1e6,
    "m": 1e6,
    "billion": 1e9,
    "bn": 1e9,
    "b": 1e9,
}

_MONEY_RE = re.compile(
    r"^\(?\s*(-)?\s*(?:US)?\$?\s*(\d[\d,]*(?:\.\d+)?)\s*(thousand|million|billion|mm|bn|k|m|b)?\s*\)?$",
    re.IGNORECASE,
)
_RATIO_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*(x|%|:\s*1)?$", re.IGNORECASE)

# Value patterns used after a label inside a line of text.
_VALUE_PATTERNS: dict[FieldKind, str] = {
    "money": r"(\(?-?\s*(?:US)?\$?\s*\d[\d,]*(?:\.\d+)?(?:\s*(?:thousand|million|billion|mm|bn|k|m)\b)?\)?)",
    "ratio": r"(\d+(?:\.\d+)?\s*(?:x\b|%|:\s*1\b)?)",
    "id": r"([A-Z0-9][A-Z0-9\-_/]*\d[A-Z0-9\-_/]*)",
    "text": r"(\S.{1,118}?)\s*$",
}
_SEPARATOR = r"\s*(?:\(\$\))?\s*[:#=\-\u2013]?\s*"
_NUMBERING = r"^\s*(?:\(?[0-9a-z]{1,3}[.)]\s+)?"


# --------------------------------------------------------------------------- normalisers


def parse_money(raw: str) -> Optional[float]:
    """Parse a dollar amount such as '$14,500,000', '14.5 million', '$9.2M' or '(1,000)'.

    Returns ``None`` when the text is not a recognisable amount. Parentheses mean negative,
    following accounting convention.
    """
    text = raw.strip().replace("\u00a0", " ")
    match = _MONEY_RE.match(text)
    if not match:
        return None
    sign, number, scale = match.groups()
    try:
        value = float(number.replace(",", ""))
    except ValueError:
        return None
    if scale:
        value *= _SCALE_WORDS[scale.lower()]
    if sign or (text.startswith("(") and text.endswith(")")):
        value = -value
    return value


def parse_ratio(raw: str) -> Optional[float]:
    """Parse a ratio such as '1.45', '1.45x', '1.45:1' or '145%' into a decimal (1.45)."""
    match = _RATIO_RE.match(raw.strip())
    if not match:
        return None
    value = float(match.group(1))
    suffix = (match.group(2) or "").strip().lower()
    if suffix == "%":
        value /= 100.0
    return value


def normalise_text(raw: str) -> Optional[str]:
    """Collapse whitespace and trim trailing punctuation; empty text becomes ``None``."""
    text = re.sub(r"\s+", " ", raw).strip().strip(" .,;:")
    return text or None


def normalise_id(raw: str) -> Optional[str]:
    """Upper-case an identifier and remove internal spaces. Must contain at least one digit."""
    text = re.sub(r"\s+", "", raw).strip(" .,;:").upper()
    if not text or not re.search(r"\d", text):
        return None
    return text


def normalise_value(kind: FieldKind, raw: str) -> Any:
    """Dispatch to the normaliser for a field kind. Returns ``None`` when parsing fails."""
    if raw is None:
        return None
    if kind == "money":
        return parse_money(raw)
    if kind == "ratio":
        return parse_ratio(raw)
    if kind == "id":
        return normalise_id(raw)
    return normalise_text(raw)


# --------------------------------------------------------------------------- label matching


def normalise_label(cell: str) -> str:
    """Normalise a label or header cell: lower case, no numbering, no trailing ':' or '($)'."""
    text = cell.lower().replace("\u00a0", " ")
    text = re.sub(_NUMBERING, "", text)
    text = re.sub(r"\(\$\)|\$", "", text)
    text = re.sub(r"[:#*]+$", "", text.strip())
    return re.sub(r"\s+", " ", text).strip()


def match_label(cell: str, specs: Iterable[FieldSpec]) -> Optional[FieldSpec]:
    """Return the spec whose label equals the normalised cell text, if any.

    Exact equality (after normalisation) is used on purpose: prefix matching would map
    'Total Development Cost per Unit' onto ``total_dev_cost``.
    """
    label = normalise_label(cell)
    if not label:
        return None
    for spec in specs:
        if label in spec.labels:
            return spec
    return None


def _label_pattern(label: str) -> str:
    """Regex for a label with flexible whitespace between its words."""
    return r"\s+".join(re.escape(part) for part in label.split())


# --------------------------------------------------------------------------- extractors


def extract_from_text(page: ParsedPage, spec: FieldSpec) -> list[ExtractedValue]:
    """Find ``label: value`` patterns line by line on one page.

    Free-text fields must start the line (after optional numbering) to avoid capturing prose such
    as 'the applicant shall ...'. Longer labels are tried first so 'applicant name' wins over
    'applicant'.
    """
    found: list[ExtractedValue] = []
    value_re = _VALUE_PATTERNS[spec.kind]
    labels = sorted(spec.labels, key=len, reverse=True)
    for line in page.text.splitlines():
        for label in labels:
            prefix = _NUMBERING if spec.kind == "text" else r"(?:^|\b)"
            sep = r"\s*[:#=\-\u2013]\s*" if spec.kind == "text" else _SEPARATOR
            pattern = rf"{prefix}{_label_pattern(label)}\b{sep}{value_re}"
            match = re.search(pattern, line, re.IGNORECASE)
            if not match:
                continue
            raw = match.group(1).strip()
            value = normalise_value(spec.kind, raw)
            if value is None:
                continue
            found.append(
                ExtractedValue(
                    field=spec.name,
                    value=value,
                    raw=raw,
                    method="rule_regex",
                    page=page.number,
                    bbox=locate_snippet(page, raw) or locate_snippet(page, line),
                    evidence=line.strip(),
                )
            )
            break  # one match per line
    return found


def _first_non_empty(cells: list[str], start: int) -> Optional[int]:
    """Index of the first non-empty cell at or after ``start``."""
    for i in range(start, len(cells)):
        if cells[i].strip():
            return i
    return None


def _looks_like_header(table: ParsedTable) -> bool:
    """A wide first row of words over a row of numbers is a header, even if few labels are known.

    Without this, a header row with one known label and unfamiliar ones next to it is read as a
    key-value row, and the neighbouring header cell ('Agency Request') becomes the value.
    """
    if len(table.rows) < 2 or len(table.rows[0]) < 3:
        return False
    first, second = table.rows[0], table.rows[1]
    if any(re.search(r"\d", c) for c in first):
        return False
    filled = [c for c in second if c.strip()]
    return bool(filled) and sum(bool(re.search(r"\d", c)) for c in filled) * 2 >= len(filled)


def extract_from_table(
    page: ParsedPage, table: ParsedTable, specs: Iterable[FieldSpec]
) -> list[ExtractedValue]:
    """Extract values from one table using two common layouts.

    * Header layout: the first row holds two or more known labels; values come from the first
      data row. A note is added when there are several data rows, because one-record-per-document
      may then be the wrong assumption.
    * Key-value layout: a row holds a known label in some cell and the value in the next
      non-empty cell to its right.
    """
    specs = list(specs)
    found: list[ExtractedValue] = []
    if not table.rows:
        return found

    header = table.rows[0]
    header_specs = [match_label(c, specs) for c in header]
    n_known = sum(s is not None for s in header_specs)
    if (n_known >= 2 or (n_known == 1 and _looks_like_header(table))) and len(table.rows) > 1:
        data_rows = [r for r in table.rows[1:] if any(c.strip() for c in r)]
        if data_rows:
            row_idx = table.rows.index(data_rows[0])
            for col, spec in enumerate(header_specs):
                if spec is None or col >= len(data_rows[0]):
                    continue
                raw = data_rows[0][col]
                value = normalise_value(spec.kind, raw)
                if value is None:
                    continue
                notes = ["multi_row_table"] if len(data_rows) > 1 else []
                found.append(
                    ExtractedValue(
                        field=spec.name,
                        value=value,
                        raw=raw,
                        method="rule_table",
                        page=page.number,
                        bbox=table.cell_bbox(row_idx, col),
                        evidence=f"{header[col]} | {raw}",
                        notes=notes,
                    )
                )
        return found

    for row_idx, row in enumerate(table.rows):
        for col, cell in enumerate(row):
            spec = match_label(cell, specs)
            if spec is None:
                continue
            value_col = _first_non_empty(row, col + 1)
            if value_col is None:
                continue
            raw = row[value_col]
            value = normalise_value(spec.kind, raw)
            if value is None:
                continue
            found.append(
                ExtractedValue(
                    field=spec.name,
                    value=value,
                    raw=raw,
                    method="rule_table",
                    page=page.number,
                    bbox=table.cell_bbox(row_idx, value_col),
                    evidence=f"{cell} | {raw}",
                )
            )
            break  # one label per row
    return found


def _dedupe(values: list[ExtractedValue]) -> list[ExtractedValue]:
    """Drop regex hits that repeat a table hit (same field, value and page).

    pdfplumber's page text also contains table text, so the same value is often found twice.
    The table hit is kept because its cell bbox is more precise.
    """
    table_keys = {(v.field, v.value, v.page) for v in values if v.method == "rule_table"}
    return [
        v
        for v in values
        if v.method == "rule_table" or (v.field, v.value, v.page) not in table_keys
    ]


def extract_rules(
    doc: ParsedDocument, specs: Optional[Iterable[FieldSpec]] = None
) -> dict[str, list[ExtractedValue]]:
    """Run table and regex rules over every page.

    Returns:
        Field name to the list of rule candidates (possibly empty), table hits first.
    """
    specs = list(specs or FIELD_SPECS.values())
    found: list[ExtractedValue] = []
    for page in doc.pages:
        for table in page.tables:
            found.extend(extract_from_table(page, table, specs))
        for spec in specs:
            found.extend(extract_from_text(page, spec))
    found = _dedupe(found)
    ocr_pages = {p.number for p in doc.pages if p.ocr}
    for value in found:
        if value.page in ocr_pages:
            value.notes.append("from_ocr")  # read from OCR text, which can misread characters
    by_field: dict[str, list[ExtractedValue]] = {s.name: [] for s in specs}
    for value in found:
        by_field[value.field].append(value)
    for values in by_field.values():
        values.sort(key=lambda v: (v.method != "rule_table", v.page or 0))
    return by_field
