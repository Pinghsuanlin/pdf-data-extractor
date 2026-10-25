"""Streamlit review UI: upload a PDF, inspect extracted fields, accept or fix flagged ones.

Run with ``make app`` (or ``streamlit run app.py``). Needs the ``app`` extra.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st

from pdf_extractor.confidence import ConfidenceConfig
from pdf_extractor.export import (
    apply_human_edits,
    audit_frame,
    display_frame,
    records_frame,
    to_excel_bytes,
)
from pdf_extractor.llm_extract import get_backend
from pdf_extractor.pipeline import PipelineConfig, extract_document
from pdf_extractor.rules import normalise_value
from pdf_extractor.schemas import FIELD_SPECS, BBox, DocumentResult

st.set_page_config(page_title="PDF data extractor", layout="wide")


def render_highlight(pdf_path: str, page_number: int, bbox: Optional[BBox]) -> Any:
    """Render one page with the value's bounding box outlined (needs pdfplumber's renderer)."""
    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number - 1]
        image = page.to_image(resolution=110)
        if bbox is not None:
            image.draw_rect(bbox, stroke="red", stroke_width=3, fill=(255, 0, 0, 40))
        return image.original


@st.cache_data(show_spinner="Extracting...")
def run_extraction(pdf_bytes: bytes, name: str, backend_name: str, policy: str, threshold: float):
    """Write the upload to a temp file and run the pipeline. Cached per file and settings."""
    tmp_dir = Path(tempfile.mkdtemp())
    path = tmp_dir / name
    path.write_bytes(pdf_bytes)
    config = PipelineConfig(
        llm_policy=policy,  # type: ignore[arg-type]
        confidence=ConfidenceConfig(review_threshold=threshold),
    )
    return extract_document(path, get_backend(backend_name), config), str(path)


st.title("PDF data extractor")
st.caption("Rules first, LLM second, every value scored and traceable to its page.")

with st.sidebar:
    st.header("Settings")
    backend_name = st.selectbox("LLM backend", ["none", "gemini", "ollama", "groq", "mistral"])
    policy = st.selectbox("LLM policy", ["hard", "all", "off"], help="Which fields go to the LLM")
    threshold = st.slider("Review threshold", 0.0, 1.0, 0.70, 0.05)

upload = st.file_uploader("Upload a PDF", type=["pdf"])
if upload is None:
    st.info("Upload a funding application or report PDF to start.")
    st.stop()

try:
    result, pdf_path = run_extraction(upload.getvalue(), upload.name, backend_name, policy, threshold)
except Exception as exc:  # show backend or parsing errors instead of a stack trace
    st.error(f"Extraction failed: {exc}")
    st.stop()

doc: DocumentResult = st.session_state.get(f"doc::{upload.name}", result)
for warning in doc.warnings:
    st.warning(warning)

col_a, col_b, col_c = st.columns(3)
col_a.metric("Record confidence", f"{doc.record_confidence:.0%}")
col_b.metric("Status", doc.data_confidence_label)
col_c.metric("Fields to review", sum(r.status != "ok" for r in doc.fields.values()))

st.subheader("Extracted fields")
st.dataframe(display_frame([doc]), use_container_width=True, hide_index=True)

st.subheader("Review")
review_rows = [
    {
        "field": name,
        "value": "" if res.value is None else str(res.value),
        "confidence": round(res.confidence, 2),
        "status": res.status,
        "reasons": "; ".join(res.reasons),
        "page": res.chosen.page if res.chosen else None,
        "accept": res.status == "ok",
    }
    for name, res in doc.fields.items()
]
edited = st.data_editor(
    pd.DataFrame(review_rows),
    disabled=["field", "confidence", "status", "reasons", "page"],
    hide_index=True,
    use_container_width=True,
    key=f"editor::{upload.name}",
)

if st.button("Apply accepted values"):
    edits: dict[str, Any] = {}
    for _, row in edited.iterrows():
        if not row["accept"]:
            continue
        current = doc.fields[row["field"]]
        text = str(row["value"]).strip()
        value = normalise_value(FIELD_SPECS[row["field"]].kind, text) if text else None
        if current.status != "ok" or value != current.value:
            edits[row["field"]] = value
    st.session_state[f"doc::{upload.name}"] = apply_human_edits(doc, edits)
    st.success(f"Applied {len(edits)} reviewer decision(s).")
    st.rerun()

st.subheader("Where did this value come from?")
field_name = st.selectbox("Field", list(doc.fields))
chosen = doc.fields[field_name].chosen
if chosen is None or chosen.page is None:
    st.write("No source location for this field.")
else:
    st.write(f"Method: `{chosen.method}`, page {chosen.page}, evidence: {chosen.evidence!r}")
    try:
        st.image(render_highlight(pdf_path, chosen.page, chosen.bbox))
    except Exception as exc:  # rendering needs pypdfium2, which ships with pdfplumber
        st.write(f"Could not render page: {exc}")

st.subheader("Download")
st.download_button(
    "Records (CSV)",
    records_frame([doc]).to_csv(index=False).encode(),
    file_name="records.csv",
)
st.download_button(
    "Audit log (CSV)",
    audit_frame([doc]).to_csv(index=False).encode(),
    file_name="audit_log.csv",
)
st.download_button("Workbook (Excel)", to_excel_bytes([doc]), file_name="extraction.xlsx")
