"""Synthetic benchmark generator: deterministic, consistent with its gold labels, scans are image-only."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pytest

pytest.importorskip("reportlab")

from pdf_extractor.evaluate import load_gold
from pdf_extractor.parse import parse_pdf
from pdf_extractor.synth import LAYOUTS, generate, make_doc


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


@pytest.fixture(scope="module")
def bench(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("bench")
    generate(out, n=10, seed=3, scanned_share=0.2)
    return out


def test_same_seed_same_documents() -> None:
    a = [make_doc(i, random.Random(11)).gold for i in range(5)]
    b = [make_doc(i, random.Random(11)).gold for i in range(5)]
    assert a == b


def test_layouts_rotate_and_files_are_written(bench: Path) -> None:
    manifest = json.loads((bench / "manifest.json").read_text())
    assert manifest["seed"] == 3
    assert {m["layout"] for m in manifest["documents"]} == set(LAYOUTS)
    assert len(list((bench / "pdfs").glob("*.pdf"))) == 10
    assert len(load_gold(bench / "gold")) == 10


def test_native_pdfs_contain_every_gold_value_as_written(bench: Path) -> None:
    for gold_file in (bench / "gold").glob("*.json"):
        data = json.loads(gold_file.read_text())
        if data["tags"]["scanned"]:
            continue
        doc = parse_pdf(bench / "pdfs" / data["source_file"], ocr="off")
        text = _squash(" ".join(p.text for p in doc.pages))
        assert _squash(str(data["fields"]["project_id"])) in text
        if data["tags"]["layout"] != "header_table":  # names wrap inside narrow table cells
            assert _squash(data["fields"]["applicant_name"]) in text


def test_scanned_pdfs_have_no_text_layer(bench: Path) -> None:
    scanned = [p for p in (bench / "pdfs").glob("*_scan_*.pdf")]
    assert len(scanned) == 2
    for path in scanned:
        doc = parse_pdf(path, ocr="off")
        assert all(not p.text.strip() for p in doc.pages)


def test_scheduled_traps_set_gold_correctly() -> None:
    doc = make_doc(3, random.Random(1))  # index % 7 == 3 -> no funding amount
    assert "no_funding_amount" in doc.traps
    assert doc.gold["funding_amount"] is None and doc.gold["leverage_ratio"] is None
    doc = make_doc(5, random.Random(1))  # index % 9 == 5 -> inconsistent leverage
    assert "leverage_inconsistent" in doc.traps
    implied = doc.gold["total_dev_cost"] / doc.gold["funding_amount"]
    assert abs(implied - doc.gold["leverage_ratio"]) > 0.2


def test_same_seed_gives_byte_identical_files(tmp_path: Path) -> None:
    for name in ("a", "b"):
        generate(tmp_path / name, n=4, seed=5, scanned_share=0.5)
    for pdf in (tmp_path / "a" / "pdfs").glob("*.pdf"):
        assert pdf.read_bytes() == (tmp_path / "b" / "pdfs" / pdf.name).read_bytes()
