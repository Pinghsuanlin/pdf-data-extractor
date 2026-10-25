"""Vision route on scanned pages, OCR-tolerant evidence, disagreement handling and segment metrics."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from pdf_extractor.confidence import score_field
from pdf_extractor.evaluate import GoldDocument, evaluate, metrics_markdown, segments_for
from pdf_extractor.llm_extract import FakeLLM, _verify_evidence, vision_pages
from pdf_extractor.parse import ParsedDocument, ParsedPage, page_from_text, parse_pdf
from pdf_extractor.pipeline import PipelineConfig, extract_parsed
from pdf_extractor.schemas import FIELD_SPECS, ExtractedValue


@pytest.fixture
def scanned_pdf(sample_pdf: Path, tmp_path: Path) -> Path:
    from pdf_extractor.synth import render_scanned

    return render_scanned(sample_pdf, tmp_path / "scanned.pdf", "light", random.Random(0))


LLM_REPLY = {
    "fields": {
        "project_id": {"value": "2099-001T", "evidence": "Application Number: 2099-001T", "page": 1},
        "total_dev_cost": {"value": "$10,000,000", "evidence": "Total Development Cost $10,000,000", "page": 1},
    }
}


def test_scanned_pages_are_sent_as_images(scanned_pdf: Path) -> None:
    doc = parse_pdf(scanned_pdf, ocr="off")
    assert vision_pages(doc, "scanned") == [1, 2]
    fake = FakeLLM(LLM_REPLY, vision=True)
    result = extract_parsed(doc, fake, PipelineConfig(llm_policy="all", vision="scanned"))
    assert fake.image_counts == [2]
    assert "attached as images" in fake.prompts[0]
    field = result.fields["total_dev_cost"]
    assert field.value == pytest.approx(10_000_000)
    # No OCR text to check the quote against, so a value read only from the image goes to review.
    assert field.status == "review" and "read_from_image" in field.reasons


def test_text_only_backend_gets_no_images(scanned_pdf: Path) -> None:
    fake = FakeLLM(LLM_REPLY, vision=False)
    extract_parsed(parse_pdf(scanned_pdf, ocr="off"), fake, PipelineConfig(llm_policy="all"))
    assert fake.image_counts == [0]


def test_native_pages_are_not_sent_as_images(sample_doc: ParsedDocument) -> None:
    assert vision_pages(sample_doc, "scanned") == []
    assert vision_pages(sample_doc, "all") == [1]
    assert vision_pages(sample_doc, "off") == []


def test_evidence_check_tolerates_ocr_noise_only_on_ocr_pages() -> None:
    ocr_page = ParsedPage(number=1, text="Tota1 Hard Costs: $17,775,000\nFunding Source: NHTF", ocr=True)
    native = ParsedPage(number=1, text=ocr_page.text, ocr=False)
    quote = "Total Hard Costs: $17,775,000"
    assert _verify_evidence(ParsedDocument("a.pdf", [ocr_page]), quote, 1) == "fuzzy"
    assert _verify_evidence(ParsedDocument("a.pdf", [native]), quote, 1) is None
    assert _verify_evidence(ParsedDocument("a.pdf", [native]), "Funding Source: NHTF", 1) == "exact"


def test_disagreement_prefers_llm_when_rule_value_is_implausible() -> None:
    spec = FIELD_SPECS["hard_construction_cost"]
    rule = [ExtractedValue(field=spec.name, value=1.0, raw="$1", method="rule_regex", page=1)]
    llm = [ExtractedValue(field=spec.name, value=17_775_000.0, raw="$17,775,000", method="llm",
                          page=1, evidence_verified=True)]
    result = score_field(spec, rule, llm)
    assert result.value == pytest.approx(17_775_000)
    assert result.status == "review"


def test_verified_llm_only_value_on_native_text_is_trusted(sample_doc: ParsedDocument) -> None:
    doc = ParsedDocument("x.pdf", [page_from_text(1, "The sponsor is Example Housing LP.")])
    fake = FakeLLM({"fields": {"applicant_name": {"value": "Example Housing LP",
                                                  "evidence": "The sponsor is Example Housing LP", "page": 1}}})
    result = extract_parsed(doc, fake, PipelineConfig(llm_policy="hard"))
    assert result.fields["applicant_name"].status == "ok"


def test_segments_break_down_accuracy(sample_doc: ParsedDocument) -> None:
    result = extract_parsed(sample_doc, None, PipelineConfig(llm_policy="off"))
    right = GoldDocument("synthetic.pdf", {"project_id": "2099-001T"},
                         tags={"layout": "form", "scanned": False, "traps": ["short_money"]})
    assert segments_for(right.tags) == ["scanned=no", "layout=form", "trap=short_money"]
    metrics = evaluate([result], [right])
    assert metrics["segments"]["scanned=no"]["accuracy"] == 1.0
    assert metrics["segments"]["trap=short_money"]["documents"] == 1
    assert "| scanned=no | 1 |" in metrics_markdown(metrics)


def _ocr_doc() -> ParsedDocument:
    page = page_from_text(1, "Application Number: 2099-001T\nTotal Development Cost: $10,000,000")
    page.ocr = True
    return ParsedDocument("scan.pdf", [page])


def test_ocr_rule_values_need_a_second_reader() -> None:
    alone = extract_parsed(_ocr_doc(), None, PipelineConfig(llm_policy="off")).fields["total_dev_cost"]
    assert alone.value == pytest.approx(10_000_000)
    assert alone.status == "review" and "ocr_unconfirmed" in alone.reasons

    agree = FakeLLM({"fields": {"total_dev_cost": {"value": "$10,000,000",
                                                   "evidence": "Total Development Cost: $10,000,000", "page": 1}}})
    confirmed = extract_parsed(_ocr_doc(), agree, PipelineConfig(llm_policy="all")).fields["total_dev_cost"]
    assert confirmed.status == "ok" and "rule_llm_agree" in confirmed.reasons
