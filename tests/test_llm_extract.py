"""LLM response parsing, evidence checks and field routing, all with FakeLLM."""

from __future__ import annotations

import json

import pytest

from pdf_extractor.llm_extract import (
    FakeLLM,
    LLMResponseError,
    build_prompt,
    get_backend,
    llm_extract,
    parse_llm_response,
    specs_for_llm,
)
from pdf_extractor.parse import ParsedDocument
from pdf_extractor.rules import extract_rules
from pdf_extractor.schemas import FIELD_SPECS


def test_parse_response_with_code_fence_and_bare_mapping() -> None:
    text = '```json\n{"applicant_name": {"value": "Example Housing LP", "page": 1}}\n```'
    parsed = parse_llm_response(text)
    assert parsed.fields["applicant_name"].value == "Example Housing LP"


def test_parse_response_drops_unknown_fields_and_blank_values() -> None:
    parsed = parse_llm_response(
        json.dumps({"fields": {"made_up": {"value": "x"}, "funding_source": {"value": "null"}}})
    )
    assert "made_up" not in parsed.fields
    assert parsed.fields["funding_source"].value is None


@pytest.mark.parametrize("bad", ["no json here", "{not: valid}", '{"fields": {"project_id": {"page": 0}}}'])
def test_parse_response_errors(bad: str) -> None:
    with pytest.raises(LLMResponseError):
        parse_llm_response(bad)


def test_llm_extract_normalises_and_verifies_evidence(sample_doc: ParsedDocument) -> None:
    fake = FakeLLM(
        {
            "fields": {
                "applicant_name": {
                    "value": "Example Housing LP",
                    "evidence": "Name of Applicant: Example Housing LP",
                    "page": 1,
                },
                "total_dev_cost": {"value": "$10,000,000", "evidence": "invented quote", "page": 1},
            }
        }
    )
    specs = [FIELD_SPECS["applicant_name"], FIELD_SPECS["total_dev_cost"]]
    found, warnings = llm_extract(sample_doc, specs, fake)
    assert warnings == []
    assert found["applicant_name"][0].evidence_verified is True
    assert found["total_dev_cost"][0].value == pytest.approx(10_000_000)
    assert found["total_dev_cost"][0].evidence_verified is False
    assert "[Page 1]" in fake.prompts[0]


def test_backend_failure_becomes_warning(sample_doc: ParsedDocument) -> None:
    def boom(_: str) -> str:
        raise TimeoutError("backend down")

    found, warnings = llm_extract(sample_doc, [FIELD_SPECS["applicant_name"]], FakeLLM(boom))
    assert found["applicant_name"] == []
    assert warnings and warnings[0].startswith("llm_backend_error")


def test_specs_for_llm_policies(sample_doc: ParsedDocument) -> None:
    rules = extract_rules(sample_doc)
    rules["leverage_ratio"] = []  # pretend the rule missed it
    hard = {s.name for s in specs_for_llm(rules, "hard")}
    assert hard == {"applicant_name", "funding_source", "leverage_ratio"}
    assert specs_for_llm(rules, "off") == []
    assert len(specs_for_llm(rules, "all")) == len(FIELD_SPECS)


def test_prompt_lists_fields_and_contract() -> None:
    prompt = build_prompt([FIELD_SPECS["project_id"]], "[Page 1]\ntext")
    assert '"project_id"' in prompt and "verbatim" in prompt


def test_get_backend_none_and_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    assert get_backend() is None
    with pytest.raises(ValueError):
        get_backend("does-not-exist")
