"""End-to-end orchestration: parse, rules, LLM (where needed), confidence."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from pdf_extractor.confidence import DEFAULT_CONFIG, ConfidenceConfig, score_document
from pdf_extractor.llm_extract import LLMBackend, LLMBackendError, llm_extract, specs_for_llm
from pdf_extractor.parse import OcrMode, ParsedDocument, parse_pdf
from pdf_extractor.rules import extract_rules
from pdf_extractor.schemas import DocumentResult

LLMPolicy = Literal["off", "hard", "all"]
VisionMode = Literal["off", "scanned", "all"]


@dataclass
class PipelineConfig:
    """Run settings.

    Attributes:
        llm_policy: ``off`` (rules only), ``hard`` (free-text fields and rule misses) or ``all``.
        ocr: OCR mode passed to the parser.
        max_context_chars: Upper bound on document text sent to the LLM.
        vision: Which pages to send as images when the backend can read them: ``off``,
            ``scanned`` (pages with no native text layer) or ``all``.
        strict_llm: stop the run when an LLM call fails (evaluation) instead of carrying on with
            rule output only (production extraction).
        confidence: Scoring thresholds and tolerances.
    """

    llm_policy: LLMPolicy = "hard"
    ocr: OcrMode = "auto"
    vision: VisionMode = "scanned"
    strict_llm: bool = False
    max_context_chars: int = 6000
    confidence: ConfidenceConfig = field(default_factory=lambda: DEFAULT_CONFIG)


def extract_parsed(
    doc: ParsedDocument,
    backend: Optional[LLMBackend] = None,
    config: Optional[PipelineConfig] = None,
) -> DocumentResult:
    """Run extraction and scoring on an already parsed document (handy for tests)."""
    config = config or PipelineConfig()
    rule_candidates = extract_rules(doc)
    llm_candidates: dict = {}
    warnings: list[str] = []
    if backend is not None and config.llm_policy != "off":
        specs = specs_for_llm(rule_candidates, config.llm_policy)
        llm_candidates, warnings = llm_extract(
            doc, specs, backend, config.max_context_chars, vision=config.vision, strict=config.strict_llm
        )
    if not any(p.text.strip() for p in doc.pages):
        warnings.append("no_text_layer: document may be scanned; install the OCR extra")
    if any(p.ocr for p in doc.pages):
        warnings.append("ocr_used: values read from scanned pages")
    return score_document(
        Path(doc.path).name, rule_candidates, llm_candidates, config.confidence, warnings
    )


def extract_document(
    path: str | Path,
    backend: Optional[LLMBackend] = None,
    config: Optional[PipelineConfig] = None,
) -> DocumentResult:
    """Parse one PDF and extract a scored record from it."""
    config = config or PipelineConfig()
    return extract_parsed(parse_pdf(path, ocr=config.ocr), backend, config)


def extract_many(
    paths: Iterable[str | Path],
    backend: Optional[LLMBackend] = None,
    config: Optional[PipelineConfig] = None,
) -> list[DocumentResult]:
    """Extract every PDF in ``paths``. A failing file becomes a warning-only empty record."""
    results: list[DocumentResult] = []
    for path in paths:
        try:
            results.append(extract_document(path, backend, config))
        except LLMBackendError:
            raise  # strict mode: never turn a failed LLM call into a rules-only result
        except Exception as exc:  # corrupt or encrypted PDFs should not stop a batch
            empty = score_document(Path(path).name, {}, {}, (config or PipelineConfig()).confidence)
            empty.warnings.append(f"parse_failed: {exc}")
            results.append(empty)
    return results
