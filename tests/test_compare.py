"""``compare`` runs several configurations on the same synthetic documents."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("reportlab")

from pdf_extractor.cli import main, run_compare
from pdf_extractor.llm_extract import FakeLLM
from pdf_extractor.pipeline import PipelineConfig


def test_compare_writes_one_column_per_run(tmp_path: Path) -> None:
    assert main(["synth", "--out", str(tmp_path / "bench"), "--n", "5", "--scanned-share", "0"]) == 0
    summary = run_compare(
        tmp_path / "bench" / "pdfs", tmp_path / "bench" / "gold", tmp_path / "out",
        ["rules", "llm", "vision"], FakeLLM('{"fields": {}}', vision=True), PipelineConfig(ocr="off"),
    )
    assert list(summary) == ["rules", "llm", "vision"]
    text = (tmp_path / "out" / "comparison.md").read_text()
    assert "| Metric | rules | llm | vision |" in text
    assert (tmp_path / "out" / "rules" / "report.md").exists()


def test_compare_skips_llm_runs_without_backend(tmp_path: Path) -> None:
    main(["synth", "--out", str(tmp_path / "bench"), "--n", "3", "--scanned-share", "0"])
    summary = run_compare(tmp_path / "bench" / "pdfs", tmp_path / "bench" / "gold", tmp_path / "out",
                          ["rules", "llm"], None, PipelineConfig(ocr="off"))
    assert list(summary) == ["rules"]
