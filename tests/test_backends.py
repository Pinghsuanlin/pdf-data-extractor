"""Gemini payloads, retries, the disk cache and backend selection. No network is used."""

from __future__ import annotations

import io
import json
import urllib.error
from typing import Any

import pytest

from pdf_extractor import llm_extract as le
from pdf_extractor.llm_extract import CachedBackend, FakeLLM, GeminiBackend, LLMResponseError, get_backend


def test_gemini_payload_has_prompt_images_and_json_mode() -> None:
    backend = GeminiBackend(api_key="test-key")
    payload = backend.build_payload("hello", [b"\x89PNG-one", b"\x89PNG-two"])
    parts = payload["contents"][0]["parts"]
    assert parts[0] == {"text": "hello"}
    assert [p["inline_data"]["mime_type"] for p in parts[1:]] == ["image/png", "image/png"]
    assert payload["generationConfig"]["responseMimeType"] == "application/json"
    assert payload["generationConfig"]["temperature"] == 0


def test_gemini_reply_text_and_blocked_reply() -> None:
    reply = {"candidates": [{"content": {"parts": [{"text": '{"fields"'}, {"text": ": {}}"}]}}]}
    assert GeminiBackend.reply_text(reply) == '{"fields": {}}'
    with pytest.raises(LLMResponseError):
        GeminiBackend.reply_text({"promptFeedback": {"blockReason": "SAFETY"}})


def test_gemini_needs_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiBackend()


def test_get_backend_wraps_in_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path))
    backend = get_backend("gemini")
    assert isinstance(backend, CachedBackend) and backend.supports_vision
    assert isinstance(get_backend("gemini", cache=False), GeminiBackend)
    assert get_backend("none") is None


class _Reply(io.BytesIO):
    def __enter__(self) -> _Reply:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://x", code, "err", {}, io.BytesIO(b"detail"))  # type: ignore[arg-type]


def test_post_json_retries_rate_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def fake_urlopen(request: Any, timeout: int) -> _Reply:
        calls.append(1)
        if len(calls) < 3:
            raise _http_error(429)
        return _Reply(json.dumps({"ok": True}).encode())

    waits: list[float] = []
    monkeypatch.setattr(le.urllib.request, "urlopen", fake_urlopen)
    assert le._post_json("https://x", {}, {}, 5, sleep=waits.append) == {"ok": True}
    assert len(calls) == 3 and waits == [2.0, 4.0]


def test_post_json_does_not_retry_auth_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def fake_urlopen(request: Any, timeout: int) -> _Reply:
        calls.append(1)
        raise _http_error(401)

    monkeypatch.setattr(le.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="HTTP 401"):
        le._post_json("https://x", {}, {}, 5, sleep=lambda s: None)
    assert len(calls) == 1


def test_cache_hits_skip_the_backend(tmp_path: Any) -> None:
    inner = FakeLLM('{"fields": {}}', vision=True)
    cached = CachedBackend(inner, tmp_path)
    assert cached.complete("p") == cached.complete("p")
    assert len(inner.prompts) == 1 and cached.hits == 1 and cached.misses == 1
    cached.complete_vision("p", [b"img-a"])
    cached.complete_vision("p", [b"img-b"])  # different image, different key
    assert len(inner.prompts) == 3


QUOTA_BODY = json.dumps({"error": {
    "code": 429,
    "message": "You exceeded your current quota, please check your plan and billing details.\nmore",
    "details": [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
         "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier", "quotaValue": "20"}]},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "41s"},
    ],
}})


def test_quota_error_names_the_quota() -> None:
    detail = le._http_error_detail(QUOTA_BODY)
    assert "PerDay" in detail and "limit=20" in detail and "retry_after=41s" in detail
    assert le._http_error_detail("not json") == "not json"


class _FailingLLM:
    name = "failing"

    def complete(self, prompt: str) -> str:
        raise RuntimeError("HTTP 429: quota")


def test_strict_mode_stops_instead_of_silently_dropping_the_llm(sample_doc: Any, sample_pdf: Any) -> None:
    from pdf_extractor.pipeline import PipelineConfig, extract_many, extract_parsed

    lenient = extract_parsed(sample_doc, _FailingLLM(), PipelineConfig(llm_policy="all"))
    assert any(w.startswith("llm_backend_error") for w in lenient.warnings)
    with pytest.raises(le.LLMBackendError):
        extract_parsed(sample_doc, _FailingLLM(), PipelineConfig(llm_policy="all", strict_llm=True))
    with pytest.raises(le.LLMBackendError):  # the batch loop must not swallow it either
        extract_many([sample_pdf], _FailingLLM(), PipelineConfig(llm_policy="all", ocr="off", strict_llm=True))
