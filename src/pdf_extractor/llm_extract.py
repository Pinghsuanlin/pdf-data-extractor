"""LLM extraction for the fields rules cannot handle reliably.

Design choices:

* **Backends are swappable and free to run.** ``ollama`` (local, e.g. ``mistral``), ``groq``,
  ``mistral`` and ``gemini`` (hosted, free tiers at the time of writing) share one small interface.
  ``FakeLLM`` returns canned output so tests never touch the network.
* **Vision for scanned pages.** A backend that can read images (``gemini``) is sent the page images
  of scanned pages alongside any OCR text, so a value OCR garbled can still be read. Everything
  else (schema check, normalisers, evidence check) is the same as the text route.
* **Robust and cheap to re-run.** Hosted calls retry on rate limits and transient errors with
  backoff, can be spaced out with ``LLM_MIN_INTERVAL_S`` for free tiers, and every response is
  cached on disk (``.cache/llm``) so re-running an evaluation costs nothing.
* **The LLM is an untrusted source.** Its JSON is validated against ``schemas.LLMExtraction``,
  values go through the same normalisers as rule output, and each value must come with a verbatim
  evidence quote. A quote that is not on the cited page is marked unverified and costs confidence,
  which is a cheap guard against invented values.
* **Context is kept small.** Only the pages that mention a requested field's labels are sent,
  up to ``max_chars``, so small local models stay inside their context window.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, Optional, Protocol, Union

from pydantic import ValidationError

from pdf_extractor.parse import ParsedDocument, ParsedPage, locate_snippet
from pdf_extractor.rules import normalise_value
from pdf_extractor.schemas import FIELD_SPECS, ExtractedValue, FieldSpec, LLMExtraction

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 120
RETRY_STATUS = {429, 500, 502, 503, 504}
MAX_RETRIES = 4
VISION_DPI = 150
MAX_VISION_PAGES = 4


class LLMResponseError(ValueError):
    """Raised when an LLM response cannot be parsed into the expected schema."""


class LLMBackendError(RuntimeError):
    """Raised in strict mode when the backend call itself fails (quota, auth, network).

    Evaluation runs use strict mode: a run where some LLM calls silently failed would report rules-only
    numbers under the LLM's name, so it is better to stop and resume later from the cache.
    """


def _http_error_detail(body: str) -> str:
    """Pull the useful part out of an API error body: the message plus any quota id."""
    try:
        err = json.loads(body).get("error", {})
    except (ValueError, AttributeError):
        return body[:500]
    parts = [str(err.get("message", "")).split("\n")[0][:300]]
    for detail in err.get("details", []) or []:
        for violation in detail.get("violations", []) or []:
            quota = violation.get("quotaId") or violation.get("quotaMetric")
            if quota:
                parts.append(f"quota={quota} limit={violation.get('quotaValue', '?')}")
        if detail.get("retryDelay"):
            parts.append(f"retry_after={detail['retryDelay']}")
    return " | ".join(p for p in parts if p)


class LLMBackend(Protocol):
    """Minimal interface every backend implements.

    Backends that can read images also set ``supports_vision = True`` and implement
    ``complete_vision(prompt, images)`` where ``images`` are PNG bytes in page order.
    """

    name: str

    def complete(self, prompt: str) -> str:
        """Return the raw text completion for ``prompt``."""
        ...


def supports_vision(backend: Optional[LLMBackend]) -> bool:
    """True when ``backend`` can take page images."""
    return bool(getattr(backend, "supports_vision", False)) and hasattr(backend, "complete_vision")


class FakeLLM:
    """Deterministic backend for tests and offline demos.

    Args:
        response: A JSON string, a dict (serialised to JSON) or a callable that maps the prompt to
            a response string.
    """

    name = "fake"

    def __init__(
        self,
        response: Union[str, dict[str, Any], Callable[[str], str]],
        vision: bool = False,
    ) -> None:
        self._response = response
        self.supports_vision = vision
        self.prompts: list[str] = []
        self.image_counts: list[int] = []

    def complete(self, prompt: str) -> str:
        """Record the prompt and return the canned response."""
        self.prompts.append(prompt)
        self.image_counts.append(0)
        if callable(self._response):
            return self._response(prompt)
        if isinstance(self._response, dict):
            return json.dumps(self._response)
        return self._response

    def complete_vision(self, prompt: str, images: Sequence[bytes]) -> str:
        """Same as ``complete`` but also records how many images were attached."""
        out = self.complete(prompt)
        self.image_counts[-1] = len(images)
        return out


def _post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: int,
    retries: int = MAX_RETRIES,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """POST a JSON payload and return the decoded reply, retrying transient failures.

    Rate limits (429) and server errors (5xx) are retried with exponential backoff (2, 4, 8, ...
    seconds), honouring a ``Retry-After`` header when the server sends one. Client errors such as a
    bad key (401/403) fail immediately, because retrying cannot fix them.
    """
    body = json.dumps(payload).encode("utf-8")
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json", **headers}, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code not in RETRY_STATUS or attempt == retries:
                body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
                detail = _http_error_detail(body)
                raise RuntimeError(f"HTTP {exc.code} from {url.split('?')[0]}: {detail}") from exc
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            wait = float(retry_after) if retry_after and retry_after.isdigit() else 2.0 ** (attempt + 1)
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries:
                raise
            wait = 2.0 ** (attempt + 1)
            logger.info("Network error (%s); retrying in %.0fs", exc, wait)
        logger.info("Retrying %s in %.0fs (attempt %d of %d)", url.split("?")[0], wait, attempt + 1, retries)
        sleep(wait)
    raise RuntimeError("unreachable")  # pragma: no cover


class _Throttle:
    """Space calls at least ``LLM_MIN_INTERVAL_S`` seconds apart (free tiers limit requests/minute)."""

    def __init__(self, min_interval: Optional[float] = None) -> None:
        env = os.getenv("LLM_MIN_INTERVAL_S", "0")
        self.min_interval = float(min_interval if min_interval is not None else env or 0)
        self._last = 0.0

    def wait(self) -> None:
        gap = time.monotonic() - self._last
        if self.min_interval and gap < self.min_interval:
            time.sleep(self.min_interval - gap)
        self._last = time.monotonic()


class OllamaBackend:
    """Local model served by Ollama (``ollama pull mistral`` then ``ollama serve``)."""

    name = "ollama"
    supports_vision = False

    def __init__(self, model: Optional[str] = None, host: Optional[str] = None) -> None:
        self.model = model or os.getenv("OLLAMA_MODEL", "mistral")
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")

    def complete(self, prompt: str) -> str:
        """Call ``/api/generate`` in JSON mode with temperature 0."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }
        reply = _post_json(f"{self.host}/api/generate", payload, {}, DEFAULT_TIMEOUT_S)
        return str(reply.get("response", ""))


class OpenAICompatibleBackend:
    """Hosted chat-completions API with an OpenAI-compatible schema (Groq, Mistral)."""

    supports_vision = False

    def __init__(self, name: str, url: str, api_key_env: str, model: str) -> None:
        self.name = name
        self.url = url
        self.model = model
        self.api_key = os.getenv(api_key_env)
        self._throttle = _Throttle()
        if not self.api_key:
            raise RuntimeError(f"Set {api_key_env} in your environment or .env to use '{name}'.")

    def complete(self, prompt: str) -> str:
        """Call the chat-completions endpoint in JSON mode with temperature 0."""
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "You extract data and reply with JSON only."},
                {"role": "user", "content": prompt},
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        self._throttle.wait()
        reply = _post_json(self.url, payload, headers, DEFAULT_TIMEOUT_S)
        return str(reply["choices"][0]["message"]["content"])


class GeminiBackend:
    """Google Gemini via the Generative Language REST API (free key from Google AI Studio).

    Reads text and page images, and is asked for JSON directly (``responseMimeType``). The model
    name is configurable with ``GEMINI_MODEL`` because Google renames models often; check the
    current id in AI Studio if a call returns 404.
    """

    name = "gemini"
    supports_vision = True
    url_template = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None) -> None:
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self._throttle = _Throttle()
        if not self.api_key:
            raise RuntimeError("Set GEMINI_API_KEY in your environment or .env to use 'gemini'.")

    def build_payload(self, prompt: str, images: Sequence[bytes] = ()) -> dict[str, Any]:
        """Request body: the prompt, then each image as inline PNG data."""
        parts: list[dict[str, Any]] = [{"text": prompt}]
        for image in images:
            parts.append(
                {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(image).decode("ascii")}}
            )
        return {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
        }

    @staticmethod
    def reply_text(reply: dict[str, Any]) -> str:
        """Concatenate the text parts of the first candidate (empty when the reply was blocked)."""
        candidates = reply.get("candidates") or []
        if not candidates:
            raise LLMResponseError(f"Gemini returned no candidates: {reply.get('promptFeedback')}")
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(str(p.get("text", "")) for p in parts)

    def _call(self, prompt: str, images: Sequence[bytes]) -> str:
        self._throttle.wait()
        url = self.url_template.format(model=self.model)
        reply = _post_json(url, self.build_payload(prompt, images), {"x-goog-api-key": self.api_key}, DEFAULT_TIMEOUT_S)
        return self.reply_text(reply)

    def complete(self, prompt: str) -> str:
        """Text-only request."""
        return self._call(prompt, ())

    def complete_vision(self, prompt: str, images: Sequence[bytes]) -> str:
        """Request with page images attached after the prompt."""
        return self._call(prompt, images)


class CachedBackend:
    """Disk cache in front of any backend, keyed on backend, model, prompt and image bytes.

    Re-running an evaluation with the same documents and prompt then makes no API calls, which
    keeps free-tier quotas for new experiments and makes results reproducible.
    """

    def __init__(self, inner: LLMBackend, cache_dir: str | Path = ".cache/llm") -> None:
        self.inner = inner
        self.name = inner.name
        self.supports_vision = supports_vision(inner)
        self.cache_dir = Path(cache_dir)
        self.hits = 0
        self.misses = 0

    def _key(self, prompt: str, images: Sequence[bytes]) -> Path:
        digest = hashlib.sha256()
        digest.update(f"{self.inner.name}|{getattr(self.inner, 'model', '')}|".encode())
        digest.update(prompt.encode("utf-8"))
        for image in images:
            digest.update(hashlib.sha256(image).digest())
        return self.cache_dir / f"{digest.hexdigest()}.json"

    def _cached(self, prompt: str, images: Sequence[bytes], call: Callable[[], str]) -> str:
        path = self._key(prompt, images)
        if path.exists():
            self.hits += 1
            return str(json.loads(path.read_text(encoding="utf-8"))["response"])
        response = call()
        self.misses += 1
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"backend": self.name, "response": response}), encoding="utf-8")
        return response

    def complete(self, prompt: str) -> str:
        return self._cached(prompt, (), lambda: self.inner.complete(prompt))

    def complete_vision(self, prompt: str, images: Sequence[bytes]) -> str:
        call = self.inner.complete_vision  # type: ignore[attr-defined]
        return self._cached(prompt, images, lambda: call(prompt, images))


def get_backend(name: Optional[str] = None, cache: bool = True) -> Optional[LLMBackend]:
    """Create a backend by name: ``none``, ``ollama``, ``groq``, ``mistral`` or ``gemini``.

    The default comes from the ``LLM_BACKEND`` environment variable, then ``none`` (rules only).
    Model names are configurable because hosted model catalogues change often. With ``cache`` on,
    responses are stored under ``LLM_CACHE_DIR`` (default ``.cache/llm``).
    """
    backend = _make_backend((name or os.getenv("LLM_BACKEND", "none")).lower())
    if backend is None or not cache:
        return backend
    return CachedBackend(backend, os.getenv("LLM_CACHE_DIR", ".cache/llm"))


def _make_backend(name: str) -> Optional[LLMBackend]:
    if name in {"none", "off", ""}:
        return None
    if name == "gemini":
        return GeminiBackend()
    if name == "ollama":
        return OllamaBackend()
    if name == "groq":
        return OpenAICompatibleBackend(
            "groq",
            "https://api.groq.com/openai/v1/chat/completions",
            "GROQ_API_KEY",
            os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        )
    if name == "mistral":
        return OpenAICompatibleBackend(
            "mistral",
            "https://api.mistral.ai/v1/chat/completions",
            "MISTRAL_API_KEY",
            os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
        )
    raise ValueError(f"Unknown LLM backend '{name}'. Use none, ollama, groq, mistral or gemini.")


# --------------------------------------------------------------------------- prompting


def select_context(doc: ParsedDocument, specs: Iterable[FieldSpec], max_chars: int = 6000) -> str:
    """Pick the pages most likely to hold the requested fields, tagged with page numbers.

    Pages are ranked by how many label mentions they contain; the first page is always included
    because cover pages often hold the applicant and application number.
    """
    labels = [label for spec in specs for label in spec.labels]

    def hits(page: ParsedPage) -> int:
        text = page.text.lower()
        return sum(text.count(label) for label in labels)

    ranked = sorted(doc.pages, key=lambda p: (p.number != 1, -hits(p), p.number))
    chunks: list[str] = []
    used = 0
    for page in ranked:
        if page.number != 1 and hits(page) == 0:
            continue
        chunk = f"[Page {page.number}]\n{page.text.strip()}\n"
        if used + len(chunk) > max_chars:
            chunk = chunk[: max(0, max_chars - used)]
        if chunk:
            chunks.append(chunk)
            used += len(chunk)
        if used >= max_chars:
            break
    return "\n".join(chunks)


def build_prompt(specs: Iterable[FieldSpec], context: str, image_pages: Sequence[int] = ()) -> str:
    """Build the extraction prompt: field definitions, output contract and the document text.

    When ``image_pages`` is given, the prompt says which pages are attached as images (in that
    order) and that any OCR text for them may contain errors.
    """
    specs = list(specs)
    field_lines = "\n".join(f'- "{s.name}" ({s.kind}): {s.description}' for s in specs)
    example = {s.name: {"value": None, "evidence": None, "page": None} for s in specs}
    source = "the document text"
    image_note = ""
    if image_pages:
        listed = ", ".join(str(n) for n in image_pages)
        source = "the document text and the attached page images"
        image_note = (
            f"5. Pages {listed} are scanned and attached as images, in that order. Read values from "
            "the images; any text shown for those pages is OCR and may contain errors.\n"
        )
    return (
        "Extract the fields below from an affordable-housing funding document.\n"
        "Rules:\n"
        f"1. Use only {source}. If a field is not stated, use null. Do not guess.\n"
        "2. 'evidence' must be a short verbatim quote (under 20 words) copied from the page.\n"
        "3. 'page' is the number in the [Page N] marker where the evidence appears.\n"
        "4. Write money as it appears (e.g. '$14,500,000'); ratios as decimals (e.g. '1.45').\n"
        f"{image_note}\n"
        f"Fields:\n{field_lines}\n\n"
        'Reply with JSON only, shaped as {"fields": ' + json.dumps(example) + "}\n\n"
        f"Document:\n{context}"
    )


def vision_pages(doc: ParsedDocument, mode: str = "scanned") -> list[int]:
    """Pages to send as images: ``off`` none, ``scanned`` pages without a native text layer
    (OCR'd or empty), ``all`` every page. Capped at ``MAX_VISION_PAGES``."""
    if mode == "off":
        return []
    if mode == "all":
        chosen = [p.number for p in doc.pages]
    else:
        chosen = [p.number for p in doc.pages if p.ocr or not p.text.strip()]
    return chosen[:MAX_VISION_PAGES]


def page_images(doc: ParsedDocument, numbers: Sequence[int], dpi: int = VISION_DPI) -> list[bytes]:
    """Render the given 1-based pages to greyscale PNG bytes. Empty when the PDF is not on disk."""
    if not numbers or not Path(doc.path).is_file():
        return []
    import pdfplumber

    images: list[bytes] = []
    with pdfplumber.open(doc.path) as pdf:
        for number in numbers:
            image = pdf.pages[number - 1].to_image(resolution=dpi).original.convert("L")
            buf = io.BytesIO()
            image.save(buf, format="PNG", optimize=True)
            images.append(buf.getvalue())
    return images


def parse_llm_response(text: str) -> LLMExtraction:
    """Parse and validate an LLM reply, tolerating code fences and leading chatter.

    Accepts either ``{"fields": {...}}`` or a bare ``{field: {...}}`` mapping.

    Raises:
        LLMResponseError: when no valid JSON object matching the schema can be recovered.
    """
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise LLMResponseError("No JSON object in LLM response.")
    try:
        data = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMResponseError(f"Invalid JSON from LLM: {exc}") from exc
    if isinstance(data, dict) and "fields" not in data:
        data = {"fields": data}
    try:
        return LLMExtraction.model_validate(data)
    except ValidationError as exc:
        raise LLMResponseError(f"LLM JSON does not match schema: {exc}") from exc


def _squash(text: str) -> str:
    """Lower-case and remove all whitespace, for robust evidence matching."""
    return re.sub(r"\s+", "", text).lower()


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def token_overlap(evidence: str, page_text: str) -> float:
    """Share of the evidence's word tokens that appear anywhere in the page text."""
    ev = _tokens(evidence)
    if not ev:
        return 0.0
    page = set(_tokens(page_text))
    return sum(t in page for t in ev) / len(ev)


def _verify_evidence(
    doc: ParsedDocument, evidence: Optional[str], page: Optional[int], ocr_min_overlap: float = 0.8
) -> Optional[str]:
    """Check the evidence quote against the cited page (or every page, if no page given).

    Returns ``"exact"`` when the quote is in the page text (ignoring case and spacing), ``"fuzzy"``
    when the page is OCR text and at least ``ocr_min_overlap`` of the quote's words appear on it
    (OCR is noisy, so exact matching would reject correct quotes), and ``None`` otherwise. A scanned
    page with no OCR text cannot verify anything, so values read only from its image stay unverified.
    """
    if not evidence:
        return None
    needle = _squash(evidence)
    pages = [p for p in doc.pages if page is None or p.number == page]
    if any(needle in _squash(p.text) for p in pages):
        return "exact"
    if any(p.ocr and token_overlap(evidence, p.text) >= ocr_min_overlap for p in pages):
        return "fuzzy"
    return None


def llm_extract(
    doc: ParsedDocument,
    specs: Iterable[FieldSpec],
    backend: LLMBackend,
    max_chars: int = 6000,
    vision: str = "off",
    strict: bool = False,
) -> tuple[dict[str, list[ExtractedValue]], list[str]]:
    """Ask the LLM for ``specs`` and convert its answer into normalised candidates.

    Args:
        vision: ``off``, ``scanned`` or ``all``; which pages to attach as images. Ignored unless the
            backend supports images.
        strict: raise ``LLMBackendError`` when the backend call fails instead of returning a
            warning (used by evaluation, where a silent failure would corrupt the results).

    Returns:
        ``(candidates_by_field, warnings)``. Backend or parsing failures become warnings and an
        empty result, so the pipeline still returns rule output and routes fields to review.
    """
    specs = list(specs)
    out: dict[str, list[ExtractedValue]] = {s.name: [] for s in specs}
    if not specs:
        return out, []
    image_pages = vision_pages(doc, vision) if supports_vision(backend) else []
    images = page_images(doc, image_pages) if image_pages else []
    if not images:
        image_pages = []
    context = select_context(doc, specs, max_chars=max_chars)
    for number in image_pages:  # make sure every attached page has a marker the model can cite
        if f"[Page {number}]" not in context:
            context += f"\n[Page {number}]\n(scanned page, see image)\n"
    prompt = build_prompt(specs, context, image_pages)
    try:
        if images:
            raw = backend.complete_vision(prompt, images)  # type: ignore[attr-defined]
        else:
            raw = backend.complete(prompt)
        parsed = parse_llm_response(raw)
    except LLMResponseError as exc:
        return out, [f"llm_parse_error: {exc}"]
    except Exception as exc:  # network errors, timeouts, auth, quota
        if strict:
            raise LLMBackendError(str(exc)) from exc
        logger.warning("LLM backend %s failed: %s", getattr(backend, "name", "?"), exc)
        return out, [f"llm_backend_error: {exc}"]

    for spec in specs:
        item = parsed.fields.get(spec.name)
        if item is None or item.value is None:
            continue
        value = normalise_value(spec.kind, item.value)
        notes: list[str] = []
        if value is None:
            notes.append("llm_value_unparseable")
        if item.page in image_pages:
            notes.append("read_from_image")
        check = _verify_evidence(doc, item.evidence, item.page)
        verified = check is not None
        if check == "fuzzy":
            notes.append("evidence_fuzzy_ocr")
        page_obj = next((p for p in doc.pages if p.number == item.page), None)
        bbox = None
        if page_obj is not None:
            bbox = locate_snippet(page_obj, item.value) or (
                locate_snippet(page_obj, item.evidence) if item.evidence else None
            )
        out[spec.name].append(
            ExtractedValue(
                field=spec.name,
                value=value,
                raw=item.value,
                method="llm",
                page=item.page,
                bbox=bbox,
                evidence=item.evidence,
                evidence_verified=verified,
                notes=notes,
            )
        )
    return out, []


def specs_for_llm(
    rule_candidates: dict[str, list[ExtractedValue]], policy: str = "hard"
) -> list[FieldSpec]:
    """Choose which fields to send to the LLM.

    Args:
        rule_candidates: Output of ``rules.extract_rules``.
        policy: ``"off"`` sends nothing; ``"hard"`` sends fields whose spec says ``always`` plus any
            field the rules missed; ``"all"`` sends every field (best agreement signal, most cost).
    """
    if policy == "off":
        return []
    specs = []
    for name, spec in FIELD_SPECS.items():
        if policy == "all" or spec.llm_policy == "always" or not rule_candidates.get(name):
            specs.append(spec)
    return specs
