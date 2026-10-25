"""Target schema, field specifications and the internal result types.

Two kinds of objects live here:

* ``FieldSpec`` / ``FIELD_SPECS``: plain, declarative descriptions of each output field
  (type, label synonyms used by the rule extractor, whether the LLM should always cross-check it).
* Pydantic models (``LLMField``, ``LLMExtraction``) that validate whatever JSON the LLM returns.
  The LLM is treated as an untrusted source, so its output is schema-checked before use.

Internal pipeline results (``ExtractedValue``, ``FieldResult``, ``DocumentResult``) are dataclasses
because they never cross a trust boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

FieldKind = Literal["id", "text", "money", "ratio"]
Method = Literal["rule_table", "rule_regex", "llm", "human"]
Status = Literal["ok", "review", "missing"]
BBox = tuple[float, float, float, float]  # (x0, top, x1, bottom) in PDF points, origin top-left

MISSING_MARKER = "[Missing]"
FLAGGED_MARKER = "[Flagged]"


@dataclass(frozen=True)
class FieldSpec:
    """Declarative description of one output field.

    Attributes:
        name: Column name in the output table.
        kind: Value type, which decides the normaliser and the format checks.
        labels: Label or header synonyms the rule extractor looks for (lower case).
        description: Plain-English meaning, also sent to the LLM in the prompt.
        required: Required fields count towards the record-level confidence.
        llm_policy: ``"always"`` asks the LLM even when a rule matched (free-text fields, where
            rules are weakest and a second opinion is worth the cost); ``"fallback"`` asks the LLM
            only when no rule matched.
    """

    name: str
    kind: FieldKind
    labels: tuple[str, ...]
    description: str
    required: bool = True
    llm_policy: Literal["always", "fallback"] = "fallback"


FIELD_SPECS: dict[str, FieldSpec] = {
    spec.name: spec
    for spec in (
        FieldSpec(
            name="project_id",
            kind="id",
            labels=(
                "application number",
                "application no",
                "application id",
                "project id",
                "project number",
                "development id",
            ),
            description="Application or project identifier assigned by the funding agency.",
        ),
        FieldSpec(
            name="applicant_name",
            kind="text",
            labels=("name of applicant", "applicant name", "applicant entity", "applicant"),
            description="Legal name of the applicant entity (not the developer or the property).",
            llm_policy="always",
        ),
        FieldSpec(
            name="total_dev_cost",
            kind="money",
            labels=(
                "total development cost",
                "total development costs",
                "total project cost",
                "tdc",
            ),
            description="Total development cost of the project, in US dollars.",
        ),
        FieldSpec(
            name="hard_construction_cost",
            kind="money",
            labels=(
                "hard construction cost",
                "hard construction costs",
                "total hard cost",
                "total hard costs",
                "hard costs",
            ),
            description="Hard (physical) construction cost, in US dollars.",
        ),
        FieldSpec(
            name="funding_source",
            kind="text",
            labels=(
                "funding source",
                "funding program",
                "program type",
                "request for applications",
                "rfa",
            ),
            description="Programme or RFA the funding comes from, e.g. 'RFA 2023-102' or 'SAIL'.",
            llm_policy="always",
        ),
        FieldSpec(
            name="funding_amount",
            kind="money",
            labels=(
                "funding amount",
                "requested amount",
                "amount requested",
                "request amount",
                "loan amount",
            ),
            description="Amount of agency funding requested or awarded, in US dollars.",
            required=False,
        ),
        FieldSpec(
            name="leverage_ratio",
            kind="ratio",
            labels=("leverage ratio", "leverage"),
            description=(
                "Leverage ratio as a plain decimal (e.g. 1.45). In this project it is defined as "
                "total development cost divided by agency funding amount."
            ),
        ),
    )
}

OUTPUT_FIELDS: tuple[str, ...] = tuple(FIELD_SPECS)


@dataclass
class ExtractedValue:
    """One candidate value for a field, from one extraction method.

    ``value`` is already normalised (``float`` for money/ratio, ``str`` otherwise). ``raw`` keeps
    the text exactly as found so the audit log can show what the number was parsed from.
    """

    field: str
    value: Any
    raw: str
    method: Method
    page: Optional[int] = None
    bbox: Optional[BBox] = None
    evidence: Optional[str] = None
    evidence_verified: Optional[bool] = None
    notes: list[str] = field(default_factory=list)


@dataclass
class FieldResult:
    """Final decision for one field after confidence scoring."""

    field: str
    value: Any
    confidence: float
    status: Status
    reasons: list[str] = field(default_factory=list)
    chosen: Optional[ExtractedValue] = None
    candidates: list[ExtractedValue] = field(default_factory=list)
    previous_value: Any = None  # set when a human edits the value in the review UI


@dataclass
class DocumentResult:
    """All field results for one source PDF (one record per document in v2)."""

    source_file: str
    fields: dict[str, FieldResult]
    warnings: list[str] = field(default_factory=list)

    @property
    def required_results(self) -> list[FieldResult]:
        """Results for the fields that count towards record confidence."""
        return [r for name, r in self.fields.items() if FIELD_SPECS[name].required]

    @property
    def needs_review(self) -> bool:
        """True when any required field is missing or below the review threshold."""
        return any(r.status != "ok" for r in self.required_results)

    @property
    def record_confidence(self) -> float:
        """Mean confidence over required fields (0 when there are none)."""
        results = self.required_results
        if not results:
            return 0.0
        return sum(r.confidence for r in results) / len(results)

    @property
    def data_confidence_label(self) -> str:
        """Human-facing label in the style of the v1 target table: '98%' or 'Review Needed'."""
        if self.needs_review:
            return "Review Needed"
        return f"{round(self.record_confidence * 100)}%"

    def values(self) -> dict[str, Any]:
        """Field name to final value (``None`` when missing)."""
        return {name: r.value for name, r in self.fields.items()}


class LLMField(BaseModel):
    """One field as returned by the LLM. Every part is optional because the LLM may not know."""

    value: Optional[str] = None
    evidence: Optional[str] = Field(
        default=None, description="Short verbatim quote from the document supporting the value."
    )
    page: Optional[int] = Field(default=None, ge=1)

    @field_validator("value", "evidence", mode="before")
    @classmethod
    def _coerce_to_str(cls, v: Any) -> Optional[str]:
        """Accept numbers from the LLM but store them as text; blank strings become None."""
        if v is None:
            return None
        text = str(v).strip()
        if text.lower() in {"", "null", "none", "n/a", "unknown"}:
            return None
        return text


class LLMExtraction(BaseModel):
    """Validated LLM response: a mapping of known field names to ``LLMField``."""

    fields: dict[str, LLMField] = Field(default_factory=dict)

    @field_validator("fields", mode="before")
    @classmethod
    def _drop_unknown_fields(cls, v: Any) -> Any:
        """Ignore any keys the LLM invents that are not in the schema."""
        if isinstance(v, dict):
            return {k: val for k, val in v.items() if k in FIELD_SPECS}
        return v
