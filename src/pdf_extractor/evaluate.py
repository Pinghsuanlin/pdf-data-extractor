"""Evaluation harness against hand-labelled gold files.

Gold files live in ``eval/gold/*.json``, one per PDF (see the template there). A value of
``"FILL_IN"`` means "not labelled yet" and is skipped; ``null`` means "the document genuinely does
not state this field", so predicting nothing is correct and predicting a value is an error.

Metrics:

* **accuracy** per field and overall: exact match for text and ids (case and whitespace
  insensitive), relative tolerance for money and ratios.
* **coverage**: share of fields present in the gold file for which the pipeline returned a value.
* **flag quality**: is the low-confidence flag useful? ``flag_precision`` is the share of flagged
  values that are wrong; compare it with ``unflagged_error_rate``. ``flag_recall`` is the share of
  wrong values that were flagged. A useful flag has precision well above the unflagged error rate.
  ``review_rate`` is the share of returned values sent to a person: the cost side of the flag.
* **segments**: when gold files carry ``tags`` (the synthetic benchmark does), accuracy and
  coverage are also reported per segment, e.g. ``scanned=yes`` vs ``scanned=no``, per layout and
  per trap, so you can see where each method breaks rather than one blended number.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from pdf_extractor.confidence import values_agree
from pdf_extractor.rules import normalise_value
from pdf_extractor.schemas import FIELD_SPECS, DocumentResult

logger = logging.getLogger(__name__)

FILL_IN = "FILL_IN"
DEFAULT_REL_TOL = 0.01


@dataclass
class GoldDocument:
    """Labels for one PDF. Only labelled fields appear in ``fields``; ``tags`` is optional."""

    source_file: str
    fields: dict[str, Any]
    tags: Optional[dict[str, Any]] = None


def segments_for(tags: Optional[dict[str, Any]]) -> list[str]:
    """Segment names a document belongs to, e.g. ['scanned=yes', 'scan=heavy', 'layout=form']."""
    if not tags:
        return []
    out = [f"scanned={'yes' if tags.get('scanned') else 'no'}"]
    if tags.get("scan_quality"):
        out.append(f"scan={tags['scan_quality']}")
    if tags.get("layout"):
        out.append(f"layout={tags['layout']}")
    out += [f"trap={t}" for t in tags.get("traps") or []]
    return out


def _is_unlabelled(value: Any) -> bool:
    return isinstance(value, str) and value.strip().upper() == FILL_IN


def load_gold(gold_dir: str | Path) -> list[GoldDocument]:
    """Load gold files, skipping unlabelled values and files with no labelled fields.

    String values for money and ratio fields are normalised (so '$14,500,000' is accepted).
    """
    docs: list[GoldDocument] = []
    for path in sorted(Path(gold_dir).glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        source = data.get("source_file")
        if not source or _is_unlabelled(source) or FILL_IN in str(source):
            logger.info("Skipping %s: source_file not filled in.", path.name)
            continue
        fields: dict[str, Any] = {}
        for name, value in (data.get("fields") or {}).items():
            if name not in FIELD_SPECS or _is_unlabelled(value):
                continue
            if isinstance(value, str):
                # Gold goes through the same normaliser as predictions, so 'Ltd.' vs 'Ltd' or
                # '$14.5M' vs 14500000 are not counted as errors.
                value = normalise_value(FIELD_SPECS[name].kind, value)
            fields[name] = value
        if fields:
            docs.append(GoldDocument(source_file=source, fields=fields, tags=data.get("tags")))
    return docs


def field_correct(name: str, predicted: Any, gold: Any, rel_tol: float = DEFAULT_REL_TOL) -> bool:
    """True when the prediction matches gold. ``None`` vs ``None`` counts as correct."""
    if gold is None:
        return predicted is None
    return values_agree(FIELD_SPECS[name].kind, predicted, gold, rel_tol)


def _new_bucket() -> dict[str, int]:
    keys = ("docs", "labelled", "correct", "gold_present", "predicted_when_present", "returned", "flagged")
    return dict.fromkeys(keys, 0)


def _ratio(num: int, den: int) -> Optional[float]:
    return round(num / den, 4) if den else None


def evaluate(
    results: list[DocumentResult], gold: list[GoldDocument], rel_tol: float = DEFAULT_REL_TOL
) -> dict[str, Any]:
    """Compare predictions with gold labels and return a metrics dictionary."""
    by_file = {r.source_file: r for r in results}
    per_field: dict[str, dict[str, int]] = {
        n: {"labelled": 0, "correct": 0, "gold_present": 0, "predicted_when_present": 0}
        for n in FIELD_SPECS
    }
    flagged = flagged_wrong = unflagged = unflagged_wrong = 0
    missing_docs: list[str] = []
    seg: dict[str, dict[str, int]] = {}

    for g in gold:
        doc = by_file.get(g.source_file)
        if doc is None:
            missing_docs.append(g.source_file)
            continue
        buckets = [seg.setdefault(name, _new_bucket()) for name in segments_for(g.tags)]
        for bucket in buckets:
            bucket["docs"] += 1
        for name, gold_value in g.fields.items():
            res = doc.fields[name]
            correct = field_correct(name, res.value, gold_value, rel_tol)
            for stats in (per_field[name], *buckets):
                stats["labelled"] += 1
                stats["correct"] += int(correct)
                if gold_value is not None:
                    stats["gold_present"] += 1
                    stats["predicted_when_present"] += int(res.value is not None)
            if res.value is None:
                continue  # flag quality is about values we returned
            for bucket in buckets:
                bucket["returned"] += 1
                bucket["flagged"] += int(res.status == "review")
            if res.status == "review":
                flagged += 1
                flagged_wrong += int(not correct)
            else:
                unflagged += 1
                unflagged_wrong += int(not correct)

    labelled = sum(s["labelled"] for s in per_field.values())
    correct = sum(s["correct"] for s in per_field.values())
    present = sum(s["gold_present"] for s in per_field.values())
    predicted = sum(s["predicted_when_present"] for s in per_field.values())
    return {
        "n_documents": len(gold) - len(missing_docs),
        "documents_without_prediction": missing_docs,
        "n_labelled_fields": labelled,
        "accuracy": _ratio(correct, labelled),
        "coverage": _ratio(predicted, present),
        "flag_precision": _ratio(flagged_wrong, flagged),
        "unflagged_error_rate": _ratio(unflagged_wrong, unflagged),
        "flag_recall": _ratio(flagged_wrong, flagged_wrong + unflagged_wrong),
        "review_rate": _ratio(flagged, flagged + unflagged),
        "n_flagged": flagged,
        "n_unflagged": unflagged,
        "segments": {
            name: {
                "documents": s["docs"],
                "labelled": s["labelled"],
                "accuracy": _ratio(s["correct"], s["labelled"]),
                "coverage": _ratio(s["predicted_when_present"], s["gold_present"]),
                "review_rate": _ratio(s["flagged"], s["returned"]),
            }
            for name, s in sorted(seg.items(), key=lambda kv: _segment_order(kv[0]))
        },
        "per_field": {
            name: {
                "labelled": s["labelled"],
                "accuracy": _ratio(s["correct"], s["labelled"]),
                "coverage": _ratio(s["predicted_when_present"], s["gold_present"]),
            }
            for name, s in per_field.items()
            if s["labelled"]
        },
    }


def _segment_order(name: str) -> tuple[int, str]:
    order = ("scanned=", "scan=", "layout=", "trap=")
    return (next((i for i, p in enumerate(order) if name.startswith(p)), len(order)), name)


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def metrics_markdown(metrics: dict[str, Any]) -> str:
    """Render metrics as a Markdown report (paste into the README Results section)."""
    lines = [
        "# Evaluation report",
        "",
        f"Documents evaluated: {metrics['n_documents']}  ",
        f"Labelled fields: {metrics['n_labelled_fields']}",
        "",
        "| Metric | Value |",
        "|:---|---:|",
        f"| Field accuracy | {_fmt(metrics['accuracy'])} |",
        f"| Coverage | {_fmt(metrics['coverage'])} |",
        f"| Flag precision (flagged values that are wrong) | {_fmt(metrics['flag_precision'])} |",
        f"| Error rate of unflagged values | {_fmt(metrics['unflagged_error_rate'])} |",
        f"| Flag recall (wrong values that were flagged) | {_fmt(metrics['flag_recall'])} |",
        f"| Review rate (returned values sent to a person) | {_fmt(metrics.get('review_rate'))} |",
        "",
        "| Field | Labelled | Accuracy | Coverage |",
        "|:---|---:|---:|---:|",
    ]
    for name, s in metrics["per_field"].items():
        lines.append(f"| {name} | {s['labelled']} | {_fmt(s['accuracy'])} | {_fmt(s['coverage'])} |")
    if metrics.get("segments"):
        lines += [
            "",
            "| Segment | Docs | Accuracy | Coverage | Review rate |",
            "|:---|---:|---:|---:|---:|",
        ]
        for name, s in metrics["segments"].items():
            lines.append(
                f"| {name} | {s['documents']} | {_fmt(s['accuracy'])} | {_fmt(s['coverage'])} "
                f"| {_fmt(s['review_rate'])} |"
            )
    if metrics["documents_without_prediction"]:
        lines += ["", "Gold files with no matching PDF: " + ", ".join(metrics["documents_without_prediction"])]
    return "\n".join(lines) + "\n"


def write_report(metrics: dict[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    """Write ``metrics.json`` and ``report.md`` to ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "metrics.json"
    md_path = out / "report.md"
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    md_path.write_text(metrics_markdown(metrics), encoding="utf-8")
    return json_path, md_path
