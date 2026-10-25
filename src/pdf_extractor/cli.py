"""Command line entry point.

* ``pdf-extract synth``: generate the synthetic benchmark (PDFs plus gold labels).
* ``pdf-extract extract``: extract records from PDFs to CSV / Excel with an audit log.
* ``pdf-extract evaluate``: score one configuration against gold labels.
* ``pdf-extract compare``: score several configurations on the same documents side by side.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

from pdf_extractor.evaluate import GoldDocument, evaluate, load_gold, metrics_markdown, write_report
from pdf_extractor.export import export_results
from pdf_extractor.llm_extract import CachedBackend, LLMBackend, LLMBackendError, get_backend, supports_vision
from pdf_extractor.pipeline import PipelineConfig, extract_many

RUNS = {
    "rules": "Rules + OCR only (no LLM)",
    "llm": "Rules + LLM reading text (OCR text on scans)",
    "vision": "Rules + LLM reading page images on scans",
}


def _load_dotenv() -> None:
    """Load ``.env`` if python-dotenv is installed; silently skip otherwise."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _pdf_paths(target: Path) -> list[Path]:
    """A single PDF, or every PDF in a directory (sorted)."""
    if target.is_file():
        return [target]
    return sorted(p for p in target.glob("*.pdf"))


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", default=None, help="none | ollama | groq | mistral | gemini")
    parser.add_argument("--llm-policy", default="hard", choices=["off", "hard", "all"])
    parser.add_argument("--ocr", default="auto", choices=["off", "auto", "force"])
    parser.add_argument("--vision", default="scanned", choices=["off", "scanned", "all"],
                        help="pages sent as images when the backend can read them")
    parser.add_argument("--no-cache", action="store_true", help="always call the LLM (no disk cache)")


def _gold_and_paths(pdfs: Path, gold_dir: Path) -> tuple[list[GoldDocument], list[Path]]:
    gold = load_gold(gold_dir)
    paths = [pdfs / g.source_file for g in gold if (pdfs / g.source_file).exists()]
    return gold, paths


def _no_gold_message(gold_dir: Path) -> str:
    return (
        f"No labelled gold files in {gold_dir}. Run `pdf-extract synth` for the synthetic benchmark, "
        "or copy eval/gold/TEMPLATE_application.json, replace every FILL_IN, and point source_file "
        "at a PDF in data/pdfs/."
    )


def run_compare(
    pdfs: Path,
    gold_dir: Path,
    out: Path,
    runs: list[str],
    backend: Optional[LLMBackend],
    base: PipelineConfig,
) -> dict[str, Any]:
    """Evaluate each configuration on the same documents and write ``comparison.md``."""
    gold, paths = _gold_and_paths(pdfs, gold_dir)
    if not gold:
        raise SystemExit(_no_gold_message(gold_dir))
    summary: dict[str, Any] = {}
    for run in runs:
        if run != "rules" and backend is None:
            logging.warning("Skipping run '%s': no LLM backend (set --backend or LLM_BACKEND).", run)
            continue
        if run == "vision" and not supports_vision(backend):
            logging.warning("Skipping run 'vision': backend '%s' cannot read images.", backend.name)
            continue
        config = PipelineConfig(
            llm_policy="off" if run == "rules" else base.llm_policy,
            ocr=base.ocr,
            vision="scanned" if run == "vision" else "off",
            strict_llm=True,
        )
        before = backend.misses if isinstance(backend, CachedBackend) else 0
        start = time.perf_counter()
        results = extract_many(paths, None if run == "rules" else backend, config)
        seconds = time.perf_counter() - start
        metrics = evaluate(results, gold)
        metrics["run"] = {"name": run, "description": RUNS[run], "seconds": round(seconds, 1),
                          "backend": None if run == "rules" else backend.name,
                          "new_api_calls": (backend.misses - before) if isinstance(backend, CachedBackend) else None}
        write_report(metrics, out / run)
        export_results(results, out / run)
        summary[run] = metrics
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "comparison.md").write_text(comparison_markdown(summary), encoding="utf-8")
    return summary


def comparison_markdown(summary: dict[str, Any]) -> str:
    """One table, one column per run: the numbers to paste into the README."""
    def pct(v: Optional[float]) -> str:
        return "n/a" if v is None else f"{v:.0%}"

    def seg(m: dict[str, Any], name: str) -> str:
        return pct((m.get("segments") or {}).get(name, {}).get("accuracy"))

    runs = list(summary)
    rows = [
        ("Field accuracy, all documents", lambda m: pct(m["accuracy"])),
        ("Accuracy, native PDFs", lambda m: seg(m, "scanned=no")),
        ("Accuracy, scanned PDFs", lambda m: seg(m, "scanned=yes")),
        ("Accuracy, heavy scans", lambda m: seg(m, "scan=heavy")),
        ("Accuracy, narrative layout", lambda m: seg(m, "layout=narrative")),
        ("Accuracy, unseen labels", lambda m: seg(m, "trap=unseen_labels")),
        ("Accuracy, per-unit distractor", lambda m: seg(m, "trap=per_unit_distractor")),
        ("Coverage", lambda m: pct(m["coverage"])),
        ("Error rate of values NOT flagged", lambda m: pct(m["unflagged_error_rate"])),
        ("Flag recall (wrong values caught)", lambda m: pct(m["flag_recall"])),
        ("Review rate (values sent to a person)", lambda m: pct(m["review_rate"])),
        ("Run time (s)", lambda m: str(m["run"]["seconds"])),
        ("New API calls (0 = all cached)", lambda m: str(m["run"]["new_api_calls"] or 0)),
    ]
    header = "| Metric | " + " | ".join(runs) + " |"
    lines = ["# Comparison", ""]
    for r in runs:
        info = summary[r]["run"]
        backend = f" (backend: {info['backend']})" if info["backend"] else ""
        lines.append(f"- **{r}**: {info['description']}{backend}")
    lines += ["", header, "|:---|" + "---:|" * len(runs)]
    for label, fn in rows:
        lines.append(f"| {label} | " + " | ".join(fn(summary[r]) for r in runs) + " |")
    n_docs = next(iter(summary.values()))["n_documents"] if summary else 0
    lines += ["", f"Documents: {n_docs}. Per-segment detail is in each run's report.md."]
    return "\n".join(lines) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    """Parse arguments and run the requested command."""
    _load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="pdf-extract", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="Extract records from PDFs")
    p_extract.add_argument("input", type=Path, help="PDF file or folder of PDFs")
    p_extract.add_argument("--out", type=Path, default=Path("outputs"))
    _add_common(p_extract)

    p_eval = sub.add_parser("evaluate", help="Score extraction against gold labels")
    p_eval.add_argument("--pdfs", type=Path, default=Path("data/pdfs"))
    p_eval.add_argument("--gold", type=Path, default=Path("eval/gold"))
    p_eval.add_argument("--out", type=Path, default=Path("eval/results"))
    _add_common(p_eval)

    p_cmp = sub.add_parser("compare", help="Score several configurations on the same documents")
    p_cmp.add_argument("--pdfs", type=Path, default=Path("eval/synthetic/pdfs"))
    p_cmp.add_argument("--gold", type=Path, default=Path("eval/synthetic/gold"))
    p_cmp.add_argument("--out", type=Path, default=Path("eval/results/compare"))
    p_cmp.add_argument("--runs", default="rules,llm,vision", help="comma list of: rules, llm, vision")
    _add_common(p_cmp)

    p_syn = sub.add_parser("synth", help="Generate the synthetic benchmark")
    p_syn.add_argument("--out", type=Path, default=Path("eval/synthetic"))
    p_syn.add_argument("--n", type=int, default=40)
    p_syn.add_argument("--seed", type=int, default=7)
    p_syn.add_argument("--scanned-share", type=float, default=0.3)

    args = parser.parse_args(argv)
    if args.command == "synth":
        from pdf_extractor.synth import generate

        docs = generate(args.out, n=args.n, seed=args.seed, scanned_share=args.scanned_share)
        n_scan = sum(d.scanned for d in docs)
        print(f"wrote {len(docs)} PDFs ({n_scan} scanned) to {args.out}/pdfs and gold labels to {args.out}/gold")
        return 0

    backend = get_backend(args.backend, cache=not args.no_cache)
    config = PipelineConfig(
        llm_policy=args.llm_policy, ocr=args.ocr, vision=args.vision,
        strict_llm=args.command in {"evaluate", "compare"},
    )
    try:
        return _run(args, backend, config)
    except LLMBackendError as exc:
        done = backend.misses if isinstance(backend, CachedBackend) else 0
        print(
            f"\nStopped: the LLM call failed, so the results would not be valid.\n  {exc}\n"
            f"{done} new answers were saved to the cache before this; rerun the same command later "
            "and it resumes from there. HTTP 429 = quota: see RUNNING.md, 'Troubleshooting'.",
            file=sys.stderr,
        )
        return 3


def _run(args: argparse.Namespace, backend: Optional[LLMBackend], config: PipelineConfig) -> int:
    """Dispatch ``extract``, ``evaluate`` and ``compare`` (``synth`` is handled before)."""
    if args.command == "compare":
        runs = [r.strip() for r in args.runs.split(",") if r.strip()]
        unknown = set(runs) - set(RUNS)
        if unknown:
            print(f"Unknown run(s): {', '.join(sorted(unknown))}. Use {', '.join(RUNS)}.", file=sys.stderr)
            return 2
        summary = run_compare(args.pdfs, args.gold, args.out, runs, backend, config)
        print((args.out / "comparison.md").read_text(encoding="utf-8"))
        return 0 if summary else 1

    if args.command == "extract":
        paths = _pdf_paths(args.input)
        if not paths:
            print(f"No PDFs found at {args.input}", file=sys.stderr)
            return 1
        results = extract_many(paths, backend, config)
        for path in export_results(results, args.out):
            print(f"wrote {path}")
        return 0

    gold, paths = _gold_and_paths(args.pdfs, args.gold)
    if not gold:
        print(_no_gold_message(args.gold), file=sys.stderr)
        return 1
    results = extract_many(paths, backend, config)
    metrics = evaluate(results, gold)
    write_report(metrics, args.out)
    export_results(results, args.out)
    print(metrics_markdown(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
