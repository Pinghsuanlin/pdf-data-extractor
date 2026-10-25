.PHONY: install test lint format typecheck benchmark compare compare-rules holdout holdout-rules extract evaluate app samples clean

# uv runs every command in the project's own environment and keeps it in sync with pyproject.toml.
# CI overrides this with PY=python.
PY ?= uv run python

install:            ## everything: core + dev + Streamlit app + OCR
	uv sync --extra dev --extra app --extra ocr

test:               ## offline test suite (no network, no LLM, no OCR needed)
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests app.py

format:
	$(PY) -m ruff format src tests app.py
	$(PY) -m ruff check --fix src tests app.py

typecheck:
	$(PY) -m mypy src

benchmark:          ## 40 development PDFs (seed 7) + 40 held-out PDFs (seed 23), 30% scanned, with gold labels
	$(PY) -m pdf_extractor synth --out eval/synthetic --n 40 --seed 7
	$(PY) -m pdf_extractor synth --out eval/holdout --n 40 --seed 23

compare-rules:      ## rules + OCR only; no API key needed
	$(PY) -m pdf_extractor compare --runs rules

compare:            ## rules vs LLM on text vs LLM on page images (uses LLM_BACKEND from .env)
	$(PY) -m pdf_extractor compare --runs rules,llm,vision

holdout-rules:      ## same as compare-rules, on the held-out set
	$(PY) -m pdf_extractor compare --runs rules --pdfs eval/holdout/pdfs --gold eval/holdout/gold --out eval/results/holdout

holdout:            ## same as compare, on the held-out set (report these numbers)
	$(PY) -m pdf_extractor compare --runs rules,llm,vision --pdfs eval/holdout/pdfs --gold eval/holdout/gold --out eval/results/holdout

extract:            ## extract every PDF in data/pdfs to outputs/
	$(PY) -m pdf_extractor extract data/pdfs --out outputs

evaluate:           ## score your own hand-labelled PDFs (eval/gold) against data/pdfs
	$(PY) -m pdf_extractor evaluate --pdfs data/pdfs --gold eval/gold --out eval/results/real

app:
	$(PY) -m streamlit run app.py

samples:            ## README images: one native and one heavy-scan page from the benchmark
	$(PY) scripts/make_samples.py

clean:
	rm -rf outputs eval/results eval/synthetic/pdfs eval/holdout/pdfs .pytest_cache .mypy_cache .ruff_cache
