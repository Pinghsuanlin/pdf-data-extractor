# Running it on your Mac

About 30 minutes end to end, most of it waiting on the free Gemini tier. Run every command from
the project folder.

```bash
cd ~/Documents/CoworkOS/career/portfolio/pdf-data-extractor
```

## 1. One-off setup (5 min)

```bash
brew install uv tesseract     # uv = Python env manager; tesseract = OCR for scanned pages
make install                  # creates .venv and uv.lock with every extra
```

`make install` writes a `uv.lock` file. Commit it: it pins exact versions so CI and anyone cloning
the repo get the same environment.

## 2. Check it works (1 min)

```bash
make test
make lint
```

Expect every test to pass and no lint errors. If anything fails, paste the output to Claude; it was
built in a sandbox without the real `pydantic` and `ruff`, so a small fix on first run is possible.

## 3. Build the benchmark (under 1 min)

```bash
make benchmark
```

Writes two sets of 40 PDFs: the development set in `eval/synthetic/pdfs/` and the held-out set in
`eval/holdout/pdfs/`. Open two in Preview to see what the system faces, for example
`eval/synthetic/pdfs/synth_000_form.pdf` and any `*_scan_heavy.pdf`. Try selecting text on the
scan: there is none, it is a picture.

Why two sets: you look at the development set while checking and tuning; the held-out set is only
scored at the end, so its numbers show how the system does on documents it was never adjusted to.
Report the held-out numbers.

## 4. Baseline: rules + OCR, no key needed (1 min)

```bash
make compare-rules     # development set
make holdout-rules     # held-out set
```

Expected, roughly (the scanned rows can move a few points with your Tesseract version):

| Metric | Development | Held-out |
|:---|---:|---:|
| Accuracy, all documents | about 57% | about 62% |
| Accuracy, native PDFs | about 68% | about 73% |
| Accuracy, scanned PDFs | about 31% | about 36% |
| Accuracy, narrative layout | about 7% | about 7% |
| Error rate of values not flagged | 0% | 0% |
| Review rate | about 19% | about 23% |

That is the story in one table: rules are almost never wrong when they let a value through, but
they miss a lot, and scans cut accuracy by more than half. Every value read from a scan goes to
review on its own; one of them is a real OCR misread worth showing in an interview:
`synth_031_kv_table_scan_light.pdf` reads the id `2024-210S` as `2024-2108`, and it is caught only
because OCR values need a second reader.

## 5. Get a free Gemini key (3 min)

1. Go to https://aistudio.google.com/apikey and sign in with a personal Google account.
2. Click **Create API key** and copy it.
3. Create your `.env` and fill it in:

```bash
cp .env.example .env
open -e .env
```

Set `LLM_BACKEND=gemini` and paste the key after `GEMINI_API_KEY=`. Save.

- `.env` is in `.gitignore`; never commit it or paste the key anywhere else.
- Free-tier prompts may be used by Google to improve its products. Fine for the synthetic
  benchmark; do not send confidential documents through the free tier.

## 6. The full comparison (20-30 min)

```bash
make compare     # development set
make holdout     # held-out set: the numbers for the README
```

Each runs three configurations on the same 40 documents: rules only, rules + Gemini reading text,
and rules + Gemini reading page images for the scans. About 50 API calls per set (100 in total),
spaced 13 seconds apart for the free tier. If you hit the daily free limit, run `make holdout` the
next day; the cache keeps everything already done. Native documents send the same prompt in the text and image runs, so the second
run reuses the cached answer and only the 12 scans make new calls.

If it stops part-way, just run `make compare` again. Every finished call is cached in
`.cache/llm/`, so nothing is paid for twice.

Output: `eval/results/compare/comparison.md` and `eval/results/holdout/comparison.md` (the summary
tables) plus, per run, `report.md`
(breakdown by layout, scan quality and trap), `review_queue.csv` and `audit_log.csv`.

## 7. Read the results before you publish (10 min)

Check these three things, in this order:

1. **Scanned row, image run vs text run.** The point of the project. If reading images doesn't beat
   reading OCR text on scans, say so in the README rather than hiding it.
2. **Error rate of values NOT flagged.** Should stay low (a few percent). If the LLM runs push it
   up, open the run's `audit_log.csv`, filter `status = ok`, and compare with the gold files to
   see which trap caused it (the per-unit distractor is the usual suspect).
3. **Review rate.** The cost side. A run that is accurate only because it sends half the values to
   a person is a weaker result; the README's "How to read it" explains the trade-off.

Do this on the development set (`eval/results/compare/`). If you change anything because of what
you see, rerun both, and still report the held-out set. Then paste
`eval/results/holdout/comparison.md` into the README's Results table and replace "LLM backend and model:
TBD" with the model you used (from `.env`).

## 8. Try the review screen (optional, 5 min)

```bash
make app
```

Opens http://localhost:8501. Upload a `*_scan_heavy.pdf` from `eval/synthetic/pdfs/`, pick
`gemini`, and look at the flagged fields with their reasons. A screenshot of this screen is a good
second image for the README.

## 9. Publish

1. Commit `uv.lock` along with everything else (step 1 created it).
2. In GitHub Desktop: **File → Add local repository**, choose this folder, and click **create a
   repository** when it says the folder isn't one yet. (Your CoworkOS repo already ignores
   `career/`, so the two repos don't clash.) Commit, then **Publish repository** to the new repo,
   unticking "Keep this code private".
3. Before publishing, check the changed-files list has no `.env`, `.venv`, `.cache` or
   `eval/synthetic/pdfs` / `eval/holdout/pdfs`. The `.gitignore` excludes them; this is just a check.
4. On GitHub, open the **Actions** tab and check the CI run is green.
5. Archive the old repo: `PDF_scraping_app` → **Settings → Archive this repository** (or make it
   private), so nobody lands on the prototype.

## Troubleshooting

| Message | Fix |
|:---|:---|
| `HTTP 404` from generativelanguage | Model id changed. In AI Studio, copy the current "Flash" model id into `GEMINI_MODEL` in `.env` |
| `Stopped: ... HTTP 429 ... quota=...PerMinute...` | Too many calls per minute. Set `LLM_MIN_INTERVAL_S=12` in `.env` and rerun; cached calls are skipped |
| `Stopped: ... HTTP 429 ... quota=...PerDay...` or `limit=0` | The model's free daily quota is used up (or is zero for your account). Check https://ai.dev/rate-limit, then either wait until tomorrow and rerun, switch `GEMINI_MODEL` to a model with free quota left (e.g. a Flash-Lite model), or enable billing with a small budget cap (about 100 calls costs cents). Note the model you used in the README |
| `HTTP 400` / `403` | Key wrong or not enabled. Recreate it in AI Studio |
| `tesseract is not installed` | `brew install tesseract`, then rerun |
| `uv: command not found` | `brew install uv`, then open a new terminal |
| Scanned rows show 0% for rules | OCR did not run; see the Tesseract line above |
