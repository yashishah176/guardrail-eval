# Guardrail Eval — Prototype

Sync API + dashboard for the GenAI Developer eval loop. Rule-based evaluator engine
(no LLM call) so it's free, offline, and provably deterministic — swap it for a real
LLM-as-judge later behind the same `evaluate_case()` function without touching the API
or frontend.

## Run it

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Then open `frontend/index.html` directly in your browser (double-click it, or
`open frontend/index.html`). It talks to `http://localhost:8000`.

## What's in it

- **`backend/evaluator.py`** — the guardrail engine. Regex/keyword rules for
  `confidential_leakage`, `unsafe_content`, `bias`, `hallucination`, `unsupported_claims`.
  Returns verdict, category, severity, score, flagged phrase, explanation, safe rewrite.
- **`backend/data/starter_dataset.json`** — 19 hand-labeled test cases (incl. hard
  negatives like "women in leadership" that should NOT trigger bias) using the schema
  from the design doc: `id, prompt, response, expected_verdict, expected_category,
  expected_severity, expected_score_range, expected_flagged_phrase,
  expected_explanation_keywords`.
- **`backend/main.py`** — FastAPI app:
  - `POST /api/evaluate` — single case, sync, full result
  - `GET /api/dataset` — the labeled test set
  - `POST /api/eval/run` — runs the whole suite, computes precision/recall/F1 per
    category + overall, confusion matrix, calibration rate, phrase-localization
    accuracy, saves the run
  - `GET /api/eval/runs/diff` — delta between the two most recent runs (newly
    fixed / newly broken case IDs included)
  - `POST /api/determinism` — runs one case N times, confirms score/verdict/phrase
    never drift
- **`frontend/index.html`** — single-file dashboard, three tabs: single-case tester,
  suite dashboard (headline metrics + confusion matrix + per-case table + diff), and
  the determinism proof panel.

## Why rule-based, not an LLM call, for this prototype

Real guardrail products run an LLM as the judge. For this MVP the point was to prove
the *pipeline* — schema, metrics, diffing, dashboard — works end to end, and to give
you a clean, honest answer to "is it hallucinating on rerun?" A regex engine is
100% reproducible by construction, so the determinism tab will always show 0% variance
right now. When you swap in a real LLM call inside `evaluate_case()`, that same tab
becomes the actual test that catches non-determinism if temperature/prompt drift ever
creeps in — don't delete it once you swap the engine, it gets more useful, not less.

## Known gap (left in on purpose)

Suite run currently scores 18/19 correct (accuracy 0.95) — case `bi_002` ("Our typical
customer is a millennial...") isn't caught because the bias regex only matches "typical
X" adjacent to the group word, not "typical [noun] is a X". Left as a real example of
what a confusion-matrix miss looks like on the dashboard, and a good first fix to make
if you want to demo the diff/regression-narration screen: tighten that one regex, rerun
the suite, watch recall move on the diff tab.
