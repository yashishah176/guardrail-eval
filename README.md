# Guardrail Eval

[![CI](https://github.com/yashishah176/guardrail-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/yashishah176/guardrail-eval/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

An evaluation harness for **GenAI chatbot guardrails**, built for the **TCS hackathon**. It checks a chatbot's response for five kinds of failure, explains *why* it was flagged, suggests a safe rewrite, and measures the guardrail itself with precision / recall / F1, a confusion matrix and run-over-run regression diffs.

![Single-case evaluator](docs/images/single-case.png)

## What it detects

| Category | Examples |
|---|---|
| `confidential_leakage` | SSNs, card numbers, API keys (`sk-…`, `AKIA…`), passwords, email/password pairs |
| `unsafe_content` | weapon instructions, self-harm encouragement, exploit walkthroughs |
| `bias` | group generalisations and stereotypes |
| `hallucination` | unsourced "according to a study…" and fabricated attributions |
| `unsupported_claims` | "guaranteed", "clinically proven", "always works" |

Each result has a verdict, category, severity band (Low → Critical), a 0–100 risk score, the exact flagged phrase, a plain-English explanation and a redacted safe rewrite.

## Measuring the guardrail

Writing rules is easy; knowing whether they work is the hard part. So the project scores the evaluator against hand-labelled datasets:

- **Suite dashboard**: per-category precision / recall / F1, confusion matrix, severity-calibration rate and phrase-localisation accuracy.
- **Regression diff**: compares the last two runs and lists newly fixed / newly broken case IDs.
- **Determinism check**: runs one case N times to prove the score never drifts, which becomes a real test once the engine is swapped for an LLM judge.

### Results

| Dataset | Cases | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Starter (rules were written against this) | 20 | 100% | 100% | 100% | 100% |
| **Held-out** (new phrasings, never used for tuning) | 20 | **75%** | **100%** | **61.5%** | **76.2%** |

The held-out set is the honest number. The rules **never produced a false positive**, including on hard negatives like "bomb calorimeter" and "a typical sprint is two weeks". But they **missed 5 of 13 rephrased failures** ("pwd is …", an Aadhaar number, "end your life", "older candidates can't learn…", "scientists at Stanford proved…"). That is the core limitation of keyword rules: high precision, brittle recall. It's the reason the next step is an LLM-as-judge behind the same interface.

![Held-out suite results](docs/images/suite-holdout.png)

## Architecture

```
frontend/index.html  ──fetch──▶  FastAPI (backend/main.py)
                                   ├─ POST /api/evaluate          single case
                                   ├─ GET  /api/dataset?name=     starter | holdout
                                   ├─ POST /api/eval/run?dataset= score a whole suite
                                   ├─ GET  /api/eval/runs/diff    regression diff
                                   └─ POST /api/determinism       N-run consistency
                                          │
                                          ▼
                                 evaluate_case(prompt, response)   backend/evaluator.py
```

`evaluate_case()` is the only thing the API depends on, so the rule engine can be replaced by an LLM judge without touching the API, metrics or dashboard.

## Run it

```bash
git clone https://github.com/yashishah176/guardrail-eval.git
cd guardrail-eval
pip install -r requirements-dev.txt
cd backend && uvicorn main:app --reload --port 8000
```

Then open `frontend/index.html` in a browser. The dashboard talks to `http://localhost:8000`.

## Tests

```bash
pytest -v
```

23 tests cover every category, hard negatives, severity bands, determinism, full redaction of multiple secrets, both datasets and every API endpoint.

## Roadmap

- **LLM-as-judge mode** (OpenAI API) behind `evaluate_case()`, compared against the rule engine on the held-out set.
- Grow the held-out set to 100+ cases, including Indian PII formats (Aadhaar, PAN) and multilingual prompts.
- Persist runs in SQLite so regression history survives restarts.

## License

MIT
