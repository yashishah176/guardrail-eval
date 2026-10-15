"""
GenAI Guardrail Eval -- prototype backend.

Run:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

Endpoints:
    POST /api/evaluate            single (prompt, response) -> full eval result   [sync]
    GET  /api/dataset             list the loaded labeled dataset
    POST /api/eval/run            run the whole dataset through the evaluator,
                                   score precision/recall/F1 + confusion matrix,
                                   save the run so it shows up in /api/eval/runs
    GET  /api/eval/runs           list saved runs (most recent first)
    GET  /api/eval/runs/diff      delta between the two most recent runs
    POST /api/determinism         run ONE case N times, prove score/verdict never drifts
"""
import json
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from evaluator import evaluate_case, CATEGORY_ORDER

BASE_DIR = Path(__file__).parent
DATASETS = {
    "starter": BASE_DIR / "data" / "starter_dataset.json",
    "holdout": BASE_DIR / "data" / "holdout_dataset.json",
}

app = FastAPI(title="GenAI Guardrail Eval Prototype")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# in-memory run history for this prototype (swap for a real DB table later --
# see eval_runs / eval_case_results in the design doc)
RUN_HISTORY: list[dict] = []


def load_dataset(name: str = "starter") -> list[dict]:
    if name not in DATASETS:
        raise HTTPException(404, f"Unknown dataset '{name}'. Options: {sorted(DATASETS)}")
    with open(DATASETS[name], encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class EvaluateRequest(BaseModel):
    prompt: str
    response: str


class DeterminismRequest(BaseModel):
    prompt: str
    response: str
    runs: int = 5


# ---------------------------------------------------------------------------
# Single-case sync evaluation
# ---------------------------------------------------------------------------

@app.post("/api/evaluate")
def evaluate(req: EvaluateRequest):
    result = evaluate_case(req.prompt, req.response)
    return result.to_dict()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@app.get("/api/dataset")
def get_dataset(name: str = "starter"):
    cases = load_dataset(name)
    by_category = {}
    for c in cases:
        cat = c.get("expected_category") or "pass/no-issue"
        by_category[cat] = by_category.get(cat, 0) + 1
    return {"count": len(cases), "by_category": by_category, "cases": cases}


# ---------------------------------------------------------------------------
# Suite run: precision / recall / F1 / confusion matrix
# ---------------------------------------------------------------------------

def _score_suite(cases: list[dict]) -> dict:
    per_case_results = []
    confusion: dict[str, dict[str, int]] = {}
    labels = CATEGORY_ORDER + ["none"]
    for a in labels:
        confusion[a] = {p: 0 for p in labels}

    tp = {c: 0 for c in CATEGORY_ORDER}
    fp = {c: 0 for c in CATEGORY_ORDER}
    fn = {c: 0 for c in CATEGORY_ORDER}

    score_in_range_count = 0
    phrase_localization_correct = 0
    phrase_localization_total = 0

    for case in cases:
        result = evaluate_case(case["prompt"], case["response"])
        predicted_category = result.category or "none"
        expected_category = case.get("expected_category") or "none"

        confusion[expected_category][predicted_category] += 1

        is_correct = (
            result.verdict == case["expected_verdict"]
            and predicted_category == expected_category
        )

        if expected_category != "none":
            if predicted_category == expected_category:
                tp[expected_category] += 1
            else:
                fn[expected_category] += 1
                if predicted_category != "none":
                    fp[predicted_category] += 1
        else:
            if predicted_category != "none":
                fp[predicted_category] += 1

        lo, hi = case.get("expected_score_range", [0, 0])
        score_ok = lo <= result.score <= hi
        if score_ok:
            score_in_range_count += 1

        expected_phrase = case.get("expected_flagged_phrase")
        if expected_phrase:
            phrase_localization_total += 1
            if result.flagged_phrase and (
                expected_phrase.lower() in result.flagged_phrase.lower()
                or result.flagged_phrase.lower() in expected_phrase.lower()
            ):
                phrase_localization_correct += 1

        per_case_results.append({
            "id": case["id"],
            "prompt": case["prompt"],
            "response": case["response"],
            "expected_verdict": case["expected_verdict"],
            "expected_category": expected_category,
            "predicted_verdict": result.verdict,
            "predicted_category": predicted_category,
            "predicted_score": result.score,
            "predicted_flagged_phrase": result.flagged_phrase,
            "predicted_explanation": result.explanation,
            "score_in_expected_range": score_ok,
            "is_correct": is_correct,
        })

    per_category_metrics = {}
    total_tp = total_fp = total_fn = 0
    for cat in CATEGORY_ORDER:
        p = tp[cat] / (tp[cat] + fp[cat]) if (tp[cat] + fp[cat]) else None
        r = tp[cat] / (tp[cat] + fn[cat]) if (tp[cat] + fn[cat]) else None
        f1 = (2 * p * r / (p + r)) if (p and r and (p + r) > 0) else None
        per_category_metrics[cat] = {
            "precision": round(p, 3) if p is not None else None,
            "recall": round(r, 3) if r is not None else None,
            "f1": round(f1, 3) if f1 is not None else None,
            "support": tp[cat] + fn[cat],
        }
        total_tp += tp[cat]
        total_fp += fp[cat]
        total_fn += fn[cat]

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    overall_f1 = (
        2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        if (overall_precision + overall_recall) > 0 else 0
    )

    accuracy = sum(1 for r in per_case_results if r["is_correct"]) / len(per_case_results)
    calibration_rate = score_in_range_count / len(per_case_results)
    phrase_localization_accuracy = (
        phrase_localization_correct / phrase_localization_total
        if phrase_localization_total else None
    )

    return {
        "total_cases": len(cases),
        "accuracy": round(accuracy, 3),
        "overall": {
            "precision": round(overall_precision, 3),
            "recall": round(overall_recall, 3),
            "f1": round(overall_f1, 3),
        },
        "per_category": per_category_metrics,
        "confusion_matrix": confusion,
        "calibration": {
            "scores_in_expected_range_rate": round(calibration_rate, 3),
        },
        "phrase_localization_accuracy": (
            round(phrase_localization_accuracy, 3)
            if phrase_localization_accuracy is not None else None
        ),
        "per_case_results": per_case_results,
    }


@app.post("/api/eval/run")
def run_suite(dataset: str = "starter"):
    cases = load_dataset(dataset)
    metrics = _score_suite(cases)
    run_record = {
        "run_id": str(uuid.uuid4())[:8],
        "suite_name": dataset,
        "timestamp": time.time(),
        "metrics": metrics,
    }
    RUN_HISTORY.append(run_record)
    return run_record


@app.get("/api/eval/runs")
def list_runs():
    return {"runs": [
        {
            "run_id": r["run_id"],
            "timestamp": r["timestamp"],
            "accuracy": r["metrics"]["accuracy"],
            "overall_f1": r["metrics"]["overall"]["f1"],
        }
        for r in reversed(RUN_HISTORY)
    ]}


@app.get("/api/eval/runs/diff")
def diff_runs():
    if len(RUN_HISTORY) < 2:
        raise HTTPException(400, "Need at least 2 runs to diff. Run the suite twice.")
    prev, curr = RUN_HISTORY[-2], RUN_HISTORY[-1]
    if prev["suite_name"] != curr["suite_name"]:
        raise HTTPException(400, "The last two runs used different datasets. Run the same dataset twice to diff.")

    def delta(a, b):
        if a is None or b is None:
            return None
        return round(b - a, 3)

    per_category_delta = {}
    for cat in CATEGORY_ORDER:
        pm, cm = prev["metrics"]["per_category"][cat], curr["metrics"]["per_category"][cat]
        per_category_delta[cat] = {
            "precision_delta": delta(pm["precision"], cm["precision"]),
            "recall_delta": delta(pm["recall"], cm["recall"]),
            "f1_delta": delta(pm["f1"], cm["f1"]),
        }

    prev_ids = {c["id"]: c["is_correct"] for c in prev["metrics"]["per_case_results"]}
    curr_ids = {c["id"]: c["is_correct"] for c in curr["metrics"]["per_case_results"]}
    newly_fixed = [cid for cid in curr_ids if prev_ids.get(cid) is False and curr_ids[cid] is True]
    newly_broken = [cid for cid in curr_ids if prev_ids.get(cid) is True and curr_ids[cid] is False]

    return {
        "prev_run_id": prev["run_id"],
        "curr_run_id": curr["run_id"],
        "accuracy_delta": delta(prev["metrics"]["accuracy"], curr["metrics"]["accuracy"]),
        "overall_f1_delta": delta(prev["metrics"]["overall"]["f1"], curr["metrics"]["overall"]["f1"]),
        "overall_recall_delta": delta(prev["metrics"]["overall"]["recall"], curr["metrics"]["overall"]["recall"]),
        "per_category_delta": per_category_delta,
        "newly_fixed_case_ids": newly_fixed,
        "newly_broken_case_ids": newly_broken,
    }


# ---------------------------------------------------------------------------
# Determinism check -- "is our evaluator hallucinating on rerun?"
# Runs the exact same case N times and confirms score/verdict never drift.
# ---------------------------------------------------------------------------

@app.post("/api/determinism")
def determinism_check(req: DeterminismRequest):
    runs = []
    for i in range(max(1, min(req.runs, 20))):
        result = evaluate_case(req.prompt, req.response)
        runs.append({"run_number": i + 1, **result.to_dict()})

    scores = {r["score"] for r in runs}
    verdicts = {r["verdict"] for r in runs}
    phrases = {r["flagged_phrase"] for r in runs}

    is_consistent = len(scores) == 1 and len(verdicts) == 1 and len(phrases) == 1

    return {
        "runs": runs,
        "is_consistent": is_consistent,
        "unique_scores": list(scores),
        "unique_verdicts": list(verdicts),
        "variance": 0 if is_consistent else "INCONSISTENT -- investigate non-determinism",
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}
