"""Scores the research graph against hand-written ground truth.

Each question lists strings that a correct answer has to contain. Accuracy is
the fraction of those strings present across all questions, which is coarse but
catches the failure we actually care about: the report inventing or dropping a
concrete fact.

Spend is metered on every model call and the run aborts at MAX_SPEND_USD.

    python -m eval.run_eval
    python -m eval.run_eval --only slo-latency canary
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from app.config import get_settings
from app.db import init_db
from app.agents import build_graph
from app.llm import BudgetExceeded, meter

QUESTIONS = Path(__file__).with_name("questions.json")


def matches(answer: str, needle: str) -> bool:
    a = answer.lower()
    n = needle.lower()
    if n in a:
        return True
    # numbers get written with or without separators
    return n.replace(",", "") in a.replace(",", "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="run a subset by id")
    ap.add_argument("--out", default="eval/results.json")
    args = ap.parse_args()

    cases = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    if args.only:
        cases = [c for c in cases if c["id"] in set(args.only)]

    # The planner reads session memory, so the tables have to exist even though
    # the eval doesn't write any reports of its own.
    init_db()

    cap = get_settings().max_spend_usd
    if cap:
        print(f"spend cap: ${cap:.2f}")
    print()

    graph = build_graph()
    results, durations = [], []
    hits = total = 0
    stopped_early = None

    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['id']}", flush=True)
        started = time.perf_counter()
        try:
            state = graph.invoke({"session_id": 0, "question": case["question"]})
        except BudgetExceeded as exc:
            stopped_early = str(exc)
            print(f"  stopped: {exc}")
            break

        elapsed = time.perf_counter() - started
        durations.append(elapsed)

        answer = state.get("draft", "")
        found = [f for f in case["must_include"] if matches(answer, f)]
        missed = [f for f in case["must_include"] if f not in found]
        hits += len(found)
        total += len(case["must_include"])

        results.append(
            {
                "id": case["id"],
                "seconds": round(elapsed, 1),
                "self_reported_confidence": state.get("confidence"),
                "revisions": state.get("revisions", 0),
                "sources": len(state.get("evidence", [])),
                "missed": missed,
                "answer": answer,
            }
        )

        spent = meter.snapshot()["cost_usd"]
        print(f"  {elapsed:.0f}s  ${spent:.2f} spent so far", flush=True)
        if missed:
            print(f"  missed: {missed}", flush=True)

    summary = {
        "questions": len(results),
        "facts_checked": total,
        "facts_found": hits,
        "accuracy": round(hits / total, 4) if total else 0.0,
        "median_seconds": round(statistics.median(durations), 1) if durations else 0,
        "mean_confidence": round(
            statistics.mean(r["self_reported_confidence"] or 0 for r in results), 3
        )
        if results
        else 0,
        "spend": meter.snapshot(),
        "stopped_early": stopped_early,
    }

    Path(args.out).write_text(
        json.dumps({"summary": summary, "results": results}, indent=2), encoding="utf-8"
    )

    print()
    for key, value in summary.items():
        print(f"{key:20} {value}")
    print(f"\nwrote {args.out}")
    return 1 if stopped_early else 0


if __name__ == "__main__":
    sys.exit(main())
