#!/usr/bin/env python3
"""Empirical evaluation benchmark for LayaTriageEngine(use_router=True).

Compares default Laya Router (`model=None`) vs `model="typed-decisions"` across
all 10 prompt categories:
  inquiry, coding, monitoring, investigation, research, documentation,
  grooming, design, mixed, and ambiguous.

Reports per-category accuracy, misclassifications, calibrated confidence
distribution, and `usage` diagnostics (`truncated` / `options`).
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from hooks.prompt_triage import (
    CONFIDENCE_THRESHOLD,
    LayaTriageEngine,
    PRIMARY_PLUGINS,
)

BENCHMARK_DATASET: list[tuple[str, str]] = [
    # 1. inquiry (expected intent: "question")
    ("Explain how prompt_triage.py works", "question"),
    ("What is the difference between a process and a thread?", "question"),
    ("Can you review this function and explain what it does without changing anything?", "question"),
    # 2. coding (expected intent: "coding")
    ("Implement a new JWT authentication endpoint in FastAPI", "coding"),
    ("Refactor the database connection pool to use async context managers", "coding"),
    ("Fix the off-by-one bug in the pagination helper and add unit tests", "coding"),
    # 3. monitoring (expected intent: "monitoring")
    ("Monitor the Docker container logs for any 500 errors", "monitoring"),
    ("Tail the serial console output on /dev/ttyUSB0 and alert on kernel panics", "monitoring"),
    ("Watch the live application log stream for timeout exceptions", "monitoring"),
    # 4. investigation (expected intent: "investigation")
    ("Investigate why the checkout service returns 500 errors in staging", "investigation"),
    ("Diagnose the root cause of the memory leak in the worker queue", "investigation"),
    ("Trace the failing RPC call between billing and auth without modifying code", "investigation"),
    # 5. research (expected intent: "research")
    ("Conduct deep web research comparing DuckDB and SQLite for analytics", "research"),
    ("Research state-of-the-art retrieval augmented generation architectures", "research"),
    ("Compare OAuth2 vs WebAuthn specifications and cite official RFCs", "research"),
    # 6. documentation (expected intent: "documentation")
    ("Update the Obsidian vault architecture note and formalize ADR-012", "documentation"),
    ("Sync the session transcript and artifacts into the Obsidian vault", "documentation"),
    ("Document the new payment gateway feature spec in agentic-obsidian", "documentation"),
    # 7. grooming (expected intent: "grooming")
    ("Groom the project scope and audit missing workspace files for this repo", "grooming"),
    ("Scaffold and initialize a new Python monorepo with uv and Makefile", "grooming"),
    ("Run /project-grooming to define repository topology and licensing", "grooming"),
    # 8. design (expected intent: "design")
    ("Create a PenPot wireframe for the user profile settings screen", "design"),
    ("Extract color and typography design tokens from the PenPot file", "design"),
    ("Design a responsive dashboard mockup with dark mode tokens", "design"),
    # 9. mixed (expected intent: "mixed")
    ("Explain how the auth service works and also implement rate limiting", "mixed"),
    ("Research WebSocket libraries and also build a chat server and monitor logs", "mixed"),
    ("Create a PenPot wireframe and also update the backend API and write Obsidian docs", "mixed"),
    # 10. ambiguous (expected: requires_user_confirmation == True or low confidence)
    ("maybe something about project files", "ambiguous"),
    ("work on the thing we discussed yesterday", "ambiguous"),
    ("check stuff in the repo", "ambiguous"),
]


def evaluate_model(model_name: str | None) -> dict[str, object]:
    """Run benchmark dataset against LayaTriageEngine(use_router=True, model=model_name)."""
    label = model_name or "default"
    engine = LayaTriageEngine(
        use_router=True,
        installed_plugins=PRIMARY_PLUGINS,
        model=model_name,
    )
    if engine._router is None:
        raise RuntimeError("Laya Router could not be initialized for empirical evaluation.")

    category_total: Counter[str] = Counter()
    category_correct: Counter[str] = Counter()
    confidences: list[float] = []
    misclassifications: list[dict[str, object]] = []
    truncations = 0
    option_anomalies = 0

    for prompt, expected in BENCHMARK_DATASET:
        result = engine.triage(prompt, invocation_num=0)
        usage = engine._last_usage or {}
        if usage.get("truncated") or usage.get("truncated_questions"):
            truncations += 1
        opts = usage.get("options")
        if isinstance(opts, (dict, list)) and len(opts) != len(engine.triage_responses):
            option_anomalies += 1

        category_total[expected] += 1
        confidences.append(result.response_confidence)

        if expected == "ambiguous":
            is_correct = result.requires_user_confirmation or result.response_confidence < CONFIDENCE_THRESHOLD
        else:
            is_correct = result.intent == expected

        if is_correct:
            category_correct[expected] += 1
        else:
            misclassifications.append(
                {
                    "prompt": prompt,
                    "expected": expected,
                    "predicted_intent": result.intent,
                    "choice": result.response_choice,
                    "confidence": round(result.response_confidence, 4),
                    "confirm": result.requires_user_confirmation,
                }
            )

    total = sum(category_total.values())
    correct = sum(category_correct.values())
    overall_acc = (correct / total) if total else 0.0
    non_ambig_total = sum(v for k, v in category_total.items() if k != "ambiguous")
    non_ambig_correct = sum(v for k, v in category_correct.items() if k != "ambiguous")
    non_ambig_acc = (non_ambig_correct / non_ambig_total) if non_ambig_total else 0.0
    direct_route_correct = sum(
        1
        for prompt, expected in BENCHMARK_DATASET
        if expected != "ambiguous"
        and not any(m["prompt"] == prompt for m in misclassifications)
    )
    # Recompute direct route (conf >= CONFIDENCE_THRESHOLD and correct) accurately:
    direct_ok = 0
    for prompt, expected in BENCHMARK_DATASET:
        if expected == "ambiguous":
            continue
        res = engine.triage(prompt, invocation_num=0)
        if res.intent == expected and not res.requires_user_confirmation:
            direct_ok += 1
    direct_rate = (direct_ok / non_ambig_total) if non_ambig_total else 0.0

    print(f"\n=== Empirical Laya Evaluation: model={label} ===")
    print(
        f"Overall Accuracy: {correct}/{total} ({overall_acc:.1%}) | "
        f"Non-Ambiguous: {non_ambig_correct}/{non_ambig_total} ({non_ambig_acc:.1%}) | "
        f"Direct-Route (>=0.80): {direct_ok}/{non_ambig_total} ({direct_rate:.1%})"
    )
    print(
        f"Confidence Distribution: min={min(confidences):.4f}, "
        f"mean={sum(confidences)/len(confidences):.4f}, max={max(confidences):.4f}"
    )
    print(f"Usage Diagnostics: truncations={truncations}, option_anomalies={option_anomalies}")
    print("Per-Category Accuracy:")
    for cat in [
        "question", "coding", "monitoring", "investigation", "research",
        "documentation", "grooming", "design", "mixed", "ambiguous",
    ]:
        c_tot = category_total[cat]
        c_ok = category_correct[cat]
        print(f"  - {cat:14s}: {c_ok}/{c_tot} ({(c_ok / c_tot) if c_tot else 0.0:.0%})")

    if misclassifications:
        print("Misclassifications:")
        for m in misclassifications:
            print(
                f"  * [{m['expected']}] -> intent={m['predicted_intent']} "
                f"(conf={m['confidence']}, confirm={m['confirm']}): {m['prompt']}"
            )
    else:
        print("Misclassifications: None")

    return {
        "model": label,
        "overall_accuracy": overall_acc,
        "non_ambiguous_accuracy": non_ambig_acc,
        "direct_route_rate": direct_rate,
        "direct_ok": direct_ok,
        "correct": correct,
        "total": total,
        "mean_confidence": sum(confidences) / len(confidences),
        "truncations": truncations,
        "option_anomalies": option_anomalies,
    }


def main() -> int:
    results = [
        evaluate_model(None),
        evaluate_model("typed-decisions"),
    ]
    best = max(
        results,
        key=lambda r: (r["direct_route_rate"], r["overall_accuracy"], r["mean_confidence"]),
    )
    print(
        f"\nRecommended Router Model: {best['model']} "
        f"(direct-route={best['direct_ok']}/27 {best['direct_route_rate']:.1%}, "
        f"overall={best['correct']}/{best['total']} {best['overall_accuracy']:.1%}, "
        f"mean_conf={best['mean_confidence']:.4f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
