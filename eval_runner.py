"""eval_runner.py — Eval runner for the ask-the-course RAG CLI.

Iterates over `questions.json` and runs `rag_ask.py` in two modes for each
question: with retrieval (default) and without retrieval (`--no-retrieval`).
Scores each run against the question's `expect` field and writes results
to `CHECKS.md`.

Scoring:
- `expect="answer"` → PASS if the output is NOT exactly the refusal string.
- `expect="refuse"` → PASS if the output IS exactly the refusal string
  (after whitespace strip).

The Step 4 baseline disagreement observation is recorded as a separate
section in CHECKS.md.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent
QUESTIONS_FILE: Path = PROJECT_ROOT / "questions.json"
CHECKS_FILE: Path = PROJECT_ROOT / "CHECKS.md"

REFUSAL_STRING: str = "I can't find that in the course documents."
TIMEOUT_SECONDS: int = 90


def run_ask(question: str, no_retrieval: bool) -> str:
    """Run rag_ask.py for `question` and return its stdout."""
    cmd = ["uv", "run", "python", "rag_ask.py", question]
    if no_retrieval:
        cmd.append("--no-retrieval")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        timeout=TIMEOUT_SECONDS,
    )
    return result.stdout


def score(question_data: dict, output: str) -> bool:
    """Return True if the run passes against the question's expectation."""
    expect = question_data["expect"]
    output_stripped = output.strip()
    if expect == "answer":
        return output_stripped != REFUSAL_STRING
    if expect == "refuse":
        return output_stripped == REFUSAL_STRING
    raise ValueError(f"Unknown expect value: {expect!r}")


def _first_line(text: str, max_chars: int = 200) -> str:
    """Return the first line of `text`, truncated for compact table display."""
    lines = text.strip().splitlines()
    if not lines:
        return ""
    first = lines[0]
    if len(first) > max_chars:
        first = first[:max_chars] + "..."
    # Escape pipes so they don't break the markdown table.
    return first.replace("|", "\\|")


def format_check_md(
    questions: list[dict],
    with_results: list[tuple[str, bool]],
    without_results: list[tuple[str, bool]],
) -> str:
    """Build the CHECKS.md content."""
    n = len(questions)
    with_pass = sum(1 for _, ok in with_results if ok)
    without_pass = sum(1 for _, ok in without_results if ok)

    parts: list[str] = [
        "# CHECKS.md",
        "",
        "Results from `python eval_runner.py` against `questions.json`. Each",
        "question is run through `rag_ask.py` in two modes (with and without",
        "retrieval) and scored against the question's `expect` field.",
        "",
        "## With retrieval",
        "",
        "| Q# | Question | Expected | PASS/FAIL | First line of output |",
        "|---|---|---|---|---|",
    ]
    for q, (out, ok) in zip(questions, with_results):
        mark = "PASS" if ok else "FAIL"
        parts.append(
            f"| {q['id']} | {q['question']} | {q['expect']} | {mark} | {_first_line(out)} |"
        )
    parts.append("")
    parts.append(f"**With-retrieval total: {with_pass}/{n}**")
    parts.append("")

    parts.extend([
        "## Without retrieval",
        "",
        "| Q# | Question | Expected | PASS/FAIL | First line of output |",
        "|---|---|---|---|---|",
    ])
    for q, (out, ok) in zip(questions, without_results):
        mark = "PASS" if ok else "FAIL"
        parts.append(
            f"| {q['id']} | {q['question']} | {q['expect']} | {mark} | {_first_line(out)} |"
        )
    parts.append("")
    parts.append(
        f"**Without-retrieval total: {without_pass}/{n}** "
        f"(only the out-of-scope question is expected to pass)"
    )
    parts.append("")

    parts.extend([
        "## Step 4 disagreement observation (baseline, no fix in v1)",
        "",
        "The retriever surfaces chunks by raw cosine similarity; it does NOT",
        "weight by recency or file path. When two chunks disagree on a fact",
        "(e.g., `weeks/01/lab.md` vs `weeks/03/lab.md` on a model name),",
        "both surface in the top-k and the LLM may report the stale answer",
        "from the older chunk. This is the documented baseline; fixing",
        "recency weighting or implementing a disagreement protocol is a",
        "future intent.",
        "",
        "Observation for this eval set: none of the five questions exercise a",
        "between-weeks factual conflict. Q4 is multi-source but draws from",
        "complementary facts in `syllabus.md` and an assignment, not from",
        "conflicting facts across weeks. A probe question that exercises the",
        "disagreement case (e.g., a deliberately contradictory fact between",
        "two weeks) is left for a future eval set.",
        "",
    ])

    return "\n".join(parts) + "\n"


def main() -> int:
    questions: list[dict] = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))

    with_results: list[tuple[str, bool]] = []
    without_results: list[tuple[str, bool]] = []

    for q in questions:
        print(f"Q{q['id']} (with retrieval)...", file=sys.stderr, flush=True)
        out = run_ask(q["question"], no_retrieval=False)
        ok = score(q, out)
        with_results.append((out, ok))
        print(f"  -> {'PASS' if ok else 'FAIL'}", file=sys.stderr, flush=True)

        print(f"Q{q['id']} (without retrieval)...", file=sys.stderr, flush=True)
        out = run_ask(q["question"], no_retrieval=True)
        ok = score(q, out)
        without_results.append((out, ok))
        print(f"  -> {'PASS' if ok else 'FAIL'}", file=sys.stderr, flush=True)

    md = format_check_md(questions, with_results, without_results)
    CHECKS_FILE.write_text(md, encoding="utf-8")

    with_pass = sum(1 for _, ok in with_results if ok)
    without_pass = sum(1 for _, ok in without_results if ok)
    n = len(questions)

    print(f"\nWith retrieval:    {with_pass}/{n}")
    print(f"Without retrieval: {without_pass}/{n}")
    print(f"Wrote {CHECKS_FILE}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
