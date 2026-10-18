# CHECKS.md

Results from `python eval_runner.py` against `questions.json`. Each
question is run through `rag_ask.py` in two modes (with and without
retrieval) and scored against the question's `expect` field.

## With retrieval

| Q# | Question | Expected | PASS/FAIL | First line of output |
|---|---|---|---|---|
| 1 | When is the midterm exam? | answer | PASS | The midterm is on **Monday, October 26** (Week 6). That class session covers image generation, deployment, and review, and serves as the mid-term meeting of the semester. |
| 2 | How many late days are students allowed to use across the semester without penalty? | answer | PASS | Each student is allocated **3 late days** per semester, usable at their discretion in 24-hour increments. Late days may be applied to any assignment and used individually or together. There is no bonu... |
| 3 | What percentage of the course grade does Team Project 1 count for? | answer | PASS | Team Project 1 counts for **15%** of the course grade. |
| 4 | Can late days be applied to Team Project 1, and what happens if a team submits after all late days are exhausted? | answer | PASS | Yes, late days can be applied to Team Project 1. The late submission policy states that late days "may be applied to any assignment," with the only exceptions being presentations and the Comprehension... |
| 5 | What university did the instructor attend for his undergraduate degree? | refuse | PASS | I can't find that in the course documents. |

**With-retrieval total: 5/5**

## Without retrieval

| Q# | Question | Expected | PASS/FAIL | First line of output |
|---|---|---|---|---|
| 1 | When is the midterm exam? | answer | FAIL | I can't find that in the course documents. |
| 2 | How many late days are students allowed to use across the semester without penalty? | answer | FAIL | I can't find that in the course documents. |
| 3 | What percentage of the course grade does Team Project 1 count for? | answer | FAIL | I can't find that in the course documents. |
| 4 | Can late days be applied to Team Project 1, and what happens if a team submits after all late days are exhausted? | answer | FAIL | I can't find that in the course documents. |
| 5 | What university did the instructor attend for his undergraduate degree? | refuse | PASS | I can't find that in the course documents. |

**Without-retrieval total: 1/5** (only the out-of-scope question is expected to pass)

## Step 4 disagreement observation (baseline, no fix in v1)

The retriever surfaces chunks by raw cosine similarity; it does NOT
weight by recency or file path. When two chunks disagree on a fact
(e.g., `weeks/01/lab.md` vs `weeks/03/lab.md` on a model name),
both surface in the top-k and the LLM may report the stale answer
from the older chunk. This is the documented baseline; fixing
recency weighting or implementing a disagreement protocol is a
future intent.

### Step 4 Probe Run:
**Question:** `"Which Xiaomi MiMo model does this course use as the fallback or second model?"`

**Observed Output:**
The program reported both models rather than preferring the current one:
- `xiaomi/mimo-v2.5` — cited from `weeks/01/setup.md > 1. Accounts and a small budget` (cosine=0.4958)
- `xiaomi/mimo-v2.6-flash` — cited from `weeks/03/lab.md > 4. The second model (5 minutes)` (cosine=0.5104)

**Possible Fixes (not implemented in v1):**
- Recency weighting: decay similarity scores by file modification time (`mtime`) or Git commit date.
- Path prioritization: boost chunks located in later week directories (e.g., `weeks/03/` over `weeks/01/`).
- Date metadata in prompt: supply publication/week dates in chunk citations and instruct the LLM to prefer the more recent source in conflicts.

