# Ask the Course — Week 4 RAG CLI

A single-shot CLI that answers questions about CPSC 415 using the course's own Markdown pages. Every answer cites the file and section heading of each source, prints the retrieved chunks with raw cosine similarity scores beneath the answer, and refuses with a fixed string when the pages do not contain the answer.

## Setup

```bash
# Install dependencies (Python 3.10+, httpx, numpy)
uv sync

# Set the OpenRouter API key (required)
export OPENROUTER_API_KEY=sk-or-v1-...

# Build the index (first run only; subsequent runs load from index.json)
uv run python rag_index.py
```

## Usage

```bash
# Build / refresh the index. Walks the corpus, parses, chunks, embeds, and
# caches to ./index.json. Re-uses the cache unless the corpus changes.
uv run python rag_index.py

# Ask a question — full RAG with retrieval
uv run python rag_ask.py "When is the midterm?"

# Ask with the --no-retrieval ablation (LLM only, no chunks in the prompt)
uv run python rag_ask.py "When is the midterm?" --no-retrieval

# Run the full eval against questions.json, writing CHECKS.md
uv run python eval_runner.py
```

`rag_ask.py` exits 0 on both answer and refusal.

## Files

| File | Purpose |
|---|---|
| `pyproject.toml`, `uv.lock` | uv project config + lockfile (Python 3.10+, `httpx`, `numpy`). |
| `intent/ask-the-course.md` | The approved intent that drives the spec and plan. |
| `spec.md` | The approved design — Indexer, Asker, behaviors, failure modes, cost. |
| `plan.md` | The order of work, per-step failure modes, and proof checks. |
| `rag_index.py` | **Indexer**: corpus walk, ATX parser, sentence-aware chunker, OpenRouter embedder, `index.json` cache, retriever. |
| `rag_ask.py` | **Asker**: CLI, two-stage refusal, prompt-grounded LLM call, output formatting. |
| `questions.json` | The five eval questions authored by the student. |
| `eval_runner.py` | Loops `questions.json` in both modes, scores pass/fail, writes `CHECKS.md`. |
| `index.json` | (Generated.) Cached chunk embeddings + metadata + corpus mtime. |
| `CHECKS.md` | (Generated.) Per-question rows + totals from the latest eval run. |

## Evaluation questions

From `questions.json` — five questions authored by the student to exercise the four behaviors the lab requires:

| # | Question | Expect | What it catches |
|---|---|---|---|
| 1 | When is the midterm exam? | answer | Single-place fact — `syllabus.md > Week 6 — Monday, October 26 (mid-term)`. |
| 2 | How many late days are students allowed to use across the semester without penalty? | answer | Single-place fact — `syllabus.md > Late Submission Policy`. |
| 3 | What percentage of the course grade does Team Project 1 count for? | answer | Single-place fact — `assignments/team-project-1.md`. |
| 4 | Can late days be applied to Team Project 1, and what happens if a team submits after all late days are exhausted? | answer | Multi-source synthesis across `syllabus.md` (Late Submission Policy) and `assignments/team-project-1.md`. |
| 5 | What university did the instructor attend for his undergraduate degree? | refuse | Out-of-scope question not present in any course document; tests exact refusal string. |

## Evaluation results

From `CHECKS.md`:

| Mode | Pass | Total | Notes |
|---|---|---|---|
| With retrieval | **5** | 5 | All four in-scope questions answered correctly; the out-of-scope question refused. |
| Without retrieval (`--no-retrieval`) | **1** | 5 | Only the out-of-scope question passes — without chunks, the LLM has no corpus information and refuses everything in-scope. |

The contrast (5/5 vs. 1/5) is the proof that the answers are grounded in the retrieved excerpts, not in the LLM's prior training.

## Step 4 disagreement observation (baseline, no fix in v1)

The retriever surfaces chunks by raw cosine similarity. It does **not** weight by recency or file path. When two chunks disagree on a fact — e.g., `weeks/01/lab.md` states one model name and `weeks/03/lab.md` states a different one — both chunks surface in the top-k because both score highly on cosine. The LLM then synthesizes from the retrieved excerpts and may report the **stale answer** from the older chunk.

This is the documented baseline for v1; fixing recency weighting or implementing a disagreement protocol is a **future intent**.

### Step 4 Probe Run
**Question asked:**
`"Which Xiaomi MiMo model does this course use as the fallback or second model?"`

**Observed Program Answer:**
The program surfaced both models rather than determining which is current:
- `xiaomi/mimo-v2.5` — cited from `weeks/01/setup.md > 1. Accounts and a small budget` (cosine=0.4958)
- `xiaomi/mimo-v2.6-flash` — cited from `weeks/03/lab.md > 4. The second model (5 minutes)` (cosine=0.5104)

**Cause:**
Both chunks score high cosine similarity due to matching terminology. The retriever uses raw cosine similarity without recency or chronological path weighting, surfacing both into the top-k context.

**One way a RAG system could prefer the current page (baseline observation, no fix in v1):**
- **Chronological / Path Boosting:** Apply a score boost to chunks based on directory hierarchy (e.g., weighting `weeks/03/` over `weeks/01/`).
- **Timestamp Decay / Prompt Grounding:** Ingest file modification dates (`mtime`) into chunk metadata and decay cosine scores over time, or supply document dates directly in the prompt and instruct the LLM: *"If two retrieved excerpts state conflicting facts, prefer the one with the later week number or more recent date."*

## Empirical tunings

Three tunings were made based on observed retrieval behavior against this corpus. All three are documented in `spec.md` (Behavior 7) and `plan.md` (Steps 3 and 5).

### 1. Cosine threshold lowered from `0.5` to `0.30`

Observed top scores for in-scope queries on this corpus cluster at **0.37–0.40** (short queries against longer chunks dilute the cosine signal). The originally proposed `0.5` would refuse every query. `0.30` keeps in-scope queries passing while still rejecting clearly unrelated text — the sanity check on `"How do I cook pasta?"` scored **0.05**.

### 2. `TOP_K` raised from `5` to `10`

The chunk carrying the midterm answer (`Week 6 — Monday, October 26 (mid-term)`) ranks **8** with score **0.33**. With `TOP_K = 5` it's outside the LLM's prompt and the model refuses (even though the corpus has the answer). With `TOP_K = 10` the chunk reaches the prompt and the model answers correctly.

### 3. Section heading prepended to chunk text

Without this, sections whose heading carries the answer but whose body is on an unrelated topic rank poorly. Example: `### Week 6 — Monday, October 26 (mid-term)` has a body about image generation. Before prepending, the chunk scored **0.17** at rank 15+. After prepending the heading to the chunk text (in `chunk_section` in `rag_index.py`), the same chunk scored **0.33** at rank 8.

The `section_heading` field on each `Chunk` is unchanged — citations still use the parsed heading — but the embedded text starts with the heading line, so the embedder captures the heading topic. `INDEX_VERSION` was bumped to `2` to invalidate the old cache.

## Code walkthrough: two-stage refusal

`rag_ask.py` implements the two-stage refusal in the `answer()` function. The contract is that both stages produce the exact same output string, so the caller (the CLI, the eval runner) doesn't need to distinguish which stage fired.

**Stage 1 — programmatic cosine threshold.** The retriever returns the top `TOP_K = 10` chunks with raw cosine scores. If the top score is below `0.30`, the function returns the refusal string **without calling the LLM**:

```python
if retrieval:
    index = get_index()
    chunks_with_scores = retrieve(question, index=index, k=TOP_K)

    # Stage 1: programmatic cosine threshold.
    if not chunks_with_scores or chunks_with_scores[0][1] < COSINE_THRESHOLD:
        return REFUSAL_STRING, [], chunks_with_scores
```

This stage handles the "no relevant chunks at all" case cheaply — no LLM call, no API cost.

**Stage 2 — prompt-grounded LLM call.** If stage 1 passes, the system prompt instructs the LLM to answer only from the retrieved excerpts and to emit the refusal string verbatim if the excerpts don't contain the answer:

```python
SYSTEM_PROMPT = """You are an assistant answering questions about CPSC 415 ...
You answer ONLY from the excerpts of the course documents provided below. ...
If the excerpts do not contain enough information to answer the question,
output EXACTLY this string and nothing else:
I can't find that in the course documents.
..."""
```

After the LLM responds, the function checks whether the response equals the refusal string verbatim. If so, it propagates the refusal:

```python
response_text = _openrouter_chat(messages).strip()
if response_text == REFUSAL_STRING:
    return REFUSAL_STRING, [], chunks_with_scores
```

This stage handles the "chunks are weakly related but don't actually answer the question" case (e.g., a query about the instructor's PhD advisor retrieves the "Contact Info" section, which is about the instructor but doesn't mention his PhD advisor).

Both stages emit the same exact string: `I can't find that in the course documents.` — same case, no surrounding punctuation, no extra whitespace. The eval runner matches with `==` after a `strip()`, so any paraphrase ("I cannot find...", "The documents do not contain...") is treated as a failure.

## Cost

Per single invocation of `rag_ask.py "..."`: ~**\$0.0018** (1 embedding call + 1 LLM call, ~5k input tokens + ~200 output tokens on `minimax/minimax-m3`). The full eval run (10 invocations) costs ~**\$0.018**. A student running ~100 questions across the semester pays ~**\$0.18** — well under the \$20 initial OpenRouter budget in `syllabus.md`.
