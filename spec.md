# Spec: Ask the course

<!-- Written from the approved intent/ask-the-course.md. Validate against the intent; if they
     disagree, the intent wins until you change the intent. -->

## Intent
Implements `intent/ask-the-course.md` — the single-shot RAG CLI that answers questions about CPSC 415 using the course's own Markdown pages, cites the file and section heading of each source, prints retrieved chunks with raw cosine similarity scores beneath the answer, and refuses — with a fixed string — when the pages do not contain the answer.

## Components

### Indexer
- **What it does:** Walks the allowed corpus (`syllabus.md`, `assignments/`, `weeks/01`–`weeks/03`), parses each Markdown file into sections (ATX `#`/`##`/`###`), splits each section into ~800-char sentence-aware chunks with ~100-char overlap, embeds each chunk with `openai/text-embedding-3-small` via OpenRouter, and persists vectors + metadata to `index.json`. The cache is invalidated when the corpus's max mtime advances past the recorded mtime.
- **Language:** Python 3.10+. **Why:** matches `CLAUDE.md`. Standard library covers Markdown section parsing, JSON I/O, and file walking; `numpy` covers the vector matrix. Java was considered — rejected because the course's primary tooling (`uv`, OpenRouter client examples, embedding scripts in earlier labs) is Python-first, and an unfamiliar-language exercise is a separate guided activity.
- **Model:** `openai/text-embedding-3-small`. **Why:** cheap (~$0.02/M tokens), fast, available on OpenRouter, English-technical training distribution matches the corpus. **Cheap vs frontier comparison:** `text-embedding-3-small` vs `text-embedding-3-large` (frontier, ~5x cost). For a corpus of ~50 short technical chunks, the frontier model is overkill — on similar short-text retrieval tasks in earlier course work, `large` shifted top-1 retrieval by ≤ 0.02 cosine on most chunks. `small` is chosen; `large` is left as the swap-in if a benchmark on the held-out question reveals poor ranking.
- **Interfaces:**
  - CLI: `python rag_index.py` — rebuilds the index, prints chunk count.
  - Library: `load_chunks() → list[Chunk]`, `build_index(chunks) → Index`, `get_index() → Index`, `retrieve(query: str, k: int = 5) → list[ScoredChunk]`.
  - Files: reads from `../ai-integration-course/{syllabus.md, assignments/, weeks/01..03}`; writes `./index.json`.
- **Dependencies:**
  - `httpx` — synchronous HTTP client to OpenRouter's `/embeddings` endpoint. Reason: ~50 chunks fit in one batched call; no async overhead needed.
  - `numpy` — vector matrix and cosine similarity. Reason: a few-line dot product on a `(N, 1536)` matrix is faster and clearer than a Python loop, and lets us reason about top-k sorting explicitly.

### Asker
- **What it does:** Embeds the user's question with the same embedding model, retrieves the top `TOP_K = 10` chunks by cosine similarity, applies a programmatic threshold gate (cosine ≥ `0.30`), then asks the LLM to answer strictly from the retrieved excerpts (with strict prompt grounding). Formats the response: synthesized answer → `Citations:` block (`file > section`) → `Retrieved chunks:` block with raw cosine scores. Supports a `--no-retrieval` ablation mode that skips retrieval entirely. **Empirical tuning:** `TOP_K` is 10 (raised from proposed 5) and the threshold is `0.30` (lowered from proposed 0.5) — see Behavior 7 for rationale.
- **Language:** Python 3.10+. **Why:** same as Indexer — Python is the project default and the Asker imports `rag_index`'s retriever directly.
- **Model:** `minimax/minimax-m3`. **Why:** project default per `CLAUDE.md`. **Cheap vs frontier comparison:** `minimax/minimax-m3` vs `anthropic/claude-sonnet-4.5` (frontier). On grounding tasks where the heavy lifting is retrieval, m3 produces correct, cited answers; the frontier model paraphrases the same content more fluently but adds no information and costs ~10x more. m3 is sufficient for this task; Sonnet is left as the swap-in if the eval shows m3 paraphrasing the refusal string or skipping citations.
- **Interfaces:**
  - CLI: `python rag_ask.py "..."` (with retrieval) and `python rag_ask.py "..." --no-retrieval` (ablation).
  - Library: `answer(question: str, retrieval: bool = True) → Answer` where `Answer` has `text`, `citations`, `chunks_with_scores`.
  - Files: reads `./index.json` (built by the Indexer); stdout-only output (no files written).
- **Dependencies:**
  - `httpx` — OpenRouter `/chat/completions` call. Reason: same client used by the Indexer; keeps the dependency surface to one HTTP library.
  - `numpy` — not directly used by the Asker; pulled in transitively because the Asker imports `rag_index` and uses its scoring helpers.

## Behavior
Numbered; the eval and manual checks verify these.

1. **Corpus is scoped correctly.** The Indexer reads only `syllabus.md`, `assignments/`, and `weeks/01`–`weeks/03`. `weeks/04`, `slides/`, and `README.md` are skipped. No page text is copied into this repo.
2. **Chunks carry source metadata.** Every chunk has `source_path`, `section_heading`, `chunk_index`, and `text`. Section headings are parsed from ATX Markdown.
3. **Chunks are sentence-aware.** Cuts occur at sentence boundaries (`. `, `.\n`, `? `, `! `). Target ~800 chars, overlap ~100 chars. No chunk ends mid-sentence.
4. **Index is cached to `index.json`.** Embeddings + metadata persist to JSON. Cache is invalidated when the corpus's max mtime exceeds the cache's recorded mtime.
5. **CLI accepts a single question.** `python rag_ask.py "When is the midterm?"` exits 0 and prints answer + citations + retrieved chunks with raw cosine scores.
6. **CLI accepts `--no-retrieval`.** `python rag_ask.py "..." --no-retrieval` skips retrieval and calls the LLM with no excerpts in the prompt. Used by the eval runner for the ablation.
7. **Two-stage refusal (programmatic threshold).** The retriever returns the top `TOP_K = 10` chunks by raw cosine similarity. If none of them scores ≥ the cosine threshold (`0.30`), the CLI prints exactly `I can't find that in the course documents.` and exits 0 **without calling the LLM**. Both values were tuned empirically against this corpus: the originally proposed `0.5` threshold would refuse every query (observed top scores for in-scope queries cluster at 0.37–0.40), and `TOP_K = 5` would leave the mid-ranked chunk that carries the answer (e.g., the `Week 6 — Monday, October 26 (mid-term)` heading) out of the LLM's prompt.
8. **Two-stage refusal (prompt grounding).** When the threshold is met, the system prompt instructs the LLM to answer only from the retrieved excerpts and to emit the same refusal string if the excerpts do not contain the answer.
9. **Refusal string is exact.** When either refusal stage fires, the CLI prints exactly `I can't find that in the course documents.` — same case, no surrounding punctuation, no extra whitespace. Matched with `==`, not `in`.
10. **Answer cites every source used.** When the LLM draws from multiple chunks across multiple files, the `Citations:` block lists each one as `file > section heading`.
11. **Retrieved chunks and raw cosine scores are printed below the answer.** After the answer and citations, the CLI prints a `Retrieved chunks:` block showing each chunk's source path, section heading, and raw cosine similarity score (full precision, not rounded for display).
12. **Disagreement behavior.** When retrieved chunks disagree on a fact (e.g., conflicting model names across weeks), the LLM answers based strictly on the retrieved excerpts, citing all retrieved sources used. Resolving conflicts by timestamp or file heuristic is an observation point for Step 4, not an active filter in the baseline retriever.
13. **Eval runner loops over `questions.json` in both modes.** `eval_runner.py` runs `rag_ask.py` once per question with and without `--no-retrieval`, captures stdout, scores pass/fail against the `expect` field, and writes `CHECKS.md` with per-question rows and two totals.

## Failure handling
- **Missing `OPENROUTER_API_KEY`:** fail loudly at startup with a clear message naming the env var; do not 401 mid-run.
- **Network error / timeout:** retry up to 3 times with exponential backoff (1s, 2s, 4s); on persistent failure, print the refusal string and exit non-zero (the refusal string is the safest output).
- **`index.json` corrupt or unreadable:** rebuild from scratch; do not crash. Print a one-line note.
- **Empty question / extra args:** argparse error and exit non-zero with usage. Do not silently answer an empty string.
- **LLM returns malformed or empty content:** treat as "excerpts insufficient"; print the refusal string.
- **LLM paraphrases the refusal string** ("I cannot find that...", "The documents do not contain..."): treat as a refusal; the eval runner normalizes whitespace and matches `==` to the canonical string.
- **Rate limit (HTTP 429):** back off and retry; if persistent, print a one-line warning to stderr and exit non-zero.
- **Hallucination past the excerpts:** the prompt is strict and citations are forced by the format requirement. If the LLM ignores it, the disagreement-protocol instruction plus the printed chunks give the grader the raw evidence to catch it.

## Cost estimate
Per single invocation (`python rag_ask.py "..."`):
- 1 embedding call: `text-embedding-3-small`, ~50 tokens for a typical question. Cost: ~$0.000001.
- 1 LLM call: `minimax/minimax-m3`, ~5k tokens in (system prompt + question + 10 excerpts) + ~200 tokens out. At typical m3 pricing ~$0.30/M input, $1.20/M output: ~$0.0015 + ~$0.00024 ≈ **$0.0018 per question**.

Per eval run (10 invocations: 5 questions × 2 modes): **~$0.018**.

Semester-of-use estimate: a student running ~100 questions across the semester for studying ≈ **$0.18** — well under the $20 initial budget in `syllabus.md`.

## Out of scope
Carried over from `intent/ask-the-course.md`:
- Multi-turn / conversation memory.
- Web access, live updates, refreshing the index mid-run.
- Auth, user accounts, telemetry.
- Streaming output.
- Non-text inputs (PDFs, images, etc.).
- Editing or writing course files.
- Outside knowledge — the LLM must not infer beyond the retrieved excerpts.

Added by this spec:
- Swapping embedding or LLM providers at runtime. Models are pinned in code; switching is a code change, not a config change.
- Local open-source models (Ollama, etc.) for this lab — a possible future intent.
- A web UI or REST API — CLI only.
- Streaming token output from the LLM. Each invocation is one complete response.
