# Intent: Ask the course

## Goal
A single-shot CLI tool, `python rag_ask.py "..."`, that answers questions about CPSC 415 by retrieving and reading the course's own Markdown pages. Every answer cites the file and section heading of each page used, prints the retrieved chunks and their raw cosine similarity scores directly below the response, and refuses — with a fixed string — when the pages do not contain the answer.

## Who it is for
The student (you) and a grader/reviewer who must verify both that the answers are correct **and** that they are grounded in the course documents. Today that verification is done by hand-searching the syllabus and weekly pages.

## Constraints
- **Language & build:** Python 3.10+, run with `uv`. `uv` is the package tool.
- **Model:** default model for the project, `minimax/minimax-m3`.
- **Corpus (strict):** only `../ai-integration-course/syllabus.md`, `../ai-integration-course/assignments/`, and `../ai-integration-course/weeks/01` through `weeks/03`. `weeks/04`, `slides/`, and `README.md` are out of scope. Pages are never copied into this repo.
- **Chunk metadata:** each indexed chunk is prepended with its source file and parsed Markdown section heading, e.g. `[weeks/03/lab.md > 4. The second model]`.
- **Citations:** every answer names each source as `file > section heading`, e.g. `weeks/02/syllabus.md > Late-work policy`.
- **Score reporting:** every answer prints the retrieved chunks with their raw cosine similarity scores directly below the response.
- **Two-stage refusal:** (1) a programmatic cosine-similarity threshold gate over the top-k chunks; (2) a strict system-prompt instruction that the LLM answers **only** from the excerpts that passed the gate, and outputs the refusal string if the excerpts do not contain the answer.
- **Refusal string:** exactly `I can't find that in the course documents.` — no surrounding text, no punctuation changes.
- **Eval contract:**
  - `python rag_ask.py "..."` runs with retrieval (default).
  - `python rag_ask.py "..." --no-retrieval` skips retrieval — the LLM receives only the system prompt and the question.
  - A runner script loops over `questions.json` in both modes and writes results into `CHECKS.md`.
- **`questions.json` shape (5 questions):** 3 single-place facts (due date, grade weight, lab step), 1 synthesized multi-source question spread across two pages, 1 out-of-scope question whose expected response is the refusal string.

## Not in scope
- **No multi-turn / conversation memory.** Each invocation is independent.
- **No web access or live updates.** The index is built from the course repo and not refreshed mid-run.
- **No auth, user accounts, or telemetry.** Local CLI only.
- **No streaming output.** Each invocation prints the full response and exits.
- **No non-text inputs.** Markdown only — no PDFs, images, audio, etc.
- **No editing or writing course files.** Read-only over `../ai-integration-course`.
- **No outside knowledge.** The LLM answers strictly from retrieved excerpts; it does not infer, speculate, or fill gaps from its own training data.

## Success looks like
1. `python rag_ask.py "When is the midterm?"` returns a short answer, a `weeks/NN/<file>.md > <section>` citation, and the retrieved chunks with cosine scores printed beneath.
2. `python rag_ask.py "Who is the instructor's PhD advisor?"` (or any question outside the corpus) prints exactly `I can't find that in the course documents.` and nothing else.
3. The eval runner, fed `questions.json`, writes `CHECKS.md` showing **5/5 pass with retrieval** and **1/5 pass without retrieval** (only the out-of-scope question passes).

## Open questions
- Embedding model, top-k, chunk size, and the exact cosine threshold value are deferred to `spec.md`.
- Whether to cache the index to disk across runs (so the second invocation is fast) is deferred to `spec.md`.

**Approved by:** Javier Romo, October 17, 2026
