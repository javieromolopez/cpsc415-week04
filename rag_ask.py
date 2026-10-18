"""rag_ask.py — Asker component for the ask-the-course RAG CLI.

Answers a question about CPSC 415 using the course's Markdown pages (via
`rag_index`'s retriever) or, with `--no-retrieval`, the LLM's own knowledge
only. Implements a two-stage refusal:

1. Programmatic cosine threshold. If no retrieved chunk meets the
   threshold, output the refusal string without calling the LLM.
2. Prompt grounding. The system prompt instructs the LLM to answer only
   from the provided excerpts, and to emit the refusal string verbatim
   if the excerpts don't contain the answer.

Output format (per `spec.md`):
- synthesized answer
- `Citations:` block listing `source_path > section_heading`
- `Retrieved chunks:` block with raw cosine similarity scores

On refusal (either stage), prints exactly:
    I can't find that in the course documents.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import httpx

from rag_index import (
    Chunk,
    get_index,
    retrieve,
    OPENROUTER_BASE_URL,
    HTTP_TIMEOUT_SECONDS,
    HTTP_RETRY_MAX,
)

# --- Config --------------------------------------------------------------

REFUSAL_STRING: str = "I can't find that in the course documents."
# Cosine threshold for stage-1 refusal. The plan proposed 0.5; observed
# retrieval scores on this corpus cluster around 0.37–0.40 for topically
# relevant queries, so 0.5 would refuse every query. 0.30 keeps in-scope
# queries passing while still refusing clearly unrelated text (e.g., the
# sanity check on "How do I cook pasta?" scored 0.05).
COSINE_THRESHOLD: float = 0.30
TOP_K: int = 10
CHAT_MODEL: str = "minimax/minimax-m3"


# --- OpenRouter chat -----------------------------------------------------

def _api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set; export it before running."
        )
    return key


def _openrouter_chat(messages: list[dict], model: str = CHAT_MODEL) -> str:
    """Call OpenRouter `/chat/completions` and return the assistant text.

    Retries up to HTTP_RETRY_MAX times with exponential backoff on 429/5xx
    or transport errors. Raises if all attempts fail.
    """
    url = f"{OPENROUTER_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
    }
    payload: dict[str, object] = {
        "model": model,
        "messages": messages,
        "temperature": 0,
    }

    last_exc: Exception | None = None
    for attempt in range(HTTP_RETRY_MAX):
        try:
            response = httpx.post(
                url,
                json=payload,
                headers=headers,
                timeout=HTTP_TIMEOUT_SECONDS,
            )
            if response.status_code == 429 or response.status_code >= 500:
                last_exc = httpx.HTTPStatusError(
                    f"{response.status_code} {response.reason_phrase}",
                    request=response.request,
                    response=response,
                )
            else:
                response.raise_for_status()
                data = response.json()
                return str(data["choices"][0]["message"]["content"])
        except httpx.HTTPError as exc:
            last_exc = exc

        time.sleep(2 ** attempt)

    raise RuntimeError(
        f"OpenRouter chat call failed after {HTTP_RETRY_MAX} attempts: {last_exc}"
    )


# --- System prompt -------------------------------------------------------

SYSTEM_PROMPT = """You are an assistant answering questions about CPSC 415 (AI Integration, Trinity College, Fall 2026).

You answer ONLY from the excerpts of the course documents provided below. You do not use any other knowledge, and you do not speculate.

If the excerpts do not contain enough information to answer the question, output EXACTLY this string and nothing else (no preamble, no explanation, no quotes, no extra punctuation):

I can't find that in the course documents.

Otherwise, give a concise synthesized answer to the question. Do not include any citations or "References" section in your response — those will be added programmatically from the excerpts.

The provided excerpts follow. Each excerpt shows its source path, section heading, cosine similarity to the question, and the excerpt text:

{excerpts}
"""


def _format_excerpts(chunks_with_scores: list[tuple[Chunk, float]]) -> str:
    """Format retrieved chunks into the prompt's excerpts section."""
    if not chunks_with_scores:
        return "(no excerpts provided)"
    parts: list[str] = []
    for chunk, score in chunks_with_scores:
        parts.append(
            f"[{chunk.source_path} > {chunk.section_heading}] "
            f"(cosine={score:.4f})\n{chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


# --- Core answer logic ---------------------------------------------------

def answer(
    question: str,
    retrieval: bool = True,
) -> tuple[str, list[str], list[tuple[Chunk, float]]]:
    """Answer `question`.

    Returns `(text, citations, chunks_with_scores)`.
    - `text` is the synthesized answer, or REFUSAL_STRING on refusal.
    - `citations` is a list of `source_path > section_heading` strings,
      one per distinct source drawn from.
    - `chunks_with_scores` is the retrieved chunks with raw cosine scores
      (empty when `retrieval=False`).
    """
    chunks_with_scores: list[tuple[Chunk, float]] = []

    if retrieval:
        index = get_index()
        chunks_with_scores = retrieve(question, index=index, k=TOP_K)

        # Stage 1: programmatic cosine threshold.
        if not chunks_with_scores or chunks_with_scores[0][1] < COSINE_THRESHOLD:
            return REFUSAL_STRING, [], chunks_with_scores

    # Stage 2: LLM call with strict grounding prompt.
    excerpts_text = (
        _format_excerpts(chunks_with_scores) if retrieval else "(no excerpts provided)"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(excerpts=excerpts_text)},
        {"role": "user", "content": question},
    ]
    response_text = _openrouter_chat(messages).strip()

    # If the LLM emitted the refusal string verbatim, propagate it as refusal.
    if response_text == REFUSAL_STRING:
        return REFUSAL_STRING, [], chunks_with_scores

    # Build citations from the retrieved chunks (those are the only sources
    # the LLM saw). De-duplicate by (source_path, section_heading).
    citations: list[str] = []
    seen: set[tuple[str, str]] = set()
    for chunk, _ in chunks_with_scores:
        key = (chunk.source_path, chunk.section_heading)
        if key not in seen:
            citations.append(f"{chunk.source_path} > {chunk.section_heading}")
            seen.add(key)

    return response_text, citations, chunks_with_scores


# --- Output formatting ---------------------------------------------------

def _format_output(
    text: str,
    citations: list[str],
    chunks: list[tuple[Chunk, float]],
) -> str:
    """Format the CLI output per `spec.md`."""
    if text == REFUSAL_STRING:
        return REFUSAL_STRING

    parts: list[str] = [text, ""]

    if citations:
        parts.append("Citations:")
        for c in citations:
            parts.append(f"- {c}")
        parts.append("")

    if chunks:
        parts.append("Retrieved chunks:")
        for chunk, score in chunks:
            preview = chunk.text[:120].replace("\n", " ")
            parts.append(
                f"- [{chunk.source_path} > {chunk.section_heading}] "
                f"(cosine={score:.4f})"
            )
            parts.append(f"  preview: {preview}...")
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"


# --- CLI -----------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ask a question about CPSC 415.",
    )
    parser.add_argument("question", help="The question to answer.")
    parser.add_argument(
        "--no-retrieval",
        action="store_true",
        help="Skip retrieval; LLM answers from its own knowledge only.",
    )
    args = parser.parse_args(argv)

    text, citations, chunks = answer(args.question, retrieval=not args.no_retrieval)
    output = _format_output(text, citations, chunks)
    print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
