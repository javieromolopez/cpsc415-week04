"""rag_index.py — Indexer component for the ask-the-course RAG CLI.

Walks the course corpus, parses ATX Markdown sections, chunks each section
on sentence boundaries, embeds chunks via OpenRouter, caches the embeddings
to `index.json`, and exposes a retriever for cosine-similarity top-k lookup.

Corpus scope (per `intent/ask-the-course.md` and `spec.md`):
- `../ai-integration-course/syllabus.md`
- `../ai-integration-course/assignments/` (all `.md` files except `README.md`)
- `../ai-integration-course/weeks/01`, `weeks/02`, `weeks/03`
  (all `.md` files except `README.md`)

Explicitly skipped: `weeks/04`, `slides/`, any non-`.md` file, any
`README.md`, and any Markdown horizontal-rule line (`---`, `***`, `___`).

Baseline disagreement observation (no fix in v1): when two chunks disagree
on a fact (e.g., `weeks/01` vs `weeks/03` on a model name), the retriever
surfaces both because both score highly on cosine similarity. The retriever
does NOT weight by recency or file path. The LLM may report the stale answer;
this is recorded in `CHECKS.md` as an observation, not a failure.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np

# Corpus root, resolved relative to this script's location.
CORPUS_ROOT = Path(__file__).resolve().parent.parent / "ai-integration-course"

# Whitelist of corpus paths.
ALLOWED_TOP_FILES: tuple[str, ...] = ("syllabus.md",)
ALLOWED_TOP_DIRS: tuple[str, ...] = ("assignments",)
ALLOWED_WEEK_DIRS: tuple[str, ...] = ("01", "02", "03")

# Chunking parameters.
CHUNK_TARGET_SIZE: int = 800
CHUNK_OVERLAP: int = 100

# Embedding + cache config.
EMBEDDING_MODEL: str = "openai/text-embedding-3-small"
EMBEDDING_DIM: int = 1536
INDEX_FILE: Path = Path("index.json")
INDEX_VERSION: int = 2  # bumped: chunks now prepend the section heading
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
EMBEDDING_BATCH_SIZE: int = 64
HTTP_TIMEOUT_SECONDS: float = 60.0
HTTP_RETRY_MAX: int = 3


@dataclass(frozen=True)
class Section:
    """One ATX-keyed section of a Markdown file."""

    source_path: str
    section_heading: str
    text: str


@dataclass(frozen=True)
class Chunk:
    """A sentence-aware chunk of a section."""

    source_path: str
    section_heading: str
    chunk_index: int
    text: str


@dataclass(frozen=True)
class Index:
    """The cached embedding index."""

    chunks: tuple[Chunk, ...]
    embeddings: np.ndarray   # shape (N, EMBEDDING_DIM), float32
    corpus_max_mtime: float
    embedding_model: str


# --- Corpus walk ---------------------------------------------------------

def _walk_corpus(root: Path) -> list[Path]:
    """Yield `.md` files under the allowed corpus paths.

    Explicitly excludes `weeks/04/`, `slides/`, any non-`.md` file, and any
    file named `README.md` (case-insensitive) — including those inside week
    directories.
    """
    if not root.exists():
        raise FileNotFoundError(f"Corpus root not found: {root}")

    out: list[Path] = []

    for name in ALLOWED_TOP_FILES:
        p = root / name
        if p.is_file():
            out.append(p)

    for name in ALLOWED_TOP_DIRS:
        d = root / name
        if d.is_dir():
            out.extend(sorted(
                p for p in d.rglob("*.md")
                if p.is_file() and p.name.lower() != "readme.md"
            ))

    for week in ALLOWED_WEEK_DIRS:
        d = root / "weeks" / week
        if d.is_dir():
            out.extend(sorted(
                p for p in d.rglob("*.md")
                if p.is_file() and p.name.lower() != "readme.md"
            ))

    return out


# --- Section parser ------------------------------------------------------

_ATX_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
# Markdown horizontal rules: ---, ***, ___, ----, etc.
_HORIZONTAL_RULE = re.compile(r"^([-*_])\s*\1\s*\1[\1\s]*$")


def _parse_sections(path: Path, text: str) -> list[Section]:
    """Parse a Markdown file into ATX-keyed sections.

    Skips headings inside fenced code blocks (``` or ~~~). Horizontal-rule
    lines (`---`, `***`, `___`) are dropped entirely so they don't end up in
    chunk text. Content before the first heading is discarded (the corpus
    always starts with a heading).
    """
    rel = path.relative_to(CORPUS_ROOT).as_posix()
    sections: list[Section] = []
    heading: str | None = None
    body: list[str] = []
    in_fence = False

    def emit() -> None:
        nonlocal heading, body
        if heading is None:
            return
        joined = "\n".join(body).strip("\n")
        sections.append(Section(source_path=rel, section_heading=heading, text=joined))
        heading = None
        body = []

    for line in text.splitlines():
        stripped = line.lstrip()
        if not in_fence and _HORIZONTAL_RULE.match(stripped):
            # Drop horizontal rules so they don't pollute chunk text.
            continue
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            body.append(line)
            continue
        if not in_fence:
            m = _ATX_HEADING.match(line)
            if m:
                emit()
                heading = m.group(2).strip()
                continue
        body.append(line)

    emit()
    return sections


# --- Sentence-aware chunker ----------------------------------------------

_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")


def _chunk_text(text: str, target_size: int, overlap: int) -> list[str]:
    """Split `text` into sentence-aware chunks of ~`target_size` chars with
    ~`overlap` chars of overlap.

    A sentence boundary is `.`, `?`, or `!` followed by whitespace or the
    end of the string. The chunker walks through these boundaries, taking
    the first one at or after `target_size` chars from each chunk's start.
    Successive chunks overlap by `overlap` chars so that a query about a
    sentence near a boundary still hits the chunk that contains the next
    paragraph.
    """
    stripped = text.strip()
    if not stripped:
        return []

    ends = [m.end() for m in _SENTENCE_END.finditer(text)]
    if not ends:
        return [stripped]

    chunks: list[str] = []
    pos = 0
    boundary_idx = 0

    while pos < len(text):
        target = pos + target_size

        if target >= len(text):
            tail = text[pos:].strip()
            if tail:
                chunks.append(tail)
            break

        chosen_end: int | None = None
        for i in range(boundary_idx, len(ends)):
            if ends[i] >= target:
                chosen_end = ends[i]
                boundary_idx = i + 1
                break

        if chosen_end is None:
            tail = text[pos:].strip()
            if tail:
                chunks.append(tail)
            break

        chunk = text[pos:chosen_end].strip()
        if chunk:
            chunks.append(chunk)

        pos = chosen_end - overlap
        if pos <= 0:
            pos = chosen_end

    return chunks


def chunk_section(section: Section) -> list[Chunk]:
    """Split a section into sentence-aware chunks carrying section metadata.

    The section heading is prepended to the chunked text so the embedder
    captures the heading topic in addition to the body. The section_heading
    field on each Chunk is unchanged — citations still use the parsed
    heading — but the embedded text starts with the heading line. This
    fixes cases where the answer lives in a short heading but the body
    text is on an unrelated topic (e.g., `### Week 6 — Monday, October 26
    (mid-term)` with a body about image generation).
    """
    full_text = f"{section.section_heading}\n\n{section.text}".strip()
    texts = _chunk_text(full_text, CHUNK_TARGET_SIZE, CHUNK_OVERLAP)
    return [
        Chunk(
            source_path=section.source_path,
            section_heading=section.section_heading,
            chunk_index=i,
            text=t,
        )
        for i, t in enumerate(texts)
    ]


def load_sections() -> list[Section]:
    """Load every ATX-keyed section from the corpus."""
    out: list[Section] = []
    for path in _walk_corpus(CORPUS_ROOT):
        text = path.read_text(encoding="utf-8")
        out.extend(_parse_sections(path, text))
    return out


def load_chunks() -> list[Chunk]:
    """Load every sentence-aware chunk from the corpus."""
    out: list[Chunk] = []
    for section in load_sections():
        out.extend(chunk_section(section))
    return out


# --- Embeddings ----------------------------------------------------------

def _api_key() -> str:
    """Read the OpenRouter API key from env; fail loudly if missing."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set; export it before running."
        )
    return key


def _openrouter_embed(texts: list[str]) -> np.ndarray:
    """Call OpenRouter `/embeddings` and return a numpy matrix of shape (N, dim).

    Retries up to HTTP_RETRY_MAX times with exponential backoff (1s, 2s, 4s)
    on HTTP 429/5xx or transport errors.
    """
    url = f"{OPENROUTER_BASE_URL}/embeddings"
    headers = {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
    }
    payload: dict[str, object] = {
        "model": EMBEDDING_MODEL,
        "input": texts,
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
                sorted_data = sorted(data["data"], key=lambda x: x["index"])
                return np.array(
                    [d["embedding"] for d in sorted_data],
                    dtype=np.float32,
                )
        except httpx.HTTPError as exc:
            last_exc = exc

        # Exponential backoff.
        time.sleep(2 ** attempt)

    raise RuntimeError(
        f"OpenRouter embedding call failed after {HTTP_RETRY_MAX} attempts: {last_exc}"
    )


def _embed_all(texts: list[str]) -> np.ndarray:
    """Embed `texts` in batches and stack into one (N, dim) matrix."""
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    parts: list[np.ndarray] = []
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i:i + EMBEDDING_BATCH_SIZE]
        parts.append(_openrouter_embed(batch))
    return np.vstack(parts)


def _corpus_max_mtime() -> float:
    """Max mtime across all files in the corpus walk."""
    files = _walk_corpus(CORPUS_ROOT)
    if not files:
        return 0.0
    return max(f.stat().st_mtime for f in files)


# --- Index build / load / save -------------------------------------------

def build_index() -> Index:
    """Build embeddings for every chunk and return a fresh Index."""
    chunks = load_chunks()
    texts = [c.text for c in chunks]
    vectors = _embed_all(texts)
    return Index(
        chunks=tuple(chunks),
        embeddings=vectors,
        corpus_max_mtime=_corpus_max_mtime(),
        embedding_model=EMBEDDING_MODEL,
    )


def save_index(index: Index, path: Path = INDEX_FILE) -> None:
    """Persist an Index to JSON. Vectors are stored as nested lists."""
    data = {
        "version": INDEX_VERSION,
        "embedding_model": index.embedding_model,
        "embedding_dim": int(index.embeddings.shape[1]) if index.embeddings.size else EMBEDDING_DIM,
        "corpus_max_mtime": index.corpus_max_mtime,
        "chunks": [
            {
                "source_path": c.source_path,
                "section_heading": c.section_heading,
                "chunk_index": c.chunk_index,
                "text": c.text,
            }
            for c in index.chunks
        ],
        "embeddings": index.embeddings.tolist(),
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def load_index(path: Path = INDEX_FILE) -> Index | None:
    """Load an Index from JSON, or return None if missing/stale/incompatible.

    The cache is considered stale when:
    - the JSON is missing or unreadable,
    - `version` doesn't match INDEX_VERSION,
    - `embedding_model` or `embedding_dim` differ from current,
    - or the recorded `corpus_max_mtime` is older than the corpus's current
      max mtime.
    """
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    if data.get("version") != INDEX_VERSION:
        return None
    if data.get("embedding_model") != EMBEDDING_MODEL:
        return None
    if data.get("embedding_dim") != EMBEDDING_DIM:
        return None
    if data.get("corpus_max_mtime", 0.0) < _corpus_max_mtime():
        return None

    chunks = tuple(Chunk(**c) for c in data["chunks"])
    embeddings = np.array(data["embeddings"], dtype=np.float32)
    return Index(
        chunks=chunks,
        embeddings=embeddings,
        corpus_max_mtime=float(data["corpus_max_mtime"]),
        embedding_model=str(data["embedding_model"]),
    )


def get_index(path: Path = INDEX_FILE) -> Index:
    """Return the cached Index, building it if missing or stale."""
    index = load_index(path)
    if index is not None:
        return index
    index = build_index()
    save_index(index, path)
    return index


# --- Retriever -----------------------------------------------------------

def retrieve(
    query: str,
    index: Index | None = None,
    k: int = 5,
) -> list[tuple[Chunk, float]]:
    """Embed `query` and return the top-k chunks by raw cosine similarity.

    Each entry is `(chunk, score)`; score is the raw cosine similarity in
    roughly `[-1, 1]`. Higher is more similar.

    Baseline disagreement observation (no fix in v1): when two chunks
    disagree on a fact, the retriever surfaces both because both score
    highly on cosine similarity. The retriever does NOT weight by recency
    or file path; the LLM may report the stale answer. This is recorded
    in `CHECKS.md` as an observation, not a failure.
    """
    if index is None:
        index = get_index()

    query_vec = _openrouter_embed([query])[0]

    norms = np.linalg.norm(index.embeddings, axis=1)
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0 or np.any(norms == 0):
        scores = np.zeros(index.embeddings.shape[0], dtype=np.float32)
    else:
        scores = (index.embeddings @ query_vec) / (norms * query_norm)

    top_indices = np.argsort(scores)[::-1][:k]
    return [(index.chunks[int(i)], float(scores[int(i)])) for i in top_indices]


# --- Main ----------------------------------------------------------------

if __name__ == "__main__":
    sections = load_sections()
    files = sorted({s.source_path for s in sections})
    print(f"Loaded {len(sections)} sections from {len(files)} files.\n")

    print("Files walked:")
    for f in files:
        print(f"  {f}")

    chunks = load_chunks()
    sizes = [len(c.text) for c in chunks]
    print(f"\nLoaded {len(chunks)} chunks.")
    print(
        f"Chunk size: min={min(sizes)}, max={max(sizes)}, "
        f"mean={sum(sizes) // len(sizes)}"
    )

    bad = [
        c for c in chunks
        if c.text.rstrip() and c.text.rstrip()[-1] not in ".!?"
    ]
    print(f"Chunks not ending in sentence punctuation: {len(bad)}")

    print("\nBuilding/loading index...")
    index = get_index()
    print(f"  chunks:    {len(index.chunks)}")
    print(f"  embedding: {index.embeddings.shape}, model={index.embedding_model}")
    print(f"  corpus_max_mtime: {index.corpus_max_mtime}")
    if INDEX_FILE.exists():
        print(f"  cache file: {INDEX_FILE} ({INDEX_FILE.stat().st_size:,} bytes)")

    print("\nCosine similarity sanity check:")
    sample_vec = index.embeddings[0]
    sample_norm = float(np.linalg.norm(sample_vec))
    self_score = float(sample_vec @ sample_vec / (sample_norm * sample_norm))
    print(f"  chunk[0] vs chunk[0]: {self_score:.6f}  (expected ~1.0)")

    unrelated_vec = _openrouter_embed(["How do I cook pasta?"])[0]
    unrelated_norm = float(np.linalg.norm(unrelated_vec))
    unrelated_score = float(sample_vec @ unrelated_vec / (sample_norm * unrelated_norm))
    print(f"  chunk[0] vs 'How do I cook pasta?': {unrelated_score:.6f}  (expected low)")

    print("\nRetrieval test: 'When is the midterm?'")
    results = retrieve("When is the midterm?", index=index, k=5)
    for i, (chunk, score) in enumerate(results, 1):
        preview = chunk.text[:100].replace("\n", " ")
        print(f"  {i}. {chunk.source_path} > {chunk.section_heading!r}")
        print(f"     score={score:.4f}  ({len(chunk.text)} chars)")
        print(f"     preview: {preview!r}...")
