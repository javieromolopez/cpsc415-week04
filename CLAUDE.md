# Project conventions

<!-- The agent reads this at the start of every session. Keep it short and current.
     Graded: does it reflect how the team actually works? -->

## What this repository is
One paragraph. Link to the current `spec.md`.

## Commands
```
# build
# test
# run
# lint
```

## Conventions

- Language: Python 3.10+ (standard library + requests/httpx if needed via uv)
- Build/Package tool: uv
- Default model: minimax/minimax-m3
- File naming: snake_case (e.g., rag_index.py, rag_ask.py)

## Working rules

For an introductory lab, follow its explicitly assigned stages; the full chain below applies to major projects. Week 1 uses its own minimal repository.

- Write or update `intent/` and `spec.md` before code. Get `plan.md` approved before implementing.
- One feature per branch and pull request. Never push to `main` directly.
- Never commit `.env` or `.claude/settings.local.json`.
- This is the Week 4 introductory lab. Stages assigned: intent, spec, and plan.
  No branches or pull requests yet. Commit to main.
- Libraries and build tools are allowed (pip or uv, Gradle or Maven). Name
  each dependency in spec.md with the reason for it. A RAG framework such as
  LangChain, LangChain4j, or Spring AI is fine, but the program must print
  the chunks it retrieved and their scores for every answer.
- The documents are the course repository at ../ai-integration-course:
  syllabus.md, assignments/, and weeks/01 through weeks/03 (the course as it
  stood before this lab). Skip weeks/04. Never copy the pages into this repo.

## Common mistakes
Things the agent got wrong before and must not repeat. Add to this list as they happen.
