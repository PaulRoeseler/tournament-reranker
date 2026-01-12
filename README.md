# Tournament Reranker

An old idea I had lying in the graveyard of my projects — finally dug up and turned into a small, dependency-light Python library for LLM-based reranking using a tournament / pyramid strategy.

You give it a *query* (or any “target context”), and a list of candidate texts (passages, documents, CVs, product descriptions, etc.). It runs a series of small “mini-rankings” and returns a **top-k** list.

![Tournament / Pyramid reranking process](pyramid_llm_tournament_tree.gif)

## How the tournament/pyramid works

* Candidates are **interleaved** into groups so early rounds mix “high” and “low” base ranks.
* Each round independently ranks small groups and keeps the top `winners_per_group`

  * winners are automatically adjusted to avoid dropping below `target_k`
* The process repeats until the pool is small, then an optional **final rerank** produces a clean top-k list.

Key knobs live on `pyramid_rerank` / `pyramid_rerank_async`:

* `group_size`
* `winners_per_group`
* `target_k`
* `final_rerank`
* `max_rounds`

> Note: `group_size` must stay under 100 because passage labels are zero-padded (`01`..`99`).


## Installation

This repo ships with a `pyproject.toml`. There are **no required runtime dependencies**.

If you want the built-in OpenAI adapter, install with the optional extra:

```bash
pip install -e ".[openai]"
```

Or using `uv`:

```bash
uv sync
```

> If you don’t need the OpenAI adapter, you can skip installing `openai` and provide your own ranker function.

## Quickstart (sync)

```python
from openai import OpenAI
from tournament_reranker import make_openai_chat_ranker, rerank_passages

client = OpenAI()  # needs OPENAI_API_KEY in env
ranker = make_openai_chat_ranker(client, model="gpt-4o-mini")

passages = [
    "Paris is the capital of France.",
    "The Eiffel Tower is in Paris.",
    "Toronto is the capital of Ontario.",
]

top_chunks = rerank_passages(
    query="Where is the Eiffel Tower?",
    passages=passages,
    ranker=ranker,
    target_k=2,
)

for c in top_chunks:
    print(c.text, c.metadata, c.base_rank)
```

## Quickstart (async)

```python
import asyncio
from openai import AsyncOpenAI
from tournament_reranker import make_openai_chat_ranker_async, rerank_passages_async

async def main():
    client = AsyncOpenAI()
    ranker = make_openai_chat_ranker_async(client, model="gpt-4o-mini")
    passages = ["Doc A", "Doc B", "Doc C"]
    ranked = await rerank_passages_async("Pick the best doc", passages, ranker, target_k=2)
    print([c.text for c in ranked])

asyncio.run(main())
```

## More context-dependent example: ranking CVs for a job posting

```python
from openai import OpenAI
from tournament_reranker import make_openai_chat_ranker, rerank_passages

job_posting = """
Senior Backend Engineer (Python)
Must-have:
- 5+ years backend experience
- Strong Python + FastAPI
- Postgres + production SQL
Nice-to-have:
- Kubernetes
- Event-driven systems
"""

cvs = [
    "Candidate A: 7 years Python, built FastAPI services, Postgres tuning, ...",
    "Candidate B: 10 years Java, some Python scripting, ...",
    "Candidate C: 6 years Python, Django, some FastAPI, heavy Kubernetes, ...",
]

metadata = [
    {"candidate_id": "A", "source": "inbox/123"},
    {"candidate_id": "B", "source": "inbox/456"},
    {"candidate_id": "C", "source": "inbox/789"},
]

client = OpenAI()
ranker = make_openai_chat_ranker(client, model="gpt-4o-mini")

top = rerank_passages(
    query=f"Rank these CVs for the following job:\n\n{job_posting}\n\nPrefer must-haves over nice-to-haves.",
    passages=cvs,
    metadata=metadata,
    ranker=ranker,
    target_k=2,
)

for c in top:
    print(c.metadata["candidate_id"], c.text[:80])
```


## What you pass in

* `query`: the target context (question, rubric, job posting, policy, spec, etc.)
* `passages`: ordered list of candidate strings (from retrieval or any candidate pool)
* Optional: `metadata` (same length as `passages`) to carry ids, sources, URLs, scores, etc.

The reranker returns `Chunk` objects with:

* `text`
* `chunk_id`
* `metadata`
* `base_rank` (original order index, useful for audit/tie-breaking)

## Plugging in a different LLM/provider

The library is intentionally simple: you can bring your own model call. A `Ranker` is:

* input: `(query: str, group: list[Chunk])`
* output: a **permutation** of `0..n-1` (best → worst)

The default prompt numbers passages `01, 02, 03, ...` (up to 99 per group) and expects a JSON array of those labels, best to worst.

Example:

```python
from tournament_reranker import Chunk, make_ranking_prompt, parse_ranking_response_to_indices

def my_ranker(query: str, group: list[Chunk]) -> list[int]:
    prompt = make_ranking_prompt(query, group)
    model_output_text = call_your_llm(prompt)  # you implement this
    return parse_ranking_response_to_indices(model_output_text, len(group))
```

If the ranker output is missing indices or invalid, the reranker raises a `ValueError`.

