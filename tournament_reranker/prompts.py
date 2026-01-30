from __future__ import annotations

from typing import List

from .parsing import label_for
from .types import Chunk


def make_ranking_prompt(query: str, group: List[Chunk]) -> str:
    """
    Enforces a JSON-only response for easy parsing.
    """
    items = []
    for i, c in enumerate(group):
        lab = label_for(i)
        items.append(f"{lab}. {c.text}")
    joined = "\n\n".join(items)

    return (
        "You are a strict passage reranker.\n"
        "Rank the passages by usefulness for answering the user's query.\n\n"
        "Output format (mandatory):\n"
        "- Return ONLY a JSON array of the passage numbers, best to worst.\n"
        "- Labels are 0-indexed and zero-padded: 00, 01, 02, ...\n"
        "- No prose, no markdown, no extra keys.\n"
        '- Example: ["01","00","02"]\n\n'
        f"Query:\n{query}\n\n"
        f"Passages:\n{joined}\n"
    )
