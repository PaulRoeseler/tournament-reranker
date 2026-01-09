from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional


@dataclass(frozen=True)
class Chunk:
    text: str
    chunk_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    base_rank: int = 0


Ranker = Callable[[str, List[Chunk]], List[int]]
AsyncRanker = Callable[[str, List[Chunk]], Awaitable[List[int]]]
PromptBuilder = Callable[[str, List[Chunk]], str]
