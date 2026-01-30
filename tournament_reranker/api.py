from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .validating import ensure_chunks, validate_inputs
from .pyramid import pyramid_rank_all, pyramid_rank_all_async, pyramid_rerank, pyramid_rerank_async
from .types import AsyncRanker, Chunk, Ranker


def rerank_passages(
    query: str,
    passages: Sequence[str],
    ranker: Ranker,
    *,
    target_k: int = 5,
    group_size: int = 5,
    winners_per_group: int = 2,
    metadata: Optional[Sequence[Dict[str, Any]]] = None,
    final_rerank: bool = True,
    max_rounds: int = 50,
) -> List[int]:
    validate_inputs(
        query,
        ranker,
        target_k=target_k,
        group_size=group_size,
        winners_per_group=winners_per_group,
        max_rounds=max_rounds,
    )
    chunks = ensure_chunks(passages, metadata=metadata)

    return pyramid_rank_all(
        query,
        chunks,
        ranker,
        target_k=target_k,
        group_size=group_size,
        winners_per_group=winners_per_group,
        final_rerank=final_rerank,
        max_rounds=max_rounds,
    )


async def rerank_passages_async(
    query: str,
    passages: Sequence[str],
    ranker: AsyncRanker,
    *,
    target_k: int = 5,
    group_size: int = 5,
    winners_per_group: int = 2,
    metadata: Optional[Sequence[Dict[str, Any]]] = None,
    final_rerank: bool = True,
    max_rounds: int = 50,
    concurrency_limit: int = 20,
) -> List[int]:
    validate_inputs(
        query,
        ranker,
        target_k=target_k,
        group_size=group_size,
        winners_per_group=winners_per_group,
        max_rounds=max_rounds,
        concurrency_limit=concurrency_limit,
    )
    chunks = ensure_chunks(passages, metadata=metadata)

    return await pyramid_rank_all_async(
        query,
        chunks,
        ranker,
        target_k=target_k,
        group_size=group_size,
        winners_per_group=winners_per_group,
        final_rerank=final_rerank,
        max_rounds=max_rounds,
        concurrency_limit=concurrency_limit,
    )
