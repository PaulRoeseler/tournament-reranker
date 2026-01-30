from __future__ import annotations

import asyncio
import math
import warnings
from typing import Dict, List, Sequence, Tuple

from .grouping import build_interleaved_groups
from .validating import validate_permutation
from .types import AsyncRanker, Chunk, Ranker


def _prepare_round(
    candidates: List[Chunk],
    group_size: int,
    winners_per_group: int,
    target_k: int,
    ensure_at_least_target: bool,
) -> Tuple[List[List[Chunk]], int]:
    groups = build_interleaved_groups(candidates, group_size)
    num_groups = len(groups)
    if num_groups <= 1:
        return groups, 0

    w = winners_per_group
    if ensure_at_least_target:
        w = max(w, math.ceil(target_k / num_groups))
    return groups, min(w, group_size)


def _select_winners(
    groups: Sequence[Sequence[Chunk]],
    orders: Sequence[Sequence[int]],
    winners_per_group: int,
) -> List[Chunk]:
    next_candidates: List[Chunk] = []
    for group, order in zip(groups, orders):
        if len(group) == 1:
            next_candidates.append(group[0])
            continue
        valid_order = validate_permutation(order, len(group))
        ranked = [group[i] for i in valid_order]
        next_candidates.extend(ranked[: min(winners_per_group, len(ranked))])
    return next_candidates


def pyramid_rerank(
    query: str,
    candidates: List[Chunk],
    ranker: Ranker,
    *,
    target_k: int,
    group_size: int,
    winners_per_group: int,
    ensure_at_least_target: bool = True,
    final_rerank: bool = True,
    max_rounds: int = 50,
) -> List[Chunk]:
    if len(candidates) <= target_k:
        if final_rerank and len(candidates) > 1:
            warnings.warn(
                (
                    f"pyramid_rerank short-circuit: got {len(candidates)} candidates "
                    f"<= target_k ({target_k}); skipping pyramid rounds"
                ),
                category=UserWarning,
                stacklevel=2,
            )
            order = validate_permutation(ranker(query, candidates), len(candidates))
            candidates = [candidates[i] for i in order]
        return candidates[:target_k]

    rounds = 0
    while len(candidates) > target_k and rounds < max_rounds:
        rounds += 1
        groups, winners_per_group_round = _prepare_round(
            candidates,
            group_size,
            winners_per_group,
            target_k,
            ensure_at_least_target,
        )
        if len(groups) <= 1:
            break

        orders = [ranker(query, g) for g in groups]
        candidates = _select_winners(groups, orders, winners_per_group_round)

        if len(candidates) <= target_k:
            break

    if final_rerank and len(candidates) > 1:
        order = validate_permutation(ranker(query, candidates), len(candidates))
        candidates = [candidates[i] for i in order]

    return candidates[:target_k]


async def pyramid_rerank_async(
    query: str,
    candidates: List[Chunk],
    ranker: AsyncRanker,
    *,
    target_k: int,
    group_size: int,
    winners_per_group: int,
    ensure_at_least_target: bool = True,
    final_rerank: bool = True,
    max_rounds: int = 50,
    concurrency_limit: int = 20,
) -> List[Chunk]:
    sem = asyncio.Semaphore(concurrency_limit)

    async def rank_group(group: List[Chunk]) -> List[int]:
        async with sem:
            return await ranker(query, group)

    if len(candidates) <= target_k:
        if final_rerank and len(candidates) > 1:
            warnings.warn(
                (
                    f"pyramid_rerank short-circuit: got {len(candidates)} candidates "
                    f"<= target_k ({target_k}); skipping pyramid rounds"
                ),
                category=UserWarning,
                stacklevel=2,
            )
            order = validate_permutation(await rank_group(candidates), len(candidates))
            candidates = [candidates[i] for i in order]
        return candidates[:target_k]

    rounds = 0
    while len(candidates) > target_k and rounds < max_rounds:
        rounds += 1
        groups, winners_per_group_round = _prepare_round(
            candidates,
            group_size,
            winners_per_group,
            target_k,
            ensure_at_least_target,
        )
        if len(groups) <= 1:
            break

        orders = await asyncio.gather(*[rank_group(g) for g in groups])
        candidates = _select_winners(groups, orders, winners_per_group_round)

        if len(candidates) <= target_k:
            break

    if final_rerank and len(candidates) > 1:
        order = validate_permutation(await rank_group(candidates), len(candidates))
        candidates = [candidates[i] for i in order]

    return candidates[:target_k]


def _finalize_rank_vector(
    *,
    eliminated: List[Tuple[int, Chunk]],
    survivors_in_order: List[Chunk],
    final_round: int,
    original_candidates: List[Chunk],
) -> List[int]:
    """
    Turn elimination data into a rank vector aligned with the original input order.
    Lower rank value means better (1 = best).
    """
    ranked: List[Tuple[int, int, Chunk]] = []

    # Survivors get the highest round number and keep their final ordering.
    for idx, chunk in enumerate(survivors_in_order):
        ranked.append((final_round, idx, chunk))

    # Eliminated candidates are sorted by the round they dropped, then base_rank.
    for round_num, chunk in eliminated:
        ranked.append((round_num, chunk.base_rank, chunk))

    ranked.sort(key=lambda t: (-t[0], t[1]))

    id_to_original_index: Dict[int, int] = {id(c): i for i, c in enumerate(original_candidates)}
    rank_vector = [0] * len(original_candidates)
    for rank_position, (_, _, chunk) in enumerate(ranked, start=1):
        orig_idx = id_to_original_index.get(id(chunk))
        if orig_idx is None:
            # Fallback to base_rank if object identity was lost for some reason.
            orig_idx = chunk.base_rank
        rank_vector[orig_idx] = rank_position

    return rank_vector


def pyramid_rank_all(
    query: str,
    candidates: List[Chunk],
    ranker: Ranker,
    *,
    target_k: int,
    group_size: int,
    winners_per_group: int,
    ensure_at_least_target: bool = True,
    final_rerank: bool = True,
    max_rounds: int = 50,
) -> List[int]:
    """
    Run tournament-style reranking but return a rank for every candidate,
    aligned to the original input order (1 = best). Ties within the same
    elimination round are broken by the candidate's base_rank (original order).
    """
    rounds = 0
    active = list(candidates)
    eliminated: List[Tuple[int, Chunk]] = []

    while len(active) > target_k and rounds < max_rounds:
        rounds += 1
        groups, winners_per_group_round = _prepare_round(
            active,
            group_size,
            winners_per_group,
            target_k,
            ensure_at_least_target,
        )
        if len(groups) <= 1:
            break

        orders = [ranker(query, g) for g in groups]
        next_candidates: List[Chunk] = []
        for group, order in zip(groups, orders):
            valid_order = validate_permutation(order, len(group))
            ranked_group = [group[i] for i in valid_order]
            winners = ranked_group[: min(winners_per_group_round, len(ranked_group))]
            losers = ranked_group[len(winners) :]
            next_candidates.extend(winners)
            eliminated.extend((rounds, c) for c in losers)

        if len(next_candidates) >= len(active):
            # No progress; avoid infinite loops.
            break
        active = next_candidates

    if final_rerank and len(active) > 1:
        order = validate_permutation(ranker(query, active), len(active))
        active = [active[i] for i in order]

    final_round = rounds + 1
    return _finalize_rank_vector(
        eliminated=eliminated,
        survivors_in_order=active,
        final_round=final_round,
        original_candidates=candidates,
    )


async def pyramid_rank_all_async(
    query: str,
    candidates: List[Chunk],
    ranker: AsyncRanker,
    *,
    target_k: int,
    group_size: int,
    winners_per_group: int,
    ensure_at_least_target: bool = True,
    final_rerank: bool = True,
    max_rounds: int = 50,
    concurrency_limit: int = 20,
) -> List[int]:
    """
    Async variant of pyramid_rank_all.
    """
    sem = asyncio.Semaphore(concurrency_limit)

    async def rank_group(group: List[Chunk]) -> List[int]:
        async with sem:
            return await ranker(query, group)

    rounds = 0
    active = list(candidates)
    eliminated: List[Tuple[int, Chunk]] = []

    while len(active) > target_k and rounds < max_rounds:
        rounds += 1
        groups, winners_per_group_round = _prepare_round(
            active,
            group_size,
            winners_per_group,
            target_k,
            ensure_at_least_target,
        )
        if len(groups) <= 1:
            break

        orders = await asyncio.gather(*[rank_group(g) for g in groups])
        next_candidates: List[Chunk] = []
        for group, order in zip(groups, orders):
            valid_order = validate_permutation(order, len(group))
            ranked_group = [group[i] for i in valid_order]
            winners = ranked_group[: min(winners_per_group_round, len(ranked_group))]
            losers = ranked_group[len(winners) :]
            next_candidates.extend(winners)
            eliminated.extend((rounds, c) for c in losers)

        if len(next_candidates) >= len(active):
            break
        active = next_candidates

    if final_rerank and len(active) > 1:
        order = validate_permutation(await rank_group(active), len(active))
        active = [active[i] for i in order]

    final_round = rounds + 1
    return _finalize_rank_vector(
        eliminated=eliminated,
        survivors_in_order=active,
        final_round=final_round,
        original_candidates=candidates,
    )
