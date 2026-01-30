from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

from .types import Chunk

CandidateInput = Union[Sequence[str], Sequence[Chunk]]


def passages_to_chunks(
    passages: Sequence[str],
    metadata: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Chunk]:
    meta = list(metadata) if metadata is not None else []
    chunks: List[Chunk] = []
    for i, text in enumerate(passages):
        meta_i = meta[i] if i < len(meta) else {}
        chunks.append(
            Chunk(
                text=text,
                chunk_id=str(i),
                metadata=meta_i,
                base_rank=i,
            )
        )
    return chunks


def ensure_chunks(
    candidates: CandidateInput,
    *,
    metadata: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Chunk]:
    cand_list = list(candidates)
    if not cand_list:
        raise ValueError("candidates must not be empty")

    all_str = all(isinstance(x, str) for x in cand_list)
    all_chunk = all(isinstance(x, Chunk) for x in cand_list)

    if not (all_str or all_chunk):
        raise TypeError("candidates must be a Sequence[str] or a Sequence[Chunk] (not mixed)")

    if all_str:
        if metadata is not None and len(metadata) != len(cand_list):
            raise ValueError("metadata must be the same length as passages")
        return passages_to_chunks(cand_list, metadata)

    # all_chunk
    if metadata is not None:
        raise ValueError("metadata is only supported when candidates are passages (Sequence[str])")
    return list(cand_list)


def validate_inputs(
    query: str,
    ranker: Any,
    *,
    target_k: int,
    group_size: int,
    winners_per_group: int,
    max_rounds: int,
    concurrency_limit: Optional[int] = None,
) -> None:
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if not callable(ranker):
        raise TypeError("ranker must be callable")

    if target_k <= 0:
        raise ValueError("target_k must be > 0")
    if group_size <= 0:
        raise ValueError("group_size must be > 0")
    if group_size >= 100:
        raise ValueError("group_size must be < 100 to keep labels under 100")
    if winners_per_group <= 0:
        raise ValueError("winners_per_group must be > 0")
    if max_rounds <= 0:
        raise ValueError("max_rounds must be > 0")
    if concurrency_limit is not None and concurrency_limit <= 0:
        raise ValueError("concurrency_limit must be > 0")


def validate_permutation(order: Sequence[int], expected_len: int) -> List[int]:
    if len(order) != expected_len:
        raise ValueError(f"ranker must return {expected_len} indices")
    if any(not isinstance(i, int) for i in order):
        raise ValueError("ranker must return integer indices")

    zero_indexed = set(range(expected_len))
    if set(order) == zero_indexed:
        return list(order)

    # Many LLM prompts (including the default one in this library) use 1-indexed
    # labels like 01, 02, 03. Accept that form too and normalize it to 0-indexed
    # so the rest of the pipeline can remain zero-based.
    one_indexed = set(range(1, expected_len + 1))
    if set(order) == one_indexed:
        return [i - 1 for i in order]

    raise ValueError("ranker must return a permutation of indices 0..n-1 (or 1..n)")
