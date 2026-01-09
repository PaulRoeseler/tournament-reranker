from __future__ import annotations

import math
from typing import List, Sequence

from .types import Chunk


def build_interleaved_groups(
    candidates: Sequence[Chunk],
    group_size: int,
) -> List[List[Chunk]]:
    n = len(candidates)
    if n == 0:
        return []

    num_groups = math.ceil(n / group_size)
    groups: List[List[Chunk]] = []

    for i in range(num_groups):
        g: List[Chunk] = []
        for j in range(group_size):
            idx = i + j * num_groups
            if idx < n:
                g.append(candidates[idx])
        if g:
            groups.append(g)

    return groups
