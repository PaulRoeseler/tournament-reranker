from __future__ import annotations

import json
import re
from typing import List


class RankingParseError(ValueError):
    """Raised when a ranking response cannot be parsed into any indices."""


def label_for(idx: int) -> str:
    """
    Convert a zero-based index into the zero-padded label used in prompts.
    """
    if idx < 0:
        raise ValueError("label index must be non-negative")
    if idx >= 100:
        raise ValueError("label index must be < 100 to keep labels at two digits")
    return f"{idx:02d}"


def parse_ranking_response_to_indices(response_text: str, n_items: int) -> List[int]:
    """
    Parse a model response into a list of integer indices/labels, preserving order.

    Accepted formats:
      1) JSON array like ["02","01","03"] or [2,1,3]
      2) Text containing an embedded JSON array somewhere
      3) Non-JSON text: extract 1–2 digit numbers via regex in appearance order

    No normalization:
      - No 1-indexed to 0-indexed conversion
      - No bounds checks against n_items
      - No dedupe / permutation validation / fill-missing
    """
    obj = _try_load_json(response_text)
    if obj is None:
        obj = _try_load_first_json_array_substring(response_text)

    if obj is not None:
        if not isinstance(obj, list):
            raise RankingParseError(f"Expected a JSON array but got {type(obj).__name__}")
        raw_items = obj
    else:
        raw_items = _parse_numbers_via_regex(response_text)

    out: List[int] = []
    for item in raw_items:
        val = _coerce_to_int(item)
        if val is not None:
            out.append(val)

    if not out:
        preview = response_text if len(response_text) <= 300 else (response_text[:300] + "…")
        raise RankingParseError(
            "Could not extract any indices from response. "
            f"n_items={n_items}. Response preview: {preview!r}"
        )

    return out


def _try_load_json(text: str) -> object | None:
    try:
        return json.loads(text)
    except Exception:
        return None


def _try_load_first_json_array_substring(text: str) -> object | None:
    for m in re.finditer(r"\[[\s\S]*?\]", text):
        try:
            return json.loads(m.group(0))
        except Exception:
            continue
    return None


def _parse_numbers_via_regex(text: str) -> List[int]:
    return [int(s) for s in re.findall(r"(?<!\d)(\d{1,2})(?!\d)", text)]


def _coerce_to_int(item: object) -> int | None:
    if isinstance(item, int):
        return item
    if isinstance(item, str):
        s = item.strip()
        if s.isdigit():
            return int(s) 
    return None
