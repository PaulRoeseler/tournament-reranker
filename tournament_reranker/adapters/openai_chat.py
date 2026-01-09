from __future__ import annotations

from typing import Any, Dict, Optional

from ..parsing import parse_ranking_response_to_indices
from ..prompts import make_ranking_prompt
from ..types import AsyncRanker, Chunk, PromptBuilder, Ranker


def make_openai_chat_ranker(
    client: Any,
    *,
    model: str,
    system_prompt: Optional[str] = None,
    prompt_builder: PromptBuilder = make_ranking_prompt,
    temperature: float = 0.0,
    max_output_tokens: int = 256,
    extra_completion_kwargs: Optional[Dict[str, Any]] = None,
) -> Ranker:
    """
    Build a sync Ranker that calls an OpenAI-style Chat Completions client.

    The `client` can be `openai.OpenAI()` or anything with:
      client.chat.completions.create(...)
    """
    sys_prompt = system_prompt or "You are a strict passage reranker. Return JSON only."
    extra_kwargs = extra_completion_kwargs or {}

    def rank(query: str, group: list[Chunk]) -> list[int]:
        prompt = prompt_builder(query, group)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_output_tokens,
            **extra_kwargs,
        )
        content = _extract_message_content(completion)
        return parse_ranking_response_to_indices(content, len(group))

    return rank


def make_openai_chat_ranker_async(
    client: Any,
    *,
    model: str,
    system_prompt: Optional[str] = None,
    prompt_builder: PromptBuilder = make_ranking_prompt,
    temperature: float = 0.0,
    max_output_tokens: int = 256,
    extra_completion_kwargs: Optional[Dict[str, Any]] = None,
) -> AsyncRanker:
    """
    Build an async Ranker for AsyncOpenAI-style clients.
    """
    sys_prompt = system_prompt or "You are a strict passage reranker. Return JSON only."
    extra_kwargs = extra_completion_kwargs or {}

    async def rank(query: str, group: list[Chunk]) -> list[int]:
        prompt = prompt_builder(query, group)
        completion = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_output_tokens,
            **extra_kwargs,
        )
        content = _extract_message_content(completion)
        return parse_ranking_response_to_indices(content, len(group))

    return rank


def _extract_message_content(completion: Any) -> str:
    try:
        content = completion.choices[0].message.content
    except Exception as e:
        raise ValueError(f"Unexpected ChatCompletion shape: {type(completion)!r}") from e

    if not isinstance(content, str) or not content.strip():
        raise ValueError("ChatCompletion content is empty or not a string")

    return content

