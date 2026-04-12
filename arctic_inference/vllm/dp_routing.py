"""Patch vLLM to pass data_parallel_rank from OpenAI API to DP router.

vLLM's DP load balancer has no session affinity -- successive turns of a
multi-turn conversation can land on different ranks, breaking prefix caching.
This module adds a ``data_parallel_rank`` field to the OpenAI request models
and threads it through to ``AsyncLLM.generate()`` via a contextvar so that
clients can pin sessions to specific DP ranks.
"""

import contextvars
from typing import Optional

# Contextvar carries the DP rank from the serving layer into the engine.
_dp_rank_var: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "_dp_rank_var", default=None)


def apply_dp_routing_patches():
    """Wire data_parallel_rank through the OpenAI serving layer."""
    _patch_protocol()
    _patch_serving_chat()
    _patch_serving_completion()
    _patch_async_llm_generate()


# ---------------------------------------------------------------------------
# 1. Add data_parallel_rank field to Pydantic request models
# ---------------------------------------------------------------------------

def _patch_protocol():
    from pydantic.fields import FieldInfo
    from vllm.entrypoints.openai.protocol import (
        ChatCompletionRequest, CompletionRequest)

    field_info = FieldInfo(
        default=None,
        description=(
            "Pin this request to a specific data-parallel rank (0-indexed). "
            "Use for session affinity so prefix caching works across turns."))

    for cls in (ChatCompletionRequest, CompletionRequest):
        cls.model_fields["data_parallel_rank"] = field_info
        cls.__annotations__["data_parallel_rank"] = Optional[int]
        cls.model_rebuild(force=True)


# ---------------------------------------------------------------------------
# 2. Wrap serving methods to set the contextvar
# ---------------------------------------------------------------------------

def _patch_serving_chat():
    from vllm.entrypoints.openai.serving_chat import OpenAIServingChat

    _orig = OpenAIServingChat.create_chat_completion

    async def create_chat_completion(self, request, raw_request=None):
        dp_rank = getattr(request, "data_parallel_rank", None)
        token = _dp_rank_var.set(dp_rank)
        try:
            return await _orig(self, request, raw_request)
        finally:
            _dp_rank_var.reset(token)

    OpenAIServingChat.create_chat_completion = create_chat_completion


def _patch_serving_completion():
    from vllm.entrypoints.openai.serving_completion import (
        OpenAIServingCompletion)

    _orig = OpenAIServingCompletion.create_completion

    async def create_completion(self, request, raw_request=None):
        dp_rank = getattr(request, "data_parallel_rank", None)
        token = _dp_rank_var.set(dp_rank)
        try:
            return await _orig(self, request, raw_request)
        finally:
            _dp_rank_var.reset(token)

    OpenAIServingCompletion.create_completion = create_completion


# ---------------------------------------------------------------------------
# 3. Patch AsyncLLM.generate() to read contextvar as fallback
# ---------------------------------------------------------------------------

def _patch_async_llm_generate():
    from vllm.v1.engine.async_llm import AsyncLLM

    _orig_generate = AsyncLLM.generate

    async def generate(self, *args, data_parallel_rank=None, **kwargs):
        if data_parallel_rank is None:
            data_parallel_rank = _dp_rank_var.get(None)
        async for item in _orig_generate(
                self, *args,
                data_parallel_rank=data_parallel_rank, **kwargs):
            yield item

    AsyncLLM.generate = generate
