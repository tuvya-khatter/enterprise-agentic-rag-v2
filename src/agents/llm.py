"""Bedrock LLM client — structured JSON responses, tool use, prompt caching, and token counting.

Prompt caching
──────────────
Claude caches content up to a cache_control marker and reuses it on subsequent calls.
We cache two things that are constant across every agent turn:
  1. The system prompt   (~300 tokens, resent every turn)
  2. The tool definitions (~500 tokens, resent every turn)

Cache pricing vs normal pricing:
  - Cache write:  1.25× input price  (25% premium to store in cache)
  - Cache read:   0.10× input price  (90% discount when reading from cache)

Minimum cacheable prefix: ~1,024 tokens (Sonnet) / ~2,048 tokens (Haiku).
If content is below the minimum, Bedrock silently skips caching — no error.
"""
import json
from typing import Any

import boto3
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import get_settings

settings = get_settings()


class BedrockLLM:
    # Regular pricing per 1M tokens
    PRICING = {
        "us.anthropic.claude-haiku-4-5-20251001-v1:0": {"input": 0.80, "output": 4.00},
        "us.anthropic.claude-sonnet-4-6-20250930-v1:0": {"input": 3.00, "output": 15.00},
    }

    # Cache pricing multipliers applied on top of the input price
    CACHE_WRITE_MULTIPLIER = 1.25   # costs 25% more than normal input to write to cache
    CACHE_READ_MULTIPLIER  = 0.10   # costs 90% less than normal input to read from cache

    def __init__(self) -> None:
        self.client = boto3.client("bedrock-runtime", region_name=settings.aws_region)
        self.model_id = settings.bedrock_generation_model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def invoke(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> tuple[str, int, int]:
        """Single-turn call (used for non-agentic paths). Returns (text, tokens_in, tokens_out)."""
        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            }
        )
        resp = self.client.invoke_model(
            modelId=self.model_id,
            body=body,
            contentType="application/json",
        )
        result = json.loads(resp["body"].read())
        text = result["content"][0]["text"]
        usage = result.get("usage", {})
        return text, usage.get("input_tokens", 0), usage.get("output_tokens", 0)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def invoke_with_tools(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> tuple[list[dict[str, Any]], str, int, int, int, int]:
        """Multi-turn call with tool use and prompt caching.

        Returns (content_blocks, stop_reason, tokens_in, tokens_out,
                 cache_write_tokens, cache_read_tokens).

        cache_write_tokens — tokens written to cache this call (first call or cache miss).
                             Charged at 1.25× normal input price.
        cache_read_tokens  — tokens read from cache this call (subsequent calls, cache hit).
                             Charged at 0.10× normal input price.

        Prompt caching strategy:
          - system prompt:    cache_control on the single system block
          - tool definitions: cache_control on the LAST tool (caches all tools as a prefix)
          Both are constant across all turns of the same agent session.
        """
        # ── System prompt: wrap as a content block list with cache_control ────
        # Claude caches this entire block. On turn 1 it's written to cache
        # (cache_write_tokens). On turns 2+ it's read from cache (cache_read_tokens).
        system_with_cache = [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ]

        # ── Tool definitions: add cache_control to the last tool ──────────────
        # cache_control on the last item caches the entire list up to that point.
        # All three tool definitions are written once and reused across turns.
        tools_with_cache = [dict(t) for t in tools]   # shallow copy — don't mutate the original
        if tools_with_cache:
            tools_with_cache[-1]["cache_control"] = {"type": "ephemeral"}

        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system_with_cache,
                "tools": tools_with_cache,
                "messages": messages,
            }
        )
        resp = self.client.invoke_model(
            modelId=self.model_id,
            body=body,
            contentType="application/json",
        )
        result = json.loads(resp["body"].read())
        content     = result["content"]
        stop_reason = result["stop_reason"]
        usage       = result.get("usage", {})

        return (
            content,
            stop_reason,
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
            usage.get("cache_creation_input_tokens", 0),   # tokens written to cache
            usage.get("cache_read_input_tokens", 0),        # tokens read from cache
        )

    def estimate_cost_usd(
        self,
        tokens_in: int,
        tokens_out: int,
        cache_write_tokens: int = 0,
        cache_read_tokens: int = 0,
    ) -> float:
        """Compute total cost accounting for cache read/write pricing.

        tokens_in         — regular (non-cached) input tokens, billed at full price
        tokens_out        — output tokens, always billed at full output price
        cache_write_tokens— tokens written to cache, billed at 1.25× input price
        cache_read_tokens — tokens read from cache, billed at 0.10× input price
        """
        pricing = self.PRICING.get(self.model_id, {"input": 1.0, "output": 5.0})
        input_price  = pricing["input"]
        output_price = pricing["output"]

        regular_cost     = tokens_in           * input_price  / 1_000_000
        output_cost      = tokens_out          * output_price / 1_000_000
        cache_write_cost = cache_write_tokens  * input_price  * self.CACHE_WRITE_MULTIPLIER / 1_000_000
        cache_read_cost  = cache_read_tokens   * input_price  * self.CACHE_READ_MULTIPLIER  / 1_000_000

        return regular_cost + output_cost + cache_write_cost + cache_read_cost

    def cache_savings_usd(
        self,
        cache_read_tokens: int,
    ) -> float:
        """How much cheaper the cache read was vs paying full price for those tokens."""
        pricing = self.PRICING.get(self.model_id, {"input": 1.0, "output": 5.0})
        full_price_cost  = cache_read_tokens * pricing["input"] / 1_000_000
        actual_read_cost = cache_read_tokens * pricing["input"] * self.CACHE_READ_MULTIPLIER / 1_000_000
        return full_price_cost - actual_read_cost
