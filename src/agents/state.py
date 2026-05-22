"""Shared state schema for the LangGraph ReAct agent."""
from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    # ── Inputs ────────────────────────────────────────────────────────────────
    query: str
    tenant_id: str
    user_id: str
    session_id: str | None

    # ── ReAct conversation ────────────────────────────────────────────────────
    # Full message history passed to Claude on every agent turn.
    # Format follows the Anthropic messages API:
    #   {"role": "user",      "content": "..."}
    #   {"role": "assistant", "content": [content_block, ...]}
    #   {"role": "user",      "content": [{"type": "tool_result", ...}]}
    messages: list[dict[str, Any]]

    # ── Pending tool dispatch ─────────────────────────────────────────────────
    # Set by agent_node when Claude picks a tool; cleared after tool_executor_node runs.
    pending_tool_name: str
    pending_tool_input: dict[str, Any]
    pending_tool_id: str        # Claude's tool_use block id — must round-trip in tool_result

    # ── Accumulated retrieval ─────────────────────────────────────────────────
    # All chunks retrieved across every search_knowledge_base call this turn.
    # Deduplicated by chunk_id. Used to build citations in the final answer.
    all_retrieved_chunks: list[dict[str, Any]]

    # ── Final outputs ─────────────────────────────────────────────────────────
    answer: str
    citations: list[dict[str, Any]]
    clarification_needed: str | None   # set when agent calls request_clarification

    # ── Control ───────────────────────────────────────────────────────────────
    is_done: bool      # True when agent calls answer_directly or request_clarification
    iteration: int     # number of agent turns taken (each turn = one Claude call)

    # ── Telemetry ─────────────────────────────────────────────────────────────
    nodes_executed: list[str]
    tokens_in: int
    tokens_out: int
    latency_ms_per_node: dict[str, int]

    # ── Cache telemetry ───────────────────────────────────────────────────────
    # Accumulated across all agent turns in a single request.
    # cache_write_tokens: tokens written to cache (turn 1 or cache miss) — billed at 1.25× input
    # cache_read_tokens:  tokens read from cache (turns 2+, cache hit)   — billed at 0.10× input
    cache_write_tokens: int
    cache_read_tokens: int
