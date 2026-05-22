"""LangGraph node implementations for the ReAct agentic loop.

Architecture
────────────
  agent_node  →  [conditional]  →  tool_executor_node  →  agent_node  → ...
                      ↓ done
                     END

The agent_node calls Claude with the full message history and a tool schema.
Claude decides which tool to call next (or terminates with answer_directly).
tool_executor_node runs the chosen tool and appends the result to messages.
The loop continues until Claude calls answer_directly or request_clarification,
or until MAX_ITERATIONS is reached.

This replaces the old fixed pipeline:
  planner → retriever → synthesizer → critic → [retry or end]
"""
import json
import time
from typing import Any

import structlog

from src.agents.llm import BedrockLLM
from src.agents.state import AgentState
from src.agents.tools import TOOL_DEFINITIONS, ToolExecutor

logger = structlog.get_logger()

# Hard cap on agent turns (each turn = one Claude call + one tool execution).
# Prevents runaway loops. Increase if you add more tools / expect deeper reasoning.
MAX_ITERATIONS = 5

AGENT_SYSTEM = """You are an enterprise knowledge assistant with access to a private document knowledge base.

Your goal: answer the user's question accurately, completely, and with citations to source documents.

## Available tools

- **search_knowledge_base** — hybrid retrieval (vector + keyword) over the tenant's document corpus.
  Call this before answering. You may call it multiple times with different queries
  to build up context for complex, multi-part questions.

- **answer_directly** — emit your final answer. Every factual claim MUST cite the chunk
  it came from using inline [1], [2], [3] notation. If the chunks don't support a claim,
  do NOT make it — say "I don't have enough information."

- **request_clarification** — ask the user for more information. Use ONLY when the query
  is genuinely ambiguous in a way that changes what you would search for.
  Do not use this to avoid searching.

## Rules

1. Always call search_knowledge_base at least once before calling answer_directly.
2. For multi-part questions, call search_knowledge_base multiple times with targeted queries —
   one for each distinct sub-topic — then synthesize across all results.
3. When you have enough context, call answer_directly with grounded, cited answer.
4. Never fabricate information. If the knowledge base doesn't contain the answer, say so.
"""


# ── Agent node ────────────────────────────────────────────────────────────────

def agent_node(state: AgentState, llm: BedrockLLM) -> AgentState:
    """One turn of the ReAct loop.

    Sends the current message history to Claude with the tool schema.
    Claude responds with either:
      - A tool_use block  → we set pending_tool_* and route to tool_executor
      - answer_directly   → we extract the answer, mark is_done=True, route to END
      - request_clarification → mark is_done=True, route to END
    """
    t0 = time.perf_counter()

    # Initialise message history on the very first turn
    if not state.get("messages"):
        state["messages"] = [{"role": "user", "content": state["query"]}]

    content_blocks, stop_reason, tin, tout, cache_write, cache_read = llm.invoke_with_tools(
        system=AGENT_SYSTEM,
        messages=state["messages"],
        tools=TOOL_DEFINITIONS,
        max_tokens=1024,
    )

    # Append Claude's response to the conversation so the next turn has full context
    state["messages"].append({"role": "assistant", "content": content_blocks})

    # Find the tool_use block Claude emitted (there will be exactly one)
    tool_block = next(
        (b for b in content_blocks if isinstance(b, dict) and b.get("type") == "tool_use"),
        None,
    )

    if tool_block is None:
        # Claude returned plain text without a tool call — treat as a final answer
        text = next(
            (b["text"] for b in content_blocks if isinstance(b, dict) and b.get("type") == "text"),
            "",
        )
        logger.warning("agent_no_tool_call", stop_reason=stop_reason)
        state["answer"] = text or "I was unable to generate a response."
        state["citations"] = _build_citations(state.get("all_retrieved_chunks", []))
        state["is_done"] = True

    elif tool_block["name"] == "answer_directly":
        # Terminal tool: extract the answer and finish
        state["answer"] = tool_block["input"]["answer"]
        state["citations"] = _build_citations(state.get("all_retrieved_chunks", []))
        state["is_done"] = True
        logger.info("agent_answered", iteration=state.get("iteration", 0))

    elif tool_block["name"] == "request_clarification":
        # Terminal tool: relay the clarification question to the caller
        state["clarification_needed"] = tool_block["input"]["question"]
        state["answer"] = f"Clarification needed: {tool_block['input']['question']}"
        state["citations"] = []
        state["is_done"] = True
        logger.info("agent_clarification_requested")

    else:
        # Non-terminal tool (e.g. search_knowledge_base): hand off to tool_executor
        state["pending_tool_name"] = tool_block["name"]
        state["pending_tool_input"] = tool_block["input"]
        state["pending_tool_id"] = tool_block["id"]
        state["is_done"] = False
        logger.info(
            "agent_tool_selected",
            tool=tool_block["name"],
            iteration=state.get("iteration", 0),
        )

    state["iteration"] = state.get("iteration", 0) + 1
    _record(state, "agent", t0, tin, tout, cache_write, cache_read)
    return state


# ── Tool executor node ────────────────────────────────────────────────────────

def tool_executor_node(state: AgentState, executor: ToolExecutor) -> AgentState:
    """Run the tool Claude chose and append the result to the message history.

    The result is wrapped in a tool_result content block so Claude sees it
    on the next turn and can reason over it.
    """
    t0 = time.perf_counter()

    tool_name = state["pending_tool_name"]
    tool_input = state["pending_tool_input"]
    tool_id = state["pending_tool_id"]

    result = executor.execute(tool_name, tool_input, state["tenant_id"])

    # Accumulate retrieved chunks for citation building later.
    # Deduplicate across multiple search calls.
    if tool_name == "search_knowledge_base" and result.get("chunks"):
        existing = state.get("all_retrieved_chunks", [])
        seen_ids = {c["chunk_id"] for c in existing}
        for chunk in result["chunks"]:
            if chunk["chunk_id"] not in seen_ids:
                existing.append(chunk)
                seen_ids.add(chunk["chunk_id"])
        state["all_retrieved_chunks"] = existing

    # Append the tool result as a user-role message (Anthropic API convention)
    state["messages"].append(
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": json.dumps(result),
                }
            ],
        }
    )

    logger.info(
        "tool_executed",
        tool=tool_name,
        chunks_found=result.get("total_found", "n/a"),
    )
    _record(state, f"tool:{tool_name}", t0, 0, 0)
    return state


# ── Conditional edge ──────────────────────────────────────────────────────────

def should_continue(state: AgentState) -> str:
    """Route after agent_node.

    Returns "end" when the agent is done or has hit the iteration cap.
    Returns "execute_tool" when a non-terminal tool is pending.
    """
    if state.get("is_done", False):
        return "end"

    # Hard safety cap — force termination if the agent loops too many times
    if state.get("iteration", 0) >= MAX_ITERATIONS:
        logger.warning("agent_iteration_cap_hit", iteration=state["iteration"])
        if not state.get("answer"):
            state["answer"] = (
                "I was unable to find sufficient information to answer your question "
                "within the allowed number of search steps."
            )
            state["citations"] = _build_citations(state.get("all_retrieved_chunks", []))
        state["is_done"] = True
        return "end"

    return "execute_tool"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_citations(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert accumulated retrieved chunks into the citation format for the API response."""
    return [
        {
            "id": str(i + 1),
            "chunk_id": c["chunk_id"],
            "document_id": c["document_id"],
            "content": c["content"][:200] + ("..." if len(c["content"]) > 200 else ""),
            "score": c["score"],
        }
        for i, c in enumerate(chunks[:10])  # cap at 10 citations in the response
    ]


def _record(
    state: AgentState,
    node: str,
    t0: float,
    tin: int,
    tout: int,
    cache_write: int = 0,
    cache_read: int = 0,
) -> None:
    latency_ms = int((time.perf_counter() - t0) * 1000)
    state.setdefault("nodes_executed", []).append(node)
    state.setdefault("latency_ms_per_node", {})[node] = latency_ms
    state["tokens_in"]          = state.get("tokens_in", 0)          + tin
    state["tokens_out"]         = state.get("tokens_out", 0)         + tout
    state["cache_write_tokens"] = state.get("cache_write_tokens", 0) + cache_write
    state["cache_read_tokens"]  = state.get("cache_read_tokens", 0)  + cache_read
