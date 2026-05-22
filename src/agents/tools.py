"""Tool definitions and executor for the agentic RAG loop.

The agent (Claude) picks tools at runtime. The executor runs them and returns
results that go back into the conversation as tool_result messages.

Tools available:
- search_knowledge_base: hybrid retrieval over the tenant's corpus
- answer_directly:       emit the final grounded answer (terminal)
- request_clarification: ask the user a question before proceeding (terminal)
"""
import json
from typing import Any

from src.retrieval.hybrid import HybridRetriever
from src.retrieval.rerank import Reranker


# ── Tool schemas (passed to Claude's tool_use API) ────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "search_knowledge_base",
        "description": (
            "Search the document knowledge base using hybrid retrieval (vector + keyword + reranking). "
            "Use this to retrieve relevant passages before answering. "
            "You can call this multiple times with different queries to gather more context — "
            "for example, once for each sub-topic in a multi-part question. "
            "Prefer specific, targeted queries over broad ones."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "One or more search queries to run in parallel. "
                        "Use multiple queries when the question has distinct sub-topics. "
                        "Example: ['AWS fault tolerance AZ redundancy', 'AWS auto scaling resilience']"
                    ),
                },
                "reasoning": {
                    "type": "string",
                    "description": (
                        "Explain why you are searching for this. "
                        "What gap in your current context are you filling?"
                    ),
                },
            },
            "required": ["queries", "reasoning"],
        },
    },
    {
        "name": "answer_directly",
        "description": (
            "Provide the final answer to the user. Call this when you have retrieved "
            "enough context to answer confidently and completely. "
            "Every factual claim MUST cite the chunk it came from using [1], [2], [3] inline format. "
            "If chunks do not support a claim, do NOT make it — say 'I don't have enough information'. "
            "Do not call this tool until you have done at least one search."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The complete answer with inline [N] citations for every factual claim.",
                },
                "reasoning": {
                    "type": "string",
                    "description": (
                        "Explain why you are confident this answer is complete and grounded. "
                        "Which chunks support your main claims?"
                    ),
                },
            },
            "required": ["answer", "reasoning"],
        },
    },
    {
        "name": "request_clarification",
        "description": (
            "Ask the user a clarifying question before proceeding. "
            "Use this ONLY when the query is genuinely ambiguous in a way that "
            "changes what you would search for — not to avoid searching. "
            "Example: 'Are you asking about AWS S3 or on-prem storage?' "
            "Do NOT use this just because the question is hard."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The clarifying question to ask the user.",
                },
                "reason": {
                    "type": "string",
                    "description": "Why this clarification is necessary to proceed.",
                },
            },
            "required": ["question", "reason"],
        },
    },
]


# ── Tool executor ─────────────────────────────────────────────────────────────

class ToolExecutor:
    """Runs the tool chosen by the agent and returns a JSON-serialisable result."""

    def __init__(self, retriever: HybridRetriever, reranker: Reranker) -> None:
        self.retriever = retriever
        self.reranker = reranker

    def execute(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tenant_id: str,
    ) -> dict[str, Any]:
        """Dispatch to the correct handler. Always returns a JSON-serialisable dict."""
        if tool_name == "search_knowledge_base":
            return self._search(tool_input["queries"], tenant_id)
        if tool_name == "answer_directly":
            # Terminal — the agent loop reads is_done from state, not here.
            return {"status": "answer_recorded"}
        if tool_name == "request_clarification":
            return {"status": "clarification_recorded"}
        return {"error": f"Unknown tool: {tool_name}"}

    # ── private ───────────────────────────────────────────────────────────────

    def _search(self, queries: list[str], tenant_id: str) -> dict[str, Any]:
        """Run hybrid retrieval for every query, deduplicate, rerank, return top-5."""
        all_chunks = []
        seen_ids: set[str] = set()

        for query in queries:
            chunks = self.retriever.retrieve(query, tenant_id, top_k=10)
            for c in chunks:
                if c.chunk_id not in seen_ids:
                    all_chunks.append(c)
                    seen_ids.add(c.chunk_id)

        if not all_chunks:
            return {"chunks": [], "total_found": 0, "note": "No matching documents found."}

        # Rerank using the first query as the anchor
        reranked = self.reranker.rerank(queries[0], all_chunks, top_k=5)

        return {
            "chunks": [
                {
                    "id": str(i + 1),
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "content": c.content,
                    "score": round(c.score, 4),
                }
                for i, c in enumerate(reranked)
            ],
            "total_found": len(reranked),
        }
