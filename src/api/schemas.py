"""Pydantic request/response schemas."""
from typing import Any

from pydantic import BaseModel, Field


class RetrievalOptions(BaseModel):
    top_k: int = Field(default=5, ge=1, le=20)
    rerank: bool = True
    filter: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None
    stream: bool = False
    max_tokens: int = Field(default=1024, ge=1, le=4096)
    retrieval: RetrievalOptions = Field(default_factory=RetrievalOptions)


class Citation(BaseModel):
    id: str
    chunk_id: str
    document_id: str
    content: str
    score: float


class ToolCall(BaseModel):
    """Summary of a single tool call made during the agent loop."""
    tool_name: str
    queries: list[str] | None = None        # present for search_knowledge_base
    chunks_found: int | None = None         # present for search_knowledge_base


class AgentTrace(BaseModel):
    tool_calls: list[ToolCall]              # ordered list of tools the agent chose
    nodes_executed: list[str]               # raw node execution log
    total_searches: int                     # how many times search_knowledge_base was called
    iterations: int                         # total agent turns
    clarification_needed: str | None        # set if agent called request_clarification
    latency_ms_per_node: dict[str, int]


class QueryMetadata(BaseModel):
    model: str
    tokens_in: int
    tokens_out: int
    cache_write_tokens: int     # tokens written to prompt cache this request
    cache_read_tokens: int      # tokens read from prompt cache this request (the savings)
    cache_savings_usd: float    # how much cheaper cache reads were vs paying full price
    cost_usd: float             # actual cost after cache discounts applied
    latency_ms: int
    request_id: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    agent_trace: AgentTrace
    metadata: QueryMetadata


class IngestRequest(BaseModel):
    document_id: str | None = None
    title: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    document_id: str
    chunks_created: int


class DocumentSummary(BaseModel):
    document_id: str
    title: str
    chunks: int
    created_at: str


class HealthResponse(BaseModel):
    status: str
    postgres: str
    bedrock: str
    version: str
