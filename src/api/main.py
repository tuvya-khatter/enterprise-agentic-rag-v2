"""FastAPI app entrypoint."""
import time
import uuid
from contextlib import asynccontextmanager

import boto3
import structlog
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import Depends, FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import text
from sqlalchemy.orm import Session

from config.settings import get_settings
from src import __version__
from src.agents.graph import build_agent
from src.agents.llm import BedrockLLM
from src.api.schemas import (
    AgentTrace,
    Citation,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryMetadata,
    QueryRequest,
    QueryResponse,
    ToolCall,
)
from src.auth.jwt import get_current_user
from src.observability.telemetry import (
    COST_COUNTER,
    QUERY_COUNTER,
    QUERY_LATENCY,
    TOKEN_COUNTER,
    setup_logging,
    setup_tracing,
)
from src.retrieval.chunking import Chunker
from src.retrieval.embeddings import BedrockEmbeddings
from src.storage.db import get_db
from src.storage.models import Chunk, Document, QueryLog

logger = structlog.get_logger()
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    setup_tracing(app)
    logger.info("startup", version=__version__, env=settings.app_env)
    yield
    logger.info("shutdown")


app = FastAPI(
    title="Enterprise Agentic RAG",
    version=__version__,
    description="Production-grade agentic RAG system with hybrid retrieval, citations, and observability.",
    lifespan=lifespan,
)


@app.post("/v1/query", response_model=QueryResponse)
def query_endpoint(
    req: QueryRequest,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> QueryResponse:
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    t_start = time.perf_counter()

    with structlog.contextvars.bound_contextvars(request_id=request_id, tenant_id=user["tenant_id"]):
        logger.info("query_started", query_length=len(req.query))

        try:
            agent = build_agent(db)
            initial_state = {
                "query": req.query,
                "tenant_id": user["tenant_id"],
                "user_id": user["user_id"],
                "session_id": req.session_id,
                "iteration": 0,
            }
            final_state = agent.invoke(initial_state)
        except Exception as exc:
            QUERY_COUNTER.labels(tenant_id=user["tenant_id"], status="error").inc()
            logger.exception("agent_failed", error=str(exc))
            raise HTTPException(status_code=500, detail="Agent execution failed") from exc

        latency_ms = int((time.perf_counter() - t_start) * 1000)
        llm = BedrockLLM()

        cache_write_tokens = final_state.get("cache_write_tokens", 0)
        cache_read_tokens  = final_state.get("cache_read_tokens", 0)

        cost = llm.estimate_cost_usd(
            tokens_in          = final_state.get("tokens_in", 0),
            tokens_out         = final_state.get("tokens_out", 0),
            cache_write_tokens = cache_write_tokens,
            cache_read_tokens  = cache_read_tokens,
        )
        savings = llm.cache_savings_usd(cache_read_tokens)

        QUERY_COUNTER.labels(tenant_id=user["tenant_id"], status="success").inc()
        QUERY_LATENCY.labels(tenant_id=user["tenant_id"]).observe(latency_ms / 1000.0)
        TOKEN_COUNTER.labels(tenant_id=user["tenant_id"], direction="in").inc(
            final_state.get("tokens_in", 0)
        )
        TOKEN_COUNTER.labels(tenant_id=user["tenant_id"], direction="out").inc(
            final_state.get("tokens_out", 0)
        )
        TOKEN_COUNTER.labels(tenant_id=user["tenant_id"], direction="cache_write").inc(
            cache_write_tokens
        )
        TOKEN_COUNTER.labels(tenant_id=user["tenant_id"], direction="cache_read").inc(
            cache_read_tokens
        )
        COST_COUNTER.labels(tenant_id=user["tenant_id"]).inc(cost)

        # Build a structured summary of every tool call the agent made
        tool_calls: list[ToolCall] = []
        total_searches = 0
        for msg in final_state.get("messages", []):
            if msg.get("role") != "assistant":
                continue
            for block in msg.get("content", []):
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                name = block["name"]
                inp = block.get("input", {})
                if name == "search_knowledge_base":
                    total_searches += 1
                    tool_calls.append(ToolCall(
                        tool_name=name,
                        queries=inp.get("queries", []),
                        chunks_found=None,  # would need to correlate with tool_results
                    ))
                elif name in ("answer_directly", "request_clarification"):
                    tool_calls.append(ToolCall(tool_name=name))

        response = QueryResponse(
            answer=final_state.get("answer", ""),
            citations=[Citation(**c) for c in final_state.get("citations", [])],
            agent_trace=AgentTrace(
                tool_calls=tool_calls,
                nodes_executed=final_state.get("nodes_executed", []),
                total_searches=total_searches,
                iterations=final_state.get("iteration", 1),
                clarification_needed=final_state.get("clarification_needed"),
                latency_ms_per_node=final_state.get("latency_ms_per_node", {}),
            ),
            metadata=QueryMetadata(
                model=settings.bedrock_generation_model,
                tokens_in=final_state.get("tokens_in", 0),
                tokens_out=final_state.get("tokens_out", 0),
                cache_write_tokens=cache_write_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_savings_usd=round(savings, 6),
                cost_usd=round(cost, 6),
                latency_ms=latency_ms,
                request_id=request_id,
            ),
        )

        log_row = QueryLog(
            user_id=user["user_id"],
            tenant_id=user["tenant_id"],
            query=req.query,
            answer=response.answer,
            citations=[c.model_dump() for c in response.citations],
            agent_trace=response.agent_trace.model_dump(),
            tokens_in=response.metadata.tokens_in,
            tokens_out=response.metadata.tokens_out,
            cost_usd=cost,
            latency_ms=latency_ms,
            request_id=request_id,
            critic_passed=final_state.get("clarification_needed") is None,  # proxy: True if agent answered
        )
        db.add(log_row)
        db.commit()

        logger.info(
            "query_completed",
            latency_ms=latency_ms,
            tokens_in=response.metadata.tokens_in,
            tokens_out=response.metadata.tokens_out,
            cache_write_tokens=cache_write_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_savings_usd=round(savings, 6),
            cost_usd=round(cost, 6),
            total_searches=response.agent_trace.total_searches,
            iterations=response.agent_trace.iterations,
            clarification_needed=response.agent_trace.clarification_needed is not None,
        )
        return response


@app.post("/v1/ingest", response_model=IngestResponse)
def ingest_endpoint(
    req: IngestRequest,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> IngestResponse:
    chunker = Chunker()
    embedder = BedrockEmbeddings()
    chunks_text = chunker.split(req.content)
    embeddings = embedder.embed_documents(chunks_text)

    doc = Document(
        tenant_id=user["tenant_id"],
        external_id=req.document_id,
        title=req.title,
        owner_id=user["user_id"],
        doc_metadata=req.metadata,
    )
    db.add(doc)
    db.flush()

    for pos, (chunk_text, emb) in enumerate(zip(chunks_text, embeddings)):
        db.add(
            Chunk(
                document_id=doc.id,
                tenant_id=user["tenant_id"],
                position=pos,
                content=chunk_text,
                embedding=emb,
                token_count=len(chunk_text) // 4,  # rough token estimate
            )
        )
    db.commit()

    logger.info("document_ingested", doc_id=str(doc.id), chunks=len(chunks_text))
    return IngestResponse(document_id=str(doc.id), chunks_created=len(chunks_text))


@app.get("/health", response_model=HealthResponse)
def health(db: Session = Depends(get_db)) -> HealthResponse:
    pg_status = "up"
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        pg_status = "down"

    bedrock_status = "up"
    try:
        boto3.client("bedrock-runtime", region_name=settings.aws_region)
    except (BotoCoreError, ClientError):
        bedrock_status = "down"

    overall = "healthy" if pg_status == "up" and bedrock_status == "up" else "degraded"
    return HealthResponse(
        status=overall,
        postgres=pg_status,
        bedrock=bedrock_status,
        version=__version__,
    )


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
