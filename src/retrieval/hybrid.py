"""Hybrid retrieval: dense (pgvector) + sparse (BM25) with Reciprocal Rank Fusion."""
from dataclasses import dataclass
from typing import Any

from rank_bm25 import BM25Okapi
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.retrieval.embeddings import BedrockEmbeddings


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: dict[str, Any]


class HybridRetriever:
    """
    Hybrid search over a per-tenant corpus:
      1. Dense retrieval via pgvector cosine similarity
      2. Sparse retrieval via BM25 over the same tenant's chunks
      3. Reciprocal Rank Fusion to combine

    The sparse BM25 index is built lazily on first query per tenant and cached.
    """

    RRF_K = 60  # standard RRF constant

    def __init__(self, db: Session, embedder: BedrockEmbeddings) -> None:
        self.db = db
        self.embedder = embedder
        self._bm25_cache: dict[str, tuple[BM25Okapi, list[dict[str, Any]]]] = {}

    def _get_bm25_for_tenant(self, tenant_id: str) -> tuple[BM25Okapi, list[dict[str, Any]]]:
        if tenant_id in self._bm25_cache:
            return self._bm25_cache[tenant_id]

        rows = self.db.execute(
            text(
                "SELECT id, document_id, content, chunk_metadata "
                "FROM chunks WHERE tenant_id = :t"
            ),
            {"t": tenant_id},
        ).mappings().all()

        corpus = [r["content"].lower().split() for r in rows]
        if not corpus:
            bm25 = BM25Okapi([[""]])
        else:
            bm25 = BM25Okapi(corpus)
        self._bm25_cache[tenant_id] = (bm25, [dict(r) for r in rows])
        return self._bm25_cache[tenant_id]

    def _dense_search(
        self, query: str, tenant_id: str, top_k: int, filter_doc_ids: list[str] | None = None
    ) -> list[RetrievedChunk]:
        q_embedding = self.embedder.embed_query(query)
        sql = text(
            """
            SELECT
                id::text AS chunk_id,
                document_id::text AS document_id,
                content,
                chunk_metadata,
                1 - (embedding <=> CAST(:q AS vector)) AS score
            FROM chunks
            WHERE tenant_id = :t
              AND (:doc_filter IS NULL OR document_id::text = ANY(:doc_filter))
            ORDER BY embedding <=> CAST(:q AS vector)
            LIMIT :k
            """
        )
        rows = self.db.execute(
            sql,
            {"q": str(q_embedding), "t": tenant_id, "doc_filter": filter_doc_ids, "k": top_k},
        ).mappings().all()
        return [
            RetrievedChunk(
                chunk_id=r["chunk_id"],
                document_id=r["document_id"],
                content=r["content"],
                score=float(r["score"]),
                metadata=r["chunk_metadata"] or {},
            )
            for r in rows
        ]

    def _sparse_search(
        self, query: str, tenant_id: str, top_k: int
    ) -> list[RetrievedChunk]:
        bm25, rows = self._get_bm25_for_tenant(tenant_id)
        if not rows:
            return []
        scores = bm25.get_scores(query.lower().split())
        ranked = sorted(zip(rows, scores), key=lambda p: p[1], reverse=True)[:top_k]
        return [
            RetrievedChunk(
                chunk_id=str(r["id"]),
                document_id=str(r["document_id"]),
                content=r["content"],
                score=float(score),
                metadata=r["chunk_metadata"] or {},
            )
            for r, score in ranked
            if score > 0
        ]

    def _reciprocal_rank_fusion(
        self, dense: list[RetrievedChunk], sparse: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        """Combine rankings using RRF formula: score = sum(1 / (k + rank))."""
        scores: dict[str, float] = {}
        chunk_lookup: dict[str, RetrievedChunk] = {}

        for rank, chunk in enumerate(dense, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1 / (self.RRF_K + rank)
            chunk_lookup[chunk.chunk_id] = chunk

        for rank, chunk in enumerate(sparse, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1 / (self.RRF_K + rank)
            chunk_lookup.setdefault(chunk.chunk_id, chunk)

        fused = []
        for chunk_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            c = chunk_lookup[chunk_id]
            c.score = score
            fused.append(c)
        return fused

    def retrieve(
        self,
        query: str,
        tenant_id: str,
        top_k: int = 10,
        dense_k: int = 25,
        sparse_k: int = 25,
        filter_doc_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        dense = self._dense_search(query, tenant_id, dense_k, filter_doc_ids)
        sparse = self._sparse_search(query, tenant_id, sparse_k)
        return self._reciprocal_rank_fusion(dense, sparse, top_k)
