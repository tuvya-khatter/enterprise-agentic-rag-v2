"""Unit tests for Reciprocal Rank Fusion logic."""
from unittest.mock import MagicMock

from src.retrieval.hybrid import HybridRetriever, RetrievedChunk


def _chunk(cid: str, content: str = "") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, document_id="d1", content=content or cid, score=1.0, metadata={}
    )


def test_rrf_rewards_top_ranked_in_both():
    retriever = HybridRetriever(db=MagicMock(), embedder=MagicMock())
    dense = [_chunk("a"), _chunk("b"), _chunk("c")]
    sparse = [_chunk("b"), _chunk("d"), _chunk("a")]
    fused = retriever._reciprocal_rank_fusion(dense, sparse, top_k=3)
    assert fused[0].chunk_id in {"a", "b"}
    assert len(fused) == 3


def test_rrf_handles_no_overlap():
    retriever = HybridRetriever(db=MagicMock(), embedder=MagicMock())
    dense = [_chunk("a")]
    sparse = [_chunk("b")]
    fused = retriever._reciprocal_rank_fusion(dense, sparse, top_k=5)
    assert {c.chunk_id for c in fused} == {"a", "b"}


def test_rrf_empty_sparse():
    retriever = HybridRetriever(db=MagicMock(), embedder=MagicMock())
    dense = [_chunk("a"), _chunk("b")]
    sparse = []
    fused = retriever._reciprocal_rank_fusion(dense, sparse, top_k=5)
    assert [c.chunk_id for c in fused] == ["a", "b"]
