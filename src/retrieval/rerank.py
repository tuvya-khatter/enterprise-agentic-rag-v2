"""Cross-encoder reranking of retrieved chunks. Uses Cohere Rerank if key provided,
otherwise falls back to a local sentence-transformers cross-encoder."""
import httpx

from config.settings import get_settings
from src.retrieval.hybrid import RetrievedChunk

settings = get_settings()


class Reranker:
    def __init__(self) -> None:
        self.use_cohere = bool(settings.cohere_api_key)
        self._local_model = None

    def _cohere_rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        docs = [c.content for c in chunks]
        resp = httpx.post(
            "https://api.cohere.com/v1/rerank",
            headers={"Authorization": f"Bearer {settings.cohere_api_key}"},
            json={
                "model": "rerank-english-v3.0",
                "query": query,
                "documents": docs,
                "top_n": top_k,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        results = resp.json()["results"]
        reranked = []
        for r in results:
            c = chunks[r["index"]]
            c.score = float(r["relevance_score"])
            reranked.append(c)
        return reranked

    def _local_rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        # Lazy import to keep cold start fast for the Cohere path
        from sentence_transformers import CrossEncoder

        if self._local_model is None:
            self._local_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        pairs = [(query, c.content) for c in chunks]
        scores = self._local_model.predict(pairs)

        for c, s in zip(chunks, scores):
            c.score = float(s)

        return sorted(chunks, key=lambda c: c.score, reverse=True)[:top_k]

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int = 5
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        if self.use_cohere:
            return self._cohere_rerank(query, chunks, top_k)
        return self._local_rerank(query, chunks, top_k)
