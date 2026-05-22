"""Retrieval evaluation: Recall@k, MRR, NDCG on labeled question-answer pairs."""
import argparse
import json
import math
from pathlib import Path

from src.retrieval.embeddings import BedrockEmbeddings
from src.retrieval.hybrid import HybridRetriever
from src.storage.db import SessionLocal


def recall_at_k(relevant_chunk_ids: set[str], retrieved_ids: list[str], k: int) -> float:
    top_k = set(retrieved_ids[:k])
    if not relevant_chunk_ids:
        return 0.0
    return len(relevant_chunk_ids & top_k) / len(relevant_chunk_ids)


def mrr(relevant_chunk_ids: set[str], retrieved_ids: list[str]) -> float:
    for i, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_chunk_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(relevant_chunk_ids: set[str], retrieved_ids: list[str], k: int) -> float:
    dcg = 0.0
    for i, cid in enumerate(retrieved_ids[:k], start=1):
        if cid in relevant_chunk_ids:
            dcg += 1.0 / math.log2(i + 1)
    ideal_dcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(relevant_chunk_ids)) + 1))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tenant", default="default")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    db = SessionLocal()
    retriever = HybridRetriever(db, BedrockEmbeddings())

    results = {"recall_at_k": [], "mrr": [], "ndcg_at_k": []}
    with args.dataset.open() as f:
        for line in f:
            row = json.loads(line)
            retrieved = retriever.retrieve(row["query"], args.tenant, top_k=args.k)
            retrieved_ids = [c.chunk_id for c in retrieved]
            relevant = set(row.get("relevant_chunk_ids", []))
            results["recall_at_k"].append(recall_at_k(relevant, retrieved_ids, args.k))
            results["mrr"].append(mrr(relevant, retrieved_ids))
            results["ndcg_at_k"].append(ndcg_at_k(relevant, retrieved_ids, args.k))

    print(f"Recall@{args.k}: {sum(results['recall_at_k']) / len(results['recall_at_k']):.3f}")
    print(f"MRR:        {sum(results['mrr']) / len(results['mrr']):.3f}")
    print(f"NDCG@{args.k}:  {sum(results['ndcg_at_k']) / len(results['ndcg_at_k']):.3f}")

    db.close()


if __name__ == "__main__":
    main()
