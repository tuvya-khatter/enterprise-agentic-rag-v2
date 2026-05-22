"""Evaluation metric correctness."""
from src.evals.run_retrieval_eval import mrr, ndcg_at_k, recall_at_k


def test_recall_at_k_perfect():
    assert recall_at_k({"a", "b"}, ["a", "b", "c"], k=3) == 1.0


def test_recall_at_k_zero():
    assert recall_at_k({"a", "b"}, ["x", "y", "z"], k=3) == 0.0


def test_mrr_first_position():
    assert mrr({"a"}, ["a", "b", "c"]) == 1.0


def test_mrr_second_position():
    assert mrr({"b"}, ["a", "b", "c"]) == 0.5


def test_ndcg_at_k_perfect():
    # When the single relevant doc is at position 1, NDCG = 1.0
    assert abs(ndcg_at_k({"a"}, ["a"], k=1) - 1.0) < 1e-6


def test_ndcg_at_k_decays_with_position():
    top_1 = ndcg_at_k({"a"}, ["a", "b"], k=5)
    top_3 = ndcg_at_k({"a"}, ["x", "y", "a"], k=5)
    assert top_1 > top_3
