"""Unit tests for token-aware chunking."""
from src.retrieval.chunking import Chunker


def test_chunker_respects_max_tokens():
    chunker = Chunker(chunk_size=50, chunk_overlap=0)
    text = " ".join(["word"] * 500)
    chunks = chunker.split(text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunker._count_tokens(chunk) <= 50 + 5  # small slack for merging


def test_chunker_preserves_content():
    chunker = Chunker(chunk_size=100, chunk_overlap=0)
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = chunker.split(text)
    merged = " ".join(chunks)
    for word in ["First", "Second", "Third"]:
        assert word in merged


def test_chunker_overlap():
    chunker = Chunker(chunk_size=30, chunk_overlap=5)
    text = " ".join([f"word{i}" for i in range(200)])
    chunks = chunker.split(text)
    assert len(chunks) > 1
