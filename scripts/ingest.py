"""Bulk-ingest documents from a directory into the RAG index."""
import argparse
import sys
from pathlib import Path
from uuid import uuid4

from pypdf import PdfReader

from src.retrieval.chunking import Chunker
from src.retrieval.embeddings import BedrockEmbeddings
from src.storage.db import SessionLocal
from src.storage.models import Chunk, Document


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tenant", default="default")
    args = parser.parse_args()

    if not args.source.exists():
        print(f"Source not found: {args.source}", file=sys.stderr)
        sys.exit(1)

    db = SessionLocal()
    chunker = Chunker()
    embedder = BedrockEmbeddings()

    files = list(args.source.rglob("*.pdf")) + list(args.source.rglob("*.txt")) + list(args.source.rglob("*.md"))
    print(f"Found {len(files)} files.")

    for f in files:
        try:
            content = read_pdf(f) if f.suffix.lower() == ".pdf" else read_text(f)
            if not content.strip():
                continue

            chunks_text = chunker.split(content)
            embeddings = embedder.embed_documents(chunks_text)

            doc = Document(
                tenant_id=args.tenant,
                external_id=f.stem,
                title=f.name,
                source_uri=str(f),
                owner_id=uuid4(),  # synthetic system owner for bulk import
            )
            db.add(doc)
            db.flush()

            for pos, (chunk_text, emb) in enumerate(zip(chunks_text, embeddings)):
                db.add(
                    Chunk(
                        document_id=doc.id,
                        tenant_id=args.tenant,
                        position=pos,
                        content=chunk_text,
                        embedding=emb,
                        token_count=len(chunk_text) // 4,
                    )
                )
            db.commit()
            print(f"  ingested {f.name}: {len(chunks_text)} chunks")
        except Exception as exc:
            db.rollback()
            print(f"  failed {f.name}: {exc}", file=sys.stderr)

    db.close()
    print("Done.")


if __name__ == "__main__":
    main()
