"""SQLAlchemy models for documents, chunks, users, and query logs."""
from datetime import datetime
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    tenant_id = Column(String(64), nullable=False, index=True)
    role = Column(String(32), nullable=False, default="user")  # user, admin
    api_key_hash = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id = Column(String(64), nullable=False, index=True)
    external_id = Column(String(255), index=True)  # caller-supplied doc id
    title = Column(String(512), nullable=False)
    source_uri = Column(String(1024))
    doc_metadata = Column(JSON, default=dict)
    visibility = Column(String(32), default="tenant")  # public, tenant, private
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False, index=True)
    tenant_id = Column(String(64), nullable=False, index=True)
    position = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    # Titan v2 embedding dimension is 1024
    embedding = Column(Vector(1024))
    token_count = Column(Integer)
    chunk_metadata = Column(JSON, default=dict)

    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index(
            "ix_chunks_embedding_cosine",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_chunks_tenant_document", "tenant_id", "document_id"),
    )


class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    tenant_id = Column(String(64), nullable=False, index=True)
    query = Column(Text, nullable=False)
    answer = Column(Text)
    citations = Column(JSON)
    agent_trace = Column(JSON)
    tokens_in = Column(Integer)
    tokens_out = Column(Integer)
    cost_usd = Column(Float)
    latency_ms = Column(Integer)
    request_id = Column(String(64), index=True)
    critic_passed = Column(Boolean)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class DocumentACL(Base):
    """Explicit per-user document shares, in addition to tenant-wide visibility."""
    __tablename__ = "document_acls"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    granted_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("ix_acls_doc_user", "document_id", "user_id", unique=True),)
