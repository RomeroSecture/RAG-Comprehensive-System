from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import (
    Column, String, Text, DateTime, JSON, Integer, 
    ForeignKey, Index, Enum as SQLEnum, Float
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import relationship

from src.shared.config.database import Base


class DocumentModel(Base):
    """SQLAlchemy model for documents."""
    
    __tablename__ = "documents"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True)
    content = Column(Text, nullable=False)
    metadata = Column(JSON, default={})
    source = Column(String(512), nullable=False, index=True)
    file_type = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    language = Column(String(10))
    embedding_model = Column(String(100))
    processing_status = Column(String(50), default="pending", nullable=False, index=True)
    error_message = Column(Text)
    
    # Relationships
    chunks = relationship("DocumentChunkModel", back_populates="document", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_documents_source_status", "source", "processing_status"),
        Index("idx_documents_created_at", "created_at"),
    )


class DocumentChunkModel(Base):
    """SQLAlchemy model for document chunks."""
    
    __tablename__ = "document_chunks"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True)
    document_id = Column(PGUUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    content = Column(Text, nullable=False)
    metadata = Column(JSON, default={})
    chunk_index = Column(Integer, nullable=False)
    start_char = Column(Integer, nullable=False)
    end_char = Column(Integer, nullable=False)
    embedding_id = Column(PGUUID(as_uuid=True), ForeignKey("embeddings.id", ondelete="SET NULL"))
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    document = relationship("DocumentModel", back_populates="chunks")
    embedding = relationship("EmbeddingModel", back_populates="chunk", uselist=False)
    
    # Indexes
    __table_args__ = (
        Index("idx_chunks_document_id", "document_id"),
        Index("idx_chunks_document_index", "document_id", "chunk_index"),
        Index("idx_chunks_embedding_id", "embedding_id"),
    )


class EmbeddingModel(Base):
    """SQLAlchemy model for embeddings."""
    
    __tablename__ = "embeddings"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True)
    chunk_id = Column(PGUUID(as_uuid=True), unique=True, nullable=False)
    model = Column(String(100), nullable=False, index=True)
    dimensions = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Note: The actual vector is stored in a separate pgvector table
    # This table stores metadata about the embedding
    
    # Relationships
    chunk = relationship("DocumentChunkModel", back_populates="embedding", uselist=False)
    vector = relationship("VectorModel", back_populates="embedding", uselist=False, cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_embeddings_chunk_id", "chunk_id"),
        Index("idx_embeddings_model", "model"),
    )


class VectorModel(Base):
    """SQLAlchemy model for vector storage using pgvector."""
    
    __tablename__ = "vectors"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True)
    embedding_id = Column(PGUUID(as_uuid=True), ForeignKey("embeddings.id", ondelete="CASCADE"), unique=True, nullable=False)
    vector = Column("vector", type_=None)  # pgvector type will be added via migration
    
    # Relationships
    embedding = relationship("EmbeddingModel", back_populates="vector", uselist=False)
    
    # Note: Vector indexes (IVFFlat, HNSW) will be created via migrations


class SparseEmbeddingModel(Base):
    """SQLAlchemy model for sparse embeddings (e.g., BM25)."""
    
    __tablename__ = "sparse_embeddings"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True)
    chunk_id = Column(PGUUID(as_uuid=True), unique=True, nullable=False)
    indices = Column(JSON, nullable=False)  # List of indices
    values = Column(JSON, nullable=False)   # List of values
    vocabulary_size = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Indexes
    __table_args__ = (
        Index("idx_sparse_embeddings_chunk_id", "chunk_id"),
    )