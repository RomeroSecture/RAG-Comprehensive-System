from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import (
    Column, String, Text, DateTime, JSON, Integer, Float,
    ForeignKey, Index, Enum as SQLEnum, Boolean
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import relationship

from src.shared.config.database import Base
from src.domain.entities.query import QueryComplexity, QueryIntent, RetrievalStrategy


class QueryModel(Base):
    """SQLAlchemy model for queries."""
    
    __tablename__ = "queries"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True)
    text = Column(Text, nullable=False)
    user_id = Column(PGUUID(as_uuid=True), index=True)
    session_id = Column(PGUUID(as_uuid=True), index=True)
    metadata = Column(JSON, default={})
    intent = Column(SQLEnum(QueryIntent), nullable=True)
    complexity = Column(SQLEnum(QueryComplexity), nullable=True)
    strategy = Column(SQLEnum(RetrievalStrategy), default=RetrievalStrategy.HYBRID)
    max_results = Column(Integer, default=10)
    similarity_threshold = Column(Float, default=0.7)
    include_metadata = Column(Boolean, default=True)
    language = Column(String(10))
    filters = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    retrieval_results = relationship("RetrievalResultModel", back_populates="query", cascade="all, delete-orphan")
    enhanced_queries = relationship("EnhancedQueryModel", back_populates="original_query", uselist=False)
    
    # Indexes
    __table_args__ = (
        Index("idx_queries_user_session", "user_id", "session_id"),
        Index("idx_queries_created_at", "created_at"),
        Index("idx_queries_strategy", "strategy"),
    )


class EnhancedQueryModel(Base):
    """SQLAlchemy model for enhanced queries."""
    
    __tablename__ = "enhanced_queries"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True)
    original_query_id = Column(PGUUID(as_uuid=True), ForeignKey("queries.id", ondelete="CASCADE"), unique=True, nullable=False)
    expanded_queries = Column(JSON, default=[])  # List of expanded query texts
    rewritten_query = Column(Text)
    hypothetical_answer = Column(Text)
    entities = Column(JSON, default=[])  # List of entity dicts
    keywords = Column(JSON, default=[])  # List of keywords
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    original_query = relationship("QueryModel", back_populates="enhanced_queries")


class RetrievalResultModel(Base):
    """SQLAlchemy model for retrieval results."""
    
    __tablename__ = "retrieval_results"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True)
    query_id = Column(PGUUID(as_uuid=True), ForeignKey("queries.id", ondelete="CASCADE"), nullable=False)
    total_results = Column(Integer, default=0)
    retrieval_strategy = Column(String(50), nullable=False)
    retrieval_time_ms = Column(Float, default=0.0)
    reranking_time_ms = Column(Float)
    metadata = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    query = relationship("QueryModel", back_populates="retrieval_results")
    retrieved_documents = relationship("RetrievedDocumentModel", back_populates="retrieval_result", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_retrieval_results_query_id", "query_id"),
        Index("idx_retrieval_results_created_at", "created_at"),
    )


class RetrievedDocumentModel(Base):
    """SQLAlchemy model for retrieved documents."""
    
    __tablename__ = "retrieved_documents"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True)
    retrieval_result_id = Column(PGUUID(as_uuid=True), ForeignKey("retrieval_results.id", ondelete="CASCADE"), nullable=False)
    document_id = Column(PGUUID(as_uuid=True), nullable=False)
    chunk_id = Column(PGUUID(as_uuid=True))
    content = Column(Text, nullable=False)
    metadata = Column(JSON, default={})
    score = Column(Float, default=0.0)
    source = Column(String(512))
    retrieval_method = Column(String(50))
    rerank_score = Column(Float)
    rank_position = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    retrieval_result = relationship("RetrievalResultModel", back_populates="retrieved_documents")
    
    # Indexes
    __table_args__ = (
        Index("idx_retrieved_docs_result_id", "retrieval_result_id"),
        Index("idx_retrieved_docs_document_id", "document_id"),
        Index("idx_retrieved_docs_score", "score"),
    )


class SelfRAGResultModel(Base):
    """SQLAlchemy model for Self-RAG results."""
    
    __tablename__ = "self_rag_results"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True)
    retrieval_result_id = Column(PGUUID(as_uuid=True), ForeignKey("retrieval_results.id", ondelete="CASCADE"), unique=True, nullable=False)
    confidence_score = Column(Float, default=0.0)
    retrieval_attempts = Column(Integer, default=1)
    reformulated_queries = Column(JSON, default=[])
    retrieval_decisions = Column(JSON, default=[])
    needs_additional_retrieval = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)