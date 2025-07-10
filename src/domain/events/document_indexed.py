from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List
from uuid import UUID


@dataclass
class DocumentIndexedEvent:
    """Domain event raised when a document is successfully indexed."""
    
    document_id: UUID
    chunk_count: int
    embedding_model: str
    processing_time_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_type": "document_indexed",
            "document_id": str(self.document_id),
            "chunk_count": self.chunk_count,
            "embedding_model": self.embedding_model,
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class DocumentChunkingCompletedEvent:
    """Domain event raised when document chunking is completed."""
    
    document_id: UUID
    chunks_created: int
    chunking_strategy: str
    chunk_size: int
    chunk_overlap: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_type": "document_chunking_completed",
            "document_id": str(self.document_id),
            "chunks_created": self.chunks_created,
            "chunking_strategy": self.chunking_strategy,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class DocumentProcessingFailedEvent:
    """Domain event raised when document processing fails."""
    
    document_id: UUID
    error_message: str
    error_type: str
    failed_at_stage: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_type": "document_processing_failed",
            "document_id": str(self.document_id),
            "error_message": self.error_message,
            "error_type": self.error_type,
            "failed_at_stage": self.failed_at_stage,
            "timestamp": self.timestamp.isoformat(),
            "retry_count": self.retry_count
        }


@dataclass
class DocumentDeletedEvent:
    """Domain event raised when a document is deleted."""
    
    document_id: UUID
    chunks_deleted: int
    embeddings_deleted: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    deleted_by: Optional[UUID] = None
    
    def to_dict(self) -> Dict[str, any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_type": "document_deleted",
            "document_id": str(self.document_id),
            "chunks_deleted": self.chunks_deleted,
            "embeddings_deleted": self.embeddings_deleted,
            "timestamp": self.timestamp.isoformat(),
            "deleted_by": str(self.deleted_by) if self.deleted_by else None
        }