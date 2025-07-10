from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID, uuid4


@dataclass
class Document:
    """Core document entity representing a document in the RAG system."""
    
    id: UUID = field(default_factory=uuid4)
    content: str = ""
    metadata: Dict[str, any] = field(default_factory=dict)
    source: str = ""
    file_type: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    language: Optional[str] = None
    chunk_ids: List[UUID] = field(default_factory=list)
    embedding_model: Optional[str] = None
    processing_status: str = "pending"
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if not self.content and self.processing_status != "error":
            raise ValueError("Document content cannot be empty")
        
        if self.file_type and self.file_type not in self.SUPPORTED_FILE_TYPES:
            raise ValueError(f"Unsupported file type: {self.file_type}")
    
    @property
    def SUPPORTED_FILE_TYPES(self) -> List[str]:
        return ["pdf", "docx", "txt", "markdown", "html", "json", "csv", "xlsx"]
    
    def update_metadata(self, new_metadata: Dict[str, any]) -> None:
        """Update document metadata."""
        self.metadata.update(new_metadata)
        self.updated_at = datetime.utcnow()
    
    def mark_as_processed(self) -> None:
        """Mark document as successfully processed."""
        self.processing_status = "completed"
        self.updated_at = datetime.utcnow()
    
    def mark_as_failed(self, error_message: str) -> None:
        """Mark document as failed to process."""
        self.processing_status = "error"
        self.error_message = error_message
        self.updated_at = datetime.utcnow()
    
    def add_chunk_id(self, chunk_id: UUID) -> None:
        """Add a chunk ID to the document."""
        if chunk_id not in self.chunk_ids:
            self.chunk_ids.append(chunk_id)
            self.updated_at = datetime.utcnow()


@dataclass
class DocumentChunk:
    """Represents a chunk of a document for processing and retrieval."""
    
    id: UUID = field(default_factory=uuid4)
    document_id: UUID = field(default_factory=uuid4)
    content: str = ""
    metadata: Dict[str, any] = field(default_factory=dict)
    chunk_index: int = 0
    start_char: int = 0
    end_char: int = 0
    embedding_id: Optional[UUID] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if not self.content:
            raise ValueError("Chunk content cannot be empty")
        
        if self.start_char < 0 or self.end_char < 0:
            raise ValueError("Character positions must be non-negative")
        
        if self.start_char >= self.end_char:
            raise ValueError("Start position must be less than end position")
    
    @property
    def char_count(self) -> int:
        """Get the character count of the chunk."""
        return self.end_char - self.start_char
    
    def to_dict(self) -> Dict[str, any]:
        """Convert chunk to dictionary representation."""
        return {
            "id": str(self.id),
            "document_id": str(self.document_id),
            "content": self.content,
            "metadata": self.metadata,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "embedding_id": str(self.embedding_id) if self.embedding_id else None,
            "created_at": self.created_at.isoformat()
        }