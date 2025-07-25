from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


class DocumentUploadRequest(BaseModel):
    """Request schema for document upload."""
    
    file_path: str = Field(..., description="Path to the document file")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    chunking_strategy: str = Field(default="recursive", description="Chunking strategy to use")
    chunk_size: int = Field(default=1000, ge=100, le=5000, description="Size of each chunk")
    chunk_overlap: int = Field(default=200, ge=0, le=1000, description="Overlap between chunks")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "file_path": "/path/to/document.pdf",
                "metadata": {"author": "John Doe", "category": "research"},
                "chunking_strategy": "recursive",
                "chunk_size": 1000,
                "chunk_overlap": 200
            }
        }
    )


class DocumentResponse(BaseModel):
    """Response schema for document information."""
    
    id: UUID
    source: str
    file_type: str
    processing_status: str
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]
    chunk_count: int = 0
    language: Optional[str] = None
    error_message: Optional[str] = None
    
    model_config = ConfigDict(from_attributes=True)


class DocumentIngestionResponse(BaseModel):
    """Response schema for document ingestion result."""
    
    document_id: UUID
    chunks_created: int
    processing_time_seconds: float
    success: bool
    error_message: Optional[str] = None
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document_id": "123e4567-e89b-12d3-a456-426614174000",
                "chunks_created": 15,
                "processing_time_seconds": 2.5,
                "success": True,
                "error_message": None
            }
        }
    )


class DocumentListResponse(BaseModel):
    """Response schema for listing documents."""
    
    documents: List[DocumentResponse]
    total: int
    page: int = 1
    page_size: int = 10
    
    model_config = ConfigDict(from_attributes=True)