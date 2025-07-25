from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


class QueryRequest(BaseModel):
    """Request schema for processing a query."""
    
    query_text: str = Field(..., min_length=1, max_length=1000, description="The query text")
    max_results: int = Field(default=10, ge=1, le=100, description="Maximum number of results")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Similarity threshold")
    include_metadata: bool = Field(default=True, description="Include metadata in results")
    strategy: str = Field(default="semantic", description="Retrieval strategy")
    generate_response: bool = Field(default=True, description="Generate a response using LLM")
    system_prompt: Optional[str] = Field(None, description="Custom system prompt for generation")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query_text": "What are the main features of RAG systems?",
                "max_results": 5,
                "similarity_threshold": 0.75,
                "include_metadata": True,
                "strategy": "hybrid",
                "generate_response": True
            }
        }
    )


class RetrievedDocumentResponse(BaseModel):
    """Response schema for a retrieved document."""
    
    id: UUID
    content: str
    similarity_score: float
    document_id: UUID
    document_source: str
    chunk_index: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = ConfigDict(from_attributes=True)


class QueryResponse(BaseModel):
    """Response schema for query results."""
    
    query_id: UUID
    retrieved_documents: List[RetrievedDocumentResponse]
    generated_response: Optional[str] = None
    processing_time_seconds: float
    total_documents_searched: int
    confidence_score: float
    success: bool
    error_message: Optional[str] = None
    
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "query_id": "123e4567-e89b-12d3-a456-426614174000",
                "retrieved_documents": [
                    {
                        "id": "123e4567-e89b-12d3-a456-426614174001",
                        "content": "RAG systems combine retrieval and generation...",
                        "similarity_score": 0.89,
                        "document_id": "123e4567-e89b-12d3-a456-426614174002",
                        "document_source": "rag_overview.pdf",
                        "chunk_index": 5,
                        "metadata": {"page": 3}
                    }
                ],
                "generated_response": "RAG systems have several key features...",
                "processing_time_seconds": 1.2,
                "total_documents_searched": 50,
                "confidence_score": 0.85,
                "success": True
            }
        }
    )


class SearchOnlyRequest(BaseModel):
    """Request schema for search-only operations."""
    
    query_text: str = Field(..., min_length=1, max_length=1000)
    max_results: int = Field(default=10, ge=1, le=100)
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query_text": "machine learning techniques",
                "max_results": 5,
                "filters": {"category": "research", "year": 2024}
            }
        }
    )