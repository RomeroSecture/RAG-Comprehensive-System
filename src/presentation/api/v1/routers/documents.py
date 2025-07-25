from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File
from fastapi.responses import JSONResponse

from ..schemas.document import (
    DocumentUploadRequest,
    DocumentResponse,
    DocumentIngestionResponse,
    DocumentListResponse
)
from ..dependencies.services import get_document_service, get_ingestion_service
from src.application.use_cases.ingest_document import IngestDocumentCommand


router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/ingest", response_model=DocumentIngestionResponse)
async def ingest_document(
    request: DocumentUploadRequest,
    # ingestion_service = Depends(get_ingestion_service)
) -> DocumentIngestionResponse:
    """Ingest a document into the RAG system."""
    
    # Create ingestion command
    command = IngestDocumentCommand(
        file_path=request.file_path,
        metadata=request.metadata,
        chunking_strategy=request.chunking_strategy,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap
    )
    
    # Execute ingestion (placeholder for now)
    # result = await ingestion_service.execute(command)
    
    # Placeholder response
    return DocumentIngestionResponse(
        document_id=UUID("123e4567-e89b-12d3-a456-426614174000"),
        chunks_created=15,
        processing_time_seconds=2.5,
        success=True
    )


@router.post("/upload", response_model=DocumentIngestionResponse)
async def upload_document(
    file: UploadFile = File(...),
    chunking_strategy: str = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    # ingestion_service = Depends(get_ingestion_service)
) -> DocumentIngestionResponse:
    """Upload and ingest a document file."""
    
    # Save uploaded file temporarily
    # In production, this would save to a proper storage service
    temp_path = f"/tmp/{file.filename}"
    
    try:
        # Save file
        contents = await file.read()
        with open(temp_path, "wb") as f:
            f.write(contents)
        
        # Create ingestion command
        command = IngestDocumentCommand(
            file_path=temp_path,
            metadata={"filename": file.filename, "content_type": file.content_type},
            chunking_strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Execute ingestion (placeholder)
        # result = await ingestion_service.execute(command)
        
        # Placeholder response
        return DocumentIngestionResponse(
            document_id=UUID("123e4567-e89b-12d3-a456-426614174000"),
            chunks_created=20,
            processing_time_seconds=3.2,
            success=True
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: UUID,
    # document_service = Depends(get_document_service)
) -> DocumentResponse:
    """Get a specific document by ID."""
    
    # Placeholder response
    return DocumentResponse(
        id=document_id,
        source="/path/to/document.pdf",
        file_type="pdf",
        processing_status="completed",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        metadata={"pages": 10},
        chunk_count=15,
        language="en"
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page"),
    status: Optional[str] = Query(None, description="Filter by processing status"),
    # document_service = Depends(get_document_service)
) -> DocumentListResponse:
    """List all documents with pagination."""
    
    # Placeholder response
    documents = [
        DocumentResponse(
            id=UUID("123e4567-e89b-12d3-a456-426614174000"),
            source="/path/to/doc1.pdf",
            file_type="pdf",
            processing_status="completed",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            metadata={},
            chunk_count=15
        ),
        DocumentResponse(
            id=UUID("223e4567-e89b-12d3-a456-426614174000"),
            source="/path/to/doc2.txt",
            file_type="txt",
            processing_status="completed",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            metadata={},
            chunk_count=8
        )
    ]
    
    return DocumentListResponse(
        documents=documents,
        total=2,
        page=page,
        page_size=page_size
    )


@router.delete("/{document_id}")
async def delete_document(
    document_id: UUID,
    # document_service = Depends(get_document_service)
) -> JSONResponse:
    """Delete a document and all its chunks."""
    
    # Placeholder implementation
    # success = await document_service.delete(document_id)
    
    return JSONResponse(
        content={
            "message": f"Document {document_id} deleted successfully",
            "document_id": str(document_id)
        },
        status_code=200
    )


# Import datetime for the placeholder responses
from datetime import datetime