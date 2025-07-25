from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
import asyncio

from ..schemas.query import (
    QueryRequest,
    QueryResponse,
    SearchOnlyRequest,
    RetrievedDocumentResponse
)
from ..dependencies.services import get_query_service
from src.application.use_cases.process_query import ProcessQueryCommand


router = APIRouter(prefix="/query", tags=["Query"])


@router.post("/search", response_model=QueryResponse)
async def search_documents(
    request: QueryRequest,
    # query_service = Depends(get_query_service)
) -> QueryResponse:
    """Search for documents and optionally generate a response."""
    
    # Create command
    command = ProcessQueryCommand(
        query_text=request.query_text,
        max_results=request.max_results,
        similarity_threshold=request.similarity_threshold,
        include_metadata=request.include_metadata,
        strategy=request.strategy,
        generate_response=request.generate_response,
        system_prompt=request.system_prompt
    )
    
    # Execute query (placeholder for now)
    # result = await query_service.execute(command)
    
    # Placeholder response
    retrieved_docs = [
        RetrievedDocumentResponse(
            id=UUID("123e4567-e89b-12d3-a456-426614174001"),
            content="RAG systems combine retrieval and generation for enhanced AI responses...",
            similarity_score=0.89,
            document_id=UUID("123e4567-e89b-12d3-a456-426614174000"),
            document_source="rag_overview.pdf",
            chunk_index=5,
            metadata={"page": 3, "section": "Introduction"}
        ),
        RetrievedDocumentResponse(
            id=UUID("223e4567-e89b-12d3-a456-426614174001"),
            content="The key components of RAG include vector stores, embedding models...",
            similarity_score=0.85,
            document_id=UUID("223e4567-e89b-12d3-a456-426614174000"),
            document_source="rag_architecture.pdf",
            chunk_index=2,
            metadata={"page": 7}
        )
    ]
    
    generated_response = None
    if request.generate_response:
        generated_response = (
            "RAG (Retrieval-Augmented Generation) systems combine the strengths of "
            "retrieval-based and generation-based approaches. They first retrieve "
            "relevant documents from a knowledge base, then use this context to "
            "generate more accurate and grounded responses."
        )
    
    return QueryResponse(
        query_id=UUID("323e4567-e89b-12d3-a456-426614174000"),
        retrieved_documents=retrieved_docs,
        generated_response=generated_response,
        processing_time_seconds=1.2,
        total_documents_searched=50,
        confidence_score=0.87,
        success=True
    )


@router.post("/search-only", response_model=List[RetrievedDocumentResponse])
async def search_only(
    request: SearchOnlyRequest,
    # query_service = Depends(get_query_service)
) -> List[RetrievedDocumentResponse]:
    """Perform search without generation (faster)."""
    
    # Placeholder response
    return [
        RetrievedDocumentResponse(
            id=UUID("123e4567-e89b-12d3-a456-426614174001"),
            content="Sample retrieved content matching the query...",
            similarity_score=0.92,
            document_id=UUID("123e4567-e89b-12d3-a456-426614174000"),
            document_source="document.pdf",
            chunk_index=3,
            metadata={"relevance": "high"}
        )
    ]


@router.post("/stream")
async def search_with_streaming(
    request: QueryRequest,
    # query_service = Depends(get_query_service)
):
    """Search and stream the response (for long generations)."""
    
    async def generate_stream():
        """Generate streaming response."""
        # First, yield the retrieved documents
        yield "data: {\"type\": \"documents\", \"count\": 2}\n\n"
        
        # Simulate streaming a generated response
        response_parts = [
            "RAG systems ",
            "combine retrieval ",
            "and generation ",
            "for enhanced ",
            "AI responses. ",
            "They provide ",
            "more accurate ",
            "and grounded ",
            "outputs."
        ]
        
        for part in response_parts:
            yield f"data: {{\"type\": \"token\", \"content\": \"{part}\"}}\n\n"
            await asyncio.sleep(0.1)  # Simulate processing time
        
        yield "data: {\"type\": \"done\"}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


@router.get("/strategies")
async def get_retrieval_strategies():
    """Get available retrieval strategies."""
    return {
        "strategies": [
            {
                "name": "semantic",
                "description": "Pure semantic similarity search using embeddings"
            },
            {
                "name": "hybrid",
                "description": "Combines semantic and keyword search"
            },
            {
                "name": "graph",
                "description": "Graph-enhanced retrieval with entity relationships"
            },
            {
                "name": "self_rag",
                "description": "Self-reflective RAG with dynamic retrieval"
            }
        ]
    }