from typing import Dict, List, Optional
from uuid import UUID
import asyncio
from dataclasses import dataclass

from src.domain.entities.query import Query
from src.domain.entities.retrieval_result import RetrievalResult, RetrievedDocument
from src.domain.repositories.document_repository import DocumentRepository
from src.domain.repositories.vector_store_repository import VectorStoreRepository
from src.domain.services.ranking_service import RankingService
from src.domain.value_objects.embedding import Embedding
from src.application.ports.embedding_service import EmbeddingService
from src.application.ports.generation_service import GenerationService


@dataclass
class ProcessQueryCommand:
    """Command for processing a query in the RAG system."""
    query_text: str
    max_results: int = 10
    similarity_threshold: float = 0.7
    include_metadata: bool = True
    strategy: str = "semantic"  # semantic, hybrid, keyword
    generate_response: bool = True
    system_prompt: Optional[str] = None


@dataclass
class ProcessQueryResult:
    """Result of query processing."""
    query_id: UUID
    retrieved_documents: List[RetrievedDocument]
    generated_response: Optional[str] = None
    processing_time_seconds: float = 0.0
    total_documents_searched: int = 0
    confidence_score: float = 0.0
    success: bool = True
    error_message: Optional[str] = None


class ProcessQueryUseCase:
    """Use case for processing queries in the RAG system."""
    
    def __init__(
        self,
        document_repository: DocumentRepository,
        vector_store_repository: VectorStoreRepository,
        embedding_service: EmbeddingService,
        ranking_service: RankingService,
        generation_service: Optional[GenerationService] = None,
    ):
        self._document_repository = document_repository
        self._vector_store_repository = vector_store_repository
        self._embedding_service = embedding_service
        self._ranking_service = ranking_service
        self._generation_service = generation_service
    
    async def execute(self, command: ProcessQueryCommand) -> ProcessQueryResult:
        """Execute the query processing workflow."""
        import time
        start_time = time.time()
        
        try:
            # 1. Create query entity
            query = Query(
                text=command.query_text,
                max_results=command.max_results,
                similarity_threshold=command.similarity_threshold,
                strategy=command.strategy,
                metadata={"include_metadata": command.include_metadata}
            )
            
            # 2. Generate query embedding
            query_embedding = await self._embedding_service.embed_text(query.text)
            query.embedding = query_embedding
            
            # 3. Perform vector search
            search_results = await self._vector_store_repository.search(
                query_embedding=query_embedding,
                k=command.max_results * 2,  # Get more results for reranking
                similarity_threshold=command.similarity_threshold
            )
            
            # 4. Retrieve full document chunks
            retrieved_documents = []
            for result in search_results:
                # Get document metadata
                document = await self._document_repository.get_by_id(
                    UUID(result.metadata.get("document_id"))
                )
                
                if document:
                    retrieved_doc = RetrievedDocument(
                        id=UUID(result.metadata.get("chunk_id", result.id)),
                        content=result.content,
                        metadata=result.metadata,
                        similarity_score=result.similarity_score,
                        document_id=document.id,
                        document_source=document.source,
                        chunk_index=result.metadata.get("chunk_index", 0)
                    )
                    retrieved_documents.append(retrieved_doc)
            
            # 5. Apply reranking if needed
            if len(retrieved_documents) > command.max_results:
                ranked_documents = await self._ranking_service.rank_documents(
                    query=query,
                    documents=retrieved_documents,
                    strategy="similarity"  # Can be enhanced with other strategies
                )
                retrieved_documents = ranked_documents[:command.max_results]
            
            # 6. Generate response if requested
            generated_response = None
            confidence_score = 0.0
            
            if command.generate_response and self._generation_service:
                response = await self._generation_service.generate_response(
                    query=query.text,
                    context=retrieved_documents,
                    system_prompt=command.system_prompt
                )
                generated_response = response.text
                confidence_score = response.confidence
            
            # 7. Calculate average similarity score as confidence
            if not confidence_score and retrieved_documents:
                confidence_score = sum(
                    doc.similarity_score.value for doc in retrieved_documents
                ) / len(retrieved_documents)
            
            processing_time = time.time() - start_time
            
            return ProcessQueryResult(
                query_id=query.id,
                retrieved_documents=retrieved_documents,
                generated_response=generated_response,
                processing_time_seconds=processing_time,
                total_documents_searched=len(search_results),
                confidence_score=confidence_score,
                success=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return ProcessQueryResult(
                query_id=query.id if 'query' in locals() else None,
                retrieved_documents=[],
                processing_time_seconds=processing_time,
                success=False,
                error_message=str(e)
            )
    
    async def search_only(self, query_text: str, max_results: int = 10) -> List[RetrievedDocument]:
        """Simplified search-only method."""
        command = ProcessQueryCommand(
            query_text=query_text,
            max_results=max_results,
            generate_response=False
        )
        
        result = await self.execute(command)
        return result.retrieved_documents if result.success else []