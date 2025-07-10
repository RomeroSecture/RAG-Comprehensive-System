from typing import Dict, List, Optional
from uuid import UUID
import asyncio
from dataclasses import dataclass

from src.domain.entities.document import Document, DocumentChunk
from src.domain.repositories.document_repository import DocumentRepository
from src.domain.repositories.embedding_repository import EmbeddingRepository
from src.domain.repositories.vector_store_repository import VectorStoreRepository
from src.domain.value_objects.embedding import Embedding
from src.application.ports.chunking_service import ChunkingService
from src.application.ports.embedding_service import EmbeddingService
from src.application.ports.parser_service import ParserService


@dataclass
class IngestDocumentCommand:
    """Command for ingesting a document into the system."""
    file_path: str
    metadata: Dict[str, str]
    chunking_strategy: str = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 200


@dataclass
class IngestDocumentResult:
    """Result of document ingestion."""
    document_id: UUID
    chunks_created: int
    processing_time_seconds: float
    success: bool
    error_message: Optional[str] = None


class IngestDocumentUseCase:
    """Use case for ingesting documents into the RAG system."""
    
    def __init__(
        self,
        document_repository: DocumentRepository,
        embedding_repository: EmbeddingRepository,
        vector_store_repository: VectorStoreRepository,
        parser_service: ParserService,
        chunking_service: ChunkingService,
        embedding_service: EmbeddingService,
    ):
        self._document_repository = document_repository
        self._embedding_repository = embedding_repository
        self._vector_store_repository = vector_store_repository
        self._parser_service = parser_service
        self._chunking_service = chunking_service
        self._embedding_service = embedding_service
    
    async def execute(self, command: IngestDocumentCommand) -> IngestDocumentResult:
        """Execute the document ingestion process."""
        import time
        start_time = time.time()
        
        try:
            # 1. Parse the document
            parsed_content = await self._parser_service.parse_document(
                command.file_path, command.metadata
            )
            
            # 2. Create document entity
            document = Document(
                content=parsed_content.content,
                metadata=parsed_content.metadata,
                source=command.file_path,
                file_type=parsed_content.file_type,
                language=parsed_content.language,
                processing_status="processing"
            )
            
            # 3. Save document to repository
            saved_document = await self._document_repository.save(document)
            
            # 4. Create chunks
            chunks = await self._chunking_service.create_chunks(
                content=parsed_content.content,
                document_id=saved_document.id,
                strategy=command.chunking_strategy,
                chunk_size=command.chunk_size,
                overlap=command.chunk_overlap
            )
            
            # 5. Process chunks in batches
            chunk_ids = []
            batch_size = 10
            
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                
                # Generate embeddings for batch
                batch_texts = [chunk.content for chunk in batch_chunks]
                embeddings = await self._embedding_service.embed_batch(batch_texts)
                
                # Save chunks and embeddings
                for chunk, embedding in zip(batch_chunks, embeddings):
                    # Save embedding
                    saved_embedding = await self._embedding_repository.save(
                        embedding, chunk.id
                    )
                    
                    # Update chunk with embedding reference
                    chunk.embedding_id = saved_embedding.id
                    chunk_ids.append(chunk.id)
                    
                    # Store in vector store
                    await self._vector_store_repository.upsert(
                        chunk_id=chunk.id,
                        embedding=embedding,
                        metadata={
                            "document_id": str(saved_document.id),
                            "chunk_index": chunk.chunk_index,
                            "content": chunk.content[:500],  # Truncate for metadata
                            **chunk.metadata
                        }
                    )
            
            # 6. Update document with chunk references
            saved_document.chunk_ids = chunk_ids
            saved_document.mark_as_processed()
            await self._document_repository.save(saved_document)
            
            processing_time = time.time() - start_time
            
            return IngestDocumentResult(
                document_id=saved_document.id,
                chunks_created=len(chunks),
                processing_time_seconds=processing_time,
                success=True
            )
            
        except Exception as e:
            # Mark document as failed if it was created
            if 'saved_document' in locals():
                saved_document.mark_as_failed(str(e))
                await self._document_repository.save(saved_document)
            
            processing_time = time.time() - start_time
            return IngestDocumentResult(
                document_id=saved_document.id if 'saved_document' in locals() else None,
                chunks_created=0,
                processing_time_seconds=processing_time,
                success=False,
                error_message=str(e)
            )