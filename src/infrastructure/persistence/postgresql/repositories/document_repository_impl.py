from typing import List, Optional
from uuid import UUID

from sqlalchemy import select, update, delete, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.domain.entities.document import Document, DocumentChunk
from src.domain.repositories.document_repository import DocumentRepository, DocumentChunkRepository
from ..models.document_model import DocumentModel, DocumentChunkModel


class PostgreSQLDocumentRepository(DocumentRepository):
    """PostgreSQL implementation of DocumentRepository."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def save(self, document: Document) -> Document:
        """Save a document to the database."""
        db_document = DocumentModel(
            id=document.id,
            content=document.content,
            metadata=document.metadata,
            source=document.source,
            file_type=document.file_type,
            created_at=document.created_at,
            updated_at=document.updated_at,
            language=document.language,
            embedding_model=document.embedding_model,
            processing_status=document.processing_status,
            error_message=document.error_message
        )
        
        self.session.add(db_document)
        await self.session.flush()
        
        return document
    
    async def get_by_id(self, document_id: UUID) -> Optional[Document]:
        """Retrieve a document by its ID."""
        result = await self.session.execute(
            select(DocumentModel).where(DocumentModel.id == document_id)
        )
        db_document = result.scalar_one_or_none()
        
        if not db_document:
            return None
        
        return self._to_domain(db_document)
    
    async def get_by_source(self, source: str) -> Optional[Document]:
        """Retrieve a document by its source."""
        result = await self.session.execute(
            select(DocumentModel).where(DocumentModel.source == source)
        )
        db_document = result.scalar_one_or_none()
        
        if not db_document:
            return None
        
        return self._to_domain(db_document)
    
    async def get_all(self, limit: int = 100, offset: int = 0) -> List[Document]:
        """Retrieve all documents with pagination."""
        result = await self.session.execute(
            select(DocumentModel)
            .order_by(DocumentModel.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        db_documents = result.scalars().all()
        
        return [self._to_domain(doc) for doc in db_documents]
    
    async def update(self, document: Document) -> Document:
        """Update an existing document."""
        await self.session.execute(
            update(DocumentModel)
            .where(DocumentModel.id == document.id)
            .values(
                content=document.content,
                metadata=document.metadata,
                updated_at=document.updated_at,
                language=document.language,
                embedding_model=document.embedding_model,
                processing_status=document.processing_status,
                error_message=document.error_message
            )
        )
        await self.session.flush()
        
        return document
    
    async def delete(self, document_id: UUID) -> bool:
        """Delete a document by its ID."""
        result = await self.session.execute(
            delete(DocumentModel).where(DocumentModel.id == document_id)
        )
        await self.session.flush()
        
        return result.rowcount > 0
    
    async def exists(self, document_id: UUID) -> bool:
        """Check if a document exists."""
        result = await self.session.execute(
            select(DocumentModel.id).where(DocumentModel.id == document_id)
        )
        return result.scalar_one_or_none() is not None
    
    async def count(self) -> int:
        """Get total count of documents."""
        from sqlalchemy import func
        result = await self.session.execute(
            select(func.count()).select_from(DocumentModel)
        )
        return result.scalar() or 0
    
    async def search_by_metadata(self, metadata_filters: dict) -> List[Document]:
        """Search documents by metadata filters."""
        query = select(DocumentModel)
        
        # Build JSON containment query for metadata
        if metadata_filters:
            query = query.where(DocumentModel.metadata.contains(metadata_filters))
        
        result = await self.session.execute(query)
        db_documents = result.scalars().all()
        
        return [self._to_domain(doc) for doc in db_documents]
    
    async def get_by_status(self, status: str) -> List[Document]:
        """Get documents by processing status."""
        result = await self.session.execute(
            select(DocumentModel).where(DocumentModel.processing_status == status)
        )
        db_documents = result.scalars().all()
        
        return [self._to_domain(doc) for doc in db_documents]
    
    def _to_domain(self, db_document: DocumentModel) -> Document:
        """Convert database model to domain entity."""
        document = Document(
            id=db_document.id,
            content=db_document.content,
            metadata=db_document.metadata,
            source=db_document.source,
            file_type=db_document.file_type,
            created_at=db_document.created_at,
            updated_at=db_document.updated_at,
            language=db_document.language,
            chunk_ids=[],  # Will be populated separately if needed
            embedding_model=db_document.embedding_model,
            processing_status=db_document.processing_status,
            error_message=db_document.error_message
        )
        return document


class PostgreSQLDocumentChunkRepository(DocumentChunkRepository):
    """PostgreSQL implementation of DocumentChunkRepository."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def save(self, chunk: DocumentChunk) -> DocumentChunk:
        """Save a document chunk."""
        db_chunk = DocumentChunkModel(
            id=chunk.id,
            document_id=chunk.document_id,
            content=chunk.content,
            metadata=chunk.metadata,
            chunk_index=chunk.chunk_index,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            embedding_id=chunk.embedding_id,
            created_at=chunk.created_at
        )
        
        self.session.add(db_chunk)
        await self.session.flush()
        
        return chunk
    
    async def save_many(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Save multiple chunks in batch."""
        db_chunks = [
            DocumentChunkModel(
                id=chunk.id,
                document_id=chunk.document_id,
                content=chunk.content,
                metadata=chunk.metadata,
                chunk_index=chunk.chunk_index,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                embedding_id=chunk.embedding_id,
                created_at=chunk.created_at
            )
            for chunk in chunks
        ]
        
        self.session.add_all(db_chunks)
        await self.session.flush()
        
        return chunks
    
    async def get_by_id(self, chunk_id: UUID) -> Optional[DocumentChunk]:
        """Retrieve a chunk by its ID."""
        result = await self.session.execute(
            select(DocumentChunkModel).where(DocumentChunkModel.id == chunk_id)
        )
        db_chunk = result.scalar_one_or_none()
        
        if not db_chunk:
            return None
        
        return self._to_domain(db_chunk)
    
    async def get_by_document_id(self, document_id: UUID) -> List[DocumentChunk]:
        """Retrieve all chunks for a document."""
        result = await self.session.execute(
            select(DocumentChunkModel)
            .where(DocumentChunkModel.document_id == document_id)
            .order_by(DocumentChunkModel.chunk_index)
        )
        db_chunks = result.scalars().all()
        
        return [self._to_domain(chunk) for chunk in db_chunks]
    
    async def get_by_ids(self, chunk_ids: List[UUID]) -> List[DocumentChunk]:
        """Retrieve multiple chunks by their IDs."""
        result = await self.session.execute(
            select(DocumentChunkModel).where(DocumentChunkModel.id.in_(chunk_ids))
        )
        db_chunks = result.scalars().all()
        
        return [self._to_domain(chunk) for chunk in db_chunks]
    
    async def delete_by_document_id(self, document_id: UUID) -> int:
        """Delete all chunks for a document. Returns count of deleted chunks."""
        result = await self.session.execute(
            delete(DocumentChunkModel).where(DocumentChunkModel.document_id == document_id)
        )
        await self.session.flush()
        
        return result.rowcount
    
    async def update_embedding_id(self, chunk_id: UUID, embedding_id: UUID) -> bool:
        """Update the embedding ID for a chunk."""
        result = await self.session.execute(
            update(DocumentChunkModel)
            .where(DocumentChunkModel.id == chunk_id)
            .values(embedding_id=embedding_id)
        )
        await self.session.flush()
        
        return result.rowcount > 0
    
    async def count_by_document_id(self, document_id: UUID) -> int:
        """Count chunks for a specific document."""
        from sqlalchemy import func
        result = await self.session.execute(
            select(func.count())
            .select_from(DocumentChunkModel)
            .where(DocumentChunkModel.document_id == document_id)
        )
        return result.scalar() or 0
    
    def _to_domain(self, db_chunk: DocumentChunkModel) -> DocumentChunk:
        """Convert database model to domain entity."""
        return DocumentChunk(
            id=db_chunk.id,
            document_id=db_chunk.document_id,
            content=db_chunk.content,
            metadata=db_chunk.metadata,
            chunk_index=db_chunk.chunk_index,
            start_char=db_chunk.start_char,
            end_char=db_chunk.end_char,
            embedding_id=db_chunk.embedding_id,
            created_at=db_chunk.created_at
        )