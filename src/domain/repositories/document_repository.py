from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from ..entities.document import Document, DocumentChunk


class DocumentRepository(ABC):
    """Abstract repository interface for document persistence."""
    
    @abstractmethod
    async def save(self, document: Document) -> Document:
        """Save a document to the repository."""
        pass
    
    @abstractmethod
    async def get_by_id(self, document_id: UUID) -> Optional[Document]:
        """Retrieve a document by its ID."""
        pass
    
    @abstractmethod
    async def get_by_source(self, source: str) -> Optional[Document]:
        """Retrieve a document by its source."""
        pass
    
    @abstractmethod
    async def get_all(self, limit: int = 100, offset: int = 0) -> List[Document]:
        """Retrieve all documents with pagination."""
        pass
    
    @abstractmethod
    async def update(self, document: Document) -> Document:
        """Update an existing document."""
        pass
    
    @abstractmethod
    async def delete(self, document_id: UUID) -> bool:
        """Delete a document by its ID."""
        pass
    
    @abstractmethod
    async def exists(self, document_id: UUID) -> bool:
        """Check if a document exists."""
        pass
    
    @abstractmethod
    async def count(self) -> int:
        """Get total count of documents."""
        pass
    
    @abstractmethod
    async def search_by_metadata(self, metadata_filters: dict) -> List[Document]:
        """Search documents by metadata filters."""
        pass
    
    @abstractmethod
    async def get_by_status(self, status: str) -> List[Document]:
        """Get documents by processing status."""
        pass


class DocumentChunkRepository(ABC):
    """Abstract repository interface for document chunks."""
    
    @abstractmethod
    async def save(self, chunk: DocumentChunk) -> DocumentChunk:
        """Save a document chunk."""
        pass
    
    @abstractmethod
    async def save_many(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Save multiple chunks in batch."""
        pass
    
    @abstractmethod
    async def get_by_id(self, chunk_id: UUID) -> Optional[DocumentChunk]:
        """Retrieve a chunk by its ID."""
        pass
    
    @abstractmethod
    async def get_by_document_id(self, document_id: UUID) -> List[DocumentChunk]:
        """Retrieve all chunks for a document."""
        pass
    
    @abstractmethod
    async def get_by_ids(self, chunk_ids: List[UUID]) -> List[DocumentChunk]:
        """Retrieve multiple chunks by their IDs."""
        pass
    
    @abstractmethod
    async def delete_by_document_id(self, document_id: UUID) -> int:
        """Delete all chunks for a document. Returns count of deleted chunks."""
        pass
    
    @abstractmethod
    async def update_embedding_id(self, chunk_id: UUID, embedding_id: UUID) -> bool:
        """Update the embedding ID for a chunk."""
        pass
    
    @abstractmethod
    async def count_by_document_id(self, document_id: UUID) -> int:
        """Count chunks for a specific document."""
        pass