import pytest
from abc import ABC
from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID, uuid4

from src.domain.entities.document import Document, DocumentChunk
from src.domain.repositories.document_repository import DocumentRepository, DocumentChunkRepository


class MockDocumentRepository(DocumentRepository):
    """Mock implementation of DocumentRepository for testing."""
    
    def __init__(self):
        self.documents = {}
        self.save_called = False
        self.get_called = False
    
    async def save(self, document: Document) -> Document:
        self.save_called = True
        self.documents[document.id] = document
        return document
    
    async def get_by_id(self, document_id: UUID) -> Optional[Document]:
        self.get_called = True
        return self.documents.get(document_id)
    
    async def get_by_source(self, source: str) -> Optional[Document]:
        for doc in self.documents.values():
            if doc.source == source:
                return doc
        return None
    
    async def get_all(self, limit: int = 100, offset: int = 0) -> List[Document]:
        docs = list(self.documents.values())
        return docs[offset:offset + limit]
    
    async def update(self, document: Document) -> Document:
        if document.id in self.documents:
            self.documents[document.id] = document
            return document
        raise ValueError(f"Document {document.id} not found")
    
    async def delete(self, document_id: UUID) -> bool:
        if document_id in self.documents:
            del self.documents[document_id]
            return True
        return False
    
    async def exists(self, document_id: UUID) -> bool:
        return document_id in self.documents
    
    async def count(self) -> int:
        return len(self.documents)
    
    async def search_by_metadata(self, metadata_filters: dict) -> List[Document]:
        results = []
        for doc in self.documents.values():
            match = True
            for key, value in metadata_filters.items():
                if key not in doc.metadata or doc.metadata[key] != value:
                    match = False
                    break
            if match:
                results.append(doc)
        return results
    
    async def get_by_status(self, status: str) -> List[Document]:
        return [doc for doc in self.documents.values() if doc.processing_status == status]


class MockDocumentChunkRepository(DocumentChunkRepository):
    """Mock implementation of DocumentChunkRepository for testing."""
    
    def __init__(self):
        self.chunks = {}
        self.save_many_called = False
    
    async def save(self, chunk: DocumentChunk) -> DocumentChunk:
        self.chunks[chunk.id] = chunk
        return chunk
    
    async def save_many(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        self.save_many_called = True
        for chunk in chunks:
            self.chunks[chunk.id] = chunk
        return chunks
    
    async def get_by_id(self, chunk_id: UUID) -> Optional[DocumentChunk]:
        return self.chunks.get(chunk_id)
    
    async def get_by_document_id(self, document_id: UUID) -> List[DocumentChunk]:
        return [chunk for chunk in self.chunks.values() if chunk.document_id == document_id]
    
    async def get_by_ids(self, chunk_ids: List[UUID]) -> List[DocumentChunk]:
        return [self.chunks[chunk_id] for chunk_id in chunk_ids if chunk_id in self.chunks]
    
    async def delete_by_document_id(self, document_id: UUID) -> int:
        to_delete = [chunk_id for chunk_id, chunk in self.chunks.items() 
                     if chunk.document_id == document_id]
        for chunk_id in to_delete:
            del self.chunks[chunk_id]
        return len(to_delete)
    
    async def update_embedding_id(self, chunk_id: UUID, embedding_id: UUID) -> bool:
        if chunk_id in self.chunks:
            self.chunks[chunk_id].embedding_id = embedding_id
            return True
        return False
    
    async def count_by_document_id(self, document_id: UUID) -> int:
        return sum(1 for chunk in self.chunks.values() if chunk.document_id == document_id)


class TestDocumentRepository:
    """Test cases for DocumentRepository interface."""
    
    @pytest.mark.asyncio
    async def test_repository_is_abstract(self):
        """Test that DocumentRepository is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            DocumentRepository()
    
    @pytest.mark.asyncio
    async def test_save_and_get_document(self):
        """Test saving and retrieving a document."""
        repo = MockDocumentRepository()
        
        document = Document(
            content="Test content",
            source="test.pdf",
            file_type="pdf"
        )
        
        # Save document
        saved_doc = await repo.save(document)
        assert saved_doc == document
        assert repo.save_called
        
        # Get document by ID
        retrieved_doc = await repo.get_by_id(document.id)
        assert retrieved_doc == document
        assert repo.get_called
    
    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self):
        """Test getting a document that doesn't exist."""
        repo = MockDocumentRepository()
        
        result = await repo.get_by_id(uuid4())
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_by_source(self):
        """Test getting a document by source."""
        repo = MockDocumentRepository()
        
        document = Document(
            content="Test content",
            source="unique-source.pdf",
            file_type="pdf"
        )
        await repo.save(document)
        
        result = await repo.get_by_source("unique-source.pdf")
        assert result == document
        
        # Test not found
        result = await repo.get_by_source("non-existent.pdf")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_all_with_pagination(self):
        """Test getting all documents with pagination."""
        repo = MockDocumentRepository()
        
        # Create multiple documents
        for i in range(5):
            doc = Document(
                content=f"Document {i}",
                source=f"doc{i}.pdf",
                file_type="pdf"
            )
            await repo.save(doc)
        
        # Test pagination
        page1 = await repo.get_all(limit=2, offset=0)
        assert len(page1) == 2
        
        page2 = await repo.get_all(limit=2, offset=2)
        assert len(page2) == 2
        
        page3 = await repo.get_all(limit=2, offset=4)
        assert len(page3) == 1
    
    @pytest.mark.asyncio
    async def test_update_document(self):
        """Test updating a document."""
        repo = MockDocumentRepository()
        
        document = Document(content="Original", source="test.pdf", file_type="pdf")
        await repo.save(document)
        
        # Update document
        document.mark_as_processed()
        updated_doc = await repo.update(document)
        
        assert updated_doc.processing_status == "completed"
        
        # Verify update
        retrieved = await repo.get_by_id(document.id)
        assert retrieved.processing_status == "completed"
    
    @pytest.mark.asyncio
    async def test_update_non_existent_document(self):
        """Test updating a document that doesn't exist."""
        repo = MockDocumentRepository()
        
        document = Document(content="Test", source="test.pdf", file_type="pdf")
        
        with pytest.raises(ValueError, match=f"Document {document.id} not found"):
            await repo.update(document)
    
    @pytest.mark.asyncio
    async def test_delete_document(self):
        """Test deleting a document."""
        repo = MockDocumentRepository()
        
        document = Document(content="Test", source="test.pdf", file_type="pdf")
        await repo.save(document)
        
        # Delete document
        result = await repo.delete(document.id)
        assert result is True
        
        # Verify deletion
        retrieved = await repo.get_by_id(document.id)
        assert retrieved is None
        
        # Delete non-existent
        result = await repo.delete(uuid4())
        assert result is False
    
    @pytest.mark.asyncio
    async def test_exists(self):
        """Test checking if a document exists."""
        repo = MockDocumentRepository()
        
        document = Document(content="Test", source="test.pdf", file_type="pdf")
        await repo.save(document)
        
        assert await repo.exists(document.id) is True
        assert await repo.exists(uuid4()) is False
    
    @pytest.mark.asyncio
    async def test_count(self):
        """Test counting documents."""
        repo = MockDocumentRepository()
        
        assert await repo.count() == 0
        
        for i in range(3):
            doc = Document(content=f"Doc {i}", source=f"doc{i}.pdf", file_type="pdf")
            await repo.save(doc)
        
        assert await repo.count() == 3
    
    @pytest.mark.asyncio
    async def test_search_by_metadata(self):
        """Test searching documents by metadata."""
        repo = MockDocumentRepository()
        
        # Create documents with different metadata
        doc1 = Document(
            content="Doc 1",
            metadata={"category": "tech", "year": 2024}
        )
        doc2 = Document(
            content="Doc 2",
            metadata={"category": "science", "year": 2024}
        )
        doc3 = Document(
            content="Doc 3",
            metadata={"category": "tech", "year": 2023}
        )
        
        for doc in [doc1, doc2, doc3]:
            await repo.save(doc)
        
        # Search by single filter
        results = await repo.search_by_metadata({"category": "tech"})
        assert len(results) == 2
        assert doc1 in results
        assert doc3 in results
        
        # Search by multiple filters
        results = await repo.search_by_metadata({"category": "tech", "year": 2024})
        assert len(results) == 1
        assert doc1 in results
    
    @pytest.mark.asyncio
    async def test_get_by_status(self):
        """Test getting documents by processing status."""
        repo = MockDocumentRepository()
        
        # Create documents with different statuses
        doc1 = Document(content="Doc 1", processing_status="pending")
        doc2 = Document(content="Doc 2", processing_status="pending")
        doc3 = Document(content="Doc 3", processing_status="completed")
        doc3.mark_as_processed()
        
        for doc in [doc1, doc2, doc3]:
            await repo.save(doc)
        
        # Get by status
        pending_docs = await repo.get_by_status("pending")
        assert len(pending_docs) == 2
        
        completed_docs = await repo.get_by_status("completed")
        assert len(completed_docs) == 1
        assert completed_docs[0] == doc3


class TestDocumentChunkRepository:
    """Test cases for DocumentChunkRepository interface."""
    
    @pytest.mark.asyncio
    async def test_chunk_repository_is_abstract(self):
        """Test that DocumentChunkRepository is abstract."""
        with pytest.raises(TypeError):
            DocumentChunkRepository()
    
    @pytest.mark.asyncio
    async def test_save_and_get_chunk(self):
        """Test saving and retrieving a chunk."""
        repo = MockDocumentChunkRepository()
        
        chunk = DocumentChunk(
            document_id=uuid4(),
            content="Test chunk content",
            chunk_index=0,
            start_char=0,
            end_char=18
        )
        
        saved_chunk = await repo.save(chunk)
        assert saved_chunk == chunk
        
        retrieved_chunk = await repo.get_by_id(chunk.id)
        assert retrieved_chunk == chunk
    
    @pytest.mark.asyncio
    async def test_save_many_chunks(self):
        """Test saving multiple chunks in batch."""
        repo = MockDocumentChunkRepository()
        
        doc_id = uuid4()
        chunks = [
            DocumentChunk(
                document_id=doc_id,
                content=f"Chunk {i}",
                chunk_index=i,
                start_char=i * 100,
                end_char=(i + 1) * 100
            )
            for i in range(3)
        ]
        
        saved_chunks = await repo.save_many(chunks)
        assert len(saved_chunks) == 3
        assert repo.save_many_called
        
        # Verify all chunks were saved
        for chunk in chunks:
            retrieved = await repo.get_by_id(chunk.id)
            assert retrieved == chunk
    
    @pytest.mark.asyncio
    async def test_get_by_document_id(self):
        """Test getting all chunks for a document."""
        repo = MockDocumentChunkRepository()
        
        doc_id = uuid4()
        other_doc_id = uuid4()
        
        # Create chunks for different documents
        chunks_doc1 = [
            DocumentChunk(
                document_id=doc_id,
                content=f"Doc1 Chunk {i}",
                chunk_index=i,
                start_char=i * 100,
                end_char=(i + 1) * 100
            )
            for i in range(3)
        ]
        
        chunk_doc2 = DocumentChunk(
            document_id=other_doc_id,
            content="Doc2 Chunk",
            chunk_index=0,
            start_char=0,
            end_char=100
        )
        
        await repo.save_many(chunks_doc1)
        await repo.save(chunk_doc2)
        
        # Get chunks for doc1
        retrieved_chunks = await repo.get_by_document_id(doc_id)
        assert len(retrieved_chunks) == 3
        for chunk in retrieved_chunks:
            assert chunk.document_id == doc_id
    
    @pytest.mark.asyncio
    async def test_get_by_ids(self):
        """Test getting multiple chunks by their IDs."""
        repo = MockDocumentChunkRepository()
        
        chunks = []
        for i in range(5):
            chunk = DocumentChunk(
                document_id=uuid4(),
                content=f"Chunk {i}",
                chunk_index=i,
                start_char=0,
                end_char=100
            )
            chunks.append(chunk)
            await repo.save(chunk)
        
        # Get specific chunks
        chunk_ids = [chunks[0].id, chunks[2].id, chunks[4].id]
        retrieved = await repo.get_by_ids(chunk_ids)
        
        assert len(retrieved) == 3
        assert chunks[0] in retrieved
        assert chunks[2] in retrieved
        assert chunks[4] in retrieved
    
    @pytest.mark.asyncio
    async def test_delete_by_document_id(self):
        """Test deleting all chunks for a document."""
        repo = MockDocumentChunkRepository()
        
        doc_id = uuid4()
        other_doc_id = uuid4()
        
        # Create chunks
        for i in range(3):
            chunk = DocumentChunk(
                document_id=doc_id,
                content=f"Chunk {i}",
                chunk_index=i,
                start_char=0,
                end_char=100
            )
            await repo.save(chunk)
        
        # Create chunk for another document
        other_chunk = DocumentChunk(
            document_id=other_doc_id,
            content="Other chunk",
            chunk_index=0,
            start_char=0,
            end_char=100
        )
        await repo.save(other_chunk)
        
        # Delete chunks for doc_id
        deleted_count = await repo.delete_by_document_id(doc_id)
        assert deleted_count == 3
        
        # Verify deletion
        remaining_chunks = await repo.get_by_document_id(doc_id)
        assert len(remaining_chunks) == 0
        
        # Verify other document's chunks remain
        other_chunks = await repo.get_by_document_id(other_doc_id)
        assert len(other_chunks) == 1
    
    @pytest.mark.asyncio
    async def test_update_embedding_id(self):
        """Test updating embedding ID for a chunk."""
        repo = MockDocumentChunkRepository()
        
        chunk = DocumentChunk(
            document_id=uuid4(),
            content="Test chunk",
            chunk_index=0,
            start_char=0,
            end_char=100
        )
        await repo.save(chunk)
        
        # Update embedding ID
        embedding_id = uuid4()
        result = await repo.update_embedding_id(chunk.id, embedding_id)
        assert result is True
        
        # Verify update
        updated_chunk = await repo.get_by_id(chunk.id)
        assert updated_chunk.embedding_id == embedding_id
        
        # Update non-existent chunk
        result = await repo.update_embedding_id(uuid4(), embedding_id)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_count_by_document_id(self):
        """Test counting chunks for a document."""
        repo = MockDocumentChunkRepository()
        
        doc_id = uuid4()
        
        # Initially no chunks
        count = await repo.count_by_document_id(doc_id)
        assert count == 0
        
        # Add chunks
        for i in range(5):
            chunk = DocumentChunk(
                document_id=doc_id,
                content=f"Chunk {i}",
                chunk_index=i,
                start_char=i * 100,
                end_char=(i + 1) * 100
            )
            await repo.save(chunk)
        
        count = await repo.count_by_document_id(doc_id)
        assert count == 5