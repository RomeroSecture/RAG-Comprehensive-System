import pytest
from datetime import datetime
from uuid import UUID, uuid4

from src.domain.entities.document import Document, DocumentChunk


class TestDocument:
    """Unit tests for Document entity."""
    
    def test_document_creation_with_valid_data(self, sample_document_data):
        """Test document creation with valid data."""
        document = Document(
            content=sample_document_data["content"],
            metadata=sample_document_data["metadata"],
            source=sample_document_data["source"],
            file_type=sample_document_data["file_type"],
            language=sample_document_data["language"]
        )
        
        assert isinstance(document.id, UUID)
        assert document.content == sample_document_data["content"]
        assert document.metadata == sample_document_data["metadata"]
        assert document.source == sample_document_data["source"]
        assert document.file_type == sample_document_data["file_type"]
        assert document.language == sample_document_data["language"]
        assert document.processing_status == "pending"
        assert document.error_message is None
        assert isinstance(document.created_at, datetime)
        assert isinstance(document.updated_at, datetime)
        assert document.chunk_ids == []
    
    def test_document_creation_with_empty_content_raises_error(self):
        """Test that creating a document with empty content raises ValueError."""
        with pytest.raises(ValueError, match="Document content cannot be empty"):
            Document(content="", source="test.pdf", file_type="pdf")
    
    def test_document_creation_with_invalid_file_type_raises_error(self):
        """Test that creating a document with unsupported file type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported file type: exe"):
            Document(
                content="Test content",
                source="test.exe",
                file_type="exe"
            )
    
    def test_document_with_error_status_allows_empty_content(self):
        """Test that document with error status can have empty content."""
        document = Document(
            content="",
            processing_status="error",
            error_message="Failed to extract content"
        )
        assert document.content == ""
        assert document.processing_status == "error"
    
    def test_supported_file_types(self):
        """Test that all expected file types are supported."""
        document = Document(content="test")
        expected_types = ["pdf", "docx", "txt", "markdown", "html", "json", "csv", "xlsx"]
        assert document.SUPPORTED_FILE_TYPES == expected_types
    
    def test_update_metadata(self, mock_datetime):
        """Test updating document metadata."""
        document = Document(content="test", metadata={"key1": "value1"})
        original_updated_at = document.updated_at
        
        new_metadata = {"key2": "value2", "key3": "value3"}
        document.update_metadata(new_metadata)
        
        assert document.metadata == {"key1": "value1", "key2": "value2", "key3": "value3"}
        assert document.updated_at != original_updated_at
    
    def test_mark_as_processed(self, mock_datetime):
        """Test marking document as processed."""
        document = Document(content="test")
        original_updated_at = document.updated_at
        
        document.mark_as_processed()
        
        assert document.processing_status == "completed"
        assert document.updated_at != original_updated_at
    
    def test_mark_as_failed(self, mock_datetime):
        """Test marking document as failed."""
        document = Document(content="test")
        error_msg = "Failed to generate embeddings"
        
        document.mark_as_failed(error_msg)
        
        assert document.processing_status == "error"
        assert document.error_message == error_msg
    
    def test_add_chunk_id(self, mock_datetime):
        """Test adding chunk IDs to document."""
        document = Document(content="test")
        chunk_id1 = uuid4()
        chunk_id2 = uuid4()
        
        document.add_chunk_id(chunk_id1)
        assert chunk_id1 in document.chunk_ids
        assert len(document.chunk_ids) == 1
        
        # Test adding duplicate chunk ID
        document.add_chunk_id(chunk_id1)
        assert len(document.chunk_ids) == 1
        
        # Test adding different chunk ID
        document.add_chunk_id(chunk_id2)
        assert chunk_id2 in document.chunk_ids
        assert len(document.chunk_ids) == 2


class TestDocumentChunk:
    """Unit tests for DocumentChunk entity."""
    
    def test_chunk_creation_with_valid_data(self, sample_chunk_data):
        """Test chunk creation with valid data."""
        chunk = DocumentChunk(
            document_id=sample_chunk_data["document_id"],
            content=sample_chunk_data["content"],
            metadata=sample_chunk_data["metadata"],
            chunk_index=sample_chunk_data["chunk_index"],
            start_char=sample_chunk_data["start_char"],
            end_char=sample_chunk_data["end_char"]
        )
        
        assert isinstance(chunk.id, UUID)
        assert chunk.document_id == sample_chunk_data["document_id"]
        assert chunk.content == sample_chunk_data["content"]
        assert chunk.metadata == sample_chunk_data["metadata"]
        assert chunk.chunk_index == sample_chunk_data["chunk_index"]
        assert chunk.start_char == sample_chunk_data["start_char"]
        assert chunk.end_char == sample_chunk_data["end_char"]
        assert chunk.embedding_id is None
        assert isinstance(chunk.created_at, datetime)
    
    def test_chunk_creation_with_empty_content_raises_error(self):
        """Test that creating a chunk with empty content raises ValueError."""
        with pytest.raises(ValueError, match="Chunk content cannot be empty"):
            DocumentChunk(
                document_id=uuid4(),
                content="",
                chunk_index=0,
                start_char=0,
                end_char=10
            )
    
    def test_chunk_creation_with_negative_positions_raises_error(self):
        """Test that negative character positions raise ValueError."""
        with pytest.raises(ValueError, match="Character positions must be non-negative"):
            DocumentChunk(
                document_id=uuid4(),
                content="test",
                chunk_index=0,
                start_char=-1,
                end_char=10
            )
    
    def test_chunk_creation_with_invalid_position_order_raises_error(self):
        """Test that start >= end positions raise ValueError."""
        with pytest.raises(ValueError, match="Start position must be less than end position"):
            DocumentChunk(
                document_id=uuid4(),
                content="test",
                chunk_index=0,
                start_char=10,
                end_char=5
            )
    
    def test_char_count_property(self):
        """Test char_count property calculation."""
        chunk = DocumentChunk(
            document_id=uuid4(),
            content="Hello, world!",
            chunk_index=0,
            start_char=100,
            end_char=113
        )
        
        assert chunk.char_count == 13
    
    def test_to_dict_conversion(self):
        """Test converting chunk to dictionary."""
        doc_id = uuid4()
        chunk_id = uuid4()
        embedding_id = uuid4()
        
        chunk = DocumentChunk(
            id=chunk_id,
            document_id=doc_id,
            content="Test content",
            metadata={"page": 1},
            chunk_index=0,
            start_char=0,
            end_char=12,
            embedding_id=embedding_id
        )
        
        chunk_dict = chunk.to_dict()
        
        assert chunk_dict["id"] == str(chunk_id)
        assert chunk_dict["document_id"] == str(doc_id)
        assert chunk_dict["content"] == "Test content"
        assert chunk_dict["metadata"] == {"page": 1}
        assert chunk_dict["chunk_index"] == 0
        assert chunk_dict["start_char"] == 0
        assert chunk_dict["end_char"] == 12
        assert chunk_dict["embedding_id"] == str(embedding_id)
        assert "created_at" in chunk_dict
    
    def test_to_dict_with_none_embedding_id(self):
        """Test to_dict conversion when embedding_id is None."""
        chunk = DocumentChunk(
            document_id=uuid4(),
            content="Test",
            chunk_index=0,
            start_char=0,
            end_char=4
        )
        
        chunk_dict = chunk.to_dict()
        assert chunk_dict["embedding_id"] is None