import pytest
from datetime import datetime, timezone

from src.domain.value_objects.retrieval_metadata import (
    ChunkingMetadata,
    ProcessingMetadata,
    DocumentMetadata
)


class TestChunkingMetadata:
    """Unit tests for ChunkingMetadata value object."""
    
    def test_chunking_metadata_creation(self):
        """Test creating chunking metadata with valid data."""
        metadata = ChunkingMetadata(
            strategy="sentence",
            chunk_size=1000,
            chunk_overlap=200,
            total_chunks=5,
            separator="\n",
            preserve_formatting=True
        )
        
        assert metadata.strategy == "sentence"
        assert metadata.chunk_size == 1000
        assert metadata.chunk_overlap == 200
        assert metadata.total_chunks == 5
        assert metadata.separator == "\n"
        assert metadata.preserve_formatting is True
    
    def test_chunking_metadata_is_immutable(self):
        """Test that chunking metadata is immutable."""
        metadata = ChunkingMetadata(
            strategy="fixed",
            chunk_size=500,
            chunk_overlap=50,
            total_chunks=10
        )
        
        with pytest.raises(AttributeError):
            metadata.chunk_size = 600
    
    def test_chunking_metadata_with_zero_chunk_size_raises_error(self):
        """Test that zero chunk size raises error."""
        with pytest.raises(ValueError, match="Chunk size must be positive"):
            ChunkingMetadata(
                strategy="fixed",
                chunk_size=0,
                chunk_overlap=0,
                total_chunks=1
            )
    
    def test_chunking_metadata_with_negative_chunk_size_raises_error(self):
        """Test that negative chunk size raises error."""
        with pytest.raises(ValueError, match="Chunk size must be positive"):
            ChunkingMetadata(
                strategy="fixed",
                chunk_size=-100,
                chunk_overlap=0,
                total_chunks=1
            )
    
    def test_chunking_metadata_with_negative_overlap_raises_error(self):
        """Test that negative overlap raises error."""
        with pytest.raises(ValueError, match="Chunk overlap cannot be negative"):
            ChunkingMetadata(
                strategy="fixed",
                chunk_size=1000,
                chunk_overlap=-50,
                total_chunks=1
            )
    
    def test_chunking_metadata_with_overlap_exceeding_size_raises_error(self):
        """Test that overlap >= chunk size raises error."""
        with pytest.raises(ValueError, match="Chunk overlap must be less than chunk size"):
            ChunkingMetadata(
                strategy="fixed",
                chunk_size=500,
                chunk_overlap=500,
                total_chunks=1
            )
        
        with pytest.raises(ValueError, match="Chunk overlap must be less than chunk size"):
            ChunkingMetadata(
                strategy="fixed",
                chunk_size=500,
                chunk_overlap=600,
                total_chunks=1
            )
    
    def test_chunking_metadata_with_zero_total_chunks_raises_error(self):
        """Test that zero total chunks raises error."""
        with pytest.raises(ValueError, match="Total chunks must be positive"):
            ChunkingMetadata(
                strategy="fixed",
                chunk_size=1000,
                chunk_overlap=200,
                total_chunks=0
            )
    
    def test_chunking_metadata_defaults(self):
        """Test default values for optional fields."""
        metadata = ChunkingMetadata(
            strategy="paragraph",
            chunk_size=2000,
            chunk_overlap=100,
            total_chunks=3
        )
        
        assert metadata.separator is None
        assert metadata.preserve_formatting is True


class TestProcessingMetadata:
    """Unit tests for ProcessingMetadata value object."""
    
    def test_processing_metadata_creation(self):
        """Test creating processing metadata with valid data."""
        timestamp = datetime.now(timezone.utc)
        metadata = ProcessingMetadata(
            processing_time_ms=1234.5,
            processor_version="1.0.0",
            timestamp=timestamp,
            pipeline_steps=["parse", "chunk", "embed"],
            errors=["Error 1"],
            warnings=["Warning 1", "Warning 2"]
        )
        
        assert metadata.processing_time_ms == 1234.5
        assert metadata.processor_version == "1.0.0"
        assert metadata.timestamp == timestamp
        assert metadata.pipeline_steps == ["parse", "chunk", "embed"]
        assert metadata.errors == ["Error 1"]
        assert metadata.warnings == ["Warning 1", "Warning 2"]
    
    def test_processing_metadata_is_immutable(self):
        """Test that processing metadata is immutable."""
        metadata = ProcessingMetadata(
            processing_time_ms=100.0,
            processor_version="1.0.0",
            timestamp=datetime.now(timezone.utc)
        )
        
        with pytest.raises(AttributeError):
            metadata.processing_time_ms = 200.0
    
    def test_processing_metadata_with_negative_time_raises_error(self):
        """Test that negative processing time raises error."""
        with pytest.raises(ValueError, match="Processing time cannot be negative"):
            ProcessingMetadata(
                processing_time_ms=-100.0,
                processor_version="1.0.0",
                timestamp=datetime.now(timezone.utc)
            )
    
    def test_processing_metadata_with_empty_version_raises_error(self):
        """Test that empty processor version raises error."""
        with pytest.raises(ValueError, match="Processor version must be specified"):
            ProcessingMetadata(
                processing_time_ms=100.0,
                processor_version="",
                timestamp=datetime.now(timezone.utc)
            )
    
    def test_has_errors_property(self):
        """Test has_errors property."""
        # With errors
        metadata_with_errors = ProcessingMetadata(
            processing_time_ms=100.0,
            processor_version="1.0.0",
            timestamp=datetime.now(timezone.utc),
            errors=["Error 1", "Error 2"]
        )
        assert metadata_with_errors.has_errors is True
        
        # Without errors
        metadata_no_errors = ProcessingMetadata(
            processing_time_ms=100.0,
            processor_version="1.0.0",
            timestamp=datetime.now(timezone.utc)
        )
        assert metadata_no_errors.has_errors is False
    
    def test_has_warnings_property(self):
        """Test has_warnings property."""
        # With warnings
        metadata_with_warnings = ProcessingMetadata(
            processing_time_ms=100.0,
            processor_version="1.0.0",
            timestamp=datetime.now(timezone.utc),
            warnings=["Warning 1"]
        )
        assert metadata_with_warnings.has_warnings is True
        
        # Without warnings
        metadata_no_warnings = ProcessingMetadata(
            processing_time_ms=100.0,
            processor_version="1.0.0",
            timestamp=datetime.now(timezone.utc)
        )
        assert metadata_no_warnings.has_warnings is False
    
    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        timestamp = datetime.now(timezone.utc)
        metadata = ProcessingMetadata(
            processing_time_ms=500.0,
            processor_version="2.0.0",
            timestamp=timestamp,
            pipeline_steps=["step1", "step2"],
            errors=["error1"],
            warnings=["warning1", "warning2"]
        )
        
        result = metadata.to_dict()
        
        assert result["processing_time_ms"] == 500.0
        assert result["processor_version"] == "2.0.0"
        assert result["timestamp"] == timestamp.isoformat()
        assert result["pipeline_steps"] == ["step1", "step2"]
        assert result["errors"] == ["error1"]
        assert result["warnings"] == ["warning1", "warning2"]
    
    def test_processing_metadata_defaults(self):
        """Test default values for lists."""
        metadata = ProcessingMetadata(
            processing_time_ms=100.0,
            processor_version="1.0.0",
            timestamp=datetime.now(timezone.utc)
        )
        
        assert metadata.pipeline_steps == []
        assert metadata.errors == []
        assert metadata.warnings == []


class TestDocumentMetadata:
    """Unit tests for DocumentMetadata value object."""
    
    def test_document_metadata_creation(self):
        """Test creating document metadata with all fields."""
        created = datetime.now(timezone.utc)
        modified = datetime.now(timezone.utc)
        
        metadata = DocumentMetadata(
            title="Test Document",
            author="John Doe",
            created_date=created,
            modified_date=modified,
            source_url="https://example.com/doc.pdf",
            file_size_bytes=1024000,
            page_count=10,
            word_count=5000,
            language="en",
            content_type="application/pdf",
            tags=["test", "example"],
            custom_fields={"department": "Engineering"}
        )
        
        assert metadata.title == "Test Document"
        assert metadata.author == "John Doe"
        assert metadata.created_date == created
        assert metadata.modified_date == modified
        assert metadata.source_url == "https://example.com/doc.pdf"
        assert metadata.file_size_bytes == 1024000
        assert metadata.page_count == 10
        assert metadata.word_count == 5000
        assert metadata.language == "en"
        assert metadata.content_type == "application/pdf"
        assert metadata.tags == ["test", "example"]
        assert metadata.custom_fields == {"department": "Engineering"}
    
    def test_document_metadata_is_immutable(self):
        """Test that document metadata is immutable."""
        metadata = DocumentMetadata(title="Test")
        
        with pytest.raises(AttributeError):
            metadata.title = "New Title"
    
    def test_document_metadata_with_minimal_fields(self):
        """Test creating document metadata with minimal fields."""
        metadata = DocumentMetadata()
        
        assert metadata.title is None
        assert metadata.author is None
        assert metadata.created_date is None
        assert metadata.tags == []
        assert metadata.custom_fields == {}
    
    def test_document_metadata_with_negative_file_size_raises_error(self):
        """Test that negative file size raises error."""
        with pytest.raises(ValueError, match="File size cannot be negative"):
            DocumentMetadata(file_size_bytes=-100)
    
    def test_document_metadata_with_negative_page_count_raises_error(self):
        """Test that negative page count raises error."""
        with pytest.raises(ValueError, match="Page count cannot be negative"):
            DocumentMetadata(page_count=-1)
    
    def test_document_metadata_with_negative_word_count_raises_error(self):
        """Test that negative word count raises error."""
        with pytest.raises(ValueError, match="Word count cannot be negative"):
            DocumentMetadata(word_count=-100)
    
    def test_add_tag(self):
        """Test adding tags (returns new instance)."""
        metadata = DocumentMetadata(tags=["tag1"])
        
        # Add new tag
        new_metadata = metadata.add_tag("tag2")
        
        # Original unchanged
        assert metadata.tags == ["tag1"]
        # New instance has both tags
        assert new_metadata.tags == ["tag1", "tag2"]
        
        # Add duplicate tag (should not duplicate)
        same_metadata = new_metadata.add_tag("tag2")
        assert same_metadata.tags == ["tag1", "tag2"]
    
    def test_with_custom_field(self):
        """Test adding custom fields (returns new instance)."""
        metadata = DocumentMetadata(custom_fields={"field1": "value1"})
        
        # Add new field
        new_metadata = metadata.with_custom_field("field2", "value2")
        
        # Original unchanged
        assert metadata.custom_fields == {"field1": "value1"}
        # New instance has both fields
        assert new_metadata.custom_fields == {"field1": "value1", "field2": "value2"}
        
        # Update existing field
        updated_metadata = new_metadata.with_custom_field("field1", "updated")
        assert updated_metadata.custom_fields == {"field1": "updated", "field2": "value2"}
    
    def test_to_dict_with_all_fields(self):
        """Test to_dict conversion with all fields."""
        created = datetime.now(timezone.utc)
        modified = datetime.now(timezone.utc)
        
        metadata = DocumentMetadata(
            title="Test",
            author="Author",
            created_date=created,
            modified_date=modified,
            source_url="https://test.com",
            file_size_bytes=1000,
            page_count=5,
            word_count=500,
            language="en",
            content_type="text/plain",
            tags=["tag1", "tag2"],
            custom_fields={"custom": "value"}
        )
        
        result = metadata.to_dict()
        
        assert result["title"] == "Test"
        assert result["author"] == "Author"
        assert result["created_date"] == created.isoformat()
        assert result["modified_date"] == modified.isoformat()
        assert result["source_url"] == "https://test.com"
        assert result["file_size_bytes"] == 1000
        assert result["page_count"] == 5
        assert result["word_count"] == 500
        assert result["language"] == "en"
        assert result["content_type"] == "text/plain"
        assert result["tags"] == ["tag1", "tag2"]
        assert result["custom"] == "value"
    
    def test_to_dict_excludes_none_values(self):
        """Test that to_dict excludes None values."""
        metadata = DocumentMetadata(
            title="Test",
            author=None,
            page_count=10
        )
        
        result = metadata.to_dict()
        
        assert "title" in result
        assert "author" not in result  # None values excluded
        assert "page_count" in result
        assert "word_count" not in result  # None values excluded
    
    def test_to_dict_includes_custom_fields(self):
        """Test that custom fields are included in to_dict."""
        metadata = DocumentMetadata(
            title="Test",
            custom_fields={
                "department": "IT",
                "project": "RAG System",
                "priority": "high"
            }
        )
        
        result = metadata.to_dict()
        
        assert result["title"] == "Test"
        assert result["department"] == "IT"
        assert result["project"] == "RAG System"
        assert result["priority"] == "high"
    
    def test_immutability_through_methods(self):
        """Test that modifications are done through methods that return new instances."""
        metadata = DocumentMetadata(
            tags=["tag1", "tag2"],
            custom_fields={"key": "value"}
        )
        
        # Test that add_tag returns new instance
        new_metadata = metadata.add_tag("tag3")
        assert metadata.tags == ["tag1", "tag2"]  # Original unchanged
        assert new_metadata.tags == ["tag1", "tag2", "tag3"]  # New instance has new tag
        
        # Test that with_custom_field returns new instance
        new_metadata2 = metadata.with_custom_field("new_key", "new_value")
        assert metadata.custom_fields == {"key": "value"}  # Original unchanged
        assert new_metadata2.custom_fields == {"key": "value", "new_key": "new_value"}  # New instance has new field