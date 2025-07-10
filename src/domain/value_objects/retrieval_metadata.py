from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ChunkingMetadata:
    """Metadata about how a document was chunked."""
    
    strategy: str
    chunk_size: int
    chunk_overlap: int
    total_chunks: int
    separator: Optional[str] = None
    preserve_formatting: bool = True
    
    def __post_init__(self):
        if self.chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        
        if self.chunk_overlap < 0:
            raise ValueError("Chunk overlap cannot be negative")
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        
        if self.total_chunks <= 0:
            raise ValueError("Total chunks must be positive")


@dataclass(frozen=True)
class ProcessingMetadata:
    """Metadata about document processing."""
    
    processing_time_ms: float
    processor_version: str
    timestamp: datetime
    pipeline_steps: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.processing_time_ms < 0:
            raise ValueError("Processing time cannot be negative")
        
        if not self.processor_version:
            raise ValueError("Processor version must be specified")
    
    @property
    def has_errors(self) -> bool:
        """Check if processing had errors."""
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        """Check if processing had warnings."""
        return len(self.warnings) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "processing_time_ms": self.processing_time_ms,
            "processor_version": self.processor_version,
            "timestamp": self.timestamp.isoformat(),
            "pipeline_steps": self.pipeline_steps,
            "errors": self.errors,
            "warnings": self.warnings
        }


@dataclass(frozen=True)
class DocumentMetadata:
    """Rich metadata for documents."""
    
    title: Optional[str] = None
    author: Optional[str] = None
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    source_url: Optional[str] = None
    file_size_bytes: Optional[int] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    language: Optional[str] = None
    content_type: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.file_size_bytes is not None and self.file_size_bytes < 0:
            raise ValueError("File size cannot be negative")
        
        if self.page_count is not None and self.page_count < 0:
            raise ValueError("Page count cannot be negative")
        
        if self.word_count is not None and self.word_count < 0:
            raise ValueError("Word count cannot be negative")
    
    def add_tag(self, tag: str) -> "DocumentMetadata":
        """Add a tag (returns new instance due to immutability)."""
        new_tags = list(self.tags)
        if tag not in new_tags:
            new_tags.append(tag)
        
        return DocumentMetadata(
            title=self.title,
            author=self.author,
            created_date=self.created_date,
            modified_date=self.modified_date,
            source_url=self.source_url,
            file_size_bytes=self.file_size_bytes,
            page_count=self.page_count,
            word_count=self.word_count,
            language=self.language,
            content_type=self.content_type,
            tags=new_tags,
            custom_fields=dict(self.custom_fields)
        )
    
    def with_custom_field(self, key: str, value: Any) -> "DocumentMetadata":
        """Add a custom field (returns new instance)."""
        new_custom_fields = dict(self.custom_fields)
        new_custom_fields[key] = value
        
        return DocumentMetadata(
            title=self.title,
            author=self.author,
            created_date=self.created_date,
            modified_date=self.modified_date,
            source_url=self.source_url,
            file_size_bytes=self.file_size_bytes,
            page_count=self.page_count,
            word_count=self.word_count,
            language=self.language,
            content_type=self.content_type,
            tags=list(self.tags),
            custom_fields=new_custom_fields
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = {
            "title": self.title,
            "author": self.author,
            "source_url": self.source_url,
            "file_size_bytes": self.file_size_bytes,
            "page_count": self.page_count,
            "word_count": self.word_count,
            "language": self.language,
            "content_type": self.content_type,
            "tags": self.tags,
            **self.custom_fields
        }
        
        if self.created_date:
            data["created_date"] = self.created_date.isoformat()
        
        if self.modified_date:
            data["modified_date"] = self.modified_date.isoformat()
        
        return {k: v for k, v in data.items() if v is not None}