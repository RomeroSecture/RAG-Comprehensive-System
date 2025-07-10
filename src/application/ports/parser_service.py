from abc import ABC, abstractmethod
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class ParsedDocument:
    """Result of document parsing."""
    content: str
    metadata: Dict[str, str]
    file_type: str
    language: Optional[str] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None


class ParserService(ABC):
    """Port for document parsing service implementations."""
    
    @abstractmethod
    async def parse_document(self, file_path: str, metadata: Dict[str, str]) -> ParsedDocument:
        """Parse a document from the given file path."""
        pass
    
    @abstractmethod
    def supports_file_type(self, file_type: str) -> bool:
        """Check if the parser supports the given file type."""
        pass
    
    @abstractmethod
    def get_supported_file_types(self) -> list[str]:
        """Get list of supported file types."""
        pass