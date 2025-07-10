from abc import ABC, abstractmethod
from typing import List
from uuid import UUID
from src.domain.entities.document import DocumentChunk


class ChunkingService(ABC):
    """Port for document chunking service implementations."""
    
    @abstractmethod
    async def create_chunks(
        self,
        content: str,
        document_id: UUID,
        strategy: str = "recursive",
        chunk_size: int = 1000,
        overlap: int = 200,
        **kwargs
    ) -> List[DocumentChunk]:
        """Create chunks from document content."""
        pass
    
    @abstractmethod
    def get_available_strategies(self) -> List[str]:
        """Get list of available chunking strategies."""
        pass
    
    @abstractmethod
    def estimate_chunk_count(self, content: str, chunk_size: int) -> int:
        """Estimate number of chunks that will be created."""
        pass