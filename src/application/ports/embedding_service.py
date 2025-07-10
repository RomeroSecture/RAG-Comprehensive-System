from abc import ABC, abstractmethod
from typing import List
from src.domain.value_objects.embedding import Embedding


class EmbeddingService(ABC):
    """Port for embedding service implementations."""
    
    @abstractmethod
    async def embed_text(self, text: str) -> Embedding:
        """Generate embedding for a single text."""
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[Embedding]:
        """Generate embeddings for a batch of texts."""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this service."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the name of the embedding model being used."""
        pass