from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from ..value_objects.embedding import Embedding, HybridEmbedding, SparseEmbedding


class EmbeddingRepository(ABC):
    """Abstract repository interface for embedding persistence."""
    
    @abstractmethod
    async def save(self, embedding: Embedding, chunk_id: UUID) -> Embedding:
        """Save an embedding associated with a chunk."""
        pass
    
    @abstractmethod
    async def save_many(self, embeddings: List[tuple[Embedding, UUID]]) -> List[Embedding]:
        """Save multiple embeddings in batch. Each tuple contains (embedding, chunk_id)."""
        pass
    
    @abstractmethod
    async def get_by_id(self, embedding_id: UUID) -> Optional[Embedding]:
        """Retrieve an embedding by its ID."""
        pass
    
    @abstractmethod
    async def get_by_chunk_id(self, chunk_id: UUID) -> Optional[Embedding]:
        """Retrieve an embedding by its associated chunk ID."""
        pass
    
    @abstractmethod
    async def get_by_chunk_ids(self, chunk_ids: List[UUID]) -> List[Embedding]:
        """Retrieve embeddings for multiple chunks."""
        pass
    
    @abstractmethod
    async def delete_by_chunk_id(self, chunk_id: UUID) -> bool:
        """Delete an embedding by its chunk ID."""
        pass
    
    @abstractmethod
    async def update_vector(self, embedding_id: UUID, new_vector: List[float]) -> bool:
        """Update the vector of an existing embedding."""
        pass
    
    @abstractmethod
    async def exists(self, embedding_id: UUID) -> bool:
        """Check if an embedding exists."""
        pass
    
    @abstractmethod
    async def count_by_model(self, model: str) -> int:
        """Count embeddings created by a specific model."""
        pass


class SparseEmbeddingRepository(ABC):
    """Abstract repository for sparse embeddings (e.g., BM25)."""
    
    @abstractmethod
    async def save(self, embedding: SparseEmbedding, chunk_id: UUID) -> SparseEmbedding:
        """Save a sparse embedding."""
        pass
    
    @abstractmethod
    async def get_by_chunk_id(self, chunk_id: UUID) -> Optional[SparseEmbedding]:
        """Retrieve a sparse embedding by chunk ID."""
        pass
    
    @abstractmethod
    async def delete_by_chunk_id(self, chunk_id: UUID) -> bool:
        """Delete a sparse embedding by chunk ID."""
        pass
    
    @abstractmethod
    async def update_vocabulary_size(self, new_size: int) -> int:
        """Update vocabulary size for all embeddings. Returns count of updated embeddings."""
        pass


class HybridEmbeddingRepository(ABC):
    """Abstract repository for hybrid embeddings."""
    
    @abstractmethod
    async def save(self, embedding: HybridEmbedding, chunk_id: UUID) -> HybridEmbedding:
        """Save a hybrid embedding."""
        pass
    
    @abstractmethod
    async def get_by_chunk_id(self, chunk_id: UUID) -> Optional[HybridEmbedding]:
        """Retrieve a hybrid embedding by chunk ID."""
        pass
    
    @abstractmethod
    async def update_weights(self, chunk_id: UUID, dense_weight: float, sparse_weight: float) -> bool:
        """Update the weights of a hybrid embedding."""
        pass