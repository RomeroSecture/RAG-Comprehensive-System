from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from ..entities.retrieval_result import RetrievedDocument
from ..value_objects.embedding import Embedding, HybridEmbedding
from ..value_objects.similarity_score import SimilarityMetric


class VectorStoreRepository(ABC):
    """Abstract repository interface for vector store operations."""
    
    @abstractmethod
    async def upsert(self, chunk_id: UUID, embedding: Embedding, metadata: Dict[str, any]) -> bool:
        """Insert or update a vector with metadata."""
        pass
    
    @abstractmethod
    async def upsert_batch(self, items: List[Tuple[UUID, Embedding, Dict[str, any]]]) -> int:
        """Batch upsert vectors. Returns count of successful upserts."""
        pass
    
    @abstractmethod
    async def search(
        self,
        query_embedding: Embedding,
        k: int = 10,
        filters: Optional[Dict[str, any]] = None,
        similarity_metric: SimilarityMetric = SimilarityMetric.COSINE
    ) -> List[RetrievedDocument]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    async def delete(self, chunk_id: UUID) -> bool:
        """Delete a vector by chunk ID."""
        pass
    
    @abstractmethod
    async def delete_batch(self, chunk_ids: List[UUID]) -> int:
        """Delete multiple vectors. Returns count of deleted vectors."""
        pass
    
    @abstractmethod
    async def get_by_id(self, chunk_id: UUID) -> Optional[Tuple[Embedding, Dict[str, any]]]:
        """Retrieve a vector and its metadata by chunk ID."""
        pass
    
    @abstractmethod
    async def count(self) -> int:
        """Get total count of vectors in the store."""
        pass
    
    @abstractmethod
    async def create_index(self, index_type: str, params: Dict[str, any]) -> bool:
        """Create an index for efficient search."""
        pass
    
    @abstractmethod
    async def optimize(self) -> bool:
        """Optimize the vector store for better performance."""
        pass


class HybridVectorStoreRepository(ABC):
    """Abstract repository for hybrid search combining vector and keyword search."""
    
    @abstractmethod
    async def hybrid_search(
        self,
        query_embedding: Embedding,
        query_text: str,
        k: int = 10,
        alpha: float = 0.5,  # Weight for vector search vs keyword search
        filters: Optional[Dict[str, any]] = None
    ) -> List[RetrievedDocument]:
        """Perform hybrid search combining vector and keyword search."""
        pass
    
    @abstractmethod
    async def keyword_search(
        self,
        query_text: str,
        k: int = 10,
        filters: Optional[Dict[str, any]] = None
    ) -> List[RetrievedDocument]:
        """Perform keyword-based search (e.g., BM25)."""
        pass
    
    @abstractmethod
    async def update_text_index(self, chunk_id: UUID, text: str) -> bool:
        """Update the text index for keyword search."""
        pass


class GraphVectorStoreRepository(ABC):
    """Abstract repository for graph-enhanced vector search."""
    
    @abstractmethod
    async def add_edge(self, from_chunk_id: UUID, to_chunk_id: UUID, edge_type: str, weight: float = 1.0) -> bool:
        """Add an edge between two chunks in the graph."""
        pass
    
    @abstractmethod
    async def graph_search(
        self,
        query_embedding: Embedding,
        k: int = 10,
        max_hops: int = 2,
        filters: Optional[Dict[str, any]] = None
    ) -> List[RetrievedDocument]:
        """Search with graph traversal for multi-hop reasoning."""
        pass
    
    @abstractmethod
    async def get_neighbors(self, chunk_id: UUID, edge_type: Optional[str] = None) -> List[UUID]:
        """Get neighboring chunks in the graph."""
        pass
    
    @abstractmethod
    async def compute_pagerank(self, damping_factor: float = 0.85) -> Dict[UUID, float]:
        """Compute PageRank scores for all chunks."""
        pass