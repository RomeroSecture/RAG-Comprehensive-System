from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from uuid import UUID, uuid4


@dataclass(frozen=True)
class Embedding:
    """Immutable value object representing a vector embedding."""
    
    id: UUID
    vector: List[float]
    model: str
    dimensions: int
    
    def __post_init__(self):
        if not self.vector:
            raise ValueError("Embedding vector cannot be empty")
        
        if self.dimensions != len(self.vector):
            raise ValueError(f"Dimensions mismatch: expected {self.dimensions}, got {len(self.vector)}")
        
        if not self.model:
            raise ValueError("Embedding model must be specified")
    
    @classmethod
    def create(cls, vector: List[float], model: str) -> "Embedding":
        """Factory method to create an embedding."""
        return cls(
            id=uuid4(),
            vector=vector,
            model=model,
            dimensions=len(vector)
        )
    
    def to_numpy(self) -> np.ndarray:
        """Convert embedding to numpy array."""
        return np.array(self.vector, dtype=np.float32)
    
    def cosine_similarity(self, other: "Embedding") -> float:
        """Calculate cosine similarity with another embedding."""
        if self.dimensions != other.dimensions:
            raise ValueError("Embeddings must have the same dimensions")
        
        a = self.to_numpy()
        b = other.to_numpy()
        
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(dot_product / (norm_a * norm_b))
    
    def euclidean_distance(self, other: "Embedding") -> float:
        """Calculate Euclidean distance to another embedding."""
        if self.dimensions != other.dimensions:
            raise ValueError("Embeddings must have the same dimensions")
        
        a = self.to_numpy()
        b = other.to_numpy()
        
        return float(np.linalg.norm(a - b))
    
    def dot_product(self, other: "Embedding") -> float:
        """Calculate dot product with another embedding."""
        if self.dimensions != other.dimensions:
            raise ValueError("Embeddings must have the same dimensions")
        
        return float(np.dot(self.to_numpy(), other.to_numpy()))


@dataclass(frozen=True)
class SparseEmbedding:
    """Immutable value object for sparse embeddings (e.g., BM25)."""
    
    id: UUID
    indices: List[int]
    values: List[float]
    vocabulary_size: int
    
    def __post_init__(self):
        if len(self.indices) != len(self.values):
            raise ValueError("Indices and values must have the same length")
        
        if any(idx < 0 or idx >= self.vocabulary_size for idx in self.indices):
            raise ValueError("Invalid index in sparse embedding")
    
    @classmethod
    def create(cls, indices: List[int], values: List[float], vocabulary_size: int) -> "SparseEmbedding":
        """Factory method to create a sparse embedding."""
        return cls(
            id=uuid4(),
            indices=indices,
            values=values,
            vocabulary_size=vocabulary_size
        )
    
    def to_dense(self) -> List[float]:
        """Convert sparse embedding to dense representation."""
        dense = [0.0] * self.vocabulary_size
        for idx, val in zip(self.indices, self.values):
            dense[idx] = val
        return dense
    
    @property
    def nnz(self) -> int:
        """Get number of non-zero elements."""
        return len(self.indices)


@dataclass(frozen=True)
class HybridEmbedding:
    """Immutable value object combining dense and sparse embeddings."""
    
    dense: Embedding
    sparse: Optional[SparseEmbedding] = None
    weight_dense: float = 0.7
    weight_sparse: float = 0.3
    
    def __post_init__(self):
        if not 0 <= self.weight_dense <= 1:
            raise ValueError("Dense weight must be between 0 and 1")
        
        if not 0 <= self.weight_sparse <= 1:
            raise ValueError("Sparse weight must be between 0 and 1")
        
        if abs(self.weight_dense + self.weight_sparse - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1")
    
    def hybrid_similarity(self, other: "HybridEmbedding") -> float:
        """Calculate hybrid similarity score."""
        dense_sim = self.dense.cosine_similarity(other.dense)
        
        if self.sparse and other.sparse:
            # Simple sparse similarity (could be improved with proper BM25)
            sparse_sim = self._sparse_similarity(self.sparse, other.sparse)
            return self.weight_dense * dense_sim + self.weight_sparse * sparse_sim
        
        return dense_sim
    
    @staticmethod
    def _sparse_similarity(a: SparseEmbedding, b: SparseEmbedding) -> float:
        """Calculate similarity between sparse embeddings."""
        if a.vocabulary_size != b.vocabulary_size:
            return 0.0
        
        # Convert to sets for efficient intersection
        a_terms = set(zip(a.indices, a.values))
        b_terms = set(zip(b.indices, b.values))
        
        # Calculate intersection
        common_terms = 0.0
        for idx, val_a in a_terms:
            for b_idx, val_b in b_terms:
                if idx == b_idx:
                    common_terms += val_a * val_b
        
        # Normalize
        norm_a = sum(v * v for _, v in a_terms) ** 0.5
        norm_b = sum(v * v for _, v in b_terms) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return common_terms / (norm_a * norm_b)