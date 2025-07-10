from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SimilarityMetric(Enum):
    """Types of similarity metrics."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    JACCARD = "jaccard"
    BM25 = "bm25"


@dataclass(frozen=True)
class SimilarityScore:
    """Immutable value object representing a similarity score."""
    
    value: float
    metric: SimilarityMetric
    min_value: float = 0.0
    max_value: float = 1.0
    
    def __post_init__(self):
        if not self.min_value <= self.value <= self.max_value:
            raise ValueError(
                f"Score {self.value} is outside valid range "
                f"[{self.min_value}, {self.max_value}] for metric {self.metric.value}"
            )
    
    @classmethod
    def cosine(cls, value: float) -> "SimilarityScore":
        """Create a cosine similarity score."""
        return cls(value=value, metric=SimilarityMetric.COSINE, min_value=-1.0, max_value=1.0)
    
    @classmethod
    def euclidean(cls, value: float, max_distance: float = 100.0) -> "SimilarityScore":
        """Create a Euclidean distance score (inverted to similarity)."""
        # Convert distance to similarity (closer = higher similarity)
        similarity = 1.0 - (value / max_distance)
        return cls(value=similarity, metric=SimilarityMetric.EUCLIDEAN)
    
    @classmethod
    def dot_product(cls, value: float) -> "SimilarityScore":
        """Create a dot product similarity score."""
        # Dot product can be unbounded, but we'll normalize it
        return cls(value=value, metric=SimilarityMetric.DOT_PRODUCT, min_value=-float('inf'), max_value=float('inf'))
    
    @classmethod
    def bm25(cls, value: float) -> "SimilarityScore":
        """Create a BM25 similarity score."""
        return cls(value=value, metric=SimilarityMetric.BM25, min_value=0.0, max_value=float('inf'))
    
    def normalize(self) -> float:
        """Normalize score to [0, 1] range."""
        if self.metric == SimilarityMetric.COSINE:
            # Convert from [-1, 1] to [0, 1]
            return (self.value + 1.0) / 2.0
        elif self.metric == SimilarityMetric.DOT_PRODUCT:
            # Sigmoid normalization for unbounded values
            import math
            return 1.0 / (1.0 + math.exp(-self.value))
        elif self.metric == SimilarityMetric.BM25:
            # Log normalization for BM25
            import math
            return math.log(1 + self.value) / math.log(1 + self.max_value)
        else:
            # Already in [0, 1] range
            return self.value
    
    def is_above_threshold(self, threshold: float) -> bool:
        """Check if score is above a given threshold."""
        return self.normalize() >= threshold
    
    def combine_with(self, other: "SimilarityScore", weight: float = 0.5) -> "SimilarityScore":
        """Combine with another score using weighted average."""
        if not 0 <= weight <= 1:
            raise ValueError("Weight must be between 0 and 1")
        
        # Both scores must be normalized for fair combination
        combined_value = weight * self.normalize() + (1 - weight) * other.normalize()
        
        return SimilarityScore(
            value=combined_value,
            metric=SimilarityMetric.COSINE,  # Default to cosine for combined scores
            min_value=0.0,
            max_value=1.0
        )


@dataclass(frozen=True)
class RerankingScore:
    """Immutable value object for reranking scores."""
    
    original_score: SimilarityScore
    rerank_score: float
    reranker_model: str
    confidence: Optional[float] = None
    
    def __post_init__(self):
        if not 0 <= self.rerank_score <= 1:
            raise ValueError("Rerank score must be between 0 and 1")
        
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        
        if not self.reranker_model:
            raise ValueError("Reranker model must be specified")
    
    @property
    def score_delta(self) -> float:
        """Calculate the difference between rerank and original scores."""
        return self.rerank_score - self.original_score.normalize()
    
    @property
    def final_score(self) -> float:
        """Get the final score after reranking."""
        return self.rerank_score
    
    def weighted_combination(self, original_weight: float = 0.3) -> float:
        """Combine original and rerank scores with weights."""
        if not 0 <= original_weight <= 1:
            raise ValueError("Weight must be between 0 and 1")
        
        return (original_weight * self.original_score.normalize() + 
                (1 - original_weight) * self.rerank_score)