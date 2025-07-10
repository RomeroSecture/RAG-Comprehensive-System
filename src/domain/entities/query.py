from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from uuid import UUID, uuid4


class QueryComplexity(Enum):
    """Query complexity levels for adaptive retrieval."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    MULTI_HOP = "multi_hop"


class QueryIntent(Enum):
    """Query intent classification."""
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    EXPLORATORY = "exploratory"
    NAVIGATIONAL = "navigational"


class RetrievalStrategy(Enum):
    """Available retrieval strategies."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    GRAPH = "graph"
    SELF_RAG = "self_rag"
    CORRECTIVE_RAG = "corrective_rag"
    LONG_RAG = "long_rag"
    MULTIMODAL = "multimodal"


@dataclass
class Query:
    """Core query entity for the RAG system."""
    
    id: UUID = field(default_factory=uuid4)
    text: str = ""
    user_id: Optional[UUID] = None
    session_id: Optional[UUID] = None
    metadata: Dict[str, any] = field(default_factory=dict)
    intent: Optional[QueryIntent] = None
    complexity: Optional[QueryComplexity] = None
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    max_results: int = 10
    similarity_threshold: float = 0.7
    include_metadata: bool = True
    language: Optional[str] = None
    filters: Dict[str, any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if not self.text:
            raise ValueError("Query text cannot be empty")
        
        if self.max_results < 1:
            raise ValueError("Max results must be at least 1")
        
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError("Similarity threshold must be between 0 and 1")
    
    def set_intent(self, intent: QueryIntent) -> None:
        """Set the query intent."""
        self.intent = intent
    
    def set_complexity(self, complexity: QueryComplexity) -> None:
        """Set the query complexity."""
        self.complexity = complexity
    
    def add_filter(self, key: str, value: any) -> None:
        """Add a metadata filter to the query."""
        self.filters[key] = value
    
    def to_retrieval_params(self) -> Dict[str, any]:
        """Convert query to retrieval parameters."""
        return {
            "query_text": self.text,
            "strategy": self.strategy.value,
            "max_results": self.max_results,
            "similarity_threshold": self.similarity_threshold,
            "filters": self.filters,
            "include_metadata": self.include_metadata
        }


@dataclass
class EnhancedQuery:
    """Enhanced query with expansions and rewrites."""
    
    original_query: Query
    expanded_queries: List[str] = field(default_factory=list)
    rewritten_query: Optional[str] = None
    hypothetical_answer: Optional[str] = None
    entities: List[Dict[str, str]] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def add_expansion(self, expanded_text: str) -> None:
        """Add an expanded query variant."""
        if expanded_text and expanded_text not in self.expanded_queries:
            self.expanded_queries.append(expanded_text)
    
    def add_entity(self, entity_text: str, entity_type: str) -> None:
        """Add an extracted entity."""
        self.entities.append({
            "text": entity_text,
            "type": entity_type
        })
    
    def get_all_query_variants(self) -> List[str]:
        """Get all query variants including original."""
        variants = [self.original_query.text]
        
        if self.rewritten_query:
            variants.append(self.rewritten_query)
        
        variants.extend(self.expanded_queries)
        
        return list(set(variants))  # Remove duplicates