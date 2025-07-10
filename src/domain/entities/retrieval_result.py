from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID, uuid4


@dataclass
class RetrievedDocument:
    """Represents a document retrieved from the vector store."""
    
    id: UUID = field(default_factory=uuid4)
    document_id: UUID = field(default_factory=uuid4)
    chunk_id: Optional[UUID] = None
    content: str = ""
    metadata: Dict[str, any] = field(default_factory=dict)
    score: float = 0.0
    source: str = ""
    retrieval_method: str = ""
    rerank_score: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if not self.content:
            raise ValueError("Retrieved document content cannot be empty")
        
        if not 0 <= self.score <= 1:
            raise ValueError("Score must be between 0 and 1")
    
    def apply_rerank_score(self, new_score: float) -> None:
        """Apply a reranking score to the document."""
        if not 0 <= new_score <= 1:
            raise ValueError("Rerank score must be between 0 and 1")
        self.rerank_score = new_score
    
    @property
    def final_score(self) -> float:
        """Get the final score (rerank if available, otherwise original)."""
        return self.rerank_score if self.rerank_score is not None else self.score
    
    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary representation."""
        return {
            "id": str(self.id),
            "document_id": str(self.document_id),
            "chunk_id": str(self.chunk_id) if self.chunk_id else None,
            "content": self.content,
            "metadata": self.metadata,
            "score": self.score,
            "rerank_score": self.rerank_score,
            "source": self.source,
            "retrieval_method": self.retrieval_method
        }


@dataclass
class RetrievalResult:
    """Complete retrieval result with documents and metadata."""
    
    id: UUID = field(default_factory=uuid4)
    query_id: UUID = field(default_factory=uuid4)
    documents: List[RetrievedDocument] = field(default_factory=list)
    total_results: int = 0
    retrieval_strategy: str = ""
    retrieval_time_ms: float = 0.0
    reranking_time_ms: Optional[float] = None
    metadata: Dict[str, any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        self.total_results = len(self.documents)
    
    def add_document(self, document: RetrievedDocument) -> None:
        """Add a retrieved document to the result."""
        self.documents.append(document)
        self.total_results = len(self.documents)
    
    def sort_by_score(self, descending: bool = True) -> None:
        """Sort documents by their final score."""
        self.documents.sort(
            key=lambda doc: doc.final_score,
            reverse=descending
        )
    
    def filter_by_threshold(self, threshold: float) -> None:
        """Filter documents by score threshold."""
        self.documents = [
            doc for doc in self.documents
            if doc.final_score >= threshold
        ]
        self.total_results = len(self.documents)
    
    def get_top_k(self, k: int) -> List[RetrievedDocument]:
        """Get top k documents by score."""
        self.sort_by_score()
        return self.documents[:k]
    
    @property
    def average_score(self) -> float:
        """Calculate average score of retrieved documents."""
        if not self.documents:
            return 0.0
        return sum(doc.final_score for doc in self.documents) / len(self.documents)
    
    @property
    def total_time_ms(self) -> float:
        """Get total processing time."""
        total = self.retrieval_time_ms
        if self.reranking_time_ms:
            total += self.reranking_time_ms
        return total


@dataclass
class SelfRAGResult(RetrievalResult):
    """Extended retrieval result for Self-RAG with confidence and attempts."""
    
    confidence_score: float = 0.0
    retrieval_attempts: int = 1
    reformulated_queries: List[str] = field(default_factory=list)
    retrieval_decisions: List[Dict[str, any]] = field(default_factory=list)
    needs_additional_retrieval: bool = False
    
    def __post_init__(self):
        super().__post_init__()
        
        if not 0 <= self.confidence_score <= 1:
            raise ValueError("Confidence score must be between 0 and 1")
    
    def add_retrieval_decision(self, decision: Dict[str, any]) -> None:
        """Add a retrieval decision made during Self-RAG process."""
        self.retrieval_decisions.append({
            **decision,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def mark_low_confidence(self) -> None:
        """Mark result as low confidence requiring additional retrieval."""
        self.needs_additional_retrieval = True