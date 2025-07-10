from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

from ..entities.query import QueryComplexity, QueryIntent, RetrievalStrategy


@dataclass
class QueryProcessedEvent:
    """Domain event raised when a query is successfully processed."""
    
    query_id: UUID
    query_text: str
    retrieval_strategy: RetrievalStrategy
    documents_retrieved: int
    total_processing_time_ms: float
    retrieval_time_ms: float
    generation_time_ms: Optional[float] = None
    reranking_time_ms: Optional[float] = None
    user_id: Optional[UUID] = None
    session_id: Optional[UUID] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_type": "query_processed",
            "query_id": str(self.query_id),
            "query_text": self.query_text,
            "retrieval_strategy": self.retrieval_strategy.value,
            "documents_retrieved": self.documents_retrieved,
            "total_processing_time_ms": self.total_processing_time_ms,
            "retrieval_time_ms": self.retrieval_time_ms,
            "generation_time_ms": self.generation_time_ms,
            "reranking_time_ms": self.reranking_time_ms,
            "user_id": str(self.user_id) if self.user_id else None,
            "session_id": str(self.session_id) if self.session_id else None,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class QueryAnalyzedEvent:
    """Domain event raised when a query is analyzed for intent and complexity."""
    
    query_id: UUID
    query_text: str
    detected_intent: QueryIntent
    detected_complexity: QueryComplexity
    entities_extracted: List[Dict[str, str]]
    keywords_extracted: List[str]
    analysis_time_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_type": "query_analyzed",
            "query_id": str(self.query_id),
            "query_text": self.query_text,
            "detected_intent": self.detected_intent.value,
            "detected_complexity": self.detected_complexity.value,
            "entities_extracted": self.entities_extracted,
            "keywords_extracted": self.keywords_extracted,
            "analysis_time_ms": self.analysis_time_ms,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class QueryEnhancedEvent:
    """Domain event raised when a query is enhanced with expansions."""
    
    query_id: UUID
    original_query: str
    expanded_queries: List[str]
    rewritten_query: Optional[str]
    hypothetical_answer: Optional[str]
    enhancement_method: str
    enhancement_time_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_type": "query_enhanced",
            "query_id": str(self.query_id),
            "original_query": self.original_query,
            "expanded_queries": self.expanded_queries,
            "rewritten_query": self.rewritten_query,
            "hypothetical_answer": self.hypothetical_answer,
            "enhancement_method": self.enhancement_method,
            "enhancement_time_ms": self.enhancement_time_ms,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class SelfRAGDecisionEvent:
    """Domain event for Self-RAG retrieval decisions."""
    
    query_id: UUID
    iteration: int
    decision: str  # "retrieve", "skip_retrieval", "reformulate", "additional_retrieval"
    confidence_score: float
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_type": "self_rag_decision",
            "query_id": str(self.query_id),
            "iteration": self.iteration,
            "decision": self.decision,
            "confidence_score": self.confidence_score,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class MultimodalQueryProcessedEvent:
    """Domain event for multimodal query processing."""
    
    query_id: UUID
    text_query: str
    image_count: int
    vision_model_used: str
    cross_modal_retrieval: bool
    processing_time_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_type": "multimodal_query_processed",
            "query_id": str(self.query_id),
            "text_query": self.text_query,
            "image_count": self.image_count,
            "vision_model_used": self.vision_model_used,
            "cross_modal_retrieval": self.cross_modal_retrieval,
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp.isoformat()
        }