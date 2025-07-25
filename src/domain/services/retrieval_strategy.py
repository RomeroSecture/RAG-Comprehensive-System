from abc import ABC, abstractmethod
from typing import List, Optional

from ..entities.query import Query, RetrievalStrategy
from ..entities.retrieval_result import RetrievalResult, RetrievedDocument
from ..value_objects.embedding import Embedding


class RetrievalStrategyService(ABC):
    """Abstract service for different retrieval strategies."""
    
    @abstractmethod
    async def retrieve(self, query: Query, query_embedding: Optional[Embedding] = None) -> RetrievalResult:
        """Execute retrieval based on the strategy."""
        pass
    
    @abstractmethod
    def supports_strategy(self, strategy: RetrievalStrategy) -> bool:
        """Check if this service supports a given strategy."""
        pass


class SemanticRetrievalStrategy(RetrievalStrategyService):
    """Pure semantic similarity search strategy."""
    
    def supports_strategy(self, strategy: RetrievalStrategy) -> bool:
        return strategy == RetrievalStrategy.SEMANTIC
    
    async def retrieve(self, query: Query, query_embedding: Optional[Embedding] = None) -> RetrievalResult:
        """Execute semantic retrieval (placeholder implementation)."""
        # This would connect to the actual vector store
        return RetrievalResult(
            query_id=query.id,
            retrieval_strategy="semantic"
        )


class HybridRetrievalStrategy(RetrievalStrategyService):
    """Hybrid search combining semantic and keyword search."""
    
    def supports_strategy(self, strategy: RetrievalStrategy) -> bool:
        return strategy == RetrievalStrategy.HYBRID
    
    async def retrieve(self, query: Query, query_embedding: Optional[Embedding] = None) -> RetrievalResult:
        """Execute hybrid retrieval (placeholder implementation)."""
        # This would combine semantic and keyword search
        return RetrievalResult(
            query_id=query.id,
            retrieval_strategy="hybrid"
        )


class GraphRetrievalStrategy(RetrievalStrategyService):
    """Graph-enhanced retrieval with multi-hop reasoning."""
    
    def supports_strategy(self, strategy: RetrievalStrategy) -> bool:
        return strategy == RetrievalStrategy.GRAPH
    
    async def retrieve(self, query: Query, query_embedding: Optional[Embedding] = None) -> RetrievalResult:
        """Execute graph-based retrieval (placeholder implementation)."""
        # This would query the knowledge graph
        return RetrievalResult(
            query_id=query.id,
            retrieval_strategy="graph"
        )


class SelfRAGStrategy(RetrievalStrategyService):
    """Self-reflective RAG with dynamic retrieval decisions."""
    
    def supports_strategy(self, strategy: RetrievalStrategy) -> bool:
        return strategy == RetrievalStrategy.SELF_RAG
    
    async def retrieve(self, query: Query, query_embedding: Optional[Embedding] = None) -> RetrievalResult:
        """Execute Self-RAG retrieval (placeholder implementation)."""
        # This would implement self-reflective retrieval
        return RetrievalResult(
            query_id=query.id,
            retrieval_strategy="self_rag"
        )


class RetrievalStrategyFactory:
    """Factory for creating retrieval strategies."""
    
    def __init__(self):
        self._strategies: List[RetrievalStrategyService] = []
    
    def register_strategy(self, strategy: RetrievalStrategyService) -> None:
        """Register a retrieval strategy."""
        self._strategies.append(strategy)
    
    def get_strategy(self, strategy_type: RetrievalStrategy) -> RetrievalStrategyService:
        """Get the appropriate retrieval strategy."""
        for strategy in self._strategies:
            if strategy.supports_strategy(strategy_type):
                return strategy
        
        raise ValueError(f"No strategy registered for {strategy_type.value}")
    
    def list_available_strategies(self) -> List[RetrievalStrategy]:
        """List all available retrieval strategies."""
        strategies = []
        for strategy_enum in RetrievalStrategy:
            for service in self._strategies:
                if service.supports_strategy(strategy_enum):
                    strategies.append(strategy_enum)
                    break
        return strategies


class RecursiveRetrievalService:
    """Service for recursive retrieval with iterative deepening."""
    
    def __init__(self, base_strategy: RetrievalStrategyService):
        self.base_strategy = base_strategy
    
    async def retrieve_recursive(
        self,
        query: Query,
        max_iterations: int = 3,
        expansion_factor: int = 2
    ) -> RetrievalResult:
        """Perform recursive retrieval with expanding search."""
        # Implementation would go here
        pass


class AdaptiveRetrievalService:
    """Service that adapts retrieval strategy based on query complexity."""
    
    def __init__(self, strategy_factory: RetrievalStrategyFactory):
        self.strategy_factory = strategy_factory
    
    def select_strategy(self, query: Query) -> RetrievalStrategy:
        """Select the best retrieval strategy for a query."""
        # Simple logic for now, can be enhanced with ML
        if query.complexity is None:
            return RetrievalStrategy.HYBRID
        
        complexity_to_strategy = {
            "simple": RetrievalStrategy.SEMANTIC,
            "moderate": RetrievalStrategy.HYBRID,
            "complex": RetrievalStrategy.SELF_RAG,
            "multi_hop": RetrievalStrategy.GRAPH
        }
        
        return complexity_to_strategy.get(
            query.complexity.value,
            RetrievalStrategy.HYBRID
        )
    
    async def retrieve_adaptive(self, query: Query) -> RetrievalResult:
        """Retrieve using adaptive strategy selection."""
        selected_strategy = self.select_strategy(query)
        query.strategy = selected_strategy
        
        strategy_service = self.strategy_factory.get_strategy(selected_strategy)
        return await strategy_service.retrieve(query)