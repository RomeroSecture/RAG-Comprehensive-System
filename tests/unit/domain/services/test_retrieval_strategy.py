import pytest
from uuid import uuid4

from src.domain.services.retrieval_strategy import (
    RetrievalStrategyService,
    SemanticRetrievalStrategy,
    HybridRetrievalStrategy,
    GraphRetrievalStrategy,
    SelfRAGStrategy,
    RetrievalStrategyFactory,
    RecursiveRetrievalService,
    AdaptiveRetrievalService
)
from src.domain.entities.query import Query, RetrievalStrategy, QueryComplexity
from src.domain.entities.retrieval_result import RetrievalResult, RetrievedDocument
from src.domain.value_objects.embedding import Embedding


class TestRetrievalStrategyService:
    """Unit tests for the abstract RetrievalStrategyService."""
    
    def test_retrieval_strategy_service_is_abstract(self):
        """Test that RetrievalStrategyService cannot be instantiated."""
        with pytest.raises(TypeError):
            RetrievalStrategyService()


class TestSemanticRetrievalStrategy:
    """Unit tests for SemanticRetrievalStrategy."""
    
    def test_semantic_strategy_supports_semantic(self):
        """Test that semantic strategy supports SEMANTIC type."""
        strategy = SemanticRetrievalStrategy()
        assert strategy.supports_strategy(RetrievalStrategy.SEMANTIC) is True
        
    def test_semantic_strategy_does_not_support_others(self):
        """Test that semantic strategy doesn't support other types."""
        strategy = SemanticRetrievalStrategy()
        assert strategy.supports_strategy(RetrievalStrategy.HYBRID) is False
        assert strategy.supports_strategy(RetrievalStrategy.GRAPH) is False
        assert strategy.supports_strategy(RetrievalStrategy.SELF_RAG) is False


class TestHybridRetrievalStrategy:
    """Unit tests for HybridRetrievalStrategy."""
    
    def test_hybrid_strategy_supports_hybrid(self):
        """Test that hybrid strategy supports HYBRID type."""
        strategy = HybridRetrievalStrategy()
        assert strategy.supports_strategy(RetrievalStrategy.HYBRID) is True
        
    def test_hybrid_strategy_does_not_support_others(self):
        """Test that hybrid strategy doesn't support other types."""
        strategy = HybridRetrievalStrategy()
        assert strategy.supports_strategy(RetrievalStrategy.SEMANTIC) is False
        assert strategy.supports_strategy(RetrievalStrategy.GRAPH) is False
        assert strategy.supports_strategy(RetrievalStrategy.SELF_RAG) is False


class TestGraphRetrievalStrategy:
    """Unit tests for GraphRetrievalStrategy."""
    
    def test_graph_strategy_supports_graph(self):
        """Test that graph strategy supports GRAPH type."""
        strategy = GraphRetrievalStrategy()
        assert strategy.supports_strategy(RetrievalStrategy.GRAPH) is True
        
    def test_graph_strategy_does_not_support_others(self):
        """Test that graph strategy doesn't support other types."""
        strategy = GraphRetrievalStrategy()
        assert strategy.supports_strategy(RetrievalStrategy.SEMANTIC) is False
        assert strategy.supports_strategy(RetrievalStrategy.HYBRID) is False
        assert strategy.supports_strategy(RetrievalStrategy.SELF_RAG) is False


class TestSelfRAGStrategy:
    """Unit tests for SelfRAGStrategy."""
    
    def test_self_rag_strategy_supports_self_rag(self):
        """Test that Self-RAG strategy supports SELF_RAG type."""
        strategy = SelfRAGStrategy()
        assert strategy.supports_strategy(RetrievalStrategy.SELF_RAG) is True
        
    def test_self_rag_strategy_does_not_support_others(self):
        """Test that Self-RAG strategy doesn't support other types."""
        strategy = SelfRAGStrategy()
        assert strategy.supports_strategy(RetrievalStrategy.SEMANTIC) is False
        assert strategy.supports_strategy(RetrievalStrategy.HYBRID) is False
        assert strategy.supports_strategy(RetrievalStrategy.GRAPH) is False


class TestRetrievalStrategyFactory:
    """Unit tests for RetrievalStrategyFactory."""
    
    def test_factory_creation(self):
        """Test creating an empty factory."""
        factory = RetrievalStrategyFactory()
        assert factory._strategies == []
    
    def test_register_strategy(self):
        """Test registering strategies."""
        factory = RetrievalStrategyFactory()
        
        semantic_strategy = SemanticRetrievalStrategy()
        hybrid_strategy = HybridRetrievalStrategy()
        
        factory.register_strategy(semantic_strategy)
        factory.register_strategy(hybrid_strategy)
        
        assert len(factory._strategies) == 2
        assert semantic_strategy in factory._strategies
        assert hybrid_strategy in factory._strategies
    
    def test_get_strategy(self):
        """Test getting the correct strategy."""
        factory = RetrievalStrategyFactory()
        
        semantic_strategy = SemanticRetrievalStrategy()
        hybrid_strategy = HybridRetrievalStrategy()
        
        factory.register_strategy(semantic_strategy)
        factory.register_strategy(hybrid_strategy)
        
        # Get semantic strategy
        result_semantic = factory.get_strategy(RetrievalStrategy.SEMANTIC)
        assert result_semantic == semantic_strategy
        
        # Get hybrid strategy
        result_hybrid = factory.get_strategy(RetrievalStrategy.HYBRID)
        assert result_hybrid == hybrid_strategy
    
    def test_get_strategy_not_registered_raises_error(self):
        """Test that getting unregistered strategy raises error."""
        factory = RetrievalStrategyFactory()
        
        # Register only semantic
        factory.register_strategy(SemanticRetrievalStrategy())
        
        # Try to get unregistered strategy
        with pytest.raises(ValueError, match="No strategy registered for graph"):
            factory.get_strategy(RetrievalStrategy.GRAPH)
    
    def test_list_available_strategies(self):
        """Test listing available strategies."""
        factory = RetrievalStrategyFactory()
        
        # Empty factory
        assert factory.list_available_strategies() == []
        
        # Add strategies
        factory.register_strategy(SemanticRetrievalStrategy())
        factory.register_strategy(HybridRetrievalStrategy())
        factory.register_strategy(GraphRetrievalStrategy())
        
        available = factory.list_available_strategies()
        assert len(available) == 3
        assert RetrievalStrategy.SEMANTIC in available
        assert RetrievalStrategy.HYBRID in available
        assert RetrievalStrategy.GRAPH in available
        assert RetrievalStrategy.SELF_RAG not in available
    
    def test_register_multiple_strategies_for_same_type(self):
        """Test that only one strategy is returned for each type."""
        factory = RetrievalStrategyFactory()
        
        # Register two semantic strategies
        strategy1 = SemanticRetrievalStrategy()
        strategy2 = SemanticRetrievalStrategy()
        
        factory.register_strategy(strategy1)
        factory.register_strategy(strategy2)
        
        # Should still list SEMANTIC only once
        available = factory.list_available_strategies()
        assert available.count(RetrievalStrategy.SEMANTIC) == 1
        
        # Should get the first registered strategy
        result = factory.get_strategy(RetrievalStrategy.SEMANTIC)
        assert result == strategy1


class TestRecursiveRetrievalService:
    """Unit tests for RecursiveRetrievalService."""
    
    def test_recursive_retrieval_creation(self):
        """Test creating recursive retrieval service."""
        base_strategy = SemanticRetrievalStrategy()
        service = RecursiveRetrievalService(base_strategy)
        
        assert service.base_strategy == base_strategy
    
    @pytest.mark.asyncio
    async def test_retrieve_recursive_placeholder(self):
        """Test recursive retrieval (placeholder test)."""
        base_strategy = SemanticRetrievalStrategy()
        service = RecursiveRetrievalService(base_strategy)
        
        query = Query(text="test query")
        
        # Since implementation is not complete, it will return None
        result = await service.retrieve_recursive(query, max_iterations=3)
        assert result is None  # Placeholder implementation returns None


class TestAdaptiveRetrievalService:
    """Unit tests for AdaptiveRetrievalService."""
    
    def test_adaptive_retrieval_creation(self):
        """Test creating adaptive retrieval service."""
        factory = RetrievalStrategyFactory()
        service = AdaptiveRetrievalService(factory)
        
        assert service.strategy_factory == factory
    
    def test_select_strategy_no_complexity(self):
        """Test strategy selection with no complexity."""
        factory = RetrievalStrategyFactory()
        service = AdaptiveRetrievalService(factory)
        
        query = Query(text="test", complexity=None)
        selected = service.select_strategy(query)
        
        assert selected == RetrievalStrategy.HYBRID
    
    def test_select_strategy_simple_complexity(self):
        """Test strategy selection with simple complexity."""
        factory = RetrievalStrategyFactory()
        service = AdaptiveRetrievalService(factory)
        
        query = Query(text="test", complexity=QueryComplexity.SIMPLE)
        selected = service.select_strategy(query)
        
        assert selected == RetrievalStrategy.SEMANTIC
    
    def test_select_strategy_moderate_complexity(self):
        """Test strategy selection with moderate complexity."""
        factory = RetrievalStrategyFactory()
        service = AdaptiveRetrievalService(factory)
        
        query = Query(text="test", complexity=QueryComplexity.MODERATE)
        selected = service.select_strategy(query)
        
        assert selected == RetrievalStrategy.HYBRID
    
    def test_select_strategy_complex_complexity(self):
        """Test strategy selection with complex complexity."""
        factory = RetrievalStrategyFactory()
        service = AdaptiveRetrievalService(factory)
        
        query = Query(text="test", complexity=QueryComplexity.COMPLEX)
        selected = service.select_strategy(query)
        
        assert selected == RetrievalStrategy.SELF_RAG
    
    def test_select_strategy_multi_hop_complexity(self):
        """Test strategy selection with multi-hop complexity."""
        factory = RetrievalStrategyFactory()
        service = AdaptiveRetrievalService(factory)
        
        query = Query(text="test", complexity=QueryComplexity.MULTI_HOP)
        selected = service.select_strategy(query)
        
        assert selected == RetrievalStrategy.GRAPH
    
    @pytest.mark.asyncio
    async def test_retrieve_adaptive(self):
        """Test adaptive retrieval."""
        # Create a mock strategy that returns a result
        class MockSemanticStrategy(SemanticRetrievalStrategy):
            async def retrieve(self, query, query_embedding=None):
                return RetrievalResult(
                    query_id=query.id,
                    documents=[
                        RetrievedDocument(content="test doc", score=0.9)
                    ],
                    retrieval_strategy="semantic"
                )
        
        factory = RetrievalStrategyFactory()
        factory.register_strategy(MockSemanticStrategy())
        
        service = AdaptiveRetrievalService(factory)
        
        # Create query with simple complexity
        query = Query(text="simple query", complexity=QueryComplexity.SIMPLE)
        
        result = await service.retrieve_adaptive(query)
        
        assert isinstance(result, RetrievalResult)
        assert result.retrieval_strategy == "semantic"
        assert len(result.documents) == 1
        assert query.strategy == RetrievalStrategy.SEMANTIC
    
    @pytest.mark.asyncio
    async def test_retrieve_adaptive_updates_query_strategy(self):
        """Test that adaptive retrieval updates the query strategy."""
        # Create mock strategies
        class MockHybridStrategy(HybridRetrievalStrategy):
            async def retrieve(self, query, query_embedding=None):
                return RetrievalResult(
                    query_id=query.id,
                    retrieval_strategy="hybrid"
                )
        
        factory = RetrievalStrategyFactory()
        factory.register_strategy(MockHybridStrategy())
        
        service = AdaptiveRetrievalService(factory)
        
        # Create query without strategy
        query = Query(text="test", strategy=None, complexity=QueryComplexity.MODERATE)
        
        await service.retrieve_adaptive(query)
        
        # Query strategy should be updated
        assert query.strategy == RetrievalStrategy.HYBRID