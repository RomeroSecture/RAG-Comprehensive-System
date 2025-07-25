import pytest
from datetime import datetime
from uuid import UUID, uuid4

from src.domain.entities.query import (
    Query, 
    EnhancedQuery,
    QueryComplexity,
    QueryIntent,
    RetrievalStrategy
)


class TestQueryEnums:
    """Test query-related enumerations."""
    
    def test_query_complexity_values(self):
        """Test QueryComplexity enum has all expected values."""
        assert QueryComplexity.SIMPLE.value == "simple"
        assert QueryComplexity.MODERATE.value == "moderate"
        assert QueryComplexity.COMPLEX.value == "complex"
        assert QueryComplexity.MULTI_HOP.value == "multi_hop"
    
    def test_query_intent_values(self):
        """Test QueryIntent enum has all expected values."""
        assert QueryIntent.FACTUAL.value == "factual"
        assert QueryIntent.ANALYTICAL.value == "analytical"
        assert QueryIntent.COMPARATIVE.value == "comparative"
        assert QueryIntent.EXPLORATORY.value == "exploratory"
        assert QueryIntent.NAVIGATIONAL.value == "navigational"
    
    def test_retrieval_strategy_values(self):
        """Test RetrievalStrategy enum has all expected values."""
        strategies = [
            ("SEMANTIC", "semantic"),
            ("KEYWORD", "keyword"),
            ("HYBRID", "hybrid"),
            ("GRAPH", "graph"),
            ("SELF_RAG", "self_rag"),
            ("CORRECTIVE_RAG", "corrective_rag"),
            ("LONG_RAG", "long_rag"),
            ("MULTIMODAL", "multimodal")
        ]
        
        for enum_name, expected_value in strategies:
            assert getattr(RetrievalStrategy, enum_name).value == expected_value


class TestQuery:
    """Unit tests for Query entity."""
    
    def test_query_creation_with_valid_data(self, sample_query_data):
        """Test query creation with valid data."""
        query = Query(
            text=sample_query_data["text"],
            user_id=sample_query_data["user_id"],
            session_id=sample_query_data["session_id"],
            metadata=sample_query_data["metadata"],
            max_results=sample_query_data["max_results"],
            similarity_threshold=sample_query_data["similarity_threshold"]
        )
        
        assert isinstance(query.id, UUID)
        assert query.text == sample_query_data["text"]
        assert query.user_id == sample_query_data["user_id"]
        assert query.session_id == sample_query_data["session_id"]
        assert query.metadata == sample_query_data["metadata"]
        assert query.max_results == sample_query_data["max_results"]
        assert query.similarity_threshold == sample_query_data["similarity_threshold"]
        assert query.strategy == RetrievalStrategy.HYBRID  # Default
        assert query.include_metadata is True  # Default
        assert query.filters == {}  # Default
        assert isinstance(query.created_at, datetime)
    
    def test_query_creation_with_empty_text_raises_error(self):
        """Test that creating a query with empty text raises ValueError."""
        with pytest.raises(ValueError, match="Query text cannot be empty"):
            Query(text="")
    
    def test_query_creation_with_invalid_max_results_raises_error(self):
        """Test that max_results < 1 raises ValueError."""
        with pytest.raises(ValueError, match="Max results must be at least 1"):
            Query(text="test query", max_results=0)
    
    def test_query_creation_with_invalid_similarity_threshold_raises_error(self):
        """Test that similarity threshold outside [0,1] raises ValueError."""
        # Test threshold > 1
        with pytest.raises(ValueError, match="Similarity threshold must be between 0 and 1"):
            Query(text="test query", similarity_threshold=1.5)
        
        # Test threshold < 0
        with pytest.raises(ValueError, match="Similarity threshold must be between 0 and 1"):
            Query(text="test query", similarity_threshold=-0.1)
    
    def test_query_with_custom_strategy(self):
        """Test creating query with custom retrieval strategy."""
        query = Query(
            text="test query",
            strategy=RetrievalStrategy.SELF_RAG
        )
        assert query.strategy == RetrievalStrategy.SELF_RAG
    
    def test_set_intent(self):
        """Test setting query intent."""
        query = Query(text="What is machine learning?")
        assert query.intent is None
        
        query.set_intent(QueryIntent.FACTUAL)
        assert query.intent == QueryIntent.FACTUAL
    
    def test_set_complexity(self):
        """Test setting query complexity."""
        query = Query(text="Complex multi-step question")
        assert query.complexity is None
        
        query.set_complexity(QueryComplexity.COMPLEX)
        assert query.complexity == QueryComplexity.COMPLEX
    
    def test_add_filter(self):
        """Test adding metadata filters."""
        query = Query(text="test query")
        assert query.filters == {}
        
        query.add_filter("category", "technology")
        assert query.filters["category"] == "technology"
        
        query.add_filter("year", 2024)
        assert query.filters["year"] == 2024
        assert len(query.filters) == 2
    
    def test_to_retrieval_params(self):
        """Test conversion to retrieval parameters."""
        query = Query(
            text="test query",
            strategy=RetrievalStrategy.SEMANTIC,
            max_results=20,
            similarity_threshold=0.8,
            include_metadata=False
        )
        query.add_filter("lang", "en")
        
        params = query.to_retrieval_params()
        
        assert params["query_text"] == "test query"
        assert params["strategy"] == "semantic"
        assert params["max_results"] == 20
        assert params["similarity_threshold"] == 0.8
        assert params["filters"] == {"lang": "en"}
        assert params["include_metadata"] is False
    
    def test_query_boundary_similarity_thresholds(self):
        """Test boundary values for similarity threshold."""
        # Test threshold = 0 (valid)
        query1 = Query(text="test", similarity_threshold=0.0)
        assert query1.similarity_threshold == 0.0
        
        # Test threshold = 1 (valid)
        query2 = Query(text="test", similarity_threshold=1.0)
        assert query2.similarity_threshold == 1.0


class TestEnhancedQuery:
    """Unit tests for EnhancedQuery entity."""
    
    def test_enhanced_query_creation(self):
        """Test creating an enhanced query."""
        original_query = Query(text="What is artificial intelligence?")
        enhanced = EnhancedQuery(original_query=original_query)
        
        assert enhanced.original_query == original_query
        assert enhanced.expanded_queries == []
        assert enhanced.rewritten_query is None
        assert enhanced.hypothetical_answer is None
        assert enhanced.entities == []
        assert enhanced.keywords == []
        assert isinstance(enhanced.created_at, datetime)
    
    def test_add_expansion(self):
        """Test adding query expansions."""
        original = Query(text="AI applications")
        enhanced = EnhancedQuery(original_query=original)
        
        enhanced.add_expansion("artificial intelligence applications")
        assert "artificial intelligence applications" in enhanced.expanded_queries
        assert len(enhanced.expanded_queries) == 1
        
        # Test duplicate expansion is not added
        enhanced.add_expansion("artificial intelligence applications")
        assert len(enhanced.expanded_queries) == 1
        
        # Test different expansion is added
        enhanced.add_expansion("machine learning applications")
        assert "machine learning applications" in enhanced.expanded_queries
        assert len(enhanced.expanded_queries) == 2
        
        # Test empty expansion is not added
        enhanced.add_expansion("")
        assert len(enhanced.expanded_queries) == 2
    
    def test_add_entity(self):
        """Test adding extracted entities."""
        original = Query(text="Tell me about OpenAI and GPT-4")
        enhanced = EnhancedQuery(original_query=original)
        
        enhanced.add_entity("OpenAI", "organization")
        enhanced.add_entity("GPT-4", "model")
        
        assert len(enhanced.entities) == 2
        assert enhanced.entities[0] == {"text": "OpenAI", "type": "organization"}
        assert enhanced.entities[1] == {"text": "GPT-4", "type": "model"}
    
    def test_get_all_query_variants(self):
        """Test getting all query variants."""
        original = Query(text="original query")
        enhanced = EnhancedQuery(original_query=original)
        
        # Initially only contains original
        variants = enhanced.get_all_query_variants()
        assert variants == ["original query"]
        
        # Add rewritten query
        enhanced.rewritten_query = "rewritten query"
        variants = enhanced.get_all_query_variants()
        assert "rewritten query" in variants
        assert len(variants) == 2
        
        # Add expansions
        enhanced.add_expansion("expanded query 1")
        enhanced.add_expansion("expanded query 2")
        variants = enhanced.get_all_query_variants()
        assert len(variants) == 4
        assert "expanded query 1" in variants
        assert "expanded query 2" in variants
        
        # Test deduplication
        enhanced.add_expansion("original query")  # Duplicate of original
        variants = enhanced.get_all_query_variants()
        assert len(variants) == 4  # Still 4, not 5
    
    def test_enhanced_query_with_all_features(self):
        """Test enhanced query with all features populated."""
        original = Query(text="What are the latest AI breakthroughs?")
        enhanced = EnhancedQuery(
            original_query=original,
            rewritten_query="Recent advances in artificial intelligence",
            hypothetical_answer="Recent AI breakthroughs include GPT-4, DALL-E 3...",
            keywords=["AI", "breakthroughs", "latest", "artificial intelligence"]
        )
        
        enhanced.add_expansion("cutting-edge AI developments")
        enhanced.add_expansion("recent machine learning innovations")
        enhanced.add_entity("GPT-4", "model")
        enhanced.add_entity("DALL-E 3", "model")
        
        assert enhanced.rewritten_query == "Recent advances in artificial intelligence"
        assert enhanced.hypothetical_answer.startswith("Recent AI breakthroughs")
        assert len(enhanced.keywords) == 4
        assert len(enhanced.expanded_queries) == 2
        assert len(enhanced.entities) == 2
        
        variants = enhanced.get_all_query_variants()
        assert len(variants) == 4  # original + rewritten + 2 expansions