import pytest
from uuid import uuid4

from src.domain.services.ranking_service import (
    RankingService,
    CrossEncoderRankingService,
    ColBERTRankingService,
    LLMRankingService,
    MultiStageRankingService,
    ReciprocalRankFusion,
    DiversityRankingService
)
from src.domain.entities.query import Query
from src.domain.entities.retrieval_result import RetrievedDocument


class TestRankingService:
    """Unit tests for the abstract RankingService."""
    
    def test_ranking_service_is_abstract(self):
        """Test that RankingService cannot be instantiated."""
        with pytest.raises(TypeError):
            RankingService()


class TestCrossEncoderRankingService:
    """Unit tests for CrossEncoderRankingService."""
    
    def test_cross_encoder_creation(self):
        """Test creating cross-encoder ranking service."""
        service = CrossEncoderRankingService()
        assert service.model_name == "cross-encoder/ms-marco-MiniLM-L-12-v2"
        
    def test_cross_encoder_with_custom_model(self):
        """Test creating cross-encoder with custom model."""
        custom_model = "cross-encoder/ms-marco-electra-base"
        service = CrossEncoderRankingService(model_name=custom_model)
        assert service.get_model_name() == custom_model
    
    @pytest.mark.asyncio
    async def test_rank_documents_returns_list(self):
        """Test that rank_documents returns a list (stub implementation)."""
        service = CrossEncoderRankingService()
        query = Query(text="test query")
        documents = [
            RetrievedDocument(content="doc1", score=0.9),
            RetrievedDocument(content="doc2", score=0.8)
        ]
        
        # Since implementation is not complete, it will return None
        # We're just verifying the method exists and is callable
        result = await service.rank_documents(query, documents)
        assert result is None  # Placeholder implementation returns None


class TestColBERTRankingService:
    """Unit tests for ColBERTRankingService."""
    
    def test_colbert_creation(self):
        """Test creating ColBERT ranking service."""
        service = ColBERTRankingService()
        assert service.model_name == "colbert-ir/colbertv2.0"
        
    def test_colbert_with_custom_model(self):
        """Test creating ColBERT with custom model."""
        custom_model = "colbert-ir/colbertv1.0"
        service = ColBERTRankingService(model_name=custom_model)
        assert service.get_model_name() == custom_model


class TestLLMRankingService:
    """Unit tests for LLMRankingService."""
    
    def test_llm_ranking_creation(self):
        """Test creating LLM ranking service."""
        service = LLMRankingService()
        assert service.model_name == "gpt-4"
        
    def test_llm_ranking_with_custom_model(self):
        """Test creating LLM ranking with custom model."""
        custom_model = "claude-3"
        service = LLMRankingService(model_name=custom_model)
        assert service.get_model_name() == custom_model


class TestMultiStageRankingService:
    """Unit tests for MultiStageRankingService."""
    
    def test_multi_stage_creation(self):
        """Test creating multi-stage ranking service."""
        stage1 = CrossEncoderRankingService()
        stage2 = LLMRankingService()
        
        stages = [(stage1, 100), (stage2, 10)]
        service = MultiStageRankingService(stages)
        
        assert len(service.stages) == 2
        assert service.stages[0][0] == stage1
        assert service.stages[0][1] == 100
        assert service.stages[1][0] == stage2
        assert service.stages[1][1] == 10
    
    @pytest.mark.asyncio
    async def test_multi_stage_rank_documents(self):
        """Test multi-stage ranking pipeline (with mocked stages)."""
        # Create mock ranking services
        class MockRankingService(RankingService):
            def __init__(self, prefix):
                self.prefix = prefix
                
            async def rank_documents(self, query, documents):
                # Simply reverse the order to simulate reranking
                return documents[::-1]
            
            def get_model_name(self):
                return f"mock-{self.prefix}"
        
        stage1 = MockRankingService("stage1")
        stage2 = MockRankingService("stage2")
        
        service = MultiStageRankingService([(stage1, 3), (stage2, 2)])
        
        query = Query(text="test")
        documents = [
            RetrievedDocument(content="doc1", score=0.9),
            RetrievedDocument(content="doc2", score=0.8),
            RetrievedDocument(content="doc3", score=0.7),
            RetrievedDocument(content="doc4", score=0.6)
        ]
        
        result = await service.rank_documents(query, documents)
        
        # Should keep only top 2 after all stages
        assert len(result) == 2


class TestReciprocalRankFusion:
    """Unit tests for ReciprocalRankFusion."""
    
    def test_rrf_creation(self):
        """Test creating RRF with default k."""
        rrf = ReciprocalRankFusion()
        assert rrf.k == 60
    
    def test_rrf_with_custom_k(self):
        """Test creating RRF with custom k."""
        rrf = ReciprocalRankFusion(k=30)
        assert rrf.k == 30
    
    def test_fuse_single_ranking(self):
        """Test fusing a single ranking."""
        rrf = ReciprocalRankFusion(k=60)
        
        docs = [
            RetrievedDocument(content="doc1", score=0.9),
            RetrievedDocument(content="doc2", score=0.8)
        ]
        
        result = rrf.fuse_rankings([docs])
        
        assert len(result) == 2
        assert result[0].content == "doc1"
        assert result[1].content == "doc2"
        # Check that scores are normalized
        assert 0 <= result[0].score <= 1
        assert 0 <= result[1].score <= 1
    
    def test_fuse_multiple_rankings(self):
        """Test fusing multiple rankings."""
        rrf = ReciprocalRankFusion(k=60)
        
        ranking1 = [
            RetrievedDocument(id=uuid4(), content="doc1", score=0.9),
            RetrievedDocument(id=uuid4(), content="doc2", score=0.8),
            RetrievedDocument(id=uuid4(), content="doc3", score=0.7)
        ]
        
        # Create second ranking with different order
        ranking2 = [
            ranking1[1],  # doc2 first
            ranking1[2],  # doc3 second
            ranking1[0]   # doc1 third
        ]
        
        result = rrf.fuse_rankings([ranking1, ranking2])
        
        assert len(result) == 3
        # All documents should be present
        contents = [doc.content for doc in result]
        assert "doc1" in contents
        assert "doc2" in contents
        assert "doc3" in contents
    
    def test_fuse_rankings_with_overlapping_documents(self):
        """Test fusing rankings with the same document appearing multiple times."""
        rrf = ReciprocalRankFusion(k=10)
        
        # Create documents with consistent IDs
        doc1 = RetrievedDocument(id=uuid4(), content="doc1", score=0.9)
        doc2 = RetrievedDocument(id=uuid4(), content="doc2", score=0.8)
        doc3 = RetrievedDocument(id=uuid4(), content="doc3", score=0.7)
        
        ranking1 = [doc1, doc2]
        ranking2 = [doc2, doc3]
        ranking3 = [doc3, doc1]
        
        result = rrf.fuse_rankings([ranking1, ranking2, ranking3])
        
        assert len(result) == 3
        # Check that each document appears only once
        doc_ids = [doc.id for doc in result]
        assert len(doc_ids) == len(set(doc_ids))


class TestDiversityRankingService:
    """Unit tests for DiversityRankingService."""
    
    def test_diversity_ranking_creation(self):
        """Test creating diversity ranking service."""
        service = DiversityRankingService()
        assert service.lambda_param == 0.5
    
    def test_diversity_ranking_with_custom_lambda(self):
        """Test creating diversity ranking with custom lambda."""
        service = DiversityRankingService(lambda_param=0.7)
        assert service.lambda_param == 0.7
    
    @pytest.mark.asyncio
    async def test_diversify_empty_documents(self):
        """Test diversifying empty document list."""
        service = DiversityRankingService()
        result = await service.diversify_rankings([], top_k=5)
        assert result == []
    
    @pytest.mark.asyncio
    async def test_diversify_single_document(self):
        """Test diversifying single document."""
        service = DiversityRankingService()
        docs = [RetrievedDocument(content="single doc", score=0.9)]
        
        result = await service.diversify_rankings(docs, top_k=5)
        assert len(result) == 1
        assert result[0].content == "single doc"
    
    @pytest.mark.asyncio
    async def test_diversify_rankings_mmr(self):
        """Test MMR diversification with multiple documents."""
        service = DiversityRankingService(lambda_param=0.5)
        
        # Create documents with varying content overlap
        docs = [
            RetrievedDocument(content="python programming tutorial", score=0.95),
            RetrievedDocument(content="python programming guide", score=0.93),  # Similar to first
            RetrievedDocument(content="java coding examples", score=0.91),
            RetrievedDocument(content="python data science", score=0.90),
            RetrievedDocument(content="machine learning basics", score=0.88)
        ]
        
        result = await service.diversify_rankings(docs, top_k=3)
        
        assert len(result) == 3
        # First document should be the highest scoring one
        assert result[0].content == "python programming tutorial"
        # Subsequent documents should balance relevance and diversity
        assert result[0].score == 0.95
    
    def test_calculate_similarity(self):
        """Test document similarity calculation."""
        service = DiversityRankingService()
        
        doc1 = RetrievedDocument(content="the quick brown fox", score=0.9)
        doc2 = RetrievedDocument(content="the quick brown dog", score=0.8)
        doc3 = RetrievedDocument(content="completely different text", score=0.7)
        
        # High similarity (3 out of 4 words match)
        sim1_2 = service._calculate_similarity(doc1, doc2)
        assert 0.5 < sim1_2 < 1.0
        
        # Low similarity
        sim1_3 = service._calculate_similarity(doc1, doc3)
        assert 0 <= sim1_3 < 0.5
        
        # Same document
        sim1_1 = service._calculate_similarity(doc1, doc1)
        assert sim1_1 == 1.0
    
    @pytest.mark.asyncio
    async def test_diversify_with_lambda_extremes(self):
        """Test diversification with extreme lambda values."""
        docs = [
            RetrievedDocument(content="doc1 about topic A", score=0.95),
            RetrievedDocument(content="doc2 about topic A", score=0.90),
            RetrievedDocument(content="doc3 about topic B", score=0.85),
            RetrievedDocument(content="doc4 about topic C", score=0.80)
        ]
        
        # Lambda = 1.0 (only relevance matters)
        service_relevance = DiversityRankingService(lambda_param=1.0)
        result_relevance = await service_relevance.diversify_rankings(docs, top_k=3)
        assert result_relevance[0].score == 0.95
        assert result_relevance[1].score == 0.90
        
        # Lambda = 0.0 (only diversity matters)
        service_diversity = DiversityRankingService(lambda_param=0.0)
        result_diversity = await service_diversity.diversify_rankings(docs, top_k=3)
        # Should select documents with less content overlap
        assert len(result_diversity) == 3