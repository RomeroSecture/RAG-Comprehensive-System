import pytest
import math

from src.domain.value_objects.similarity_score import (
    SimilarityScore,
    SimilarityMetric,
    RerankingScore
)


class TestSimilarityMetric:
    """Test SimilarityMetric enum."""
    
    def test_similarity_metric_values(self):
        """Test that all similarity metrics have correct values."""
        assert SimilarityMetric.COSINE.value == "cosine"
        assert SimilarityMetric.EUCLIDEAN.value == "euclidean"
        assert SimilarityMetric.DOT_PRODUCT.value == "dot_product"
        assert SimilarityMetric.JACCARD.value == "jaccard"
        assert SimilarityMetric.BM25.value == "bm25"


class TestSimilarityScore:
    """Unit tests for SimilarityScore value object."""
    
    def test_similarity_score_creation(self):
        """Test creating similarity score with valid data."""
        score = SimilarityScore(
            value=0.85,
            metric=SimilarityMetric.COSINE,
            min_value=-1.0,
            max_value=1.0
        )
        
        assert score.value == 0.85
        assert score.metric == SimilarityMetric.COSINE
        assert score.min_value == -1.0
        assert score.max_value == 1.0
    
    def test_similarity_score_is_immutable(self):
        """Test that similarity score is immutable."""
        score = SimilarityScore.cosine(0.9)
        
        with pytest.raises(AttributeError):
            score.value = 0.8
        
        with pytest.raises(AttributeError):
            score.metric = SimilarityMetric.EUCLIDEAN
    
    def test_similarity_score_out_of_range_raises_error(self):
        """Test that values outside valid range raise error."""
        # Value too high
        with pytest.raises(ValueError, match="Score 1.5 is outside valid range"):
            SimilarityScore(value=1.5, metric=SimilarityMetric.COSINE)
        
        # Value too low
        with pytest.raises(ValueError, match="Score -0.5 is outside valid range"):
            SimilarityScore(value=-0.5, metric=SimilarityMetric.EUCLIDEAN)
    
    def test_cosine_factory_method(self):
        """Test cosine similarity score factory method."""
        # Valid cosine score
        score = SimilarityScore.cosine(0.75)
        assert score.value == 0.75
        assert score.metric == SimilarityMetric.COSINE
        assert score.min_value == -1.0
        assert score.max_value == 1.0
        
        # Negative cosine score
        score_neg = SimilarityScore.cosine(-0.5)
        assert score_neg.value == -0.5
        
        # Boundary values
        score_min = SimilarityScore.cosine(-1.0)
        assert score_min.value == -1.0
        
        score_max = SimilarityScore.cosine(1.0)
        assert score_max.value == 1.0
    
    def test_euclidean_factory_method(self):
        """Test Euclidean distance score factory method."""
        # Test conversion from distance to similarity
        score = SimilarityScore.euclidean(25.0, max_distance=100.0)
        assert score.value == 0.75  # 1 - (25/100)
        assert score.metric == SimilarityMetric.EUCLIDEAN
        
        # Zero distance = maximum similarity
        score_zero = SimilarityScore.euclidean(0.0, max_distance=100.0)
        assert score_zero.value == 1.0
        
        # Max distance = minimum similarity
        score_max = SimilarityScore.euclidean(100.0, max_distance=100.0)
        assert score_max.value == 0.0
    
    def test_dot_product_factory_method(self):
        """Test dot product score factory method."""
        score = SimilarityScore.dot_product(5.5)
        assert score.value == 5.5
        assert score.metric == SimilarityMetric.DOT_PRODUCT
        assert score.min_value == float('-inf')
        assert score.max_value == float('inf')
        
        # Negative dot product
        score_neg = SimilarityScore.dot_product(-3.0)
        assert score_neg.value == -3.0
    
    def test_bm25_factory_method(self):
        """Test BM25 score factory method."""
        score = SimilarityScore.bm25(12.5)
        assert score.value == 12.5
        assert score.metric == SimilarityMetric.BM25
        assert score.min_value == 0.0
        assert score.max_value == float('inf')
    
    def test_normalize_cosine(self):
        """Test normalization of cosine similarity scores."""
        # Test -1 -> 0
        score = SimilarityScore.cosine(-1.0)
        assert pytest.approx(score.normalize()) == 0.0
        
        # Test 0 -> 0.5
        score = SimilarityScore.cosine(0.0)
        assert pytest.approx(score.normalize()) == 0.5
        
        # Test 1 -> 1
        score = SimilarityScore.cosine(1.0)
        assert pytest.approx(score.normalize()) == 1.0
        
        # Test intermediate value
        score = SimilarityScore.cosine(0.5)
        assert pytest.approx(score.normalize()) == 0.75
    
    def test_normalize_dot_product(self):
        """Test normalization of dot product scores using sigmoid."""
        # Test 0 -> 0.5
        score = SimilarityScore.dot_product(0.0)
        assert pytest.approx(score.normalize()) == 0.5
        
        # Test positive value
        score = SimilarityScore.dot_product(2.0)
        expected = 1.0 / (1.0 + math.exp(-2.0))
        assert pytest.approx(score.normalize()) == expected
        
        # Test negative value
        score = SimilarityScore.dot_product(-2.0)
        expected = 1.0 / (1.0 + math.exp(2.0))
        assert pytest.approx(score.normalize()) == expected
    
    def test_normalize_bm25(self):
        """Test normalization of BM25 scores."""
        # For BM25 with max_value=inf, this should handle gracefully
        score = SimilarityScore.bm25(10.0)
        normalized = score.normalize()
        assert 0 <= normalized <= 1  # Should be in valid range
    
    def test_normalize_euclidean(self):
        """Test normalization of Euclidean scores."""
        # Euclidean scores are already in [0, 1]
        score = SimilarityScore.euclidean(30.0, max_distance=100.0)
        assert score.value == 0.7
        assert score.normalize() == 0.7
    
    def test_is_above_threshold(self):
        """Test threshold checking."""
        score = SimilarityScore.cosine(0.6)
        
        # Normalized value is (0.6 + 1) / 2 = 0.8
        assert score.is_above_threshold(0.7) is True
        assert score.is_above_threshold(0.8) is True
        assert score.is_above_threshold(0.9) is False
    
    def test_combine_with(self):
        """Test combining two similarity scores."""
        score1 = SimilarityScore.cosine(0.8)
        score2 = SimilarityScore.euclidean(20.0, max_distance=100.0)
        
        # Equal weight combination
        combined = score1.combine_with(score2, weight=0.5)
        # score1 normalized: (0.8 + 1) / 2 = 0.9
        # score2 normalized: 0.8
        # Combined: 0.5 * 0.9 + 0.5 * 0.8 = 0.85
        assert pytest.approx(combined.value) == 0.85
        assert combined.metric == SimilarityMetric.COSINE
        
        # Weighted combination
        combined_weighted = score1.combine_with(score2, weight=0.7)
        # 0.7 * 0.9 + 0.3 * 0.8 = 0.63 + 0.24 = 0.87
        assert pytest.approx(combined_weighted.value) == 0.87
    
    def test_combine_with_invalid_weight_raises_error(self):
        """Test that invalid weights raise error."""
        score1 = SimilarityScore.cosine(0.8)
        score2 = SimilarityScore.cosine(0.6)
        
        with pytest.raises(ValueError, match="Weight must be between 0 and 1"):
            score1.combine_with(score2, weight=1.5)
        
        with pytest.raises(ValueError, match="Weight must be between 0 and 1"):
            score1.combine_with(score2, weight=-0.1)


class TestRerankingScore:
    """Unit tests for RerankingScore value object."""
    
    def test_reranking_score_creation(self):
        """Test creating reranking score with valid data."""
        original = SimilarityScore.cosine(0.7)
        rerank = RerankingScore(
            original_score=original,
            rerank_score=0.85,
            reranker_model="cross-encoder-v1",
            confidence=0.95
        )
        
        assert rerank.original_score == original
        assert rerank.rerank_score == 0.85
        assert rerank.reranker_model == "cross-encoder-v1"
        assert rerank.confidence == 0.95
    
    def test_reranking_score_is_immutable(self):
        """Test that reranking score is immutable."""
        original = SimilarityScore.cosine(0.7)
        rerank = RerankingScore(
            original_score=original,
            rerank_score=0.85,
            reranker_model="model"
        )
        
        with pytest.raises(AttributeError):
            rerank.rerank_score = 0.9
    
    def test_reranking_score_invalid_rerank_score_raises_error(self):
        """Test that invalid rerank scores raise error."""
        original = SimilarityScore.cosine(0.7)
        
        # Score > 1
        with pytest.raises(ValueError, match="Rerank score must be between 0 and 1"):
            RerankingScore(
                original_score=original,
                rerank_score=1.5,
                reranker_model="model"
            )
        
        # Score < 0
        with pytest.raises(ValueError, match="Rerank score must be between 0 and 1"):
            RerankingScore(
                original_score=original,
                rerank_score=-0.1,
                reranker_model="model"
            )
    
    def test_reranking_score_invalid_confidence_raises_error(self):
        """Test that invalid confidence values raise error."""
        original = SimilarityScore.cosine(0.7)
        
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            RerankingScore(
                original_score=original,
                rerank_score=0.85,
                reranker_model="model",
                confidence=1.5
            )
    
    def test_reranking_score_empty_model_raises_error(self):
        """Test that empty model name raises error."""
        original = SimilarityScore.cosine(0.7)
        
        with pytest.raises(ValueError, match="Reranker model must be specified"):
            RerankingScore(
                original_score=original,
                rerank_score=0.85,
                reranker_model=""
            )
    
    def test_score_delta(self):
        """Test score delta calculation."""
        original = SimilarityScore.cosine(0.6)  # Normalized: 0.8
        rerank = RerankingScore(
            original_score=original,
            rerank_score=0.9,
            reranker_model="model"
        )
        
        # Delta = 0.9 - 0.8 = 0.1
        assert pytest.approx(rerank.score_delta) == 0.1
        
        # Test negative delta
        rerank_lower = RerankingScore(
            original_score=original,
            rerank_score=0.7,
            reranker_model="model"
        )
        assert pytest.approx(rerank_lower.score_delta) == -0.1
    
    def test_final_score(self):
        """Test final score property."""
        original = SimilarityScore.cosine(0.6)
        rerank = RerankingScore(
            original_score=original,
            rerank_score=0.85,
            reranker_model="model"
        )
        
        assert rerank.final_score == 0.85
    
    def test_weighted_combination(self):
        """Test weighted combination of original and rerank scores."""
        original = SimilarityScore.cosine(0.6)  # Normalized: 0.8
        rerank = RerankingScore(
            original_score=original,
            rerank_score=0.9,
            reranker_model="model"
        )
        
        # Default weight (0.3 original, 0.7 rerank)
        combined = rerank.weighted_combination()
        # 0.3 * 0.8 + 0.7 * 0.9 = 0.24 + 0.63 = 0.87
        assert pytest.approx(combined) == 0.87
        
        # Custom weight
        combined_custom = rerank.weighted_combination(original_weight=0.5)
        # 0.5 * 0.8 + 0.5 * 0.9 = 0.4 + 0.45 = 0.85
        assert pytest.approx(combined_custom) == 0.85
    
    def test_weighted_combination_invalid_weight_raises_error(self):
        """Test that invalid weight raises error."""
        original = SimilarityScore.cosine(0.6)
        rerank = RerankingScore(
            original_score=original,
            rerank_score=0.9,
            reranker_model="model"
        )
        
        with pytest.raises(ValueError, match="Weight must be between 0 and 1"):
            rerank.weighted_combination(original_weight=1.5)
    
    def test_reranking_without_confidence(self):
        """Test reranking score without confidence value."""
        original = SimilarityScore.cosine(0.7)
        rerank = RerankingScore(
            original_score=original,
            rerank_score=0.85,
            reranker_model="model"
            # No confidence specified
        )
        
        assert rerank.confidence is None