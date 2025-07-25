import pytest
from uuid import uuid4

from src.domain.services.evaluation_service import (
    RetrievalMetrics,
    GenerationMetrics,
    EvaluationService,
    StandardRetrievalEvaluator,
    RAGASEvaluator,
    ComparativeEvaluator
)
from src.domain.entities.query import Query
from src.domain.entities.retrieval_result import RetrievalResult, RetrievedDocument


class TestRetrievalMetrics:
    """Unit tests for RetrievalMetrics dataclass."""
    
    def test_retrieval_metrics_creation(self):
        """Test creating retrieval metrics."""
        metrics = RetrievalMetrics()
        
        assert metrics.precision_at_k == {}
        assert metrics.recall_at_k == {}
        assert metrics.f1_at_k == {}
        assert metrics.ndcg_at_k == {}
        assert metrics.mrr == 0.0
        assert metrics.average_precision == 0.0
    
    def test_retrieval_metrics_with_values(self):
        """Test creating retrieval metrics with values."""
        metrics = RetrievalMetrics(
            precision_at_k={1: 1.0, 5: 0.8, 10: 0.7},
            recall_at_k={1: 0.2, 5: 0.6, 10: 0.9},
            f1_at_k={1: 0.33, 5: 0.69, 10: 0.79},
            ndcg_at_k={1: 1.0, 5: 0.85, 10: 0.78},
            mrr=0.75,
            average_precision=0.82
        )
        
        assert metrics.get_precision(5) == 0.8
        assert metrics.get_recall(10) == 0.9
        assert metrics.get_f1(1) == 0.33
        assert metrics.get_ndcg(5) == 0.85
        assert metrics.mrr == 0.75
        assert metrics.average_precision == 0.82
    
    def test_get_metrics_missing_k(self):
        """Test getting metrics for missing k value."""
        metrics = RetrievalMetrics(
            precision_at_k={5: 0.8},
            recall_at_k={5: 0.6}
        )
        
        assert metrics.get_precision(10) == 0.0
        assert metrics.get_recall(10) == 0.0
        assert metrics.get_f1(10) == 0.0
        assert metrics.get_ndcg(10) == 0.0


class TestGenerationMetrics:
    """Unit tests for GenerationMetrics dataclass."""
    
    def test_generation_metrics_creation(self):
        """Test creating generation metrics."""
        metrics = GenerationMetrics()
        
        assert metrics.faithfulness == 0.0
        assert metrics.answer_relevancy == 0.0
        assert metrics.context_precision == 0.0
        assert metrics.context_recall == 0.0
        assert metrics.hallucination_score == 0.0
        assert metrics.citation_accuracy == 0.0
        assert metrics.coherence_score == 0.0
    
    def test_generation_metrics_with_values(self):
        """Test creating generation metrics with values."""
        metrics = GenerationMetrics(
            faithfulness=0.9,
            answer_relevancy=0.85,
            context_precision=0.8,
            context_recall=0.75,
            hallucination_score=0.1,
            citation_accuracy=0.95,
            coherence_score=0.88
        )
        
        assert metrics.faithfulness == 0.9
        assert metrics.answer_relevancy == 0.85
        assert metrics.context_precision == 0.8
        assert metrics.context_recall == 0.75
        assert metrics.hallucination_score == 0.1
        assert metrics.citation_accuracy == 0.95
        assert metrics.coherence_score == 0.88
    
    def test_overall_quality_calculation(self):
        """Test overall quality score calculation."""
        metrics = GenerationMetrics(
            faithfulness=0.9,
            answer_relevancy=0.85,
            context_precision=0.8,
            context_recall=0.75,
            hallucination_score=0.1,
            citation_accuracy=0.95
        )
        
        # Calculate expected score
        expected = (
            0.3 * 0.9 +      # faithfulness
            0.3 * 0.85 +     # answer_relevancy
            0.15 * 0.8 +     # context_precision
            0.15 * 0.75 +    # context_recall
            0.1 * 0.95       # citation_accuracy
        ) * (1 - 0.1)        # penalty for hallucination
        
        assert pytest.approx(metrics.overall_quality, rel=1e-6) == expected
    
    def test_overall_quality_with_no_hallucination(self):
        """Test overall quality with no hallucination."""
        metrics = GenerationMetrics(
            faithfulness=1.0,
            answer_relevancy=1.0,
            context_precision=1.0,
            context_recall=1.0,
            hallucination_score=0.0,
            citation_accuracy=1.0
        )
        
        assert metrics.overall_quality == 1.0
    
    def test_overall_quality_with_high_hallucination(self):
        """Test overall quality with high hallucination."""
        metrics = GenerationMetrics(
            faithfulness=0.9,
            answer_relevancy=0.9,
            context_precision=0.9,
            context_recall=0.9,
            hallucination_score=0.5,  # 50% hallucination
            citation_accuracy=0.9
        )
        
        # Should be significantly penalized
        assert metrics.overall_quality < 0.5


class TestEvaluationService:
    """Unit tests for the abstract EvaluationService."""
    
    def test_evaluation_service_is_abstract(self):
        """Test that EvaluationService cannot be instantiated."""
        with pytest.raises(TypeError):
            EvaluationService()


class TestStandardRetrievalEvaluator:
    """Unit tests for StandardRetrievalEvaluator."""
    
    @pytest.mark.asyncio
    async def test_evaluate_retrieval_empty_results(self):
        """Test evaluating retrieval with no results."""
        evaluator = StandardRetrievalEvaluator()
        
        query = Query(text="test query")
        retrieved_documents = []
        relevant_documents = ["doc1", "doc2", "doc3"]
        
        metrics = await evaluator.evaluate_retrieval(
            query, retrieved_documents, relevant_documents
        )
        
        # All metrics should be 0
        assert metrics.mrr == 0.0
        assert metrics.average_precision == 0.0
        for k in [1, 3, 5, 10, 20]:
            assert metrics.get_precision(k) == 0.0
            assert metrics.get_recall(k) == 0.0
    
    @pytest.mark.asyncio
    async def test_evaluate_retrieval_perfect_results(self):
        """Test evaluating retrieval with perfect results."""
        evaluator = StandardRetrievalEvaluator()
        
        query = Query(text="test query")
        
        # Create documents matching relevant IDs
        doc_ids = [uuid4() for _ in range(3)]
        retrieved_documents = [
            RetrievedDocument(document_id=doc_ids[0], content="doc1", score=0.9),
            RetrievedDocument(document_id=doc_ids[1], content="doc2", score=0.8),
            RetrievedDocument(document_id=doc_ids[2], content="doc3", score=0.7)
        ]
        relevant_documents = [str(doc_id) for doc_id in doc_ids]
        
        metrics = await evaluator.evaluate_retrieval(
            query, retrieved_documents, relevant_documents
        )
        
        # Perfect scores
        assert metrics.mrr == 1.0
        assert metrics.average_precision == 1.0
        assert metrics.get_precision(1) == 1.0
        assert metrics.get_precision(3) == 1.0
        assert metrics.get_recall(3) == 1.0
        assert metrics.get_f1(3) == 1.0
    
    @pytest.mark.asyncio
    async def test_evaluate_retrieval_partial_results(self):
        """Test evaluating retrieval with partial matches."""
        evaluator = StandardRetrievalEvaluator()
        
        query = Query(text="test query")
        
        # Create mix of relevant and irrelevant documents
        relevant_ids = [uuid4() for _ in range(3)]
        irrelevant_ids = [uuid4() for _ in range(2)]
        
        retrieved_documents = [
            RetrievedDocument(document_id=relevant_ids[0], content="relevant1", score=0.9),
            RetrievedDocument(document_id=irrelevant_ids[0], content="irrelevant1", score=0.8),
            RetrievedDocument(document_id=relevant_ids[1], content="relevant2", score=0.7),
            RetrievedDocument(document_id=irrelevant_ids[1], content="irrelevant2", score=0.6),
            RetrievedDocument(document_id=relevant_ids[2], content="relevant3", score=0.5)
        ]
        
        relevant_documents = [str(doc_id) for doc_id in relevant_ids]
        
        metrics = await evaluator.evaluate_retrieval(
            query, retrieved_documents, relevant_documents
        )
        
        # Check specific metrics
        assert metrics.get_precision(1) == 1.0  # First doc is relevant
        assert metrics.get_precision(5) == 0.6  # 3 out of 5 are relevant
        assert metrics.get_recall(5) == 1.0     # All 3 relevant docs retrieved
        assert metrics.mrr == 1.0               # First result is relevant
    
    @pytest.mark.asyncio
    async def test_evaluate_retrieval_no_relevant_documents(self):
        """Test evaluating when there are no relevant documents."""
        evaluator = StandardRetrievalEvaluator()
        
        query = Query(text="test query")
        retrieved_documents = [
            RetrievedDocument(content="doc1", score=0.9),
            RetrievedDocument(content="doc2", score=0.8)
        ]
        relevant_documents = []
        
        metrics = await evaluator.evaluate_retrieval(
            query, retrieved_documents, relevant_documents
        )
        
        # Metrics should handle empty relevant set gracefully
        assert metrics.mrr == 0.0
        assert metrics.average_precision == 0.0
    
    def test_calculate_ndcg(self):
        """Test NDCG calculation."""
        evaluator = StandardRetrievalEvaluator()
        
        # Perfect ranking
        retrieved_ids = ["rel1", "rel2", "rel3", "irrel1", "irrel2"]
        relevant_set = {"rel1", "rel2", "rel3"}
        
        ndcg = evaluator._calculate_ndcg(retrieved_ids, relevant_set, k=5)
        assert 0 < ndcg <= 1.0
        
        # Worst ranking (all irrelevant)
        retrieved_ids_bad = ["irrel1", "irrel2", "irrel3", "irrel4", "irrel5"]
        ndcg_bad = evaluator._calculate_ndcg(retrieved_ids_bad, relevant_set, k=5)
        assert ndcg_bad == 0.0
    
    def test_calculate_mrr(self):
        """Test MRR calculation."""
        evaluator = StandardRetrievalEvaluator()
        
        # First result is relevant
        docs = [
            RetrievedDocument(document_id=uuid4(), content="relevant", score=0.9),
            RetrievedDocument(document_id=uuid4(), content="irrelevant", score=0.8)
        ]
        relevant_set = {str(docs[0].document_id)}
        
        mrr = evaluator._calculate_mrr(docs, relevant_set)
        assert mrr == 1.0
        
        # Second result is relevant
        docs[0], docs[1] = docs[1], docs[0]
        relevant_set = {str(docs[1].document_id)}
        
        mrr = evaluator._calculate_mrr(docs, relevant_set)
        assert mrr == 0.5
        
        # No relevant results
        mrr_none = evaluator._calculate_mrr(docs, set())
        assert mrr_none == 0.0
    
    def test_calculate_average_precision(self):
        """Test Average Precision calculation."""
        evaluator = StandardRetrievalEvaluator()
        
        # Create documents
        relevant_ids = [uuid4() for _ in range(3)]
        docs = [
            RetrievedDocument(document_id=relevant_ids[0], content="rel1", score=0.9),
            RetrievedDocument(document_id=uuid4(), content="irrel1", score=0.8),
            RetrievedDocument(document_id=relevant_ids[1], content="rel2", score=0.7),
            RetrievedDocument(document_id=relevant_ids[2], content="rel3", score=0.6),
            RetrievedDocument(document_id=uuid4(), content="irrel2", score=0.5)
        ]
        
        relevant_set = {str(doc_id) for doc_id in relevant_ids}
        
        ap = evaluator._calculate_average_precision(docs, relevant_set)
        
        # Calculate expected AP
        # Precision at position 1: 1/1 = 1.0
        # Precision at position 3: 2/3 = 0.667
        # Precision at position 4: 3/4 = 0.75
        # AP = (1.0 + 0.667 + 0.75) / 3 = 0.806
        assert 0.8 < ap < 0.82
    
    @pytest.mark.asyncio
    async def test_evaluate_generation_placeholder(self):
        """Test generation evaluation (placeholder implementation)."""
        evaluator = StandardRetrievalEvaluator()
        
        query = Query(text="test query")
        generated_answer = "This is a generated answer."
        context_documents = [
            RetrievedDocument(content="context1", score=0.9),
            RetrievedDocument(content="context2", score=0.8)
        ]
        
        metrics = await evaluator.evaluate_generation(
            query, generated_answer, context_documents
        )
        
        # Check that placeholder returns reasonable values
        assert 0 <= metrics.faithfulness <= 1
        assert 0 <= metrics.answer_relevancy <= 1
        assert 0 <= metrics.context_precision <= 1
        assert 0 <= metrics.context_recall <= 1
        assert 0 <= metrics.hallucination_score <= 1
        assert 0 <= metrics.citation_accuracy <= 1
        assert 0 <= metrics.coherence_score <= 1
        assert 0 <= metrics.overall_quality <= 1


class TestRAGASEvaluator:
    """Unit tests for RAGASEvaluator."""
    
    def test_ragas_evaluator_creation(self):
        """Test creating RAGAS evaluator."""
        evaluator = RAGASEvaluator()
        assert evaluator.llm_model == "gpt-4"
    
    def test_ragas_evaluator_with_custom_model(self):
        """Test creating RAGAS evaluator with custom model."""
        evaluator = RAGASEvaluator(llm_model="claude-3")
        assert evaluator.llm_model == "claude-3"


class TestComparativeEvaluator:
    """Unit tests for ComparativeEvaluator."""
    
    def test_comparative_evaluator_creation(self):
        """Test creating comparative evaluator."""
        base_evaluator = StandardRetrievalEvaluator()
        comp_evaluator = ComparativeEvaluator(base_evaluator)
        
        assert comp_evaluator.evaluator == base_evaluator
    
    @pytest.mark.asyncio
    async def test_compare_strategies(self):
        """Test comparing multiple strategies."""
        base_evaluator = StandardRetrievalEvaluator()
        comp_evaluator = ComparativeEvaluator(base_evaluator)
        
        query = Query(text="test query")
        
        # Create mock results for different strategies
        relevant_id = uuid4()
        
        semantic_result = RetrievalResult(
            documents=[
                RetrievedDocument(document_id=relevant_id, content="relevant", score=0.9),
                RetrievedDocument(document_id=uuid4(), content="irrelevant", score=0.8)
            ]
        )
        
        hybrid_result = RetrievalResult(
            documents=[
                RetrievedDocument(document_id=uuid4(), content="irrelevant1", score=0.95),
                RetrievedDocument(document_id=relevant_id, content="relevant", score=0.85),
                RetrievedDocument(document_id=uuid4(), content="irrelevant2", score=0.75)
            ]
        )
        
        strategy_results = {
            "semantic": (semantic_result, "Semantic answer"),
            "hybrid": (hybrid_result, "Hybrid answer")
        }
        
        ground_truth = [str(relevant_id)]
        
        comparison = await comp_evaluator.compare_strategies(
            query, strategy_results, ground_truth
        )
        
        assert "semantic" in comparison
        assert "hybrid" in comparison
        
        # Each strategy should have both retrieval and generation metrics
        for strategy_name, (retrieval_metrics, generation_metrics) in comparison.items():
            assert isinstance(retrieval_metrics, RetrievalMetrics)
            assert isinstance(generation_metrics, GenerationMetrics)
        
        # Semantic should have better MRR (relevant doc is first)
        semantic_retrieval, _ = comparison["semantic"]
        hybrid_retrieval, _ = comparison["hybrid"]
        
        assert semantic_retrieval.mrr > hybrid_retrieval.mrr
    
    @pytest.mark.asyncio
    async def test_compare_strategies_empty(self):
        """Test comparing with no strategies."""
        base_evaluator = StandardRetrievalEvaluator()
        comp_evaluator = ComparativeEvaluator(base_evaluator)
        
        query = Query(text="test query")
        strategy_results = {}
        ground_truth = ["doc1"]
        
        comparison = await comp_evaluator.compare_strategies(
            query, strategy_results, ground_truth
        )
        
        assert comparison == {}