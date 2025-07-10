from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..entities.query import Query
from ..entities.retrieval_result import RetrievalResult, RetrievedDocument


@dataclass
class RetrievalMetrics:
    """Metrics for evaluating retrieval quality."""
    
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    f1_at_k: Dict[int, float] = field(default_factory=dict)
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    average_precision: float = 0.0
    
    def get_precision(self, k: int) -> float:
        """Get precision at k."""
        return self.precision_at_k.get(k, 0.0)
    
    def get_recall(self, k: int) -> float:
        """Get recall at k."""
        return self.recall_at_k.get(k, 0.0)
    
    def get_f1(self, k: int) -> float:
        """Get F1 score at k."""
        return self.f1_at_k.get(k, 0.0)
    
    def get_ndcg(self, k: int) -> float:
        """Get NDCG at k."""
        return self.ndcg_at_k.get(k, 0.0)


@dataclass
class GenerationMetrics:
    """Metrics for evaluating generation quality."""
    
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    hallucination_score: float = 0.0
    citation_accuracy: float = 0.0
    coherence_score: float = 0.0
    
    @property
    def overall_quality(self) -> float:
        """Calculate overall generation quality score."""
        weights = {
            "faithfulness": 0.3,
            "answer_relevancy": 0.3,
            "context_precision": 0.15,
            "context_recall": 0.15,
            "citation_accuracy": 0.1
        }
        
        score = (
            weights["faithfulness"] * self.faithfulness +
            weights["answer_relevancy"] * self.answer_relevancy +
            weights["context_precision"] * self.context_precision +
            weights["context_recall"] * self.context_recall +
            weights["citation_accuracy"] * self.citation_accuracy
        )
        
        # Penalize for hallucination
        score *= (1 - self.hallucination_score)
        
        return score


class EvaluationService(ABC):
    """Abstract service for RAG evaluation."""
    
    @abstractmethod
    async def evaluate_retrieval(
        self,
        query: Query,
        retrieved_documents: List[RetrievedDocument],
        relevant_documents: List[str]
    ) -> RetrievalMetrics:
        """Evaluate retrieval quality against ground truth."""
        pass
    
    @abstractmethod
    async def evaluate_generation(
        self,
        query: Query,
        generated_answer: str,
        context_documents: List[RetrievedDocument],
        reference_answer: Optional[str] = None
    ) -> GenerationMetrics:
        """Evaluate generation quality."""
        pass


class StandardRetrievalEvaluator(EvaluationService):
    """Standard retrieval evaluation metrics implementation."""
    
    async def evaluate_retrieval(
        self,
        query: Query,
        retrieved_documents: List[RetrievedDocument],
        relevant_documents: List[str]
    ) -> RetrievalMetrics:
        """Calculate standard retrieval metrics."""
        metrics = RetrievalMetrics()
        
        # Convert relevant documents to set for efficiency
        relevant_set = set(relevant_documents)
        
        # Calculate metrics at different k values
        k_values = [1, 3, 5, 10, 20]
        
        for k in k_values:
            if k > len(retrieved_documents):
                continue
            
            # Get top k retrieved documents
            top_k_docs = retrieved_documents[:k]
            retrieved_ids = [str(doc.document_id) for doc in top_k_docs]
            
            # Calculate precision at k
            relevant_in_top_k = sum(1 for doc_id in retrieved_ids if doc_id in relevant_set)
            precision = relevant_in_top_k / k if k > 0 else 0.0
            metrics.precision_at_k[k] = precision
            
            # Calculate recall at k
            recall = relevant_in_top_k / len(relevant_set) if relevant_set else 0.0
            metrics.recall_at_k[k] = recall
            
            # Calculate F1 at k
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
            metrics.f1_at_k[k] = f1
            
            # Calculate NDCG at k
            ndcg = self._calculate_ndcg(retrieved_ids, relevant_set, k)
            metrics.ndcg_at_k[k] = ndcg
        
        # Calculate MRR (Mean Reciprocal Rank)
        metrics.mrr = self._calculate_mrr(retrieved_documents, relevant_set)
        
        # Calculate Average Precision
        metrics.average_precision = self._calculate_average_precision(
            retrieved_documents, relevant_set
        )
        
        return metrics
    
    def _calculate_ndcg(
        self,
        retrieved_ids: List[str],
        relevant_set: set,
        k: int
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        # Calculate DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            if doc_id in relevant_set:
                # Using binary relevance (1 if relevant, 0 otherwise)
                dcg += 1.0 / (i + 2)  # log2(i+2) approximation
        
        # Calculate IDCG (ideal DCG)
        idcg = sum(1.0 / (i + 2) for i in range(min(len(relevant_set), k)))
        
        # Calculate NDCG
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_mrr(
        self,
        retrieved_documents: List[RetrievedDocument],
        relevant_set: set
    ) -> float:
        """Calculate Mean Reciprocal Rank."""
        for i, doc in enumerate(retrieved_documents):
            if str(doc.document_id) in relevant_set:
                return 1.0 / (i + 1)
        return 0.0
    
    def _calculate_average_precision(
        self,
        retrieved_documents: List[RetrievedDocument],
        relevant_set: set
    ) -> float:
        """Calculate Average Precision."""
        if not relevant_set:
            return 0.0
        
        num_relevant = 0
        sum_precision = 0.0
        
        for i, doc in enumerate(retrieved_documents):
            if str(doc.document_id) in relevant_set:
                num_relevant += 1
                precision_at_i = num_relevant / (i + 1)
                sum_precision += precision_at_i
        
        return sum_precision / len(relevant_set) if relevant_set else 0.0
    
    async def evaluate_generation(
        self,
        query: Query,
        generated_answer: str,
        context_documents: List[RetrievedDocument],
        reference_answer: Optional[str] = None
    ) -> GenerationMetrics:
        """Evaluate generation quality (placeholder implementation)."""
        # This would integrate with LLM-based evaluation or other metrics
        return GenerationMetrics(
            faithfulness=0.85,
            answer_relevancy=0.90,
            context_precision=0.80,
            context_recall=0.75,
            hallucination_score=0.05,
            citation_accuracy=0.95,
            coherence_score=0.88
        )


class RAGASEvaluator(EvaluationService):
    """RAGAS (RAG Assessment) framework implementation."""
    
    def __init__(self, llm_model: str = "gpt-4"):
        self.llm_model = llm_model
    
    async def evaluate_retrieval(
        self,
        query: Query,
        retrieved_documents: List[RetrievedDocument],
        relevant_documents: List[str]
    ) -> RetrievalMetrics:
        """Evaluate using RAGAS retrieval metrics."""
        # Would implement RAGAS-specific retrieval evaluation
        pass
    
    async def evaluate_generation(
        self,
        query: Query,
        generated_answer: str,
        context_documents: List[RetrievedDocument],
        reference_answer: Optional[str] = None
    ) -> GenerationMetrics:
        """Evaluate using RAGAS generation metrics."""
        # Would implement RAGAS-specific generation evaluation
        pass


class ComparativeEvaluator:
    """Service for comparing different RAG strategies."""
    
    def __init__(self, evaluator: EvaluationService):
        self.evaluator = evaluator
    
    async def compare_strategies(
        self,
        query: Query,
        strategy_results: Dict[str, tuple[RetrievalResult, str]],
        ground_truth: List[str],
        reference_answer: Optional[str] = None
    ) -> Dict[str, tuple[RetrievalMetrics, GenerationMetrics]]:
        """
        Compare multiple RAG strategies.
        strategy_results: Dict mapping strategy name to (retrieval_result, generated_answer)
        """
        comparison_results = {}
        
        for strategy_name, (retrieval_result, generated_answer) in strategy_results.items():
            # Evaluate retrieval
            retrieval_metrics = await self.evaluator.evaluate_retrieval(
                query,
                retrieval_result.documents,
                ground_truth
            )
            
            # Evaluate generation
            generation_metrics = await self.evaluator.evaluate_generation(
                query,
                generated_answer,
                retrieval_result.documents,
                reference_answer
            )
            
            comparison_results[strategy_name] = (retrieval_metrics, generation_metrics)
        
        return comparison_results