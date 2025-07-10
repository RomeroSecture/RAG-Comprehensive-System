from abc import ABC, abstractmethod
from typing import List, Optional, Protocol

from ..entities.query import Query
from ..entities.retrieval_result import RetrievedDocument
from ..value_objects.similarity_score import RerankingScore


class Reranker(Protocol):
    """Protocol for reranking models."""
    
    async def rerank(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_k: Optional[int] = None
    ) -> List[tuple[RetrievedDocument, float]]:
        """Rerank documents and return with new scores."""
        ...


class RankingService(ABC):
    """Abstract service for document ranking and reranking."""
    
    @abstractmethod
    async def rank_documents(
        self,
        query: Query,
        documents: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        """Rank or rerank documents based on the query."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the name of the ranking model."""
        pass


class CrossEncoderRankingService(RankingService):
    """Ranking service using cross-encoder models for high precision."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        self.model_name = model_name
    
    def get_model_name(self) -> str:
        return self.model_name
    
    async def rank_documents(
        self,
        query: Query,
        documents: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        """Rerank documents using cross-encoder."""
        # Implementation would use actual cross-encoder model
        pass


class ColBERTRankingService(RankingService):
    """Ranking service using ColBERT for efficient late interaction."""
    
    def __init__(self, model_name: str = "colbert-ir/colbertv2.0"):
        self.model_name = model_name
    
    def get_model_name(self) -> str:
        return self.model_name
    
    async def rank_documents(
        self,
        query: Query,
        documents: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        """Rerank documents using ColBERT late interaction."""
        # Implementation would use actual ColBERT model
        pass


class LLMRankingService(RankingService):
    """Ranking service using LLM for reranking (RankGPT approach)."""
    
    def __init__(self, model_name: str = "gpt-4"):
        self.model_name = model_name
    
    def get_model_name(self) -> str:
        return self.model_name
    
    async def rank_documents(
        self,
        query: Query,
        documents: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        """Rerank documents using LLM-based ranking."""
        # Implementation would use LLM for ranking
        pass


class MultiStageRankingService:
    """Service for multi-stage ranking pipeline."""
    
    def __init__(self, stages: List[tuple[RankingService, int]]):
        """
        Initialize with ranking stages.
        Each stage is a tuple of (RankingService, top_k_to_keep).
        """
        self.stages = stages
    
    async def rank_documents(
        self,
        query: Query,
        documents: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        """Apply multi-stage ranking pipeline."""
        current_docs = documents
        
        for ranking_service, top_k in self.stages:
            # Apply ranking
            current_docs = await ranking_service.rank_documents(query, current_docs)
            
            # Keep only top_k documents
            if top_k and len(current_docs) > top_k:
                current_docs = current_docs[:top_k]
        
        return current_docs


class ReciprocalRankFusion:
    """Service for fusing rankings from multiple sources."""
    
    def __init__(self, k: int = 60):
        """
        Initialize RRF with parameter k.
        k is a parameter that controls the fusion behavior.
        """
        self.k = k
    
    def fuse_rankings(
        self,
        rankings: List[List[RetrievedDocument]]
    ) -> List[RetrievedDocument]:
        """
        Fuse multiple rankings using Reciprocal Rank Fusion.
        Each input is a ranked list of documents.
        """
        # Calculate RRF scores
        doc_scores = {}
        
        for ranking in rankings:
            for rank, doc in enumerate(ranking):
                doc_id = doc.id
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        "doc": doc,
                        "score": 0.0
                    }
                
                # RRF formula: 1 / (k + rank)
                doc_scores[doc_id]["score"] += 1.0 / (self.k + rank + 1)
        
        # Sort by RRF score
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        
        # Update document scores with RRF scores
        result_docs = []
        for item in sorted_docs:
            doc = item["doc"]
            # Normalize RRF score to [0, 1]
            doc.score = item["score"] / len(rankings)
            result_docs.append(doc)
        
        return result_docs


class DiversityRankingService:
    """Service for promoting diversity in rankings using MMR."""
    
    def __init__(self, lambda_param: float = 0.5):
        """
        Initialize with diversity parameter.
        lambda_param controls the trade-off between relevance and diversity.
        """
        self.lambda_param = lambda_param
    
    async def diversify_rankings(
        self,
        documents: List[RetrievedDocument],
        top_k: int = 10
    ) -> List[RetrievedDocument]:
        """Apply Maximal Marginal Relevance (MMR) for diversity."""
        if not documents:
            return []
        
        # Start with the most relevant document
        selected = [documents[0]]
        candidates = documents[1:]
        
        while len(selected) < top_k and candidates:
            # Calculate MMR scores for remaining candidates
            mmr_scores = []
            
            for candidate in candidates:
                # Relevance score (already in the document)
                relevance = candidate.final_score
                
                # Maximum similarity to already selected documents
                max_sim = max(
                    self._calculate_similarity(candidate, selected_doc)
                    for selected_doc in selected
                )
                
                # MMR score
                mmr = self.lambda_param * relevance - (1 - self.lambda_param) * max_sim
                mmr_scores.append((candidate, mmr))
            
            # Select document with highest MMR score
            mmr_scores.sort(key=lambda x: x[1], reverse=True)
            selected_doc = mmr_scores[0][0]
            
            selected.append(selected_doc)
            candidates.remove(selected_doc)
        
        return selected
    
    def _calculate_similarity(
        self,
        doc1: RetrievedDocument,
        doc2: RetrievedDocument
    ) -> float:
        """Calculate similarity between two documents."""
        # Simplified similarity based on content overlap
        # In practice, this would use embeddings or other features
        words1 = set(doc1.content.lower().split())
        words2 = set(doc2.content.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0