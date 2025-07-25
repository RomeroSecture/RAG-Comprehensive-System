import pytest
from datetime import datetime, timezone
from uuid import UUID, uuid4

from src.domain.entities.retrieval_result import (
    RetrievedDocument,
    RetrievalResult,
    SelfRAGResult
)


class TestRetrievedDocument:
    """Unit tests for RetrievedDocument entity."""
    
    def test_retrieved_document_creation(self):
        """Test creating retrieved document with valid data."""
        doc_id = uuid4()
        chunk_id = uuid4()
        
        document = RetrievedDocument(
            document_id=doc_id,
            chunk_id=chunk_id,
            content="This is retrieved content",
            metadata={"page": 1, "section": "intro"},
            score=0.85,
            source="document.pdf",
            retrieval_method="semantic",
            rerank_score=0.92
        )
        
        assert isinstance(document.id, UUID)
        assert document.document_id == doc_id
        assert document.chunk_id == chunk_id
        assert document.content == "This is retrieved content"
        assert document.metadata == {"page": 1, "section": "intro"}
        assert document.score == 0.85
        assert document.source == "document.pdf"
        assert document.retrieval_method == "semantic"
        assert document.rerank_score == 0.92
        assert isinstance(document.created_at, datetime)
    
    def test_retrieved_document_with_empty_content_raises_error(self):
        """Test that empty content raises error."""
        with pytest.raises(ValueError, match="Retrieved document content cannot be empty"):
            RetrievedDocument(content="")
    
    def test_retrieved_document_with_invalid_score_raises_error(self):
        """Test that score outside [0,1] raises error."""
        # Score > 1
        with pytest.raises(ValueError, match="Score must be between 0 and 1"):
            RetrievedDocument(content="test", score=1.5)
        
        # Score < 0
        with pytest.raises(ValueError, match="Score must be between 0 and 1"):
            RetrievedDocument(content="test", score=-0.1)
    
    def test_apply_rerank_score(self):
        """Test applying rerank score."""
        document = RetrievedDocument(content="test", score=0.7)
        assert document.rerank_score is None
        
        document.apply_rerank_score(0.9)
        assert document.rerank_score == 0.9
    
    def test_apply_invalid_rerank_score_raises_error(self):
        """Test that invalid rerank score raises error."""
        document = RetrievedDocument(content="test", score=0.7)
        
        with pytest.raises(ValueError, match="Rerank score must be between 0 and 1"):
            document.apply_rerank_score(1.5)
        
        with pytest.raises(ValueError, match="Rerank score must be between 0 and 1"):
            document.apply_rerank_score(-0.1)
    
    def test_final_score_property(self):
        """Test final score property."""
        # Without rerank score
        document = RetrievedDocument(content="test", score=0.7)
        assert document.final_score == 0.7
        
        # With rerank score
        document.apply_rerank_score(0.9)
        assert document.final_score == 0.9
    
    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        doc_id = uuid4()
        chunk_id = uuid4()
        
        document = RetrievedDocument(
            id=uuid4(),
            document_id=doc_id,
            chunk_id=chunk_id,
            content="test content",
            metadata={"key": "value"},
            score=0.8,
            source="test.pdf",
            retrieval_method="hybrid",
            rerank_score=0.85
        )
        
        result = document.to_dict()
        
        assert result["document_id"] == str(doc_id)
        assert result["chunk_id"] == str(chunk_id)
        assert result["content"] == "test content"
        assert result["metadata"] == {"key": "value"}
        assert result["score"] == 0.8
        assert result["rerank_score"] == 0.85
        assert result["source"] == "test.pdf"
        assert result["retrieval_method"] == "hybrid"
    
    def test_to_dict_with_none_chunk_id(self):
        """Test to_dict with None chunk_id."""
        document = RetrievedDocument(
            content="test",
            score=0.5,
            chunk_id=None
        )
        
        result = document.to_dict()
        assert result["chunk_id"] is None


class TestRetrievalResult:
    """Unit tests for RetrievalResult entity."""
    
    def test_retrieval_result_creation(self):
        """Test creating retrieval result with valid data."""
        query_id = uuid4()
        docs = [
            RetrievedDocument(content="doc1", score=0.9),
            RetrievedDocument(content="doc2", score=0.8)
        ]
        
        result = RetrievalResult(
            query_id=query_id,
            documents=docs,
            retrieval_strategy="hybrid",
            retrieval_time_ms=150.5,
            reranking_time_ms=50.3,
            metadata={"model": "all-MiniLM-L6"}
        )
        
        assert isinstance(result.id, UUID)
        assert result.query_id == query_id
        assert result.documents == docs
        assert result.total_results == 2
        assert result.retrieval_strategy == "hybrid"
        assert result.retrieval_time_ms == 150.5
        assert result.reranking_time_ms == 50.3
        assert result.metadata == {"model": "all-MiniLM-L6"}
        assert isinstance(result.created_at, datetime)
    
    def test_retrieval_result_total_results_auto_calculation(self):
        """Test that total_results is automatically calculated."""
        docs = [
            RetrievedDocument(content="doc1", score=0.9),
            RetrievedDocument(content="doc2", score=0.8),
            RetrievedDocument(content="doc3", score=0.7)
        ]
        
        result = RetrievalResult(documents=docs)
        assert result.total_results == 3
    
    def test_add_document(self):
        """Test adding documents to result."""
        result = RetrievalResult()
        assert result.total_results == 0
        
        doc1 = RetrievedDocument(content="doc1", score=0.9)
        result.add_document(doc1)
        assert result.total_results == 1
        assert doc1 in result.documents
        
        doc2 = RetrievedDocument(content="doc2", score=0.8)
        result.add_document(doc2)
        assert result.total_results == 2
        assert doc2 in result.documents
    
    def test_sort_by_score_descending(self):
        """Test sorting documents by score in descending order."""
        docs = [
            RetrievedDocument(content="low", score=0.5),
            RetrievedDocument(content="high", score=0.9),
            RetrievedDocument(content="medium", score=0.7)
        ]
        
        result = RetrievalResult(documents=docs)
        result.sort_by_score(descending=True)
        
        assert result.documents[0].score == 0.9
        assert result.documents[1].score == 0.7
        assert result.documents[2].score == 0.5
    
    def test_sort_by_score_ascending(self):
        """Test sorting documents by score in ascending order."""
        docs = [
            RetrievedDocument(content="low", score=0.5),
            RetrievedDocument(content="high", score=0.9),
            RetrievedDocument(content="medium", score=0.7)
        ]
        
        result = RetrievalResult(documents=docs)
        result.sort_by_score(descending=False)
        
        assert result.documents[0].score == 0.5
        assert result.documents[1].score == 0.7
        assert result.documents[2].score == 0.9
    
    def test_sort_by_score_with_rerank(self):
        """Test sorting considers rerank scores."""
        doc1 = RetrievedDocument(content="doc1", score=0.9)
        doc1.apply_rerank_score(0.6)  # Lower rerank score
        
        doc2 = RetrievedDocument(content="doc2", score=0.5)
        doc2.apply_rerank_score(0.8)  # Higher rerank score
        
        result = RetrievalResult(documents=[doc1, doc2])
        result.sort_by_score()
        
        # doc2 should be first due to higher rerank score
        assert result.documents[0].content == "doc2"
        assert result.documents[1].content == "doc1"
    
    def test_filter_by_threshold(self):
        """Test filtering documents by score threshold."""
        docs = [
            RetrievedDocument(content="low", score=0.3),
            RetrievedDocument(content="medium", score=0.6),
            RetrievedDocument(content="high", score=0.9)
        ]
        
        result = RetrievalResult(documents=docs)
        result.filter_by_threshold(0.5)
        
        assert result.total_results == 2
        assert all(doc.score >= 0.5 for doc in result.documents)
        assert result.documents[0].content == "medium"
        assert result.documents[1].content == "high"
    
    def test_get_top_k(self):
        """Test getting top k documents."""
        docs = [
            RetrievedDocument(content="doc1", score=0.5),
            RetrievedDocument(content="doc2", score=0.9),
            RetrievedDocument(content="doc3", score=0.7),
            RetrievedDocument(content="doc4", score=0.6)
        ]
        
        result = RetrievalResult(documents=docs)
        top_2 = result.get_top_k(2)
        
        assert len(top_2) == 2
        assert top_2[0].score == 0.9
        assert top_2[1].score == 0.7
    
    def test_average_score_property(self):
        """Test average score calculation."""
        docs = [
            RetrievedDocument(content="doc1", score=0.8),
            RetrievedDocument(content="doc2", score=0.6),
            RetrievedDocument(content="doc3", score=0.7)
        ]
        
        result = RetrievalResult(documents=docs)
        assert pytest.approx(result.average_score, rel=1e-6) == 0.7
    
    def test_average_score_empty_documents(self):
        """Test average score with no documents."""
        result = RetrievalResult()
        assert result.average_score == 0.0
    
    def test_average_score_with_rerank(self):
        """Test average score considers rerank scores."""
        doc1 = RetrievedDocument(content="doc1", score=0.8)
        doc1.apply_rerank_score(0.9)
        
        doc2 = RetrievedDocument(content="doc2", score=0.6)
        # No rerank score
        
        result = RetrievalResult(documents=[doc1, doc2])
        # Average of 0.9 and 0.6
        assert result.average_score == 0.75
    
    def test_total_time_ms_property(self):
        """Test total time calculation."""
        # Without reranking
        result1 = RetrievalResult(retrieval_time_ms=100.0)
        assert result1.total_time_ms == 100.0
        
        # With reranking
        result2 = RetrievalResult(
            retrieval_time_ms=100.0,
            reranking_time_ms=50.0
        )
        assert result2.total_time_ms == 150.0


class TestSelfRAGResult:
    """Unit tests for SelfRAGResult entity."""
    
    def test_self_rag_result_creation(self):
        """Test creating Self-RAG result with valid data."""
        result = SelfRAGResult(
            confidence_score=0.85,
            retrieval_attempts=3,
            reformulated_queries=["query1", "query2"],
            needs_additional_retrieval=False
        )
        
        assert result.confidence_score == 0.85
        assert result.retrieval_attempts == 3
        assert result.reformulated_queries == ["query1", "query2"]
        assert result.needs_additional_retrieval is False
        assert result.retrieval_decisions == []
    
    def test_self_rag_result_inherits_from_retrieval_result(self):
        """Test that SelfRAGResult inherits RetrievalResult functionality."""
        docs = [
            RetrievedDocument(content="doc1", score=0.9),
            RetrievedDocument(content="doc2", score=0.8)
        ]
        
        result = SelfRAGResult(
            documents=docs,
            confidence_score=0.9
        )
        
        # Should have RetrievalResult properties
        assert result.total_results == 2
        assert pytest.approx(result.average_score, rel=1e-6) == 0.85
        
        # Can use RetrievalResult methods
        result.sort_by_score()
        assert result.documents[0].score == 0.9
    
    def test_self_rag_result_with_invalid_confidence_raises_error(self):
        """Test that invalid confidence score raises error."""
        # Confidence > 1
        with pytest.raises(ValueError, match="Confidence score must be between 0 and 1"):
            SelfRAGResult(confidence_score=1.5)
        
        # Confidence < 0
        with pytest.raises(ValueError, match="Confidence score must be between 0 and 1"):
            SelfRAGResult(confidence_score=-0.1)
    
    def test_add_retrieval_decision(self):
        """Test adding retrieval decisions."""
        result = SelfRAGResult(confidence_score=0.7)
        
        decision1 = {
            "action": "reformulate",
            "reason": "low confidence",
            "confidence": 0.6
        }
        
        result.add_retrieval_decision(decision1)
        
        assert len(result.retrieval_decisions) == 1
        assert result.retrieval_decisions[0]["action"] == "reformulate"
        assert result.retrieval_decisions[0]["reason"] == "low confidence"
        assert result.retrieval_decisions[0]["confidence"] == 0.6
        assert "timestamp" in result.retrieval_decisions[0]
        
        # Add another decision
        decision2 = {
            "action": "retrieve_more",
            "reason": "insufficient results"
        }
        
        result.add_retrieval_decision(decision2)
        assert len(result.retrieval_decisions) == 2
    
    def test_mark_low_confidence(self):
        """Test marking result as low confidence."""
        result = SelfRAGResult(confidence_score=0.4)
        assert result.needs_additional_retrieval is False
        
        result.mark_low_confidence()
        assert result.needs_additional_retrieval is True
    
    def test_self_rag_result_defaults(self):
        """Test default values for SelfRAGResult."""
        result = SelfRAGResult()
        
        assert result.confidence_score == 0.0
        assert result.retrieval_attempts == 1
        assert result.reformulated_queries == []
        assert result.retrieval_decisions == []
        assert result.needs_additional_retrieval is False