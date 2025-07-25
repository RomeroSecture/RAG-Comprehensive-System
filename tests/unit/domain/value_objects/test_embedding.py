import pytest
import numpy as np
from uuid import UUID, uuid4

from src.domain.value_objects.embedding import Embedding, SparseEmbedding, HybridEmbedding


class TestEmbedding:
    """Unit tests for Embedding value object."""
    
    def test_embedding_creation_with_valid_data(self, sample_embedding_data):
        """Test creating embedding with valid data."""
        embedding_id = uuid4()
        embedding = Embedding(
            id=embedding_id,
            vector=sample_embedding_data["vector"],
            model=sample_embedding_data["model"],
            dimensions=sample_embedding_data["dimensions"]
        )
        
        assert embedding.id == embedding_id
        assert embedding.vector == sample_embedding_data["vector"]
        assert embedding.model == sample_embedding_data["model"]
        assert embedding.dimensions == sample_embedding_data["dimensions"]
    
    def test_embedding_is_immutable(self):
        """Test that embedding is immutable (frozen dataclass)."""
        embedding = Embedding.create([1.0, 2.0, 3.0], "test-model")
        
        with pytest.raises(AttributeError):
            embedding.vector = [4.0, 5.0, 6.0]
        
        with pytest.raises(AttributeError):
            embedding.model = "new-model"
    
    def test_embedding_with_empty_vector_raises_error(self):
        """Test that empty vector raises ValueError."""
        with pytest.raises(ValueError, match="Embedding vector cannot be empty"):
            Embedding(id=uuid4(), vector=[], model="test", dimensions=0)
    
    def test_embedding_with_dimension_mismatch_raises_error(self):
        """Test that dimension mismatch raises ValueError."""
        with pytest.raises(ValueError, match="Dimensions mismatch: expected 5, got 3"):
            Embedding(
                id=uuid4(),
                vector=[1.0, 2.0, 3.0],
                model="test",
                dimensions=5
            )
    
    def test_embedding_with_empty_model_raises_error(self):
        """Test that empty model name raises ValueError."""
        with pytest.raises(ValueError, match="Embedding model must be specified"):
            Embedding(
                id=uuid4(),
                vector=[1.0, 2.0, 3.0],
                model="",
                dimensions=3
            )
    
    def test_create_factory_method(self):
        """Test the create factory method."""
        vector = [1.0, 2.0, 3.0, 4.0]
        model = "test-model"
        
        embedding = Embedding.create(vector, model)
        
        assert isinstance(embedding.id, UUID)
        assert embedding.vector == vector
        assert embedding.model == model
        assert embedding.dimensions == len(vector)
    
    def test_to_numpy(self):
        """Test conversion to numpy array."""
        vector = [1.0, 2.0, 3.0]
        embedding = Embedding.create(vector, "test-model")
        
        np_array = embedding.to_numpy()
        
        assert isinstance(np_array, np.ndarray)
        assert np_array.dtype == np.float32
        assert np_array.tolist() == vector
    
    def test_cosine_similarity_identical_vectors(self):
        """Test cosine similarity with identical vectors."""
        vector = [1.0, 2.0, 3.0]
        emb1 = Embedding.create(vector, "test-model")
        emb2 = Embedding.create(vector, "test-model")
        
        similarity = emb1.cosine_similarity(emb2)
        assert pytest.approx(similarity, rel=1e-6) == 1.0
    
    def test_cosine_similarity_orthogonal_vectors(self):
        """Test cosine similarity with orthogonal vectors."""
        emb1 = Embedding.create([1.0, 0.0, 0.0], "test-model")
        emb2 = Embedding.create([0.0, 1.0, 0.0], "test-model")
        
        similarity = emb1.cosine_similarity(emb2)
        assert pytest.approx(similarity, abs=1e-6) == 0.0
    
    def test_cosine_similarity_opposite_vectors(self):
        """Test cosine similarity with opposite vectors."""
        emb1 = Embedding.create([1.0, 2.0, 3.0], "test-model")
        emb2 = Embedding.create([-1.0, -2.0, -3.0], "test-model")
        
        similarity = emb1.cosine_similarity(emb2)
        assert pytest.approx(similarity, rel=1e-6) == -1.0
    
    def test_cosine_similarity_dimension_mismatch_raises_error(self):
        """Test cosine similarity with different dimensions raises error."""
        emb1 = Embedding.create([1.0, 2.0, 3.0], "test-model")
        emb2 = Embedding.create([1.0, 2.0], "test-model")
        
        with pytest.raises(ValueError, match="Embeddings must have the same dimensions"):
            emb1.cosine_similarity(emb2)
    
    def test_cosine_similarity_zero_vector(self):
        """Test cosine similarity with zero vector."""
        emb1 = Embedding.create([1.0, 2.0, 3.0], "test-model")
        emb2 = Embedding.create([0.0, 0.0, 0.0], "test-model")
        
        similarity = emb1.cosine_similarity(emb2)
        assert similarity == 0.0
    
    def test_euclidean_distance_identical_vectors(self):
        """Test Euclidean distance with identical vectors."""
        vector = [1.0, 2.0, 3.0]
        emb1 = Embedding.create(vector, "test-model")
        emb2 = Embedding.create(vector, "test-model")
        
        distance = emb1.euclidean_distance(emb2)
        assert pytest.approx(distance, abs=1e-6) == 0.0
    
    def test_euclidean_distance_different_vectors(self):
        """Test Euclidean distance with different vectors."""
        emb1 = Embedding.create([0.0, 0.0, 0.0], "test-model")
        emb2 = Embedding.create([3.0, 4.0, 0.0], "test-model")
        
        distance = emb1.euclidean_distance(emb2)
        assert pytest.approx(distance, abs=1e-6) == 5.0  # 3-4-5 triangle
    
    def test_euclidean_distance_dimension_mismatch_raises_error(self):
        """Test Euclidean distance with different dimensions raises error."""
        emb1 = Embedding.create([1.0, 2.0, 3.0], "test-model")
        emb2 = Embedding.create([1.0, 2.0], "test-model")
        
        with pytest.raises(ValueError, match="Embeddings must have the same dimensions"):
            emb1.euclidean_distance(emb2)
    
    def test_dot_product(self):
        """Test dot product calculation."""
        emb1 = Embedding.create([2.0, 3.0, 4.0], "test-model")
        emb2 = Embedding.create([1.0, 2.0, 3.0], "test-model")
        
        dot_prod = emb1.dot_product(emb2)
        # 2*1 + 3*2 + 4*3 = 2 + 6 + 12 = 20
        assert pytest.approx(dot_prod, abs=1e-6) == 20.0
    
    def test_dot_product_dimension_mismatch_raises_error(self):
        """Test dot product with different dimensions raises error."""
        emb1 = Embedding.create([1.0, 2.0, 3.0], "test-model")
        emb2 = Embedding.create([1.0, 2.0], "test-model")
        
        with pytest.raises(ValueError, match="Embeddings must have the same dimensions"):
            emb1.dot_product(emb2)


class TestSparseEmbedding:
    """Unit tests for SparseEmbedding value object."""
    
    def test_sparse_embedding_creation(self):
        """Test creating sparse embedding with valid data."""
        indices = [1, 5, 10]
        values = [0.5, 1.0, 0.3]
        vocab_size = 20
        
        sparse_emb = SparseEmbedding(
            id=uuid4(),
            indices=indices,
            values=values,
            vocabulary_size=vocab_size
        )
        
        assert sparse_emb.indices == indices
        assert sparse_emb.values == values
        assert sparse_emb.vocabulary_size == vocab_size
    
    def test_sparse_embedding_is_immutable(self):
        """Test that sparse embedding is immutable."""
        sparse_emb = SparseEmbedding.create([1, 2], [0.5, 0.5], 10)
        
        with pytest.raises(AttributeError):
            sparse_emb.indices = [3, 4]
    
    def test_sparse_embedding_mismatched_lengths_raises_error(self):
        """Test that mismatched indices/values lengths raise error."""
        with pytest.raises(ValueError, match="Indices and values must have the same length"):
            SparseEmbedding(
                id=uuid4(),
                indices=[1, 2, 3],
                values=[0.5, 0.5],  # Only 2 values for 3 indices
                vocabulary_size=10
            )
    
    def test_sparse_embedding_invalid_index_raises_error(self):
        """Test that invalid indices raise error."""
        # Test negative index
        with pytest.raises(ValueError, match="Invalid index in sparse embedding"):
            SparseEmbedding(
                id=uuid4(),
                indices=[-1, 2],
                values=[0.5, 0.5],
                vocabulary_size=10
            )
        
        # Test index >= vocabulary size
        with pytest.raises(ValueError, match="Invalid index in sparse embedding"):
            SparseEmbedding(
                id=uuid4(),
                indices=[5, 10],
                values=[0.5, 0.5],
                vocabulary_size=10  # Max valid index is 9
            )
    
    def test_create_factory_method(self):
        """Test the create factory method."""
        indices = [0, 3, 7]
        values = [1.0, 2.0, 3.0]
        vocab_size = 10
        
        sparse_emb = SparseEmbedding.create(indices, values, vocab_size)
        
        assert isinstance(sparse_emb.id, UUID)
        assert sparse_emb.indices == indices
        assert sparse_emb.values == values
        assert sparse_emb.vocabulary_size == vocab_size
    
    def test_to_dense_conversion(self):
        """Test conversion to dense representation."""
        sparse_emb = SparseEmbedding.create(
            indices=[1, 3, 5],
            values=[0.5, 1.0, 0.3],
            vocabulary_size=8
        )
        
        dense = sparse_emb.to_dense()
        
        assert len(dense) == 8
        assert dense == [0.0, 0.5, 0.0, 1.0, 0.0, 0.3, 0.0, 0.0]
    
    def test_nnz_property(self):
        """Test non-zero elements count."""
        sparse_emb = SparseEmbedding.create(
            indices=[1, 5, 10, 15],
            values=[0.5, 1.0, 0.3, 0.7],
            vocabulary_size=20
        )
        
        assert sparse_emb.nnz == 4


class TestHybridEmbedding:
    """Unit tests for HybridEmbedding value object."""
    
    def test_hybrid_embedding_creation(self):
        """Test creating hybrid embedding."""
        dense = Embedding.create([1.0, 2.0, 3.0], "dense-model")
        sparse = SparseEmbedding.create([0, 2], [0.5, 1.0], 5)
        
        hybrid = HybridEmbedding(
            dense=dense,
            sparse=sparse,
            weight_dense=0.7,
            weight_sparse=0.3
        )
        
        assert hybrid.dense == dense
        assert hybrid.sparse == sparse
        assert hybrid.weight_dense == 0.7
        assert hybrid.weight_sparse == 0.3
    
    def test_hybrid_embedding_is_immutable(self):
        """Test that hybrid embedding is immutable."""
        dense = Embedding.create([1.0, 2.0], "model")
        hybrid = HybridEmbedding(dense=dense)
        
        with pytest.raises(AttributeError):
            hybrid.weight_dense = 0.5
    
    def test_hybrid_embedding_invalid_weights_raise_error(self):
        """Test that invalid weights raise errors."""
        dense = Embedding.create([1.0, 2.0], "model")
        
        # Test weight < 0
        with pytest.raises(ValueError, match="Dense weight must be between 0 and 1"):
            HybridEmbedding(dense=dense, weight_dense=-0.1)
        
        # Test weight > 1
        with pytest.raises(ValueError, match="Sparse weight must be between 0 and 1"):
            HybridEmbedding(dense=dense, weight_sparse=1.5)
        
        # Test weights don't sum to 1
        with pytest.raises(ValueError, match="Weights must sum to 1"):
            HybridEmbedding(dense=dense, weight_dense=0.6, weight_sparse=0.5)
    
    def test_hybrid_embedding_dense_only(self):
        """Test hybrid embedding with only dense component."""
        dense = Embedding.create([1.0, 2.0, 3.0], "model")
        hybrid = HybridEmbedding(dense=dense)
        
        assert hybrid.sparse is None
        assert hybrid.weight_dense == 0.7  # Default
        assert hybrid.weight_sparse == 0.3  # Default
    
    def test_hybrid_similarity_dense_only(self):
        """Test hybrid similarity with only dense embeddings."""
        dense1 = Embedding.create([1.0, 0.0, 0.0], "model")
        dense2 = Embedding.create([1.0, 0.0, 0.0], "model")
        
        hybrid1 = HybridEmbedding(dense=dense1)
        hybrid2 = HybridEmbedding(dense=dense2)
        
        similarity = hybrid1.hybrid_similarity(hybrid2)
        assert pytest.approx(similarity, rel=1e-6) == 1.0
    
    def test_hybrid_similarity_with_sparse(self):
        """Test hybrid similarity with both dense and sparse components."""
        # Create dense embeddings
        dense1 = Embedding.create([1.0, 0.0], "model")
        dense2 = Embedding.create([1.0, 0.0], "model")
        
        # Create sparse embeddings
        sparse1 = SparseEmbedding.create([0, 2], [1.0, 1.0], 5)
        sparse2 = SparseEmbedding.create([0, 2], [1.0, 1.0], 5)
        
        # Create hybrid embeddings
        hybrid1 = HybridEmbedding(
            dense=dense1,
            sparse=sparse1,
            weight_dense=0.5,
            weight_sparse=0.5
        )
        hybrid2 = HybridEmbedding(
            dense=dense2,
            sparse=sparse2,
            weight_dense=0.5,
            weight_sparse=0.5
        )
        
        similarity = hybrid1.hybrid_similarity(hybrid2)
        # Both dense and sparse similarities are 1.0, so result should be 1.0
        assert pytest.approx(similarity, rel=1e-6) == 1.0
    
    def test_sparse_similarity_different_vocab_sizes(self):
        """Test sparse similarity with different vocabulary sizes."""
        sparse1 = SparseEmbedding.create([0, 1], [1.0, 1.0], 5)
        sparse2 = SparseEmbedding.create([0, 1], [1.0, 1.0], 10)
        
        similarity = HybridEmbedding._sparse_similarity(sparse1, sparse2)
        assert similarity == 0.0
    
    def test_sparse_similarity_no_overlap(self):
        """Test sparse similarity with no overlapping terms."""
        sparse1 = SparseEmbedding.create([0, 1], [1.0, 1.0], 5)
        sparse2 = SparseEmbedding.create([2, 3], [1.0, 1.0], 5)
        
        similarity = HybridEmbedding._sparse_similarity(sparse1, sparse2)
        assert similarity == 0.0
    
    def test_sparse_similarity_partial_overlap(self):
        """Test sparse similarity with partial overlap."""
        sparse1 = SparseEmbedding.create([0, 1, 2], [1.0, 1.0, 1.0], 5)
        sparse2 = SparseEmbedding.create([1, 2, 3], [1.0, 1.0, 1.0], 5)
        
        similarity = HybridEmbedding._sparse_similarity(sparse1, sparse2)
        # Expected: 2 common terms out of 3 each
        # Dot product = 2.0, norms = sqrt(3) each
        # Similarity = 2.0 / (sqrt(3) * sqrt(3)) = 2/3
        assert pytest.approx(similarity, rel=1e-6) == 2/3