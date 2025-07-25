import pytest
from typing import List, Optional, Dict
from uuid import UUID, uuid4

from src.domain.value_objects.embedding import Embedding, SparseEmbedding, HybridEmbedding
from src.domain.repositories.embedding_repository import (
    EmbeddingRepository,
    SparseEmbeddingRepository,
    HybridEmbeddingRepository
)


class MockEmbeddingRepository(EmbeddingRepository):
    """Mock implementation of EmbeddingRepository for testing."""
    
    def __init__(self):
        self.embeddings = {}
        self.chunk_embeddings = {}  # chunk_id -> embedding_id mapping
    
    async def save(self, embedding: Embedding, chunk_id: UUID) -> Embedding:
        self.embeddings[embedding.id] = embedding
        self.chunk_embeddings[chunk_id] = embedding.id
        return embedding
    
    async def save_many(self, embeddings: List[tuple[Embedding, UUID]]) -> List[Embedding]:
        saved = []
        for embedding, chunk_id in embeddings:
            saved_embedding = await self.save(embedding, chunk_id)
            saved.append(saved_embedding)
        return saved
    
    async def get_by_id(self, embedding_id: UUID) -> Optional[Embedding]:
        return self.embeddings.get(embedding_id)
    
    async def get_by_chunk_id(self, chunk_id: UUID) -> Optional[Embedding]:
        embedding_id = self.chunk_embeddings.get(chunk_id)
        if embedding_id:
            return self.embeddings.get(embedding_id)
        return None
    
    async def get_by_chunk_ids(self, chunk_ids: List[UUID]) -> List[Embedding]:
        embeddings = []
        for chunk_id in chunk_ids:
            embedding = await self.get_by_chunk_id(chunk_id)
            if embedding:
                embeddings.append(embedding)
        return embeddings
    
    async def delete_by_chunk_id(self, chunk_id: UUID) -> bool:
        if chunk_id in self.chunk_embeddings:
            embedding_id = self.chunk_embeddings[chunk_id]
            del self.chunk_embeddings[chunk_id]
            if embedding_id in self.embeddings:
                del self.embeddings[embedding_id]
            return True
        return False
    
    async def update_vector(self, embedding_id: UUID, new_vector: List[float]) -> bool:
        if embedding_id in self.embeddings:
            old_embedding = self.embeddings[embedding_id]
            # Create new embedding with updated vector (immutable)
            new_embedding = Embedding(
                id=old_embedding.id,
                vector=new_vector,
                model=old_embedding.model,
                dimensions=len(new_vector)
            )
            self.embeddings[embedding_id] = new_embedding
            return True
        return False
    
    async def exists(self, embedding_id: UUID) -> bool:
        return embedding_id in self.embeddings
    
    async def count_by_model(self, model: str) -> int:
        return sum(1 for emb in self.embeddings.values() if emb.model == model)


class MockSparseEmbeddingRepository(SparseEmbeddingRepository):
    """Mock implementation of SparseEmbeddingRepository."""
    
    def __init__(self):
        self.embeddings: Dict[UUID, SparseEmbedding] = {}
    
    async def save(self, embedding: SparseEmbedding, chunk_id: UUID) -> SparseEmbedding:
        self.embeddings[chunk_id] = embedding
        return embedding
    
    async def get_by_chunk_id(self, chunk_id: UUID) -> Optional[SparseEmbedding]:
        return self.embeddings.get(chunk_id)
    
    async def delete_by_chunk_id(self, chunk_id: UUID) -> bool:
        if chunk_id in self.embeddings:
            del self.embeddings[chunk_id]
            return True
        return False
    
    async def update_vocabulary_size(self, new_size: int) -> int:
        count = 0
        for chunk_id, old_emb in list(self.embeddings.items()):
            # Create new embedding with updated vocabulary size
            new_emb = SparseEmbedding(
                id=old_emb.id,
                indices=old_emb.indices,
                values=old_emb.values,
                vocabulary_size=new_size
            )
            self.embeddings[chunk_id] = new_emb
            count += 1
        return count


class MockHybridEmbeddingRepository(HybridEmbeddingRepository):
    """Mock implementation of HybridEmbeddingRepository."""
    
    def __init__(self):
        self.embeddings: Dict[UUID, HybridEmbedding] = {}
    
    async def save(self, embedding: HybridEmbedding, chunk_id: UUID) -> HybridEmbedding:
        self.embeddings[chunk_id] = embedding
        return embedding
    
    async def get_by_chunk_id(self, chunk_id: UUID) -> Optional[HybridEmbedding]:
        return self.embeddings.get(chunk_id)
    
    async def update_weights(self, chunk_id: UUID, dense_weight: float, sparse_weight: float) -> bool:
        if chunk_id in self.embeddings:
            old_emb = self.embeddings[chunk_id]
            # Create new embedding with updated weights
            new_emb = HybridEmbedding(
                dense=old_emb.dense,
                sparse=old_emb.sparse,
                weight_dense=dense_weight,
                weight_sparse=sparse_weight
            )
            self.embeddings[chunk_id] = new_emb
            return True
        return False


class TestEmbeddingRepository:
    """Test cases for EmbeddingRepository interface."""
    
    @pytest.mark.asyncio
    async def test_repository_is_abstract(self):
        """Test that EmbeddingRepository is abstract."""
        with pytest.raises(TypeError):
            EmbeddingRepository()
    
    @pytest.mark.asyncio
    async def test_save_and_get_embedding(self):
        """Test saving and retrieving an embedding."""
        repo = MockEmbeddingRepository()
        
        embedding = Embedding.create(
            vector=[0.1, 0.2, 0.3, 0.4],
            model="test-model"
        )
        chunk_id = uuid4()
        
        # Save embedding
        saved = await repo.save(embedding, chunk_id)
        assert saved == embedding
        
        # Get by ID
        retrieved = await repo.get_by_id(embedding.id)
        assert retrieved == embedding
        
        # Get by chunk ID
        retrieved_by_chunk = await repo.get_by_chunk_id(chunk_id)
        assert retrieved_by_chunk == embedding
    
    @pytest.mark.asyncio
    async def test_save_many_embeddings(self):
        """Test saving multiple embeddings in batch."""
        repo = MockEmbeddingRepository()
        
        embeddings_data = []
        for i in range(3):
            embedding = Embedding.create(
                vector=[float(i) * 0.1] * 4,
                model="test-model"
            )
            chunk_id = uuid4()
            embeddings_data.append((embedding, chunk_id))
        
        # Save many
        saved = await repo.save_many(embeddings_data)
        assert len(saved) == 3
        
        # Verify all were saved
        for embedding, chunk_id in embeddings_data:
            retrieved = await repo.get_by_chunk_id(chunk_id)
            assert retrieved == embedding
    
    @pytest.mark.asyncio
    async def test_get_by_chunk_ids(self):
        """Test retrieving embeddings for multiple chunks."""
        repo = MockEmbeddingRepository()
        
        # Save embeddings
        chunk_ids = []
        embeddings = []
        for i in range(5):
            embedding = Embedding.create(
                vector=[float(i)] * 4,
                model="test-model"
            )
            chunk_id = uuid4()
            await repo.save(embedding, chunk_id)
            chunk_ids.append(chunk_id)
            embeddings.append(embedding)
        
        # Get specific embeddings
        selected_ids = [chunk_ids[0], chunk_ids[2], chunk_ids[4]]
        retrieved = await repo.get_by_chunk_ids(selected_ids)
        
        assert len(retrieved) == 3
        assert embeddings[0] in retrieved
        assert embeddings[2] in retrieved
        assert embeddings[4] in retrieved
    
    @pytest.mark.asyncio
    async def test_delete_by_chunk_id(self):
        """Test deleting an embedding by chunk ID."""
        repo = MockEmbeddingRepository()
        
        embedding = Embedding.create(vector=[0.1, 0.2], model="test")
        chunk_id = uuid4()
        await repo.save(embedding, chunk_id)
        
        # Delete
        result = await repo.delete_by_chunk_id(chunk_id)
        assert result is True
        
        # Verify deletion
        retrieved = await repo.get_by_chunk_id(chunk_id)
        assert retrieved is None
        
        # Delete non-existent
        result = await repo.delete_by_chunk_id(uuid4())
        assert result is False
    
    @pytest.mark.asyncio
    async def test_update_vector(self):
        """Test updating an embedding vector."""
        repo = MockEmbeddingRepository()
        
        embedding = Embedding.create(
            vector=[0.1, 0.2, 0.3],
            model="test-model"
        )
        chunk_id = uuid4()
        await repo.save(embedding, chunk_id)
        
        # Update vector
        new_vector = [0.4, 0.5, 0.6]
        result = await repo.update_vector(embedding.id, new_vector)
        assert result is True
        
        # Verify update
        updated = await repo.get_by_id(embedding.id)
        assert updated.vector == new_vector
        assert updated.dimensions == 3
        
        # Update non-existent
        result = await repo.update_vector(uuid4(), new_vector)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_exists(self):
        """Test checking if an embedding exists."""
        repo = MockEmbeddingRepository()
        
        embedding = Embedding.create([0.1, 0.2], "test")
        chunk_id = uuid4()
        await repo.save(embedding, chunk_id)
        
        assert await repo.exists(embedding.id) is True
        assert await repo.exists(uuid4()) is False
    
    @pytest.mark.asyncio
    async def test_count_by_model(self):
        """Test counting embeddings by model."""
        repo = MockEmbeddingRepository()
        
        # Save embeddings with different models
        models = ["model-a", "model-a", "model-b", "model-a", "model-c"]
        for i, model in enumerate(models):
            embedding = Embedding.create(
                vector=[float(i)],
                model=model
            )
            await repo.save(embedding, uuid4())
        
        assert await repo.count_by_model("model-a") == 3
        assert await repo.count_by_model("model-b") == 1
        assert await repo.count_by_model("model-c") == 1
        assert await repo.count_by_model("model-d") == 0


class TestSparseEmbeddingRepository:
    """Test cases for SparseEmbeddingRepository interface."""
    
    @pytest.mark.asyncio
    async def test_repository_is_abstract(self):
        """Test that SparseEmbeddingRepository is abstract."""
        with pytest.raises(TypeError):
            SparseEmbeddingRepository()
    
    @pytest.mark.asyncio
    async def test_save_and_get_sparse_embedding(self):
        """Test saving and retrieving a sparse embedding."""
        repo = MockSparseEmbeddingRepository()
        
        embedding = SparseEmbedding.create(
            indices=[0, 5, 10],
            values=[0.5, 1.0, 0.3],
            vocabulary_size=100
        )
        chunk_id = uuid4()
        
        # Save
        saved = await repo.save(embedding, chunk_id)
        assert saved == embedding
        
        # Get
        retrieved = await repo.get_by_chunk_id(chunk_id)
        assert retrieved == embedding
    
    @pytest.mark.asyncio
    async def test_delete_sparse_embedding(self):
        """Test deleting a sparse embedding."""
        repo = MockSparseEmbeddingRepository()
        
        embedding = SparseEmbedding.create([1, 2], [0.5, 0.5], 10)
        chunk_id = uuid4()
        await repo.save(embedding, chunk_id)
        
        # Delete
        result = await repo.delete_by_chunk_id(chunk_id)
        assert result is True
        
        # Verify
        assert await repo.get_by_chunk_id(chunk_id) is None
        
        # Delete non-existent
        assert await repo.delete_by_chunk_id(uuid4()) is False
    
    @pytest.mark.asyncio
    async def test_update_vocabulary_size(self):
        """Test updating vocabulary size for all embeddings."""
        repo = MockSparseEmbeddingRepository()
        
        # Save multiple embeddings
        for i in range(3):
            embedding = SparseEmbedding.create(
                indices=[i, i + 1],
                values=[0.5, 0.5],
                vocabulary_size=100
            )
            await repo.save(embedding, uuid4())
        
        # Update vocabulary size
        count = await repo.update_vocabulary_size(200)
        assert count == 3
        
        # Verify all embeddings were updated
        for chunk_id, embedding in repo.embeddings.items():
            assert embedding.vocabulary_size == 200


class TestHybridEmbeddingRepository:
    """Test cases for HybridEmbeddingRepository interface."""
    
    @pytest.mark.asyncio
    async def test_repository_is_abstract(self):
        """Test that HybridEmbeddingRepository is abstract."""
        with pytest.raises(TypeError):
            HybridEmbeddingRepository()
    
    @pytest.mark.asyncio
    async def test_save_and_get_hybrid_embedding(self):
        """Test saving and retrieving a hybrid embedding."""
        repo = MockHybridEmbeddingRepository()
        
        dense = Embedding.create([0.1, 0.2, 0.3], "dense-model")
        sparse = SparseEmbedding.create([0, 2], [0.5, 0.8], 10)
        
        hybrid = HybridEmbedding(
            dense=dense,
            sparse=sparse,
            weight_dense=0.7,
            weight_sparse=0.3
        )
        chunk_id = uuid4()
        
        # Save
        saved = await repo.save(hybrid, chunk_id)
        assert saved == hybrid
        
        # Get
        retrieved = await repo.get_by_chunk_id(chunk_id)
        assert retrieved == hybrid
    
    @pytest.mark.asyncio
    async def test_update_weights(self):
        """Test updating weights of a hybrid embedding."""
        repo = MockHybridEmbeddingRepository()
        
        dense = Embedding.create([0.1, 0.2], "model")
        hybrid = HybridEmbedding(dense=dense)
        chunk_id = uuid4()
        
        await repo.save(hybrid, chunk_id)
        
        # Update weights
        result = await repo.update_weights(chunk_id, 0.6, 0.4)
        assert result is True
        
        # Verify
        updated = await repo.get_by_chunk_id(chunk_id)
        assert updated.weight_dense == 0.6
        assert updated.weight_sparse == 0.4
        
        # Update non-existent
        result = await repo.update_weights(uuid4(), 0.5, 0.5)
        assert result is False