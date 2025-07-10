# Testing Strategy Guide - RAG Comprehensive System

## 🎯 Testing Philosophy

### Test Pyramid
```
         /\
        /e2e\        <- End-to-End Tests (10%)
       /------\
      /  int   \     <- Integration Tests (30%)
     /----------\
    /    unit    \   <- Unit Tests (60%)
   /--------------\
```

### Testing Principles
1. **Fast Feedback**: Unit tests should run in milliseconds
2. **Isolation**: Tests should not depend on external services
3. **Deterministic**: Same input always produces same output
4. **Comprehensive**: Cover happy paths, edge cases, and error scenarios
5. **Maintainable**: Tests should be easy to understand and update

## 🧪 Test Categories

### 1. Unit Tests

#### Domain Layer Testing
```python
# tests/unit/domain/test_document.py
import pytest
from datetime import datetime
from uuid import uuid4
from src.domain.entities.document import Document, DocumentChunk

class TestDocument:
    """Test Document entity business logic"""
    
    def test_create_valid_document(self):
        """Test creating a valid document"""
        doc = Document(
            id=uuid4(),
            content="Test content",
            source="test.pdf",
            file_type="pdf",
            metadata={"author": "Test Author"}
        )
        
        assert doc.content == "Test content"
        assert doc.processing_status == "pending"
        assert doc.file_type == "pdf"
    
    def test_document_requires_content(self):
        """Test that document content cannot be empty"""
        with pytest.raises(ValueError, match="content cannot be empty"):
            Document(
                content="",
                source="test.pdf",
                file_type="pdf"
            )
    
    def test_invalid_file_type(self):
        """Test validation of supported file types"""
        with pytest.raises(ValueError, match="Unsupported file type"):
            Document(
                content="Test",
                source="test.exe",
                file_type="exe"
            )
    
    def test_mark_as_processed(self):
        """Test document status transitions"""
        doc = Document(content="Test", source="test.pdf", file_type="pdf")
        original_updated_at = doc.updated_at
        
        doc.mark_as_processed()
        
        assert doc.processing_status == "completed"
        assert doc.updated_at > original_updated_at
    
    def test_mark_as_failed(self):
        """Test error handling in document processing"""
        doc = Document(content="Test", source="test.pdf", file_type="pdf")
        error_msg = "Failed to parse PDF"
        
        doc.mark_as_failed(error_msg)
        
        assert doc.processing_status == "error"
        assert doc.error_message == error_msg

class TestDocumentChunk:
    """Test DocumentChunk value constraints"""
    
    def test_chunk_position_validation(self):
        """Test chunk position constraints"""
        with pytest.raises(ValueError, match="Start position must be less than end"):
            DocumentChunk(
                document_id=uuid4(),
                content="Test",
                chunk_index=0,
                start_char=100,
                end_char=50  # Invalid: end before start
            )
    
    @pytest.mark.parametrize("start,end,expected", [
        (0, 100, 100),
        (50, 150, 100),
        (1000, 1500, 500)
    ])
    def test_chunk_char_count(self, start, end, expected):
        """Test character count calculation"""
        chunk = DocumentChunk(
            document_id=uuid4(),
            content="x" * expected,
            chunk_index=0,
            start_char=start,
            end_char=end
        )
        assert chunk.char_count == expected
```

#### Value Objects Testing
```python
# tests/unit/domain/test_embedding.py
import pytest
import numpy as np
from src.domain.value_objects.embedding import Embedding, HybridEmbedding

class TestEmbedding:
    """Test Embedding value object"""
    
    def test_create_valid_embedding(self):
        """Test creating a valid embedding"""
        vector = [0.1, 0.2, 0.3, 0.4]
        embedding = Embedding.create(vector=vector, model="test-model")
        
        assert embedding.vector == vector
        assert embedding.model == "test-model"
        assert embedding.dimensions == 4
    
    def test_embedding_immutability(self):
        """Test that embeddings are immutable"""
        embedding = Embedding.create([0.1, 0.2], "model")
        
        with pytest.raises(AttributeError):
            embedding.vector = [0.3, 0.4]
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation"""
        emb1 = Embedding.create([1.0, 0.0], "model")
        emb2 = Embedding.create([0.0, 1.0], "model")
        emb3 = Embedding.create([1.0, 0.0], "model")
        
        # Orthogonal vectors
        assert abs(emb1.cosine_similarity(emb2)) < 0.001
        
        # Identical vectors
        assert abs(emb1.cosine_similarity(emb3) - 1.0) < 0.001
    
    def test_dimension_mismatch(self):
        """Test error on dimension mismatch"""
        emb1 = Embedding.create([1.0, 0.0], "model")
        emb2 = Embedding.create([1.0, 0.0, 0.0], "model")
        
        with pytest.raises(ValueError, match="same dimensions"):
            emb1.cosine_similarity(emb2)
```

#### Service Layer Testing
```python
# tests/unit/domain/test_ranking_service.py
import pytest
from unittest.mock import Mock, AsyncMock
from src.domain.services.ranking_service import (
    ReciprocalRankFusion, DiversityRankingService
)

class TestReciprocalRankFusion:
    """Test RRF ranking fusion"""
    
    def test_fuse_single_ranking(self):
        """Test RRF with single ranking"""
        rrf = ReciprocalRankFusion(k=60)
        
        docs = [Mock(id=i) for i in range(5)]
        rankings = [docs]
        
        result = rrf.fuse_rankings(rankings)
        
        assert len(result) == 5
        assert [doc.id for doc in result] == [0, 1, 2, 3, 4]
    
    def test_fuse_multiple_rankings(self):
        """Test RRF with multiple rankings"""
        rrf = ReciprocalRankFusion(k=60)
        
        # Create mock documents
        doc_a, doc_b, doc_c = Mock(id='a'), Mock(id='b'), Mock(id='c')
        
        # Different rankings
        ranking1 = [doc_a, doc_b, doc_c]
        ranking2 = [doc_b, doc_c, doc_a]
        ranking3 = [doc_c, doc_a, doc_b]
        
        result = rrf.fuse_rankings([ranking1, ranking2, ranking3])
        
        # All documents should be present
        assert len(result) == 3
        assert set(doc.id for doc in result) == {'a', 'b', 'c'}
        
        # Check RRF scores are normalized
        assert all(0 <= doc.score <= 1 for doc in result)

class TestDiversityRanking:
    """Test MMR diversity ranking"""
    
    @pytest.mark.asyncio
    async def test_diversity_selection(self):
        """Test that MMR promotes diversity"""
        mmr = DiversityRankingService(lambda_param=0.5)
        
        # Create documents with content
        docs = [
            Mock(content="machine learning algorithms", final_score=0.9),
            Mock(content="machine learning models", final_score=0.85),
            Mock(content="deep neural networks", final_score=0.8),
            Mock(content="database systems", final_score=0.75)
        ]
        
        result = await mmr.diversify_rankings(docs, top_k=3)
        
        assert len(result) == 3
        # First doc should be highest scored
        assert result[0].final_score == 0.9
        # Should prefer diverse content over similar
        assert "database" in result[2].content
```

### 2. Integration Tests

#### Repository Integration Tests
```python
# tests/integration/test_document_repository.py
import pytest
from uuid import uuid4
from sqlalchemy.ext.asyncio import AsyncSession
from src.domain.entities.document import Document
from src.infrastructure.persistence.postgresql.repositories import (
    PostgreSQLDocumentRepository
)

@pytest.mark.asyncio
class TestDocumentRepositoryIntegration:
    """Test PostgreSQL document repository"""
    
    async def test_save_and_retrieve_document(self, db_session: AsyncSession):
        """Test saving and retrieving a document"""
        repo = PostgreSQLDocumentRepository(db_session)
        
        # Create document
        doc = Document(
            id=uuid4(),
            content="Integration test content",
            source="test.pdf",
            file_type="pdf",
            metadata={"test": True}
        )
        
        # Save
        saved_doc = await repo.save(doc)
        await db_session.commit()
        
        # Retrieve
        retrieved = await repo.get_by_id(doc.id)
        
        assert retrieved is not None
        assert retrieved.id == doc.id
        assert retrieved.content == doc.content
        assert retrieved.metadata["test"] is True
    
    async def test_update_document(self, db_session: AsyncSession):
        """Test updating document properties"""
        repo = PostgreSQLDocumentRepository(db_session)
        
        # Create and save
        doc = Document(
            content="Original content",
            source="test.pdf",
            file_type="pdf"
        )
        await repo.save(doc)
        await db_session.commit()
        
        # Update
        doc.content = "Updated content"
        doc.mark_as_processed()
        await repo.update(doc)
        await db_session.commit()
        
        # Verify
        updated = await repo.get_by_id(doc.id)
        assert updated.content == "Updated content"
        assert updated.processing_status == "completed"
    
    async def test_search_by_metadata(self, db_session: AsyncSession):
        """Test metadata-based search"""
        repo = PostgreSQLDocumentRepository(db_session)
        
        # Create documents with different metadata
        docs = [
            Document(
                content=f"Doc {i}",
                source=f"doc{i}.pdf",
                file_type="pdf",
                metadata={"category": "technical" if i % 2 == 0 else "general"}
            )
            for i in range(5)
        ]
        
        for doc in docs:
            await repo.save(doc)
        await db_session.commit()
        
        # Search by metadata
        technical_docs = await repo.search_by_metadata({"category": "technical"})
        
        assert len(technical_docs) == 3
        assert all(doc.metadata["category"] == "technical" for doc in technical_docs)
```

#### Vector Store Integration Tests
```python
# tests/integration/test_vector_store.py
import pytest
import numpy as np
from src.infrastructure.persistence.vector_stores import PgVectorStore
from src.domain.value_objects.embedding import Embedding

@pytest.mark.asyncio
class TestPgVectorStoreIntegration:
    """Test pgvector integration"""
    
    async def test_vector_similarity_search(self, vector_store: PgVectorStore):
        """Test vector similarity search"""
        # Create test embeddings
        embeddings = [
            Embedding.create([1.0, 0.0, 0.0], "test-model"),
            Embedding.create([0.0, 1.0, 0.0], "test-model"),
            Embedding.create([0.0, 0.0, 1.0], "test-model"),
            Embedding.create([0.9, 0.1, 0.0], "test-model"),  # Similar to first
        ]
        
        # Store embeddings
        for i, emb in enumerate(embeddings):
            await vector_store.upsert(
                chunk_id=uuid4(),
                embedding=emb,
                metadata={"index": i}
            )
        
        # Search for similar to first embedding
        query = Embedding.create([1.0, 0.0, 0.0], "test-model")
        results = await vector_store.search(query, k=2)
        
        assert len(results) == 2
        # First result should be exact match
        assert results[0].score > 0.99
        # Second should be the similar one
        assert results[1].metadata["index"] == 3
        assert results[1].score > 0.85
    
    async def test_batch_operations(self, vector_store: PgVectorStore):
        """Test batch insert performance"""
        # Create batch of embeddings
        batch_items = [
            (uuid4(), Embedding.create(np.random.rand(128).tolist(), "model"), {"batch": i})
            for i in range(100)
        ]
        
        # Batch insert
        count = await vector_store.upsert_batch(batch_items)
        
        assert count == 100
        
        # Verify retrieval
        total = await vector_store.count()
        assert total >= 100
```

#### External Service Integration Tests
```python
# tests/integration/test_openai_service.py
import pytest
from unittest.mock import patch, AsyncMock
from src.infrastructure.external.openai import OpenAIEmbeddingService

@pytest.mark.asyncio
class TestOpenAIIntegration:
    """Test OpenAI service integration"""
    
    @patch('openai.AsyncOpenAI')
    async def test_embedding_generation(self, mock_openai):
        """Test embedding generation with mocked OpenAI"""
        # Mock response
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client
        mock_client.embeddings.create.return_value = AsyncMock(
            data=[AsyncMock(embedding=[0.1] * 1536)]
        )
        
        service = OpenAIEmbeddingService("test-key")
        embedding = await service.embed_text("Test text")
        
        assert len(embedding.vector) == 1536
        assert embedding.model == "text-embedding-3-large"
        mock_client.embeddings.create.assert_called_once()
    
    @pytest.mark.integration
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), 
                        reason="Requires OPENAI_API_KEY")
    async def test_real_openai_embedding(self):
        """Test with real OpenAI API (requires API key)"""
        service = OpenAIEmbeddingService(os.getenv("OPENAI_API_KEY"))
        
        embedding = await service.embed_text("Machine learning is fascinating")
        
        assert len(embedding.vector) == 1536
        assert all(-1 <= x <= 1 for x in embedding.vector)
```

### 3. End-to-End Tests

#### API E2E Tests
```python
# tests/e2e/test_search_flow.py
import pytest
from httpx import AsyncClient
from tests.factories import DocumentFactory, UserFactory

@pytest.mark.e2e
@pytest.mark.asyncio
class TestSearchE2E:
    """End-to-end search flow tests"""
    
    async def test_complete_search_flow(self, 
                                       api_client: AsyncClient,
                                       test_user_token: str):
        """Test complete document upload and search flow"""
        headers = {"Authorization": f"Bearer {test_user_token}"}
        
        # 1. Upload document
        with open("tests/fixtures/sample.pdf", "rb") as f:
            response = await api_client.post(
                "/api/v1/documents",
                files={"file": ("sample.pdf", f, "application/pdf")},
                data={"metadata": '{"category": "test"}'},
                headers=headers
            )
        
        assert response.status_code == 201
        doc_id = response.json()["id"]
        
        # 2. Wait for processing (with timeout)
        for _ in range(30):  # 30 second timeout
            status_response = await api_client.get(
                f"/api/v1/documents/{doc_id}/status",
                headers=headers
            )
            if status_response.json()["status"] == "completed":
                break
            await asyncio.sleep(1)
        else:
            pytest.fail("Document processing timeout")
        
        # 3. Search for content
        search_response = await api_client.post(
            "/api/v1/search",
            json={
                "query": "machine learning",
                "filters": {"category": "test"}
            },
            headers=headers
        )
        
        assert search_response.status_code == 200
        results = search_response.json()
        assert len(results["results"]) > 0
        assert results["results"][0]["document_id"] == doc_id
    
    async def test_multimodal_search_flow(self,
                                         api_client: AsyncClient,
                                         test_user_token: str):
        """Test multimodal search with images"""
        headers = {"Authorization": f"Bearer {test_user_token}"}
        
        # Upload document with images
        with open("tests/fixtures/document_with_images.pdf", "rb") as f:
            response = await api_client.post(
                "/api/v1/documents",
                files={"file": ("doc.pdf", f, "application/pdf")},
                data={
                    "metadata": '{"type": "multimodal"}',
                    "processing_options": '{"extract_images": true}'
                },
                headers=headers
            )
        
        doc_id = response.json()["id"]
        
        # Wait for processing
        await self._wait_for_processing(api_client, doc_id, headers)
        
        # Search with text + image
        with open("tests/fixtures/query_image.png", "rb") as img:
            search_response = await api_client.post(
                "/api/v1/search/multimodal",
                files={"image_query": ("query.png", img, "image/png")},
                data={
                    "text_query": "similar diagrams",
                    "search_mode": "hybrid"
                },
                headers=headers
            )
        
        results = search_response.json()
        assert any(r["type"] == "image" for r in results["results"])
```

#### WebSocket E2E Tests
```python
# tests/e2e/test_websocket_flow.py
import pytest
import websockets
import json

@pytest.mark.e2e
@pytest.mark.asyncio
class TestWebSocketE2E:
    """Test WebSocket real-time features"""
    
    async def test_streaming_search(self, ws_url: str, test_user_token: str):
        """Test streaming search responses"""
        async with websockets.connect(ws_url) as ws:
            # Authenticate
            await ws.send(json.dumps({
                "type": "auth",
                "token": f"Bearer {test_user_token}"
            }))
            
            auth_response = await ws.recv()
            assert json.loads(auth_response)["type"] == "auth_success"
            
            # Send search request
            await ws.send(json.dumps({
                "type": "search_stream",
                "data": {
                    "query": "explain machine learning",
                    "stream_response": True
                }
            }))
            
            # Collect streamed chunks
            chunks = []
            complete = False
            
            while not complete:
                message = json.loads(await ws.recv())
                
                if message["type"] == "response_chunk":
                    chunks.append(message["data"]["text"])
                elif message["type"] == "search_complete":
                    complete = True
                    final_result = message["data"]
            
            # Verify complete response
            full_response = "".join(chunks)
            assert len(full_response) > 100
            assert final_result["confidence_score"] > 0.7
```

### 4. Performance Tests

#### Load Testing
```python
# tests/performance/test_load.py
import asyncio
import time
from locust import HttpUser, task, between
import statistics

class RAGSystemUser(HttpUser):
    """Simulated user for load testing"""
    wait_time = between(1, 3)
    
    def on_start(self):
        """Login and get token"""
        response = self.client.post("/api/v1/auth/login", json={
            "username": "test_user",
            "password": "test_password"
        })
        self.token = response.json()["access_token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}
    
    @task(weight=70)
    def search_documents(self):
        """Most common operation - searching"""
        queries = [
            "machine learning basics",
            "neural network architecture",
            "data preprocessing techniques",
            "model evaluation metrics"
        ]
        
        response = self.client.post(
            "/api/v1/search",
            json={
                "query": random.choice(queries),
                "max_results": 10
            },
            headers=self.headers
        )
        
        assert response.status_code == 200
    
    @task(weight=20)
    def upload_document(self):
        """Less common - document upload"""
        with open("tests/fixtures/small_doc.pdf", "rb") as f:
            response = self.client.post(
                "/api/v1/documents",
                files={"file": ("doc.pdf", f, "application/pdf")},
                headers=self.headers
            )
        
        assert response.status_code in [201, 202]
    
    @task(weight=10)
    def check_document_status(self):
        """Check processing status"""
        # Assume we have some document IDs
        doc_id = "550e8400-e29b-41d4-a716-446655440000"
        response = self.client.get(
            f"/api/v1/documents/{doc_id}/status",
            headers=self.headers
        )
```

#### Benchmark Tests
```python
# tests/performance/test_benchmarks.py
import pytest
import asyncio
import time
from statistics import mean, stdev

@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmarks for critical operations"""
    
    @pytest.mark.asyncio
    async def test_retrieval_latency(self, retrieval_service, benchmark_queries):
        """Benchmark retrieval latency"""
        latencies = []
        
        for query in benchmark_queries:
            start = time.perf_counter()
            await retrieval_service.retrieve(query)
            latency = (time.perf_counter() - start) * 1000  # ms
            latencies.append(latency)
        
        print(f"\nRetrieval Latency Benchmarks:")
        print(f"Mean: {mean(latencies):.2f}ms")
        print(f"StdDev: {stdev(latencies):.2f}ms")
        print(f"Min: {min(latencies):.2f}ms")
        print(f"Max: {max(latencies):.2f}ms")
        print(f"P95: {sorted(latencies)[int(len(latencies)*0.95)]:.2f}ms")
        
        # Assert performance requirements
        assert mean(latencies) < 200  # Mean under 200ms
        assert sorted(latencies)[int(len(latencies)*0.95)] < 500  # P95 under 500ms
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, api_client):
        """Test system under concurrent load"""
        async def single_search():
            start = time.perf_counter()
            response = await api_client.post("/api/v1/search", json={
                "query": "test query",
                "max_results": 10
            })
            return time.perf_counter() - start, response.status_code
        
        # Run concurrent searches
        tasks = [single_search() for _ in range(100)]
        results = await asyncio.gather(*tasks)
        
        latencies = [r[0] for r in results]
        status_codes = [r[1] for r in results]
        
        # All should succeed
        assert all(code == 200 for code in status_codes)
        
        # Performance should degrade gracefully
        assert mean(latencies) < 1.0  # Under 1 second average
```

## 🧩 Test Utilities

### Test Fixtures
```python
# tests/conftest.py
import pytest
import asyncio
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from httpx import AsyncClient

from src.presentation.api.main import app
from src.shared.config.database import Base
from src.shared.config.settings import get_settings

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def test_db_engine():
    """Create test database engine"""
    engine = create_async_engine(
        "postgresql+asyncpg://postgres:postgres@localhost/rag_test",
        echo=False
    )
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()

@pytest.fixture
async def db_session(test_db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Provide database session for tests"""
    async with AsyncSession(test_db_engine) as session:
        yield session
        await session.rollback()

@pytest.fixture
async def api_client() -> AsyncGenerator[AsyncClient, None]:
    """Provide API client for tests"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.fixture
def mock_embedding_service(mocker):
    """Mock embedding service for tests"""
    mock = mocker.patch('src.infrastructure.external.openai.OpenAIEmbeddingService')
    mock.embed_text.return_value = Embedding.create([0.1] * 1536, "test-model")
    return mock
```

### Test Factories
```python
# tests/factories.py
import factory
from factory import fuzzy
from datetime import datetime, timedelta
from src.domain.entities.document import Document

class DocumentFactory(factory.Factory):
    """Factory for creating test documents"""
    class Meta:
        model = Document
    
    id = factory.Faker('uuid4')
    content = factory.Faker('text', max_nb_chars=1000)
    source = factory.Sequence(lambda n: f"test_doc_{n}.pdf")
    file_type = fuzzy.FuzzyChoice(['pdf', 'docx', 'txt'])
    metadata = factory.Dict({
        'category': fuzzy.FuzzyChoice(['technical', 'general', 'research']),
        'author': factory.Faker('name'),
        'created_date': fuzzy.FuzzyDateTime(
            datetime.now() - timedelta(days=365),
            datetime.now()
        )
    })
    processing_status = 'completed'
    
    @factory.post_generation
    def chunks(self, create, extracted, **kwargs):
        """Generate chunks after document creation"""
        if not create:
            return
        
        if extracted:
            self.chunk_ids = [chunk.id for chunk in extracted]
        else:
            # Create default chunks
            chunks = ChunkFactory.create_batch(5, document_id=self.id)
            self.chunk_ids = [chunk.id for chunk in chunks]
```

### Test Helpers
```python
# tests/helpers.py
import asyncio
from typing import List
from httpx import AsyncClient

async def wait_for_condition(condition_func, timeout=30, interval=1):
    """Wait for a condition to become true"""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if await condition_func():
            return True
        await asyncio.sleep(interval)
    
    return False

async def upload_test_documents(api_client: AsyncClient, 
                              count: int = 10) -> List[str]:
    """Helper to upload multiple test documents"""
    document_ids = []
    
    for i in range(count):
        with open(f"tests/fixtures/sample_{i % 3}.pdf", "rb") as f:
            response = await api_client.post(
                "/api/v1/documents",
                files={"file": (f"doc_{i}.pdf", f, "application/pdf")},
                data={"metadata": f'{{"index": {i}}}'}
            )
            document_ids.append(response.json()["id"])
    
    return document_ids

def assert_valid_embedding(embedding):
    """Assert embedding has valid structure"""
    assert hasattr(embedding, 'vector')
    assert hasattr(embedding, 'model')
    assert len(embedding.vector) > 0
    assert all(isinstance(x, float) for x in embedding.vector)
```

## 📊 Test Coverage Requirements

### Coverage Targets
- **Overall**: 90%+
- **Domain Layer**: 95%+
- **Application Layer**: 90%+
- **Infrastructure Layer**: 85%+
- **Presentation Layer**: 80%+

### Coverage Configuration
```ini
# .coveragerc
[run]
source = src
omit = 
    */tests/*
    */migrations/*
    */__init__.py
    */config/*

[report]
precision = 2
show_missing = true
skip_covered = false

[html]
directory = htmlcov

[xml]
output = coverage.xml
```

## 🔄 Continuous Testing

### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: unit-tests
        name: Run unit tests
        entry: poetry run pytest tests/unit -x
        language: system
        pass_filenames: false
        always_run: true
```

### CI Pipeline Tests
```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Run tests
      run: |
        poetry run pytest tests/ \
          --cov=src \
          --cov-report=term \
          --cov-report=xml \
          --cov-report=html \
          --junitxml=test-results.xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## 🐛 Test Debugging

### Debug Configuration
```python
# pytest.ini
[tool:pytest]
minversion = 6.0
addopts = 
    -ra 
    -q 
    --strict-markers
    --tb=short
    --maxfail=1
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Markers
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    slow: Slow tests
    benchmark: Performance benchmarks

# Logging
log_cli = true
log_cli_level = INFO
```

### Debug Helpers
```python
# tests/debug_helpers.py
import pdb
import logging

def debug_on_failure(func):
    """Decorator to drop into debugger on test failure"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            pdb.post_mortem()
            raise
    return wrapper

# Usage
@debug_on_failure
def test_complex_logic():
    # Test code here
    pass
```