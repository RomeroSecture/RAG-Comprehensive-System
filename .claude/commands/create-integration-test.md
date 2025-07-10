---
allowed-tools: ["Write", "Read", "Edit", "Bash", "TodoWrite"]
description: "Crea tests de integración completos para un componente del sistema RAG"
---

Crea tests de integración para: $ARGUMENTS

## 🧪 Creación de Tests de Integración

### 1. **Análisis del Componente**
Identifica:
- Qué componente/feature se va a testear
- Dependencias externas (DB, Redis, APIs)
- Flujos críticos a validar
- Casos edge y manejo de errores

### 2. **Setup de Test Fixtures**

```python
# tests/integration/conftest.py
import pytest
import asyncio
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer
from testcontainers.compose import DockerCompose

from src.infrastructure.config import Settings
from src.infrastructure.database import get_session


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_containers():
    """Start test containers for integration tests."""
    with DockerCompose(".", compose_file_name="docker-compose.test.yml") as compose:
        # Wait for services to be ready
        compose.wait_for("postgres")
        compose.wait_for("redis")
        compose.wait_for("neo4j")
        
        yield compose


@pytest.fixture(scope="function")
async def test_db(test_containers) -> AsyncGenerator[AsyncSession, None]:
    """Provide test database session."""
    # Create test database
    engine = create_async_engine(
        "postgresql+asyncpg://test:test@localhost:5433/test_rag",
        echo=False
    )
    
    # Run migrations
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Provide session
    async with AsyncSession(engine) as session:
        yield session
    
    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()


@pytest.fixture
async def test_redis(test_containers):
    """Provide test Redis client."""
    import redis.asyncio as redis
    
    client = redis.Redis(
        host="localhost",
        port=6380,  # Test Redis port
        decode_responses=True
    )
    
    yield client
    
    await client.flushall()
    await client.close()


@pytest.fixture
async def test_vector_store(test_db):
    """Provide test vector store."""
    from src.infrastructure.vector_store import PgVectorStore
    
    store = PgVectorStore(test_db)
    await store.initialize()
    
    yield store
    
    await store.clear()
```

### 3. **Test de Flujo Completo RAG**

```python
# tests/integration/test_rag_pipeline.py
import pytest
from pathlib import Path
from uuid import uuid4

from src.application.use_cases.process_query import ProcessQueryUseCase
from src.domain.entities import Document, Query
from src.infrastructure.embeddings import OpenAIEmbeddingService
from src.infrastructure.llm import OpenAILLMService


@pytest.mark.integration
class TestRAGPipeline:
    """Integration tests for complete RAG pipeline."""
    
    @pytest.fixture
    async def setup_test_documents(self, test_vector_store):
        """Setup test documents in vector store."""
        documents = [
            Document(
                id=uuid4(),
                content="Python is a high-level programming language.",
                metadata={"source": "test1.txt", "type": "programming"}
            ),
            Document(
                id=uuid4(),
                content="Machine learning is a subset of artificial intelligence.",
                metadata={"source": "test2.txt", "type": "ai"}
            ),
            Document(
                id=uuid4(),
                content="FastAPI is a modern web framework for Python.",
                metadata={"source": "test3.txt", "type": "framework"}
            )
        ]
        
        # Generate embeddings and store
        embedding_service = OpenAIEmbeddingService()
        for doc in documents:
            embedding = await embedding_service.generate_embedding(doc.content)
            await test_vector_store.add_document(doc, embedding)
        
        return documents
    
    async def test_end_to_end_query_processing(
        self,
        setup_test_documents,
        test_vector_store,
        test_redis
    ):
        """Test complete query processing flow."""
        # Arrange
        query = Query(
            text="What is Python?",
            metadata={"user_id": "test_user"}
        )
        
        use_case = ProcessQueryUseCase(
            vector_store=test_vector_store,
            llm_service=OpenAILLMService(),
            cache=test_redis
        )
        
        # Act
        result = await use_case.execute(query)
        
        # Assert
        assert result is not None
        assert result.answer is not None
        assert len(result.sources) > 0
        assert "Python" in result.answer
        assert result.metadata["retrieval_time"] > 0
        assert result.metadata["generation_time"] > 0
    
    async def test_hybrid_search_integration(
        self,
        setup_test_documents,
        test_vector_store,
        test_db
    ):
        """Test hybrid search with keyword and vector search."""
        # Arrange
        from src.infrastructure.search import HybridSearchService
        
        search_service = HybridSearchService(
            vector_store=test_vector_store,
            keyword_store=test_db
        )
        
        # Act
        results = await search_service.search(
            query="Python programming",
            k=5,
            strategy="hybrid_rrf"
        )
        
        # Assert
        assert len(results) > 0
        assert results[0].relevance_score > 0
        assert "Python" in results[0].document.content
    
    async def test_document_ingestion_pipeline(
        self,
        test_vector_store,
        test_db,
        tmp_path
    ):
        """Test document ingestion from file to vector store."""
        # Arrange
        test_file = tmp_path / "test_document.txt"
        test_file.write_text("This is a test document about RAG systems.")
        
        from src.application.use_cases.ingest_document import IngestDocumentUseCase
        
        use_case = IngestDocumentUseCase(
            vector_store=test_vector_store,
            document_repo=test_db
        )
        
        # Act
        result = await use_case.execute(str(test_file))
        
        # Assert
        assert result.success is True
        assert result.document_id is not None
        assert result.chunks_created > 0
        
        # Verify in vector store
        search_results = await test_vector_store.search("RAG systems", k=1)
        assert len(search_results) == 1
```

### 4. **Test de Componentes Infrastructure**

```python
# tests/integration/test_vector_store.py
@pytest.mark.integration
class TestPgVectorStore:
    """Integration tests for PostgreSQL vector store."""
    
    async def test_add_and_search_documents(self, test_vector_store):
        """Test adding documents and searching."""
        # Arrange
        doc = Document(
            content="Integration testing is important",
            metadata={"test": True}
        )
        embedding = [0.1] * 1536  # Mock embedding
        
        # Act
        await test_vector_store.add_document(doc, embedding)
        results = await test_vector_store.search(
            query_embedding=[0.1] * 1536,
            k=1
        )
        
        # Assert
        assert len(results) == 1
        assert results[0].document.content == doc.content
    
    async def test_similarity_threshold(self, test_vector_store):
        """Test similarity threshold filtering."""
        # Add documents with different similarities
        # Test threshold filtering
        pass
    
    async def test_metadata_filtering(self, test_vector_store):
        """Test filtering by metadata."""
        # Add documents with different metadata
        # Test filtering
        pass
```

### 5. **Test de Resiliencia y Error Handling**

```python
# tests/integration/test_resilience.py
@pytest.mark.integration
class TestSystemResilience:
    """Test system resilience and error handling."""
    
    async def test_database_connection_failure(self, monkeypatch):
        """Test handling of database connection failures."""
        # Simulate connection failure
        # Verify graceful degradation
        pass
    
    async def test_llm_api_timeout(self, httpx_mock):
        """Test handling of LLM API timeouts."""
        # Mock timeout
        # Verify fallback behavior
        pass
    
    async def test_concurrent_requests(self, test_app_client):
        """Test system under concurrent load."""
        import asyncio
        
        async def make_request():
            return await test_app_client.post(
                "/api/v1/query",
                json={"text": "Test query"}
            )
        
        # Make 100 concurrent requests
        tasks = [make_request() for _ in range(100)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify success rate
        successful = sum(1 for r in responses if not isinstance(r, Exception))
        assert successful > 95  # 95% success rate
```

### 6. **Test de Performance**

```python
# tests/integration/test_performance.py
@pytest.mark.integration
@pytest.mark.performance
class TestPerformance:
    """Performance integration tests."""
    
    async def test_query_latency_sla(self, test_app_client, setup_test_documents):
        """Test that query latency meets SLA."""
        import time
        
        latencies = []
        for _ in range(10):
            start = time.time()
            response = await test_app_client.post(
                "/api/v1/query",
                json={"text": "What is Python?"}
            )
            latency = (time.time() - start) * 1000  # ms
            latencies.append(latency)
            
            assert response.status_code == 200
        
        # Check P95 latency
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        assert p95_latency < 200  # 200ms SLA
    
    async def test_bulk_ingestion_throughput(self, test_vector_store):
        """Test bulk document ingestion throughput."""
        # Test ingesting 1000 documents
        # Measure throughput
        pass
```

### 7. **Test Helpers y Utilities**

```python
# tests/integration/helpers.py
from typing import List
from src.domain.entities import Document

async def create_test_corpus(vector_store, num_documents: int = 100) -> List[Document]:
    """Create a test corpus of documents."""
    documents = []
    for i in range(num_documents):
        doc = Document(
            content=f"Test document {i} with content about topic {i % 10}",
            metadata={"index": i, "category": f"cat_{i % 5}"}
        )
        documents.append(doc)
        
        # Add to vector store
        embedding = await generate_test_embedding(doc.content)
        await vector_store.add_document(doc, embedding)
    
    return documents

async def generate_test_embedding(text: str) -> List[float]:
    """Generate deterministic test embedding."""
    # Simple hash-based embedding for testing
    import hashlib
    hash_hex = hashlib.sha256(text.encode()).hexdigest()
    # Convert to float vector
    return [int(hash_hex[i:i+2], 16) / 255.0 for i in range(0, 256, 2)][:1536]
```

### 8. **Configuración de CI/CD**

```yaml
# .github/workflows/integration-tests.yml
name: Integration Tests

on: [push, pull_request]

jobs:
  integration-tests:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: pgvector/pgvector:pg15
        env:
          POSTGRES_PASSWORD: test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install poetry
          poetry install
      
      - name: Run integration tests
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          poetry run pytest tests/integration -v --cov
```

### 9. **Ejecutar Tests**

```bash
# Ejecutar todos los tests de integración
!pytest tests/integration -v

# Ejecutar con marca específica
!pytest tests/integration -m "integration" -v

# Ejecutar con reporte de coverage
!pytest tests/integration --cov=src --cov-report=html

# Ejecutar tests de performance
!pytest tests/integration -m "performance" -v
```

## 📋 Checklist de Validación

- [ ] Tests cubren flujos principales end-to-end
- [ ] Manejo de errores está testeado
- [ ] Tests de performance validan SLAs
- [ ] Fixtures son reutilizables y eficientes
- [ ] Tests son independientes entre sí
- [ ] Cleanup se ejecuta correctamente
- [ ] CI/CD ejecuta tests automáticamente