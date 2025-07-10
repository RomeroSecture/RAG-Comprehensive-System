# Phase 1: Core Infrastructure & Basic RAG Implementation Guide

## 📋 Objetivo de la Fase

Establecer la infraestructura base del sistema RAG con funcionalidad básica de búsqueda semántica y generación, siguiendo Clean Architecture y mejores prácticas.

## 🎯 Entregables Clave

1. Infraestructura completa con PostgreSQL + pgvector
2. Pipeline de procesamiento de documentos
3. Sistema de embeddings con OpenAI
4. API REST funcional con FastAPI
5. Tests unitarios e integración (>70% coverage)

## 📅 Timeline Detallado

### Semana 1-2: Foundation Setup

#### Día 1-3: Completar Infrastructure Layer

```python
# 1. Completar repositorios PostgreSQL faltantes
src/infrastructure/persistence/postgresql/repositories/
├── embedding_repository_impl.py
├── vector_store_repository_impl.py
└── __init__.py

# 2. Implementar vector store con pgvector
class PgVectorStore(VectorStoreRepository):
    async def upsert(self, chunk_id: UUID, embedding: Embedding, metadata: Dict):
        # Usar pgvector para almacenar embeddings
        pass
    
    async def search(self, query_embedding: Embedding, k: int = 10):
        # Búsqueda por similitud coseno con pgvector
        pass
```

#### Día 4-5: Application Layer - Use Cases Básicos

```python
# Estructura de use cases
src/application/use_cases/
├── ingest_document.py
├── process_query.py
├── search_documents.py
└── __init__.py

# Ejemplo: IngestDocumentUseCase
class IngestDocumentUseCase:
    def __init__(self,
                 document_repo: DocumentRepository,
                 chunk_repo: DocumentChunkRepository,
                 chunking_service: ChunkingService,
                 embedding_service: EmbeddingService,
                 vector_store: VectorStoreRepository):
        self.document_repo = document_repo
        self.chunk_repo = chunk_repo
        self.chunking_service = chunking_service
        self.embedding_service = embedding_service
        self.vector_store = vector_store
    
    async def execute(self, file_path: str, metadata: Dict) -> Document:
        # 1. Parse documento
        # 2. Crear entidad Document
        # 3. Chunking
        # 4. Generar embeddings
        # 5. Almacenar en vector store
        # 6. Guardar en repositorios
        pass
```

#### Día 6-7: Presentation Layer - FastAPI Setup

```python
# src/presentation/api/main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    yield
    # Shutdown
    await close_db()

app = FastAPI(
    title="RAG Comprehensive System",
    version="0.1.0",
    lifespan=lifespan
)

# Routers
app.include_router(documents_router, prefix="/api/v1/documents")
app.include_router(search_router, prefix="/api/v1/search")
app.include_router(health_router, prefix="/api/v1/health")
```

#### Día 8-10: Docker Environment & Database Migrations

```yaml
# docker/Dockerfile.api
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev

COPY . .

CMD ["uvicorn", "src.presentation.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```sql
-- migrations/001_enable_pgvector.sql
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- Create vector column
ALTER TABLE vectors ADD COLUMN vector vector(1536);

-- Create indexes
CREATE INDEX ON vectors USING ivfflat (vector vector_cosine_ops) WITH (lists = 100);
```

### Semana 3-4: Document Processing Pipeline

#### Día 11-13: Multi-format Parsers

```python
# src/infrastructure/external/parsers/
├── base_parser.py
├── pdf_parser.py
├── docx_parser.py
├── text_parser.py
├── markdown_parser.py
└── parser_factory.py

# Ejemplo: PDFParser
class PDFParser(BaseParser):
    async def parse(self, file_path: str) -> ParsedDocument:
        # Usar PyPDF2 o pdfplumber
        # Extraer texto y metadata
        # Manejar imágenes con OCR si está habilitado
        pass
```

#### Día 14-16: Chunking Strategies

```python
# src/application/services/chunking_service.py
class ChunkingService:
    def __init__(self):
        self.strategies = {
            'fixed': FixedSizeChunker(),
            'semantic': SemanticChunker(),
            'recursive': RecursiveCharacterChunker(),
            'sentence': SentenceChunker()
        }
    
    async def chunk_document(self, 
                           content: str, 
                           strategy: str = 'recursive',
                           chunk_size: int = 1000,
                           overlap: int = 200) -> List[DocumentChunk]:
        chunker = self.strategies[strategy]
        return await chunker.chunk(content, chunk_size, overlap)
```

#### Día 17-20: Background Jobs con Celery

```python
# src/infrastructure/messaging/celery_app.py
from celery import Celery
from src.shared.config.settings import settings

celery_app = Celery(
    'rag_system',
    broker=settings.redis.url,
    backend=settings.redis.url
)

# src/infrastructure/messaging/tasks/document_tasks.py
@celery_app.task(bind=True, max_retries=3)
def process_document_task(self, document_id: str):
    try:
        # Procesar documento asincrónicamente
        pass
    except Exception as exc:
        # Retry con exponential backoff
        raise self.retry(exc=exc, countdown=60 * 2 ** self.request.retries)
```

### Semana 5-6: Basic RAG Implementation

#### Día 21-23: OpenAI Integration

```python
# src/infrastructure/external/openai/embedding_service.py
class OpenAIEmbeddingService(EmbeddingService):
    def __init__(self, api_key: str, model: str = "text-embedding-3-large"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
    
    async def embed_text(self, text: str) -> Embedding:
        response = await self.client.embeddings.create(
            model=self.model,
            input=text
        )
        vector = response.data[0].embedding
        return Embedding.create(vector=vector, model=self.model)
    
    async def embed_batch(self, texts: List[str]) -> List[Embedding]:
        # Batch processing con rate limiting
        pass
```

#### Día 24-26: Generation Pipeline

```python
# src/application/services/generation_service.py
class GenerationService:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
    
    async def generate_response(self,
                              query: str,
                              context: List[RetrievedDocument],
                              system_prompt: Optional[str] = None) -> GeneratedResponse:
        # 1. Construir prompt con contexto
        prompt = self._build_prompt(query, context, system_prompt)
        
        # 2. Generar respuesta
        response = await self.llm_service.generate(prompt)
        
        # 3. Extraer citas
        citations = self._extract_citations(response, context)
        
        return GeneratedResponse(
            text=response,
            citations=citations,
            confidence=0.85
        )
```

#### Día 27-30: Testing Suite

```python
# tests/unit/domain/test_document.py
import pytest
from src.domain.entities.document import Document

class TestDocument:
    def test_create_document_valid(self):
        doc = Document(
            content="Test content",
            source="test.pdf",
            file_type="pdf"
        )
        assert doc.content == "Test content"
        assert doc.processing_status == "pending"
    
    def test_create_document_empty_content(self):
        with pytest.raises(ValueError):
            Document(content="", source="test.pdf", file_type="pdf")

# tests/integration/test_document_ingestion.py
@pytest.mark.asyncio
async def test_ingest_pdf_document(test_db, test_files):
    use_case = IngestDocumentUseCase(...)
    
    document = await use_case.execute(
        file_path=test_files / "sample.pdf",
        metadata={"category": "test"}
    )
    
    assert document.processing_status == "completed"
    assert len(document.chunk_ids) > 0
```

## 🔧 Configuración y Setup

### 1. Configuración de pgvector

```bash
# Instalar pgvector en PostgreSQL
CREATE EXTENSION vector;

# Configurar índices
CREATE INDEX ON vectors USING ivfflat (vector vector_cosine_ops) WITH (lists = 100);
```

### 2. Variables de Entorno

```env
# .env.development
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/rag_system
REDIS_URL=redis://localhost:6379
OPENAI_API_KEY=your_key_here
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
```

### 3. Comandos Make

```makefile
# Comandos útiles para desarrollo
setup-dev:
	poetry install
	docker-compose up -d postgres redis
	poetry run alembic upgrade head

test-unit:
	poetry run pytest tests/unit -v --cov=src

test-integration:
	poetry run pytest tests/integration -v

run-dev:
	poetry run uvicorn src.presentation.api.main:app --reload
```

## 📊 Métricas de Éxito Phase 1

### Funcionalidad
- [ ] Ingesta de al menos 3 formatos de documento (PDF, DOCX, TXT)
- [ ] Búsqueda semántica funcional con pgvector
- [ ] API REST con endpoints CRUD básicos
- [ ] Procesamiento asíncrono de documentos

### Calidad
- [ ] Test coverage > 70%
- [ ] Sin errores críticos en linting
- [ ] Documentación de API con OpenAPI
- [ ] Logs estructurados funcionando

### Performance
- [ ] Ingesta de documento < 5 segundos
- [ ] Búsqueda < 500ms para 10 resultados
- [ ] Capacidad para 10,000 documentos

## 🚨 Riesgos y Mitigaciones

### Riesgo 1: Configuración de pgvector
**Mitigación**: Tener Dockerfile con pgvector pre-instalado

### Riesgo 2: Rate limits de OpenAI
**Mitigación**: Implementar retry logic y caching de embeddings

### Riesgo 3: Manejo de documentos grandes
**Mitigación**: Streaming y procesamiento por chunks

## 📝 Checklist de Completación

- [ ] Infrastructure layer completo con repositorios
- [ ] Application layer con use cases básicos
- [ ] Presentation layer con FastAPI configurado
- [ ] Docker environment funcionando
- [ ] Pipeline de ingesta multi-formato
- [ ] Integración con OpenAI embeddings
- [ ] Vector search con pgvector
- [ ] Tests unitarios e integración
- [ ] Documentación básica de API
- [ ] CI/CD básico con GitHub Actions