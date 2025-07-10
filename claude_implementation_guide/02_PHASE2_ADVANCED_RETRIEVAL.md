# Phase 2: Advanced Retrieval Strategies Implementation Guide

## 📋 Objetivo de la Fase

Implementar estrategias avanzadas de retrieval incluyendo búsqueda híbrida, reranking multi-etapa, GraphRAG, y un sistema completo de evaluación para optimizar la calidad de recuperación.

## 🎯 Entregables Clave

1. Búsqueda híbrida (vector + keyword con BM25)
2. Pipeline de reranking multi-etapa (ColBERT + Cross-encoder)
3. GraphRAG con Neo4j para multi-hop reasoning
4. RAG-Fusion y query expansion
5. Framework de evaluación RAGAS
6. Sistema de A/B testing

## 📅 Timeline Detallado

### Semana 7-8: Hybrid & Multi-stage Retrieval

#### Día 1-3: BM25 Implementation

```python
# src/infrastructure/persistence/vector_stores/bm25_store.py
from rank_bm25 import BM25Okapi
import pickle

class BM25Store:
    def __init__(self):
        self.index = None
        self.document_ids = []
        self.tokenized_docs = []
    
    async def build_index(self, documents: List[DocumentChunk]):
        # Tokenizar documentos
        self.tokenized_docs = [self._tokenize(doc.content) for doc in documents]
        self.document_ids = [doc.id for doc in documents]
        
        # Construir índice BM25
        self.index = BM25Okapi(self.tokenized_docs)
        
        # Persistir índice
        await self._save_index()
    
    async def search(self, query: str, k: int = 10) -> List[tuple[UUID, float]]:
        tokenized_query = self._tokenize(query)
        scores = self.index.get_scores(tokenized_query)
        
        # Top-k resultados
        top_indices = np.argsort(scores)[-k:][::-1]
        
        results = [
            (self.document_ids[i], float(scores[i]))
            for i in top_indices
        ]
        
        return results
    
    def _tokenize(self, text: str) -> List[str]:
        # Tokenización simple, mejorar con spaCy
        return text.lower().split()
```

#### Día 4-6: Hybrid Search Implementation

```python
# src/infrastructure/persistence/vector_stores/hybrid_vector_store.py
class HybridVectorStore(HybridVectorStoreRepository):
    def __init__(self,
                 vector_store: VectorStoreRepository,
                 bm25_store: BM25Store,
                 rrf_k: int = 60):
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.rrf = ReciprocalRankFusion(k=rrf_k)
    
    async def hybrid_search(self,
                          query_embedding: Embedding,
                          query_text: str,
                          k: int = 10,
                          alpha: float = 0.5) -> List[RetrievedDocument]:
        # 1. Vector search
        vector_results = await self.vector_store.search(
            query_embedding, k=k*2
        )
        
        # 2. Keyword search (BM25)
        keyword_results = await self.bm25_store.search(
            query_text, k=k*2
        )
        
        # 3. Fusion con RRF o weighted combination
        if alpha == 0.5:
            # Use RRF
            fused_results = self.rrf.fuse_rankings([
                vector_results,
                keyword_results
            ])
        else:
            # Weighted combination
            fused_results = self._weighted_fusion(
                vector_results, keyword_results, alpha
            )
        
        return fused_results[:k]
```

#### Día 7-9: ColBERT Integration

```python
# src/infrastructure/external/colbert/colbert_reranker.py
from colbert.infra import Run, RunConfig
from colbert.modeling.checkpoint import Checkpoint

class ColBERTReranker:
    def __init__(self, model_path: str = "colbert-ir/colbertv2.0"):
        self.config = RunConfig(experiment="rag_reranking")
        self.checkpoint = Checkpoint(model_path, colbert_config=self.config)
    
    async def rerank(self,
                    query: str,
                    documents: List[RetrievedDocument],
                    k: int = 10) -> List[RetrievedDocument]:
        # Preparar textos para ColBERT
        doc_texts = [doc.content for doc in documents]
        
        # Late interaction scoring
        scores = await self._compute_colbert_scores(query, doc_texts)
        
        # Reordenar documentos
        sorted_indices = np.argsort(scores)[::-1][:k]
        
        reranked_docs = []
        for idx in sorted_indices:
            doc = documents[idx]
            doc.apply_rerank_score(float(scores[idx]))
            reranked_docs.append(doc)
        
        return reranked_docs
    
    async def _compute_colbert_scores(self, 
                                    query: str, 
                                    documents: List[str]) -> np.ndarray:
        # Encoding con ColBERT
        Q = self.checkpoint.queryFromText([query])
        D = self.checkpoint.docFromText(documents)
        
        # Late interaction
        scores = self.checkpoint.score(Q, D)
        
        return scores.cpu().numpy()
```

#### Día 10-12: Multi-stage Reranking Pipeline

```python
# src/application/services/multi_stage_reranking_service.py
class MultiStageRerankingService:
    def __init__(self,
                 colbert_reranker: ColBERTReranker,
                 cross_encoder: CrossEncoderReranker,
                 llm_reranker: Optional[LLMReranker] = None):
        self.stages = [
            (colbert_reranker, 50),    # Stage 1: ColBERT top-50
            (cross_encoder, 10),        # Stage 2: Cross-encoder top-10
        ]
        if llm_reranker:
            self.stages.append((llm_reranker, 5))  # Stage 3: LLM top-5
    
    async def rerank_pipeline(self,
                            query: str,
                            documents: List[RetrievedDocument]) -> List[RetrievedDocument]:
        current_docs = documents
        
        for reranker, top_k in self.stages:
            # Aplicar reranking
            current_docs = await reranker.rerank(
                query, current_docs, k=top_k
            )
            
            # Log métricas de cada etapa
            logger.info(f"Stage {reranker.__class__.__name__}: {len(current_docs)} docs")
        
        return current_docs
```

### Semana 9-10: GraphRAG & Knowledge Graphs

#### Día 13-15: Neo4j Integration

```python
# src/infrastructure/external/neo4j/graph_store.py
from neo4j import AsyncGraphDatabase

class Neo4jGraphStore:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    
    async def create_node(self, chunk_id: UUID, content: str, metadata: dict):
        async with self.driver.session() as session:
            await session.run(
                """
                CREATE (n:Chunk {
                    id: $chunk_id,
                    content: $content,
                    metadata: $metadata
                })
                """,
                chunk_id=str(chunk_id),
                content=content,
                metadata=metadata
            )
    
    async def create_edge(self, 
                         from_id: UUID, 
                         to_id: UUID, 
                         edge_type: str,
                         properties: dict = None):
        async with self.driver.session() as session:
            await session.run(
                f"""
                MATCH (a:Chunk {{id: $from_id}})
                MATCH (b:Chunk {{id: $to_id}})
                CREATE (a)-[r:{edge_type} $properties]->(b)
                """,
                from_id=str(from_id),
                to_id=str(to_id),
                properties=properties or {}
            )
```

#### Día 16-18: Entity Extraction & Graph Construction

```python
# src/application/services/knowledge_graph_service.py
import spacy
from typing import List, Tuple

class KnowledgeGraphService:
    def __init__(self, graph_store: Neo4jGraphStore):
        self.graph_store = graph_store
        self.nlp = spacy.load("en_core_web_lg")
    
    async def build_graph_from_documents(self, documents: List[Document]):
        for doc in documents:
            # 1. Extraer entidades
            entities = self._extract_entities(doc.content)
            
            # 2. Crear nodos para chunks
            for chunk in doc.chunks:
                await self.graph_store.create_node(
                    chunk.id, chunk.content, chunk.metadata
                )
            
            # 3. Crear relaciones entre chunks
            await self._create_chunk_relations(doc.chunks)
            
            # 4. Crear relaciones basadas en entidades
            await self._create_entity_relations(doc.chunks, entities)
    
    def _extract_entities(self, text: str) -> List[Tuple[str, str]]:
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities
    
    async def _create_chunk_relations(self, chunks: List[DocumentChunk]):
        # Relaciones secuenciales
        for i in range(len(chunks) - 1):
            await self.graph_store.create_edge(
                chunks[i].id,
                chunks[i + 1].id,
                "FOLLOWS",
                {"weight": 1.0}
            )
```

#### Día 19-20: Graph-Enhanced Retrieval

```python
# src/infrastructure/persistence/vector_stores/graph_vector_store.py
class GraphEnhancedVectorStore(GraphVectorStoreRepository):
    def __init__(self, 
                 vector_store: VectorStoreRepository,
                 graph_store: Neo4jGraphStore):
        self.vector_store = vector_store
        self.graph_store = graph_store
    
    async def graph_search(self,
                         query_embedding: Embedding,
                         k: int = 10,
                         max_hops: int = 2) -> List[RetrievedDocument]:
        # 1. Initial vector search
        initial_results = await self.vector_store.search(
            query_embedding, k=k
        )
        
        # 2. Expand with graph traversal
        expanded_results = []
        for doc in initial_results[:5]:  # Top 5 para expansión
            neighbors = await self._get_multi_hop_neighbors(
                doc.chunk_id, max_hops
            )
            expanded_results.extend(neighbors)
        
        # 3. Compute PageRank scores
        pagerank_scores = await self.compute_pagerank()
        
        # 4. Combine scores
        final_results = self._combine_scores(
            initial_results, expanded_results, pagerank_scores
        )
        
        return final_results[:k]
    
    async def _get_multi_hop_neighbors(self, 
                                     chunk_id: UUID, 
                                     max_hops: int) -> List[UUID]:
        query = f"""
        MATCH (start:Chunk {{id: $chunk_id}})
        MATCH path = (start)-[*1..{max_hops}]-(neighbor:Chunk)
        RETURN DISTINCT neighbor.id as id
        """
        
        async with self.graph_store.driver.session() as session:
            result = await session.run(query, chunk_id=str(chunk_id))
            return [record["id"] async for record in result]
```

### Semana 11-12: RAG Orchestration & Evaluation

#### Día 21-23: Query Complexity Analyzer

```python
# src/application/services/query_analyzer_service.py
class QueryAnalyzerService:
    def __init__(self, 
                 classifier_model: str = "cross-encoder/query-complexity"):
        self.classifier = self._load_classifier(classifier_model)
        self.intent_detector = self._load_intent_detector()
    
    async def analyze_query(self, query: Query) -> EnhancedQuery:
        # 1. Detectar complejidad
        complexity = await self._detect_complexity(query.text)
        query.set_complexity(complexity)
        
        # 2. Detectar intent
        intent = await self._detect_intent(query.text)
        query.set_intent(intent)
        
        # 3. Extraer entidades
        entities = await self._extract_entities(query.text)
        
        # 4. Generar expansiones
        expansions = await self._generate_expansions(query.text)
        
        # 5. Crear HyDE (Hypothetical Document Embeddings)
        hyde_answer = await self._generate_hypothetical_answer(query.text)
        
        enhanced = EnhancedQuery(
            original_query=query,
            expanded_queries=expansions,
            hypothetical_answer=hyde_answer,
            entities=entities
        )
        
        return enhanced
    
    async def _detect_complexity(self, text: str) -> QueryComplexity:
        # Usar modelo o heurísticas
        features = self._extract_features(text)
        
        if features['num_entities'] > 3 or features['has_comparison']:
            return QueryComplexity.COMPLEX
        elif features['is_factual'] and features['num_tokens'] < 10:
            return QueryComplexity.SIMPLE
        else:
            return QueryComplexity.MODERATE
```

#### Día 24-26: RAG-Fusion Implementation

```python
# src/application/services/rag_fusion_service.py
class RAGFusionService:
    def __init__(self,
                 query_generator: QueryGeneratorService,
                 retriever: RetrievalStrategyService,
                 fusion: ReciprocalRankFusion):
        self.query_generator = query_generator
        self.retriever = retriever
        self.fusion = fusion
    
    async def retrieve_with_fusion(self,
                                 original_query: Query,
                                 num_queries: int = 5) -> RetrievalResult:
        # 1. Generar múltiples queries
        generated_queries = await self.query_generator.generate_queries(
            original_query.text, num_queries
        )
        
        # 2. Retrieve para cada query
        all_retrievals = []
        for query_text in [original_query.text] + generated_queries:
            query_obj = Query(text=query_text)
            results = await self.retriever.retrieve(query_obj)
            all_retrievals.append(results.documents)
        
        # 3. Fusion de resultados
        fused_documents = self.fusion.fuse_rankings(all_retrievals)
        
        # 4. Crear resultado final
        return RetrievalResult(
            query_id=original_query.id,
            documents=fused_documents,
            retrieval_strategy="rag_fusion",
            metadata={"num_queries": len(all_retrievals)}
        )
```

#### Día 27-30: RAGAS Evaluation Framework

```python
# src/application/services/evaluation/ragas_evaluator.py
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

class RAGASEvaluator:
    def __init__(self, llm_model: str = "gpt-4"):
        self.llm_model = llm_model
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]
    
    async def evaluate_rag_pipeline(self,
                                  test_dataset: List[EvaluationSample]) -> Dict[str, float]:
        # Preparar datos para RAGAS
        questions = [sample.question for sample in test_dataset]
        answers = [sample.generated_answer for sample in test_dataset]
        contexts = [sample.retrieved_contexts for sample in test_dataset]
        ground_truths = [sample.ground_truth for sample in test_dataset]
        
        # Evaluar con RAGAS
        results = evaluate(
            questions=questions,
            answers=answers,
            contexts=contexts,
            ground_truths=ground_truths,
            metrics=self.metrics
        )
        
        return results
    
    async def evaluate_retrieval_strategies(self,
                                          strategies: List[str],
                                          test_queries: List[Query]) -> Dict[str, Dict[str, float]]:
        strategy_results = {}
        
        for strategy in strategies:
            metrics = await self._evaluate_strategy(strategy, test_queries)
            strategy_results[strategy] = metrics
        
        return strategy_results
```

## 🔧 Configuración Avanzada

### 1. Neo4j Setup

```bash
# docker-compose.yml addition
neo4j:
  image: neo4j:5.0
  environment:
    NEO4J_AUTH: neo4j/password
    NEO4J_PLUGINS: '["apoc", "graph-data-science"]'
  ports:
    - "7474:7474"
    - "7687:7687"
  volumes:
    - neo4j_data:/data
```

### 2. ColBERT Configuration

```python
# config/colbert_config.py
COLBERT_CONFIG = {
    "model": "colbert-ir/colbertv2.0",
    "index_path": "./indexes/colbert",
    "doc_maxlen": 300,
    "query_maxlen": 32,
    "nway": 64,
    "checkpoint": "downloads/colbertv2.0"
}
```

### 3. Evaluation Datasets

```python
# scripts/create_evaluation_dataset.py
def create_evaluation_dataset():
    dataset = [
        {
            "question": "What is machine learning?",
            "ground_truth": "Machine learning is a subset of AI...",
            "relevant_docs": ["doc1", "doc2", "doc3"]
        },
        # Más ejemplos...
    ]
    
    # Guardar como JSON para evaluación
    with open("evaluation_dataset.json", "w") as f:
        json.dump(dataset, f)
```

## 📊 Métricas de Éxito Phase 2

### Funcionalidad
- [ ] Búsqueda híbrida funcionando (vector + BM25)
- [ ] Reranking con ColBERT operacional
- [ ] GraphRAG con Neo4j integrado
- [ ] RAG-Fusion implementado
- [ ] Framework de evaluación RAGAS

### Calidad de Retrieval
- [ ] Precision@10 > 0.8
- [ ] NDCG@10 > 0.75
- [ ] MRR > 0.85
- [ ] Reducción de latencia con ColBERT vs Cross-encoder

### Performance
- [ ] Búsqueda híbrida < 300ms
- [ ] Reranking < 200ms para 50 documentos
- [ ] Graph traversal < 100ms para 2 hops

## 🚨 Riesgos y Mitigaciones

### Riesgo 1: Complejidad de Neo4j
**Mitigación**: Comenzar con grafos simples, usar Neo4j Aura para hosting

### Riesgo 2: Latencia de reranking
**Mitigación**: Implementar caching y batch processing

### Riesgo 3: Calidad de entity extraction
**Mitigación**: Fine-tuning de modelos spaCy con datos específicos

## 📝 Checklist de Completación

- [ ] BM25 indexing implementado
- [ ] Búsqueda híbrida con RRF
- [ ] ColBERT reranker integrado
- [ ] Multi-stage reranking pipeline
- [ ] Neo4j configurado y conectado
- [ ] Entity extraction con spaCy
- [ ] Graph construction pipeline
- [ ] Graph-enhanced retrieval
- [ ] Query complexity analyzer
- [ ] RAG-Fusion funcionando
- [ ] RAGAS evaluation framework
- [ ] A/B testing infrastructure
- [ ] Benchmarks documentados