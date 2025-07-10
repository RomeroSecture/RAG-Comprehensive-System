---
allowed-tools: ["Bash", "Read", "Write", "Task", "TodoWrite"]
description: "Ejecuta benchmarks de performance completos del sistema RAG"
---

Ejecuta benchmarks exhaustivos de performance para el sistema RAG.

## 🚀 Benchmark Suite Completa

### 1. **Preparación del Entorno**
```bash
# Asegurar que los servicios están corriendo
!docker-compose ps

# Limpiar caché para resultados consistentes
!redis-cli FLUSHALL

# Crear dataset de prueba si no existe
!python scripts/create_benchmark_dataset.py
```

### 2. **Benchmarks de Ingesta de Documentos**

#### Test de Velocidad de Procesamiento
```python
# benchmark_ingestion.py
import time
import asyncio
from pathlib import Path

async def benchmark_document_ingestion():
    test_files = {
        "small_pdf": ("test_1page.pdf", 1),
        "medium_pdf": ("test_10pages.pdf", 10),
        "large_pdf": ("test_100pages.pdf", 100),
        "docx": ("test_document.docx", 5),
        "txt": ("test_large.txt", 1)
    }
    
    results = {}
    for file_type, (filename, pages) in test_files.items():
        start = time.time()
        # Procesar documento
        await ingest_document(Path(f"benchmarks/data/{filename}"))
        elapsed = time.time() - start
        
        results[file_type] = {
            "time": elapsed,
            "pages": pages,
            "pages_per_second": pages / elapsed
        }
    
    return results
```

### 3. **Benchmarks de Embeddings**

#### Generación de Embeddings
```python
# benchmark_embeddings.py
async def benchmark_embeddings():
    test_texts = [
        ("short", "This is a short text.", 1),
        ("medium", "Medium text " * 50, 1),
        ("long", "Long text " * 500, 1),
        ("batch_small", ["Text " + str(i) for i in range(10)], 10),
        ("batch_large", ["Text " + str(i) for i in range(100)], 100)
    ]
    
    results = {}
    for test_name, text, count in test_texts:
        start = time.time()
        embeddings = await generate_embeddings(text)
        elapsed = time.time() - start
        
        results[test_name] = {
            "time": elapsed,
            "count": count,
            "embeddings_per_second": count / elapsed,
            "avg_time_per_embedding": elapsed / count
        }
    
    return results
```

### 4. **Benchmarks de Retrieval**

#### Búsqueda Vectorial
```python
# benchmark_retrieval.py
async def benchmark_vector_search():
    query_complexities = {
        "simple": "What is machine learning?",
        "medium": "Explain the differences between supervised and unsupervised learning in detail",
        "complex": "How do transformer models handle long-range dependencies compared to RNNs and what are the computational trade-offs?"
    }
    
    k_values = [5, 10, 20, 50, 100]
    results = {}
    
    for complexity, query in query_complexities.items():
        results[complexity] = {}
        for k in k_values:
            times = []
            for _ in range(10):  # 10 runs para promedio
                start = time.time()
                docs = await vector_search(query, k=k)
                elapsed = time.time() - start
                times.append(elapsed)
            
            results[complexity][f"k_{k}"] = {
                "avg_time": sum(times) / len(times),
                "min_time": min(times),
                "max_time": max(times),
                "p95_time": sorted(times)[int(len(times) * 0.95)]
            }
    
    return results
```

#### Búsqueda Híbrida
```python
# benchmark_hybrid_search.py
async def benchmark_hybrid_search():
    strategies = ["vector_only", "keyword_only", "hybrid_rrf", "hybrid_weighted"]
    results = {}
    
    for strategy in strategies:
        times = []
        for _ in range(20):
            start = time.time()
            docs = await hybrid_search(
                query="machine learning applications",
                strategy=strategy,
                k=10
            )
            elapsed = time.time() - start
            times.append(elapsed)
        
        results[strategy] = {
            "avg_time": sum(times) / len(times),
            "p50_time": sorted(times)[len(times) // 2],
            "p95_time": sorted(times)[int(len(times) * 0.95)],
            "p99_time": sorted(times)[int(len(times) * 0.99)]
        }
    
    return results
```

### 5. **Benchmarks de Reranking**

```python
# benchmark_reranking.py
async def benchmark_reranking():
    rerankers = ["none", "cross_encoder", "colbert", "mmr"]
    doc_counts = [10, 20, 50, 100]
    
    results = {}
    for reranker in rerankers:
        results[reranker] = {}
        for count in doc_counts:
            # Obtener documentos candidatos
            candidates = await vector_search("test query", k=count)
            
            start = time.time()
            reranked = await rerank_documents(
                candidates, 
                "test query",
                method=reranker
            )
            elapsed = time.time() - start
            
            results[reranker][f"docs_{count}"] = {
                "time": elapsed,
                "docs_per_second": count / elapsed
            }
    
    return results
```

### 6. **Benchmarks End-to-End**

```python
# benchmark_e2e.py
async def benchmark_end_to_end():
    test_scenarios = [
        {
            "name": "simple_rag",
            "query": "What is Python?",
            "strategy": "basic",
            "expected_latency": 200  # ms
        },
        {
            "name": "advanced_rag",
            "query": "Compare Python and JavaScript for web development",
            "strategy": "hybrid_rerank",
            "expected_latency": 500
        },
        {
            "name": "self_rag",
            "query": "Explain quantum computing with examples",
            "strategy": "self_rag",
            "expected_latency": 1000
        }
    ]
    
    results = {}
    for scenario in test_scenarios:
        times = []
        for _ in range(10):
            start = time.time()
            response = await process_rag_query(
                scenario["query"],
                strategy=scenario["strategy"]
            )
            elapsed = time.time() - start
            times.append(elapsed * 1000)  # Convert to ms
        
        avg_latency = sum(times) / len(times)
        results[scenario["name"]] = {
            "avg_latency_ms": avg_latency,
            "p95_latency_ms": sorted(times)[int(len(times) * 0.95)],
            "meets_sla": avg_latency <= scenario["expected_latency"],
            "expected_latency_ms": scenario["expected_latency"]
        }
    
    return results
```

### 7. **Benchmarks de Concurrencia**

```python
# benchmark_concurrency.py
async def benchmark_concurrent_load():
    concurrent_users = [1, 10, 50, 100, 200]
    results = {}
    
    for users in concurrent_users:
        start = time.time()
        tasks = []
        
        for i in range(users):
            query = f"Test query {i % 10}"  # 10 query variations
            tasks.append(process_rag_query(query))
        
        responses = await asyncio.gather(*tasks)
        elapsed = time.time() - start
        
        successful = sum(1 for r in responses if r.status == "success")
        results[f"users_{users}"] = {
            "total_time": elapsed,
            "requests_per_second": users / elapsed,
            "success_rate": successful / users,
            "avg_time_per_request": elapsed / users
        }
    
    return results
```

### 8. **Benchmark de Memoria**

```bash
# Monitorear uso de memoria durante las pruebas
!python scripts/memory_profiler.py --duration 300 --interval 1
```

### 9. **Generar Reporte Completo**

```markdown
# RAG System Performance Benchmark Report
**Date**: [Timestamp]
**Environment**: [Dev/Staging/Prod]
**Version**: [Git commit hash]

## 📊 Executive Summary
- **Overall Performance**: [PASS/FAIL]
- **SLA Compliance**: X%
- **Bottlenecks Identified**: [List]

## 🚀 Benchmark Results

### Document Ingestion
| File Type | Pages | Time (s) | Pages/sec |
|-----------|-------|----------|-----------|
| Small PDF | 1     | X.XX     | X.XX      |
| Large PDF | 100   | X.XX     | X.XX      |

### Embedding Generation
| Test Case | Count | Time (s) | Embeddings/sec |
|-----------|-------|----------|----------------|
| Single    | 1     | X.XX     | X.XX           |
| Batch     | 100   | X.XX     | X.XX           |

### Vector Search Performance
| Query Type | k  | Avg Time (ms) | P95 (ms) |
|------------|----|--------------:|----------|
| Simple     | 10 | X.X          | X.X      |
| Complex    | 10 | X.X          | X.X      |

### Reranking Performance
| Method        | Docs | Time (ms) | Docs/sec |
|---------------|------|-----------|----------|
| Cross-Encoder | 20   | X.X       | X.X      |
| ColBERT       | 20   | X.X       | X.X      |

### End-to-End Latency
| Scenario     | Avg (ms) | P95 (ms) | SLA Met |
|--------------|----------|----------|---------|
| Simple RAG   | X.X      | X.X      | ✅      |
| Advanced RAG | X.X      | X.X      | ✅      |
| Self-RAG     | X.X      | X.X      | ❌      |

### Concurrency Testing
| Users | RPS  | Success Rate | Avg Response (ms) |
|-------|------|--------------|-------------------|
| 1     | X.X  | 100%         | X.X               |
| 100   | X.X  | 99.X%        | X.X               |
| 200   | X.X  | 9X.X%        | X.X               |

## 📈 Performance Trends
[Graphs comparing with previous benchmarks]

## 🔍 Analysis
### Strengths
- [What's performing well]

### Weaknesses  
- [Areas needing optimization]

### Recommendations
1. [Specific optimization suggestions]
2. [Configuration changes]
3. [Architecture improvements]

## 💾 System Resources
- **Peak Memory Usage**: X GB
- **CPU Utilization**: X%
- **Disk I/O**: X MB/s
- **Network**: X Mbps

## 🎯 Next Steps
- [ ] Optimize [specific component]
- [ ] Re-run benchmarks after changes
- [ ] Update SLA targets based on results
```

### 10. **Comparación con Objetivos**

```python
# Verificar contra SLAs definidos
sla_targets = {
    "simple_query_latency_ms": 200,
    "complex_query_latency_ms": 500,
    "ingestion_pages_per_second": 10,
    "concurrent_users": 100,
    "success_rate": 0.999
}

# Generar alertas si no se cumplen SLAs
```

## 🔄 Ejecución Automatizada

```bash
# Script para ejecutar todos los benchmarks
!python scripts/run_all_benchmarks.py --output reports/benchmark_$(date +%Y%m%d_%H%M%S).json
```

Este benchmark proporcionará métricas detalladas para optimizar el sistema y asegurar que cumple con los requisitos de performance.