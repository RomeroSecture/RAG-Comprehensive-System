---
allowed-tools: ["WebSearch", "WebFetch", "Write", "Read", "TodoWrite"]
description: "Busca soluciones, mejores prácticas y ejemplos de implementación RAG en internet"
---

Busca información sobre: $ARGUMENTS

## 🔍 Búsqueda Inteligente de Soluciones RAG

### 1. **Análisis de la Consulta**
Determina el tipo de búsqueda:
- Solución a un problema específico
- Mejores prácticas de implementación
- Ejemplos de código
- Documentación técnica
- Benchmarks y comparativas

### 2. **Estrategia de Búsqueda**

#### Para Problemas Técnicos
```python
search_queries = [
    f"{problem} solution {technology} 2024",
    f"fix {error_message} {framework}",
    f"{problem} best practices {language}",
    f"Stack Overflow {problem} {technology}"
]
```

#### Para Mejores Prácticas
```python
search_queries = [
    "RAG best practices 2024",
    "Retrieval Augmented Generation optimization techniques",
    "Self-RAG implementation guide",
    "GraphRAG architecture patterns",
    "Multimodal RAG examples"
]
```

#### Para Ejemplos de Código
```python
search_queries = [
    f"{component} implementation example GitHub",
    f"open source RAG {feature} Python",
    f"{framework} RAG tutorial code",
    f"production RAG system architecture"
]
```

### 3. **Búsqueda Web Estructurada**

Ejecutar búsquedas con filtros específicos:

```python
# Buscar en sitios confiables
trusted_domains = [
    "github.com",
    "arxiv.org",
    "huggingface.co",
    "langchain.readthedocs.io",
    "stackoverflow.com",
    "medium.com/@technical",
    "towardsdatascience.com"
]

# Buscar documentación oficial
official_docs = [
    "docs.openai.com",
    "platform.openai.com",
    "anthropic.com/docs",
    "fastapi.tiangolo.com",
    "pgvector.github.io"
]
```

### 4. **Procesamiento de Resultados**

Para cada resultado relevante:

1. **Extraer Información Clave**:
   - Título y descripción
   - Fecha de publicación
   - Autor/fuente
   - Relevancia para el problema

2. **Analizar Contenido**:
   ```
   WebFetch con prompts específicos:
   - "Extract the solution or implementation details"
   - "Identify code examples and their purpose"
   - "List prerequisites and dependencies"
   - "Find performance metrics or benchmarks"
   ```

3. **Evaluar Aplicabilidad**:
   - ¿Es compatible con nuestro stack?
   - ¿Qué tan reciente es la información?
   - ¿Hay casos de uso similares?
   - ¿Cuáles son las limitaciones?

### 5. **Búsquedas Especializadas**

#### Para Papers Académicos
```
site:arxiv.org "Retrieval Augmented Generation" 2024
site:aclanthology.org RAG improvements
site:paperswithcode.com RAG benchmarks
```

#### Para Código Open Source
```
site:github.com "RAG implementation" language:Python stars:>100
site:huggingface.co RAG model
site:github.com awesome-rag curated list
```

#### Para Soluciones de Producción
```
"RAG production deployment" kubernetes
"RAG system architecture" "high scale"
"RAG performance optimization" "real world"
```

### 6. **Síntesis de Resultados**

Crear un reporte estructurado:

```markdown
# Resultados de Búsqueda: [Tema]
**Fecha**: [Timestamp]
**Consulta Original**: [Query]

## 📊 Resumen Ejecutivo
[Resumen de 2-3 párrafos con los hallazgos principales]

## 🎯 Soluciones Encontradas

### Opción 1: [Nombre]
- **Fuente**: [URL]
- **Fecha**: [Cuándo se publicó]
- **Resumen**: [Descripción breve]
- **Pros**: [Ventajas]
- **Contras**: [Desventajas]
- **Código de Ejemplo**:
```python
# Código relevante extraído
```

### Opción 2: [Nombre]
[Similar estructura...]

## 💡 Mejores Prácticas Identificadas
1. [Práctica 1 con explicación]
2. [Práctica 2 con explicación]
3. [Práctica 3 con explicación]

## 🔧 Implementación Recomendada
Basado en la investigación, la mejor aproximación sería:
[Recomendación detallada]

## 📚 Recursos Adicionales
- [Link 1]: [Descripción]
- [Link 2]: [Descripción]
- [Link 3]: [Descripción]

## ⚠️ Consideraciones
- [Advertencias o limitaciones encontradas]
- [Posibles problemas a evitar]
- [Dependencias o prerequisitos]

## 🔄 Siguientes Pasos
1. [Acción recomendada 1]
2. [Acción recomendada 2]
3. [Acción recomendada 3]
```

### 7. **Guardar Resultados**

Guardar en directorio de investigación:
```
./rag-comprehensive-system/research/
├── embedding_optimization_2024-01-15.md
├── pgvector_setup_issues_2024-01-16.md
├── selfrag_implementation_2024-01-17.md
└── index.md  # Índice de todas las búsquedas
```

### 8. **Integración con TodoWrite**

Si se encuentran tareas accionables:
- Agregar a la lista de TODOs
- Priorizar según impacto
- Vincular con la investigación

### 9. **Búsquedas Contextuales**

Adaptar búsquedas según la fase actual:
- **Fase 1**: Setup, configuración, arquitectura básica
- **Fase 2**: Retrieval avanzado, reranking, optimización
- **Fase 3**: Multimodal, Self-RAG, características avanzadas
- **Fase 4**: Producción, escalabilidad, monitoreo

### 10. **Cache de Búsquedas**

Mantener cache de búsquedas recientes:
```json
{
  "query": "original search",
  "timestamp": "2024-01-15T10:00:00Z",
  "results_summary": "...",
  "full_report_path": "research/report.md",
  "actionable_items": []
}
```

## 🎯 Resultado Esperado

Un reporte completo y accionable que:
- Resuelve el problema o pregunta planteada
- Proporciona múltiples opciones cuando aplique
- Incluye código ejemplo listo para usar
- Identifica mejores prácticas actuales
- Sugiere pasos concretos a seguir

## 📤 Salida Estándar

```bash
# Inicio del comando
COMMAND_NAME="search-rag-solutions"
START_TIME=$(date +%s)
mkdir -p .orchestration

# Variables de tracking
SOLUTIONS_FOUND=0
SOLUTION_IMPLEMENTED=false

# ... ejecutar búsquedas y análisis ...

# Al finalizar
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Determinar siguiente acción
if [ $SOLUTIONS_FOUND -gt 0 ]; then
    if [ "$SOLUTION_IMPLEMENTED" = true ]; then
        NEXT_CMD="test-rag-pipeline"
        NEXT_REASON="Solution implemented, needs testing"
    else
        NEXT_CMD="continue-rag-implementation"
        NEXT_REASON="Solution found, ready to implement"
    fi
    STATUS="success"
else
    NEXT_CMD="handle-blocker"
    NEXT_REASON="No solution found, needs manual intervention"
    STATUS="partial"
fi

cat > .orchestration/${COMMAND_NAME}_result.json << EOF
{
  "command": "${COMMAND_NAME}",
  "status": "${STATUS}",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "duration_seconds": ${DURATION},
  "arguments": "$ARGUMENTS",
  "results": {
    "query": "$ARGUMENTS",
    "solutions_found": ${SOLUTIONS_FOUND},
    "sources_consulted": ${SOURCES_COUNT},
    "solution_implemented": ${SOLUTION_IMPLEMENTED},
    "report_saved_to": "${REPORT_PATH}"
  },
  "next_recommended_action": {
    "command": "${NEXT_CMD}",
    "reason": "${NEXT_REASON}",
    "priority": "high"
  },
  "summary": "Found ${SOLUTIONS_FOUND} solutions for: $ARGUMENTS"
}
EOF
```