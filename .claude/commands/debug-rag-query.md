---
allowed-tools: ["Read", "Bash", "Edit", "TodoWrite"]
description: "Debugea una query específica del sistema RAG paso a paso"
---

Debugea la siguiente query en el sistema RAG: $ARGUMENTS

Proceso de debugging detallado:

1. **Trazar el Flujo de la Query**:
   - Log de entrada de la query
   - Preprocesamiento aplicado
   - Transformaciones de texto

2. **Análisis de Embeddings**:
   - Vector generado para la query
   - Modelo utilizado
   - Tiempo de generación

3. **Proceso de Retrieval**:
   - Documentos recuperados (top-k)
   - Scores de similitud
   - Filtros aplicados

4. **Reranking (si aplica)**:
   - Algoritmo utilizado
   - Cambios en el orden
   - Scores finales

5. **Generación de Respuesta**:
   - Prompt final construido
   - Contexto incluido
   - Respuesta del LLM

6. **Métricas de Performance**:
   ```
   Query Debug Report:
   ==================
   - Query Processing: Xms
   - Embedding Generation: Xms
   - Vector Search: Xms
   - Reranking: Xms
   - LLM Generation: Xms
   - Total Time: Xms
   
   Retrieved Documents:
   1. [Doc ID] - Score: X.XX - Title: ...
   2. [Doc ID] - Score: X.XX - Title: ...
   ```

7. **Identificar Problemas**:
   - ¿La query fue bien procesada?
   - ¿Los documentos recuperados son relevantes?
   - ¿El contexto es suficiente?
   - ¿La respuesta es coherente?

8. **Sugerencias de Mejora**:
   - Optimizaciones de query
   - Ajustes en parámetros de retrieval
   - Mejoras en prompts

Proporciona un análisis completo con visualización del flujo y recomendaciones específicas.