---
allowed-tools: ["Bash", "Read", "Edit", "TodoWrite"]
description: "Ejecuta tests completos del pipeline RAG con métricas de calidad"
---

Ejecuta una suite completa de tests para el pipeline RAG.

Pasos a seguir:

1. **Tests Unitarios del Pipeline**:
   ```bash
   !pytest tests/test_rag_pipeline.py -v
   ```

2. **Tests de Calidad de Retrieval**:
   - Medir precision@k para diferentes valores de k
   - Calcular recall de documentos relevantes
   - Evaluar MRR (Mean Reciprocal Rank)

3. **Tests de Performance**:
   - Tiempo de indexación de documentos
   - Latencia de búsquedas vectoriales
   - Throughput del sistema completo

4. **Tests de Integración**:
   - Flujo completo: documento → embedding → retrieval → respuesta
   - Verificar manejo de errores
   - Validar respuestas del LLM

5. **Evaluación de Calidad**:
   - Crear conjunto de prueba con preguntas y respuestas esperadas
   - Medir BLEU/ROUGE scores si aplica
   - Evaluar coherencia y relevancia de respuestas

6. **Reporte de Métricas**:
   ```
   === RAG Pipeline Test Report ===
   - Retrieval Accuracy: X%
   - Average Latency: Xms
   - Test Coverage: X%
   - Failed Tests: X
   - Performance Regression: ±X%
   ```

7. **Análisis de Fallos**:
   - Identificar patrones en tests fallidos
   - Sugerir mejoras basadas en resultados
   - Crear issues para problemas encontrados

Genera un reporte detallado con todas las métricas y recomendaciones.