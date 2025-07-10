---
allowed-tools: ["Read", "Edit", "MultiEdit", "Bash", "TodoWrite"]
description: "Optimiza el proceso de generación y almacenamiento de embeddings"
---

Optimiza el sistema de embeddings para el archivo o módulo: $ARGUMENTS

Tareas a realizar:

1. **Análisis del Sistema Actual**:
   - Identifica el modelo de embeddings utilizado
   - Revisa el proceso de generación actual
   - Detecta ineficiencias en el almacenamiento

2. **Optimizaciones de Procesamiento**:
   - Implementa batch processing si no existe
   - Añade paralelización donde sea posible
   - Optimiza el preprocesamiento de textos

3. **Mejoras de Almacenamiento**:
   - Evalúa el formato de almacenamiento actual
   - Implementa compresión si es aplicable
   - Optimiza índices para búsquedas rápidas

4. **Caché y Reutilización**:
   - Implementa sistema de caché para embeddings frecuentes
   - Evita recálculos innecesarios
   - Añade invalidación inteligente de caché

5. **Testing y Validación**:
   - Crea tests para verificar calidad de embeddings
   - Mide mejoras de performance
   - Valida que no hay pérdida de precisión

Documenta todos los cambios con comentarios claros sobre las mejoras implementadas.