---
allowed-tools: ["Write", "Edit", "Read", "Bash", "TodoWrite"]
description: "Crea un nuevo agente RAG con configuración y estructura estándar"
---

Crea un nuevo agente RAG llamado: $ARGUMENTS

Sigue esta estructura estándar para el sistema:

1. **Crear Estructura de Archivos**:
   ```
   agents/
   └── $ARGUMENTS/
       ├── __init__.py
       ├── agent.py          # Lógica principal del agente
       ├── prompts.py        # Templates de prompts
       ├── config.py         # Configuración del agente
       ├── tools.py          # Herramientas específicas
       └── tests/
           └── test_agent.py
   ```

2. **Implementar Agent Base**:
   - Hereda de la clase base del sistema
   - Configura el modelo LLM específico
   - Define el contexto y memoria del agente
   - Implementa métodos de procesamiento RAG

3. **Configurar Prompts**:
   - System prompt específico del dominio
   - Templates para diferentes tipos de consultas
   - Instrucciones de formato de respuesta

4. **Integrar con el Sistema RAG**:
   - Conectar con el vector store existente
   - Configurar estrategias de retrieval
   - Implementar reranking si es necesario

5. **Añadir al Registry**:
   - Registrar el agente en el sistema principal
   - Configurar rutas en FastAPI
   - Actualizar documentación de la API

6. **Tests Básicos**:
   - Test de inicialización
   - Test de consulta simple
   - Test de integración con RAG

Asegúrate de seguir las convenciones del proyecto y usar los patrones existentes.