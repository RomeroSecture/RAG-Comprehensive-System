---
allowed-tools: ["Read", "Write", "Edit", "Grep"]
description: "Genera o actualiza documentación de la API FastAPI del sistema RAG"
---

Genera/actualiza la documentación de la API para: $ARGUMENTS

Tareas de documentación:

1. **Análisis de Endpoints**:
   - Identifica todos los endpoints de la API
   - Extrae parámetros, tipos y validaciones
   - Documenta códigos de respuesta

2. **Generar OpenAPI Schema**:
   - Asegura que todos los endpoints tienen docstrings
   - Añade ejemplos de request/response
   - Documenta modelos Pydantic utilizados

3. **Crear Documentación Markdown**:
   ```markdown
   # API Documentation - [Componente]
   
   ## Endpoints
   
   ### POST /api/v1/[endpoint]
   - **Descripción**: ...
   - **Parámetros**:
     - `param1` (tipo): descripción
   - **Request Body**:
     ```json
     {
       "field": "value"
     }
     ```
   - **Response**:
     ```json
     {
       "result": "..."
     }
     ```
   - **Errores**: 400, 404, 500
   ```

4. **Ejemplos de Uso**:
   - cURL commands
   - Python requests
   - JavaScript fetch

5. **Guías de Integración**:
   - Autenticación requerida
   - Rate limiting
   - Best practices

6. **Actualizar README**:
   - Añadir link a la documentación
   - Actualizar tabla de contenidos
   - Incluir quickstart guide

Asegúrate de que la documentación sea clara, completa y con ejemplos prácticos.