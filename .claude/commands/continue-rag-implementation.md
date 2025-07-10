---
allowed-tools: ["Bash", "Read", "Write", "MultiEdit", "TodoWrite", "Grep", "LS"]
description: "Continúa la implementación del sistema RAG según las fases definidas en la guía"
---

# Continuación de Implementación RAG

Continúa la implementación del sistema RAG basándose en el estado actual.

## 🔍 Análisis del Estado Actual

1. **Usar LS** para verificar estructura del proyecto en `src/`
2. **Usar Bash** para contar entidades implementadas:
   ```bash
   find src/domain/entities -name "*.py" | grep -v __init__ | wc -l
   ```
3. **Usar Bash** para verificar casos de uso:
   ```bash
   find src/application/use_cases -name "*.py" | grep -v __init__ | wc -l
   ```
4. **Usar Bash** para verificar infraestructura:
   ```bash
   find src/infrastructure -name "*.py" | wc -l
   ```

## 🎯 Determinación de la Siguiente Fase

**Usar la lógica de decisión** basada en el análisis anterior:

### Si entidades < 3:
- **Usar Write** para implementar entidades faltantes
- **Usar Write** para crear tests unitarios correspondientes
- **Usar TodoWrite** para trackear progreso de entidades

### Si entidades completas pero casos de uso < 2:
- **Usar Write** para implementar casos de uso de búsqueda
- **Usar Write** para implementar casos de uso de indexación
- **Usar Write** para crear tests de casos de uso

### Si casos de uso completos pero sin infraestructura:
- **Usar Write** para implementar repositorios concretos
- **Usar Write** para configurar base de datos
- **Usar Write** para integrar servicios de embeddings

### Si infraestructura lista pero sin API:
- **Usar Write** para crear endpoints FastAPI
- **Usar Write** para implementar middleware
- **Usar Write** para configurar documentación automática

## ⚙️ Implementación Específica por Fase

### Fase 1: Implementación de Dominio

**Si falta entidad Document**:
1. **Usar Write** para crear `src/domain/entities/document.py` con:
   - Clase Document con dataclass
   - Métodos factory para creación
   - Validaciones de negocio
   - Métodos de actualización

**Si falta entidad Query**:
2. **Usar Write** para crear `src/domain/entities/query.py` con:
   - Clase Query para búsquedas
   - Filtros y parámetros
   - Validaciones de entrada

**Si falta entidad RAGResponse**:
3. **Usar Write** para crear `src/domain/entities/rag_response.py` con:
   - Resultado de búsquedas
   - Metadatos de relevancia
   - Tracking de origen

### Fase 2: Implementación de Casos de Uso

**Caso de uso de búsqueda**:
1. **Usar Write** para crear `src/application/use_cases/search_documents.py`:
   - Clase SearchDocuments con dependency injection
   - Método execute async
   - Validaciones de entrada
   - Manejo de errores

**Caso de uso de indexación**:
2. **Usar Write** para crear `src/application/use_cases/index_document.py`:
   - Procesamiento de documentos
   - Generación de embeddings
   - Almacenamiento en repositorio

**Servicios de aplicación**:
3. **Usar Write** para crear servicios auxiliares en `src/application/services/`

### Fase 3: Implementación de Infraestructura

**Repositorio de documentos**:
1. **Usar Write** para crear `src/infrastructure/repositories/postgres_document_repository.py`:
   - Implementación con asyncpg
   - Integración con pgvector
   - Manejo de conexiones
   - Queries optimizados

**Configuración de base de datos**:
2. **Usar Write** para crear `src/infrastructure/database/`:
   - Configuración de conexión
   - Migraciones
   - Pooling de conexiones

**Servicios externos**:
3. **Usar Write** para crear `src/infrastructure/services/`:
   - Cliente para embeddings
   - Configuración de APIs externas

### Fase 4: Implementación de API

**Endpoints de búsqueda**:
1. **Usar Write** para crear `src/presentation/api/routes/search.py`:
   - Router FastAPI para búsquedas
   - Validación con Pydantic
   - Dependency injection
   - Manejo de errores HTTP

**Schemas de API**:
2. **Usar Write** para crear `src/presentation/schemas/`:
   - Modelos Pydantic para requests
   - Modelos para responses
   - Validaciones de entrada

**Configuración principal**:
3. **Usar Write** para crear `src/presentation/api/main.py`:
   - Aplicación FastAPI principal
   - Middleware
   - Configuración CORS

## ✅ Verificación y Testing

Después de cada implementación:

1. **Usar Bash** para ejecutar tests unitarios:
   ```bash
   python -m pytest tests/unit/ -v
   ```

2. **Si tests fallan**, usar Read para analizar errores y Edit para corregir

3. **Usar Bash** para verificar cobertura:
   ```bash
   python -m pytest --cov=src tests/ --cov-report=term-missing
   ```

4. **Usar TodoWrite** para marcar tareas completadas y actualizar progreso

## 🔄 Ciclo Continuo

**Al completar implementación actual**:

1. **Usar Write** para actualizar progreso en `.orchestration/state/`
2. **Usar Bash** para hacer commit si todo funciona:
   ```bash
   git add . && git commit -m "feat: implementar [descripción]"
   ```
3. **Usar TodoWrite** para planificar siguiente iteración
4. **Determinar siguiente fase** y continuar automáticamente

---

**Ejecuta este comando de forma autónoma, tomando decisiones basadas en el estado actual y continuando hasta completar la fase o encontrar un blocker.**