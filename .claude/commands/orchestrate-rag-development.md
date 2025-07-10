---
allowed-tools: ["Bash", "Read", "Write", "MultiEdit", "TodoWrite", "Grep", "LS", "Task", "WebSearch", "WebFetch"]
description: "Orquestador maestro que ejecuta automáticamente el desarrollo del sistema RAG"
---

# Orquestador RAG - Ejecución Automática

Ejecuta de forma autónoma el desarrollo del sistema RAG siguiendo la arquitectura limpia.

## 📋 Procesamiento de Argumentos

Analiza los argumentos recibidos: $ARGUMENTS

1. **Usar Bash** para extraer parámetros:
   - `--duration Xh`: Límite de tiempo en horas
   - `--until-phase-complete`: Continuar hasta completar fase actual
   - `--focus "área"`: Enfocar en área específica

2. **Calcular tiempo límite** si se especifica duration y almacenar en variable.

## 🚀 Inicialización del Orquestador

1. **Usar LS** para verificar si existe `.orchestration/` 
2. **Si no existe, usar Bash** para crear estructura:
   ```
   mkdir -p .orchestration/{state,logs,results,backups}
   ```

3. **Usar Write** para crear/actualizar estado de sesión:
   ```json
   {
     "session_id": "rag-session-[timestamp]",
     "started_at": "[timestamp ISO]", 
     "current_phase": "initialization",
     "execution_history": [],
     "pending_tasks": [],
     "metrics": {"coverage": 0, "lines_of_code": 0, "entities": 0}
   }
   ```

## 🔍 Análisis del Estado Actual

1. **Usar LS** para inventariar estructura del proyecto
2. **Usar Bash** para contar archivos Python: `find src/ -name "*.py" | wc -l`
3. **Usar Bash** para contar líneas de código: `wc -l src/**/*.py`
4. **Usar Bash** para ejecutar tests: `python -m pytest tests/ -q`

## 🎯 Decisión Inteligente de Siguiente Acción

Basado en el análisis, determinar prioridad:

### Si no existe `src/`:
- **Usar Bash** para crear estructura de Clean Architecture
- **Usar Write** para crear archivos base

### Si existe estructura pero faltan entidades:
- **Usar Write** para implementar entidades faltantes
- **Usar Write** para crear tests unitarios correspondientes

### Si existen entidades pero faltan repositorios:
- **Usar Write** para crear interfaces de repositorio
- **Usar Write** para implementar casos de uso

### Si falta capa de infraestructura:
- **Usar Write** para implementar repositorios concretos
- **Usar Write** para configurar base de datos

### Si falta API:
- **Usar Write** para crear endpoints FastAPI
- **Usar Write** para configurar middleware y validaciones

## ⚙️ Ejecución de Tareas por Fase

### Fase 1: Setup y Dominio
1. **Usar TodoWrite** para crear lista de tareas de esta fase
2. **Marcar tarea actual como in_progress**
3. **Ejecutar implementación** usando Write/MultiEdit
4. **Usar Bash** para ejecutar tests después de cada implementación
5. **Marcar tarea como completed** al terminar
6. **Actualizar métricas** en archivo de estado

### Fase 2: Aplicación y Casos de Uso  
1. **Repetir proceso** con tareas de casos de uso
2. **Usar Write** para implementar servicios de aplicación
3. **Crear tests de integración** con Write

### Fase 3: Infraestructura
1. **Usar Write** para implementar repositorios concretos
2. **Configurar base de datos** con archivos de configuración
3. **Usar Write** para crear migraciones si es necesario

### Fase 4: Presentación (API)
1. **Usar Write** para crear controllers FastAPI
2. **Implementar middleware** y manejo de errores
3. **Crear documentación automática** de API

## 🔄 Ciclo de Verificación Continua

Después de cada implementación significativa:

1. **Usar Bash** para ejecutar tests: `python -m pytest tests/ -v`
2. **Si tests fallan**: Analizar error y corregir usando Edit
3. **Usar Bash** para verificar linting: `ruff check src/` (si existe)
4. **Actualizar archivo de estado** con Write usando métricas actuales

## ⏰ Control de Tiempo (si aplica)

Si se especificó `--duration`:
1. **Usar Bash** para verificar tiempo transcurrido: `date +%s`
2. **Si se alcanza límite**: Generar reporte final y terminar
3. **Cada 15 minutos**: Actualizar estado y mostrar progreso

## 📊 Monitoreo de Progreso

Durante la ejecución:

1. **Contar archivos creados/modificados**
2. **Actualizar métricas en tiempo real**:
   - Líneas de código
   - Cobertura de tests  
   - Entidades implementadas
   - Endpoints creados

3. **Usar Write** para actualizar archivo de estado cada 5 tareas completadas

## 🎁 Finalización

Al completar todas las tareas de la fase o alcanzar límite de tiempo:

1. **Usar Bash** para generar reporte final de métricas
2. **Usar Write** para crear resumen de sesión
3. **Mostrar progreso alcanzado** y próximos pasos recomendados

## 📝 Registro de Actividad

Durante toda la ejecución:
- **Usar Write** para registrar cada acción en `.orchestration/logs/session-[id].log`
- **Documentar errores** y cómo se resolvieron
- **Guardar estado** después de cada tarea importante

---

**Ejecuta este flujo automáticamente, tomando decisiones basadas en el estado actual del proyecto y continuando hasta alcanzar el objetivo o límite de tiempo especificado.**