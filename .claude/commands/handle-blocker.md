---
allowed-tools: ["Read", "Write", "WebSearch", "WebFetch", "TodoWrite", "Bash", "Task"]
description: "Maneja blockers técnicos buscando soluciones, documentando el problema y proponiendo alternativas"
---

Maneja el siguiente blocker: $ARGUMENTS

## 🚧 Sistema de Manejo de Blockers

### 1. **Análisis del Blocker**

#### Clasificación del Problema
```python
blocker_types = {
    "dependency_conflict": "Conflicto entre dependencias",
    "api_limitation": "Limitación de API externa",
    "performance_issue": "Problema de rendimiento",
    "architecture_violation": "Violación de arquitectura",
    "missing_feature": "Feature no disponible en librería",
    "integration_error": "Error de integración",
    "environment_issue": "Problema de entorno/configuración",
    "knowledge_gap": "Falta de conocimiento técnico"
}
```

#### Recolección de Información
1. **Error exacto**: Mensaje completo y stack trace
2. **Contexto**: Qué se estaba intentando hacer
3. **Entorno**: Versiones, sistema operativo, configuración
4. **Intentos previos**: Qué se ha probado sin éxito
5. **Impacto**: Qué tareas están bloqueadas

### 2. **Búsqueda de Soluciones**

#### Búsqueda Específica del Error
```bash
# Si hay un mensaje de error específico
error_message="[mensaje de error]"
search_queries=[
    "$error_message solution",
    "$error_message $technology fix",
    "site:stackoverflow.com $error_message",
    "site:github.com/issues $error_message"
]
```

#### Búsqueda de Alternativas
```python
alternative_searches = [
    f"alternative to {blocked_library} Python",
    f"{feature} implementation without {dependency}",
    f"workaround {limitation} {framework}",
    f"how to {goal} when {constraint}"
]
```

### 3. **Análisis de Soluciones Encontradas**

Para cada solución potencial, evaluar:

#### Viabilidad
- ¿Es compatible con nuestro stack?
- ¿Requiere cambios mayores?
- ¿Cuánto tiempo tomaría implementar?
- ¿Introduce nuevas dependencias?

#### Riesgo
- ¿Qué tan probado está el approach?
- ¿Hay casos de éxito documentados?
- ¿Qué problemas podría introducir?
- ¿Es una solución temporal o permanente?

#### Impacto
- ¿Resuelve completamente el blocker?
- ¿Afecta otras partes del sistema?
- ¿Cambia la arquitectura planeada?
- ¿Impacta el performance?

### 4. **Documentación del Blocker**

Crear archivo detallado:
```markdown
# Blocker: [Título Descriptivo]
**ID**: BLK-[timestamp]
**Fecha**: [Fecha actual]
**Severidad**: Critical|High|Medium|Low
**Estado**: Active|Investigating|Workaround|Resolved

## Descripción
[Descripción detallada del problema]

## Contexto
- **Tarea afectada**: [Qué se estaba implementando]
- **Componente**: [Parte del sistema afectada]
- **Fase del proyecto**: [Phase X, Week Y]

## Detalles Técnicos
### Error/Problema
```
[Stack trace o mensaje de error completo]
```

### Entorno
- Python: X.X
- Framework: X.X
- OS: [Sistema operativo]
- [Otras versiones relevantes]

## Intentos de Solución
1. **Intento 1**: [Qué se probó]
   - Resultado: [Qué pasó]
   - Razón del fallo: [Por qué no funcionó]

2. **Intento 2**: [Siguiente intento]
   - Resultado: [Qué pasó]
   - Aprendizaje: [Qué se aprendió]

## Investigación
### Recursos Consultados
- [URL 1]: [Resumen de lo encontrado]
- [URL 2]: [Resumen de lo encontrado]

### Soluciones Potenciales
#### Opción 1: [Nombre]
- **Descripción**: [Qué implica]
- **Pros**: [Ventajas]
- **Contras**: [Desventajas]
- **Tiempo estimado**: [Horas/días]
- **Riesgo**: [Alto/Medio/Bajo]

#### Opción 2: [Nombre]
[Similar estructura]

## Recomendación
[Cuál opción se recomienda y por qué]

## Plan de Acción
1. [Paso 1 específico]
2. [Paso 2 específico]
3. [Verificación de que funciona]

## Workaround Temporal
Si no hay solución inmediata:
```python
# Código del workaround
```

## Impacto en el Proyecto
- **Tareas bloqueadas**: [Lista]
- **Retraso estimado**: [Tiempo]
- **Dependencias afectadas**: [Lista]

## Lecciones Aprendidas
- [Qué se aprendió de este blocker]
- [Cómo prevenir similares en el futuro]
```

### 5. **Implementar Solución o Workaround**

#### Si hay solución viable:
1. Crear branch para la solución
2. Implementar cambios necesarios
3. Añadir tests para verificar
4. Documentar la solución

#### Si NO hay solución inmediata:
1. Implementar workaround temporal
2. Marcar con TODO para resolver después
3. Crear issue para tracking
4. Continuar con tareas no bloqueadas

### 6. **Actualizar Estado del Proyecto**

#### En TodoWrite:
- Marcar tarea actual como bloqueada
- Añadir nueva tarea para resolver blocker
- Reordenar prioridades si es necesario

#### En archivo de estado:
```json
{
  "blockers": [
    {
      "id": "BLK-20240115-1",
      "title": "Cannot install pgvector extension",
      "severity": "high",
      "tasks_affected": ["vector_store_implementation"],
      "workaround_available": true,
      "estimated_resolution": "2 days"
    }
  ]
}
```

### 7. **Comunicación y Escalación**

Si el blocker es crítico:
1. Documentar impacto en timeline
2. Identificar alternativas arquitecturales
3. Preparar resumen ejecutivo
4. Sugerir ajustes al plan si necesario

### 8. **Prevención Futura**

Analizar el blocker para prevención:
- ¿Se pudo haber detectado antes?
- ¿Faltó investigación previa?
- ¿Hay suposiciones incorrectas?
- ¿Qué validaciones agregar?

### 9. **Base de Conocimiento**

Mantener registro de blockers resueltos:
```
.orchestration/blockers/
├── resolved/
│   ├── BLK-001-pgvector-setup.md
│   ├── BLK-002-async-conflict.md
│   └── BLK-003-memory-limit.md
├── active/
│   └── BLK-004-cuda-support.md
└── index.md
```

### 10. **Métricas de Blockers**

Trackear para mejorar:
- Tiempo promedio de resolución
- Blockers por fase del proyecto
- Categorías más comunes
- Efectividad de workarounds

## 🎯 Resultado Esperado

Un blocker bien manejado que:
- Está completamente documentado
- Tiene solución o workaround implementado
- No detiene el progreso del proyecto
- Aporta aprendizaje al equipo
- Previene problemas similares futuros

## 📤 Salida Estándar

```bash
# Inicio del comando
COMMAND_NAME="handle-blocker"
START_TIME=$(date +%s)
mkdir -p .orchestration

# Variables de tracking
BLOCKER_RESOLVED=false
WORKAROUND_FOUND=false
ISSUE_CREATED=false

# ... manejar el blocker ...

# Al finalizar
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Determinar siguiente acción
if [ "$BLOCKER_RESOLVED" = true ]; then
    NEXT_CMD="continue-rag-implementation"
    NEXT_REASON="Blocker resolved, can continue development"
    STATUS="success"
elif [ "$WORKAROUND_FOUND" = true ]; then
    NEXT_CMD="continue-rag-implementation"
    NEXT_REASON="Workaround implemented, can continue with caution"
    STATUS="partial"
else
    NEXT_CMD="search-rag-solutions"
    NEXT_REASON="Need to search for more solutions"
    STATUS="blocked"
fi

cat > .orchestration/${COMMAND_NAME}_result.json << EOF
{
  "command": "${COMMAND_NAME}",
  "status": "${STATUS}",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "duration_seconds": ${DURATION},
  "arguments": "$ARGUMENTS",
  "results": {
    "blocker_id": "${BLOCKER_ID}",
    "blocker_resolved": ${BLOCKER_RESOLVED},
    "workaround_found": ${WORKAROUND_FOUND},
    "issue_created": ${ISSUE_CREATED},
    "issue_url": "${ISSUE_URL}",
    "documentation_path": "${DOC_PATH}"
  },
  "blockers": [
    {
      "id": "${BLOCKER_ID}",
      "severity": "${SEVERITY}",
      "status": "${BLOCKER_STATUS}"
    }
  ],
  "next_recommended_action": {
    "command": "${NEXT_CMD}",
    "reason": "${NEXT_REASON}",
    "priority": "high"
  },
  "summary": "Blocker ${BLOCKER_ID}: ${BLOCKER_STATUS}"
}
EOF
```