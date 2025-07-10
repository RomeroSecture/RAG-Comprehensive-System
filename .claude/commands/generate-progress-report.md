---
allowed-tools: ["Bash", "Read", "Write", "LS", "Grep", "TodoWrite"]
description: "Genera reporte automático del progreso del proyecto RAG con métricas y análisis"
---

# Generador de Reporte de Progreso RAG

Genera automáticamente un reporte completo del estado y progreso del proyecto.

## 📊 Análisis de Métricas del Proyecto

1. **Usar LS** para inventariar estructura completa:
   ```
   src/, tests/, docs/, scripts/
   ```

2. **Usar Bash** para calcular métricas de código:
   ```bash
   # Contar archivos Python
   PYTHON_FILES=$(find src/ -name "*.py" | wc -l)
   
   # Contar líneas de código
   LOC=$(find src/ -name "*.py" -exec wc -l {} + | tail -1 | awk '{print $1}')
   
   # Contar tests
   TEST_FILES=$(find tests/ -name "test_*.py" | wc -l)
   ```

3. **Usar Bash** para obtener cobertura de tests:
   ```bash
   python -m pytest --cov=src --cov-report=json --quiet
   ```

## 🏗️ Análisis de Arquitectura

1. **Usar LS** para verificar cumplimiento de Clean Architecture:
   - Domain entities: `src/domain/entities/`
   - Use cases: `src/application/use_cases/`
   - Infrastructure: `src/infrastructure/`
   - Presentation: `src/presentation/`

2. **Usar Bash** para contar componentes por capa:
   ```bash
   ENTITIES=$(find src/domain/entities -name "*.py" | grep -v __init__ | wc -l)
   USECASES=$(find src/application/use_cases -name "*.py" | grep -v __init__ | wc -l)
   REPOS=$(find src/infrastructure -name "*repository*.py" | wc -l)
   ENDPOINTS=$(find src/presentation -name "*.py" | grep -v __init__ | wc -l)
   ```

## 🧪 Análisis de Testing

1. **Usar Bash** para ejecutar suite completa de tests:
   ```bash
   python -m pytest tests/ --tb=short -q
   ```

2. **Usar Bash** para obtener estadísticas detalladas:
   ```bash
   # Tests por categoría
   UNIT_TESTS=$(find tests/unit -name "test_*.py" | wc -l)
   INTEGRATION_TESTS=$(find tests/integration -name "test_*.py" | wc -l || echo 0)
   E2E_TESTS=$(find tests/e2e -name "test_*.py" | wc -l || echo 0)
   ```

## 📈 Análisis de Progreso

1. **Usar Read** para leer estado actual de orquestación:
   ```
   .orchestration/state/current_session.json
   ```

2. **Calcular progreso por fase**:
   - Fase 1 (Domain): Entidades completadas / Total esperado
   - Fase 2 (Application): Use cases / Total esperado
   - Fase 3 (Infrastructure): Repositorios / Total esperado
   - Fase 4 (Presentation): Endpoints / Total esperado

## 🔍 Análisis de Calidad

1. **Usar Bash** para ejecutar linters:
   ```bash
   # Verificar estilo de código
   ruff check src/ --output-format=json 2>/dev/null || echo '{"violations": 0}'
   
   # Verificar type hints
   mypy src/ --ignore-missing-imports 2>/dev/null || echo "Type check completed"
   ```

2. **Usar Grep** para detectar code smells:
   ```bash
   # Buscar TODOs y FIXMEs
   grep -r "TODO\|FIXME\|XXX" src/ | wc -l
   ```

## 📝 Generación del Reporte

1. **Usar Write** para crear reporte markdown:
   ```markdown
   # 📊 Reporte de Progreso RAG - [TIMESTAMP]
   
   ## 🎯 Resumen Ejecutivo
   - **Progreso General**: X%
   - **Fase Actual**: [Fase actual]
   - **Salud del Proyecto**: 🟢/🟡/🔴
   
   ## 📈 Métricas de Código
   - **Archivos Python**: X
   - **Líneas de código**: X
   - **Cobertura de tests**: X%
   - **Tests implementados**: X
   
   ## 🏗️ Estado de Arquitectura
   - **Entidades**: X/3 ✅
   - **Use Cases**: X/5 🔄
   - **Repositorios**: X/3 ⏳
   - **Endpoints**: X/8 ⏳
   
   ## 🧪 Calidad del Código
   - **Tests pasando**: ✅ X/X
   - **Violaciones de linting**: X
   - **Type hints**: X%
   - **Code smells**: X pendientes
   
   ## 🚀 Próximos Pasos
   1. [Siguiente tarea prioritaria]
   2. [Segunda tarea]
   3. [Tercera tarea]
   
   ## ⚠️ Blockers y Riesgos
   - [Blocker 1 si existe]
   - [Riesgo identificado]
   ```

## 📊 Dashboard Visual

1. **Usar Write** para crear dashboard en JSON:
   ```json
   {
     "timestamp": "...",
     "overall_progress": 75,
     "phase_progress": {
       "domain": 100,
       "application": 60,
       "infrastructure": 0,
       "presentation": 0
     },
     "metrics": {
       "files": X,
       "lines_of_code": X,
       "test_coverage": X,
       "tests_count": X
     },
     "health_indicators": {
       "tests_passing": true,
       "linting_clean": true,
       "architecture_compliant": true
     }
   }
   ```

## 🔄 Actualización de Estado

1. **Usar Write** para actualizar estado de orquestación con nuevas métricas

2. **Usar TodoWrite** para actualizar lista de tareas basada en análisis

## 📧 Notificaciones y Alertas

**Generar alertas si**:
- Cobertura de tests < 90%
- Violaciones de linting > 5
- Tests fallando
- Arquitectura no cumple estándares

---

**Ejecuta este comando para generar automáticamente un reporte completo del progreso del proyecto.**