---
allowed-tools: ["Bash", "Read", "Write", "Grep", "LS", "TodoWrite"]
description: "Valida que la arquitectura del sistema RAG sigue principios de Clean Architecture"
---

# Validación de Arquitectura RAG

Valida automáticamente que el código sigue los principios de Clean Architecture.

## 🔍 Análisis de Dependencias

1. **Usar Bash** para verificar dependencias circulares:
   ```bash
   python -c "
   import ast
   import os
   # Script para detectar dependencias circulares
   "
   ```

2. **Usar Grep** para verificar que domain no importa infrastructure:
   ```bash
   grep -r "from.*infrastructure" src/domain/ || echo "✅ Domain limpio"
   ```

3. **Usar Grep** para verificar que domain no importa presentation:
   ```bash
   grep -r "from.*presentation" src/domain/ || echo "✅ Domain independiente"
   ```

## 📋 Validación de Capas

### Capa Domain
1. **Usar LS** para verificar estructura de domain:
   ```
   src/domain/
   ├── entities/
   ├── repositories/
   └── services/
   ```

2. **Usar Read** para verificar que entidades no tienen dependencias externas
3. **Usar Grep** para buscar imports no permitidos en domain

### Capa Application  
1. **Usar LS** para verificar estructura de application:
   ```
   src/application/
   ├── use_cases/
   ├── services/
   └── dtos/
   ```

2. **Usar Grep** para verificar que use cases solo dependen de domain

### Capa Infrastructure
1. **Usar LS** para verificar implementaciones concretas
2. **Usar Read** para verificar que implementa interfaces de domain

### Capa Presentation
1. **Usar LS** para verificar controllers y schemas
2. **Usar Grep** para verificar que no accede directamente a infrastructure

## 🧪 Validación de Testing

1. **Usar Bash** para verificar cobertura de tests:
   ```bash
   python -m pytest --cov=src --cov-report=term-missing
   ```

2. **Usar LS** para verificar estructura de tests:
   ```
   tests/
   ├── unit/
   ├── integration/
   └── e2e/
   ```

## 📊 Métricas de Calidad

1. **Usar Bash** para calcular complejidad ciclomática:
   ```bash
   python -c "
   # Cálculo de complejidad usando radon
   import subprocess
   result = subprocess.run(['radon', 'cc', 'src/', '-a'], capture_output=True, text=True)
   print(result.stdout)
   "
   ```

2. **Usar Bash** para verificar type hints:
   ```bash
   mypy src/ --ignore-missing-imports
   ```

## 🔧 Validación de Configuración

1. **Usar Read** para verificar que existe configuración:
   - `.env.example`
   - `pyproject.toml`
   - `requirements.txt`

2. **Usar Bash** para verificar que dependencias son consistentes

## 📈 Generación de Reporte

1. **Usar Write** para crear reporte de validación:
   ```json
   {
     "timestamp": "...",
     "architecture_score": 95,
     "violations": [],
     "suggestions": [],
     "dependency_analysis": {...}
   }
   ```

2. **Usar TodoWrite** para crear tareas de refactoring si es necesario

## ⚠️ Detección de Code Smells

1. **Usar Grep** para buscar patrones problemáticos:
   - Clases muy grandes (>500 líneas)
   - Métodos muy largos (>50 líneas)
   - Duplicación de código
   - Magic numbers

2. **Usar Bash** para ejecutar linters:
   ```bash
   ruff check src/ --output-format=json
   ```

## ✅ Criterios de Aprobación

**La arquitectura es válida si**:
- ✅ No hay dependencias circulares
- ✅ Domain es independiente
- ✅ Cobertura de tests > 90%
- ✅ Type hints en 100% del código
- ✅ Sin violaciones críticas de linting

---

**Ejecuta este comando para validar automáticamente la arquitectura y generar reporte de calidad.**