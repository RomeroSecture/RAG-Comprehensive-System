---
allowed-tools: ["Bash", "Read", "Write", "LS"]
description: "Prueba el sistema completo de orquestación y comunicación entre comandos"
---

Ejecuta una prueba completa del sistema de orquestación.

## 🧪 Test del Sistema de Orquestación

### 1. **Verificar Estructura**
```bash
# Crear directorio si no existe
!mkdir -p ./rag-comprehensive-system/.orchestration

# Verificar que existe
!ls -la ./rag-comprehensive-system/.orchestration/
```

### 2. **Test de Argumentos del Orquestador**
```bash
echo "=== Test 1: Procesamiento de argumentos ==="

# Test con duración
echo "Test --duration:"
# Simular: /project:orchestrate-rag-development --duration 2h

# Test con focus
echo "Test --focus:"
# Simular: /project:orchestrate-rag-development --focus "testing"

# Test con múltiples argumentos
echo "Test múltiple:"
# Simular: /project:orchestrate-rag-development --duration 3h --focus "domain entities"
```

### 3. **Test de Comunicación Entre Comandos**

#### Simular salida de un comando
```bash
# Simular que manage-github-repo terminó
cat > .orchestration/manage-github-repo_result.json << EOF
{
  "command": "manage-github-repo",
  "status": "success",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "duration_seconds": 45,
  "arguments": "commit-changes",
  "results": {
    "action_taken": "commit",
    "commits_created": 2,
    "pr_created": false
  },
  "next_recommended_action": {
    "command": "test-rag-pipeline",
    "reason": "New commits require testing",
    "priority": "high"
  },
  "summary": "Created 2 commits successfully"
}
EOF

echo "✅ Archivo de resultado creado"
```

#### Verificar lectura del resultado
```bash
# Leer el archivo como lo haría el orquestador
if [ -f ".orchestration/manage-github-repo_result.json" ]; then
    STATUS=$(cat .orchestration/manage-github-repo_result.json | grep -o '"status": "[^"]*"' | cut -d'"' -f4)
    NEXT_CMD=$(cat .orchestration/manage-github-repo_result.json | grep -A3 "next_recommended_action" | grep "command" | cut -d'"' -f4)
    
    echo "📊 Status leído: $STATUS"
    echo "➡️  Siguiente comando recomendado: $NEXT_CMD"
else
    echo "❌ No se pudo leer el archivo de resultado"
fi
```

### 4. **Test de Cadena de Comandos**

Simular una cadena completa de ejecución:

```bash
echo "=== Test de Cadena de Comandos ==="

# 1. validate-architecture falla
cat > .orchestration/validate-architecture_result.json << EOF
{
  "command": "validate-architecture",
  "status": "failed",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "results": {
    "violations_found": 8
  },
  "next_recommended_action": {
    "command": "refactor-code",
    "reason": "Multiple architecture violations found",
    "priority": "high"
  }
}
EOF

# 2. refactor-code tiene éxito parcial
cat > .orchestration/refactor-code_result.json << EOF
{
  "command": "refactor-code",
  "status": "partial",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "next_recommended_action": {
    "command": "validate-architecture",
    "reason": "Re-validate after refactoring",
    "priority": "high"
  }
}
EOF

# 3. Verificar la cadena
echo "Cadena de comandos creada:"
!ls -la .orchestration/*.json | grep -E "(validate|refactor)"
```

### 5. **Test de Estado Global**

```bash
# Crear estado global del orquestador
cat > .orchestration/orchestrator_state.json << EOF
{
  "session_id": "test-$(date +%s)",
  "started_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "current_phase": 1,
  "current_week": 2,
  "commands_executed": [
    "setup-development-env",
    "validate-architecture",
    "refactor-code"
  ],
  "duration_limit_hours": 4,
  "time_remaining_hours": 2.5,
  "focus_area": "testing",
  "last_command": "refactor-code",
  "next_planned_command": "validate-architecture"
}
EOF

echo "✅ Estado global creado"
```

### 6. **Verificación Final**

```bash
echo "=== Resumen de la Prueba ==="
echo "1. Directorio .orchestration: $([ -d .orchestration ] && echo '✅ OK' || echo '❌ FALTA')"
echo "2. Archivos de resultado: $(ls .orchestration/*_result.json 2>/dev/null | wc -l) encontrados"
echo "3. Estado del orquestador: $([ -f .orchestration/orchestrator_state.json ] && echo '✅ OK' || echo '❌ FALTA')"

# Listar todos los archivos creados
echo -e "\nArchivos en .orchestration:"
!ls -la .orchestration/
```

### 7. **Limpieza (Opcional)**
```bash
# Para limpiar después de las pruebas
# !rm -rf .orchestration/test_*
```

## 📊 Resultado Esperado

Si todo funciona correctamente:
- ✅ El directorio `.orchestration` existe
- ✅ Los comandos pueden escribir archivos JSON de resultado
- ✅ Los archivos siguen el formato estándar
- ✅ El orquestador puede leer los resultados
- ✅ Las recomendaciones de siguiente acción funcionan
- ✅ El estado global se mantiene correctamente

## 🔍 Diagnóstico de Problemas

Si algo falla:
1. Verificar permisos del directorio
2. Verificar que los comandos incluyen `mkdir -p .orchestration`
3. Verificar formato JSON válido
4. Verificar que las rutas son correctas