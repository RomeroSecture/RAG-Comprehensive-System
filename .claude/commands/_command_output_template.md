# 📋 Template de Salida Estándar para Comandos

Todos los comandos deben escribir su resultado en el siguiente formato JSON al finalizar:

```bash
# Al final de cada comando, escribir resultado:
cat > .orchestration/${COMMAND_NAME}_result.json << EOF
{
  "command": "${COMMAND_NAME}",
  "status": "success|failed|partial|blocked",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "duration_seconds": ${DURATION},
  "arguments": "$ARGUMENTS",
  "results": {
    "files_created": [],
    "files_modified": [],
    "tests_added": 0,
    "tests_passed": 0,
    "coverage": 0.0,
    "custom_metrics": {}
  },
  "errors": [],
  "warnings": [],
  "blockers": [],
  "next_recommended_action": {
    "command": "suggested-command",
    "reason": "why this command should run next",
    "priority": "high|medium|low"
  },
  "summary": "Brief description of what was accomplished"
}
EOF
```

## Ejemplo de implementación:

```bash
# Al inicio del comando
COMMAND_NAME="manage-github-repo"
START_TIME=$(date +%s)

# ... ejecutar lógica del comando ...

# Al final del comando
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Crear directorio si no existe
mkdir -p .orchestration

# Escribir resultado
cat > .orchestration/${COMMAND_NAME}_result.json << EOF
{
  "command": "${COMMAND_NAME}",
  "status": "success",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "duration_seconds": ${DURATION},
  "arguments": "$ARGUMENTS",
  "results": {
    "files_created": [],
    "files_modified": ["src/main.py", "tests/test_main.py"],
    "commits_created": 2,
    "pr_created": true,
    "pr_url": "https://github.com/user/repo/pull/123"
  },
  "errors": [],
  "warnings": [],
  "blockers": [],
  "next_recommended_action": {
    "command": "test-rag-pipeline",
    "reason": "New code committed, tests should be run",
    "priority": "high"
  },
  "summary": "Created 2 commits and opened PR #123"
}
EOF
```