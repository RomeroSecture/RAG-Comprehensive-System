---
allowed-tools: ["Bash", "Read", "Write", "Edit", "TodoWrite", "Grep", "LS"]
description: "Gestiona el repositorio GitHub siguiendo mejores prácticas: commits, branches, PRs, issues y releases"
---

# Gestión del Repositorio GitHub

Gestiona automáticamente el repositorio GitHub con mejores prácticas.

**Acción solicitada**: $ARGUMENTS

## 🔍 Análisis del Estado Actual

1. **Usar Bash** para verificar estado de Git:
   ```bash
   git status --porcelain
   git branch --show-current
   git remote -v
   ```

2. **Usar Bash** para verificar cambios pendientes:
   ```bash
   git diff --stat
   git diff --cached --stat
   ```

## 📋 Gestión de Branches

### Crear Nueva Branch
**Si la acción es "create-branch"**:

1. **Usar Bash** para crear branch con naming convention:
   ```bash
   # Determinar tipo de branch basado en cambios
   BRANCH_TYPE="feature"  # o fix, docs, refactor
   BRANCH_NAME="${BRANCH_TYPE}/$(date +%Y%m%d)-description"
   git checkout -b "$BRANCH_NAME"
   ```

### Cambiar de Branch
**Si la acción contiene branch name**:

1. **Usar Bash** para cambiar branch de forma segura:
   ```bash
   git stash push -m "Auto-stash before branch change"
   git checkout [branch-name]
   git stash pop || echo "No stash to restore"
   ```

## 💾 Gestión de Commits

### Auto-Commit Inteligente
**Si la acción es "commit" o "auto-commit"**:

1. **Usar Bash** para analizar cambios:
   ```bash
   # Detectar tipo de cambios
   git diff --name-only | head -10
   ```

2. **Determinar tipo de commit** basado en archivos cambiados:
   - src/domain/ → "feat(domain): "
   - tests/ → "test: "
   - docs/ → "docs: "
   - .gitignore, requirements.txt → "chore: "

3. **Usar Bash** para hacer commit con mensaje convencional:
   ```bash
   git add .
   git commit -m "feat(rag): implementar entidades de dominio
   
   - Añadir Document, Chunk, Query entities
   - Implementar tests unitarios
   - Seguir principios Clean Architecture"
   ```

### Commit con Mensaje Específico
**Si la acción contiene mensaje**:

1. **Usar Bash** para commit con mensaje proporcionado:
   ```bash
   git add .
   git commit -m "$PROVIDED_MESSAGE"
   ```

## 🔄 Sincronización Remota

### Push Changes
**Si la acción es "push"**:

1. **Usar Bash** para push seguro:
   ```bash
   CURRENT_BRANCH=$(git branch --show-current)
   git push -u origin "$CURRENT_BRANCH"
   ```

### Pull Latest Changes
**Si la acción es "pull" o "sync"**:

1. **Usar Bash** para pull con rebase:
   ```bash
   git pull --rebase origin main
   ```

## 🔀 Gestión de Pull Requests

### Crear PR Automático
**Si la acción es "create-pr"**:

1. **Usar Bash** para determinar cambios desde main:
   ```bash
   git log main..HEAD --oneline
   git diff main..HEAD --stat
   ```

2. **Usar Bash** para crear PR usando GitHub CLI:
   ```bash
   gh pr create \
     --title "$(git log -1 --pretty=format:'%s')" \
     --body "$(cat <<EOF
   ## 📝 Descripción
   $(git log main..HEAD --oneline)
   
   ## 🧪 Testing
   - [ ] Tests unitarios passing
   - [ ] Tests de integración passing
   - [ ] Coverage > 90%
   
   ## ✅ Checklist
   - [ ] Sigue Clean Architecture
   - [ ] Documentación actualizada
   - [ ] Sin breaking changes
   EOF
   )" \
     --draft
   ```

## 🏷️ Gestión de Tags y Releases

### Crear Release
**Si la acción es "release"**:

1. **Usar Bash** para determinar próxima versión:
   ```bash
   LAST_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "v0.0.0")
   # Incrementar versión patch automáticamente
   ```

2. **Usar Bash** para crear tag y release:
   ```bash
   NEW_VERSION="v0.1.0"
   git tag -a "$NEW_VERSION" -m "Release $NEW_VERSION"
   git push origin "$NEW_VERSION"
   
   gh release create "$NEW_VERSION" \
     --title "Release $NEW_VERSION" \
     --notes "$(git log ${LAST_TAG}..HEAD --oneline)"
   ```

## 🐛 Gestión de Issues

### Crear Issue Automático
**Si la acción es "create-issue"**:

1. **Usar Bash** para crear issue basado en TODOs:
   ```bash
   gh issue create \
     --title "Implementar [feature/bugfix]" \
     --body "$(cat <<EOF
   ## 📋 Descripción
   [Descripción automática basada en contexto]
   
   ## 🎯 Criterios de Aceptación
   - [ ] Criterio 1
   - [ ] Criterio 2
   
   ## 🔗 Referencias
   - Relacionado con: #[issue-number]
   EOF
   )" \
     --label "enhancement"
   ```

## 📊 Reportes y Estadísticas

### Generar Reporte de Actividad
**Si la acción es "report"**:

1. **Usar Bash** para generar estadísticas:
   ```bash
   echo "## 📊 Reporte del Repositorio"
   echo "- Commits esta semana: $(git log --since='1 week ago' --oneline | wc -l)"
   echo "- PRs abiertas: $(gh pr list --state open | wc -l)"
   echo "- Issues abiertas: $(gh issue list --state open | wc -l)"
   ```

2. **Usar Write** para guardar reporte en `.orchestration/reports/`

## 🔧 Configuración y Hooks

### Setup de Repositorio
**Si la acción es "setup"**:

1. **Usar Write** para crear `.gitignore`:
   ```
   __pycache__/
   *.pyc
   .env
   .venv/
   .pytest_cache/
   htmlcov/
   .coverage
   ```

2. **Usar Write** para crear templates de PR e Issue en `.github/`

3. **Usar Bash** para configurar hooks:
   ```bash
   # Configurar pre-commit hooks
   pre-commit install
   ```

## ✅ Validación Final

**Después de cada operación**:

1. **Usar Bash** para verificar estado:
   ```bash
   git status
   git log --oneline -3
   ```

2. **Usar TodoWrite** para actualizar progreso

3. **Usar Write** para registrar acción en log de orquestación

---

**Ejecuta este comando con la acción específica como argumento para gestionar automáticamente el repositorio GitHub.**