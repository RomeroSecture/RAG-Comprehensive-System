---
allowed-tools: ["Bash", "Write", "Read", "Edit", "TodoWrite"]
description: "Configura el entorno de desarrollo completo para el sistema RAG"
---

# Setup del Entorno de Desarrollo RAG

Configura automáticamente el entorno de desarrollo completo.

## 🔍 Verificación de Prerequisitos

1. **Usar Bash** para verificar Python:
   ```bash
   python --version
   ```

2. **Usar Bash** para verificar dependencias del sistema:
   ```bash
   which docker && echo "Docker disponible" || echo "Docker no encontrado"
   which git && echo "Git disponible" || echo "Git no encontrado"
   ```

## 📦 Configuración de Gestión de Dependencias

1. **Usar Write** para crear `pyproject.toml`:
   ```toml
   [tool.poetry]
   name = "rag-comprehensive-system"
   version = "0.1.0"
   description = "Sistema RAG con Clean Architecture"
   
   [tool.poetry.dependencies]
   python = "^3.11"
   fastapi = "^0.104.1"
   uvicorn = "^0.24.0"
   # ... resto de dependencias
   ```

2. **Usar Bash** para instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## 🏗️ Estructura de Proyecto

1. **Usar Bash** para crear estructura completa:
   ```bash
   mkdir -p src/{domain,application,infrastructure,presentation}
   mkdir -p tests/{unit,integration,e2e}
   mkdir -p docs/{architecture,api,guides}
   mkdir -p scripts/{dev,deploy,migration}
   ```

2. **Usar Write** para crear archivos `__init__.py` en cada directorio Python

## 🔧 Configuración de Herramientas de Desarrollo

**Linting y formateo**:
1. **Usar Write** para crear `.pre-commit-config.yaml`
2. **Usar Write** para crear `pyproject.toml` con configuración de ruff
3. **Usar Write** para crear `mypy.ini` para type checking

**Testing**:
1. **Usar Write** para crear `pytest.ini` con configuración de tests
2. **Usar Write** para crear `conftest.py` con fixtures compartidos

## 🐳 Configuración Docker

1. **Usar Write** para crear `Dockerfile`:
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   CMD ["uvicorn", "src.presentation.api.main:app", "--host", "0.0.0.0"]
   ```

2. **Usar Write** para crear `docker-compose.yml`:
   ```yaml
   version: '3.8'
   services:
     app:
       build: .
       ports:
         - "8000:8000"
     postgres:
       image: pgvector/pgvector:pg15
       environment:
         POSTGRES_DB: rag_db
         POSTGRES_USER: rag_user
         POSTGRES_PASSWORD: rag_pass
   ```

## ⚙️ Variables de Entorno

1. **Usar Write** para crear `.env.example`:
   ```env
   DATABASE_URL=postgresql://rag_user:rag_pass@localhost/rag_db
   OPENAI_API_KEY=your_openai_key_here
   EMBEDDING_MODEL=text-embedding-ada-002
   ```

2. **Usar Write** para crear `.env` (ignorado por git)

## 🗄️ Configuración de Base de Datos

1. **Usar Bash** para verificar PostgreSQL:
   ```bash
   docker-compose up -d postgres
   ```

2. **Usar Write** para crear script de migración inicial en `scripts/migration/001_initial.sql`

## 📋 Scripts de Desarrollo

1. **Usar Write** para crear `scripts/dev/start.sh`:
   ```bash
   #!/bin/bash
   docker-compose up -d postgres
   uvicorn src.presentation.api.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Usar Write** para crear `scripts/dev/test.sh`:
   ```bash
   #!/bin/bash
   python -m pytest tests/ -v --cov=src --cov-report=html
   ```

3. **Usar Bash** para hacer scripts ejecutables:
   ```bash
   chmod +x scripts/dev/*.sh
   ```

## ✅ Verificación Final

1. **Usar Bash** para probar instalación:
   ```bash
   python -c "import fastapi; print('FastAPI OK')"
   python -c "import asyncpg; print('AsyncPG OK')"
   ```

2. **Usar Bash** para ejecutar tests básicos:
   ```bash
   python -m pytest tests/ -v
   ```

3. **Usar TodoWrite** para marcar setup como completado

## 📝 Documentación

1. **Usar Write** para actualizar README.md con instrucciones de setup
2. **Usar Write** para crear CONTRIBUTING.md con guías de desarrollo

---

**Ejecuta este comando para configurar automáticamente todo el entorno de desarrollo necesario.**