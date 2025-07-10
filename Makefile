.PHONY: help setup dev test lint format security clean deploy

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Initial project setup
	poetry install
	pre-commit install
	docker-compose up -d postgres redis
	poetry run alembic upgrade head

dev: ## Run development server
	docker-compose up -d
	poetry run uvicorn src.presentation.api.main:app --reload --host 0.0.0.0 --port 8000

test: ## Run all tests with coverage
	poetry run pytest tests/ --cov=src --cov-report=html --cov-report=term --cov-fail-under=90

test-unit: ## Run unit tests only
	poetry run pytest tests/unit/ -v

test-integration: ## Run integration tests only
	poetry run pytest tests/integration/ -v

test-e2e: ## Run end-to-end tests only
	poetry run pytest tests/e2e/ -v

lint: ## Run linting checks
	poetry run flake8 src tests
	poetry run mypy src
	poetry run bandit -r src/

format: ## Format code with black and isort
	poetry run black src tests
	poetry run isort src tests

security: ## Run security checks
	poetry run safety check
	poetry run bandit -r src/

clean: ## Clean up temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +

migrate: ## Create and apply database migrations
	poetry run alembic revision --autogenerate -m "$(message)"
	poetry run alembic upgrade head

rollback: ## Rollback last migration
	poetry run alembic downgrade -1

docker-build: ## Build Docker images
	docker build -t rag-system:latest -f docker/Dockerfile.api .
	docker build -t rag-worker:latest -f docker/Dockerfile.worker .
	docker build -t rag-frontend:latest -f docker/Dockerfile.frontend ./frontend

docker-push: ## Push Docker images to registry
	docker tag rag-system:latest $(REGISTRY)/rag-system:latest
	docker push $(REGISTRY)/rag-system:latest

deploy: ## Deploy to Kubernetes
	kubectl apply -f k8s/base/
	kubectl apply -f k8s/overlays/$(ENV)/

logs: ## Show API logs
	docker-compose logs -f api

shell: ## Open Python shell with project context
	poetry run ipython

db-shell: ## Open database shell
	docker-compose exec postgres psql -U postgres -d rag_system

redis-cli: ## Open Redis CLI
	docker-compose exec redis redis-cli

monitoring: ## Start monitoring stack
	docker-compose -f docker-compose.monitoring.yml up -d