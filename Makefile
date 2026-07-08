# Retrospect — Developer Makefile
#
# Usage:
#   make up         — Build and start the full stack (production image).
#   make dev        — Build and start with hot-reload (dev image + override).
#   make down       — Stop all services (preserves volumes).
#   make down-clean — Stop all services and delete all volumes.
#   make logs       — Tail api service logs.
#   make lint       — Run ruff check + mypy.
#   make format     — Auto-format code with ruff.
#   make test       — Run unit tests (fast, no Ollama or live API needed).
#   make eval       — Run RAG evaluation tests (requires running stack + Ollama).
#   make clean      — Remove all local tooling caches.

.PHONY: up down down-clean dev logs lint format test eval clean

up:
	docker compose up -d --build

down:
	docker compose down

down-clean:
	docker compose down -v

dev:
	docker compose -f docker-compose.yml -f docker-compose.override.yml up --build

logs:
	docker compose logs -f api

lint:
	ruff check app/ tests/ && mypy app/

format:
	ruff format app/ tests/

test:
	docker exec retrospect-rag-api-1 python -m pytest tests/ -m 'not eval' -v

eval:
	@echo "Installing deepeval and ragas inside the api container..."
	docker exec retrospect-rag-api-1 pip install -q 'pyarrow<21.0.0' 'deepeval>=1.4.0' 'ragas>=0.1.7'
	@echo "Running RAG evaluation tests..."
	docker exec -e DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE=1800 -e DEEPEVAL_TASK_GATHER_BUFFER_SECONDS_OVERRIDE=300 retrospect-rag-api-1 python -m pytest tests/test_rag_eval.py -m eval -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov .coverage
