# Retrospect — Developer Makefile
#
# Usage:
#   make up             — Build and start the full stack (production image).
#   make dev            — Build and start with hot-reload (dev image + override).
#   make down           — Stop all services (preserves volumes).
#   make down-clean     — Stop all services and delete all volumes.
#   make logs           — Tail api service logs.
#   make lint           — Run ruff check + mypy.
#   make format         — Auto-format code with ruff.
#   make test           — Run unit tests (fast, no Ollama or live API needed).
#   make eval           — Run DeepEval + Ragas evaluation (needs running stack + Ollama).
#   make eval-baseline  — Same suite with the retrieval loop and temporal filters off.
#   make eval-treatment — Same suite with the defaults restored, for the A/B delta.
#   make clean          — Remove all local tooling caches.

.PHONY: up down down-clean dev logs lint format test eval eval-baseline eval-treatment clean

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
	@echo "Installing the DeepEval + Ragas harnesses inside the api container..."
	docker cp requirements-eval.txt retrospect-rag-api-1:/tmp/requirements-eval.txt
	docker exec retrospect-rag-api-1 pip install -q -r /tmp/requirements-eval.txt
	@echo "Running RAG evaluation tests..."
	docker exec -e DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE=1800 -e DEEPEVAL_TASK_GATHER_BUFFER_SECONDS_OVERRIDE=300 retrospect-rag-api-1 python -m pytest tests/test_rag_eval.py -m eval -v

# Ablation baseline: retrieval loop off, temporal filters not applied. The API
# service is recreated so the new settings are actually read, since they live in
# the server process and not in the pytest invocation. Run `make eval-treatment`
# afterwards to restore the defaults and score the full configuration.
eval-baseline:
	RETRIEVAL_MAX_ATTEMPTS=1 TEMPORAL_FILTERING_ENABLED=false docker compose up -d --force-recreate --no-deps api
	@echo "Waiting for the api service to report healthy..."
	@until [ "$$(docker inspect -f '{{.State.Health.Status}}' retrospect-rag-api-1 2>/dev/null)" = "healthy" ]; do sleep 2; done
	@$(MAKE) eval

eval-treatment:
	docker compose up -d --force-recreate --no-deps api
	@echo "Waiting for the api service to report healthy..."
	@until [ "$$(docker inspect -f '{{.State.Health.Status}}' retrospect-rag-api-1 2>/dev/null)" = "healthy" ]; do sleep 2; done
	@$(MAKE) eval

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov .coverage
