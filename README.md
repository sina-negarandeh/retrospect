# Retrospect RAG

This project was built as a production-grade ML engineering demonstration, showcasing end-to-end RAG system design from ingestion pipeline architecture through hybrid retrieval, cross-encoder reranking, and LLM-as-judge evaluation.

Retrospect is a modular RAG system for personal journals and diaries.

This project demonstrates a robust architecture for securely ingesting private, first-person entries, zero-shot structured extraction of JSON metadata (emotions, topics, people, places), and orchestrating empathetic LLM generation. Everything is wrapped within a scalable FastAPI service and fully containerized using Docker.

## Architecture: Offline & Online Phases

The system is cleanly decoupled into two distinct operational phases:

### 1. Offline Phase (Data Ingestion & Indexing)
- **Document Processing**: Parses raw diary text files, performing structural and token-based chunking.
- **LLM Metadata Extraction**: Uses a local model (`gemma4:26b-mlx` via Ollama) for zero-shot structured extraction of JSON metadata tracking emotions, sentiment, people, and locations.
- **Embedding Generation**: Converts diary chunks into dense vector representations using `embeddinggemma` via Ollama, and sparse SPLADE vectors via `fastembed`.
- **Vector Storage**: Indexes dense vectors, sparse SPLADE vectors, and extracted metadata filters into **Qdrant** for hybrid search retrieval.

### 2. Online Phase (Query & Generation)
- **Query Rewriting**: Intercepts natural language questions, stripping conversational filler and enriching the core concepts with probable diary synonyms (e.g., from "sad" to "crying, heartbroken") while identifying deterministic metadata filters.
- **4-Path Hybrid Retrieval**: Executes dense and sparse searches across both filtered and unfiltered result sets in a single `query_points` call, with Qdrant-native RRF fusion, eliminating multiple round-trips and pushing all fusion logic to the database layer.
- **Small-to-Big Retrieval**: Retrieves highly specific, smaller chunks via hybrid search, but feeds their larger, complete parent documents to the LLM to provide broader context, deduplicating parent IDs on the fly.
- **Cross-Encoder Re-ranking**: Utilizes a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) to re-rank the fused top results for maximum relevance against the rewritten query.

## Technology Stack

- **Python 3.12**
- **Framework**: FastAPI
- **LLM/Orchestration**: LangChain & LangGraph
- **Inference**: Ollama
- **Embedding Model**: `embeddinggemma` (dense) + SPLADE via `fastembed` (sparse)
- **Generation Model**: `gemma4:26b-mlx`
- **Vector Store**: Qdrant
- **Observability**: MLflow
- **Containerization**: Docker & Docker Compose
- **Tooling**: Make, Pytest, Ruff, Mypy

## Project Structure

```text
.
├── app/                  # FastAPI application code
│   ├── api/              # API routers and endpoints
│   ├── config.py         # App configuration settings
│   ├── domain/           # Core domain models (Pydantic)
│   ├── graph/            # LangGraph pipeline definition
│   ├── main.py           # FastAPI application entry point
│   ├── prompts.py        # Centralized LLM instruction templates
│   └── services/         # Business logic and external service integrations
├── data/                 # Data directory for document ingestion
├── docker/               # Dockerfiles and container configurations
├── evals/                # Evaluation configurations and scripts
├── tests/                # Unit and integration tests
├── .env.example          # Example environment variables
├── docker-compose.yml    # Production Docker services
├── docker-compose.override.yml # Development Docker overrides (hot-reload)
├── Makefile              # Helper commands
└── pyproject.toml        # Python project dependencies and tool configuration
```

## Getting Started

### Prerequisites
- Docker and Docker Compose
- [Ollama](https://ollama.com/) installed and running on your host machine (default port: `11434`)
- Python 3.12 (if running locally without Docker)

### 1. Environment Setup

Copy the example environment file and fill in the required values:
```bash
cp .env.example .env
```

> **Note:** `HF_TOKEN` is used by the application during startup to dynamically download the `ms-marco-MiniLM-L-6-v2` Cross-Encoder model from Hugging Face for re-ranking.

### 2. Running the Application

You can use the provided `Makefile` to manage the application lifecycle:

**Production Mode:**
Builds and starts the full stack (API, Qdrant, MLflow) in detached mode.
```bash
make up
```

**Development Mode (Hot-Reload):**
Builds and starts the stack using the dev override file, which enables live code reloading for the FastAPI service.
```bash
make dev
```

### 3. Stopping the Services

To stop all services while preserving your data (Qdrant vectors, MLflow tracking info):
```bash
make down
```

To stop all services and **wipe all data volumes**:
```bash
make down-clean
```

## Testing & Evaluation

Run unit tests (fast, does not require live API/Ollama):
```bash
make test
```

Run comprehensive RAG evaluation tests (requires running the full stack + Ollama):
```bash
make eval
```

## Code Quality

Format code using Ruff:
```bash
make format
```

Run linting (Ruff) and type checking (Mypy):
```bash
make lint
```
