from __future__ import annotations

from fastembed import SparseTextEmbedding
from langchain_ollama import OllamaEmbeddings

from app.config import get_settings


def get_embeddings_model() -> OllamaEmbeddings:
    settings = get_settings()
    return OllamaEmbeddings(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embedding_model,
    )


def get_sparse_embeddings_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
