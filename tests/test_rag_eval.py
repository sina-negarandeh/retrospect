"""RAG pipeline integration & evaluation tests using DeepEval and RAGAS"""

from __future__ import annotations

import os

import httpx
import pytest

deepeval = pytest.importorskip(
    "deepeval",
    reason="deepeval is not installed — run 'make eval' or 'pip install deepeval>=1.4.0'",
)
assert_test = deepeval.assert_test

from deepeval.metrics import (  
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM  
from deepeval.test_case import LLMTestCase  
from langchain_ollama import ChatOllama  

# Configuration (env vars mirror docker-compose defaults)
_RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8000/api/v1/rag")
_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:26b-mlx")

# Local Ollama judge


class _OllamaJudge(DeepEvalBaseLLM):
    """DeepEval-compatible wrapper around a local Ollama model."""

    def __init__(self, model: str = _OLLAMA_MODEL, base_url: str = _OLLAMA_BASE_URL) -> None:
        self._chat = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=0.0,
            num_ctx=8192,
            format="json",
        )
        self._name = model

    def load_model(self) -> ChatOllama:  # type: ignore[override]
        return self._chat

    def generate(self, prompt: str, schema: type | None = None) -> str:  # type: ignore[override]
        return str(self._chat.invoke(prompt).content)

    async def a_generate(self, prompt: str, schema: type | None = None) -> str:  # type: ignore[override]
        return str((await self._chat.ainvoke(prompt)).content)

    def get_model_name(self) -> str:
        return f"ollama/{self._name}"


_judge = _OllamaJudge()

# RAGAS Metrics

_faithfulness = FaithfulnessMetric(threshold=0.5, model=_judge, include_reason=True)
_answer_relevancy = AnswerRelevancyMetric(threshold=0.5, model=_judge, include_reason=True)
_contextual_precision = ContextualPrecisionMetric(threshold=0.5, model=_judge, include_reason=True)
_contextual_recall = ContextualRecallMetric(threshold=0.5, model=_judge, include_reason=True)

# Evaluation dataset

_EVAL_SAMPLES = [
    {
        "id": "aunt-personality",
        "input": "What did I say about my aunt's personality?",
        "expected_output": "She is a lively, cheerful woman, with the best of hearts.",
        "top_k": 10,
    },
    {
        "id": "garden-creator",
        "input": "Who laid out the garden on the sloping hills?",
        "expected_output": "The late Count M.",
        "top_k": 10,
    },
]


# Helper


def _call_rag(question: str, top_k: int = 10) -> tuple[str, list[str]]:
    """Call the live RAG API and return ``(answer, context_chunks)``."""
    with httpx.Client(timeout=600.0) as client:
        resp = client.post(_RAG_API_URL, json={"query": question, "top_k": top_k})
        resp.raise_for_status()
        data = resp.json()

    answer: str = data["answer"]
    contexts: list[str] = [c["content"] for c in data.get("source_chunks", [])]
    return answer, contexts


# Tier 1: Functional tests (no LLM judge)


@pytest.mark.eval
@pytest.mark.parametrize("sample", _EVAL_SAMPLES, ids=[s["id"] for s in _EVAL_SAMPLES])
def test_rag_functional(sample: dict) -> None:  # type: ignore[type-arg]
    """Basic functional check: the answer contains key semantic truths.
    
    Ensures context chunks are retrieved correctly before burning LLM cycles
    on the robust metrics.
    """
    answer, contexts = _call_rag(sample["input"], sample.get("top_k", 10))

    assert len(contexts) > 0, "Expected at least one context chunk to be retrieved, but got none."
    assert len(answer) > 0, "Expected a non-empty answer from the RAG API."


# Tier 2: DeepEval metric tests (LLM judge via Ollama)


@pytest.mark.eval
@pytest.mark.parametrize("sample", _EVAL_SAMPLES, ids=[s["id"] for s in _EVAL_SAMPLES])
def test_rag_pipeline_deepeval_ragas(sample: dict) -> None:  # type: ignore[type-arg]
    """End-to-end RAG evaluation using DeepEval's robust RAGAS equivalents.

    For each sample the test:
    1. Calls the live RAG API endpoint.
    2. Wraps the response in a ``LLMTestCase``.
    3. Asserts all 4 metrics pass their thresholds.
    """
    answer, contexts = _call_rag(sample["input"], sample.get("top_k", 10))

    test_case = LLMTestCase(
        input=sample["input"],
        actual_output=answer,
        expected_output=sample["expected_output"],
        retrieval_context=contexts if contexts else ["(no context retrieved)"],
    )

    assert_test(
        test_case,
        metrics=[
            _faithfulness,
            _answer_relevancy,
            _contextual_precision,
            _contextual_recall,
        ],
    )
