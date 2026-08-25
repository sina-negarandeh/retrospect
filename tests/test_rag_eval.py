"""RAG pipeline integration & evaluation tests using DeepEval and RAGAS.

Two independent judge harnesses score the same live API responses:

* **DeepEval** asserts per-sample thresholds and fails the test with a
  per-metric reason, which is what you want in CI.
* **Ragas** scores the dataset as a whole and returns a score frame, which is
  what you want when comparing two retrieval configurations against each other.

Both are driven by a local Ollama judge, so nothing leaves the machine and no
OpenAI key is required. Each harness is guarded independently — a missing
library skips only its own tier.

Every test here is marked ``eval`` and therefore excluded from ``make test``;
run them with ``make eval`` against a live stack.
"""

from __future__ import annotations

import os
from functools import cache
from typing import Any

import httpx
import pytest
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Configuration (env vars mirror docker-compose defaults)
_RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8000/api/v1/rag")
_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:26b-mlx")
_OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "embeddinggemma:300m")

# A 26B model on consumer hardware is slow, and every metric is several judge
# calls per sample. Mirrors the DeepEval overrides set in the Makefile.
_JUDGE_TIMEOUT_S = int(os.getenv("EVAL_JUDGE_TIMEOUT_SECONDS", "1800"))

_THRESHOLD = 0.5


# Optional judge harnesses


try:
    import deepeval
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )
    from deepeval.models.base_model import DeepEvalBaseLLM
    from deepeval.test_case import LLMTestCase

    _DEEPEVAL_ERROR: str | None = None
except ImportError as exc:  # pragma: no cover - depends on the environment
    _DEEPEVAL_ERROR = str(exc)
    DeepEvalBaseLLM = object  # type: ignore[assignment, misc]

try:
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )
    from ragas.run_config import RunConfig

    _RAGAS_ERROR: str | None = None
except ImportError as exc:  # pragma: no cover - depends on the environment
    _RAGAS_ERROR = str(exc)

requires_deepeval = pytest.mark.skipif(
    _DEEPEVAL_ERROR is not None,
    reason=f"deepeval is not installed — run 'make eval' ({_DEEPEVAL_ERROR})",
)
requires_ragas = pytest.mark.skipif(
    _RAGAS_ERROR is not None,
    reason=f"ragas is not installed — run 'make eval' ({_RAGAS_ERROR})",
)


# Local Ollama judge


def _build_judge_llm() -> ChatOllama:
    """Judge model shared by both harnesses — deterministic, JSON-constrained."""
    return ChatOllama(
        model=_OLLAMA_MODEL,
        base_url=_OLLAMA_BASE_URL,
        temperature=0.0,
        num_ctx=8192,
        format="json",
    )


def _build_judge_embeddings() -> OllamaEmbeddings:
    """Embeddings for the metrics that need them (answer relevancy)."""
    return OllamaEmbeddings(model=_OLLAMA_EMBEDDING_MODEL, base_url=_OLLAMA_BASE_URL)


class _OllamaJudge(DeepEvalBaseLLM):  # type: ignore[misc, valid-type]
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


@cache
def _call_rag(question: str, top_k: int = 10) -> tuple[str, tuple[str, ...]]:
    """Call the live RAG API and return ``(answer, context_chunks)``.

    Cached because both judge harnesses score the same responses: generating
    them twice would double an already slow run and, worse, let the two tiers
    grade two different answers.
    """
    with httpx.Client(timeout=600.0) as client:
        resp = client.post(_RAG_API_URL, json={"query": question, "top_k": top_k})
        resp.raise_for_status()
        data = resp.json()

    answer: str = data["answer"]
    contexts = tuple(c["content"] for c in data.get("source_chunks", []))
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
@requires_deepeval
@pytest.mark.parametrize("sample", _EVAL_SAMPLES, ids=[s["id"] for s in _EVAL_SAMPLES])
def test_rag_pipeline_deepeval(sample: dict) -> None:  # type: ignore[type-arg]
    """Per-sample RAG evaluation using DeepEval's RAGAS-equivalent metrics.

    For each sample the test:
    1. Calls the live RAG API endpoint.
    2. Wraps the response in a ``LLMTestCase``.
    3. Asserts all 4 metrics pass their thresholds.
    """
    answer, contexts = _call_rag(sample["input"], sample.get("top_k", 10))

    judge = _OllamaJudge()
    test_case = LLMTestCase(
        input=sample["input"],
        actual_output=answer,
        expected_output=sample["expected_output"],
        retrieval_context=list(contexts) if contexts else ["(no context retrieved)"],
    )

    deepeval.assert_test(
        test_case,
        metrics=[
            FaithfulnessMetric(threshold=_THRESHOLD, model=judge, include_reason=True),
            AnswerRelevancyMetric(threshold=_THRESHOLD, model=judge, include_reason=True),
            ContextualPrecisionMetric(threshold=_THRESHOLD, model=judge, include_reason=True),
            ContextualRecallMetric(threshold=_THRESHOLD, model=judge, include_reason=True),
        ],
    )


# Tier 3: Ragas metric tests (dataset-level, LLM judge via Ollama)

# Ragas reports under its own metric names; these are the result-frame columns.
_RAGAS_THRESHOLDS = {
    "faithfulness": _THRESHOLD,
    "answer_relevancy": _THRESHOLD,
    "llm_context_precision_with_reference": _THRESHOLD,
    "context_recall": _THRESHOLD,
}


@pytest.mark.eval
@requires_ragas
def test_rag_pipeline_ragas() -> None:
    """Dataset-level RAG evaluation using the Ragas library itself.

    Where the DeepEval tier fails one sample at a time, this scores the whole
    set in one pass and asserts on the mean per metric — the number you would
    actually quote when comparing two retrieval configurations.
    """
    samples = []
    for sample in _EVAL_SAMPLES:
        answer, contexts = _call_rag(sample["input"], sample.get("top_k", 10))
        samples.append(
            SingleTurnSample(
                user_input=sample["input"],
                response=answer,
                retrieved_contexts=list(contexts) if contexts else ["(no context retrieved)"],
                reference=sample["expected_output"],
            )
        )

    # max_workers=1: a single local Ollama instance serialises requests anyway,
    # and concurrent judge calls only add timeout pressure.
    run_config = RunConfig(timeout=_JUDGE_TIMEOUT_S, max_workers=1, max_retries=2)

    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ],
        llm=LangchainLLMWrapper(_build_judge_llm(), run_config=run_config),
        embeddings=LangchainEmbeddingsWrapper(_build_judge_embeddings(), run_config=run_config),
        run_config=run_config,
        # Surface every metric's outcome in one report rather than aborting on
        # the first judge hiccup; unscored metrics are caught explicitly below.
        raise_exceptions=False,
        show_progress=False,
    )

    scores = result.to_pandas()
    print("\nRagas scores:\n" + scores[list(_RAGAS_THRESHOLDS)].to_string(index=False))

    failures: list[str] = []
    for metric, threshold in _RAGAS_THRESHOLDS.items():
        column: Any = scores[metric]
        if column.isna().all():
            # A silent all-NaN column would otherwise pass, since NaN < x is
            # False — an unscored metric must fail loudly, not vanish.
            failures.append(f"{metric}: judge produced no parseable score")
            continue
        mean_score = float(column.mean())
        if mean_score < threshold:
            failures.append(f"{metric}: {mean_score:.3f} < {threshold}")

    assert not failures, "Ragas thresholds not met — " + "; ".join(failures)
