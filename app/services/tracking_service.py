from __future__ import annotations

import logging

import mlflow
import mlflow.langchain
from packaging.version import Version

logger = logging.getLogger(__name__)

_REQUIRED = "2.17.2"
if Version(mlflow.__version__) < Version(_REQUIRED):
    raise RuntimeError(
        f"MLflow >= {_REQUIRED} is required (found {mlflow.__version__}). "
        "Run 'pip install -U mlflow' and restart."
    )

mlflow.langchain.autolog(disable=True)


class TrackingService:

    @staticmethod
    def record_request_metrics(
        *,
        session_id: str,
        total_latency_ms: float,
        llm_latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        model: str,
    ) -> None:
        try:
            mlflow.update_current_trace(
                tags={
                    "session_id": session_id,
                    "model": model,
                },
                attributes={
                    "total_latency_ms": total_latency_ms,
                    "llm_latency_ms": llm_latency_ms,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
            )
        except Exception:
            # Tracking must never crash the request path.
            logger.warning("MLflow trace update failed for session=%s", session_id, exc_info=True)
