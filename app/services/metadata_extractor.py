from __future__ import annotations

import json
import logging

import mlflow
from langchain_ollama import ChatOllama
from pydantic import ValidationError
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.domain.models import JournalMetadata

logger = logging.getLogger(__name__)


class LLMMetadataExtractor:
    """Uses a local LLM to extract structured metadata from text."""

    def __init__(self) -> None:
        settings = get_settings()
        self.llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=0.0,
            format="json",
            num_ctx=8192,
        )

        from app.prompts import METADATA_EXTRACTION_PROMPT
        self.prompt_template = METADATA_EXTRACTION_PROMPT

    @mlflow.trace(name="Extract_Metadata", span_type="TOOL")
    async def extract_metadata(self, text: str) -> dict[str, list[str] | str]:
        if span := mlflow.get_current_active_span():
            span.set_inputs({"text_length": len(text)})

        prompt = self.prompt_template.format(text=text)

        try:
            async for attempt in AsyncRetrying(
                wait=wait_exponential(multiplier=1, min=2, max=10),
                stop=stop_after_attempt(3),
                reraise=True,
            ):
                with attempt:
                    response = await self.llm.ainvoke(prompt)
            content = str(response.content).strip()

            # Parse the JSON
            data = json.loads(content)

            # Validate against Pydantic model to ensure safe defaults and types
            validated = JournalMetadata(**data)
            result = validated.model_dump()

            # Log clean span outputs.
            if span := mlflow.get_current_active_span():
                span.set_outputs(result)

            return result

        except json.JSONDecodeError as e:
            logger.error("LLM did not return valid JSON for metadata extraction: %s", e)
        except ValidationError as e:
            logger.error("Extracted metadata did not match expected schema: %s", e)
        except Exception as e:
            logger.error("Error during LLM metadata extraction: %s", e)

        # Return empty safe defaults on failure
        return JournalMetadata().model_dump()
