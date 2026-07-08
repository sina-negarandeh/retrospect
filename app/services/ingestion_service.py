from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from app.config import get_settings
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import get_embeddings_model
from app.services.metadata_extractor import LLMMetadataExtractor
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    total_found: int = 0
    already_existed: int = 0
    ingested: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0


async def _drop_and_recreate_collection() -> None:
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.http import models as rest

    settings = get_settings()
    client = AsyncQdrantClient(url=settings.qdrant_url)

    collections_response = await client.get_collections()
    existing = {c.name for c in collections_response.collections}
    if settings.qdrant_collection_name in existing:
        logger.info("Dropping collection: %s", settings.qdrant_collection_name)
        await client.delete_collection(settings.qdrant_collection_name)

    embeddings_model = get_embeddings_model()
    dummy_embed = await embeddings_model.aembed_query("dimension probe")
    dimension = len(dummy_embed)

    await client.create_collection(
        collection_name=settings.qdrant_collection_name,
        vectors_config={
            "dense": rest.VectorParams(
                size=dimension,
                distance=rest.Distance.COSINE,
            )
        },
        # Sparse vectors configuration required for hybrid search.
        sparse_vectors_config={
            "sparse": rest.SparseVectorParams()
        },
    )
    logger.info(
        "Collection %s recreated with dense dim=%d and sparse vectors.",
        settings.qdrant_collection_name,
        dimension,
    )


def _expected_chunk_ids_for_document(doc_id: str, num_chunks: int) -> list[str]:
    ids: list[str] = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}-parent"))]
    for i in range(num_chunks):
        ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}-child-{i}")))
    return ids


async def run_ingestion(
    *,
    limit: int | None = None,
    wipe_first: bool = False,
    data_dir: str | Path | None = None,
) -> IngestionResult:
    result = IngestionResult()

    resolved_dir = Path(data_dir) if data_dir is not None else Path("./data")

    try:
        if wipe_first:
            logger.info("wipe_first=True — dropping and recreating collection.")
            await _drop_and_recreate_collection()

        vector_store = VectorStore()
        await vector_store.initialize()
        # Initialize DocumentProcessor once for the ingestion batch
        processor = DocumentProcessor()

        # 1. Load raw documents
        documents = processor.load_documents(resolved_dir)
        result.total_found = len(documents)

        if not documents:
            logger.info("No documents found in %s.", resolved_dir)
            return result

        parent_sentinel_ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc.id}-parent"))
            for doc in documents
        ]
        existing_doc_ids = await vector_store.get_existing_document_ids(parent_sentinel_ids)
        missing_docs = [doc for doc in documents if doc.id not in existing_doc_ids]
        result.already_existed = len(documents) - len(missing_docs)

        if not missing_docs:
            logger.info("All documents already in vector store. Nothing to do.")
            return result

        # 3. Apply limit
        if limit is not None and limit > 0:
            result.skipped = max(0, len(missing_docs) - limit)
            missing_docs = missing_docs[:limit]
            logger.info(
                "limit=%d applied — processing %d of %d missing documents.",
                limit,
                len(missing_docs),
                len(missing_docs) + result.skipped,
            )

        logger.info("Starting metadata extraction for %d documents...", len(missing_docs))

        # 4. Extract metadata + embed + upsert
        extractor = LLMMetadataExtractor()
        for i, doc in enumerate(missing_docs):
            try:
                logger.info(
                    "Extracting metadata for %s (%d/%d)...",
                    doc.id,
                    i + 1,
                    len(missing_docs),
                )
                metadata = await extractor.extract_metadata(doc.content)
                doc.metadata.update(metadata)

                chunks = processor.chunk_document(doc)
                
                # Append the parent chunk to mark this document as complete.
                parent_chunk = processor.create_parent_sentinel(doc)
                if parent_chunk:
                    chunks.append(parent_chunk)

                await vector_store.upsert_chunks(chunks)

                logger.info("Upserted %s (%d chunks) into Qdrant.", doc.id, len(chunks))
                result.ingested += 1

            except Exception as doc_err:
                msg = f"Failed to ingest {doc.id}: {doc_err}"
                logger.error(msg, exc_info=True)
                result.errors.append(msg)

        logger.info(
            "Ingestion complete. ingested=%d skipped=%d errors=%d",
            result.ingested,
            result.skipped,
            len(result.errors),
        )

    except Exception as e:
        msg = f"Ingestion pipeline error: {e}"
        logger.error(msg, exc_info=True)
        result.errors.append(msg)

    return result
