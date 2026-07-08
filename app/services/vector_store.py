from __future__ import annotations

import asyncio
import logging
from typing import Any

import mlflow
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as rest
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.domain.models import Chunk
from app.services.embedding_service import get_embeddings_model, get_sparse_embeddings_model

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages Qdrant vector database operations asynchronously."""

    def __init__(self) -> None:
        settings = get_settings()
        self.collection_name = settings.qdrant_collection_name
        self.client = AsyncQdrantClient(url=settings.qdrant_url)
        self.embeddings_model = get_embeddings_model()
        self.sparse_embeddings_model = get_sparse_embeddings_model()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the client and ensure the collection exists.

        This must be called before using the store. It will fail hard
        if Qdrant is unreachable, preventing a zombie application state.
        """
        if self._initialized:
            return

        try:
            collections_response = await self.client.get_collections()
            collection_names = [c.name for c in collections_response.collections]

            if self.collection_name not in collection_names:
                logger.info("Creating Qdrant collection: %s", self.collection_name)

                # Determine embedding dimension
                dummy_embed = await self.embeddings_model.aembed_query("dummy")
                dimension = len(dummy_embed)

                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": rest.VectorParams(
                            size=dimension,
                            distance=rest.Distance.COSINE,
                        )
                    },
                    sparse_vectors_config={
                        "sparse": rest.SparseVectorParams()
                    },
                )
                logger.info(
                    "Collection %s created with dense dimension %d and sparse vectors.",
                    self.collection_name,
                    dimension,
                )

            self._initialized = True

        except Exception as e:
            logger.critical("Failed to connect to Qdrant or initialize collection: %s", e)
            raise RuntimeError(f"Qdrant initialization failed: {e}") from e

    async def count_documents(self) -> int:
        """Return the number of points in the Qdrant collection."""
        try:
            response = await self.client.count(collection_name=self.collection_name)
            return response.count
        except Exception:
            logger.exception("Error counting documents in Qdrant")
            raise

    async def get_existing_document_ids(self, expected_chunk_ids: list[str]) -> set[str]:
        """Check Qdrant for a list of chunk IDs and return the set of document IDs that exist."""
        if not expected_chunk_ids:
            return set()

        existing_doc_ids = set()
        settings = get_settings()
        scroll_batch_size = settings.qdrant_scroll_batch_size

        for i in range(0, len(expected_chunk_ids), scroll_batch_size):
            batch = expected_chunk_ids[i : i + scroll_batch_size]
            try:
                async for attempt in AsyncRetrying(
                    wait=wait_exponential(multiplier=1, min=2, max=10),
                    stop=stop_after_attempt(3),
                    reraise=True,
                ):
                    with attempt:
                        # with_vectors=False is correctly parsed in recent qdrant-client versions
                        res, _ = await self.client.scroll(
                            collection_name=self.collection_name,
                            scroll_filter=rest.Filter(must=[rest.HasIdCondition(has_id=batch)]),
                            with_payload=["document_id"],
                            with_vectors=False,
                            limit=len(batch),
                        )
                for point in res:
                    doc_id = point.payload.get("document_id") if point.payload else None
                    if doc_id:
                        existing_doc_ids.add(str(doc_id))
            except Exception:
                logger.exception("Could not retrieve existing batch from Qdrant")
                raise

        return existing_doc_ids

    async def upsert_chunks(self, chunks: list[Chunk]) -> None:
        """Embed and upsert chunks to the vector database using Delta Ingestion."""
        if not chunks:
            return

        # Delta Ingestion: Find which chunks actually need to be embedded
        existing_ids = set()
        chunk_ids = [chunk.id for chunk in chunks]
        settings = get_settings()
        scroll_batch_size = settings.qdrant_scroll_batch_size

        for i in range(0, len(chunk_ids), scroll_batch_size):
            batch = chunk_ids[i : i + scroll_batch_size]
            try:
                async for attempt in AsyncRetrying(
                    wait=wait_exponential(multiplier=1, min=2, max=10),
                    stop=stop_after_attempt(3),
                    reraise=True,
                ):
                    with attempt:
                        res, _ = await self.client.scroll(
                            collection_name=self.collection_name,
                            scroll_filter=rest.Filter(must=[rest.HasIdCondition(has_id=batch)]),
                            with_payload=False,
                            with_vectors=False,
                            limit=len(batch),
                        )
                existing_ids.update({str(p.id) for p in res})
            except Exception:
                logger.exception("Could not retrieve existing batch from Qdrant")
                raise

        new_chunks = [c for c in chunks if c.id not in existing_ids]

        if not new_chunks:
            logger.info("All chunks already exist in vector store. Skipping embedding.")
            return

        logger.info("Embedding and upserting %d new chunks...", len(new_chunks))

        enriched_texts = []
        for chunk in new_chunks:
            date = chunk.metadata.get("date", "")
            topics_raw = chunk.metadata.get("topics", [])
            emotions_raw = chunk.metadata.get("emotions", [])
            topics = ", ".join(topics_raw) if isinstance(topics_raw, list) else ""
            emotions = ", ".join(emotions_raw) if isinstance(emotions_raw, list) else ""
            
            header_parts = []
            if date:
                header_parts.append(f"Date: {date}")
            if topics:
                header_parts.append(f"Topics: {topics}")
            if emotions:
                header_parts.append(f"Emotions: {emotions}")
                
            header = " | ".join(header_parts)
            enriched_text = f"{header}\n\n{chunk.content}" if header else chunk.content
            enriched_texts.append(enriched_text)

        batch_size = settings.vector_store_batch_size
        dense_embeddings = []
        sparse_embeddings = []
        
        def _get_sparse(texts: list[str]) -> list[Any]:
            return list(self.sparse_embeddings_model.embed(texts))
            
        for i in range(0, len(enriched_texts), batch_size):
            batch_texts = enriched_texts[i : i + batch_size]
    
            batch_dense, batch_sparse = await asyncio.gather(
                self.embeddings_model.aembed_documents(batch_texts),
                asyncio.to_thread(_get_sparse, batch_texts),
            )
            
            dense_embeddings.extend(batch_dense)
            sparse_embeddings.extend(batch_sparse)

        points = [
            rest.PointStruct(
                id=chunk.id,
                vector={
                    "dense": dense_emb,
                    "sparse": rest.SparseVector(
                        indices=sparse_emb.indices.tolist(),
                        values=sparse_emb.values.tolist(),
                    )
                },
                payload={
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    **chunk.metadata,
                },
            )
            for chunk, dense_emb, sparse_emb in zip(new_chunks, dense_embeddings, sparse_embeddings, strict=False)
        ]

        for i in range(0, len(points), batch_size):
            batch_points = points[i : i + batch_size]
            async for attempt in AsyncRetrying(
                wait=wait_exponential(multiplier=1, min=2, max=10),
                stop=stop_after_attempt(3),
                reraise=True,
            ):
                with attempt:
                    await self.client.upsert(
                        collection_name=self.collection_name,
                        points=batch_points,
                    )
        logger.info("Upserted %d points into Qdrant.", len(points))

    @mlflow.trace(name="Qdrant_VectorSearch", span_type="RETRIEVER")
    async def similarity_search(
        self, query: str, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[Chunk]:
        """Search the vector database for chunks similar to the query."""
        try:
            # Generate both query vectors
            query_embedding = await self.embeddings_model.aembed_query(query)
            
            def _get_sparse_query(text: str) -> Any:
                return next(self.sparse_embeddings_model.query_embed(text))
            
            sparse_query = await asyncio.to_thread(_get_sparse_query, query)
            sparse_vector = rest.SparseVector(
                indices=sparse_query.indices.tolist(),
                values=sparse_query.values.tolist()
            )

            qdrant_filter = None
            if filters:
                must_conditions: list[Any] = []
                for list_field in ["topics", "people", "places", "emotions"]:
                    if filters.get(list_field):
                        must_conditions.append(
                            rest.FieldCondition(
                                key=list_field, match=rest.MatchAny(any=filters[list_field])
                            )
                        )
                if filters.get("sentiment"):
                    must_conditions.append(
                        rest.FieldCondition(
                            key="sentiment", match=rest.MatchValue(value=filters["sentiment"])
                        )
                    )
                if must_conditions:
                    qdrant_filter = rest.Filter(must=must_conditions)

            # Build prefetch pipelines for Hybrid Search with Filter Fusion
            prefetch_queries = []
            
            if qdrant_filter:
                # Add only filtered paths
                prefetch_queries.extend([
                    rest.Prefetch(query=query_embedding, using="dense", limit=top_k, filter=qdrant_filter),
                    rest.Prefetch(query=sparse_vector, using="sparse", limit=top_k, filter=qdrant_filter),
                ])
            else:
                # Add only unfiltered paths
                prefetch_queries.extend([
                    rest.Prefetch(query=query_embedding, using="dense", limit=top_k),
                    rest.Prefetch(query=sparse_vector, using="sparse", limit=top_k),
                ])

            search_result = await self.client.query_points(
                collection_name=self.collection_name,
                prefetch=prefetch_queries,
                query=rest.FusionQuery(fusion=rest.Fusion.RRF),
                limit=top_k,
            )

            chunks = []
            for hit in search_result.points:
                payload = hit.payload or {}
                document_id = payload.get("document_id", "unknown")

                content = payload.get("content", "")
                metadata = {k: v for k, v in payload.items() if k != "content"}
                metadata["score"] = hit.score

                chunks.append(
                    Chunk(
                        id=str(hit.id),
                        document_id=document_id,
                        content=content,
                        metadata=metadata,
                    )
                )

            return chunks
        except Exception:
            logger.exception("Error during similarity search")
            raise
