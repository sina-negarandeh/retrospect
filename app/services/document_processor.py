from __future__ import annotations

import logging
import uuid
from pathlib import Path

from huggingface_hub import hf_hub_download
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from tokenizers import Tokenizer

from app.config import get_settings
from app.domain.models import Chunk, Document

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Loads and chunks markdown documents."""

    def __init__(self) -> None:
        settings = get_settings()
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap

        logger.info(f"Loading tokenizer: {settings.tokenizer_model}")
        tokenizer_json_path = hf_hub_download(
            repo_id=settings.tokenizer_model,
            filename="tokenizer.json",
            token=settings.hf_token,
        )
        self._tokenizer = Tokenizer.from_file(tokenizer_json_path)
        logger.info("Tokenizer loaded successfully via tokenizers library.")

    def load_documents(self, data_dir: str | Path) -> list[Document]:
        """Load all markdown files from the specified directory."""
        directory = Path(data_dir)
        documents: list[Document] = []
        if not directory.exists() or not directory.is_dir():
            logger.warning(f"Data directory {directory} does not exist or is not a directory.")
            return documents

        for file_path in directory.glob("**/*.md"):
            try:
                content = file_path.read_text(encoding="utf-8")
                doc = Document(
                    id=file_path.name,
                    content=content,
                    metadata={
                        "source_path": str(file_path),
                        "date": file_path.stem,
                    },
                )
                documents.append(doc)
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")

        logger.info(f"Loaded {len(documents)} documents from {directory}")
        return documents

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split a document into chunks using Markdown structure and recursive character splitting."""
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        
        # Split by markdown headers
        md_docs = markdown_splitter.split_text(document.content)
        
        # Split by tokens recursively
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=lambda x: len(self._tokenizer.encode(x, add_special_tokens=False).ids),
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        
        split_docs = text_splitter.split_documents(md_docs)
        
        chunks = []
        for i, split_doc in enumerate(split_docs):
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document.id}-child-{i}"))
            
            # Combine Markdown headers into the metadata
            metadata = {
                **document.metadata,
                "chunk_index": i,
                "is_child": True,
                "parent_content": document.content
            }
            for key, val in split_doc.metadata.items():
                metadata[key] = val
                
            chunk = Chunk(
                id=chunk_id,
                document_id=document.id,
                content=split_doc.page_content,
                metadata=metadata,
            )
            chunks.append(chunk)
            
        return chunks

    def create_parent_sentinel(self, document: Document) -> Chunk:
        """Create a parent sentinel chunk to mark the document as fully ingested."""
        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document.id}-parent"))
        return Chunk(
            id=chunk_id,
            document_id=document.id,
            content="",  # Empty content for the sentinel
            metadata={
                **document.metadata,
                "is_parent_sentinel": True,
            },
        )

    def process_directory(self, data_dir: str | Path) -> list[Chunk]:
        """Load all documents from a directory and chunk them."""
        documents = self.load_documents(data_dir)
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        logger.info("Created %d chunks from %d documents.", len(all_chunks), len(documents))
        return all_chunks
