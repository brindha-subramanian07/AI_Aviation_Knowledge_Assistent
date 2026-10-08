"""
Document Manager Module for AI-Powered Aviation Knowledge Assistant
Orchestrates document uploading, indexing registry, re-indexing, deletion,
and lifecycle tracking.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.chunker import TextChunker
from src.config import DOCUMENTS_METADATA_FILE, PROCESSED_DIR, UPLOADS_DIR
from src.document_processor import DocumentProcessor, ProcessedDocument
from src.embeddings import EmbeddingService, get_embedding_service
from src.vector_store import VectorStore, get_vector_store
from utils.helpers import format_file_size, get_current_timestamp


@dataclass
class DocumentRecord:
    """Metadata record for an uploaded document."""
    document_id: str
    filename: str
    document_type: str
    file_size_bytes: int
    file_size_formatted: str
    total_pages: int
    total_chunks: int
    raw_hash: str
    uploaded_at: str
    status: str  # "Indexed", "Indexed (OCR Warning)", "Error"
    file_path: str
    ocr_warning: bool = False
    warning_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentRecord":
        return cls(**data)


class DocumentManager:
    """Coordinates file storage, document extraction, vectorization, and registry."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedding_service: Optional[EmbeddingService] = None,
        registry_path: Path = DOCUMENTS_METADATA_FILE
    ):
        self.vector_store = vector_store or get_vector_store()
        self.embedding_service = embedding_service or get_embedding_service()
        self.processor = DocumentProcessor()
        self.chunker = TextChunker()
        self.registry_path = registry_path
        self._ensure_registry()

    def _ensure_registry(self):
        """Initializes empty JSON registry if not present."""
        if not self.registry_path.exists():
            self._save_registry({})

    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """Loads document metadata registry from disk."""
        try:
            if self.registry_path.exists():
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _save_registry(self, data: Dict[str, Dict[str, Any]]):
        """Saves document metadata registry to disk."""
        try:
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving registry: {e}")

    def list_documents(self) -> List[DocumentRecord]:
        """Returns all registered documents sorted by upload date descending."""
        registry = self._load_registry()
        records = []
        for doc_id, doc_dict in registry.items():
            try:
                records.append(DocumentRecord.from_dict(doc_dict))
            except Exception:
                continue
        # Sort newest first
        records.sort(key=lambda r: r.uploaded_at, reverse=True)
        return records

    def get_document(self, document_id: str) -> Optional[DocumentRecord]:
        """Retrieves a single document record by ID."""
        registry = self._load_registry()
        doc_dict = registry.get(document_id)
        if doc_dict:
            return DocumentRecord.from_dict(doc_dict)
        return None

    def upload_and_index(
        self,
        file_bytes: bytes,
        filename: str,
        allow_reindex: bool = False
    ) -> Tuple[bool, str, Optional[DocumentRecord]]:
        """
        Executes the end-to-end indexing pipeline:
        1. Validates and extracts text from file
        2. Checks for duplicates via content hash
        3. Chunks text into semantic overlapping sections
        4. Generates dense embeddings
        5. Stores in persistent ChromaDB collection
        6. Persists raw file to data/uploads/ and updates registry
        """
        try:
            # 1. Extract and process
            processed: ProcessedDocument = self.processor.process_file(file_bytes, filename)
            doc_id = processed.document_id

            # 2. Duplicate detection
            registry = self._load_registry()
            if doc_id in registry and not allow_reindex:
                existing = DocumentRecord.from_dict(registry[doc_id])
                return False, f"Document '{filename}' is already indexed with ID {doc_id}.", existing

            # 3. Chunking
            chunks = self.chunker.chunk_document(processed)
            if not chunks:
                if processed.ocr_warning:
                    msg = processed.warning_message or "Document appears to be scanned and has no extractable text."
                    return False, f"Cannot index: {msg}", None
                return False, f"No extractable text chunks found in '{filename}'.", None

            # 4. If reindexing, clean existing chunks first
            if self.vector_store.has_document(doc_id):
                self.vector_store.delete_document(doc_id)

            # 5. Generate embeddings
            chunk_texts = [c.text for c in chunks]
            embeddings = self.embedding_service.encode_texts(chunk_texts, batch_size=32)

            # 6. Store in ChromaDB
            self.vector_store.add_chunks(chunks, embeddings)

            # 7. Save raw upload to data/uploads/
            file_path = UPLOADS_DIR / f"{doc_id}_{filename}"
            with open(file_path, "wb") as f:
                f.write(file_bytes)

            # 8. Update registry
            status = "Indexed (OCR Warning)" if processed.ocr_warning else "Indexed"
            record = DocumentRecord(
                document_id=doc_id,
                filename=filename,
                document_type=processed.document_type,
                file_size_bytes=len(file_bytes),
                file_size_formatted=format_file_size(len(file_bytes)),
                total_pages=processed.total_pages,
                total_chunks=len(chunks),
                raw_hash=processed.raw_hash,
                uploaded_at=get_current_timestamp(),
                status=status,
                file_path=str(file_path),
                ocr_warning=processed.ocr_warning,
                warning_message=processed.warning_message
            )

            registry[doc_id] = record.to_dict()
            self._save_registry(registry)

            # Save processed chunks summary for quick preview
            preview_path = PROCESSED_DIR / f"{doc_id}_summary.json"
            with open(preview_path, "w", encoding="utf-8") as f:
                json.dump({
                    "document_id": doc_id,
                    "filename": filename,
                    "chunks": [c.to_metadata() for c in chunks]
                }, f, indent=2)

            return True, f"Successfully indexed '{filename}' ({len(chunks)} chunks, {processed.total_pages} pages).", record

        except Exception as e:
            return False, f"Failed to index '{filename}': {str(e)}", None

    def reindex_document(self, document_id: str) -> Tuple[bool, str]:
        """
        Re-indexes an existing document:
        1. Reads the saved raw file from disk
        2. Deletes old chunks from ChromaDB
        3. Re-extracts, re-chunks, re-embeds, and re-upserts
        """
        registry = self._load_registry()
        if document_id not in registry:
            return False, f"Document '{document_id}' not found in registry."

        doc_info = registry[document_id]
        file_path = Path(doc_info["file_path"])

        if not file_path.exists():
            return False, f"Original file for '{document_id}' not found on disk at {file_path}."

        try:
            with open(file_path, "rb") as f:
                file_bytes = f.read()

            success, msg, _ = self.upload_and_index(
                file_bytes=file_bytes,
                filename=doc_info["filename"],
                allow_reindex=True
            )
            return success, msg
        except Exception as e:
            return False, f"Error re-indexing '{document_id}': {str(e)}"

    def delete_document(self, document_id: str) -> Tuple[bool, str]:
        """
        Permanently deletes a document from the system:
        1. Deletes all vectors from ChromaDB
        2. Removes the raw file from data/uploads/
        3. Removes metadata from registry
        """
        registry = self._load_registry()
        if document_id not in registry:
            return False, f"Document '{document_id}' not found in registry."

        doc_info = registry[document_id]
        filename = doc_info.get("filename", document_id)

        try:
            # Delete from ChromaDB
            deleted_chunks = self.vector_store.delete_document(document_id)

            # Delete file on disk if exists
            file_path = Path(doc_info.get("file_path", ""))
            if file_path.exists():
                try:
                    file_path.unlink()
                except Exception:
                    pass

            # Delete processed summary if exists
            summary_path = PROCESSED_DIR / f"{document_id}_summary.json"
            if summary_path.exists():
                try:
                    summary_path.unlink()
                except Exception:
                    pass

            # Remove from registry
            del registry[document_id]
            self._save_registry(registry)

            return True, f"Successfully deleted '{filename}' ({deleted_chunks} chunks removed)."
        except Exception as e:
            return False, f"Error deleting document '{document_id}': {str(e)}"

    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Calculates system-wide dashboard statistics."""
        docs = self.list_documents()
        total_docs = len(docs)
        total_chunks = sum(d.total_chunks for d in docs)
        indexed_docs = sum(1 for d in docs if "Indexed" in d.status)
        last_updated = docs[0].uploaded_at if docs else "No documents yet"

        vector_stats = self.vector_store.get_collection_stats()

        return {
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "indexed_documents": indexed_docs,
            "last_updated": last_updated,
            "chroma_chunks": vector_stats.get("total_chunks", 0),
            "supported_types": "PDF, DOCX, TXT"
        }


def get_document_manager() -> DocumentManager:
    """Helper factory to retrieve shared DocumentManager instance."""
    return DocumentManager()
