"""
Vector Store Module for AI-Powered Aviation Knowledge Assistant
Manages persistent storage, indexing, deletion, and querying of vector embeddings
using ChromaDB.
"""

from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings
from src.chunker import DocumentChunk
from src.config import CHROMA_COLLECTION_NAME, CHROMA_DIR


class VectorStore:
    """Persistent ChromaDB client for aviation knowledge chunks."""

    def __init__(
        self,
        persist_directory: str = str(CHROMA_DIR),
        collection_name: str = CHROMA_COLLECTION_NAME
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._initialize()

    def _initialize(self):
        """Initializes persistent ChromaDB client and creates or gets collection."""
        try:
            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False, is_persistent=True)
            )
            # Use cosine distance space for normalized embeddings
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ChromaDB at '{self.persist_directory}': {str(e)}")

    @property
    def collection(self):
        """Returns the Chroma collection instance."""
        if self._collection is None:
            self._initialize()
        return self._collection

    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]]
    ) -> int:
        """
        Adds or upserts document chunks with their embeddings and metadata into ChromaDB.
        Returns the number of chunks added.
        """
        if not chunks:
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks provided but {len(embeddings)} embeddings provided."
            )

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [chunk.to_metadata() for chunk in chunks]

        try:
            # Upsert ensures that if chunks with the same ID exist, they are updated cleanly
            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            return len(ids)
        except Exception as e:
            raise RuntimeError(f"Error adding chunks to ChromaDB collection: {str(e)}")

    def delete_document(self, document_id: str) -> int:
        """
        Deletes all chunks belonging to a specific document ID.
        Returns the number of deleted chunks.
        """
        try:
            # First find matching chunk IDs
            existing = self.collection.get(
                where={"document_id": document_id},
                include=["metadatas"]
            )
            chunk_ids = existing.get("ids", [])
            if chunk_ids:
                self.collection.delete(ids=chunk_ids)
            return len(chunk_ids)
        except Exception as e:
            raise RuntimeError(f"Failed to delete document '{document_id}' from vector store: {str(e)}")

    def has_document(self, document_id: str) -> bool:
        """Checks if any chunks for this document already exist in ChromaDB."""
        try:
            result = self.collection.get(
                where={"document_id": document_id},
                limit=1
            )
            return len(result.get("ids", [])) > 0
        except Exception:
            return False

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 6,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Queries ChromaDB for the closest chunks to the query embedding.
        Returns Chroma query results containing ids, documents, metadatas, and distances.
        """
        try:
            total_items = self.collection.count()
            if total_items == 0:
                return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

            # Ensure top_k does not exceed total items in collection
            k = min(top_k, total_items)

            query_kwargs: Dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": k,
                "include": ["documents", "metadatas", "distances"]
            }
            if where_filter:
                query_kwargs["where"] = where_filter

            results = self.collection.query(**query_kwargs)
            return results
        except Exception as e:
            raise RuntimeError(f"Error querying vector store: {str(e)}")

    def get_collection_stats(self) -> Dict[str, Any]:
        """Returns statistics: total chunk count, indexed document IDs, and chunk counts."""
        try:
            total_chunks = self.collection.count()
            if total_chunks == 0:
                return {
                    "total_chunks": 0,
                    "document_chunk_counts": {},
                    "unique_documents_count": 0
                }

            # Retrieve metadatas to count chunks per document
            all_data = self.collection.get(include=["metadatas"])
            metadatas = all_data.get("metadatas", []) or []

            doc_counts: Dict[str, int] = {}
            for meta in metadatas:
                if meta and "document_id" in meta:
                    doc_id = meta["document_id"]
                    doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

            return {
                "total_chunks": total_chunks,
                "document_chunk_counts": doc_counts,
                "unique_documents_count": len(doc_counts)
            }
        except Exception as e:
            return {
                "total_chunks": 0,
                "document_chunk_counts": {},
                "unique_documents_count": 0,
                "error": str(e)
            }

    def get_chunks_for_document(self, document_id: str) -> List[Dict[str, Any]]:
        """Retrieves all chunks and text for a given document ID."""
        try:
            results = self.collection.get(
                where={"document_id": document_id},
                include=["documents", "metadatas"]
            )
            chunks = []
            ids = results.get("ids", [])
            docs = results.get("documents", [])
            metas = results.get("metadatas", [])

            for i in range(len(ids)):
                chunks.append({
                    "id": ids[i],
                    "text": docs[i] if i < len(docs) else "",
                    "metadata": metas[i] if i < len(metas) else {}
                })
            # Sort by chunk_index
            chunks.sort(key=lambda c: c["metadata"].get("chunk_index", 0))
            return chunks
        except Exception as e:
            raise RuntimeError(f"Error fetching chunks for document '{document_id}': {str(e)}")


def get_vector_store() -> VectorStore:
    """Helper factory to retrieve the shared VectorStore instance."""
    return VectorStore()
