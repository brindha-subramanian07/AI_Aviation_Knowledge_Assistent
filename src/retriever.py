"""
Retriever Module for AI-Powered Aviation Knowledge Assistant
Performs semantic query search, similarity thresholding, deduplication,
and context assembly for RAG generation.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from src.config import DEFAULT_TOP_K, DEFAULT_RELEVANCE_DISTANCE_THRESHOLD
from src.embeddings import EmbeddingService, get_embedding_service
from src.vector_store import VectorStore, get_vector_store


@dataclass
class RetrievedChunk:
    """Represents a single retrieved chunk with provenance and relevance score."""
    chunk_id: str
    document_id: str
    source: str
    page: int
    document_type: str
    text: str
    distance: float
    similarity_score: float  # Computed as max(0.0, 1.0 - distance)

    def to_citation(self) -> str:
        """Formatted human-readable citation string."""
        return f"{self.source} — Page {self.page}"


@dataclass
class RetrievalResult:
    """Aggregated retrieval payload passed to the RAG engine."""
    query: str
    chunks: List[RetrievedChunk]
    unique_sources: List[Dict[str, Any]]
    context_text: str
    has_results: bool


class KnowledgeRetriever:
    """Orchestrates query vectorization and semantic search across the knowledge base."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedding_service: Optional[EmbeddingService] = None,
        default_top_k: int = DEFAULT_TOP_K,
        distance_threshold: float = DEFAULT_RELEVANCE_DISTANCE_THRESHOLD
    ):
        self.vector_store = vector_store or get_vector_store()
        self.embedding_service = embedding_service or get_embedding_service()
        self.default_top_k = default_top_k
        self.distance_threshold = distance_threshold

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        document_id_filter: Optional[str] = None
    ) -> RetrievalResult:
        """
        Executes semantic retrieval for a query string:
        1. Encodes query into dense vector embedding
        2. Queries ChromaDB for top_k nearest neighbors
        3. Applies cosine distance threshold
        4. Deduplicates near-identical chunks
        5. Assembles citations and context text
        """
        clean_query = query.strip()
        if not clean_query:
            return RetrievalResult(
                query=query,
                chunks=[],
                unique_sources=[],
                context_text="",
                has_results=False
            )

        k = top_k if top_k is not None else self.default_top_k

        # Check if vector store has any records
        stats = self.vector_store.get_collection_stats()
        if stats.get("total_chunks", 0) == 0:
            return RetrievalResult(
                query=query,
                chunks=[],
                unique_sources=[],
                context_text="",
                has_results=False
            )

        # 1. Generate query embedding
        query_embedding = self.embedding_service.encode_query(clean_query)

        # 2. Query vector store
        where_filter = None
        if document_id_filter:
            where_filter = {"document_id": document_id_filter}

        query_res = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=k,
            where_filter=where_filter
        )

        ids_list = query_res.get("ids", [[]])[0]
        docs_list = query_res.get("documents", [[]])[0]
        metas_list = query_res.get("metadatas", [[]])[0]
        distances_list = query_res.get("distances", [[]])[0]

        retrieved_chunks: List[RetrievedChunk] = []
        seen_texts = set()

        # 3. Filter by distance threshold and deduplicate
        for i in range(len(ids_list)):
            chunk_id = ids_list[i]
            text = docs_list[i] if i < len(docs_list) else ""
            meta = metas_list[i] if i < len(metas_list) else {}
            dist = float(distances_list[i]) if i < len(distances_list) else 1.0

            # Cosine distance: 0 is exact match, 1 is orthogonal, >1 is divergent
            if dist > self.distance_threshold:
                continue

            # Deduplication: normalized text prefix comparison
            normalized_snippet = text.strip()[:150].lower()
            if normalized_snippet in seen_texts:
                continue
            seen_texts.add(normalized_snippet)

            # Similarity score: 1.0 - distance
            similarity = max(0.0, round(1.0 - dist, 4))

            chunk = RetrievedChunk(
                chunk_id=chunk_id,
                document_id=meta.get("document_id", "unknown"),
                source=meta.get("source", "Unknown Document"),
                page=int(meta.get("page", 1)),
                document_type=meta.get("document_type", "UNKNOWN"),
                text=text,
                distance=dist,
                similarity_score=similarity
            )
            retrieved_chunks.append(chunk)

        # 4. Extract unique source references
        unique_sources_map = {}
        for c in retrieved_chunks:
            key = (c.source, c.page)
            if key not in unique_sources_map:
                unique_sources_map[key] = {
                    "source": c.source,
                    "page": c.page,
                    "document_id": c.document_id,
                    "document_type": c.document_type,
                    "highest_similarity": c.similarity_score
                }
            else:
                if c.similarity_score > unique_sources_map[key]["highest_similarity"]:
                    unique_sources_map[key]["highest_similarity"] = c.similarity_score

        unique_sources = list(unique_sources_map.values())

        # 5. Format prompt-ready context text
        context_blocks = []
        for idx, c in enumerate(retrieved_chunks, 1):
            block = (
                f"[DOCUMENT CHUNK {idx}]\n"
                f"Source: {c.source}\n"
                f"Page: {c.page}\n"
                f"Content:\n{c.text}\n"
            )
            context_blocks.append(block)

        context_text = "\n---\n".join(context_blocks)

        return RetrievalResult(
            query=clean_query,
            chunks=retrieved_chunks,
            unique_sources=unique_sources,
            context_text=context_text,
            has_results=len(retrieved_chunks) > 0
        )


def get_retriever() -> KnowledgeRetriever:
    """Helper factory to retrieve KnowledgeRetriever instance."""
    return KnowledgeRetriever()
