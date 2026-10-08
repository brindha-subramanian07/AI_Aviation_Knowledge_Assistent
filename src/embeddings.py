"""
Embeddings Generation Module for AI-Powered Aviation Knowledge Assistant
Uses sentence-transformers (all-MiniLM-L6-v2) to generate normalized vector embeddings
for semantic document indexing and query retrieval.
"""

from typing import List, Union
import numpy as np
from src.config import EMBEDDING_MODEL_NAME

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class EmbeddingService:
    """Singleton-style wrapper for SentenceTransformer embedding generation."""
    _instance = None
    _model = None

    def __new__(cls, model_name: str = EMBEDDING_MODEL_NAME):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            cls._instance.model_name = model_name
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        """Loads the sentence-transformers model into memory."""
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is not installed. Please install it using `pip install sentence-transformers`."
            )
        try:
            self._model = SentenceTransformer(self.model_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load embedding model '{self.model_name}': {str(e)}"
            )

    @property
    def dimension(self) -> int:
        """Returns the embedding vector dimensionality (e.g., 384 for all-MiniLM-L6-v2)."""
        if self._model is not None:
            return self._model.get_sentence_embedding_dimension()
        return 384

    def encode_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> List[List[float]]:
        """
        Generates normalized embedding vectors for a list of text strings.
        Normalizing embeddings ensures cosine similarity corresponds directly to dot product.
        """
        if not texts:
            return []

        if self._model is None:
            self._load_model()

        try:
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
            # Convert to standard Python float lists for ChromaDB compatibility
            return embeddings.tolist()
        except Exception as e:
            raise RuntimeError(f"Error generating embeddings for text batch: {str(e)}")

    def encode_query(self, query: str) -> List[float]:
        """Generates a normalized embedding vector for a single search query."""
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        if self._model is None:
            self._load_model()

        try:
            embedding = self._model.encode(
                query.strip(),
                normalize_embeddings=True,
                convert_to_numpy=True
            )
            return embedding.tolist()
        except Exception as e:
            raise RuntimeError(f"Error generating query embedding: {str(e)}")


def get_embedding_service(model_name: str = EMBEDDING_MODEL_NAME) -> EmbeddingService:
    """Helper factory to retrieve the shared EmbeddingService instance."""
    return EmbeddingService(model_name=model_name)
