"""
BGE-M3 Dense Embedder for Vietnamese legal documents.

Uses BAAI/bge-m3 via sentence-transformers to produce normalized dense embeddings
(float32, norm ≈ 1.0) for both legal corpus chunks and search queries.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "BAAI/bge-m3"


class BGEEmbedder:
    """
    Dense text embedder using the BAAI/bge-m3 model.

    Features:
      - Model cached at class level so the large weights are loaded only once.
      - Device auto-selection (CUDA if available, otherwise CPU).
      - Dynamic embedding dimension resolution (no hardcoding).
      - Strict L2 normalization to ensure inner product equals cosine similarity.
      - Validation against NaN and Inf values.
    """

    _cached_model = None
    _cached_model_name: Optional[str] = None
    _cached_device: Optional[str] = None

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
    ) -> None:
        """
        Initialize the BGEEmbedder.

        Args:
            model_name: Hugging Face model identifier (default: BAAI/bge-m3).
            device: Target torch device ('cpu', 'cuda', etc.). If None, auto-detects.
            normalize_embeddings: Whether to apply L2 normalization to output vectors.
        """
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings

        if device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = self._get_or_load_model(self.model_name, self.device)
        if hasattr(self.model, "get_embedding_dimension"):
            self._dimension = self.model.get_embedding_dimension()
        else:
            self._dimension = self.model.get_sentence_embedding_dimension()
        logger.info(
            "BGEEmbedder ready: model=%s, device=%s, dimension=%d",
            self.model_name,
            self.device,
            self._dimension,
        )

    @classmethod
    def _get_or_load_model(cls, model_name: str, device: str):
        """Load model once and cache across instances."""
        if (
            cls._cached_model is not None
            and cls._cached_model_name == model_name
            and cls._cached_device == device
        ):
            return cls._cached_model

        from sentence_transformers import SentenceTransformer

        logger.info("Loading SentenceTransformer model '%s' on %s...", model_name, device)
        model = SentenceTransformer(model_name, device=device)
        cls._cached_model = model
        cls._cached_model_name = model_name
        cls._cached_device = device
        return model

    @property
    def dimension(self) -> int:
        """Return the dynamic embedding dimension of the model."""
        return self._dimension

    @staticmethod
    def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
        """
        Apply explicit L2 normalization to a 2D numpy array.

        Args:
            vectors: 2D array of shape (N, D) with float32 dtype.

        Returns:
            Normalized 2D array where each row has L2 norm ≈ 1.0.
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero for all-zero vectors
        norms = np.where(norms == 0, 1.0, norms)
        normalized = vectors / norms
        return normalized.astype(np.float32)

    @staticmethod
    def _validate_embeddings(embeddings: np.ndarray) -> None:
        """Ensure embeddings have valid dtype and contain no NaN or Inf."""
        if embeddings.dtype != np.float32:
            raise ValueError(f"Expected float32 embeddings, got {embeddings.dtype}")
        if np.isnan(embeddings).any():
            raise ValueError("Embedding matrix contains NaN values")
        if np.isinf(embeddings).any():
            raise ValueError("Embedding matrix contains infinite (Inf) values")

    def encode_documents(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """
        Encode a list of document strings into dense vectors.

        Args:
            texts: List of text strings to embed.
            batch_size: Number of texts per forward pass.
            show_progress_bar: Whether to display sentence_transformers progress bar.

        Returns:
            2D numpy array of shape (len(texts), dimension), float32, normalized.
        """
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)

        raw_embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )

        embeddings = np.asarray(raw_embeddings, dtype=np.float32)

        if self.normalize_embeddings:
            embeddings = self._normalize_vectors(embeddings)

        self._validate_embeddings(embeddings)
        return embeddings

    def encode_queries(
        self,
        queries: Union[list[str], str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Encode one or more search queries into dense vectors.

        Args:
            queries: A single query string or list of query strings.
            batch_size: Batch size for encoding.

        Returns:
            2D numpy array of shape (N, dimension), float32, normalized.
        """
        if isinstance(queries, str):
            queries = [queries]

        if not queries:
            return np.empty((0, self._dimension), dtype=np.float32)

        raw_embeddings = self.model.encode(
            queries,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )

        embeddings = np.asarray(raw_embeddings, dtype=np.float32)

        if self.normalize_embeddings:
            embeddings = self._normalize_vectors(embeddings)

        self._validate_embeddings(embeddings)
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query string into a 2D numpy array of shape (1, dimension).

        Args:
            query: Query string.

        Returns:
            2D numpy array of shape (1, dimension), float32, normalized.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        return self.encode_queries([query])
