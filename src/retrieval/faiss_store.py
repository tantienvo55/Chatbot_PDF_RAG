"""
FAISS Vector Store and Dense Retrieval for Vietnamese legal chunks.

Uses faiss.IndexFlatIP on normalized BGE-M3 vectors so Inner Product equals
cosine similarity. Provides deterministic index position to chunk_id mapping,
corpus integrity verification via SHA256, and validation routines.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np

from .bge_embedder import BGEEmbedder

logger = logging.getLogger(__name__)

INDEX_FILENAME = "faiss.index"
METADATA_FILENAME = "metadata.json"
MANIFEST_FILENAME = "index_manifest.json"


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 hex digest of a file in streaming chunks."""
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class FAISSStore:
    """
    Manages a dense FAISS IndexFlatIP store and position-to-metadata mapping.
    """

    def __init__(
        self,
        index: faiss.Index,
        metadata: list[dict[str, Any]],
        manifest: dict[str, Any],
        embedder: Optional[BGEEmbedder] = None,
    ) -> None:
        """
        Initialize FAISSStore with loaded components.

        Use `load()` or `build()` to instantiate.
        """
        self.index = index
        self.metadata = metadata
        self.manifest = manifest
        self._embedder = embedder

        self._validate_store()

    def _validate_store(self) -> None:
        """Validate internal consistency between index, metadata, and manifest."""
        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                f"FAISS index vector count ({self.index.ntotal}) does not match "
                f"metadata record count ({len(self.metadata)})"
            )

        expected_dim = self.manifest.get("embedding_dimension")
        if expected_dim is not None and self.index.d != expected_dim:
            raise ValueError(
                f"FAISS index dimension ({self.index.d}) does not match "
                f"manifest embedding dimension ({expected_dim})"
            )

        if self.index.ntotal == 0:
            raise ValueError("FAISS index is empty (0 vectors)")

    @property
    def embedder(self) -> BGEEmbedder:
        """Lazily initialize or return embedder."""
        if self._embedder is None:
            model_name = self.manifest.get("embedding_model", "BAAI/bge-m3")
            self._embedder = BGEEmbedder(model_name=model_name)
        return self._embedder

    def save(self, output_dir: Path) -> None:
        """
        Persist FAISS index, metadata, and index manifest to directory.

        Args:
            output_dir: Destination directory (e.g. data/vector_store).
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        index_path = output_dir / INDEX_FILENAME
        metadata_path = output_dir / METADATA_FILENAME
        manifest_path = output_dir / MANIFEST_FILENAME

        # Save FAISS index
        faiss.write_index(self.index, str(index_path))
        logger.info("Saved FAISS index to %s (%d vectors)", index_path, self.index.ntotal)

        # Save metadata (deterministic JSON array preserving index order)
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        logger.info("Saved metadata records to %s (%d items)", metadata_path, len(self.metadata))

        # Save manifest
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(self.manifest, f, ensure_ascii=False, indent=2)
        logger.info("Saved index manifest to %s", manifest_path)

    @classmethod
    def load(
        cls,
        store_dir: Path,
        embedder: Optional[BGEEmbedder] = None,
        expected_corpus_path: Optional[Path] = None,
    ) -> FAISSStore:
        """
        Load an existing FAISS store from disk.

        Args:
            store_dir: Directory containing faiss.index, metadata.json, index_manifest.json.
            embedder: Optional pre-loaded BGEEmbedder instance.
            expected_corpus_path: Optional path to verify current corpus SHA256 against manifest.

        Returns:
            Instantiated and verified FAISSStore.
        """
        index_path = store_dir / INDEX_FILENAME
        metadata_path = store_dir / METADATA_FILENAME
        manifest_path = store_dir / MANIFEST_FILENAME

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index file not found: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        if not manifest_path.exists():
            raise FileNotFoundError(f"Index manifest file not found: {manifest_path}")

        # Load index
        index = faiss.read_index(str(index_path))

        # Load metadata
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Load manifest
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        # Check corpus hash if requested or present
        if expected_corpus_path and expected_corpus_path.exists():
            current_hash = compute_file_sha256(expected_corpus_path)
            manifest_hash = manifest.get("corpus_sha256")
            if manifest_hash and current_hash != manifest_hash:
                logger.warning(
                    "Corpus hash mismatch: legal_chunks.json has changed since index creation! "
                    "(manifest: %s..., current: %s...). Run index rebuild when convenient.",
                    manifest_hash[:12],
                    current_hash[:12],
                )

        return cls(index=index, metadata=metadata, manifest=manifest, embedder=embedder)

    @classmethod
    def build(
        cls,
        chunks: list[dict[str, Any]],
        embedder: BGEEmbedder,
        corpus_path: Optional[Path] = None,
        batch_size: int = 32,
        show_progress_bar: bool = True,
    ) -> FAISSStore:
        """
        Build a new FAISSStore from a list of legal chunk dictionaries.

        Args:
            chunks: List of LegalChunk dicts from legal_chunks.json.
            embedder: Initialized BGEEmbedder instance.
            corpus_path: Optional path to the source legal_chunks.json for hashing.
            batch_size: Embedding batch size.
            show_progress_bar: Whether to show progress bar during embedding.

        Returns:
            Ready FAISSStore instance.
        """
        if not chunks:
            raise ValueError("Cannot build FAISS store from empty chunks list")

        texts: list[str] = []
        for c in chunks:
            # Prefer content_with_context, fallback to content
            text = c.get("content_with_context") or c.get("content", "")
            texts.append(text)

        logger.info("Encoding %d chunks with BGE-M3 (batch_size=%d)...", len(texts), batch_size)
        embeddings = embedder.encode_documents(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )

        dim = embedder.dimension
        if embeddings.shape[1] != dim:
            raise ValueError(f"Embedding dimension mismatch: expected {dim}, got {embeddings.shape[1]}")

        # Create IndexFlatIP (exact cosine similarity on normalized vectors)
        logger.info("Building FAISS IndexFlatIP (dim=%d, count=%d)...", dim, len(chunks))
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        # Corpus hash
        corpus_sha256 = compute_file_sha256(corpus_path) if corpus_path and corpus_path.exists() else None

        manifest = {
            "embedding_model": embedder.model_name,
            "index_type": "IndexFlatIP",
            "normalized": True,
            "embedding_dimension": dim,
            "num_vectors": index.ntotal,
            "source_file": str(corpus_path.as_posix()) if corpus_path else "data/chunks/legal_chunks.json",
            "corpus_sha256": corpus_sha256,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Keep metadata in exact same order as vectors
        metadata = list(chunks)

        return cls(index=index, metadata=metadata, manifest=manifest, embedder=embedder)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Perform dense top-K retrieval for a user query.

        Args:
            query: Non-empty search query string.
            top_k: Number of ranked chunks to retrieve (default: 5).

        Returns:
            List of result dicts ranked by cosine similarity score descending.
        """
        if query is None or not query.strip():
            raise ValueError("Query cannot be empty or whitespace only.")

        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        total_vectors = self.index.ntotal
        actual_k = min(top_k, total_vectors)

        # Encode & normalize query vector (1, D)
        query_vec = self.embedder.encode_query(query.strip())

        # Search FAISS index
        distances, indices = self.index.search(query_vec, actual_k)

        results: list[dict[str, Any]] = []
        seen_chunk_ids: set[str] = set()

        for rank, (score, pos) in enumerate(zip(distances[0], indices[0]), start=1):
            if pos < 0 or pos >= len(self.metadata):
                logger.warning("FAISS returned out-of-bounds position: %d", pos)
                continue

            chunk_meta = self.metadata[pos]
            chunk_id = chunk_meta.get("chunk_id", f"POS_{pos}")

            if chunk_id in seen_chunk_ids:
                logger.error("Duplicate chunk_id '%s' encountered in dense search results", chunk_id)
                continue
            seen_chunk_ids.add(chunk_id)

            item: dict[str, Any] = {
                "rank": rank,
                "score": float(score),
                "chunk_id": chunk_id,
                "doc_number": chunk_meta.get("doc_number"),
                "document_type": chunk_meta.get("document_type"),
                "article": chunk_meta.get("article"),
                "article_title": chunk_meta.get("article_title"),
                "clause": chunk_meta.get("clause"),
                "point": chunk_meta.get("point"),
                "content": chunk_meta.get("content"),
                "content_with_context": chunk_meta.get("content_with_context"),
                "source_file": chunk_meta.get("source_file"),
                "start_page": chunk_meta.get("start_page"),
                "end_page": chunk_meta.get("end_page"),
                "issue_date": chunk_meta.get("issue_date"),
                "effective_date": chunk_meta.get("effective_date"),
                "legal_status": chunk_meta.get("legal_status"),
                "amends_document": chunk_meta.get("amends_document"),
                "amended_article": chunk_meta.get("amended_article"),
                "amended_clause": chunk_meta.get("amended_clause"),
                "amended_point": chunk_meta.get("amended_point"),
            }
            if "merged_from_chunk_ids" in chunk_meta:
                item["merged_from_chunk_ids"] = chunk_meta["merged_from_chunk_ids"]

            results.append(item)

        return results


def build_vector_store(
    chunks_path: Path,
    output_dir: Path,
    embedder: Optional[BGEEmbedder] = None,
    batch_size: int = 32,
) -> FAISSStore:
    """
    Load legal_chunks.json, build FAISS index, and persist to output_dir.

    Args:
        chunks_path: Path to data/chunks/legal_chunks.json.
        output_dir: Path to data/vector_store/.
        embedder: Optional BGEEmbedder instance.
        batch_size: Embedding batch size.

    Returns:
        Built FAISSStore.
    """
    if not chunks_path.exists():
        raise FileNotFoundError(f"Input corpus not found: {chunks_path}")

    with chunks_path.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    if embedder is None:
        embedder = BGEEmbedder()

    store = FAISSStore.build(
        chunks=chunks,
        embedder=embedder,
        corpus_path=chunks_path,
        batch_size=batch_size,
    )
    store.save(output_dir)
    return store


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    project_root = Path(__file__).resolve().parents[2]
    chunks_path = project_root / "data" / "chunks" / "legal_chunks.json"
    output_dir = project_root / "data" / "vector_store"

    print(f"Building FAISS vector store from {chunks_path}...")
    store = build_vector_store(chunks_path, output_dir, batch_size=32)
    print(f"Done! Created index with {store.index.ntotal} vectors at {output_dir}")
