"""
Tests for FAISS vector store and dense retrieval module (src/retrieval/faiss_store.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import pytest

from src.retrieval.bge_embedder import BGEEmbedder
from src.retrieval.faiss_store import (
    INDEX_FILENAME,
    MANIFEST_FILENAME,
    METADATA_FILENAME,
    FAISSStore,
    compute_file_sha256,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = PROJECT_ROOT / "data" / "chunks" / "legal_chunks.json"


@pytest.fixture(scope="session")
def shared_embedder():
    """Session-scoped embedder fixture so model is loaded only once across tests."""
    return BGEEmbedder()


@pytest.fixture(scope="session")
def sample_chunks():
    """Small representative sample of 5 legal chunks for fast unit tests."""
    assert CORPUS_PATH.exists(), f"Corpus not found at {CORPUS_PATH}"
    with CORPUS_PATH.open("r", encoding="utf-8") as f:
        full_corpus = json.load(f)
    assert len(full_corpus) > 10
    # Select 5 distinct chunks
    return full_corpus[:5]


@pytest.fixture
def sample_faiss_store(sample_chunks, shared_embedder, tmp_path):
    """Build and return a temporary FAISSStore from sample chunks."""
    store = FAISSStore.build(
        chunks=sample_chunks,
        embedder=shared_embedder,
        corpus_path=CORPUS_PATH,
        batch_size=8,
        show_progress_bar=False,
    )
    store.save(tmp_path)
    return store, tmp_path


# ═══════════════════════════════════════════════════════════════════════════════
# B. Corpus Verification Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCorpusIntegrity:
    def test_load_legal_chunks_json(self):
        """legal_chunks.json must exist and load as valid JSON array."""
        assert CORPUS_PATH.exists()
        with CORPUS_PATH.open("r", encoding="utf-8") as f:
            chunks = json.load(f)
        assert isinstance(chunks, list)
        assert len(chunks) == 2305

    def test_corpus_not_empty(self):
        """Corpus must have more than 2000 chunks."""
        with CORPUS_PATH.open("r", encoding="utf-8") as f:
            chunks = json.load(f)
        assert len(chunks) > 2000

    def test_chunk_ids_unique(self):
        """Every chunk in legal_chunks.json must have a unique chunk_id."""
        with CORPUS_PATH.open("r", encoding="utf-8") as f:
            chunks = json.load(f)
        chunk_ids = [c["chunk_id"] for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_content_and_content_with_context_exist(self):
        """All chunks must possess content, and content_with_context should be populated."""
        with CORPUS_PATH.open("r", encoding="utf-8") as f:
            chunks = json.load(f)
        for c in chunks:
            assert c.get("content"), f"Empty content in chunk {c.get('chunk_id')}"
            assert c.get("content_with_context"), f"Missing content_with_context in {c.get('chunk_id')}"


# ═══════════════════════════════════════════════════════════════════════════════
# C. FAISS Store Index Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestFAISSIndex:
    def test_build_index_flat_ip(self, sample_faiss_store, shared_embedder, sample_chunks):
        """Store builds an IndexFlatIP with correct dimension and vector count."""
        store, _ = sample_faiss_store
        assert isinstance(store.index, faiss.IndexFlatIP)
        assert store.index.ntotal == len(sample_chunks)
        assert store.index.d == shared_embedder.dimension

    def test_save_and_load_round_trip(self, sample_faiss_store, shared_embedder, sample_chunks):
        """Saved index, metadata, and manifest reload accurately."""
        store, store_dir = sample_faiss_store
        reloaded = FAISSStore.load(store_dir, embedder=shared_embedder)

        assert reloaded.index.ntotal == store.index.ntotal
        assert reloaded.index.d == store.index.d
        assert len(reloaded.metadata) == len(sample_chunks)
        assert reloaded.manifest["embedding_model"] == "BAAI/bge-m3"

    def test_load_missing_files_raises_error(self, tmp_path):
        """Loading from an empty directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            FAISSStore.load(tmp_path)


# ═══════════════════════════════════════════════════════════════════════════════
# D. Mapping Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestFAISSMapping:
    def test_position_maps_to_correct_chunk_id(self, sample_faiss_store, sample_chunks):
        """Index vector at position i matches sample_chunks[i]."""
        store, _ = sample_faiss_store
        for i, chunk in enumerate(sample_chunks):
            assert store.metadata[i]["chunk_id"] == chunk["chunk_id"]
            assert store.metadata[i]["doc_number"] == chunk["doc_number"]

    def test_metadata_count_matches_index_ntotal(self, sample_faiss_store):
        """Metadata list length equals index.ntotal."""
        store, _ = sample_faiss_store
        assert len(store.metadata) == store.index.ntotal

    def test_no_duplicate_chunk_ids_in_metadata(self, sample_faiss_store):
        """Chunk IDs in metadata have no duplicates."""
        store, _ = sample_faiss_store
        ids = [m["chunk_id"] for m in store.metadata]
        assert len(ids) == len(set(ids))


# ═══════════════════════════════════════════════════════════════════════════════
# E. Manifest Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestIndexManifest:
    def test_manifest_metadata_fields(self, sample_faiss_store, shared_embedder, sample_chunks):
        """Manifest contains all required technical metadata."""
        store, _ = sample_faiss_store
        manifest = store.manifest
        assert manifest["embedding_model"] == "BAAI/bge-m3"
        assert manifest["index_type"] == "IndexFlatIP"
        assert manifest["normalized"] is True
        assert manifest["embedding_dimension"] == shared_embedder.dimension
        assert manifest["num_vectors"] == len(sample_chunks)
        assert "created_at" in manifest

    def test_corpus_sha256_hash_computation(self):
        """compute_file_sha256 returns valid 64-char lowercase hex string."""
        sha = compute_file_sha256(CORPUS_PATH)
        assert isinstance(sha, str)
        assert len(sha) == 64
        assert all(c in "0123456789abcdef" for c in sha)


# ═══════════════════════════════════════════════════════════════════════════════
# F. Dense Search Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDenseSearch:
    def test_search_returns_top_k(self, sample_faiss_store):
        """Search returns exactly top_k results when top_k <= ntotal."""
        store, _ = sample_faiss_store
        results = store.search("Quy định chung về trật tự giao thông", top_k=3)
        assert len(results) == 3

    def test_search_ranked_by_score_descending(self, sample_faiss_store):
        """Results are ranked in descending order of similarity score."""
        store, _ = sample_faiss_store
        results = store.search("Quy định chung", top_k=4)
        scores = [r["score"] for r in results]
        ranks = [r["rank"] for r in results]
        assert ranks == list(range(1, len(results) + 1))
        assert scores == sorted(scores, reverse=True)

    def test_search_result_metadata_fields(self, sample_faiss_store):
        """Each search result dictionary contains all essential legal metadata."""
        store, _ = sample_faiss_store
        results = store.search("xử phạt vi phạm", top_k=1)
        assert len(results) == 1
        res = results[0]
        for field in [
            "rank",
            "score",
            "chunk_id",
            "doc_number",
            "article",
            "content",
            "content_with_context",
            "source_file",
        ]:
            assert field in res, f"Missing field '{field}' in search result"

    def test_no_duplicate_chunk_ids_in_results(self, sample_faiss_store):
        """Search results do not contain duplicate chunk_ids."""
        store, _ = sample_faiss_store
        results = store.search("quy định", top_k=5)
        result_ids = [r["chunk_id"] for r in results]
        assert len(result_ids) == len(set(result_ids))

    def test_empty_query_raises_value_error(self, sample_faiss_store):
        """Empty or whitespace query raises ValueError."""
        store, _ = sample_faiss_store
        with pytest.raises(ValueError):
            store.search("")
        with pytest.raises(ValueError):
            store.search("    ")
        with pytest.raises(ValueError):
            store.search(None)

    def test_invalid_top_k_raises_value_error(self, sample_faiss_store):
        """Negative or zero top_k raises ValueError."""
        store, _ = sample_faiss_store
        with pytest.raises(ValueError):
            store.search("Luật giao thông", top_k=0)
        with pytest.raises(ValueError):
            store.search("Luật giao thông", top_k=-5)

    def test_top_k_larger_than_corpus_clamped(self, sample_faiss_store, sample_chunks):
        """top_k > ntotal does not crash; returns min(top_k, ntotal) results."""
        store, _ = sample_faiss_store
        results = store.search("phương tiện giao thông", top_k=100)
        assert len(results) == len(sample_chunks)
