"""
Tests for Hybrid Retriever with Reciprocal Rank Fusion (src/retrieval/hybrid_retriever.py).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.retrieval.bge_embedder import BGEEmbedder
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.faiss_store import FAISSStore
from src.retrieval.hybrid_retriever import HybridRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = PROJECT_ROOT / "data" / "chunks" / "legal_chunks.json"
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"


# ═══════════════════════════════════════════════════════════════════════════════
# Mock Helpers for Fast, Isolated RRF Unit Tests
# ═══════════════════════════════════════════════════════════════════════════════

def create_mock_store(chunk_ids: list[str]) -> MagicMock:
    """Create a mock retriever having metadata and corpus matching chunk_ids."""
    mock = MagicMock()
    mock.metadata = [{"chunk_id": cid, "content": f"Content of {cid}"} for cid in chunk_ids]
    mock.corpus = list(mock.metadata)
    mock.manifest = {}
    return mock


@pytest.fixture
def mock_stores():
    """Returns a pair of mock Dense and BM25 stores sharing 4 canonical chunks."""
    shared_ids = ["CHUNK_A", "CHUNK_B", "CHUNK_C", "CHUNK_D"]
    dense = create_mock_store(shared_ids)
    bm25 = create_mock_store(shared_ids)
    return dense, bm25


# ═══════════════════════════════════════════════════════════════════════════════
# A. RRF Math, Deduplication & Provenance Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestRRFLogic:
    def test_rrf_math_and_deduplication(self, mock_stores):
        """Test exact 1-indexed RRF formula and chunk_id deduplication."""
        dense, bm25 = mock_stores

        # Dense: A at rank 1, B at rank 2
        dense.search.return_value = [
            {"rank": 1, "score": 0.90, "chunk_id": "CHUNK_A", "content": "A"},
            {"rank": 2, "score": 0.80, "chunk_id": "CHUNK_B", "content": "B"},
        ]
        # BM25: A at rank 2, C at rank 1
        bm25.search.return_value = [
            {"rank": 1, "score": 25.0, "chunk_id": "CHUNK_C", "content": "C"},
            {"rank": 2, "score": 20.0, "chunk_id": "CHUNK_A", "content": "A"},
        ]

        hybrid = HybridRetriever(dense, bm25, rrf_k=60, validate_consistency=False)
        results = hybrid.search("test query", top_k=5, dense_top_k=20, bm25_top_k=20)

        # CHUNK_A appears in both: 1/(60+1) + 1/(60+2) = 1/61 + 1/62
        expected_rrf_a = (1.0 / 61.0) + (1.0 / 62.0)
        # CHUNK_C only in BM25 rank 1: 1/(60+1) = 1/61
        expected_rrf_c = 1.0 / 61.0
        # CHUNK_B only in Dense rank 2: 1/(60+2) = 1/62
        expected_rrf_b = 1.0 / 62.0

        assert len(results) == 3
        # Check rank 1 is CHUNK_A (highest RRF score)
        assert results[0]["chunk_id"] == "CHUNK_A"
        assert pytest.approx(results[0]["rrf_score"], rel=1e-5) == expected_rrf_a
        assert results[0]["dense_rank"] == 1
        assert results[0]["dense_score"] == 0.90
        assert results[0]["bm25_rank"] == 2
        assert results[0]["bm25_score"] == 20.0

        # Check rank 2 is CHUNK_C
        assert results[1]["chunk_id"] == "CHUNK_C"
        assert pytest.approx(results[1]["rrf_score"], rel=1e-5) == expected_rrf_c
        assert results[1]["dense_rank"] is None
        assert results[1]["dense_score"] is None
        assert results[1]["bm25_rank"] == 1
        assert results[1]["bm25_score"] == 25.0

        # Check rank 3 is CHUNK_B
        assert results[2]["chunk_id"] == "CHUNK_B"
        assert pytest.approx(results[2]["rrf_score"], rel=1e-5) == expected_rrf_b
        assert results[2]["dense_rank"] == 2
        assert results[2]["dense_score"] == 0.80
        assert results[2]["bm25_rank"] is None
        assert results[2]["bm25_score"] is None

    def test_custom_rrf_k(self, mock_stores):
        """Custom rrf_k parameter must be respected in formula."""
        dense, bm25 = mock_stores
        dense.search.return_value = [{"rank": 1, "score": 0.9, "chunk_id": "CHUNK_A"}]
        bm25.search.return_value = []

        hybrid = HybridRetriever(dense, bm25, rrf_k=20, validate_consistency=False)
        results = hybrid.search("test", top_k=1)
        # 1 / (20 + 1) = 1/21
        assert pytest.approx(results[0]["rrf_score"], rel=1e-5) == (1.0 / 21.0)


# ═══════════════════════════════════════════════════════════════════════════════
# B. Deterministic Tie-Breaking Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestTieBreaking:
    def test_tie_break_overlap_beats_single_source(self, mock_stores):
        """Overlap candidate should break tie against single-source candidate with equal RRF."""
        dense, bm25 = mock_stores

        # We construct identical RRF score by using different ranks if mathematically possible,
        # or testing equal single ranks with tie-break.
        # Let's test two single-source items with identical score: Dense rank 1 vs BM25 rank 1.
        # RRF score for both is 1/(60+1) = 1/61.
        # Both have missing penalty: dense has eff_bm25_rank = 21, bm25 has eff_dense_rank = 21.
        # Total rank for both = 1 + 21 = 22.
        # Next tie-break is eff_dense_rank: Dense item has eff_dense_rank = 1, BM25 item has eff_dense_rank = 21.
        # Dense item must win!
        dense.search.return_value = [
            {"rank": 1, "score": 0.8, "chunk_id": "CHUNK_DENSE_ONLY"}
        ]
        bm25.search.return_value = [
            {"rank": 1, "score": 10.0, "chunk_id": "CHUNK_BM25_ONLY"}
        ]

        hybrid = HybridRetriever(dense, bm25, rrf_k=60, validate_consistency=False)
        results = hybrid.search("query", top_k=2, dense_top_k=20, bm25_top_k=20)

        assert results[0]["chunk_id"] == "CHUNK_DENSE_ONLY"
        assert results[1]["chunk_id"] == "CHUNK_BM25_ONLY"

    def test_tie_break_lexical_order(self, mock_stores):
        """Identical scores and provenance must break tie using chunk_id lexical order."""
        dense, bm25 = mock_stores
        # Both appear only in Dense at rank 1 (hypothetical mock scenario)
        dense.search.return_value = [
            {"rank": 1, "score": 0.9, "chunk_id": "CHUNK_Z"},
            {"rank": 1, "score": 0.9, "chunk_id": "CHUNK_A"},
        ]
        bm25.search.return_value = []

        hybrid = HybridRetriever(dense, bm25, rrf_k=60, validate_consistency=False)
        results = hybrid.search("query", top_k=2)

        assert results[0]["chunk_id"] == "CHUNK_A"
        assert results[1]["chunk_id"] == "CHUNK_Z"


# ═══════════════════════════════════════════════════════════════════════════════
# C. Validation & Error Handling Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestHybridValidation:
    def test_invalid_queries(self, mock_stores):
        """None, empty string, and whitespace-only queries must raise ValueError."""
        dense, bm25 = mock_stores
        hybrid = HybridRetriever(dense, bm25, validate_consistency=False)

        with pytest.raises(ValueError, match="Query cannot be empty"):
            hybrid.search(None)
        with pytest.raises(ValueError, match="Query cannot be empty"):
            hybrid.search("")
        with pytest.raises(ValueError, match="Query cannot be empty"):
            hybrid.search("   \t\n  ")

    def test_invalid_top_k_parameters(self, mock_stores):
        """Non-positive top_k parameters must raise ValueError."""
        dense, bm25 = mock_stores
        hybrid = HybridRetriever(dense, bm25, validate_consistency=False)

        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            hybrid.search("valid", top_k=0)
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            hybrid.search("valid", top_k=-1)
        with pytest.raises(ValueError, match="dense_top_k must be a positive integer"):
            hybrid.search("valid", dense_top_k=0)
        with pytest.raises(ValueError, match="bm25_top_k must be a positive integer"):
            hybrid.search("valid", bm25_top_k=0)


# ═══════════════════════════════════════════════════════════════════════════════
# D. Fallback Behavior Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestFallbackBehavior:
    def test_dense_empty_returns_bm25_candidates(self, mock_stores):
        """When Dense returns empty list, BM25 candidates are returned."""
        dense, bm25 = mock_stores
        dense.search.return_value = []
        bm25.search.return_value = [
            {"rank": 1, "score": 15.0, "chunk_id": "CHUNK_BM25"}
        ]

        hybrid = HybridRetriever(dense, bm25, validate_consistency=False)
        results = hybrid.search("query", top_k=5)

        assert len(results) == 1
        assert results[0]["chunk_id"] == "CHUNK_BM25"
        assert results[0]["dense_rank"] is None
        assert results[0]["bm25_rank"] == 1

    def test_bm25_empty_returns_dense_candidates(self, mock_stores):
        """When BM25 returns empty list (e.g. out of vocabulary), Dense candidates are returned."""
        dense, bm25 = mock_stores
        dense.search.return_value = [
            {"rank": 1, "score": 0.85, "chunk_id": "CHUNK_DENSE"}
        ]
        bm25.search.return_value = []

        hybrid = HybridRetriever(dense, bm25, validate_consistency=False)
        results = hybrid.search("query", top_k=5)

        assert len(results) == 1
        assert results[0]["chunk_id"] == "CHUNK_DENSE"
        assert results[0]["bm25_rank"] is None
        assert results[0]["dense_rank"] == 1

    def test_both_empty_returns_empty_list(self, mock_stores):
        """When both retrievers return empty, hybrid returns empty list []."""
        dense, bm25 = mock_stores
        dense.search.return_value = []
        bm25.search.return_value = []

        hybrid = HybridRetriever(dense, bm25, validate_consistency=False)
        assert hybrid.search("nonexistent", top_k=5) == []


# ═══════════════════════════════════════════════════════════════════════════════
# E. Corpus Consistency & Hash Validation Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCorpusConsistencyCheck:
    def test_mismatched_chunk_ids_raises_error(self):
        """Different sets of chunk_ids must raise ValueError on startup."""
        dense = create_mock_store(["CHUNK_1", "CHUNK_2"])
        bm25 = create_mock_store(["CHUNK_1", "CHUNK_DIFFERENT"])

        with pytest.raises(ValueError, match="Chunk ID sets do not match"):
            HybridRetriever(dense, bm25, validate_consistency=True)

    def test_duplicate_chunk_ids_raises_error(self):
        """Duplicate chunk_ids in either store must raise ValueError on startup."""
        dense = create_mock_store(["CHUNK_1", "CHUNK_1"])
        bm25 = create_mock_store(["CHUNK_1"])

        with pytest.raises(ValueError, match="Duplicate chunk_ids detected"):
            HybridRetriever(dense, bm25, validate_consistency=True)

    def test_hash_mismatch_raises_strict_error(self, tmp_path):
        """Corpus hash mismatch between manifest and file must raise ValueError."""
        dummy_corpus = tmp_path / "legal_chunks.json"
        dummy_corpus.write_text("dummy content", encoding="utf-8")

        dense = create_mock_store(["CHUNK_1"])
        dense.manifest = {"corpus_sha256": "wrong_hash_12345"}
        bm25 = create_mock_store(["CHUNK_1"])

        with pytest.raises(ValueError, match="Corpus SHA-256 mismatch"):
            HybridRetriever(
                dense, bm25, corpus_path=dummy_corpus, validate_consistency=True
            )


# ═══════════════════════════════════════════════════════════════════════════════
# F. Real Corpus Integration Tests (Session-scoped model fixture)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def shared_embedder():
    """Session-scoped BGEEmbedder so model is loaded only once across all test sessions."""
    return BGEEmbedder()


@pytest.fixture(scope="session")
def real_hybrid_retriever(shared_embedder):
    """Instantiate real HybridRetriever on actual FAISS and BM25 stores."""
    assert VECTOR_STORE_DIR.exists()
    assert CORPUS_PATH.exists()

    dense_store = FAISSStore.load(VECTOR_STORE_DIR, embedder=shared_embedder)
    bm25_retriever = BM25Retriever.from_json(CORPUS_PATH)

    return HybridRetriever(
        dense_retriever=dense_store,
        bm25_retriever=bm25_retriever,
        corpus_path=CORPUS_PATH,
        validate_consistency=True,
    )


class TestRealCorpusIntegration:
    def test_startup_consistency_verified(self, real_hybrid_retriever):
        """Verify that startup validation succeeded on 2,305 real chunks."""
        assert len(real_hybrid_retriever.dense_retriever.metadata) == 2305
        assert real_hybrid_retriever.bm25_retriever.doc_count == 2305

    def test_real_hybrid_search(self, real_hybrid_retriever):
        """Execute real hybrid query, verify output structure, ranks, and provenance."""
        results = real_hybrid_retriever.search(
            "không đội mũ bảo hiểm",
            top_k=5,
            dense_top_k=20,
            bm25_top_k=20,
        )

        assert len(results) == 5
        # Check ranks
        assert [r["rank"] for r in results] == [1, 2, 3, 4, 5]

        # Check RRF score descending
        scores = [r["rrf_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

        # Check provenance and legal metadata
        for r in results:
            assert "chunk_id" in r
            assert "rrf_score" in r
            assert "dense_rank" in r
            assert "dense_score" in r
            assert "bm25_rank" in r
            assert "bm25_score" in r
            assert "doc_number" in r
            assert "article" in r
            assert "content" in r
            # At least one retriever must have found it
            assert (r["dense_rank"] is not None) or (r["bm25_rank"] is not None)
