"""
Tests for BM25 lexical retriever and Vietnamese tokenizer (src/retrieval/bm25_retriever.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.retrieval.bm25_retriever import (
    BM25Retriever,
    vietnamese_bm25_tokenizer,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = PROJECT_ROOT / "data" / "chunks" / "legal_chunks.json"
FAISS_METADATA_PATH = PROJECT_ROOT / "data" / "vector_store" / "metadata.json"


@pytest.fixture(scope="session")
def sample_chunks():
    """Session fixture loading 5 representative legal chunks for fast tests."""
    assert CORPUS_PATH.exists(), f"Corpus not found at {CORPUS_PATH}"
    with CORPUS_PATH.open("r", encoding="utf-8") as f:
        chunks = json.load(f)
    assert len(chunks) == 2305
    return chunks[:5]


@pytest.fixture(scope="session")
def sample_retriever(sample_chunks):
    """Build a BM25Retriever on sample chunks."""
    return BM25Retriever(corpus=sample_chunks)


@pytest.fixture(scope="session")
def full_retriever():
    """Build a BM25Retriever once across the session on full 2,305 chunks."""
    return BM25Retriever.from_json(CORPUS_PATH)


# ═══════════════════════════════════════════════════════════════════════════════
# A. Tokenizer Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestVietnameseTokenizer:
    def test_case_invariance(self):
        """Uppercase, lowercase, and title case must produce identical token representations."""
        t1 = vietnamese_bm25_tokenizer("Không đội mũ bảo hiểm")
        t2 = vietnamese_bm25_tokenizer("không đội mũ bảo hiểm")
        t3 = vietnamese_bm25_tokenizer("KHÔNG ĐỘI MŨ BẢO HIỂM")
        assert t1 == t2 == t3

    def test_unigram_and_bigram_generation(self):
        """Test specific user-requested bigram behavior for key phrases."""
        tokens_mu = vietnamese_bm25_tokenizer("mũ bảo hiểm")
        assert tokens_mu == ["mũ", "bảo", "hiểm", "mũ_bảo", "bảo_hiểm"]

        tokens_con = vietnamese_bm25_tokenizer("nồng độ cồn")
        assert tokens_con == ["nồng", "độ", "cồn", "nồng_độ", "độ_cồn"]

    def test_single_word_no_bigrams(self):
        """Single word input should produce only the unigram without error."""
        tokens = vietnamese_bm25_tokenizer("luật")
        assert tokens == ["luật"]

    def test_empty_and_whitespace_input(self):
        """Empty string or whitespace-only strings must return empty list."""
        assert vietnamese_bm25_tokenizer("") == []
        assert vietnamese_bm25_tokenizer("   \n\t  ") == []
        assert vietnamese_bm25_tokenizer(None) == []

    def test_vietnamese_unicode_preservation(self):
        """All Vietnamese accented characters must be preserved without distortion."""
        sample = "Luật Trật tự, an toàn giao thông đường bộ Việt Nam"
        tokens = vietnamese_bm25_tokenizer(sample)
        # Check unigrams are present
        assert "trật" in tokens
        assert "tự" in tokens
        assert "toàn" in tokens
        assert "đường" in tokens
        assert "bộ" in tokens
        assert "việt" in tokens
        assert "nam" in tokens
        # Check bigrams
        assert "trật_tự" in tokens
        assert "an_toàn" in tokens
        assert "giao_thông" in tokens
        assert "đường_bộ" in tokens

    def test_decimal_and_currency_preservation(self):
        """Decimal numbers and currency amounts must be preserved."""
        tokens = vietnamese_bm25_tokenizer("0,25 miligam/1 lít khí thở")
        assert "0,25" in tokens
        assert "miligam" in tokens
        assert "lít" in tokens
        assert "khí" in tokens
        assert "thở" in tokens

        tokens_money = vietnamese_bm25_tokenizer("Phạt tiền từ 100.000 đồng đến 5.000.000 đồng")
        assert "100.000" in tokens_money
        assert "5.000.000" in tokens_money

    def test_legal_doc_codes_and_dates_preservation(self):
        """Legal document codes and dates must be preserved as tokens."""
        tokens = vietnamese_bm25_tokenizer("Nghị định số 168/2024/NĐ-CP ngày 26/12/2024")
        assert "168/2024/nđ-cp" in tokens
        assert "26/12/2024" in tokens

        tokens_law = vietnamese_bm25_tokenizer("Luật số 36/2024/QH15")
        assert "36/2024/qh15" in tokens_law

    def test_punctuation_stripping(self):
        """Punctuation marks must not remain attached to words."""
        tokens = vietnamese_bm25_tokenizer("Điều 11, Khoản 4: 'Tín hiệu đèn màu đỏ là cấm đi!'")
        assert "điều" in tokens
        assert "11" in tokens
        assert "khoản" in tokens
        assert "4" in tokens
        assert "cấm" in tokens
        assert "đi" in tokens
        for t in tokens:
            assert not any(c in t for c in [",", ":", "'", "!"])


# ═══════════════════════════════════════════════════════════════════════════════
# B. Corpus Verification Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCorpusIntegrity:
    def test_corpus_exists_and_size(self):
        """legal_chunks.json must exist and contain exactly 2,305 chunks."""
        assert CORPUS_PATH.exists()
        with CORPUS_PATH.open("r", encoding="utf-8") as f:
            chunks = json.load(f)
        assert isinstance(chunks, list)
        assert len(chunks) == 2305

    def test_chunk_ids_unique(self):
        """Every chunk in legal_chunks.json must have a unique chunk_id."""
        with CORPUS_PATH.open("r", encoding="utf-8") as f:
            chunks = json.load(f)
        chunk_ids = [c["chunk_id"] for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_chunks_have_content(self):
        """Every chunk must have non-empty content or content_with_context."""
        with CORPUS_PATH.open("r", encoding="utf-8") as f:
            chunks = json.load(f)
        for c in chunks:
            text = c.get("content_with_context") or c.get("content") or ""
            assert text.strip(), f"Chunk {c.get('chunk_id')} has empty text"


# ═══════════════════════════════════════════════════════════════════════════════
# C. BM25 Build & Mapping Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestBM25Build:
    def test_empty_corpus_raises_error(self):
        """Building BM25 with empty corpus must raise ValueError."""
        with pytest.raises(ValueError, match="Corpus cannot be empty"):
            BM25Retriever(corpus=[])

    def test_build_doc_count(self, full_retriever):
        """BM25Retriever doc_count must equal total corpus size (2,305)."""
        assert full_retriever.doc_count == 2305
        assert len(full_retriever.corpus) == 2305
        assert len(full_retriever.tokenized_corpus) == 2305

    def test_deterministic_position_mapping(self, full_retriever):
        """Position i in BM25 retriever must exactly match chunk i from legal_chunks.json."""
        with CORPUS_PATH.open("r", encoding="utf-8") as f:
            chunks = json.load(f)
        for i in [0, 100, 500, 1000, 2000, 2304]:
            assert full_retriever.corpus[i]["chunk_id"] == chunks[i]["chunk_id"]


# ═══════════════════════════════════════════════════════════════════════════════
# D. Search Behavior Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestBM25Search:
    def test_search_top_k_count(self, full_retriever):
        """Search must return exactly top_k results when enough matches exist."""
        results = full_retriever.search("không đội mũ bảo hiểm", top_k=5)
        assert len(results) == 5

    def test_ranks_sequential(self, full_retriever):
        """Ranks must start at 1 and increment consecutively."""
        results = full_retriever.search("nồng độ cồn", top_k=5)
        ranks = [r["rank"] for r in results]
        assert ranks == [1, 2, 3, 4, 5]

    def test_scores_numeric_and_descending(self, full_retriever):
        """Scores must be positive floats and ordered non-increasing."""
        results = full_retriever.search("giấy phép lái xe bị trừ điểm", top_k=5)
        assert len(results) > 0
        scores = [r["score"] for r in results]
        for s in scores:
            assert isinstance(s, float)
            assert s > 0.0
        assert scores == sorted(scores, reverse=True)

    def test_no_duplicate_chunk_ids(self, full_retriever):
        """Search results must never contain duplicate chunk_ids."""
        results = full_retriever.search("xử phạt vi phạm giao thông", top_k=10)
        chunk_ids = [r["chunk_id"] for r in results]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_result_metadata_fields(self, full_retriever):
        """Retrieved results must retain all essential legal metadata fields."""
        results = full_retriever.search("tín hiệu đèn màu đỏ", top_k=3)
        assert len(results) > 0
        required_keys = [
            "rank", "score", "chunk_id", "doc_number", "document_type",
            "article", "content", "content_with_context", "source_file",
            "start_page", "end_page", "issue_date", "effective_date", "legal_status"
        ]
        for r in results:
            for k in required_keys:
                assert k in r, f"Missing key '{k}' in search result"


# ═══════════════════════════════════════════════════════════════════════════════
# E. Validation & Error Handling Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestBM25Validation:
    def test_query_none_raises_error(self, sample_retriever):
        """None query must raise ValueError."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            sample_retriever.search(None)

    def test_query_empty_raises_error(self, sample_retriever):
        """Empty string query must raise ValueError."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            sample_retriever.search("")

    def test_query_whitespace_raises_error(self, sample_retriever):
        """Whitespace-only query must raise ValueError."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            sample_retriever.search("    \t\n   ")

    def test_top_k_zero_raises_error(self, sample_retriever):
        """top_k <= 0 must raise ValueError."""
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            sample_retriever.search("giao thông", top_k=0)
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            sample_retriever.search("giao thông", top_k=-5)

    def test_top_k_greater_than_corpus(self, sample_retriever):
        """top_k larger than corpus size must be clamped without error."""
        results = sample_retriever.search("quy định", top_k=9999)
        assert len(results) <= sample_retriever.doc_count


# ═══════════════════════════════════════════════════════════════════════════════
# F. Zero Match Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestZeroMatch:
    def test_unrelated_nonsense_query_returns_empty_list(self, full_retriever):
        """A completely unrelated/nonsense query must return an empty list []."""
        results = full_retriever.search("xyzqwertynonexistenttoken123456789", top_k=5)
        assert results == []


# ═══════════════════════════════════════════════════════════════════════════════
# G. Corpus Consistency with FAISS Vector Store
# ═══════════════════════════════════════════════════════════════════════════════

class TestCorpusConsistencyWithFAISS:
    def test_chunk_ids_match_faiss_metadata(self, full_retriever):
        """BM25 corpus chunk_ids must match FAISS metadata chunk_ids 1:1 without loading BGE-M3."""
        assert FAISS_METADATA_PATH.exists(), f"FAISS metadata not found at {FAISS_METADATA_PATH}"
        with FAISS_METADATA_PATH.open("r", encoding="utf-8") as f:
            faiss_metadata = json.load(f)

        assert len(full_retriever.corpus) == len(faiss_metadata) == 2305

        bm25_ids = [c["chunk_id"] for c in full_retriever.corpus]
        faiss_ids = [m["chunk_id"] for m in faiss_metadata]

        # Order must be identical
        assert bm25_ids == faiss_ids
        # Set must be identical
        assert set(bm25_ids) == set(faiss_ids)
