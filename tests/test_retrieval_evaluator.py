"""
Unit tests for retrieval evaluation module (src/evaluation/retrieval_evaluator.py).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.evaluation.retrieval_evaluator import (
    aggregate_metrics,
    compare_query_performance,
    compute_query_metrics,
    evaluate_retriever,
    normalize_article_id,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DATASET_PATH = PROJECT_ROOT / "evaluation" / "retrieval_eval.json"
CORPUS_PATH = PROJECT_ROOT / "data" / "chunks" / "legal_chunks.json"


# ═══════════════════════════════════════════════════════════════════════════════
# A. Metric Computation Tests (Synthetic Data)
# ═══════════════════════════════════════════════════════════════════════════════

class TestMetricComputation:
    def test_hit_metrics(self):
        """Test Hit@1, Hit@3, Hit@5 when match is at rank 2."""
        retrieved = [
            {"chunk_id": "C_OTHER_1", "doc_number": "D1", "article": "1"},
            {"chunk_id": "C_TARGET", "doc_number": "D1", "article": "2"},
            {"chunk_id": "C_OTHER_2", "doc_number": "D1", "article": "3"},
        ]
        metrics = compute_query_metrics(
            retrieved_chunks=retrieved,
            relevant_chunk_ids=["C_TARGET"],
            relevant_articles=[{"doc_number": "D1", "article": "2"}],
            top_k=5,
        )

        assert metrics["hit_1"] == 0.0
        assert metrics["hit_3"] == 1.0
        assert metrics["hit_5"] == 1.0
        assert metrics["first_relevant_rank"] == 2
        assert pytest.approx(metrics["mrr_5"]) == 0.5
        assert pytest.approx(metrics["recall_5"]) == 1.0
        assert metrics["article_hit_5"] == 1.0

    def test_mrr_at_various_ranks(self):
        """Test MRR@5 with target at ranks 1, 3, 5, and not in top 5."""
        # Rank 1
        m1 = compute_query_metrics(
            [{"chunk_id": "C1"}], ["C1"], [], top_k=5
        )
        assert pytest.approx(m1["mrr_5"]) == 1.0

        # Rank 3
        m3 = compute_query_metrics(
            [{"chunk_id": "C0"}, {"chunk_id": "C1"}, {"chunk_id": "C2"}],
            ["C2"],
            [],
            top_k=5,
        )
        assert pytest.approx(m3["mrr_5"]) == 1.0 / 3.0

        # Rank 5
        m5 = compute_query_metrics(
            [{"chunk_id": f"C{i}"} for i in range(5)],
            ["C4"],
            [],
            top_k=5,
        )
        assert pytest.approx(m5["mrr_5"]) == 0.2

        # Not in top 5 (at rank 6)
        m_none = compute_query_metrics(
            [{"chunk_id": f"C{i}"} for i in range(6)],
            ["C5"],
            [],
            top_k=5,
        )
        assert m_none["mrr_5"] == 0.0
        assert m_none["hit_5"] == 0.0

    def test_multiple_relevant_chunks_recall(self):
        """Test recall calculation when query has 3 relevant chunks."""
        relevant = ["C1", "C2", "C3"]
        retrieved = [
            {"chunk_id": "C1"},  # hit 1
            {"chunk_id": "C_OTHER"},
            {"chunk_id": "C3"},  # hit 2
            {"chunk_id": "C_EXTRA"},
        ]
        metrics = compute_query_metrics(retrieved, relevant, [], top_k=5)

        assert metrics["hit_1"] == 1.0
        assert metrics["hit_5"] == 1.0
        # 2 out of 3 hits in top 5
        assert pytest.approx(metrics["recall_5"]) == 2.0 / 3.0

    def test_no_match(self):
        """Test metrics when no relevant chunk is retrieved."""
        metrics = compute_query_metrics(
            [{"chunk_id": "X1"}, {"chunk_id": "X2"}],
            ["TARGET"],
            [{"doc_number": "D1", "article": "1"}],
            top_k=5,
        )
        assert metrics["hit_1"] == 0.0
        assert metrics["hit_5"] == 0.0
        assert metrics["mrr_5"] == 0.0
        assert metrics["recall_5"] == 0.0
        assert metrics["article_hit_5"] == 0.0
        assert metrics["first_relevant_rank"] is None

    def test_empty_ground_truth_raises_error(self):
        """Empty relevant_chunk_ids must raise ValueError."""
        with pytest.raises(ValueError, match="relevant_chunk_ids cannot be empty"):
            compute_query_metrics([], [], [])


# ═══════════════════════════════════════════════════════════════════════════════
# B. Article Normalization & Article-Hit Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestArticleNormalization:
    def test_normalize_article_id(self):
        """Normalize variations of article string representation."""
        assert normalize_article_id("Điều 11") == "11"
        assert normalize_article_id("11") == "11"
        assert normalize_article_id("điều 11") == "11"
        assert normalize_article_id("  Điều   11  ") == "11"
        assert normalize_article_id("Điều 11a") == "11a"
        assert normalize_article_id(None) == ""
        assert normalize_article_id("") == ""

    def test_article_hit_different_clause_same_article(self):
        """Article-hit is 1.0 even if chunk is different clause of same article."""
        retrieved = [
            {
                "chunk_id": "L36_DIEU11_KHOAN2",
                "doc_number": "36/2024/QH15",
                "article": "Điều 11",
            }
        ]
        # Target chunk is Khoan 4 Diem c, but target article is Dieu 11
        metrics = compute_query_metrics(
            retrieved_chunks=retrieved,
            relevant_chunk_ids=["L36_DIEU11_KHOAN4_DIEMC"],
            relevant_articles=[{"doc_number": "36/2024/QH15", "article": "11"}],
            top_k=5,
        )
        assert metrics["hit_5"] == 0.0  # chunk-level miss
        assert metrics["article_hit_5"] == 1.0  # article-level hit!


# ═══════════════════════════════════════════════════════════════════════════════
# C. Discrepancy Comparison Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerformanceComparison:
    def test_hit5_precedence(self):
        """Higher Hit@5 wins immediately."""
        a = {"hit_5": 1.0, "mrr_5": 0.2, "recall_5": 0.5}
        b = {"hit_5": 0.0, "mrr_5": 0.0, "recall_5": 0.0}
        assert compare_query_performance(a, b) == 1
        assert compare_query_performance(b, a) == -1

    def test_mrr_precedence_when_hit_tied(self):
        """When Hit@5 is tied, higher MRR@5 wins."""
        # A hit at rank 1 (MRR=1.0), B hit at rank 3 (MRR=0.33)
        a = {"hit_5": 1.0, "mrr_5": 1.0, "recall_5": 0.5}
        b = {"hit_5": 1.0, "mrr_5": 0.333, "recall_5": 0.5}
        assert compare_query_performance(a, b) == 1
        assert compare_query_performance(b, a) == -1

    def test_recall_precedence_when_mrr_tied(self):
        """When Hit@5 and MRR@5 tied, higher Recall@5 wins."""
        # Both first hit at rank 1, but A has 2 hits (Recall=1.0), B has 1 hit (Recall=0.5)
        a = {"hit_5": 1.0, "mrr_5": 1.0, "recall_5": 1.0}
        b = {"hit_5": 1.0, "mrr_5": 1.0, "recall_5": 0.5}
        assert compare_query_performance(a, b) == 1
        assert compare_query_performance(b, a) == -1

    def test_exact_tie(self):
        """Identical metrics return 0."""
        a = {"hit_5": 1.0, "mrr_5": 0.5, "recall_5": 0.5}
        b = {"hit_5": 1.0, "mrr_5": 0.5, "recall_5": 0.5}
        assert compare_query_performance(a, b) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# D. Aggregate & Mock Evaluator Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvaluationAggregation:
    def test_aggregate_metrics(self):
        """Test averaging across query results."""
        results = [
            {"hit_1": 1.0, "hit_3": 1.0, "hit_5": 1.0, "mrr_5": 1.0, "recall_5": 1.0, "article_hit_5": 1.0},
            {"hit_1": 0.0, "hit_3": 1.0, "hit_5": 1.0, "mrr_5": 0.5, "recall_5": 0.5, "article_hit_5": 1.0},
            {"hit_1": 0.0, "hit_3": 0.0, "hit_5": 0.0, "mrr_5": 0.0, "recall_5": 0.0, "article_hit_5": 0.0},
        ]
        summary = aggregate_metrics(results)
        assert pytest.approx(summary["hit_1"]) == 1.0 / 3.0
        assert pytest.approx(summary["hit_3"]) == 2.0 / 3.0
        assert pytest.approx(summary["hit_5"]) == 2.0 / 3.0
        assert pytest.approx(summary["mrr_5"]) == 1.5 / 3.0
        assert pytest.approx(summary["recall_5"]) == 1.5 / 3.0
        assert pytest.approx(summary["article_hit_5"]) == 2.0 / 3.0

    def test_evaluate_retriever_mock(self):
        """Test evaluate_retriever end-to-end with a mock retriever."""
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            {"chunk_id": "CHUNK_HIT", "doc_number": "DOC1", "article": "Điều 1"}
        ]
        dataset = [
            {
                "id": "Q1",
                "query": "query 1",
                "relevant_chunk_ids": ["CHUNK_HIT"],
                "relevant_articles": [{"doc_number": "DOC1", "article": "1"}],
            }
        ]
        per_query, summary = evaluate_retriever(mock_retriever, dataset, top_k=5)
        assert len(per_query) == 1
        assert per_query[0]["hit_1"] == 1.0
        assert summary["hit_1"] == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# E. Real Evaluation Dataset Integrity Sanity Check
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvaluationDatasetSanity:
    def test_dataset_exists_and_valid(self):
        """Verify evaluation/retrieval_eval.json exists and loads valid JSON."""
        assert EVAL_DATASET_PATH.exists(), f"Dataset missing at {EVAL_DATASET_PATH}"
        with EVAL_DATASET_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert 20 <= len(data) <= 30

    def test_query_ids_unique(self):
        """All query IDs in retrieval_eval.json must be unique."""
        with EVAL_DATASET_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)

        ids = [item["id"] for item in data]
        assert len(ids) == len(set(ids))

    def test_all_ground_truth_chunks_exist_in_corpus(self):
        """Every relevant_chunk_id in retrieval_eval.json must exist in legal_chunks.json."""
        with EVAL_DATASET_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)

        with CORPUS_PATH.open("r", encoding="utf-8") as f:
            corpus = json.load(f)

        corpus_chunk_ids = {c["chunk_id"] for c in corpus}

        for item in data:
            for cid in item["relevant_chunk_ids"]:
                assert cid in corpus_chunk_ids, (
                    f"Ground truth chunk '{cid}' for query '{item['id']}' not found in corpus!"
                )
