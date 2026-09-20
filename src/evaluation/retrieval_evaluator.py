"""
Retrieval evaluation module for Vietnamese traffic law RAG.

Provides deterministic evaluation of retrieval quality using standard IR metrics:
Hit@1, Hit@3, Hit@5, MRR@5, Recall@5, and Article-Hit@5.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


def normalize_article_id(article: Optional[str]) -> str:
    """
    Normalize article representation to canonical number.

    Examples:
        'Điều 11' -> '11'
        '11' -> '11'
        'điều 11' -> '11'
        'Điều 11a' -> '11a'
    """
    if not article:
        return ""
    cleaned = article.strip().lower()
    # Remove prefix 'điều' or 'dieu'
    cleaned = re.sub(r"^(?:điều|dieu)\s*", "", cleaned)
    return cleaned.strip()


def compute_query_metrics(
    retrieved_chunks: list[dict[str, Any]],
    relevant_chunk_ids: list[str],
    relevant_articles: list[dict[str, str]],
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Compute retrieval metrics for a single query.

    Args:
        retrieved_chunks: Ranked list of retrieved chunk dictionaries.
        relevant_chunk_ids: List of ground-truth relevant chunk_ids.
        relevant_articles: List of ground-truth relevant articles
                           [{"doc_number": "...", "article": "..."}].
        top_k: Maximum rank depth to evaluate (default: 5).

    Returns:
        Dictionary containing Hit@1, Hit@3, Hit@5, MRR@5, Recall@5,
        Article-Hit@5, first_relevant_rank, and retrieved_chunk_ids.
    """
    if not relevant_chunk_ids:
        raise ValueError("relevant_chunk_ids cannot be empty.")

    target_chunks = set(relevant_chunk_ids)

    # Extract retrieved chunk IDs in rank order up to top_k
    seen_ids: set[str] = set()
    retrieved_ids: list[str] = []
    top_chunks = retrieved_chunks[:top_k]

    for c in top_chunks:
        cid = c.get("chunk_id")
        if cid and cid not in seen_ids:
            seen_ids.add(cid)
            retrieved_ids.append(cid)

    # 1. Chunk-level Hit@K
    hit_1 = 1.0 if any(cid in target_chunks for cid in retrieved_ids[:1]) else 0.0
    hit_3 = 1.0 if any(cid in target_chunks for cid in retrieved_ids[:3]) else 0.0
    hit_5 = 1.0 if any(cid in target_chunks for cid in retrieved_ids[:5]) else 0.0

    # 2. MRR@5
    first_rank: Optional[int] = None
    for rank_idx, cid in enumerate(retrieved_ids[:5], start=1):
        if cid in target_chunks:
            first_rank = rank_idx
            break

    mrr_5 = (1.0 / first_rank) if first_rank is not None else 0.0

    # 3. Recall@5
    hits_in_5 = len(set(retrieved_ids[:5]).intersection(target_chunks))
    recall_5 = hits_in_5 / len(target_chunks) if target_chunks else 0.0

    # 4. Article-level Hit@5
    # Normalize ground-truth articles: (doc_number, normalized_article)
    target_articles = {
        (ra.get("doc_number"), normalize_article_id(ra.get("article")))
        for ra in relevant_articles
    }

    article_hit_5 = 0.0
    for c in top_chunks[:5]:
        doc_num = c.get("doc_number")
        art_norm = normalize_article_id(c.get("article"))
        if (doc_num, art_norm) in target_articles:
            article_hit_5 = 1.0
            break

    return {
        "hit_1": hit_1,
        "hit_3": hit_3,
        "hit_5": hit_5,
        "mrr_5": mrr_5,
        "recall_5": recall_5,
        "article_hit_5": article_hit_5,
        "first_relevant_rank": first_rank,
        "retrieved_chunk_ids": retrieved_ids[:5],
    }


def compare_query_performance(
    metrics_a: dict[str, Any], metrics_b: dict[str, Any]
) -> int:
    """
    Compare performance between two retrievers on the same query.

    Precedence order:
    1. Hit@5
    2. MRR@5 (if Hit@5 tied)
    3. Recall@5 (if MRR@5 tied)

    Returns:
        1 if A > B, -1 if A < B, 0 if tied.
    """
    # 1. Compare Hit@5
    h_a = metrics_a.get("hit_5", 0.0)
    h_b = metrics_b.get("hit_5", 0.0)
    if h_a > h_b:
        return 1
    if h_a < h_b:
        return -1

    # 2. Compare MRR@5
    m_a = metrics_a.get("mrr_5", 0.0)
    m_b = metrics_b.get("mrr_5", 0.0)
    if m_a > m_b:
        return 1
    if m_a < m_b:
        return -1

    # 3. Compare Recall@5
    r_a = metrics_a.get("recall_5", 0.0)
    r_b = metrics_b.get("recall_5", 0.0)
    if r_a > r_b:
        return 1
    if r_a < r_b:
        return -1

    return 0


def aggregate_metrics(query_results: list[dict[str, Any]]) -> dict[str, float]:
    """
    Compute mean of all metrics across all queries.
    """
    if not query_results:
        return {
            "hit_1": 0.0,
            "hit_3": 0.0,
            "hit_5": 0.0,
            "mrr_5": 0.0,
            "recall_5": 0.0,
            "article_hit_5": 0.0,
        }

    n = len(query_results)
    return {
        "hit_1": sum(r["hit_1"] for r in query_results) / n,
        "hit_3": sum(r["hit_3"] for r in query_results) / n,
        "hit_5": sum(r["hit_5"] for r in query_results) / n,
        "mrr_5": sum(r["mrr_5"] for r in query_results) / n,
        "recall_5": sum(r["recall_5"] for r in query_results) / n,
        "article_hit_5": sum(r["article_hit_5"] for r in query_results) / n,
    }


def evaluate_retriever(
    retriever: Any,
    eval_dataset: list[dict[str, Any]],
    top_k: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """
    Run evaluation on an arbitrary retriever instance.

    Args:
        retriever: Retriever instance with a search(query, top_k=...) method.
        eval_dataset: List of query ground truth dictionaries.
        top_k: Top K results to retrieve and score.

    Returns:
        Tuple of (per_query_results, summary_metrics).
    """
    per_query: list[dict[str, Any]] = []

    for item in eval_dataset:
        qid = item["id"]
        query = item["query"]
        relevant_chunks = item["relevant_chunk_ids"]
        relevant_articles = item.get("relevant_articles", [])

        # Execute search
        retrieved = retriever.search(query, top_k=top_k)

        # Calculate metrics
        metrics = compute_query_metrics(
            retrieved_chunks=retrieved,
            relevant_chunk_ids=relevant_chunks,
            relevant_articles=relevant_articles,
            top_k=top_k,
        )

        record = {
            "query_id": qid,
            "query": query,
            "relevant_chunk_ids": relevant_chunks,
            **metrics,
        }
        per_query.append(record)

    summary = aggregate_metrics(per_query)
    return per_query, summary
