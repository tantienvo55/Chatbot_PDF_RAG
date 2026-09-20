"""
Evaluation module for retrieval benchmarking.
"""

from .retrieval_evaluator import (
    aggregate_metrics,
    compare_query_performance,
    compute_query_metrics,
    evaluate_retriever,
    normalize_article_id,
)

__all__ = [
    "compute_query_metrics",
    "normalize_article_id",
    "compare_query_performance",
    "aggregate_metrics",
    "evaluate_retriever",
]
