"""
Evaluation script to benchmark Dense, BM25, and Hybrid Retrieval across 25 legal queries.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.retrieval_evaluator import (
    compare_query_performance,
    evaluate_retriever,
)
from src.retrieval.bge_embedder import BGEEmbedder
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.faiss_store import FAISSStore
from src.retrieval.hybrid_retriever import HybridRetriever

EVAL_DATASET_PATH = PROJECT_ROOT / "evaluation" / "retrieval_eval.json"
CORPUS_PATH = PROJECT_ROOT / "data" / "chunks" / "legal_chunks.json"
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"


def preview_snippet(text: str | None, max_len: int = 120) -> str:
    if not text:
        return ""
    cleaned = text.replace("\n", " ").strip()
    return cleaned[:max_len] + ("…" if len(cleaned) > max_len else "")


def main() -> None:
    print("=" * 85)
    print("PROMPT 7.5 — RETRIEVAL BENCHMARK EVALUATION")
    print("=" * 85)

    # 1. Load evaluation dataset
    print(f"Loading evaluation dataset from {EVAL_DATASET_PATH}...")
    with EVAL_DATASET_PATH.open("r", encoding="utf-8") as f:
        eval_data = json.load(f)

    print(f"Total evaluation queries: {len(eval_data)}")

    # Sanity check
    with CORPUS_PATH.open("r", encoding="utf-8") as f:
        corpus = json.load(f)
    corpus_chunk_ids = {c["chunk_id"] for c in corpus}

    for item in eval_data:
        for cid in item["relevant_chunk_ids"]:
            if cid not in corpus_chunk_ids:
                raise ValueError(f"Ground truth chunk '{cid}' not found in corpus!")
    print("Sanity check passed: All ground-truth chunks exist in legal_chunks.json.\n")

    # 2. Load retrievers
    print("Loading Dense Retriever (FAISS + BGE-M3)...")
    embedder = BGEEmbedder()
    dense_store = FAISSStore.load(VECTOR_STORE_DIR, embedder=embedder)

    print("Loading BM25 Retriever (BM25Okapi)...")
    bm25_retriever = BM25Retriever.from_json(CORPUS_PATH)

    print("Initializing HybridRetriever (RRF k=60)...")
    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_store,
        bm25_retriever=bm25_retriever,
        rrf_k=60,
        corpus_path=CORPUS_PATH,
        validate_consistency=True,
    )
    print("All 3 retrievers initialized successfully.\n")

    # 3. Evaluate Dense Retriever
    print("-" * 85)
    print("Evaluating Dense Retrieval...")
    t0 = time.perf_counter()
    dense_per_query, dense_summary = evaluate_retriever(dense_store, eval_data, top_k=5)
    dense_time = time.perf_counter() - t0
    dense_avg_latency = (dense_time / len(eval_data)) * 1000

    # 4. Evaluate BM25 Retriever
    print("Evaluating BM25 Retrieval...")
    t0 = time.perf_counter()
    bm25_per_query, bm25_summary = evaluate_retriever(bm25_retriever, eval_data, top_k=5)
    bm25_time = time.perf_counter() - t0
    bm25_avg_latency = (bm25_time / len(eval_data)) * 1000

    # 5. Evaluate Hybrid Retriever
    print("Evaluating Hybrid Retrieval (RRF)...")
    t0 = time.perf_counter()
    hybrid_per_query, hybrid_summary = evaluate_retriever(hybrid_retriever, eval_data, top_k=5)
    hybrid_time = time.perf_counter() - t0
    hybrid_avg_latency = (hybrid_time / len(eval_data)) * 1000

    # 6. Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "dense_results.json").open("w", encoding="utf-8") as f:
        json.dump(dense_per_query, f, ensure_ascii=False, indent=2)
    with (RESULTS_DIR / "bm25_results.json").open("w", encoding="utf-8") as f:
        json.dump(bm25_per_query, f, ensure_ascii=False, indent=2)
    with (RESULTS_DIR / "hybrid_results.json").open("w", encoding="utf-8") as f:
        json.dump(hybrid_per_query, f, ensure_ascii=False, indent=2)

    summary_data = {
        "num_queries": len(eval_data),
        "dense": {**dense_summary, "avg_latency_ms": dense_avg_latency, "total_time_s": dense_time},
        "bm25": {**bm25_summary, "avg_latency_ms": bm25_avg_latency, "total_time_s": bm25_time},
        "hybrid": {**hybrid_summary, "avg_latency_ms": hybrid_avg_latency, "total_time_s": hybrid_time},
    }
    with (RESULTS_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    # Save summary CSV
    with (RESULTS_DIR / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Retriever", "Hit@1", "Hit@3", "Hit@5", "MRR@5", "Recall@5", "Article_Hit@5", "Avg_Latency_ms"])
        for name, summ, lat in [
            ("Dense (BGE-M3 + FAISS)", dense_summary, dense_avg_latency),
            ("BM25 (BM25Okapi)", bm25_summary, bm25_avg_latency),
            ("Hybrid (RRF k=60)", hybrid_summary, hybrid_avg_latency),
        ]:
            writer.writerow([
                name,
                f"{summ['hit_1']:.4f}",
                f"{summ['hit_3']:.4f}",
                f"{summ['hit_5']:.4f}",
                f"{summ['mrr_5']:.4f}",
                f"{summ['recall_5']:.4f}",
                f"{summ['article_hit_5']:.4f}",
                f"{lat:.2f}",
            ])

    print(f"\nAll results saved to: {RESULTS_DIR}/\n")

    # 7. Print summary table
    print("=" * 85)
    print("RETRIEVAL EVALUATION SUMMARY TABLE")
    print("=" * 85)
    header = f"{'Retriever':<25} | {'Hit@1':<7} | {'Hit@3':<7} | {'Hit@5':<7} | {'MRR@5':<7} | {'Recall@5':<8} | {'Article-Hit@5':<13} | {'Latency':<8}"
    print(header)
    print("-" * len(header))
    for name, summ, lat in [
        ("Dense (FAISS)", dense_summary, dense_avg_latency),
        ("BM25 (Okapi)", bm25_summary, bm25_avg_latency),
        ("Hybrid (RRF)", hybrid_summary, hybrid_avg_latency),
    ]:
        print(
            f"{name:<25} | "
            f"{summ['hit_1'] * 100:>6.2f}% | "
            f"{summ['hit_3'] * 100:>6.2f}% | "
            f"{summ['hit_5'] * 100:>6.2f}% | "
            f"{summ['mrr_5']:>7.4f} | "
            f"{summ['recall_5'] * 100:>7.2f}% | "
            f"{summ['article_hit_5'] * 100:>12.2f}% | "
            f"{lat:>5.2f} ms"
        )
    print("=" * 85 + "\n")

    # 8. Discrepancy & Error Analysis
    hybrid_misses = []
    dense_better = []
    bm25_better = []
    all_three_miss = []

    for i, item in enumerate(eval_data):
        qid = item["id"]
        q_text = item["query"]
        expected = item["relevant_chunk_ids"]

        d_m = dense_per_query[i]
        b_m = bm25_per_query[i]
        h_m = hybrid_per_query[i]

        # Hybrid miss
        if h_m["hit_5"] == 0.0:
            hybrid_misses.append((qid, q_text, expected, d_m, b_m, h_m))

        # All 3 miss
        if d_m["hit_5"] == 0.0 and b_m["hit_5"] == 0.0 and h_m["hit_5"] == 0.0:
            all_three_miss.append((qid, q_text, expected, d_m, b_m, h_m))

        # Dense > Hybrid
        if compare_query_performance(d_m, h_m) == 1:
            dense_better.append((qid, q_text, expected, d_m, b_m, h_m))

        # BM25 > Hybrid
        if compare_query_performance(b_m, h_m) == 1:
            bm25_better.append((qid, q_text, expected, d_m, b_m, h_m))

    print("=" * 85)
    print("PER-QUERY DISCREPANCY & ERROR ANALYSIS")
    print("=" * 85)
    print(f"1. Queries where Hybrid missed in Top 5 (Hit@5 = 0): {len(hybrid_misses)}")
    for qid, q_text, exp, d_m, b_m, h_m in hybrid_misses:
        print(f"   - [{qid}] '{q_text}'")
        print(f"     Expected chunks : {exp}")
        print(f"     Dense Top 5     : {d_m['retrieved_chunk_ids']}")
        print(f"     BM25 Top 5      : {b_m['retrieved_chunk_ids']}")
        print(f"     Hybrid Top 5    : {h_m['retrieved_chunk_ids']}")

    print(f"\n2. Queries where Dense outperformed Hybrid: {len(dense_better)}")
    for qid, q_text, exp, d_m, b_m, h_m in dense_better:
        print(f"   - [{qid}] '{q_text}'")
        print(f"     Expected chunks : {exp}")
        print(f"     Dense  : Hit@5={d_m['hit_5']}, MRR@5={d_m['mrr_5']:.4f}, Recall@5={d_m['recall_5']:.4f}")
        print(f"     Hybrid : Hit@5={h_m['hit_5']}, MRR@5={h_m['mrr_5']:.4f}, Recall@5={h_m['recall_5']:.4f}")

    print(f"\n3. Queries where BM25 outperformed Hybrid: {len(bm25_better)}")
    for qid, q_text, exp, d_m, b_m, h_m in bm25_better:
        print(f"   - [{qid}] '{q_text}'")
        print(f"     Expected chunks : {exp}")
        print(f"     BM25   : Hit@5={b_m['hit_5']}, MRR@5={b_m['mrr_5']:.4f}, Recall@5={b_m['recall_5']:.4f}")
        print(f"     Hybrid : Hit@5={h_m['hit_5']}, MRR@5={h_m['mrr_5']:.4f}, Recall@5={h_m['recall_5']:.4f}")

    print(f"\n4. Queries where all 3 retrievers missed in Top 5: {len(all_three_miss)}")
    for qid, q_text, exp, d_m, b_m, h_m in all_three_miss:
        print(f"   - [{qid}] '{q_text}'")
        print(f"     Expected chunks : {exp}")
        print(f"     Dense Top 5     : {d_m['retrieved_chunk_ids']}")
        print(f"     BM25 Top 5      : {b_m['retrieved_chunk_ids']}")
        print(f"     Hybrid Top 5    : {h_m['retrieved_chunk_ids']}")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
