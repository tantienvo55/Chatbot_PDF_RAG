"""
Smoke test comparing Dense, BM25, and Hybrid (RRF) Retrieval across 5 legal queries.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.bge_embedder import BGEEmbedder
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.faiss_store import FAISSStore
from src.retrieval.hybrid_retriever import HybridRetriever

SMOKE_QUERIES = [
    "Có được vượt đèn đỏ không?",
    "Không đội mũ bảo hiểm bị xử phạt như thế nào?",
    "Quy định về nồng độ cồn khi lái xe là gì?",
    "Giấy phép lái xe bị trừ điểm trong trường hợp nào?",
    "Người điều khiển xe máy có nghĩa vụ gì?",
]


def preview_text(text: str, max_len: int = 150) -> str:
    """Clean and preview text snippet."""
    cleaned = text.replace("\n", " ").strip()
    if len(cleaned) > max_len:
        return cleaned[:max_len] + "…"
    return cleaned


def run_smoke_test(vector_store_dir: Path, chunks_path: Path) -> None:
    print("=" * 85)
    print("PROMPT 7 — HYBRID RETRIEVAL (RRF) SMOKE TEST & COMPARATIVE ANALYSIS")
    print("=" * 85)

    print("Loading BGE-M3 Embedder & FAISS Store...")
    embedder = BGEEmbedder()
    dense_store = FAISSStore.load(vector_store_dir, embedder=embedder)

    print("Loading BM25 Retriever...")
    bm25_retriever = BM25Retriever.from_json(chunks_path)

    print("Initializing HybridRetriever (rrf_k=60, dense_top_k=20, bm25_top_k=20)...")
    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_store,
        bm25_retriever=bm25_retriever,
        rrf_k=60,
        corpus_path=chunks_path,
        validate_consistency=True,
    )
    print("Corpus consistency & SHA-256 hash verified successfully.")
    print("=" * 85 + "\n")

    dense_latencies: list[float] = []
    bm25_latencies: list[float] = []
    hybrid_latencies: list[float] = []

    for idx, query in enumerate(SMOKE_QUERIES, start=1):
        print(f"QUERY [{idx}/5]: '{query}'")
        print("=" * 85)

        # 1. Dense search
        t_d0 = time.perf_counter()
        dense_candidates = dense_store.search(query, top_k=20)
        dense_dur = (time.perf_counter() - t_d0) * 1000
        dense_latencies.append(dense_dur)

        # 2. BM25 search
        t_b0 = time.perf_counter()
        bm25_candidates = bm25_retriever.search(query, top_k=20)
        bm25_dur = (time.perf_counter() - t_b0) * 1000
        bm25_latencies.append(bm25_dur)

        # 3. Hybrid search
        t_h0 = time.perf_counter()
        hybrid_results = hybrid_retriever.search(
            query, top_k=5, dense_top_k=20, bm25_top_k=20
        )
        hybrid_dur = (time.perf_counter() - t_h0) * 1000
        hybrid_latencies.append(hybrid_dur)

        # Overlap analysis
        dense_ids = [r["chunk_id"] for r in dense_candidates]
        bm25_ids = [r["chunk_id"] for r in bm25_candidates]
        overlap_ids = set(dense_ids).intersection(set(bm25_ids))
        unique_count = len(set(dense_ids).union(set(bm25_ids)))

        print(f"Candidate Analysis (Pool: dense_k=20, bm25_k=20):")
        print(f"  - Dense candidates : {len(dense_candidates)}")
        print(f"  - BM25 candidates  : {len(bm25_candidates)}")
        print(f"  - Candidate Overlap: {len(overlap_ids)} chunks")
        print(f"  - Unique Candidates: {unique_count} chunks")
        print(f"Latencies: Dense={dense_dur:.2f}ms | BM25={bm25_dur:.2f}ms | Hybrid Total={hybrid_dur:.2f}ms")
        print("-" * 85)

        # Print Dense Top 5
        print("[A] DENSE TOP 5 (FAISS + BGE-M3):")
        for r in dense_candidates[:5]:
            print(
                f"  Rank {r['rank']} | Score: {r['score']:.4f} | "
                f"{r['doc_number']} {r['article']} {r.get('clause') or ''} {r.get('point') or ''} "
                f"({r['chunk_id']})\n"
                f"    Preview: {preview_text(r['content'])}"
            )

        # Print BM25 Top 5
        print("\n[B] BM25 TOP 5 (BM25Okapi):")
        for r in bm25_candidates[:5]:
            print(
                f"  Rank {r['rank']} | Score: {r['score']:.4f} | "
                f"{r['doc_number']} {r['article']} {r.get('clause') or ''} {r.get('point') or ''} "
                f"({r['chunk_id']})\n"
                f"    Preview: {preview_text(r['content'])}"
            )

        # Print Hybrid Top 5
        print("\n[C] HYBRID TOP 5 (RRF k=60):")
        for r in hybrid_results:
            dense_info = (
                f"Rank {r['dense_rank']} (score {r['dense_score']:.4f})"
                if r["dense_rank"] is not None
                else "None"
            )
            bm25_info = (
                f"Rank {r['bm25_rank']} (score {r['bm25_score']:.4f})"
                if r["bm25_rank"] is not None
                else "None"
            )
            print(
                f"  Rank {r['rank']} | RRF Score: {r['rrf_score']:.6f}\n"
                f"    Dense: {dense_info}\n"
                f"    BM25 : {bm25_info}\n"
                f"    Document: {r['doc_number']} | {r['article']} {r.get('clause') or ''} {r.get('point') or ''} | Chunk ID: {r['chunk_id']}\n"
                f"    Content preview: {preview_text(r['content'])}"
            )

        # Query-specific tracking
        if idx == 1:
            tracked_id = "L36_DIEU11_KHOAN4_DIEMC"
            d_rank = next((r["rank"] for r in dense_candidates if r["chunk_id"] == tracked_id), None)
            b_rank = next((r["rank"] for r in bm25_candidates if r["chunk_id"] == tracked_id), None)
            h_rank = next((r["rank"] for r in hybrid_results if r["chunk_id"] == tracked_id), None)
            print(f"\n  >> TRACKING '{tracked_id}' ('Tín hiệu đèn màu đỏ là cấm đi'):")
            print(f"     Dense rank : {d_rank if d_rank else '> 20'}")
            print(f"     BM25 rank  : {b_rank if b_rank else '> 20'}")
            print(f"     Hybrid rank: {h_rank if h_rank else '> 5'}")

        print("\n" + "=" * 85 + "\n")

    print("PERFORMANCE SUMMARY:")
    print(f"  Average Dense Latency : {sum(dense_latencies) / len(dense_latencies):.2f} ms")
    print(f"  Average BM25 Latency  : {sum(bm25_latencies) / len(bm25_latencies):.2f} ms")
    print(f"  Average Hybrid Latency: {sum(hybrid_latencies) / len(hybrid_latencies):.2f} ms")
    print("=" * 85)


if __name__ == "__main__":
    vector_dir = PROJECT_ROOT / "data" / "vector_store"
    chunks_file = PROJECT_ROOT / "data" / "chunks" / "legal_chunks.json"
    run_smoke_test(vector_dir, chunks_file)
