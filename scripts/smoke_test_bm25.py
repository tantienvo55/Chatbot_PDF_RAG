"""
Smoke test for BM25 Lexical Retrieval on 5 real Vietnamese traffic law queries.
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

from src.retrieval.bm25_retriever import BM25Retriever

SMOKE_QUERIES = [
    "Có được vượt đèn đỏ không?",
    "Không đội mũ bảo hiểm bị xử phạt như thế nào?",
    "Quy định về nồng độ cồn khi lái xe là gì?",
    "Giấy phép lái xe bị trừ điểm trong trường hợp nào?",
    "Người điều khiển xe máy có nghĩa vụ gì?",
]


def run_smoke_test(corpus_path: Path) -> None:
    print("=" * 80)
    print("BM25 LEXICAL RETRIEVAL — SMOKE TEST")
    print("=" * 80)

    # 1. Measure BM25 build time
    t0 = time.perf_counter()
    retriever = BM25Retriever.from_json(corpus_path)
    build_time = time.perf_counter() - t0

    print(f"Corpus path: {corpus_path}")
    print(f"Indexed documents: {retriever.doc_count}")
    print(f"BM25 build time: {build_time:.4f} seconds")
    print("=" * 80 + "\n")

    # 2. Run queries and measure search latencies
    latencies: list[float] = []

    for idx, query in enumerate(SMOKE_QUERIES, start=1):
        q_start = time.perf_counter()
        results = retriever.search(query, top_k=5)
        q_duration = time.perf_counter() - q_start
        latencies.append(q_duration)

        print(f"Query [{idx}/5]: {query}")
        print(f"Search latency: {q_duration * 1000:.2f} ms")
        print("-" * 60)
        print("Top 5:")
        for r in results:
            content_preview = r["content"][:160].replace("\n", " ")
            if len(r["content"]) > 160:
                content_preview += "…"
            print(f"  Rank: {r['rank']}")
            print(f"  BM25 Score: {r['score']:.4f}")
            print(f"  Document: {r['doc_number']}")
            print(f"  Article: {r['article']}")
            print(f"  Clause: {r['clause']}")
            print(f"  Point: {r['point']}")
            print(f"  Chunk ID: {r['chunk_id']}")
            print(f"  Content preview: {content_preview}")
            print()
        print("=" * 80 + "\n")

    avg_latency = (sum(latencies) / len(latencies)) * 1000
    print(f"Performance Summary:")
    print(f"  BM25 Build Time: {build_time:.4f}s ({build_time * 1000:.1f}ms)")
    print(f"  Average Search Latency: {avg_latency:.2f}ms across {len(SMOKE_QUERIES)} queries")
    print("=" * 80)


if __name__ == "__main__":
    corpus_file = PROJECT_ROOT / "data" / "chunks" / "legal_chunks.json"
    run_smoke_test(corpus_file)
