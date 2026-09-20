"""
Smoke test for BGE-M3 Dense Retrieval with FAISS on 5 real legal queries.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.bge_embedder import BGEEmbedder
from src.retrieval.faiss_store import FAISSStore

SMOKE_QUERIES = [
    "Có được vượt đèn đỏ không?",
    "Không đội mũ bảo hiểm bị xử phạt như thế nào?",
    "Quy định về nồng độ cồn khi lái xe là gì?",
    "Giấy phép lái xe bị trừ điểm trong trường hợp nào?",
    "Người điều khiển xe máy có nghĩa vụ gì?",
]


def run_smoke_test(store_dir: Path, embedder: BGEEmbedder | None = None) -> None:
    print("=" * 80)
    print("BGE-M3 + FAISS DENSE RETRIEVAL — SMOKE TEST")
    print("=" * 80)

    store = FAISSStore.load(store_dir, embedder=embedder)
    print(f"Loaded store from {store_dir}")
    print(f"Index vectors: {store.index.ntotal} | Dimension: {store.index.d}")
    print(f"Model: {store.manifest.get('embedding_model')}")
    print("=" * 80 + "\n")

    for idx, query in enumerate(SMOKE_QUERIES, start=1):
        print(f"Query [{idx}/5]: {query}")
        print("-" * 60)
        results = store.search(query, top_k=5)
        print("Top 5:")
        for r in results:
            content_preview = r["content"][:160].replace("\n", " ")
            if len(r["content"]) > 160:
                content_preview += "…"
            print(f"  Rank: {r['rank']}")
            print(f"  Score: {r['score']:.4f}")
            print(f"  Document: {r['doc_number']}")
            print(f"  Article: {r['article']}")
            print(f"  Clause: {r['clause']}")
            print(f"  Point: {r['point']}")
            print(f"  Chunk ID: {r['chunk_id']}")
            print(f"  Content preview: {content_preview}")
            print()
        print("=" * 80 + "\n")


if __name__ == "__main__":
    vector_store_dir = PROJECT_ROOT / "data" / "vector_store"
    run_smoke_test(vector_store_dir)
