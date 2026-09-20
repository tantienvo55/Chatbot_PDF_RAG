"""
Dense retrieval module using BGE-M3 embeddings and FAISS index.
"""

from .bge_embedder import BGEEmbedder
from .bm25_retriever import BM25Retriever, vietnamese_bm25_tokenizer
from .faiss_store import FAISSStore
from .hybrid_retriever import HybridRetriever

__all__ = [
    "BGEEmbedder",
    "FAISSStore",
    "BM25Retriever",
    "HybridRetriever",
    "vietnamese_bm25_tokenizer",
]


