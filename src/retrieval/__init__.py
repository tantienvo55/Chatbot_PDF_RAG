"""
Dense retrieval module using BGE-M3 embeddings and FAISS index.
"""

from .bge_embedder import BGEEmbedder
from .faiss_store import FAISSStore

__all__ = ["BGEEmbedder", "FAISSStore"]
