"""
Hybrid Retrieval module using Reciprocal Rank Fusion (RRF).

Combines dense semantic retrieval (FAISSStore backed by BAAI/bge-m3) and
lexical retrieval (BM25Retriever with Vietnamese legal tokenizer).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Optional

from .bm25_retriever import BM25Retriever
from .faiss_store import FAISSStore

logger = logging.getLogger(__name__)


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hex digest of a file in streaming chunks."""
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class HybridRetriever:
    """
    Hybrid retriever combining Dense (FAISS) and Lexical (BM25) search
    using pure rank-based Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        dense_retriever: FAISSStore,
        bm25_retriever: BM25Retriever,
        rrf_k: int = 60,
        corpus_path: Optional[Path] = None,
        validate_consistency: bool = True,
    ) -> None:
        """
        Initialize HybridRetriever.

        Args:
            dense_retriever: Loaded FAISSStore instance.
            bm25_retriever: Loaded BM25Retriever instance.
            rrf_k: RRF constant parameter (default: 60).
            corpus_path: Optional path to legal_chunks.json for hash check.
            validate_consistency: Whether to validate corpus consistency and hash on startup.
        """
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k
        self.corpus_path = corpus_path

        if validate_consistency:
            self._validate_corpus_consistency()

    def _validate_corpus_consistency(self) -> None:
        """
        Validate consistency between Dense and Lexical stores:
        1. Canonical chunk_id set equality and absence of duplicates.
        2. Strict SHA-256 hash match between manifest and current corpus file.
        """
        dense_chunk_ids = [m.get("chunk_id") for m in self.dense_retriever.metadata]
        bm25_chunk_ids = [m.get("chunk_id") for m in self.bm25_retriever.corpus]

        # 1. Duplicate check
        if len(dense_chunk_ids) != len(set(dense_chunk_ids)):
            raise ValueError("Duplicate chunk_ids detected in FAISS metadata.")
        if len(bm25_chunk_ids) != len(set(bm25_chunk_ids)):
            raise ValueError("Duplicate chunk_ids detected in BM25 corpus.")

        # 2. Size and Set equality
        if len(dense_chunk_ids) != len(bm25_chunk_ids):
            raise ValueError(
                f"Corpus size mismatch between FAISS ({len(dense_chunk_ids)}) "
                f"and BM25 ({len(bm25_chunk_ids)})."
            )

        dense_id_set = set(dense_chunk_ids)
        bm25_id_set = set(bm25_chunk_ids)

        if dense_id_set != bm25_id_set:
            diff_dense = dense_id_set - bm25_id_set
            diff_bm25 = bm25_id_set - dense_id_set
            raise ValueError(
                f"Chunk ID sets do not match between FAISS and BM25. "
                f"In FAISS only: {len(diff_dense)}, in BM25 only: {len(diff_bm25)}."
            )

        # 3. Strict SHA-256 hash check if available
        manifest = getattr(self.dense_retriever, "manifest", {}) or {}
        expected_sha = manifest.get("corpus_sha256")

        target_corpus_path = self.corpus_path
        if target_corpus_path is None and "source_file" in manifest:
            src_candidate = Path(manifest["source_file"])
            if src_candidate.exists():
                target_corpus_path = src_candidate

        if expected_sha and target_corpus_path and target_corpus_path.exists():
            actual_sha = compute_sha256(target_corpus_path)
            if actual_sha != expected_sha:
                raise ValueError(
                    f"Corpus SHA-256 mismatch for '{target_corpus_path}'. "
                    f"Manifest expected '{expected_sha}', but current file hash is '{actual_sha}'. "
                    f"FAISS index must be rebuilt."
                )

        logger.info(
            "HybridRetriever consistency verified: %d canonical chunks, hash verified.",
            len(dense_id_set),
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        dense_top_k: int = 20,
        bm25_top_k: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Execute Hybrid Retrieval using Reciprocal Rank Fusion (RRF).

        Args:
            query: Non-empty search query string.
            top_k: Number of final ranked results to return (default: 5).
            dense_top_k: Number of candidate chunks from FAISS (default: 20).
            bm25_top_k: Number of candidate chunks from BM25 (default: 20).

        Returns:
            List of result dicts ranked by RRF score descending.
        """
        # 1. Query validation
        if query is None or not query.strip():
            raise ValueError("Query cannot be empty or whitespace only.")

        # 2. Top-k validation
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")
        if dense_top_k <= 0:
            raise ValueError(f"dense_top_k must be a positive integer, got {dense_top_k}")
        if bm25_top_k <= 0:
            raise ValueError(f"bm25_top_k must be a positive integer, got {bm25_top_k}")

        corpus_size = len(self.dense_retriever.metadata)
        actual_dense_k = min(dense_top_k, corpus_size)
        actual_bm25_k = min(bm25_top_k, corpus_size)
        actual_final_k = min(top_k, corpus_size)

        # 3. Retrieve candidates from both sources
        dense_results = self.dense_retriever.search(query.strip(), top_k=actual_dense_k)
        bm25_results = self.bm25_retriever.search(query.strip(), top_k=actual_bm25_k)

        # 4. Fallback handling
        if not dense_results and not bm25_results:
            return []

        # 5. Candidate fusion & RRF calculation
        candidates: dict[str, dict[str, Any]] = {}

        # Process Dense candidates (1-indexed ranks)
        for r in dense_results:
            cid = r["chunk_id"]
            rank = r["rank"]
            score = r["score"]
            rrf_contrib = 1.0 / (self.rrf_k + rank)

            candidates[cid] = {
                "chunk_id": cid,
                "dense_rank": rank,
                "dense_score": score,
                "bm25_rank": None,
                "bm25_score": None,
                "rrf_score": rrf_contrib,
                "metadata": r,
            }

        # Process BM25 candidates (1-indexed ranks)
        for r in bm25_results:
            cid = r["chunk_id"]
            rank = r["rank"]
            score = r["score"]
            rrf_contrib = 1.0 / (self.rrf_k + rank)

            if cid in candidates:
                candidates[cid]["bm25_rank"] = rank
                candidates[cid]["bm25_score"] = score
                candidates[cid]["rrf_score"] += rrf_contrib
            else:
                candidates[cid] = {
                    "chunk_id": cid,
                    "dense_rank": None,
                    "dense_score": None,
                    "bm25_rank": rank,
                    "bm25_score": score,
                    "rrf_score": rrf_contrib,
                    "metadata": r,
                }

        # 6. Deterministic tie-breaking
        # Missing rank penalties:
        missing_dense_penalty = dense_top_k + 1
        missing_bm25_penalty = bm25_top_k + 1

        def sort_key(c: dict[str, Any]) -> tuple:
            # Overlap priority: present in both retrievers gets 0, single-source gets 1
            is_in_both = 0 if (c["dense_rank"] is not None and c["bm25_rank"] is not None) else 1

            # Effective penalized ranks
            eff_dense_rank = c["dense_rank"] if c["dense_rank"] is not None else missing_dense_penalty
            eff_bm25_rank = c["bm25_rank"] if c["bm25_rank"] is not None else missing_bm25_penalty
            total_rank = eff_dense_rank + eff_bm25_rank

            return (
                -c["rrf_score"],
                is_in_both,
                total_rank,
                eff_dense_rank,
                eff_bm25_rank,
                c["chunk_id"],
            )

        ranked_candidates = sorted(candidates.values(), key=sort_key)

        # 7. Build output records
        results: list[dict[str, Any]] = []
        for rank_idx, c in enumerate(ranked_candidates[:actual_final_k], start=1):
            meta = c["metadata"]
            item: dict[str, Any] = {
                "rank": rank_idx,
                "rrf_score": float(c["rrf_score"]),
                "chunk_id": c["chunk_id"],
                "dense_rank": c["dense_rank"],
                "dense_score": float(c["dense_score"]) if c["dense_score"] is not None else None,
                "bm25_rank": c["bm25_rank"],
                "bm25_score": float(c["bm25_score"]) if c["bm25_score"] is not None else None,
                "doc_number": meta.get("doc_number"),
                "document_type": meta.get("document_type"),
                "article": meta.get("article"),
                "article_title": meta.get("article_title"),
                "clause": meta.get("clause"),
                "point": meta.get("point"),
                "content": meta.get("content"),
                "content_with_context": meta.get("content_with_context"),
                "source_file": meta.get("source_file"),
                "start_page": meta.get("start_page"),
                "end_page": meta.get("end_page"),
                "issue_date": meta.get("issue_date"),
                "effective_date": meta.get("effective_date"),
                "legal_status": meta.get("legal_status"),
                "amends_document": meta.get("amends_document"),
                "amended_article": meta.get("amended_article"),
                "amended_clause": meta.get("amended_clause"),
                "amended_point": meta.get("amended_point"),
            }
            if "merged_from_chunk_ids" in meta:
                item["merged_from_chunk_ids"] = meta["merged_from_chunk_ids"]

            results.append(item)

        return results
