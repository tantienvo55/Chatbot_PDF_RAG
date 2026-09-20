"""
BM25 Lexical Retrieval for Vietnamese legal chunks.

Implements a deterministic, Unicode-aware Vietnamese tokenizer preserving legal
entity codes, decimal numbers, currency amounts, and dates, with unigrams and
word bigrams for enhanced lexical matching. Integrates rank-bm25 (BM25Okapi)
over the legal chunks corpus.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# Regex pattern for legal tokens:
# 1. Document citation codes: e.g. 168/2024/nđ-cp, 36/2024/qh15, 238/2026/nđ-cp
# 2. Date formats: e.g. 26/12/2024, 01/01/2025
# 3. Formatted decimals & currency: e.g. 0,25, 100.000, 5.000.000
# 4. Standard Unicode words, syllables, numbers: [^\W_]+ (Unicode alphanumeric letters & digits)
LEGAL_TOKEN_PATTERN = re.compile(
    r"\b\d+/\d+/[^\W_]+(?:-[^\W_]+)*\b|"  # Legal doc code e.g. 168/2024/nđ-cp
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b|"       # Dates e.g. 26/12/2024
    r"\b\d+(?:[.,]\d+)+\b|"               # Decimals / currency e.g. 0,25, 100.000
    r"[^\W_]+",                            # Any Unicode word/syllable/number
    re.UNICODE,
)


def vietnamese_bm25_tokenizer(text: str, include_bigrams: bool = True) -> list[str]:
    """
    Unicode-aware Vietnamese lexical tokenizer for legal texts.

    Workflow:
    1. Unicode NFC normalization + lowercase.
    2. Extracts base lexical tokens (words, decimals, dates, legal codes)
       using Unicode-aware matching without manual character ranges.
    3. Adds consecutive word bigrams (e.g. 'mũ bảo hiểm' -> ['mũ', 'bảo', 'hiểm',
       'mũ_bảo', 'bảo_hiểm']) for enhanced lexical phrase matching.

    Args:
        text: Input string to tokenize.
        include_bigrams: Whether to generate consecutive word bigrams (default: True).

    Returns:
        List of lexical token strings.
    """
    if not text:
        return []

    # 1. Unicode NFC normalization and lowercasing
    norm_text = unicodedata.normalize("NFC", text).lower()

    # 2. Extract base unigram tokens
    unigrams = LEGAL_TOKEN_PATTERN.findall(norm_text)

    if not include_bigrams or len(unigrams) < 2:
        return unigrams

    # 3. Consecutive word bigrams
    bigrams = [f"{unigrams[i]}_{unigrams[i+1]}" for i in range(len(unigrams) - 1)]

    return unigrams + bigrams


class BM25Retriever:
    """
    Lexical retriever using BM25Okapi for Vietnamese legal chunks.

    Maintains deterministic 1:1 index position to chunk metadata mapping
    identical to FAISS vector index order.
    """

    def __init__(
        self,
        corpus: list[dict[str, Any]],
        tokenizer: Optional[Callable[[str], list[str]]] = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """
        Initialize BM25Retriever with chunk records and build BM25Okapi index.

        Args:
            corpus: List of chunk metadata dictionaries.
            tokenizer: Tokenizer function (default: vietnamese_bm25_tokenizer).
            k1: BM25 k1 parameter (default: 1.5).
            b: BM25 b parameter (default: 0.75).
        """
        if not corpus:
            raise ValueError("Corpus cannot be empty.")

        self.corpus = list(corpus)
        self.tokenizer = tokenizer or vietnamese_bm25_tokenizer
        self.k1 = k1
        self.b = b

        # Extract text for indexing: content_with_context preferred, fallback to content
        self.corpus_texts: list[str] = []
        for chunk in self.corpus:
            text = chunk.get("content_with_context") or chunk.get("content") or ""
            self.corpus_texts.append(text)

        # Tokenize corpus once
        logger.info("Tokenizing %d documents for BM25...", len(self.corpus_texts))
        self.tokenized_corpus: list[list[str]] = [
            self.tokenizer(text) for text in self.corpus_texts
        ]

        # Build BM25Okapi model once
        logger.info("Building BM25Okapi index (k1=%.2f, b=%.2f)...", self.k1, self.b)
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)

    @classmethod
    def from_json(
        cls,
        chunks_path: Path,
        tokenizer: Optional[Callable[[str], list[str]]] = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> BM25Retriever:
        """
        Build BM25Retriever from a legal chunks JSON file.

        Args:
            chunks_path: Path to legal_chunks.json.
            tokenizer: Optional custom tokenizer.
            k1: BM25 k1 parameter.
            b: BM25 b parameter.

        Returns:
            Instantiated BM25Retriever.
        """
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found at: {chunks_path}")

        with chunks_path.open("r", encoding="utf-8") as f:
            chunks = json.load(f)

        if not isinstance(chunks, list) or len(chunks) == 0:
            raise ValueError(f"Invalid or empty chunks file at: {chunks_path}")

        logger.info("Loaded %d legal chunks from %s", len(chunks), chunks_path)
        return cls(corpus=chunks, tokenizer=tokenizer, k1=k1, b=b)

    @property
    def doc_count(self) -> int:
        """Return total number of documents in BM25 index."""
        return len(self.corpus)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Perform BM25 lexical top-K retrieval for a user query.

        Args:
            query: Non-empty search query string.
            top_k: Number of top ranked chunks to retrieve (default: 5).

        Returns:
            List of result dicts ranked by raw BM25 score descending.
            Returns empty list [] if query has zero match (all scores <= 0).
        """
        if query is None or not query.strip():
            raise ValueError("Query cannot be empty or whitespace only.")

        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        actual_k = min(top_k, len(self.corpus))

        # Tokenize query using exact same tokenizer
        query_tokens = self.tokenizer(query.strip())
        if not query_tokens:
            return []

        # Calculate BM25 scores
        raw_scores = self.bm25.get_scores(query_tokens)
        scores_arr = np.asarray(raw_scores, dtype=np.float32)

        # Check if all scores are <= 0 (no lexical match in vocabulary)
        if len(scores_arr) == 0 or np.max(scores_arr) <= 0.0:
            return []

        # Rank indices in descending order of BM25 score
        ranked_indices = np.argsort(scores_arr)[::-1]

        results: list[dict[str, Any]] = []
        seen_chunk_ids: set[str] = set()

        for pos in ranked_indices:
            score = float(scores_arr[pos])
            # Drop zero or negative score results
            if score <= 0.0:
                break

            if len(results) >= actual_k:
                break

            chunk_meta = self.corpus[pos]
            chunk_id = chunk_meta.get("chunk_id", f"POS_{pos}")

            if chunk_id in seen_chunk_ids:
                logger.error("Duplicate chunk_id '%s' encountered in BM25 results", chunk_id)
                continue
            seen_chunk_ids.add(chunk_id)

            rank = len(results) + 1
            item: dict[str, Any] = {
                "rank": rank,
                "score": score,
                "chunk_id": chunk_id,
                "doc_number": chunk_meta.get("doc_number"),
                "document_type": chunk_meta.get("document_type"),
                "article": chunk_meta.get("article"),
                "article_title": chunk_meta.get("article_title"),
                "clause": chunk_meta.get("clause"),
                "point": chunk_meta.get("point"),
                "content": chunk_meta.get("content"),
                "content_with_context": chunk_meta.get("content_with_context"),
                "source_file": chunk_meta.get("source_file"),
                "start_page": chunk_meta.get("start_page"),
                "end_page": chunk_meta.get("end_page"),
                "issue_date": chunk_meta.get("issue_date"),
                "effective_date": chunk_meta.get("effective_date"),
                "legal_status": chunk_meta.get("legal_status"),
                "amends_document": chunk_meta.get("amends_document"),
                "amended_article": chunk_meta.get("amended_article"),
                "amended_clause": chunk_meta.get("amended_clause"),
                "amended_point": chunk_meta.get("amended_point"),
            }
            if "merged_from_chunk_ids" in chunk_meta:
                item["merged_from_chunk_ids"] = chunk_meta["merged_from_chunk_ids"]

            results.append(item)

        return results
