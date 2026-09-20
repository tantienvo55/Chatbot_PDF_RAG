"""
RAG Generator orchestrator module for Vietnamese Traffic Law QA.
Integrates QueryAnalyzer, ConversationState, Retrieval (dense/bm25/hybrid),
PromptBuilder, Local Qwen, strict Citation Consistency Validation, and 4 response modes.
"""

import time
from typing import Any, Optional, Union
import re

from ..query.query_analyzer import QueryAnalyzer, OUT_OF_SCOPE_MESSAGE
from ..query.clarification import ConversationState, resolve_clarification
from .qwen_client import QwenClient
from .prompt_builder import PromptBuilder


class CitationValidationError(Exception):
    """Raised when citation chunks are inconsistent with retrieved context chunks."""
    pass


INSUFFICIENT_CONTEXT_MESSAGE = (
    "Tôi chưa tìm thấy đủ căn cứ trong các văn bản pháp luật hiện có để trả lời câu hỏi này. "
    "Bạn có thể cung cấp thêm chi tiết (như loại phương tiện hoặc hành vi cụ thể) để tôi hỗ trợ tra cứu chính xác hơn."
)

# Multi-pattern regex for identifying when model states context is insufficient
INSUFFICIENT_PATTERNS = [
    r"chưa\s+tìm\s+thấy\s+đủ\s+căn\s+cứ",
    r"không\s+tìm\s+thấy\s+đủ\s+căn\s+cứ",
    r"không\s+có\s+đủ\s+căn\s+cứ",
    r"không\s+có\s+thông\s+tin\s+trong",
    r"không\s+được\s+đề\s+cập\s+trong",
    r"không\s+tìm\s+thấy\s+trong\s+tài\s+liệu",
    r"tài\s+liệu\s+không\s+quy\s+định",
    r"tài\s+liệu\s+được\s+cung\s+cấp\s+không",
    r"context\s+không\s+có",
    r"không\s+đủ\s+thông\s+tin\s+để\s+trả\s+lời",
    r"chưa\s+đủ\s+cơ\s+sở\s+để\s+khẳng\s+định",
]
INSUFFICIENT_RES = [re.compile(p, re.IGNORECASE) for p in INSUFFICIENT_PATTERNS]


class RAGGenerator:
    """
    Orchestrates end-to-end grounded RAG answer generation with local Qwen and clarification handling.
    """

    def __init__(
        self,
        retriever: Any,
        qwen_client: Optional[QwenClient] = None,
        query_analyzer: Optional[QueryAnalyzer] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        default_retrieval_mode: str = "dense",
        top_k: int = 5,
    ) -> None:
        """
        Initialize RAGGenerator.

        Args:
            retriever: Single retriever (FAISSStore, BM25Retriever, or HybridRetriever)
                       OR dict mapping mode name to retriever {'dense': ..., 'bm25': ..., 'hybrid': ...}.
            qwen_client: QwenClient instance (defaults to standard local client).
            query_analyzer: QueryAnalyzer instance (defaults to new QueryAnalyzer).
            prompt_builder: PromptBuilder instance (defaults to standard PromptBuilder).
            default_retrieval_mode: Mode to use ('dense', 'bm25', 'hybrid'). Default: 'dense'.
            top_k: Top-K retrieved chunks to use in context (default: 5).
        """
        self.retrievers: dict[str, Any]
        if isinstance(retriever, dict):
            self.retrievers = retriever
        else:
            self.retrievers = {default_retrieval_mode: retriever}

        self.default_retrieval_mode = default_retrieval_mode
        self.qwen_client = qwen_client or QwenClient()
        self.query_analyzer = query_analyzer or QueryAnalyzer()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.top_k = top_k

        # Hierarchical parent clause lookup to supply penalty context for point-level chunks
        self._parent_clause_map: dict[tuple[Any, Any, str], dict[str, Any]] = {}
        corpus_source = None
        for r in self.retrievers.values():
            if hasattr(r, "metadata") and isinstance(r.metadata, list):
                corpus_source = r.metadata
                break
            elif hasattr(r, "corpus") and isinstance(r.corpus, list):
                corpus_source = r.corpus
                break
            elif hasattr(r, "dense_retriever") and hasattr(r.dense_retriever, "metadata"):
                corpus_source = r.dense_retriever.metadata
                break

        if corpus_source:
            for c in corpus_source:
                if c.get("clause") and not c.get("point"):
                    cl = str(c.get("clause")).strip().rstrip(".")
                    key = (c.get("doc_number"), c.get("article"), cl)
                    if key not in self._parent_clause_map:
                        self._parent_clause_map[key] = c

    def _get_retriever(self, mode: str) -> Any:
        if mode in self.retrievers:
            return self.retrievers[mode]
        # Fallback to default if requested mode is not explicitly loaded
        if self.default_retrieval_mode in self.retrievers:
            return self.retrievers[self.default_retrieval_mode]
        raise ValueError(f"No retriever available for mode '{mode}'")

    def _detect_model_insufficient(self, answer: str) -> bool:
        """Check if model explicitly indicated that provided context is insufficient."""
        for pat in INSUFFICIENT_RES:
            if pat.search(answer):
                return True
        return False

    def _extract_citations(
        self,
        retrieved_chunks: list[dict[str, Any]],
        answer: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Extract authoritative citation metadata strictly from retrieved chunks.
        Never parses invented citations from free-form LLM text.
        """
        citations: list[dict[str, Any]] = []
        for r in retrieved_chunks:
            c = {
                "chunk_id": r.get("chunk_id"),
                "doc_number": r.get("doc_number"),
                "document_type": r.get("document_type"),
                "article": r.get("article"),
                "article_title": r.get("article_title"),
                "clause": r.get("clause"),
                "point": r.get("point"),
                "source_file": r.get("source_file"),
                "start_page": r.get("start_page"),
                "end_page": r.get("end_page"),
            }
            citations.append(c)
        return citations

    def _validate_citations(
        self,
        citations: list[dict[str, Any]],
        retrieved_chunks: list[dict[str, Any]],
    ) -> None:
        """
        Validate citation consistency: set(citation.chunk_id) ⊆ set(retrieved_chunk.chunk_id).
        Raises CitationValidationError if any citation is not present in retrieved chunks.
        """
        retrieved_ids = {r.get("chunk_id") for r in retrieved_chunks}
        citation_ids = {c.get("chunk_id") for c in citations}

        diff = citation_ids - retrieved_ids
        if diff:
            raise CitationValidationError(
                f"Citation validation failed: chunk_ids {diff} appear in citations but were not retrieved!"
            )

    def generate(
        self,
        query: str,
        state: Optional[ConversationState] = None,
        retrieval_mode: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Execute end-to-end RAG pipeline for a user query.

        Args:
            query: User's query text or clarification answer.
            state: Optional ConversationState for multi-turn dialogue.
            retrieval_mode: Override for retrieval mode ('dense', 'bm25', 'hybrid').
            top_k: Override for number of chunks to retrieve.

        Returns:
            Structured dictionary matching Prompt 8 specifications with latency metrics.
        """
        start_total = time.perf_counter()
        active_mode = retrieval_mode or self.default_retrieval_mode
        k = top_k or self.top_k

        # 1. Multi-turn clarification resolution
        resolved_query = query.strip()
        if state is not None and state.clarification_pending and state.original_query:
            resolved_query = resolve_clarification(state.original_query, query, state)

        # 2. Query Understanding & Pre-retrieval analysis
        t0 = time.perf_counter()
        analysis = self.query_analyzer.analyze(resolved_query)
        analysis_ms = (time.perf_counter() - t0) * 1000.0

        # Status: OUT_OF_SCOPE
        if analysis["is_out_of_scope"]:
            total_ms = (time.perf_counter() - start_total) * 1000.0
            return {
                "status": "OUT_OF_SCOPE",
                "query": query,
                "resolved_query": resolved_query,
                "answer": analysis.get("out_of_scope_message") or OUT_OF_SCOPE_MESSAGE,
                "retrieval_mode": active_mode,
                "citations": [],
                "retrieved_chunks": [],
                "clarification": None,
                "latency": {
                    "query_analysis_ms": round(analysis_ms, 2),
                    "retrieval_ms": 0.0,
                    "generation_ms": 0.0,
                    "total_ms": round(total_ms, 2),
                },
            }

        # Status: CLARIFY
        if analysis["is_ambiguous"]:
            if state is not None:
                state.original_query = resolved_query
                state.missing_slots = analysis["missing_slots"]
                state.clarification_pending = True
                state.last_clarification_question = analysis["clarification_question"]

            total_ms = (time.perf_counter() - start_total) * 1000.0
            return {
                "status": "CLARIFY",
                "query": query,
                "resolved_query": resolved_query,
                "answer": None,
                "retrieval_mode": active_mode,
                "citations": [],
                "retrieved_chunks": [],
                "clarification": {
                    "question": analysis["clarification_question"],
                    "missing_slots": analysis["missing_slots"],
                },
                "latency": {
                    "query_analysis_ms": round(analysis_ms, 2),
                    "retrieval_ms": 0.0,
                    "generation_ms": 0.0,
                    "total_ms": round(total_ms, 2),
                },
            }

        # 3. Retrieval
        retriever = self._get_retriever(active_mode)
        t_ret = time.perf_counter()
        retrieved_chunks = retriever.search(resolved_query, top_k=k)
        retrieval_ms = (time.perf_counter() - t_ret) * 1000.0

        # Hierarchically enrich point-level chunks with parent clause chunks to provide complete penalty context
        if self._parent_clause_map and retrieved_chunks:
            enriched = list(retrieved_chunks)
            seen_ids = {c["chunk_id"] for c in enriched}
            for r in retrieved_chunks:
                if r.get("point") and r.get("clause"):
                    cl = str(r.get("clause")).strip().rstrip(".")
                    key = (r.get("doc_number"), r.get("article"), cl)
                    parent = self._parent_clause_map.get(key)
                    if parent and parent.get("chunk_id") not in seen_ids:
                        seen_ids.add(parent["chunk_id"])
                        enriched.append(parent)
            retrieved_chunks = enriched

        # Status: INSUFFICIENT_CONTEXT (0 chunks returned)
        if not retrieved_chunks:
            total_ms = (time.perf_counter() - start_total) * 1000.0
            return {
                "status": "INSUFFICIENT_CONTEXT",
                "query": query,
                "resolved_query": resolved_query,
                "answer": INSUFFICIENT_CONTEXT_MESSAGE,
                "retrieval_mode": active_mode,
                "citations": [],
                "retrieved_chunks": [],
                "clarification": None,
                "latency": {
                    "query_analysis_ms": round(analysis_ms, 2),
                    "retrieval_ms": round(retrieval_ms, 2),
                    "generation_ms": 0.0,
                    "total_ms": round(total_ms, 2),
                },
            }

        # 4. Build Context & User Prompt
        system_prompt = self.prompt_builder.system_prompt
        user_prompt = self.prompt_builder.build_user_prompt(resolved_query, retrieved_chunks)

        # 5. Local Qwen Generation
        t_gen = time.perf_counter()
        answer = self.qwen_client.generate(system_prompt, user_prompt)
        generation_ms = (time.perf_counter() - t_gen) * 1000.0

        # 6. Check if model signaled insufficient context
        if self._detect_model_insufficient(answer):
            status = "INSUFFICIENT_CONTEXT"
        else:
            status = "ANSWER"

        # 7. Extract citations & Validate consistency
        citations = self._extract_citations(retrieved_chunks, answer)
        self._validate_citations(citations, retrieved_chunks)

        # 8. Reset state if it was active
        if state is not None:
            state.reset()

        total_ms = (time.perf_counter() - start_total) * 1000.0
        return {
            "status": status,
            "query": query,
            "resolved_query": resolved_query,
            "answer": answer,
            "retrieval_mode": active_mode,
            "citations": citations,
            "retrieved_chunks": retrieved_chunks,
            "clarification": None,
            "latency": {
                "query_analysis_ms": round(analysis_ms, 2),
                "retrieval_ms": round(retrieval_ms, 2),
                "generation_ms": round(generation_ms, 2),
                "total_ms": round(total_ms, 2),
            },
        }
