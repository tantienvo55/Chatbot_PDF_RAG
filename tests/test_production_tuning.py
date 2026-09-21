"""
Unit tests for Production Tuning enhancements (Prompt 8.6).
Mock-based tests verifying streaming, think-tag suppression, TTFT, num_predict,
keep-alive, dynamic top-k, context deduplication, and streaming invariants.
"""

import json
from typing import Any
from unittest.mock import MagicMock
import pytest
import httpx

from src.generation.qwen_client import QwenClient, StreamThinkFilter, warm_up_model
from src.generation.prompt_builder import PromptBuilder
from src.generation.rag_generator import RAGGenerator
from src.query.query_analyzer import QueryAnalyzer, select_retrieval_top_k
from src.query.clarification import ConversationState


# =============================================================================
# 1. STREAM THINK FILTER TESTS
# =============================================================================

def test_stream_think_filter_split_across_chunks():
    """Verify that <think>...</think> split arbitrarily across chunks is completely suppressed."""
    filter_think = StreamThinkFilter()
    chunks = [
        "<th",
        "ink>\nStep 1: thinking...\n",
        "Step 2: still thinking...\n",
        "</th",
        "ink>\nXin chào bạn!",
    ]
    output = []
    for c in chunks:
        out = filter_think.feed(c)
        if out:
            output.append(out)
    final = filter_think.finalize()
    if final:
        output.append(final)

    result = "".join(output)
    assert "<think>" not in result
    assert "</think>" not in result
    assert "thinking" not in result
    assert "Xin chào bạn!" in result


def test_stream_think_filter_no_think_tags():
    """Verify standard text without think tags passes through transparently."""
    filter_think = StreamThinkFilter()
    chunks = ["Theo quy định ", "tại Điều 11, ", "đèn đỏ là cấm đi."]
    output = []
    for c in chunks:
        out = filter_think.feed(c)
        if out:
            output.append(out)
    final = filter_think.finalize()
    if final:
        output.append(final)

    result = "".join(output)
    assert result == "Theo quy định tại Điều 11, đèn đỏ là cấm đi."


def test_stream_think_filter_bracket_characters_not_think():
    """Verify that '<' characters not forming <think> are preserved."""
    filter_think = StreamThinkFilter()
    chunks = ["Khi tốc độ v < 50 km/h và ", "khoảng cách < 10m."]
    output = []
    for c in chunks:
        out = filter_think.feed(c)
        if out:
            output.append(out)
    final = filter_think.finalize()
    if final:
        output.append(final)

    result = "".join(output)
    assert "v < 50 km/h" in result
    assert "< 10m" in result


# =============================================================================
# 2. QWEN CLIENT STREAMING & CONFIG TESTS
# =============================================================================

def test_qwen_client_num_predict_and_keep_alive_payload():
    """Verify num_predict and keep_alive options are propagated in API payload."""
    mock_http = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "message": {"role": "assistant", "content": "Test answer."}
    }
    mock_http.post.return_value = mock_resp

    client = QwenClient(
        num_predict=384,
        keep_alive="10m",
        client=mock_http,
    )
    client.generate("sys", "user")

    call_kwargs = mock_http.post.call_args[1]
    payload = call_kwargs["json"]
    assert payload["options"]["num_predict"] == 384
    assert payload["keep_alive"] == "10m"


def test_qwen_client_generate_stream_tokens_and_ttft():
    """Verify generate_stream yields tokens and records TTFT."""
    mock_http = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200

    stream_lines = [
        json.dumps({"message": {"content": "Theo "}, "done": False}),
        json.dumps({"message": {"content": "Luật "}, "done": False}),
        json.dumps({"message": {"content": "36/2024."}, "done": True, "eval_count": 5}),
    ]
    mock_resp.iter_lines.return_value = stream_lines

    # Context manager for client.stream
    mock_stream_ctx = MagicMock()
    mock_stream_ctx.__enter__.return_value = mock_resp
    mock_http.stream.return_value = mock_stream_ctx

    client = QwenClient(client=mock_http)
    tokens = list(client.generate_stream("sys", "user"))

    assert tokens == ["Theo ", "Luật ", "36/2024."]
    assert client.last_ttft_ms is not None
    assert client.last_ttft_ms >= 0.0
    assert client.last_metrics.get("eval_count") == 5


def test_warm_up_model_safe_handling():
    """Verify warm_up_model catches errors and returns True/False without crashing."""
    # Failure case: connection error
    mock_fail_http = MagicMock(spec=httpx.Client)
    mock_fail_http.post.side_effect = httpx.ConnectError("Ollama not running")
    client_fail = QwenClient(client=mock_fail_http)
    assert warm_up_model(client_fail) is False

    # Success case
    mock_succ_http = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"message": {"content": "hi"}}
    mock_succ_http.post.return_value = mock_resp
    client_succ = QwenClient(client=mock_succ_http)
    assert warm_up_model(client_succ) is True


# =============================================================================
# 3. DYNAMIC TOP-K & CALLER OVERRIDE TESTS
# =============================================================================

def test_select_retrieval_top_k_rules():
    """Verify dynamic top-k rules: complex inquiries get 5, simple get 3."""
    analyzer = QueryAnalyzer()

    # Complex: Alcohol penalty inquiries -> 5
    a1 = analyzer.analyze("Nồng độ cồn xe máy phạt bao nhiêu?")
    assert select_retrieval_top_k(a1) == 5

    # Complex: Points deduction inquiries -> 5
    a2 = analyzer.analyze("Giấy phép lái xe bị trừ điểm trong trường hợp nào?")
    assert select_retrieval_top_k(a2) == 5

    # Complex: Multiple conditions / enumeration cues -> 5
    a3 = analyzer.analyze("Những trường hợp nào không được vượt xe?")
    assert select_retrieval_top_k(a3) == 5

    # Simple: General traffic rule / signal meaning -> 3
    a4 = analyzer.analyze("Đèn đỏ có được đi không?")
    assert select_retrieval_top_k(a4) == 3

    # Simple: Definition -> 3
    a5 = analyzer.analyze("Thế nào là dừng xe và đỗ xe?")
    assert select_retrieval_top_k(a5) == 3


def test_caller_explicit_top_k_override_respected():
    """Verify that caller-specified top_k is NEVER overridden by dynamic top-k."""
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [
        {"chunk_id": "C1", "content": "Điều 1", "doc_number": "36", "article": "Điều 1"}
    ]
    mock_qwen = MagicMock()
    mock_qwen.generate.return_value = "Câu trả lời."

    generator = RAGGenerator(
        retriever=mock_retriever,
        qwen_client=mock_qwen,
        enable_dynamic_top_k=True,
    )

    # Simple query would dynamically choose 3, but caller specifies top_k=7
    generator.generate("Đèn đỏ có được đi không?", top_k=7)
    mock_retriever.search.assert_called_with("Đèn đỏ có được đi không?", top_k=7)

    # Complex query would dynamically choose 5, but caller specifies top_k=2
    generator.generate("Nồng độ cồn xe máy phạt bao nhiêu?", top_k=2)
    mock_retriever.search.assert_called_with("Nồng độ cồn xe máy phạt bao nhiêu?", top_k=2)


# =============================================================================
# 4. CONTEXT DEDUPLICATION & TRIMMING TESTS
# =============================================================================

def test_context_chunk_deduplication():
    """Verify that PromptBuilder removes duplicate chunk_ids from context."""
    pb = PromptBuilder(max_context_chars=5000)
    chunks = [
        {"chunk_id": "CHUNK_A", "content": "Nội dung A", "doc_number": "168", "article": "Điều 7"},
        {"chunk_id": "CHUNK_A", "content": "Nội dung A trùng lặp", "doc_number": "168", "article": "Điều 7"},
        {"chunk_id": "CHUNK_B", "content": "Nội dung B", "doc_number": "168", "article": "Điều 8"},
    ]
    context = pb.build_context(chunks)

    # CHUNK_A should appear only once as a source block
    assert context.count("[SOURCE 1]") == 1
    assert context.count("[SOURCE 2]") == 1
    assert "[SOURCE 3]" not in context
    assert "Nội dung B" in context


def test_context_boundary_truncation():
    """Verify context stops adding chunks at chunk boundary when exceeding max_context_chars."""
    pb = PromptBuilder(max_context_chars=200)
    chunks = [
        {"chunk_id": "C1", "content": "A" * 100, "doc_number": "36", "article": "Điều 1"},
        {"chunk_id": "C2", "content": "B" * 150, "doc_number": "36", "article": "Điều 2"},
    ]
    context = pb.build_context(chunks)
    assert "[SOURCE 1]" in context
    assert "[SOURCE 2]" not in context  # C2 excluded cleanly without cutting mid-chunk


# =============================================================================
# 5. STREAMING INVARIANT & EVENT SCHEMA TESTS
# =============================================================================

def test_rag_streaming_answer_invariant_and_events():
    """
    CRITICAL INVARIANT TEST:
    Verify that final_response['answer'] exactly equals the concatenation of all token events,
    and event sequence adheres to status -> token -> final.
    """
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [
        {
            "chunk_id": "L36_D11",
            "content": "Tín hiệu đèn màu đỏ là cấm đi.",
            "doc_number": "36/2024/QH15",
            "article": "Điều 11",
            "document_type": "law",
        }
    ]

    mock_qwen = MagicMock()
    # Stream returns 4 token pieces
    mock_qwen.generate_stream.return_value = iter([
        "Đèn đỏ ",
        "là tín hiệu ",
        "cấm đi ",
        "theo quy định.",
    ])
    mock_qwen.last_ttft_ms = 42.5
    mock_qwen.last_metrics = {"eval_count": 10}

    generator = RAGGenerator(
        retriever=mock_retriever,
        qwen_client=mock_qwen,
    )

    events = list(generator.generate_stream("Đèn đỏ có được đi không?"))

    # Event types verification
    event_types = [e["type"] for e in events]
    assert event_types[0] == "status"
    assert event_types[-1] == "final"
    assert "token" in event_types

    # Collect token pieces
    tokens = [e["text"] for e in events if e["type"] == "token"]
    assert tokens == ["Đèn đỏ ", "là tín hiệu ", "cấm đi ", "theo quy định."]

    # Final event payload
    final_event = events[-1]
    response = final_event["response"]
    assert response["status"] == "ANSWER"

    # INVARIANT: final answer == concat(tokens)
    assert response["answer"] == "".join(tokens)

    # Citations validity
    assert len(response["citations"]) == 1
    assert response["citations"][0]["chunk_id"] == "L36_D11"

    # Latency tracking includes TTFT
    assert response["latency"]["ttft_ms"] == 42.5


def test_rag_streaming_clarify_bypasses_llm():
    """Verify that ambiguous queries emit CLARIFY immediately without calling Qwen."""
    mock_retriever = MagicMock()
    mock_qwen = MagicMock()

    generator = RAGGenerator(retriever=mock_retriever, qwen_client=mock_qwen)
    events = list(generator.generate_stream("Nồng độ cồn phạt bao nhiêu?"))

    # LLM and retrieval must NOT be invoked
    mock_retriever.search.assert_not_called()
    mock_qwen.generate_stream.assert_not_called()

    event_types = [e["type"] for e in events]
    assert event_types == ["status", "status", "final"]
    assert events[-1]["response"]["status"] == "CLARIFY"
    assert events[-1]["response"]["clarification"] is not None


def test_rag_streaming_out_of_scope_bypasses_llm():
    """Verify that out-of-scope queries emit OUT_OF_SCOPE immediately without calling Qwen."""
    mock_retriever = MagicMock()
    mock_qwen = MagicMock()

    generator = RAGGenerator(retriever=mock_retriever, qwen_client=mock_qwen)
    events = list(generator.generate_stream("Thuế thu nhập cá nhân tính thế nào?"))

    # LLM and retrieval must NOT be invoked
    mock_retriever.search.assert_not_called()
    mock_qwen.generate_stream.assert_not_called()

    event_types = [e["type"] for e in events]
    assert event_types == ["status", "status", "final"]
    assert events[-1]["response"]["status"] == "OUT_OF_SCOPE"


def test_rag_streaming_answer_exact_match_tokens_with_think_tags():
    """
    CRITICAL INVARIANT TEST (Adjustment 5):
    Verify that streaming final answer must exactly equal the concatenation
    of user-visible token events after think-tag suppression.
    """
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [
        {
            "chunk_id": "L36_D11",
            "content": "Tín hiệu đèn màu đỏ là cấm đi.",
            "doc_number": "36/2024/QH15",
            "article": "Điều 11",
            "document_type": "law",
        }
    ]

    mock_http = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200

    stream_lines = [
        json.dumps({"message": {"content": "<th"}, "done": False}),
        json.dumps({"message": {"content": "ink>\nTôi đang suy nghĩ nội bộ...\n"}, "done": False}),
        json.dumps({"message": {"content": "</think>\nTheo "}, "done": False}),
        json.dumps({"message": {"content": "quy định tại "}, "done": False}),
        json.dumps({"message": {"content": "Điều 11 Luật 36/2024, "}, "done": False}),
        json.dumps({"message": {"content": "đèn đỏ là cấm đi."}, "done": True, "eval_count": 12}),
    ]
    mock_resp.iter_lines.return_value = stream_lines

    mock_stream_ctx = MagicMock()
    mock_stream_ctx.__enter__.return_value = mock_resp
    mock_http.stream.return_value = mock_stream_ctx

    qwen = QwenClient(client=mock_http)
    generator = RAGGenerator(
        retriever=mock_retriever,
        qwen_client=qwen,
    )

    events = list(generator.generate_stream("Đèn đỏ có được đi không?"))

    # Extract all streamed tokens
    streamed_tokens = [e["text"] for e in events if e["type"] == "token"]
    concatenated_tokens = "".join(streamed_tokens)

    # Verify no think tags in streamed tokens
    assert "<think>" not in concatenated_tokens
    assert "</think>" not in concatenated_tokens
    assert "suy nghĩ nội bộ" not in concatenated_tokens

    # Verify final response matches invariant
    final_event = [e for e in events if e["type"] == "final"][0]
    final_response = final_event["response"]

    assert final_response["status"] == "ANSWER"
    # STRICT INVARIANT ASSERTION
    assert final_response["answer"] == concatenated_tokens
    assert final_response["answer"] == "Theo quy định tại Điều 11 Luật 36/2024, đèn đỏ là cấm đi."

