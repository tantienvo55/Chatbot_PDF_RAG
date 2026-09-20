"""
Unit tests for RAGGenerator orchestration, response modes, citation security, and validation.
"""

import pytest
from unittest.mock import MagicMock

from src.generation.rag_generator import (
    RAGGenerator,
    CitationValidationError,
    INSUFFICIENT_CONTEXT_MESSAGE,
)
from src.generation.qwen_client import QwenClient
from src.query.query_analyzer import QueryAnalyzer
from src.query.clarification import ConversationState


@pytest.fixture
def mock_retrieved_chunks() -> list[dict]:
    return [
        {
            "rank": 1,
            "score": 0.85,
            "chunk_id": "ND168_DIEU7_KHOAN6_DIEMA",
            "doc_number": "168/2024/NĐ-CP",
            "document_type": "Nghị định",
            "article": "Điều 7",
            "article_title": "Xử phạt người điều khiển xe mô tô, xe gắn máy vi phạm quy định về giao thông đường bộ",
            "clause": "Khoản 6",
            "point": "Điểm a",
            "content": "Phạt tiền từ 2.000.000 đồng đến 3.000.000 đồng đối với người điều khiển xe thực hiện hành vi vi phạm: Điều khiển xe trên đường mà trong máu hoặc hơi thở có nồng độ cồn nhưng chưa vượt quá 50 miligam/100 mililit máu hoặc chưa vượt quá 0,25 miligam/1 lít khí thở.",
            "content_with_context": "[Nghị định 168/2024/NĐ-CP Điều 7 Khoản 6 Điểm a] Phạt tiền từ 2.000.000 đồng đến 3.000.000 đồng...",
            "source_file": "168_2024_ND-CP.pdf",
            "start_page": 22,
            "end_page": 22,
        },
        {
            "rank": 2,
            "score": 0.78,
            "chunk_id": "ND168_DIEU7_KHOAN7_DIEMB",
            "doc_number": "168/2024/NĐ-CP",
            "document_type": "Nghị định",
            "article": "Điều 7",
            "article_title": "Xử phạt người điều khiển xe mô tô, xe gắn máy...",
            "clause": "Khoản 7",
            "point": "Điểm b",
            "content": "Phạt tiền từ 4.000.000 đồng đến 5.000.000 đồng...",
            "content_with_context": "[Nghị định 168/2024/NĐ-CP Điều 7 Khoản 7 Điểm b] Phạt tiền từ 4.000.000 đồng đến 5.000.000 đồng...",
            "source_file": "168_2024_ND-CP.pdf",
            "start_page": 23,
            "end_page": 23,
        }
    ]


@pytest.fixture
def mock_retriever(mock_retrieved_chunks: list[dict]) -> MagicMock:
    retriever = MagicMock()
    retriever.search.return_value = mock_retrieved_chunks
    return retriever


@pytest.fixture
def mock_qwen() -> MagicMock:
    client = MagicMock(spec=QwenClient)
    client.generate.return_value = (
        "Theo Điểm a Khoản 6 Điều 7 Nghị định 168/2024/NĐ-CP, người điều khiển xe mô tô, "
        "xe gắn máy vi phạm nồng độ cồn chưa vượt quá 0,25 mg/l khí thở bị phạt từ 2.000.000 đến 3.000.000 đồng."
    )
    return client


def test_rag_generator_answer_mode(
    mock_retriever: MagicMock,
    mock_qwen: MagicMock,
    mock_retrieved_chunks: list[dict],
) -> None:
    """Test successful ANSWER mode."""
    generator = RAGGenerator(retriever=mock_retriever, qwen_client=mock_qwen)
    res = generator.generate("Nồng độ cồn xe máy phạt bao nhiêu?")

    assert res["status"] == "ANSWER"
    assert "2.000.000 đến 3.000.000 đồng" in res["answer"]
    assert res["clarification"] is None
    assert len(res["retrieved_chunks"]) == 2
    assert len(res["citations"]) == 2

    # Citations metadata structure verification
    first_cit = res["citations"][0]
    assert first_cit["chunk_id"] == "ND168_DIEU7_KHOAN6_DIEMA"
    assert first_cit["doc_number"] == "168/2024/NĐ-CP"
    assert first_cit["article"] == "Điều 7"
    assert first_cit["clause"] == "Khoản 6"
    assert first_cit["point"] == "Điểm a"
    assert first_cit["start_page"] == 22

    # Latency tracking
    assert "total_ms" in res["latency"]
    assert "retrieval_ms" in res["latency"]
    assert "generation_ms" in res["latency"]
    assert res["latency"]["total_ms"] >= 0


def test_rag_generator_clarify_mode(mock_retriever: MagicMock, mock_qwen: MagicMock) -> None:
    """Test CLARIFY mode does not call retrieval or generation."""
    generator = RAGGenerator(retriever=mock_retriever, qwen_client=mock_qwen)
    res = generator.generate("Nồng độ cồn phạt bao nhiêu?")

    assert res["status"] == "CLARIFY"
    assert res["answer"] is None
    assert res["clarification"] is not None
    assert "vehicle_type" in res["clarification"]["missing_slots"]
    assert res["citations"] == []
    assert res["retrieved_chunks"] == []

    # Verification that retriever and LLM were not invoked
    mock_retriever.search.assert_not_called()
    mock_qwen.generate.assert_not_called()


def test_rag_generator_out_of_scope_mode(mock_retriever: MagicMock, mock_qwen: MagicMock) -> None:
    """Test OUT_OF_SCOPE mode does not call retrieval or generation."""
    generator = RAGGenerator(retriever=mock_retriever, qwen_client=mock_qwen)
    res = generator.generate("Thuế thu nhập cá nhân tính thế nào?")

    assert res["status"] == "OUT_OF_SCOPE"
    assert "giao thông đường bộ" in res["answer"]
    assert res["citations"] == []
    assert res["retrieved_chunks"] == []

    mock_retriever.search.assert_not_called()
    mock_qwen.generate.assert_not_called()


def test_rag_generator_insufficient_context_empty_retrieval(mock_qwen: MagicMock) -> None:
    """Test INSUFFICIENT_CONTEXT when retrieval returns 0 chunks."""
    empty_retriever = MagicMock()
    empty_retriever.search.return_value = []

    generator = RAGGenerator(retriever=empty_retriever, qwen_client=mock_qwen)
    res = generator.generate("Đèn đỏ có được đi không?")

    assert res["status"] == "INSUFFICIENT_CONTEXT"
    assert "Tôi chưa tìm thấy đủ căn cứ" in res["answer"]
    assert res["citations"] == []
    assert res["retrieved_chunks"] == []
    mock_qwen.generate.assert_not_called()


def test_rag_generator_insufficient_context_model_signal(mock_retriever: MagicMock) -> None:
    """Test INSUFFICIENT_CONTEXT when model indicates insufficient basis."""
    qwen = MagicMock(spec=QwenClient)
    qwen.generate.return_value = (
        "Tôi chưa tìm thấy đủ căn cứ trong các tài liệu được cung cấp để khẳng định mức phạt này."
    )

    generator = RAGGenerator(retriever=mock_retriever, qwen_client=qwen)
    res = generator.generate("Đèn đỏ có được đi không?")

    assert res["status"] == "INSUFFICIENT_CONTEXT"
    assert len(res["retrieved_chunks"]) > 0


def test_rag_generator_multi_pattern_insufficient_signals(mock_retriever: MagicMock) -> None:
    """Test various natural phrasing signals for insufficient context."""
    test_phrases = [
        "Tài liệu được cung cấp không có thông tin về nội dung này.",
        "Hiện tại không có đủ căn cứ pháp lý trong tài liệu để trả lời câu hỏi.",
        "Quy định này không được đề cập trong văn bản được trích dẫn.",
        "Chưa đủ cơ sở để khẳng định quy định đối với trường hợp này.",
    ]
    for phrase in test_phrases:
        qwen = MagicMock(spec=QwenClient)
        qwen.generate.return_value = phrase
        generator = RAGGenerator(retriever=mock_retriever, qwen_client=qwen)
        res = generator.generate("Đèn đỏ có được đi không?")
        assert res["status"] == "INSUFFICIENT_CONTEXT", f"Failed for phrase: {phrase}"


def test_citation_consistency_validation_success(
    mock_retriever: MagicMock,
    mock_qwen: MagicMock,
) -> None:
    """Citation consistency passes when citations match retrieved chunks."""
    generator = RAGGenerator(retriever=mock_retriever, qwen_client=mock_qwen)
    res = generator.generate("Nồng độ cồn xe máy phạt bao nhiêu?")
    retrieved_ids = {r["chunk_id"] for r in res["retrieved_chunks"]}
    citation_ids = {c["chunk_id"] for c in res["citations"]}
    assert citation_ids.issubset(retrieved_ids)


def test_citation_consistency_validation_failure() -> None:
    """Citation consistency validator raises CitationValidationError on invalid chunk_id."""
    generator = RAGGenerator(retriever=MagicMock(), qwen_client=MagicMock())
    retrieved = [{"chunk_id": "VALID_CHUNK_1"}]
    invalid_citations = [{"chunk_id": "INVALID_CHUNK_999"}]

    with pytest.raises(CitationValidationError, match="Citation validation failed"):
        generator._validate_citations(invalid_citations, retrieved)


def test_citation_security_hallucination_defense(mock_retriever: MagicMock) -> None:
    """
    If Qwen hallucinates an invented article/decree in its text response,
    the authoritative citations list still ONLY contains verified retrieved chunk metadata.
    """
    hallucinating_qwen = MagicMock(spec=QwenClient)
    hallucinating_qwen.generate.return_value = (
        "Theo Điều 999 Nghị định 999/2099/NĐ-CP quy định mức phạt là 10.000.000 đồng."
    )
    generator = RAGGenerator(retriever=mock_retriever, qwen_client=hallucinating_qwen)
    res = generator.generate("Nồng độ cồn xe máy phạt bao nhiêu?")

    assert res["status"] == "ANSWER"
    # The citations list must NOT contain any hallucinated Điều 999
    for cit in res["citations"]:
        assert cit["chunk_id"] in ["ND168_DIEU7_KHOAN6_DIEMA", "ND168_DIEU7_KHOAN7_DIEMB"]
        assert cit["doc_number"] == "168/2024/NĐ-CP"
        assert cit["article"] == "Điều 7"


def test_multi_turn_clarification_dialogue(
    mock_retriever: MagicMock,
    mock_qwen: MagicMock,
) -> None:
    """
    Multi-turn conversation flow:
    Turn 1: 'Nồng độ cồn phạt bao nhiêu?' -> CLARIFY
    Turn 2: 'Xe máy' -> Context reconstructed & ANSWER returned
    """
    state = ConversationState()
    generator = RAGGenerator(retriever=mock_retriever, qwen_client=mock_qwen)

    # Turn 1
    t1 = generator.generate("Nồng độ cồn phạt bao nhiêu?", state=state)
    assert t1["status"] == "CLARIFY"
    assert state.clarification_pending is True
    assert "vehicle_type" in state.missing_slots
    mock_retriever.search.assert_not_called()

    # Turn 2: User responds
    t2 = generator.generate("Xe máy", state=state)
    assert t2["status"] == "ANSWER"
    assert "nồng độ cồn" in t2["resolved_query"].lower()
    assert ("mô tô" in t2["resolved_query"].lower() or "xe máy" in t2["resolved_query"].lower())
    assert state.clarification_pending is False

    # Retriever was called with reconstructed query
    mock_retriever.search.assert_called_once()
    retrieved_query_arg = mock_retriever.search.call_args[0][0]
    assert "nồng độ cồn" in retrieved_query_arg.lower()


def test_retrieval_mode_dispatch(mock_retrieved_chunks: list[dict], mock_qwen: MagicMock) -> None:
    """Test dispatching between dense, bm25, and hybrid retrievers."""
    dense_mock = MagicMock()
    dense_mock.search.return_value = mock_retrieved_chunks
    bm25_mock = MagicMock()
    bm25_mock.search.return_value = mock_retrieved_chunks
    hybrid_mock = MagicMock()
    hybrid_mock.search.return_value = mock_retrieved_chunks

    retrievers_dict = {
        "dense": dense_mock,
        "bm25": bm25_mock,
        "hybrid": hybrid_mock,
    }

    generator = RAGGenerator(retriever=retrievers_dict, qwen_client=mock_qwen, default_retrieval_mode="dense")

    # Call with dense
    res_dense = generator.generate("Nồng độ cồn xe máy phạt bao nhiêu?", retrieval_mode="dense")
    assert res_dense["retrieval_mode"] == "dense"
    dense_mock.search.assert_called_once()
    bm25_mock.search.assert_not_called()

    # Call with bm25
    res_bm25 = generator.generate("Nồng độ cồn xe máy phạt bao nhiêu?", retrieval_mode="bm25")
    assert res_bm25["retrieval_mode"] == "bm25"
    bm25_mock.search.assert_called_once()

    # Call with hybrid
    res_hybrid = generator.generate("Nồng độ cồn xe máy phạt bao nhiêu?", retrieval_mode="hybrid")
    assert res_hybrid["retrieval_mode"] == "hybrid"
    hybrid_mock.search.assert_called_once()
