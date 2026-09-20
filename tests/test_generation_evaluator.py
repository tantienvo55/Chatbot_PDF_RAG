"""
Unit tests for GenerationEvaluator module (Prompt 8.5).
Fully mocked/synthetic tests without calling live LLMs or BGE-M3.
"""

from typing import Any
import pytest

from src.evaluation.generation_evaluator import GenerationEvaluator, normalize_text, extract_money_figures
from src.query.clarification import ConversationState


class MockRAGGenerator:
    """Mock RAGGenerator for testing evaluation flows."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.call_count = 0

    def generate(self, query: str, state: Any = None, **kwargs) -> dict[str, Any]:
        resp = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return resp


def test_normalize_text():
    assert normalize_text("  Nghị định 168/2024/NĐ-CP  ") == "nghị định 168/2024/nđ-cp"
    assert normalize_text("") == ""


def test_extract_money_figures():
    text = "Phạt tiền từ 2.000.000 đồng đến 3.000.000 đồng hoặc 400 nghìn đồng và 6 triệu đồng."
    figures = extract_money_figures(text)
    assert "2.000.000" in figures
    assert "3.000.000" in figures
    assert any("400" in f for f in figures)
    assert any("6" in f for f in figures)


def test_retrieval_support_complete():
    evaluator = GenerationEvaluator()
    retrieved = [
        {"chunk_id": "C1", "content": "Phạt 2 triệu"},
        {"chunk_id": "C2", "content": "Phạt 6 triệu"},
        {"chunk_id": "C3", "content": "Phạt 8 triệu"},
    ]
    cit_reqs = [
        {"claim": "Mức 1", "acceptable_chunk_ids": ["C1", "P1"]},
        {"claim": "Mức 2", "acceptable_chunk_ids": ["C2", "P2"]},
        {"claim": "Mức 3", "acceptable_chunk_ids": ["C3", "P3"]},
    ]
    complete, missing = evaluator.evaluate_retrieval_support(retrieved, citation_requirements=cit_reqs)
    assert complete is True
    assert len(missing) == 0


def test_retrieval_support_incomplete():
    evaluator = GenerationEvaluator()
    # Missing C3/P3
    retrieved = [
        {"chunk_id": "C1", "content": "Phạt 2 triệu"},
        {"chunk_id": "C2", "content": "Phạt 6 triệu"},
    ]
    cit_reqs = [
        {"claim": "Mức 1", "acceptable_chunk_ids": ["C1", "P1"]},
        {"claim": "Mức 2", "acceptable_chunk_ids": ["C2", "P2"]},
        {"claim": "Mức 3", "acceptable_chunk_ids": ["C3", "P3"]},
    ]
    complete, missing = evaluator.evaluate_retrieval_support(retrieved, citation_requirements=cit_reqs)
    assert complete is False
    assert len(missing) == 1
    assert "Mức 3" in missing[0]


def test_completeness_gated_by_retrieval_support_when_incomplete():
    evaluator = GenerationEvaluator()
    case = {
        "id": "T01",
        "query": "Nồng độ cồn xe máy phạt bao nhiêu?",
        "expected_status": "ANSWER",
        "required_topics": ["2.000.000", "6.000.000", "8.000.000"],
        "citation_requirements": [
            {"claim": "Mức 1", "acceptable_chunk_ids": ["C1"]},
            {"claim": "Mức 2", "acceptable_chunk_ids": ["C2"]},
            {"claim": "Mức 3", "acceptable_chunk_ids": ["C3"]},
        ],
    }
    # Retrieved chunks only has C1 and C2 (C3 missing)
    mock_resp = {
        "status": "ANSWER",
        "answer": "Mức phạt là từ 2.000.000 đến 3.000.000 đồng và 6.000.000 đến 8.000.000 đồng.",
        "citations": [{"chunk_id": "C1", "doc_number": "168", "article": "Điều 7"}],
        "retrieved_chunks": [
            {"chunk_id": "C1", "content": "2.000.000 đồng", "doc_number": "168", "article": "Điều 7"},
            {"chunk_id": "C2", "content": "6.000.000 đồng", "doc_number": "168", "article": "Điều 7"},
        ],
        "latency": {"total_ms": 50.0},
    }
    generator = MockRAGGenerator([mock_resp])
    result = evaluator.evaluate_case(case, generator)

    assert result["retrieval_support_complete"] is False
    assert result["generation_completeness_evaluable"] is False
    assert result["completeness_status"] == "RETRIEVAL_SUPPORT_INCOMPLETE"
    # Crucial adjustment: LLM is NOT marked as failing completeness
    assert result["completeness_pass"] is None


def test_completeness_pass_when_retrieval_complete_and_all_topics_present():
    evaluator = GenerationEvaluator()
    case = {
        "id": "T02",
        "query": "Nồng độ cồn xe máy phạt bao nhiêu?",
        "expected_status": "ANSWER",
        "required_topics": ["2.000.000", "6.000.000", "8.000.000"],
        "citation_requirements": [
            {"claim": "Mức 1", "acceptable_chunk_ids": ["C1"]},
            {"claim": "Mức 2", "acceptable_chunk_ids": ["C2"]},
            {"claim": "Mức 3", "acceptable_chunk_ids": ["C3"]},
        ],
    }
    mock_resp = {
        "status": "ANSWER",
        "answer": "Phạt 2.000.000 đến 3.000.000đ; phạt 6.000.000 đến 8.000.000đ; và 8.000.000 đến 10.000.000đ.",
        "citations": [{"chunk_id": "C1", "doc_number": "168", "article": "Điều 7"}],
        "retrieved_chunks": [
            {"chunk_id": "C1", "content": "2.000.000 đồng", "doc_number": "168", "article": "Điều 7"},
            {"chunk_id": "C2", "content": "6.000.000 đồng", "doc_number": "168", "article": "Điều 7"},
            {"chunk_id": "C3", "content": "8.000.000 đồng", "doc_number": "168", "article": "Điều 7"},
        ],
        "latency": {"total_ms": 60.0},
    }
    generator = MockRAGGenerator([mock_resp])
    result = evaluator.evaluate_case(case, generator)

    assert result["retrieval_support_complete"] is True
    assert result["generation_completeness_evaluable"] is True
    assert result["completeness_status"] == "COMPLETENESS_PASS"
    assert result["completeness_pass"] is True
    assert len(result["required_topics_missing"]) == 0


def test_completeness_fail_when_retrieval_complete_but_topic_omitted():
    evaluator = GenerationEvaluator()
    case = {
        "id": "T03",
        "query": "Nồng độ cồn xe máy phạt bao nhiêu?",
        "expected_status": "ANSWER",
        "required_topics": ["2.000.000", "6.000.000", "10.000.000"],
        "citation_requirements": [
            {"claim": "Mức 1", "acceptable_chunk_ids": ["C1"]},
            {"claim": "Mức 2", "acceptable_chunk_ids": ["C2"]},
            {"claim": "Mức 3", "acceptable_chunk_ids": ["C3"]},
        ],
    }
    # LLM only generated bracket 1 and 2, omitting 10.000.000
    mock_resp = {
        "status": "ANSWER",
        "answer": "Phạt 2.000.000 đến 3.000.000đ và 6.000.000 đến 8.000.000đ.",
        "citations": [{"chunk_id": "C1", "doc_number": "168", "article": "Điều 7"}],
        "retrieved_chunks": [
            {"chunk_id": "C1", "content": "2.000.000 đồng", "doc_number": "168", "article": "Điều 7"},
            {"chunk_id": "C2", "content": "6.000.000 đồng", "doc_number": "168", "article": "Điều 7"},
            {"chunk_id": "C3", "content": "10.000.000 đồng", "doc_number": "168", "article": "Điều 7"},
        ],
        "latency": {"total_ms": 60.0},
    }
    generator = MockRAGGenerator([mock_resp])
    result = evaluator.evaluate_case(case, generator)

    assert result["retrieval_support_complete"] is True
    assert result["generation_completeness_evaluable"] is True
    assert result["completeness_status"] == "COMPLETENESS_FAIL"
    assert result["completeness_pass"] is False
    assert "10.000.000" in result["required_topics_missing"]


def test_claim_aware_citation_coverage():
    evaluator = GenerationEvaluator()
    # Acceptable chunks include point or parent clause
    cit_reqs = [
        {
            "claim": "Mức 1",
            "acceptable_chunk_ids": ["ND168_K6_DIEMA", "ND168_K6"],
        },
        {
            "claim": "Mức 2",
            "acceptable_chunk_ids": ["ND168_K8_DIEMB", "ND168_K8"],
        },
    ]

    # Citations cite parent clause for Mức 1 and point for Mức 2
    citations = [
        {"chunk_id": "ND168_K6"},
        {"chunk_id": "ND168_K8_DIEMB"},
    ]
    pass_cov, score, unsatisfied = evaluator.check_citation_coverage(citations, citation_requirements=cit_reqs)
    assert pass_cov is True
    assert score == 1.0
    assert len(unsatisfied) == 0

    # If only 1 claim cited
    citations_partial = [{"chunk_id": "ND168_K6"}]
    pass_cov2, score2, unsatisfied2 = evaluator.check_citation_coverage(citations_partial, citation_requirements=cit_reqs)
    assert pass_cov2 is False
    assert score2 == 0.5
    assert len(unsatisfied2) == 1
    assert unsatisfied2[0] == "Mức 2"


def test_citation_correctness():
    evaluator = GenerationEvaluator()
    retrieved = [
        {"chunk_id": "C1", "doc_number": "168/2024", "article": "Điều 7"},
        {"chunk_id": "C2", "doc_number": "36/2024", "article": "Điều 11"},
    ]

    # Valid citation
    valid_cits = [
        {"chunk_id": "C1", "doc_number": "168/2024", "article": "Điều 7"},
    ]
    is_valid, errs = evaluator.check_citation_correctness(valid_cits, retrieved)
    assert is_valid is True
    assert len(errs) == 0

    # Hallucinated chunk_id
    hallu_cits = [
        {"chunk_id": "C_FAKE", "doc_number": "168/2024", "article": "Điều 7"},
    ]
    is_valid2, errs2 = evaluator.check_citation_correctness(hallu_cits, retrieved)
    assert is_valid2 is False
    assert any("C_FAKE" in e for e in errs2)

    # Metadata mismatch
    mismatch_cits = [
        {"chunk_id": "C1", "doc_number": "WRONG_DOC", "article": "Điều 7"},
    ]
    is_valid3, errs3 = evaluator.check_citation_correctness(mismatch_cits, retrieved)
    assert is_valid3 is False
    assert any("Metadata mismatch" in e for e in errs3)


def test_forbidden_claims_and_ungrounded_fines():
    evaluator = GenerationEvaluator()
    answer = "Người vi phạm bị phạt tù chung thân và số tiền 50.000.000 đồng."
    retrieved = [
        {"chunk_id": "C1", "content": "Phạt tiền từ 2.000.000 đồng đến 3.000.000 đồng."},
    ]

    forbidden = ["tù chung thân"]
    found = evaluator.check_forbidden_claims(answer, forbidden)
    assert "tù chung thân" in found

    ungrounded = evaluator.check_fine_groundedness(answer, retrieved)
    assert "50.000.000" in ungrounded


def test_multi_turn_flow_evaluation():
    evaluator = GenerationEvaluator()
    case = {
        "id": "G011",
        "query": "Vượt đèn đỏ phạt bao nhiêu?",
        "expected_status": "CLARIFY",
        "is_multi_turn": True,
        "follow_up_query": "Xe máy",
        "follow_up_expected_status": "ANSWER",
        "follow_up_required_topics": ["4.000.000", "6.000.000"],
    }

    # Turn 1 returns CLARIFY, Turn 2 returns ANSWER
    r1 = {
        "status": "CLARIFY",
        "query": "Vượt đèn đỏ phạt bao nhiêu?",
        "clarification": {"question": "Bạn điều khiển loại xe gì?"},
        "retrieved_chunks": [],
        "citations": [],
        "latency": {"total_ms": 10.0},
    }
    r2 = {
        "status": "ANSWER",
        "query": "Xe máy",
        "answer": "Xe máy vượt đèn đỏ bị phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng.",
        "citations": [{"chunk_id": "C_RED", "doc_number": "168", "article": "Điều 7"}],
        "retrieved_chunks": [
            {"chunk_id": "C_RED", "content": "4.000.000 đến 6.000.000 đồng", "doc_number": "168", "article": "Điều 7"}
        ],
        "latency": {"total_ms": 100.0},
    }
    generator = MockRAGGenerator([r1, r2])
    result = evaluator.evaluate_case(case, generator)

    assert result["is_multi_turn"] is True
    assert result["multi_turn_resolved"] is True
    assert result["actual_status"] == "ANSWER"
    assert result["status_correct"] is True
    assert result["completeness_pass"] is True


def test_aggregate_summary():
    evaluator = GenerationEvaluator()
    results = [
        {
            "id": "1",
            "status_correct": True,
            "category": "general_rule",
            "expected_status": "ANSWER",
            "actual_status": "ANSWER",
            "retrieval_support_complete": True,
            "generation_completeness_evaluable": True,
            "completeness_pass": True,
            "citation_valid": True,
            "citation_coverage_score": 1.0,
            "groundedness_pass": True,
            "manual_review_required": False,
            "latency": {"generation_ms": 1000.0, "total_ms": 1200.0},
            "context_chars": 500,
            "answer_chars": 200,
        },
        {
            "id": "2",
            "status_correct": True,
            "category": "clarification_required",
            "expected_status": "CLARIFY",
            "actual_status": "CLARIFY",
            "citation_valid": True,
            "groundedness_pass": True,
            "manual_review_required": False,
            "latency": {"generation_ms": 0.0, "total_ms": 50.0},
            "context_chars": 0,
            "answer_chars": 0,
        },
        {
            "id": "3",
            "status_correct": True,
            "category": "out_of_scope",
            "expected_status": "OUT_OF_SCOPE",
            "actual_status": "OUT_OF_SCOPE",
            "refusal_correct": True,
            "citation_valid": True,
            "groundedness_pass": True,
            "manual_review_required": False,
            "latency": {"generation_ms": 0.0, "total_ms": 30.0},
            "context_chars": 0,
            "answer_chars": 50,
        },
    ]

    summary = evaluator.aggregate_summary(results)
    assert summary["total_cases"] == 3
    assert summary["status_accuracy"] == 1.0
    assert summary["clarification_required_accuracy"] == 1.0
    assert summary["completeness_pass_rate"] == 1.0
    assert summary["citation_correctness_rate"] == 1.0
    assert summary["groundedness_pass_rate"] == 1.0
    assert summary["refusal_correctness_rate"] == 1.0
    assert summary["latency"]["generation_ms"]["max"] == 1000.0
