"""
Unit tests for QueryAnalyzer, slot extraction, ambiguity detection, and clarification resolution.
"""

import pytest
from src.query.query_analyzer import QueryAnalyzer
from src.query.clarification import ConversationState, resolve_clarification, parse_slot_from_answer


@pytest.fixture
def analyzer() -> QueryAnalyzer:
    return QueryAnalyzer()


def test_empty_query_raises(analyzer: QueryAnalyzer) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        analyzer.analyze("")
    with pytest.raises(ValueError, match="cannot be empty"):
        analyzer.analyze("   ")


def test_query_a_alcohol_missing_vehicle(analyzer: QueryAnalyzer) -> None:
    """Test A: 'Nồng độ cồn phạt bao nhiêu?' -> CLARIFY, missing vehicle_type."""
    res = analyzer.analyze("Nồng độ cồn phạt bao nhiêu?")
    assert res["is_out_of_scope"] is False
    assert res["is_ambiguous"] is True
    assert "vehicle_type" in res["missing_slots"]
    assert res["clarification_question"] is not None
    assert "loại phương tiện" in res["clarification_question"].lower() or "ô tô" in res["clarification_question"].lower()


def test_query_b_alcohol_with_vehicle(analyzer: QueryAnalyzer) -> None:
    """Test B: 'Nồng độ cồn xe máy phạt bao nhiêu?' -> has vehicle_type, not ambiguous."""
    res = analyzer.analyze("Nồng độ cồn xe máy phạt bao nhiêu?")
    assert res["is_out_of_scope"] is False
    assert res["is_ambiguous"] is False
    assert res["missing_slots"] == []
    assert res["extracted_slots"].get("vehicle_type") == "xe mô tô/xe gắn máy"
    assert res["clarification_question"] is None


def test_query_c_red_light_missing_vehicle(analyzer: QueryAnalyzer) -> None:
    """Test C: 'Vượt đèn đỏ phạt bao nhiêu?' -> CLARIFY, missing vehicle_type."""
    res = analyzer.analyze("Vượt đèn đỏ phạt bao nhiêu?")
    assert res["is_out_of_scope"] is False
    assert res["is_ambiguous"] is True
    assert "vehicle_type" in res["missing_slots"]
    assert res["clarification_question"] is not None
    assert "ô tô" in res["clarification_question"].lower()


def test_query_d_red_light_general_rule_no_clarify(analyzer: QueryAnalyzer) -> None:
    """Test D: 'Đèn đỏ có được đi không?' -> general road rule, NOT ambiguous."""
    res = analyzer.analyze("Đèn đỏ có được đi không?")
    assert res["is_out_of_scope"] is False
    assert res["is_ambiguous"] is False
    assert res["missing_slots"] == []
    assert res["clarification_question"] is None


def test_query_e_helmet_ambiguity(analyzer: QueryAnalyzer) -> None:
    """Test E: 'Không đội mũ bảo hiểm phạt bao nhiêu?' -> ambiguous on vehicle / actor."""
    res = analyzer.analyze("Không đội mũ bảo hiểm phạt bao nhiêu?")
    assert res["is_out_of_scope"] is False
    assert res["is_ambiguous"] is True
    assert ("vehicle_type" in res["missing_slots"] or "actor_role" in res["missing_slots"])
    assert res["clarification_question"] is not None


def test_query_f_out_of_scope_tax(analyzer: QueryAnalyzer) -> None:
    """Test F: 'Thuế thu nhập cá nhân tính thế nào?' -> OUT_OF_SCOPE."""
    res = analyzer.analyze("Thuế thu nhập cá nhân tính thế nào?")
    assert res["is_out_of_scope"] is True
    assert res["is_ambiguous"] is False
    assert res["out_of_scope_message"] is not None
    assert "giao thông đường bộ" in res["out_of_scope_message"].lower()


def test_query_out_of_scope_divorce(analyzer: QueryAnalyzer) -> None:
    """Test divorce domain is out of scope."""
    res = analyzer.analyze("Thủ tục ly hôn và phân chia tài sản giải quyết thế nào?")
    assert res["is_out_of_scope"] is True
    assert res["is_ambiguous"] is False


def test_query_out_of_scope_real_estate(analyzer: QueryAnalyzer) -> None:
    """Test land/real estate domain is out of scope."""
    res = analyzer.analyze("Tranh chấp đất đai và thủ tục cấp sổ đỏ cần giấy tờ gì?")
    assert res["is_out_of_scope"] is True
    assert res["is_ambiguous"] is False


def test_query_g_missing_violation_for_points(analyzer: QueryAnalyzer) -> None:
    """Test G: 'Bị trừ bao nhiêu điểm?' -> missing violation_type."""
    res = analyzer.analyze("Bị trừ bao nhiêu điểm?")
    assert res["is_out_of_scope"] is False
    assert res["is_ambiguous"] is True
    assert "violation_type" in res["missing_slots"]
    assert "hành vi" in res["clarification_question"].lower()


def test_non_over_clarification_definitions(analyzer: QueryAnalyzer) -> None:
    """General definitions and conditions do not prompt for clarification."""
    queries = [
        "Thế nào là dừng xe, đỗ xe theo quy định?",
        "Giấy phép lái xe bị trừ điểm trong trường hợp nào?",
        "Tín hiệu đèn vàng có ý nghĩa gì?",
        "Khoảng cách an toàn giữa hai xe là bao nhiêu?",
    ]
    for q in queries:
        res = analyzer.analyze(q)
        assert res["is_out_of_scope"] is False, f"Failed for {q}"
        assert res["is_ambiguous"] is False, f"Over-clarified for {q}"
        assert res["missing_slots"] == [], f"Unexpected missing slots for {q}"


def test_multi_turn_clarification_alcohol_motorcycle() -> None:
    """Multi-turn: Turn 1 clarify -> Turn 2 user says 'Xe máy' -> reconstructed query."""
    analyzer = QueryAnalyzer()
    state = ConversationState()

    # Turn 1
    query_1 = "Nồng độ cồn phạt bao nhiêu?"
    analysis_1 = analyzer.analyze(query_1)
    assert analysis_1["is_ambiguous"] is True
    state.original_query = query_1
    state.missing_slots = analysis_1["missing_slots"]
    state.clarification_pending = True

    # Turn 2: User responds
    user_response = "Xe máy"
    resolved_query = resolve_clarification(state.original_query, user_response, state)

    assert "nồng độ cồn" in resolved_query.lower()
    assert "mô tô" in resolved_query.lower() or "xe máy" in resolved_query.lower()
    assert state.clarification_pending is False
    assert state.collected_slots.get("vehicle_type") == "xe mô tô, xe gắn máy"


def test_multi_turn_clarification_red_light_car() -> None:
    """Multi-turn: Turn 1 clarify -> Turn 2 user says '1' or 'Ô tô' -> reconstructed query."""
    analyzer = QueryAnalyzer()
    state = ConversationState()

    query_1 = "Vượt đèn đỏ phạt bao nhiêu?"
    analysis_1 = analyzer.analyze(query_1)
    assert analysis_1["is_ambiguous"] is True
    state.original_query = query_1
    state.missing_slots = analysis_1["missing_slots"]
    state.clarification_pending = True

    user_response = "Ô tô"
    resolved_query = resolve_clarification(state.original_query, user_response, state)

    assert "vượt đèn đỏ" in resolved_query.lower()
    assert "ô tô" in resolved_query.lower()
    assert state.clarification_pending is False
    assert state.collected_slots.get("vehicle_type") == "xe ô tô"


def test_multi_turn_clarification_points_deduction() -> None:
    """Multi-turn: Turn 1 clarify -> Turn 2 user says 'vượt đèn đỏ'."""
    state = ConversationState(
        original_query="Bị trừ bao nhiêu điểm?",
        missing_slots=["violation_type"],
        clarification_pending=True,
    )
    resolved = resolve_clarification(state.original_query, "Vượt đèn đỏ", state)
    assert "vượt đèn đỏ" in resolved.lower()
    assert "trừ bao nhiêu điểm" in resolved.lower()
    assert state.clarification_pending is False
