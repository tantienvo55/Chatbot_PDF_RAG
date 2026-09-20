"""
Conversation state and clarification resolution module for multi-turn RAG dialogue.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import unicodedata
import re


@dataclass
class ConversationState:
    """
    In-memory state tracking clarification turns for a conversation session.
    """
    original_query: Optional[str] = None
    missing_slots: list[str] = field(default_factory=list)
    collected_slots: dict[str, Any] = field(default_factory=dict)
    clarification_pending: bool = False
    last_clarification_question: Optional[str] = None

    def reset(self) -> None:
        """Reset conversation state to empty baseline."""
        self.original_query = None
        self.missing_slots.clear()
        self.collected_slots.clear()
        self.clarification_pending = False
        self.last_clarification_question = None


def _normalize_text(text: str) -> str:
    """Normalize text into NFC and lowercased stripped string."""
    return unicodedata.normalize("NFC", text).strip().lower()


def parse_slot_from_answer(
    clarification_answer: str,
    missing_slots: list[str],
) -> dict[str, str]:
    """
    Parse slot values from a user's short follow-up answer.

    Args:
        clarification_answer: Raw text response from user (e.g. 'Xe máy', '1', 'ô tô').
        missing_slots: List of missing slot names awaiting answer.

    Returns:
        Dictionary mapping resolved slot name to standardized value.
    """
    norm = _normalize_text(clarification_answer)
    resolved: dict[str, str] = {}

    if "vehicle_type" in missing_slots:
        # Numeric choices or textual vehicle types
        if norm in ("1", "ô tô", "oto", "xe ô tô", "xe oto", "xe con", "xe tải", "xe khách"):
            resolved["vehicle_type"] = "xe ô tô"
        elif norm in ("2", "xe máy", "mô tô", "xe mô tô", "xe gắn máy", "xe hai bánh", "xe 2 bánh"):
            resolved["vehicle_type"] = "xe mô tô, xe gắn máy"
        elif norm in ("3", "xe đạp", "xe thô sơ", "xe đạp điện"):
            resolved["vehicle_type"] = "xe đạp, xe thô sơ"
        elif "ô tô" in norm or "oto" in norm or "xe hơi" in norm:
            resolved["vehicle_type"] = "xe ô tô"
        elif "máy" in norm or "mô tô" in norm or "gắn máy" in norm:
            resolved["vehicle_type"] = "xe mô tô, xe gắn máy"
        elif "đạp" in norm or "thô sơ" in norm:
            resolved["vehicle_type"] = "xe đạp, xe thô sơ"
        elif "đi bộ" in norm:
            resolved["vehicle_type"] = "người đi bộ"
        else:
            resolved["vehicle_type"] = clarification_answer.strip()

    if "violation_type" in missing_slots:
        if "cồn" in norm or "rượu" in norm or "bia" in norm:
            resolved["violation_type"] = "nồng độ cồn"
        elif "đèn đỏ" in norm:
            resolved["violation_type"] = "vượt đèn đỏ"
        elif "tốc độ" in norm:
            resolved["violation_type"] = "chạy quá tốc độ"
        elif "mũ" in norm or "bảo hiểm" in norm:
            resolved["violation_type"] = "không đội mũ bảo hiểm"
        elif "ngược chiều" in norm:
            resolved["violation_type"] = "đi ngược chiều"
        else:
            resolved["violation_type"] = clarification_answer.strip()

    if "actor_role" in missing_slots:
        if "người lái" in norm or "điều khiển" in norm or "tài xế" in norm:
            resolved["actor_role"] = "người điều khiển"
        elif "ngồi sau" in norm or "được chở" in norm or "đi cùng" in norm:
            resolved["actor_role"] = "người được chở"
        elif "chủ xe" in norm or "chủ phương tiện" in norm:
            resolved["actor_role"] = "chủ phương tiện"
        else:
            resolved["actor_role"] = clarification_answer.strip()

    return resolved


def resolve_clarification(
    original_query: str,
    clarification_answer: str,
    state: Optional[ConversationState] = None,
) -> str:
    """
    Reconstruct a contextual query by merging original user intent with follow-up clarification.

    Args:
        original_query: The original ambiguous query string.
        clarification_answer: User's answer to the clarification question.
        state: Optional ConversationState tracking session.

    Returns:
        Fully reconstructed, unambiguous query string suitable for retrieval.
    """
    if not original_query or not original_query.strip():
        return clarification_answer.strip()

    orig = original_query.strip()
    ans = clarification_answer.strip()

    missing_slots = state.missing_slots if state else ["vehicle_type"]
    parsed_slots = parse_slot_from_answer(ans, missing_slots)

    # Update state if present
    if state is not None:
        state.collected_slots.update(parsed_slots)
        state.clarification_pending = False

    # Check vehicle_type resolution
    if "vehicle_type" in parsed_slots:
        v_type = parsed_slots["vehicle_type"]
        # Remove trailing punctuation from original query
        clean_orig = re.sub(r"[?!.,;]+$", "", orig).strip()
        return f"{clean_orig} đối với {v_type}?"

    # Check violation_type resolution
    if "violation_type" in parsed_slots:
        viol = parsed_slots["violation_type"]
        clean_orig = re.sub(r"[?!.,;]+$", "", orig).strip()
        # E.g. "Bị trừ bao nhiêu điểm" + "vượt đèn đỏ" -> "Hành vi vượt đèn đỏ bị trừ bao nhiêu điểm?"
        return f"Hành vi {viol} {clean_orig.lower()}?"

    # Check actor_role resolution
    if "actor_role" in parsed_slots:
        role = parsed_slots["actor_role"]
        clean_orig = re.sub(r"[?!.,;]+$", "", orig).strip()
        return f"{clean_orig} đối với {role}?"

    # Fallback contextual merge
    clean_orig = re.sub(r"[?!.,;]+$", "", orig).strip()
    return f"{clean_orig} ({ans})?"
