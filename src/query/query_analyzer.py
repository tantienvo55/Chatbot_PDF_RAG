"""
Query analyzer module for pre-retrieval intent classification, out-of-scope detection,
slot extraction, and ambiguity/clarification handling.
"""

from typing import Any, Optional
import unicodedata
import re


# Non-traffic legal domains: tax, divorce, land/real estate, labor, corporate, criminal non-traffic
OUT_OF_SCOPE_PATTERNS = [
    # Tax / Finance
    r"thuế\s+(thu\s+nhập|tncn|thu\s+nhập\s+cá\s+nhân|giá\s+trị\s+gia\s+tăng|vat|doanh\s+nghiệp|nhà\s+đất)",
    r"quyết\s+toán\s+thuế",
    r"giảm\s+trừ\s+gia\s+cảnh",
    r"hoàn\s+thuế",
    r"kê\s+khai\s+thuế",
    r"hóa\s+đơn\s+điện\s+tử",
    # Real estate / Land
    r"sổ\s+đỏ",
    r"sổ\s+hồng",
    r"tranh\s+chấp\s+đất\s+đai",
    r"chuyển\s+nhượng\s+quyền\s+sử\s+dụng\s+đất",
    r"thừa\s+kế\s+(nhà|đất|tài\s+sản)",
    r"bất\s+động\s+sản",
    # Marriage & Family
    r"ly\s+hôn",
    r"chia\s+tài\s+sản\s+khi\s+ly\s+hôn",
    r"quyền\s+nuôi\s+con",
    r"kết\s+hôn",
    r"hôn\s+nhân\s+(và\s+)?gia\s+đình",
    # Labor & Social Insurance
    r"hợp\s+đồng\s+lao\s+động",
    r"bảo\s+hiểm\s+xã\s+hội",
    r"trợ\s+cấp\s+thất\s+nghiệp",
    r"nghỉ\s+thai\s+sản",
    r"sa\s+thải\s+lao\s+động",
    r"tiền\s+lương\s+làm\s+thêm\s+giờ",
    # Non-traffic criminal offenses
    r"buôn\s+lậu",
    r"ma\s+túy",
    r"giết\s+người",
    r"cướp\s+giật\s+tài\s+sản",
    r"lừa\s+đảo\s+chiếm\s+đoạt",
    r"đánh\s+bạc",
    r"tổ\s+chức\s+đánh\s+bạc",
    # Corporate
    r"thành\s+lập\s+công\s+ty",
    r"phá\s+sản\s+doanh\s+nghiệp",
    r"giấy\s+phép\s+kinh\s+doanh",
]

OUT_OF_SCOPE_MESSAGE = (
    "Hệ thống hiện chỉ hỗ trợ tra cứu pháp luật giao thông đường bộ trong bộ tài liệu đã nạp "
    "(Luật TTATGTĐB 36/2024/QH15, Nghị định 168/2024/NĐ-CP và các văn bản liên quan)."
)


class QueryAnalyzer:
    """
    Analyzes queries before retrieval to identify intent, detect out-of-scope domains,
    extract legal slots, and determine whether clarification is genuinely required.
    """

    def __init__(self) -> None:
        self.out_of_scope_res = [re.compile(p, re.IGNORECASE) for p in OUT_OF_SCOPE_PATTERNS]

    def analyze(self, query: str) -> dict[str, Any]:
        """
        Perform complete pre-retrieval analysis on query.

        Args:
            query: Input user query.

        Returns:
            Structured analysis dictionary containing intent, slots, ambiguity, and clarification.
        """
        if query is None or not query.strip():
            raise ValueError("Query cannot be empty or whitespace only.")

        normalized = unicodedata.normalize("NFC", query).strip().lower()

        # 1. Out-of-Scope check
        is_out_of_scope = self._check_out_of_scope(normalized)
        if is_out_of_scope:
            return {
                "intent": "out_of_scope",
                "is_out_of_scope": True,
                "is_ambiguous": False,
                "missing_slots": [],
                "extracted_slots": {},
                "normalized_query": normalized,
                "clarification_question": None,
                "out_of_scope_message": OUT_OF_SCOPE_MESSAGE,
            }

        # 2. Extract legal slots
        extracted_slots = self._extract_slots(normalized)

        # 3. Check for general legal rule / definition queries (strict non-over-clarification)
        is_general_rule = self._is_general_rule_query(normalized, extracted_slots)

        # 4. Check ambiguity & missing slots
        is_ambiguous, missing_slots = self._check_ambiguity(normalized, extracted_slots, is_general_rule)

        # 5. Build clarification question if ambiguous
        clarification_question = None
        if is_ambiguous and missing_slots:
            clarification_question = self._build_clarification_question(missing_slots, extracted_slots)

        # 6. Intent labeling
        intent = self._determine_intent(normalized, extracted_slots, is_ambiguous)

        return {
            "intent": intent,
            "is_out_of_scope": False,
            "is_ambiguous": is_ambiguous,
            "missing_slots": missing_slots,
            "extracted_slots": extracted_slots,
            "normalized_query": normalized,
            "clarification_question": clarification_question,
            "out_of_scope_message": None,
        }

    def _check_out_of_scope(self, query_norm: str) -> bool:
        """Check whether query falls outside road traffic law."""
        # Exception: if query explicitly mentions traffic entities, don't flag as out of scope
        traffic_keywords = [
            "giao thông", "đèn đỏ", "nồng độ cồn", "bằng lái", "gplx", "xe máy",
            "ô tô", "mũ bảo hiểm", "tốc độ", "làn đường", "biển báo", "dừng xe", "đỗ xe"
        ]
        has_traffic_context = any(k in query_norm for k in traffic_keywords)

        for pat in self.out_of_scope_res:
            if pat.search(query_norm):
                if not has_traffic_context:
                    return True
        return False

    def _extract_slots(self, query_norm: str) -> dict[str, Any]:
        """Extract legal slots from normalized query."""
        slots: dict[str, Any] = {}

        # 1. Vehicle Type
        # Order matters: check more specific first
        if re.search(r"\b(xe\s+máy\s+điện|xe\s+đạp\s+điện)\b", query_norm):
            slots["vehicle_type"] = "xe máy điện/xe đạp điện"
        elif re.search(r"\b(ô\s+tô|oto|xe\s+hơi|xe\s+con|xe\s+tải|xe\s+khách|xe\s+buýt|xe\s+container|xe\s+đầu\s+kéo)\b", query_norm):
            slots["vehicle_type"] = "xe ô tô"
        elif re.search(r"\b(xe\s+máy|mô\s+tô|xe\s+mô\s+tô|xe\s+gắn\s+máy|xe\s+hai\s+bánh|xe\s+2\s+bánh)\b", query_norm):
            slots["vehicle_type"] = "xe mô tô/xe gắn máy"
        elif re.search(r"\b(xe\s+đạp|xe\s+thô\s+sơ|súc\s+vật\s+kéo)\b", query_norm):
            slots["vehicle_type"] = "xe đạp/xe thô sơ"
        elif re.search(r"\b(người\s+đi\s+bộ)\b", query_norm):
            slots["vehicle_type"] = "người đi bộ"

        # 2. Violation Type
        if re.search(r"\b(nồng\s+độ\s+cồn|uống\s+rượu|uống\s+bia|say\s+xỉn|hơi\s+thở\s+có\s+cồn|máu\s+có\s+cồn)\b", query_norm):
            slots["violation_type"] = "nồng độ cồn"
        elif re.search(r"\b(vượt\s+đèn\s+đỏ|đèn\s+đỏ|tín\s+hiệu\s+đèn)\b", query_norm):
            slots["violation_type"] = "vượt đèn đỏ"
        elif re.search(r"\b(mũ\s+bảo\s+hiểm|nón\s+bảo\s+hiểm)\b", query_norm):
            slots["violation_type"] = "không đội mũ bảo hiểm"
        elif re.search(r"\b(quá\s+tốc\s+độ|chạy\s+quá\s+tốc\s+độ|vi\s+phạm\s+tốc\s+độ|bắn\s+tốc\s+độ)\b", query_norm):
            slots["violation_type"] = "chạy quá tốc độ"
        elif re.search(r"\b(ngược\s+chiều|đi\s+ngược\s+chiều)\b", query_norm):
            slots["violation_type"] = "đi ngược chiều"
        elif re.search(r"\b(lùi\s+xe|lùi\s+trên\s+cao\s+tốc)\b", query_norm):
            slots["violation_type"] = "lùi xe trên cao tốc"
        elif re.search(r"\b(trừ\s+điểm|trừ\s+bao\s+nhiêu\s+điểm|điểm\s+gplx|phục\s+hồi\s+điểm)\b", query_norm):
            slots["violation_type"] = "trừ điểm giấy phép lái xe"
        elif re.search(r"\b(dừng\s+xe|đỗ\s+xe|dừng\s+đỗ|đậu\s+xe)\b", query_norm):
            slots["violation_type"] = "dừng đỗ xe"
        elif re.search(r"\b(giấy\s+phép\s+lái\s+xe|bằng\s+lái|không\s+mang\s+bằng|không\s+có\s+bằng)\b", query_norm):
            slots["violation_type"] = "giấy phép lái xe"

        # 3. Actor Role
        if re.search(r"\b(người\s+lái|người\s+điều\s+khiển|tài\s+xế)\b", query_norm):
            slots["actor_role"] = "người điều khiển"
        elif re.search(r"\b(người\s+ngồi\s+sau|người\s+được\s+chở|ngồi\s+trên\s+xe)\b", query_norm):
            slots["actor_role"] = "người được chở"
        elif re.search(r"\b(chủ\s+xe|chủ\s+phương\s+tiện|giao\s+xe\s+cho\s+người\s+khác)\b", query_norm):
            slots["actor_role"] = "chủ phương tiện"

        # 4. Alcohol Level (if mentioned)
        alc_match = re.search(r"(\d+[\.,]?\d*)\s*(mg|miligam|g|gam)?\s*(/\s*(100\s*ml|l|lít))?", query_norm)
        if alc_match and slots.get("violation_type") == "nồng độ cồn":
            slots["alcohol_level"] = alc_match.group(0).strip()

        return slots

    def _is_general_rule_query(self, query_norm: str, slots: dict[str, Any]) -> bool:
        """
        Determine if query asks about general traffic rules, definitions, or signal meanings
        which apply universally without requiring vehicle or role clarification.
        """
        # Questions asking whether something is permitted / allowed / signal meanings
        general_patterns = [
            r"(có\s+được\s+đi\s+không|được\s+đi\s+không|có\s+được\s+phép\s+không)",
            r"(ý\s+nghĩa\s+gì|như\s+thế\s+nào\s+là|thế\s+nào\s+là|khái\s+niệm)",
            r"(quy\s+định\s+thế\s+nào|quy\s+định\s+chung|nguyên\s+tắc\s+chung)",
            r"(khoảng\s+cách\s+an\s+toàn\s+là\s+bao\s+nhiêu|khi\s+nào\s+được\s+chuyển\s+làn)",
            r"(trong\s+trường\s+hợp\s+nào|điều\s+kiện\s+gì)",
        ]
        for pat in general_patterns:
            if re.search(pat, query_norm):
                # E.g. "Đèn đỏ có được đi không?", "Thế nào là dừng xe, đỗ xe?"
                # "Giấy phép lái xe bị trừ điểm trong trường hợp nào?"
                # Do not treat as penalty inquiry requiring vehicle clarification
                return True

        return False

    def _check_ambiguity(
        self,
        query_norm: str,
        slots: dict[str, Any],
        is_general_rule: bool,
    ) -> tuple[bool, list[str]]:
        """
        Determine if ambiguity exists that materially alters the legal penalty or conclusion.

        Returns:
            (is_ambiguous, missing_slots)
        """
        if is_general_rule:
            return False, []

        is_penalty_inquiry = bool(re.search(
            r"\b(phạt\s+bao\s+nhiêu|mức\s+phạt|bị\s+phạt\s+bao\s+nhiêu|bị\s+phạt\s+mấy\s+tiền|"
            r"xử\s+phạt\s+thế\s+nào|bị\s+phạt\s+thế\s+nào|mức\s+xử\s+phạt|phạt\s+tiền\s+bao\s+nhiêu|"
            r"tịch\s+thu|tước\s+bằng|bị\s+trừ\s+bao\s+nhiêu\s+điểm)\b",
            query_norm
        ))

        violation = slots.get("violation_type")
        vehicle = slots.get("vehicle_type")
        actor = slots.get("actor_role")

        missing: list[str] = []

        # Case 1: Point deduction amount requested without specific violation
        # E.g. "Bị trừ bao nhiêu điểm?", "Mức trừ điểm là bao nhiêu?"
        if re.search(r"\b(bị\s+trừ\s+bao\s+nhiêu\s+điểm|trừ\s+mấy\s+điểm|mức\s+trừ\s+điểm\s+là\s+bao\s+nhiêu)\b", query_norm):
            if violation == "trừ điểm giấy phép lái xe" and not any(
                v in query_norm for v in ["đèn đỏ", "cồn", "tốc độ", "ngược chiều", "nồng độ", "mũ bảo hiểm", "lùi xe"]
            ):
                missing.append("violation_type")
                return True, missing

        # Case 2: Helmet penalty inquiry
        # E.g. "Không đội mũ bảo hiểm phạt bao nhiêu?"
        # Penalty depends on vehicle class (motorcycle vs electric bike / bicycle)
        if violation == "không đội mũ bảo hiểm" and is_penalty_inquiry:
            if not vehicle:
                missing.append("vehicle_type")
                return True, missing

        # Case 3: Vehicle-bracketed penalty inquiries (alcohol, red light, speeding, wrong-way)
        # Decree 168 sets fundamentally different fines for cars vs motorbikes vs bicycles
        if is_penalty_inquiry and violation in ("nồng độ cồn", "vượt đèn đỏ", "chạy quá tốc độ", "đi ngược chiều", "lùi xe trên cao tốc"):
            if not vehicle:
                missing.append("vehicle_type")
                return True, missing

        # Case 4: Alcohol penalty inquiry even without explicit "phạt bao nhiêu"
        # E.g. "Nồng độ cồn phạt bao nhiêu?" or "Nồng độ cồn xử lý thế nào?"
        if violation == "nồng độ cồn" and not vehicle:
            # If asking about sanctions/penalties in general
            if is_penalty_inquiry or any(w in query_norm for w in ["phạt", "xử lý", "mức"]):
                missing.append("vehicle_type")
                return True, missing

        # If uncertain whether a slot is truly required: do not invent the slot, prefer not to ask
        return False, []

    def _build_clarification_question(
        self,
        missing_slots: list[str],
        slots: dict[str, Any],
    ) -> str:
        """Construct user-friendly, non-technical clarification question with choices."""
        violation = slots.get("violation_type")

        if "violation_type" in missing_slots:
            return "Bạn đang muốn hỏi hành vi vi phạm giao thông cụ thể nào bị trừ điểm giấy phép lái xe?"

        if "vehicle_type" in missing_slots and "actor_role" in missing_slots:
            if violation == "không đội mũ bảo hiểm":
                return (
                    "Bạn đang hỏi mức phạt không đội mũ bảo hiểm đối với:\n"
                    "1. Người điều khiển xe hay người ngồi sau (được chở)?\n"
                    "2. Áp dụng cho xe mô tô/xe gắn máy hay xe đạp điện?"
                )

        if "vehicle_type" in missing_slots:
            if violation == "nồng độ cồn":
                return (
                    "Bạn đang hỏi mức phạt nồng độ cồn đối với loại phương tiện nào:\n"
                    "1. Ô tô\n"
                    "2. Xe mô tô / xe gắn máy\n"
                    "3. Xe đạp / xe thô sơ\n"
                    "4. Phương tiện khác?"
                )
            elif violation == "vượt đèn đỏ":
                return (
                    "Bạn đang hỏi mức phạt vượt đèn đỏ đối với loại phương tiện nào:\n"
                    "1. Ô tô\n"
                    "2. Xe mô tô / xe gắn máy\n"
                    "3. Phương tiện khác?"
                )
            elif violation == "chạy quá tốc độ":
                return (
                    "Bạn đang hỏi mức phạt vi phạm tốc độ đối với loại phương tiện nào:\n"
                    "1. Ô tô\n"
                    "2. Xe mô tô / xe gắn máy\n"
                    "3. Phương tiện khác?"
                )
            else:
                return "Bạn đang muốn hỏi mức phạt áp dụng cho loại phương tiện nào: ô tô, xe mô tô/xe gắn máy hay phương tiện khác?"

        if "actor_role" in missing_slots:
            return "Bạn đang hỏi mức phạt áp dụng cho người điều khiển phương tiện hay người được chở (người ngồi sau)?"

        return "Vui lòng cung cấp thêm thông tin chi tiết để hệ thống tra cứu quy định pháp luật chính xác nhất."

    def _determine_intent(
        self,
        query_norm: str,
        slots: dict[str, Any],
        is_ambiguous: bool,
    ) -> str:
        """Determine human-readable intent classification."""
        if is_ambiguous:
            return "clarification_needed"
        violation = slots.get("violation_type")
        if violation:
            return f"inquiry_{violation.replace(' ', '_')}"
        return "general_traffic_law_inquiry"


def select_retrieval_top_k(analysis: dict[str, Any]) -> int:
    """
    Deterministically select appropriate retrieval top-K based on query analysis.
    Uses top_k=5 for complex multi-bracket/multi-condition inquiries,
    and top_k=3 for simple definitions, signals, or single-action rules.

    Args:
        analysis: Analysis dictionary returned by QueryAnalyzer.analyze().

    Returns:
        Integer top_k value (3 or 5).
    """
    if not analysis or analysis.get("is_out_of_scope"):
        return 3

    slots = analysis.get("extracted_slots", {})
    violation = slots.get("violation_type")
    query_norm = analysis.get("normalized_query", "")

    # 1. Any penalty inquiry or sanction inquiry requires top_k=5 to capture specific clause/point brackets
    is_penalty = bool(re.search(
        r"\b(phạt\s+bao\s+nhiêu|mức\s+phạt|bị\s+phạt|xử\s+phạt|tiền\s+phạt|trừ\s+điểm)\b",
        query_norm
    ))
    if is_penalty:
        return 5

    # 2. Complex multi-bracket inquiries: alcohol violations, license points deduction
    if violation in ("nồng độ cồn", "trừ điểm giấy phép lái xe"):
        return 5

    # 3. Queries with explicit multi-part or enumeration cues
    multi_cues = [
        r"\b(các\s+mức|mấy\s+mức|những\s+mức|bao\s+nhiêu\s+mức)\b",
        r"\b(những\s+trường\s+hợp|các\s+trường\s+hợp|trường\s+hợp\s+nào)\b",
        r"\b(quy\s+định\s+nào|những\s+quy\s+tắc|các\s+quy\s+tắc)\b",
        r"\b(điều\s+kiện\s+gì|những\s+điều\s+kiện|các\s+điều\s+kiện)\b",
        r"\b(liệt\s+kê|tổng\s+hợp|toàn\s+bộ)\b",
    ]
    for pat in multi_cues:
        if re.search(pat, query_norm):
            return 5

    # 4. Simple rules, single actions, definitions, signal meanings -> top_k=3
    return 3
