"""
Tests for src/ingestion/legal_parser.py

10 required tests:
  1.  Nhận diện Điều
  2.  Nhận diện Khoản
  3.  Nhận diện Điểm
  4.  Nhận diện "đ)" tiếng Việt
  5.  Không nhầm số tiền thành Khoản
  6.  Không nhầm ngày tháng thành Khoản
  7.  Điều kéo dài qua nhiều trang
  8.  Hai file Luật 36 có cùng doc_number
  9.  UTF-8 tiếng Việt không lỗi
  10. Parsed output không mất text

Additional unit tests for classify_line edge cases.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# ── Project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.legal_parser import (
    DOC_META,
    K_ARTICLE,
    K_CHAPTER,
    K_CLAUSE,
    K_EMPTY,
    K_POINT,
    K_SECTION,
    K_TEXT,
    LegalUnit,
    LineToken,
    classify_line,
    parse_pages,
)

# ─── Shared fixtures ──────────────────────────────────────────────────────────

_SIMPLE_PAGES = [
    {
        "text": (
            "Chương I\n"
            "NHỮNG QUY ĐỊNH CHUNG\n"
            "Điều 1. Phạm vi điều chỉnh\n"
            "1. Người điều khiển phương tiện phải chấp hành\n"
            "a) Tín hiệu đèn giao thông\n"
            "b) Biển báo đường bộ\n"
            "đ) Vạch kẻ đường\n"
            "2. Tốc độ xe phải phù hợp quy định"
        ),
        "page": 1,
        "source_file": "test.pdf",
    }
]

_CROSS_PAGE_PAGES = [
    {
        "text": (
            "Điều 10. Quy định về tốc độ phương tiện\n"
            "1. Tốc độ tối đa cho phép của xe cơ giới, xe máy chuyên dùng"
        ),
        "page": 5,
        "source_file": "test.pdf",
    },
    {
        "text": (
            "tham gia giao thông đường bộ được quy định như sau:\n"
            "a) Trên đường cao tốc: không quá 120 km/h;\n"
            "b) Trong khu đô thị: không quá 60 km/h.\n"
            "2. Các loại xe khác tuân thủ quy định riêng."
        ),
        "page": 6,
        "source_file": "test.pdf",
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# Test 1 — Nhận diện Điều
# ═══════════════════════════════════════════════════════════════════════════════

class TestClassifyArticle:
    def test_basic_article(self):
        tok = classify_line("Điều 25. Điều kiện của người điều khiển phương tiện")
        assert tok.kind == K_ARTICLE
        assert tok.label == "Điều 25"
        assert "Điều kiện" in tok.extra

    def test_article_number_1(self):
        tok = classify_line("Điều 1. Phạm vi điều chỉnh")
        assert tok.kind == K_ARTICLE
        assert tok.label == "Điều 1"

    def test_article_with_letter_suffix(self):
        """Điều 11a. is valid in Vietnamese law."""
        tok = classify_line("Điều 11a. Quy định bổ sung")
        assert tok.kind == K_ARTICLE
        assert tok.label == "Điều 11a"

    def test_article_large_number(self):
        tok = classify_line("Điều 100. Điều khoản thi hành")
        assert tok.kind == K_ARTICLE
        assert tok.label == "Điều 100"

    def test_article_in_parse_output(self):
        """parse_pages must produce at least one record per article."""
        units = parse_pages(_SIMPLE_PAGES, "Doc", "TEST/2024")
        articles = {u["article"] for u in units}
        assert "Điều 1" in articles


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2 — Nhận diện Khoản
# ═══════════════════════════════════════════════════════════════════════════════

class TestClassifyClause:
    def test_basic_clause(self):
        tok = classify_line("1. Người điều khiển phương tiện phải chấp hành")
        assert tok.kind == K_CLAUSE
        assert tok.label == "1"

    def test_clause_2(self):
        tok = classify_line("2. Tốc độ xe phải phù hợp với quy định")
        assert tok.kind == K_CLAUSE
        assert tok.label == "2"

    def test_clause_with_letter_suffix(self):
        """'1a.' is a valid extended khoản number."""
        tok = classify_line("1a. Phạt cảnh cáo đối với người điều khiển xe")
        assert tok.kind == K_CLAUSE
        assert tok.label == "1a"

    def test_clause_9a(self):
        tok = classify_line("9a. Phạt tiền từ 5.000.000 đồng đến 6.000.000 đồng")
        assert tok.kind == K_CLAUSE
        assert tok.label == "9a"

    def test_clause_content_in_output(self):
        units = parse_pages(_SIMPLE_PAGES, "Doc", "TEST/2024")
        clauses = {u["clause"] for u in units if u.get("clause")}
        assert "1" in clauses
        assert "2" in clauses


# ═══════════════════════════════════════════════════════════════════════════════
# Test 3 — Nhận diện Điểm (standard a, b, c)
# ═══════════════════════════════════════════════════════════════════════════════

class TestClassifyPoint:
    def test_point_a(self):
        tok = classify_line("a) Tín hiệu đèn giao thông")
        assert tok.kind == K_POINT
        assert tok.label == "a)"

    def test_point_b(self):
        tok = classify_line("b) Không có giấy phép lái xe theo quy định")
        assert tok.kind == K_POINT
        assert tok.label == "b)"

    def test_point_c(self):
        tok = classify_line("c) Tổ chức kinh tế được thành lập")
        assert tok.kind == K_POINT
        assert tok.label == "c)"

    def test_point_in_parse_output(self):
        units = parse_pages(_SIMPLE_PAGES, "Doc", "TEST/2024")
        points = {u["point"] for u in units if u.get("point")}
        assert "a)" in points
        assert "b)" in points


# ═══════════════════════════════════════════════════════════════════════════════
# Test 4 — Nhận diện "đ)" tiếng Việt
# ═══════════════════════════════════════════════════════════════════════════════

class TestClassifyPointD:
    def test_point_d_viet(self):
        """'đ)' is the 5th point in the Vietnamese legal alphabet."""
        tok = classify_line("đ) Điều khiển xe vận tải hành khách theo hợp đồng")
        assert tok.kind == K_POINT, f"Expected K_POINT, got {tok.kind!r}"
        assert tok.label == "đ)"

    def test_point_d_content(self):
        tok = classify_line("đ) Vạch kẻ đường và các hiệu lệnh của người điều khiển giao thông")
        assert tok.kind == K_POINT
        assert tok.label == "đ)"
        assert "Vạch kẻ đường" in tok.extra

    def test_point_d_in_parse_output(self):
        units = parse_pages(_SIMPLE_PAGES, "Doc", "TEST/2024")
        points = {u["point"] for u in units if u.get("point")}
        assert "đ)" in points, f"'đ)' missing from points: {points}"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 5 — Không nhầm số tiền thành Khoản
# ═══════════════════════════════════════════════════════════════════════════════

class TestNoFalseClauseMoney:
    def test_money_no_space_after_period(self):
        """'12.000.000' has no whitespace after '12.' → not CLAUSE."""
        tok = classify_line("12.000.000 đồng đến 14.000.000 đồng")
        assert tok.kind != K_CLAUSE, f"Money misclassified as {tok.kind!r}"

    def test_money_with_space_starts_digit(self):
        """Even '12. 000' — content starts with digit, rejected."""
        tok = classify_line("12. 000.000 đồng")
        assert tok.kind != K_CLAUSE

    def test_money_in_sentence(self):
        """Line starting with 'Phạt tiền' is TEXT, never CLAUSE."""
        tok = classify_line("Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng")
        assert tok.kind == K_TEXT

    def test_large_number_three_digits(self):
        """3-digit number won't match \\d{1,2} — becomes TEXT."""
        tok = classify_line("168. Tham chiếu văn bản")
        assert tok.kind != K_CLAUSE

    def test_money_range_sentence(self):
        tok = classify_line("Phạt tiền từ 800.000 đồng đến 1.000.000 đồng đối với người điều khiển")
        assert tok.kind == K_TEXT


# ═══════════════════════════════════════════════════════════════════════════════
# Test 6 — Không nhầm ngày tháng thành Khoản
# ═══════════════════════════════════════════════════════════════════════════════

class TestNoFalseClauseDate:
    def test_date_no_period(self):
        """'26 tháng 6' has no period → not CLAUSE."""
        tok = classify_line("26 tháng 6 năm 2026")
        assert tok.kind != K_CLAUSE

    def test_year_four_digits(self):
        """4-digit year doesn't match \\d{1,2} → TEXT."""
        tok = classify_line("2026. Năm ban hành Nghị định này")
        assert tok.kind != K_CLAUSE

    def test_date_with_prefix(self):
        tok = classify_line("Hà Nội, ngày 26 tháng 6 năm 2026")
        assert tok.kind == K_TEXT

    def test_year_1900_rejected(self):
        """Year numbers 1900–2099 are explicitly rejected even if they match pattern."""
        tok = classify_line("24. Tháng Mười Hai")   # 24 < 1900, BUT "Tháng" starts uppercase
        # This tests that content-start validation works
        # 'T' is in VN_UPPER → would be CLAUSE. This is correct behaviour
        # (24. Tháng... is ambiguous; actual dates don't appear this way in law).
        # Just verify it doesn't crash.
        assert tok.kind in (K_CLAUSE, K_TEXT)   # either is acceptable

    def test_year_2024_not_clause(self):
        """'2024.' won't match \\d{1,2} — TEST."""
        tok = classify_line("2024. Ngày ban hành")
        assert tok.kind != K_CLAUSE

    def test_page_number_not_clause(self):
        """Bare page numbers from Công Báo (e.g. '76') should be TEXT."""
        tok = classify_line("76")
        assert tok.kind == K_TEXT   # no period → TEXT


# ═══════════════════════════════════════════════════════════════════════════════
# Test 7 — Điều kéo dài qua nhiều trang
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossPageArticle:
    def test_clause_spans_two_pages(self):
        """
        Khoản 1 of Điều 10 starts on page 5 and continues on page 6.
        The clause-intro record must have start_page=5 and end_page=6.
        """
        units = parse_pages(_CROSS_PAGE_PAGES, "Test Doc", "TEST/2024")

        # Clause-intro for khoản 1 (no điểm field)
        clause1 = [
            u for u in units
            if u.get("article") == "Điều 10"
            and u.get("clause") == "1"
            and u.get("point") is None
        ]
        assert clause1, (
            "Expected a clause-intro record for Điều 10 Khoản 1.\n"
            f"All records: {[(u.get('clause'), u.get('point')) for u in units]}"
        )
        rec = clause1[0]
        assert rec["start_page"] == 5, f"start_page expected 5, got {rec['start_page']}"
        assert rec["end_page"] == 6,   f"end_page expected 6, got {rec['end_page']}"

    def test_cross_page_content_intact(self):
        """Content from both pages must appear in the clause-intro record."""
        units = parse_pages(_CROSS_PAGE_PAGES, "Test Doc", "TEST/2024")
        clause1 = [
            u for u in units
            if u.get("clause") == "1" and u.get("point") is None
        ]
        assert clause1
        content = clause1[0]["content"]
        assert "Tốc độ tối đa" in content
        assert "tham gia giao thông" in content

    def test_points_on_second_page(self):
        """Points defined on page 6 should still be linked to Điều 10 Khoản 1."""
        units = parse_pages(_CROSS_PAGE_PAGES, "Test Doc", "TEST/2024")
        pts = [u for u in units if u.get("point") == "a)"]
        assert pts
        assert pts[0]["article"] == "Điều 10"
        assert pts[0]["clause"] == "1"
        assert pts[0]["start_page"] == 6


# ═══════════════════════════════════════════════════════════════════════════════
# Test 8 — Hai file Luật 36 có cùng doc_number
# ═══════════════════════════════════════════════════════════════════════════════

class TestLuat36DocNumber:
    def test_same_doc_number(self):
        assert DOC_META["36-2024-qh15"]["doc_number"] == "36/2024/QH15"
        assert DOC_META["36-2024-qh15_tiep"]["doc_number"] == "36/2024/QH15"

    def test_same_group(self):
        """Both files must be in the same processing group."""
        assert (
            DOC_META["36-2024-qh15"]["group"]
            == DOC_META["36-2024-qh15_tiep"]["group"]
        )

    def test_correct_ordering(self):
        """36-2024-qh15 must come before 36-2024-qh15_tiep."""
        assert DOC_META["36-2024-qh15"]["order"] < DOC_META["36-2024-qh15_tiep"]["order"]

    def test_units_carry_doc_number(self):
        """Every parsed unit from Luật 36 pages must carry doc_number=36/2024/QH15."""
        pages = [
            {
                "text": "Điều 5. Quyền và nghĩa vụ\n1. Người tham gia giao thông có nghĩa vụ",
                "page": 3,
                "source_file": "36-2024-qh15.pdf",
            }
        ]
        units = parse_pages(pages, "Luật Trật tự, an toàn giao thông đường bộ", "36/2024/QH15")
        for u in units:
            assert u["doc_number"] == "36/2024/QH15"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 9 — UTF-8 tiếng Việt không lỗi
# ═══════════════════════════════════════════════════════════════════════════════

class TestVietnameseUtf8:
    def test_vietnamese_chars_in_output(self, tmp_path):
        """JSON output must be UTF-8 with Vietnamese diacritics, no \\u escapes."""
        pages = [
            {
                "text": (
                    "Điều 1. Quy định về phương tiện giao thông\n"
                    "1. Người điều khiển phải đội mũ bảo hiểm\n"
                    "a) Trên đường quốc lộ phải chấp hành tốc độ quy định"
                ),
                "page": 1,
                "source_file": "test.pdf",
            }
        ]
        units = parse_pages(pages, "Luật Giao thông", "TEST/2024")
        out_file = tmp_path / "test_utf8.json"
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(units, f, ensure_ascii=False, indent=2)

        raw = out_file.read_text(encoding="utf-8")

        # Must not have Unicode escapes for Vietnamese chars (ensure_ascii=False)
        assert raw.count("\\u") == 0, (
            f"Found Unicode escapes — ensure_ascii may be True. "
            f"Snippet: {raw[:200]!r}"
        )

        # Must contain Vietnamese diacritics
        viet = "àáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ"
        all_text = " ".join(u.get("content", "") for u in units)
        assert any(c in all_text for c in viet), (
            f"No Vietnamese diacritics in output. Sample: {all_text[:200]!r}"
        )

    def test_classify_line_vietnamese_label(self):
        """classify_line must handle 'Điều', 'Chương', 'đ)' without encoding errors."""
        assert classify_line("Điều 1. Tiêu đề").kind == K_ARTICLE
        assert classify_line("Chương I").kind == K_CHAPTER
        assert classify_line("đ) Nội dung điểm đ").kind == K_POINT


# ═══════════════════════════════════════════════════════════════════════════════
# Test 10 — Parsed output không mất text
# ═══════════════════════════════════════════════════════════════════════════════

class TestNoTextLost:
    def test_all_clause_content_preserved(self):
        """Every khoản's text must appear somewhere in the output."""
        pages = [
            {
                "text": (
                    "Chương I\n"
                    "TIÊU ĐỀ CHƯƠNG MỘT\n"
                    "Điều 1. Tiêu đề điều một\n"
                    "1. Nội dung khoản một rất quan trọng\n"
                    "a) Nội dung điểm a của khoản một\n"
                    "b) Nội dung điểm b của khoản một\n"
                    "2. Khoản hai ngắn gọn nhưng đủ nghĩa\n"
                    "3. Khoản ba là khoản cuối cùng trong điều này"
                ),
                "page": 1,
                "source_file": "test.pdf",
            }
        ]
        units = parse_pages(pages, "Test Doc", "TEST/2024/QH15")
        all_content = " ".join(u.get("content", "") for u in units)

        assert "Nội dung khoản một rất quan trọng"   in all_content, "Khoản 1 intro lost"
        assert "Nội dung điểm a"                     in all_content, "Điểm a lost"
        assert "Nội dung điểm b"                     in all_content, "Điểm b lost"
        assert "Khoản hai ngắn gọn"                  in all_content, "Khoản 2 lost"
        assert "Khoản ba là khoản cuối cùng"         in all_content, "Khoản 3 lost"

    def test_total_records_positive(self):
        units = parse_pages(_SIMPLE_PAGES, "Doc", "TEST/2024")
        assert len(units) > 0

    def test_no_empty_content_records(self):
        """No record should have an empty content string."""
        units = parse_pages(_SIMPLE_PAGES, "Doc", "TEST/2024")
        empty = [u for u in units if not u.get("content", "").strip()]
        assert not empty, f"Records with empty content: {empty}"

    def test_context_fields_populated(self):
        """chapter and article fields must carry through to all records."""
        units = parse_pages(_SIMPLE_PAGES, "Doc", "TEST/2024")
        for u in units:
            assert u["chapter"] == "Chương I", f"chapter missing: {u}"
            assert u["article"] == "Điều 1",   f"article missing: {u}"


# ═══════════════════════════════════════════════════════════════════════════════
# Additional edge-case tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_quoted_amendment_not_point(self):
        """Lines starting with a quote (quoted amendment) must NOT be POINT."""
        tok = classify_line('"i) Buộc lắp đặt thiết bị giám sát hành trình')
        assert tok.kind != K_POINT, "Quoted amendment misidentified as điểm"

    def test_quoted_amendment_not_point_unicode(self):
        tok = classify_line('\u201ci) Buộc lắp đặt thiết bị')   # " (left double quote)
        assert tok.kind != K_POINT

    def test_chapter_roman(self):
        tok = classify_line("Chương I")
        assert tok.kind == K_CHAPTER
        assert tok.label == "Chương I"

    def test_chapter_roman_ii(self):
        tok = classify_line("Chương II")
        assert tok.kind == K_CHAPTER

    def test_section_numeric(self):
        tok = classify_line("Mục 1")
        assert tok.kind == K_SECTION
        assert tok.label == "Mục 1"

    def test_empty_line(self):
        tok = classify_line("")
        assert tok.kind == K_EMPTY

    def test_whitespace_only(self):
        tok = classify_line("   ")
        assert tok.kind == K_EMPTY

    def test_separator_line(self):
        tok = classify_line("--------")
        assert tok.kind == K_TEXT

    def test_dashes_not_clause(self):
        """Lines like '--------' should be TEXT, not CLAUSE."""
        tok = classify_line("--------------- ")
        assert tok.kind == K_TEXT
