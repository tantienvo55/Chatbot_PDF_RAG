"""
Tests for src/chunking/legal_chunker.py

16 required test cases:
  1.  Điều ngắn tạo chunk hợp lệ
  2.  Điều dài tách theo Khoản
  3.  Khoản giữ đúng Điều cha
  4.  Điểm giữ đúng Khoản cha
  5.  chunk_id unique
  6.  chunk_id deterministic
  7.  UTF-8 tiếng Việt
  8.  content không rỗng
  9.  content không mất text pháp lý quan trọng
  10. start_page/end_page đúng
  11. Luật 36 hai source file vẫn cùng doc_id
  12. NĐ 238 có amends_document = "168/2024/NĐ-CP"
  13. Nhận diện amended_article khi text có mẫu sửa đổi rõ ràng
  14. Không tự gán amended_article nếu source không xác định
  15. JSON save/load round-trip
  16. Không sinh duplicate chunks
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# ── Make src importable from any working directory ────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking.legal_chunker import (
    DOC_CONFIG,
    CHUNK_ID_PREFIX,
    audit_chunks,
    build_chunk_id,
    build_content_with_context,
    chunk_all,
    chunk_document,
    chunk_unit,
    compute_stats,
    extract_amendment_info,
    is_sentence_fragment,
    merge_fragment_chunks,
)

# ─── Fixture helpers ──────────────────────────────────────────────────────────

def _make_unit(
    doc_number: str = "36/2024/QH15",
    doc_title: str = "Luật Trật tự, an toàn giao thông đường bộ",
    chapter: str = "Chương I",
    chapter_title: str = "NHỮNG QUY ĐỊNH CHUNG",
    section: str = "",
    section_title: str = "",
    article: str = "Điều 1",
    article_title: str = "Phạm vi điều chỉnh",
    clause: str | None = None,
    point: str | None = None,
    content: str = "Luật này quy định về quy tắc giao thông đường bộ.",
    source_file: str = "36-2024-qh15.pdf",
    start_page: int = 1,
    end_page: int = 1,
) -> dict:
    return {
        "doc_title": doc_title,
        "doc_number": doc_number,
        "chapter": chapter,
        "chapter_title": chapter_title,
        "section": section,
        "section_title": section_title,
        "article": article,
        "article_title": article_title,
        "clause": clause,
        "point": point,
        "content": content,
        "source_file": source_file,
        "start_page": start_page,
        "end_page": end_page,
    }


def _make_parsed_doc(doc_number: str, units: list[dict]) -> dict:
    """Build a minimal parsed document dict as legal_parser would produce."""
    titles = {
        "36/2024/QH15":  "Luật Trật tự, an toàn giao thông đường bộ",
        "168/2024/NĐ-CP": "Nghị định xử phạt vi phạm hành chính",
        "238/2026/NĐ-CP": "Nghị định sửa đổi, bổ sung một số điều của Nghị định số 168/2024/NĐ-CP",
    }
    return {
        "doc_number": doc_number,
        "doc_title":  titles.get(doc_number, doc_number),
        "stats":      {},
        "units":      units,
    }


# ─── Shared test data ─────────────────────────────────────────────────────────

_SIMPLE_ARTICLE_UNIT = _make_unit(
    article="Điều 1",
    article_title="Phạm vi điều chỉnh",
    clause=None,
    point=None,
    content="Luật này quy định về quy tắc giao thông.",
)

_CLAUSE_UNITS = [
    _make_unit(article="Điều 2", article_title="Giải thích từ ngữ", clause="1",
               content="Trật tự, an toàn giao thông đường bộ là..."),
    _make_unit(article="Điều 2", article_title="Giải thích từ ngữ", clause="2",
               content="Phương tiện giao thông đường bộ là các loại xe."),
    _make_unit(article="Điều 2", article_title="Giải thích từ ngữ", clause="3",
               content="Người tham gia giao thông bao gồm..."),
]

_POINT_UNITS = [
    _make_unit(article="Điều 3", article_title="Nguyên tắc cơ bản",
               clause="1", point="a)", content="Tín hiệu đèn giao thông"),
    _make_unit(article="Điều 3", article_title="Nguyên tắc cơ bản",
               clause="1", point="b)", content="Biển báo đường bộ"),
    _make_unit(article="Điều 3", article_title="Nguyên tắc cơ bản",
               clause="1", point="đ)", content="Vạch kẻ đường"),
]

_ND238_UNIT = _make_unit(
    doc_number="238/2026/NĐ-CP",
    doc_title="Nghị định sửa đổi, bổ sung một số điều của Nghị định số 168/2024/NĐ-CP",
    article="Điều 2",
    article_title="Sửa đổi, bổ sung một số điểm, khoản của Điều 6",
    clause="1",
    point=None,
    content="Bổ sung khoản 1a vào trước khoản 1 như sau...",
    source_file="238_2026_ND-CP_712521.pdf",
)

_ND238_SPECIFIC_UNIT = _make_unit(
    doc_number="238/2026/NĐ-CP",
    doc_title="Nghị định sửa đổi, bổ sung một số điều của Nghị định số 168/2024/NĐ-CP",
    article="Điều 3",
    article_title="Sửa đổi, bổ sung điểm b khoản 3 Điều 13",
    clause=None,
    point=None,
    content="Nội dung sửa đổi điểm b khoản 3 Điều 13 của NĐ 168.",
    source_file="238_2026_ND-CP_712521.pdf",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Test 1 — Điều ngắn tạo chunk hợp lệ
# ═══════════════════════════════════════════════════════════════════════════════

class TestShortArticleChunk:
    def test_chunk_has_all_required_fields(self):
        """A chunk from a simple article-level unit must have all required metadata fields."""
        cfg = DOC_CONFIG["36/2024/QH15"]
        chunk = chunk_unit(_SIMPLE_ARTICLE_UNIT, "L36", cfg, {})

        required = {
            "chunk_id", "doc_id", "doc_title", "doc_number", "document_type",
            "issue_date", "effective_date", "legal_status",
            "chapter", "section", "article", "article_title",
            "clause", "point", "content", "content_with_context",
            "source_file", "start_page", "end_page",
            "amends_document", "amended_article", "amended_clause", "amended_point",
        }
        missing = required - set(chunk.keys())
        assert not missing, f"Missing fields: {missing}"

    def test_article_level_chunk_has_no_clause_or_point(self):
        cfg = DOC_CONFIG["36/2024/QH15"]
        chunk = chunk_unit(_SIMPLE_ARTICLE_UNIT, "L36", cfg, {})
        assert chunk["clause"] is None
        assert chunk["point"] is None

    def test_chunk_id_starts_with_correct_prefix(self):
        cfg = DOC_CONFIG["36/2024/QH15"]
        chunk = chunk_unit(_SIMPLE_ARTICLE_UNIT, "L36", cfg, {})
        assert chunk["chunk_id"].startswith("L36_")

    def test_document_type_is_law(self):
        cfg = DOC_CONFIG["36/2024/QH15"]
        chunk = chunk_unit(_SIMPLE_ARTICLE_UNIT, "L36", cfg, {})
        assert chunk["document_type"] == "law"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2 — Điều dài tách theo Khoản
# ═══════════════════════════════════════════════════════════════════════════════

class TestLongArticleSplitByClause:
    def test_each_clause_becomes_separate_chunk(self):
        """Three clause-level units from the same article produce 3 separate chunks."""
        doc = _make_parsed_doc("36/2024/QH15", _CLAUSE_UNITS)
        chunks = chunk_document(doc, {})
        assert len(chunks) == 3

    def test_chunks_have_distinct_clause_labels(self):
        doc = _make_parsed_doc("36/2024/QH15", _CLAUSE_UNITS)
        chunks = chunk_document(doc, {})
        clause_labels = [c["clause"] for c in chunks]
        assert clause_labels == ["1", "2", "3"]

    def test_chunks_have_distinct_chunk_ids(self):
        doc = _make_parsed_doc("36/2024/QH15", _CLAUSE_UNITS)
        chunks = chunk_document(doc, {})
        ids = [c["chunk_id"] for c in chunks]
        assert len(set(ids)) == len(ids), f"Duplicate ids: {ids}"

    def test_clause_ids_contain_KHOAN(self):
        doc = _make_parsed_doc("36/2024/QH15", _CLAUSE_UNITS)
        chunks = chunk_document(doc, {})
        for c in chunks:
            assert "KHOAN" in c["chunk_id"], (
                f"Expected 'KHOAN' in id '{c['chunk_id']}'"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 3 — Khoản giữ đúng Điều cha
# ═══════════════════════════════════════════════════════════════════════════════

class TestClauseParentArticle:
    def test_clause_chunk_carries_parent_article(self):
        """Clause-level chunk must retain the article label of its parent."""
        doc = _make_parsed_doc("36/2024/QH15", _CLAUSE_UNITS)
        chunks = chunk_document(doc, {})
        for c in chunks:
            assert c["article"] == "Điều 2", (
                f"Expected article='Điều 2', got '{c['article']}'"
            )

    def test_clause_chunk_carries_article_title(self):
        doc = _make_parsed_doc("36/2024/QH15", _CLAUSE_UNITS)
        chunks = chunk_document(doc, {})
        for c in chunks:
            assert c["article_title"] == "Giải thích từ ngữ"

    def test_clause_chunk_carries_chapter(self):
        doc = _make_parsed_doc("36/2024/QH15", _CLAUSE_UNITS)
        chunks = chunk_document(doc, {})
        for c in chunks:
            assert c["chapter"] == "Chương I"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 4 — Điểm giữ đúng Khoản cha
# ═══════════════════════════════════════════════════════════════════════════════

class TestPointParentClause:
    def test_point_chunk_has_correct_clause(self):
        """Point-level chunks must carry the clause label they belong to."""
        doc = _make_parsed_doc("36/2024/QH15", _POINT_UNITS)
        chunks = chunk_document(doc, {})
        for c in chunks:
            assert c["clause"] == "1", (
                f"Expected clause='1', got '{c['clause']}'"
            )

    def test_point_chunk_has_correct_article(self):
        doc = _make_parsed_doc("36/2024/QH15", _POINT_UNITS)
        chunks = chunk_document(doc, {})
        for c in chunks:
            assert c["article"] == "Điều 3"

    def test_point_labels_preserved(self):
        doc = _make_parsed_doc("36/2024/QH15", _POINT_UNITS)
        chunks = chunk_document(doc, {})
        points = [c["point"] for c in chunks]
        assert "a)" in points
        assert "b)" in points
        assert "đ)" in points

    def test_point_ids_contain_DIEM(self):
        doc = _make_parsed_doc("36/2024/QH15", _POINT_UNITS)
        chunks = chunk_document(doc, {})
        for c in chunks:
            assert "DIEM" in c["chunk_id"], (
                f"Expected 'DIEM' in id '{c['chunk_id']}'"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 5 — chunk_id unique
# ═══════════════════════════════════════════════════════════════════════════════

class TestChunkIdUnique:
    def test_all_ids_unique_single_doc(self):
        """All chunk_ids within one document must be unique."""
        units = _CLAUSE_UNITS + _POINT_UNITS
        doc = _make_parsed_doc("36/2024/QH15", units)
        chunks = chunk_document(doc, {})
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), f"Duplicates: {[i for i in ids if ids.count(i) > 1]}"

    def test_all_ids_unique_multi_doc(self):
        """chunk_ids must be globally unique across all three documents."""
        counter: dict = {}

        units_l36 = [_make_unit(doc_number="36/2024/QH15", article="Điều 1")]
        units_nd168 = [_make_unit(doc_number="168/2024/NĐ-CP", article="Điều 1",
                                   doc_title="NĐ 168")]
        units_nd238 = [_make_unit(doc_number="238/2026/NĐ-CP", article="Điều 1",
                                   doc_title="NĐ 238")]

        cfg_l36   = DOC_CONFIG["36/2024/QH15"]
        cfg_nd168 = DOC_CONFIG["168/2024/NĐ-CP"]
        cfg_nd238 = DOC_CONFIG["238/2026/NĐ-CP"]

        c1 = chunk_unit(units_l36[0],   "L36",   cfg_l36,   counter)
        c2 = chunk_unit(units_nd168[0], "ND168", cfg_nd168, counter)
        c3 = chunk_unit(units_nd238[0], "ND238", cfg_nd238, counter)

        ids = [c1["chunk_id"], c2["chunk_id"], c3["chunk_id"]]
        assert len(ids) == len(set(ids)), f"Cross-doc duplicate: {ids}"

    def test_compute_stats_reports_zero_duplicates(self):
        units = _CLAUSE_UNITS + _POINT_UNITS
        doc = _make_parsed_doc("36/2024/QH15", units)
        chunks = chunk_document(doc, {})
        stats = compute_stats(chunks)
        assert stats["duplicate_chunk_ids"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Test 6 — chunk_id deterministic
# ═══════════════════════════════════════════════════════════════════════════════

class TestChunkIdDeterministic:
    def test_same_input_same_id(self):
        """Same unit + fresh counter must always produce the same chunk_id."""
        cfg = DOC_CONFIG["36/2024/QH15"]
        id1 = chunk_unit(_SIMPLE_ARTICLE_UNIT, "L36", cfg, {})["chunk_id"]
        id2 = chunk_unit(_SIMPLE_ARTICLE_UNIT, "L36", cfg, {})["chunk_id"]
        assert id1 == id2

    def test_id_format_article_only(self):
        """Article-level chunk for Điều 25 must produce 'L36_DIEU25'."""
        unit = _make_unit(article="Điều 25", clause=None, point=None)
        cfg = DOC_CONFIG["36/2024/QH15"]
        chunk = chunk_unit(unit, "L36", cfg, {})
        assert chunk["chunk_id"] == "L36_DIEU25"

    def test_id_format_clause(self):
        """Clause-level chunk must produce 'ND168_DIEU6_KHOAN2'."""
        unit = _make_unit(
            doc_number="168/2024/NĐ-CP",
            doc_title="NĐ 168",
            article="Điều 6", clause="2", point=None,
        )
        cfg = DOC_CONFIG["168/2024/NĐ-CP"]
        chunk = chunk_unit(unit, "ND168", cfg, {})
        assert chunk["chunk_id"] == "ND168_DIEU6_KHOAN2"

    def test_id_format_point(self):
        """Point-level chunk must produce 'ND168_DIEU6_KHOAN2_DIEMA'."""
        unit = _make_unit(
            doc_number="168/2024/NĐ-CP",
            doc_title="NĐ 168",
            article="Điều 6", clause="2", point="a)",
        )
        cfg = DOC_CONFIG["168/2024/NĐ-CP"]
        chunk = chunk_unit(unit, "ND168", cfg, {})
        assert chunk["chunk_id"] == "ND168_DIEU6_KHOAN2_DIEMA"

    def test_id_format_point_d_viet(self):
        """Vietnamese 'đ)' must produce 'DIEMD' in chunk_id (đ → D)."""
        unit = _make_unit(article="Điều 3", clause="1", point="đ)")
        cfg = DOC_CONFIG["36/2024/QH15"]
        chunk = chunk_unit(unit, "L36", cfg, {})
        assert "DIEMD" in chunk["chunk_id"]

    def test_collision_resolved_with_version_suffix(self):
        """Duplicate raw keys get '_v2' suffix."""
        counter: dict = {}
        cfg = DOC_CONFIG["36/2024/QH15"]
        id1 = chunk_unit(_SIMPLE_ARTICLE_UNIT, "L36", cfg, counter)["chunk_id"]
        id2 = chunk_unit(_SIMPLE_ARTICLE_UNIT, "L36", cfg, counter)["chunk_id"]
        assert id1 != id2
        assert id2.endswith("_v2")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 7 — UTF-8 tiếng Việt
# ═══════════════════════════════════════════════════════════════════════════════

class TestUtf8Vietnamese:
    def test_vietnamese_content_preserved_in_chunk(self):
        """Vietnamese diacritics must survive chunking without encoding loss."""
        content = "Người điều khiển phương tiện phải chấp hành tín hiệu đèn giao thông."
        unit = _make_unit(content=content)
        cfg = DOC_CONFIG["36/2024/QH15"]
        chunk = chunk_unit(unit, "L36", cfg, {})
        assert chunk["content"] == content

    def test_json_round_trip_no_unicode_escapes(self, tmp_path):
        """JSON output must use ensure_ascii=False — no \\u escapes for Vietnamese."""
        content = "Trật tự, an toàn giao thông đường bộ."
        unit = _make_unit(content=content)
        doc = _make_parsed_doc("36/2024/QH15", [unit])
        chunks = chunk_document(doc, {})

        out = tmp_path / "chunks.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        raw = out.read_text(encoding="utf-8")
        assert "\\u" not in raw, f"Unicode escapes found in output"
        assert "đường" in raw or "tr" in raw   # Vietnamese chars present

    def test_content_with_context_contains_viet_headers(self):
        """content_with_context must contain Vietnamese structural labels."""
        unit = _make_unit(article="Điều 5", article_title="Quy tắc cơ bản",
                          clause="2", content="Nội dung khoản hai.")
        ctx = build_content_with_context(unit)
        assert "Điều 5" in ctx
        assert "Khoản 2" in ctx
        assert "Nội dung khoản hai" in ctx


# ═══════════════════════════════════════════════════════════════════════════════
# Test 8 — content không rỗng
# ═══════════════════════════════════════════════════════════════════════════════

class TestContentNotEmpty:
    def test_chunk_content_not_empty(self):
        cfg = DOC_CONFIG["36/2024/QH15"]
        chunk = chunk_unit(_SIMPLE_ARTICLE_UNIT, "L36", cfg, {})
        assert chunk["content"].strip(), "content must not be empty"

    def test_compute_stats_reports_zero_empty(self):
        units = [
            _make_unit(article="Điều 1", content="Nội dung một."),
            _make_unit(article="Điều 2", content="Nội dung hai."),
        ]
        doc = _make_parsed_doc("36/2024/QH15", units)
        chunks = chunk_document(doc, {})
        stats = compute_stats(chunks)
        assert stats["empty_chunks"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Test 9 — content không mất text pháp lý quan trọng
# ═══════════════════════════════════════════════════════════════════════════════

class TestContentPreservation:
    def test_original_content_preserved_verbatim(self):
        """The 'content' field must be identical to the source unit's content."""
        original = "Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng đối với người điều khiển."
        unit = _make_unit(content=original)
        cfg = DOC_CONFIG["36/2024/QH15"]
        chunk = chunk_unit(unit, "L36", cfg, {})
        assert chunk["content"] == original

    def test_no_text_lost_across_all_units(self):
        """Total character count of content must equal sum of unit contents."""
        units = _CLAUSE_UNITS + _POINT_UNITS
        total_orig = sum(len(u["content"]) for u in units)
        doc = _make_parsed_doc("36/2024/QH15", units)
        chunks = chunk_document(doc, {})
        total_chunked = sum(len(c["content"]) for c in chunks)
        assert total_chunked == total_orig, (
            f"Content loss: {total_orig} chars in → {total_chunked} chars out"
        )

    def test_content_with_context_contains_original(self):
        """content_with_context must include the original content text."""
        original = "Tốc độ tối đa trên đường cao tốc là 120 km/h."
        unit = _make_unit(content=original)
        ctx = build_content_with_context(unit)
        assert original in ctx


# ═══════════════════════════════════════════════════════════════════════════════
# Test 10 — start_page/end_page đúng
# ═══════════════════════════════════════════════════════════════════════════════

class TestPageNumbers:
    def test_start_end_page_preserved(self):
        unit = _make_unit(start_page=5, end_page=7)
        cfg = DOC_CONFIG["36/2024/QH15"]
        chunk = chunk_unit(unit, "L36", cfg, {})
        assert chunk["start_page"] == 5
        assert chunk["end_page"] == 7

    def test_single_page_chunk(self):
        unit = _make_unit(start_page=12, end_page=12)
        cfg = DOC_CONFIG["36/2024/QH15"]
        chunk = chunk_unit(unit, "L36", cfg, {})
        assert chunk["start_page"] == chunk["end_page"] == 12

    def test_cross_page_chunk(self):
        """Cross-page units (start != end) must preserve both page numbers."""
        unit = _make_unit(start_page=3, end_page=5)
        cfg = DOC_CONFIG["36/2024/QH15"]
        chunk = chunk_unit(unit, "L36", cfg, {})
        assert chunk["start_page"] == 3
        assert chunk["end_page"] == 5


# ═══════════════════════════════════════════════════════════════════════════════
# Test 11 — Luật 36 hai source file vẫn cùng doc_id
# ═══════════════════════════════════════════════════════════════════════════════

class TestLuat36TwoSourceFiles:
    def test_both_parts_produce_same_doc_id(self):
        """Units from both 36-2024-qh15.pdf and 36-2024-qh15_tiep.pdf
        must carry the same doc_id = '36/2024/QH15'."""
        unit_part1 = _make_unit(
            doc_number="36/2024/QH15",
            source_file="36-2024-qh15.pdf",
            article="Điều 10",
        )
        unit_part2 = _make_unit(
            doc_number="36/2024/QH15",
            source_file="36-2024-qh15_tiep.pdf",
            article="Điều 50",
        )
        doc = _make_parsed_doc("36/2024/QH15", [unit_part1, unit_part2])
        chunks = chunk_document(doc, {})
        for c in chunks:
            assert c["doc_id"] == "36/2024/QH15"
            assert c["doc_number"] == "36/2024/QH15"

    def test_source_file_field_preserved(self):
        """source_file field must distinguish the two PDF parts."""
        unit1 = _make_unit(source_file="36-2024-qh15.pdf", article="Điều 1")
        unit2 = _make_unit(source_file="36-2024-qh15_tiep.pdf", article="Điều 60")
        doc = _make_parsed_doc("36/2024/QH15", [unit1, unit2])
        chunks = chunk_document(doc, {})
        src_files = {c["source_file"] for c in chunks}
        assert "36-2024-qh15.pdf" in src_files
        assert "36-2024-qh15_tiep.pdf" in src_files


# ═══════════════════════════════════════════════════════════════════════════════
# Test 12 — NĐ 238 có amends_document = "168/2024/NĐ-CP"
# ═══════════════════════════════════════════════════════════════════════════════

class TestNd238AmendsDocument:
    def test_amends_document_field_set(self):
        """Every chunk from NĐ 238 must have amends_document = '168/2024/NĐ-CP'."""
        cfg = DOC_CONFIG["238/2026/NĐ-CP"]
        chunk = chunk_unit(_ND238_UNIT, "ND238", cfg, {})
        assert chunk["amends_document"] == "168/2024/NĐ-CP"

    def test_document_type_is_amending_decree(self):
        cfg = DOC_CONFIG["238/2026/NĐ-CP"]
        chunk = chunk_unit(_ND238_UNIT, "ND238", cfg, {})
        assert chunk["document_type"] == "amending_decree"

    def test_nd168_has_no_amends_document(self):
        """NĐ 168 is the original decree — amends_document must be None."""
        unit = _make_unit(doc_number="168/2024/NĐ-CP", doc_title="NĐ 168")
        cfg = DOC_CONFIG["168/2024/NĐ-CP"]
        chunk = chunk_unit(unit, "ND168", cfg, {})
        assert chunk["amends_document"] is None

    def test_luat36_has_no_amends_document(self):
        cfg = DOC_CONFIG["36/2024/QH15"]
        chunk = chunk_unit(_SIMPLE_ARTICLE_UNIT, "L36", cfg, {})
        assert chunk["amends_document"] is None


# ═══════════════════════════════════════════════════════════════════════════════
# Test 13 — Nhận diện amended_article khi text có mẫu sửa đổi rõ ràng
# ═══════════════════════════════════════════════════════════════════════════════

class TestAmendmentExtraction:
    def test_article_title_with_dieu_number(self):
        """'Sửa đổi, bổ sung ... Điều 6' → amended_article = 'Điều 6'."""
        art, kh, pt = extract_amendment_info(
            "Sửa đổi, bổ sung một số điểm, khoản của Điều 6",
            "238/2026/NĐ-CP",
        )
        assert art == "Điều 6"

    def test_article_title_with_khoan_and_dieu(self):
        """'Sửa đổi, bổ sung điểm b khoản 3 Điều 14' → all three fields."""
        art, kh, pt = extract_amendment_info(
            "Sửa đổi, bổ sung điểm b khoản 3 Điều 14",
            "238/2026/NĐ-CP",
        )
        assert art == "Điều 14"
        assert kh == "3"
        assert pt == "b)"

    def test_article_title_khoan_only(self):
        """'khoản 3 Điều 3' → amended_clause='3', amended_point=None."""
        art, kh, pt = extract_amendment_info(
            "Sửa đổi, bổ sung một số điểm của khoản 3 Điều 3",
            "238/2026/NĐ-CP",
        )
        assert art == "Điều 3"
        assert kh == "3"
        assert pt is None

    def test_chunk_carries_amended_article(self):
        """chunk_unit must propagate the extracted amendment fields."""
        cfg = DOC_CONFIG["238/2026/NĐ-CP"]
        chunk = chunk_unit(_ND238_SPECIFIC_UNIT, "ND238", cfg, {})
        assert chunk["amended_article"] == "Điều 13"
        assert chunk["amended_clause"] == "3"
        assert chunk["amended_point"] == "b)"

    def test_various_patterns(self):
        """Run all real patterns found in NĐ 238."""
        patterns = [
            ("Sửa đổi, bổ sung điểm a khoản 2 Điều 17", "Điều 17", "2", "a)"),
            ("Sửa đổi bổ sung điểm a khoản 4 Điều 18",  "Điều 18", "4", "a)"),
            ("Sửa đổi, bổ sung một số điểm, khoản của Điều 20", "Điều 20", None, None),
        ]
        for title, exp_art, exp_kh, exp_pt in patterns:
            art, kh, pt = extract_amendment_info(title, "238/2026/NĐ-CP")
            assert art == exp_art, f"title={title!r}: got {art!r}"
            if exp_kh is not None:
                assert kh == exp_kh, f"title={title!r}: got kh={kh!r}"
            if exp_pt is not None:
                assert pt == exp_pt, f"title={title!r}: got pt={pt!r}"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 14 — Không tự gán amended_article nếu source không xác định
# ═══════════════════════════════════════════════════════════════════════════════

class TestNoFalseAmendment:
    def test_non_amending_doc_has_no_amendment_fields(self):
        """Luật 36 chunks must never have amendment metadata."""
        unit = _make_unit(
            article_title="Phạm vi điều chỉnh",
        )
        cfg = DOC_CONFIG["36/2024/QH15"]
        chunk = chunk_unit(unit, "L36", cfg, {})
        assert chunk["amended_article"] is None
        assert chunk["amended_clause"] is None
        assert chunk["amended_point"] is None

    def test_nd168_has_no_amendment_fields(self):
        """NĐ 168 is the amended decree, not the amending one."""
        unit = _make_unit(doc_number="168/2024/NĐ-CP", doc_title="NĐ 168",
                          article_title="Điều 6. Xử phạt người điều khiển xe")
        cfg = DOC_CONFIG["168/2024/NĐ-CP"]
        chunk = chunk_unit(unit, "ND168", cfg, {})
        assert chunk["amended_article"] is None

    def test_empty_article_title_gives_no_amendment(self):
        """Missing article_title in an NĐ 238 unit → amendment fields are None."""
        unit = _make_unit(
            doc_number="238/2026/NĐ-CP",
            doc_title="NĐ 238",
            article_title="",   # no title info
        )
        art, kh, pt = extract_amendment_info("", "238/2026/NĐ-CP")
        assert art is None
        assert kh is None
        assert pt is None

    def test_extract_amendment_returns_none_for_unknown_doc(self):
        """Unknown doc_number must always return (None, None, None)."""
        art, kh, pt = extract_amendment_info("Sửa đổi Điều 5", "UNKNOWN/DOC")
        assert art is None
        assert kh is None
        assert pt is None


# ═══════════════════════════════════════════════════════════════════════════════
# Test 15 — JSON save/load round-trip
# ═══════════════════════════════════════════════════════════════════════════════

class TestJsonRoundTrip:
    def test_save_and_reload_preserves_all_fields(self, tmp_path):
        """Saving chunks to JSON and reloading must produce identical dicts."""
        units = _CLAUSE_UNITS + _POINT_UNITS
        doc = _make_parsed_doc("36/2024/QH15", units)
        chunks = chunk_document(doc, {})

        out = tmp_path / "chunks.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        with out.open(encoding="utf-8") as f:
            reloaded = json.load(f)

        assert len(reloaded) == len(chunks)
        for orig, loaded in zip(chunks, reloaded):
            assert orig == loaded, f"Mismatch for chunk_id={orig['chunk_id']}"

    def test_jsonl_round_trip(self, tmp_path):
        """JSONL format: each line parses to the same dict."""
        units = [_SIMPLE_ARTICLE_UNIT]
        doc = _make_parsed_doc("36/2024/QH15", units)
        chunks = chunk_document(doc, {})

        out = tmp_path / "chunks.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

        loaded = []
        with out.open(encoding="utf-8") as f:
            for line in f:
                loaded.append(json.loads(line))

        assert loaded == chunks

    def test_chunk_all_creates_output_files(self, tmp_path):
        """chunk_all() must create legal_chunks.json and legal_chunks.jsonl."""
        parsed_dir = tmp_path / "parsed"
        parsed_dir.mkdir()
        output_dir = tmp_path / "chunks"

        units = [_SIMPLE_ARTICLE_UNIT]
        parsed_doc = _make_parsed_doc("36/2024/QH15", units)
        with (parsed_dir / "36_2024_QH15.json").open("w", encoding="utf-8") as f:
            json.dump(parsed_doc, f, ensure_ascii=False)

        chunks = chunk_all(parsed_dir, output_dir)

        assert (output_dir / "legal_chunks.json").exists()
        assert (output_dir / "legal_chunks.jsonl").exists()
        assert len(chunks) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Test 16 — Không sinh duplicate chunks
# ═══════════════════════════════════════════════════════════════════════════════

class TestNoDuplicateChunks:
    def test_no_duplicate_chunk_ids_single_doc(self):
        """A realistic mix of article/clause/point units must produce 0 duplicates."""
        units = [
            _make_unit(article="Điều 1", clause=None, point=None, content="A1"),
            _make_unit(article="Điều 2", clause="1", point=None, content="A2K1"),
            _make_unit(article="Điều 2", clause="2", point=None, content="A2K2"),
            _make_unit(article="Điều 3", clause="1", point="a)", content="A3K1Pa"),
            _make_unit(article="Điều 3", clause="1", point="b)", content="A3K1Pb"),
        ]
        doc = _make_parsed_doc("36/2024/QH15", units)
        chunks = chunk_document(doc, {})
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), (
            f"Duplicates: {[i for i in ids if ids.count(i) > 1]}"
        )

    def test_no_duplicate_chunk_ids_across_docs(self):
        """Chunk ids from all three documents combined must be globally unique."""
        units_l36 = [_make_unit(doc_number="36/2024/QH15", article="Điều 1")]
        units_nd168 = [_make_unit(
            doc_number="168/2024/NĐ-CP", doc_title="NĐ 168", article="Điều 1")]
        units_nd238 = [_make_unit(
            doc_number="238/2026/NĐ-CP", doc_title="NĐ 238", article="Điều 1")]

        counter: dict = {}
        cfg_l36   = DOC_CONFIG["36/2024/QH15"]
        cfg_nd168 = DOC_CONFIG["168/2024/NĐ-CP"]
        cfg_nd238 = DOC_CONFIG["238/2026/NĐ-CP"]

        chunks = [
            chunk_unit(units_l36[0],   "L36",   cfg_l36,   counter),
            chunk_unit(units_nd168[0], "ND168", cfg_nd168, counter),
            chunk_unit(units_nd238[0], "ND238", cfg_nd238, counter),
        ]
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), f"Cross-doc duplicates: {ids}"

    def test_compute_stats_detects_duplicates(self):
        """compute_stats must correctly count artificially injected duplicates."""
        units = [_SIMPLE_ARTICLE_UNIT]
        doc = _make_parsed_doc("36/2024/QH15", units)
        chunks = chunk_document(doc, {})

        # Artificially inject a duplicate
        chunks.append(chunks[0].copy())
        stats = compute_stats(chunks)
        assert stats["duplicate_chunk_ids"] >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt 4.5 — Quality Audit Tests (17–26)
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# Test 17 — Không còn chunk vô nghĩa 2–5 ký tự sau cleanup
# ═══════════════════════════════════════════════════════════════════════════════

class TestNoMeaninglessShortChunks:
    def test_is_sentence_fragment_detects_lowercase_start(self):
        """Strings starting with lowercase that are short are fragments."""
        assert is_sentence_fragment("phép, chứng chỉ hành nghề") is True
        assert is_sentence_fragment("khi tuần tra, kiểm soát") is True
        assert is_sentence_fragment("gia giao thông đường bộ") is True

    def test_is_sentence_fragment_rejects_uppercase_start(self):
        """Short strings starting with uppercase are valid điểm items."""
        assert is_sentence_fragment("Biển báo hiệu đường bộ;") is False
        assert is_sentence_fragment("Chở người bệnh đi cấp cứu;") is False
        assert is_sentence_fragment("Dừng xe, đỗ xe trên cầu;") is False

    def test_is_sentence_fragment_rejects_digit_start(self):
        """Content starting with a digit (e.g. money amount) is not a fragment."""
        assert is_sentence_fragment("12.000.000 đồng") is False

    def test_is_sentence_fragment_rejects_long_content(self):
        """Content >= 50 chars is never classified as a fragment."""
        long_content = "phép, chứng chỉ hành nghề của những người có liên quan"
        assert len(long_content) >= 50
        assert is_sentence_fragment(long_content) is False

    def test_merge_removes_fragment_chunks(self):
        """After merge_fragment_chunks, lowercase-start short article chunks are absorbed."""
        units = [
            _make_unit(article="Điều 3", clause=None, point=None,
                       content="Phạm vi điều chỉnh của luật này bao gồm các quy định về giao thông."),
            _make_unit(article="Điều 3", clause=None, point=None,
                       content="phạm vi xử phạt"),  # fragment: lowercase start, same article
        ]
        doc = _make_parsed_doc("36/2024/QH15", units)
        chunks = chunk_document(doc, {})
        cleaned, merged, removed = merge_fragment_chunks(chunks)

        assert len(removed) == 1
        assert len(cleaned) == 1
        assert "phạm vi xử phạt" in cleaned[0]["content"]

    def test_audit_chunks_counts_fragments(self):
        """audit_chunks must correctly count fragment candidates."""
        units = [
            _make_unit(article="Điều 5", clause=None, point=None,
                       content="Nội dung chính của điều này."),
            _make_unit(article="Điều 5", clause=None, point=None,
                       content="phần bị cắt"),  # fragment
        ]
        doc = _make_parsed_doc("36/2024/QH15", units)
        chunks = chunk_document(doc, {})
        report = audit_chunks(chunks)
        assert report["fragment_candidates"] != []


# ═══════════════════════════════════════════════════════════════════════════════
# Test 18 — Fragment merge đúng parent article
# ═══════════════════════════════════════════════════════════════════════════════

class TestFragmentMergeCorrectParent:
    def test_fragment_appended_to_correct_chunk(self):
        """Fragment text is appended to the nearest preceding same-article chunk."""
        units = [
            _make_unit(article="Điều 10", clause=None, point=None,
                       content="Nội dung chính của điều 10 là những quy định chung."),
            _make_unit(article="Điều 10", clause=None, point=None,
                       content="tiếp nối nội dung"),  # fragment
        ]
        doc = _make_parsed_doc("36/2024/QH15", units)
        chunks = chunk_document(doc, {})
        cleaned, _, removed = merge_fragment_chunks(chunks)

        assert len(cleaned) == 1
        assert "Nội dung chính" in cleaned[0]["content"]
        assert "tiếp nối nội dung" in cleaned[0]["content"]

    def test_merge_keeps_original_chunk_id(self):
        """The merged chunk retains the chunk_id of the target (not the fragment)."""
        units = [
            _make_unit(article="Điều 10", clause=None, point=None,
                       content="Nội dung chính của Điều 10 này gồm nhiều phần."),
            _make_unit(article="Điều 10", clause=None, point=None,
                       content="phần phụ tiếp theo"),
        ]
        doc = _make_parsed_doc("36/2024/QH15", units)
        chunks = chunk_document(doc, {})
        original_id = chunks[0]["chunk_id"]
        cleaned, _, _ = merge_fragment_chunks(chunks)

        assert cleaned[0]["chunk_id"] == original_id

    def test_clause_level_short_content_not_merged(self):
        """Clause- or point-level short chunks are NOT fragments — they are
        valid legal enumeration items and must not be merged."""
        units = [
            _make_unit(article="Điều 11", clause="2", point="c)",
                       content="Biển báo hiệu đường bộ;"),  # 23 chars, uppercase
            _make_unit(article="Điều 11", clause="2", point="b)",
                       content="Tín hiệu đèn giao thông;"),
        ]
        doc = _make_parsed_doc("36/2024/QH15", units)
        chunks = chunk_document(doc, {})
        cleaned, _, removed = merge_fragment_chunks(chunks)

        assert len(removed) == 0, "No clause/point-level chunks should be merged"
        assert len(cleaned) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Test 19 — Không merge qua Điều khác
# ═══════════════════════════════════════════════════════════════════════════════

class TestNoMergeAcrossArticles:
    def test_fragment_without_same_article_predecessor_kept(self):
        """If there is no preceding chunk with the same article, the fragment
        must be kept as-is (not discarded, not merged into a different article)."""
        units = [
            _make_unit(article="Điều 1", clause=None, point=None,
                       content="Điều 1 có nội dung rất quan trọng về phạm vi điều chỉnh."),
            _make_unit(article="Điều 2", clause=None, point=None,
                       content="tiếp nối từ điều 1"),  # fragment, but article=Điều 2
        ]
        doc = _make_parsed_doc("36/2024/QH15", units)
        chunks = chunk_document(doc, {})
        cleaned, _, removed = merge_fragment_chunks(chunks)

        assert len(removed) == 0
        assert len(cleaned) == 2

    def test_fragment_merged_into_same_article_not_different(self):
        """Fragment of Điều 10 must not be merged into Điều 9."""
        units = [
            _make_unit(article="Điều 9", clause=None, point=None,
                       content="Nội dung Điều 9 là các quy tắc giao thông cơ bản."),
            _make_unit(article="Điều 10", clause=None, point=None,
                       content="nội dung tiếp theo"),  # fragment of Dieu 10
        ]
        doc = _make_parsed_doc("36/2024/QH15", units)
        chunks = chunk_document(doc, {})
        cleaned, _, removed = merge_fragment_chunks(chunks)

        assert len(removed) == 0
        assert len(cleaned) == 2
        dieu9 = next(c for c in cleaned if c["article"] == "Điều 9")
        assert "nội dung tiếp theo" not in dieu9["content"]


# ═══════════════════════════════════════════════════════════════════════════════
# Test 20 — source_file không mất sau merge
# ═══════════════════════════════════════════════════════════════════════════════

class TestSourceFilePreservedAfterMerge:
    def test_source_file_of_target_chunk_preserved(self):
        """Merge must not clear the target chunk's source_file field."""
        units = [
            _make_unit(article="Điều 5", source_file="36-2024-qh15.pdf",
                       clause=None, point=None,
                       content="Điều 5 quy định về các nguyên tắc cơ bản của luật."),
            _make_unit(article="Điều 5", source_file="36-2024-qh15.pdf",
                       clause=None, point=None,
                       content="phần tiếp theo của điều"),
        ]
        doc = _make_parsed_doc("36/2024/QH15", units)
        chunks = chunk_document(doc, {})
        cleaned, _, _ = merge_fragment_chunks(chunks)

        assert cleaned[0]["source_file"] == "36-2024-qh15.pdf"

    def test_all_chunks_have_source_file_after_merge(self):
        """No chunk should lose its source_file after the merge pipeline."""
        units = _CLAUSE_UNITS + _POINT_UNITS
        doc = _make_parsed_doc("36/2024/QH15", units)
        chunks = chunk_document(doc, {})
        cleaned, _, _ = merge_fragment_chunks(chunks)

        for c in cleaned:
            assert c.get("source_file"), (
                f"chunk {c['chunk_id']} lost source_file after merge"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 21 — Page metadata không mất sau merge
# ═══════════════════════════════════════════════════════════════════════════════

class TestPageMetadataAfterMerge:
    def test_start_page_preserved_after_merge(self):
        """start_page of the merged target must remain unchanged."""
        units = [
            _make_unit(article="Điều 7", clause=None, point=None,
                       start_page=5, end_page=5,
                       content="Nội dung Điều 7 rất quan trọng với hệ thống pháp luật."),
            _make_unit(article="Điều 7", clause=None, point=None,
                       start_page=6, end_page=6,
                       content="phần tiếp nối qua trang"),  # cross-page fragment
        ]
        doc = _make_parsed_doc("36/2024/QH15", units)
        chunks = chunk_document(doc, {})
        cleaned, _, removed = merge_fragment_chunks(chunks)

        assert len(removed) == 1
        assert cleaned[0]["start_page"] == 5
        assert cleaned[0]["end_page"] == 6

    def test_no_chunk_loses_page_info(self):
        """All surviving chunks must have non-None start_page and end_page."""
        units = _CLAUSE_UNITS + _POINT_UNITS
        doc = _make_parsed_doc("36/2024/QH15", units)
        chunks = chunk_document(doc, {})
        cleaned, _, _ = merge_fragment_chunks(chunks)
        for c in cleaned:
            assert c["start_page"] is not None
            assert c["end_page"] is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Test 22 — chunk_id sau cleanup vẫn unique
# ═══════════════════════════════════════════════════════════════════════════════

class TestChunkIdUniqueAfterCleanup:
    def test_no_duplicate_chunk_ids_after_merge(self):
        """merge_fragment_chunks must not produce duplicate chunk_ids."""
        units = [
            _make_unit(article="Điều 1", clause=None, point=None,
                       content="Nội dung của Điều 1 là những quy định chung."),
            _make_unit(article="Điều 1", clause=None, point=None,
                       content="phần tiếp"),   # fragment
            _make_unit(article="Điều 2", clause="1", point=None,
                       content="Khoản 1 Điều 2 có nội dung xử lý."),
            _make_unit(article="Điều 2", clause="2", point=None,
                       content="Khoản 2 Điều 2 có nội dung khác."),
        ]
        doc = _make_parsed_doc("36/2024/QH15", units)
        chunks = chunk_document(doc, {})
        cleaned, _, _ = merge_fragment_chunks(chunks)

        ids = [c["chunk_id"] for c in cleaned]
        assert len(ids) == len(set(ids)), f"Duplicate chunk_ids after merge: {ids}"

    def test_compute_stats_zero_duplicates_after_merge(self):
        """compute_stats must report 0 duplicates after merge."""
        units = _CLAUSE_UNITS + _POINT_UNITS
        doc = _make_parsed_doc("36/2024/QH15", units)
        chunks = chunk_document(doc, {})
        cleaned, _, _ = merge_fragment_chunks(chunks)
        stats = compute_stats(cleaned)
        assert stats["duplicate_chunk_ids"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Test 23 — Content vẫn UTF-8 sau merge
# ═══════════════════════════════════════════════════════════════════════════════

class TestUtf8AfterMerge:
    def test_merged_content_preserves_vietnamese(self):
        """After merge, the combined content must still encode correctly as UTF-8."""
        main_content = "Người điều khiển phương tiện phải chấp hành tín hiệu."
        frag_content = "quy tắc giao thông"
        units = [
            _make_unit(article="Điều 8", clause=None, point=None,
                       content=main_content),
            _make_unit(article="Điều 8", clause=None, point=None,
                       content=frag_content),
        ]
        doc = _make_parsed_doc("36/2024/QH15", units)
        chunks = chunk_document(doc, {})
        cleaned, _, removed = merge_fragment_chunks(chunks)

        assert len(removed) == 1
        merged_content = cleaned[0]["content"]
        encoded = merged_content.encode("utf-8")
        decoded = encoded.decode("utf-8")
        assert decoded == merged_content
        assert "Người" in decoded
        assert "quy tắc" in decoded

    def test_json_dump_no_unicode_escapes_after_merge(self, tmp_path):
        """JSON output of merged chunks must not contain backslash-u escapes."""
        units = [
            _make_unit(article="Điều 8", clause=None, point=None,
                       content="Người điều khiển phải chấp hành đèn đỏ."),
            _make_unit(article="Điều 8", clause=None, point=None,
                       content="và biển báo"),
        ]
        doc = _make_parsed_doc("36/2024/QH15", units)
        chunks = chunk_document(doc, {})
        cleaned, _, _ = merge_fragment_chunks(chunks)

        import json as _json
        out = tmp_path / "merged.json"
        with out.open("w", encoding="utf-8") as f:
            _json.dump(cleaned, f, ensure_ascii=False)
        raw = out.read_text(encoding="utf-8")
        assert chr(92) + "u" not in raw  # check no \uXXXX escapes


# ═══════════════════════════════════════════════════════════════════════════════
# Test 24 — Amendment metadata vẫn còn sau cleanup
# ═══════════════════════════════════════════════════════════════════════════════

class TestAmendmentMetadataAfterCleanup:
    def test_amends_document_preserved_after_merge_pipeline(self):
        """amends_document must survive the merge_fragment_chunks pipeline."""
        cfg = DOC_CONFIG["238/2026/NĐ-CP"]
        chunk = chunk_unit(_ND238_UNIT, "ND238", cfg, {})
        cleaned, _, _ = merge_fragment_chunks([chunk])

        assert cleaned[0]["amends_document"] == "168/2024/NĐ-CP"

    def test_amendment_count_unchanged_after_merge(self):
        """After merge, the number of amendment chunks should not decrease."""
        nd238_units = [
            _make_unit(
                doc_number="238/2026/NĐ-CP",
                doc_title="NĐ 238",
                article="Điều 2",
                article_title="Sửa đổi, bổ sung khoản 3 Điều 6",
                clause="1", point=None,
                content="Nội dung sửa đổi khoản 3 Điều 6.",
            ),
            _make_unit(
                doc_number="238/2026/NĐ-CP",
                doc_title="NĐ 238",
                article="Điều 3",
                article_title="Sửa đổi, bổ sung điểm b khoản 8 Điều 13",
                clause=None, point=None,
                content="Nội dung sửa đổi điểm b.",
            ),
        ]
        doc = _make_parsed_doc("238/2026/NĐ-CP", nd238_units)
        chunks = chunk_document(doc, {})
        amendment_count_before = sum(1 for c in chunks if c.get("amends_document"))
        cleaned, _, _ = merge_fragment_chunks(chunks)
        amendment_count_after = sum(1 for c in cleaned if c.get("amends_document"))
        assert amendment_count_after == amendment_count_before


# ═══════════════════════════════════════════════════════════════════════════════
# Test 25 — issue_date được extract cho cả 3 documents
# ═══════════════════════════════════════════════════════════════════════════════

class TestIssueDateExtracted:
    def test_luat36_issue_date_not_null(self):
        """Luật 36 issue_date must be '2024-06-27' (from raw text page 1)."""
        cfg = DOC_CONFIG["36/2024/QH15"]
        assert cfg["issue_date"] == "2024-06-27"

    def test_nd168_issue_date_not_null(self):
        """NĐ 168 issue_date must be '2024-12-26'."""
        cfg = DOC_CONFIG["168/2024/NĐ-CP"]
        assert cfg["issue_date"] == "2024-12-26"

    def test_nd238_issue_date_not_null(self):
        """NĐ 238 issue_date must be '2026-06-26'."""
        cfg = DOC_CONFIG["238/2026/NĐ-CP"]
        assert cfg["issue_date"] == "2026-06-26"

    def test_issue_date_propagated_to_chunk(self):
        """issue_date must appear in every chunk produced from that document."""
        for doc_number, expected_date in [
            ("36/2024/QH15",   "2024-06-27"),
            ("168/2024/NĐ-CP", "2024-12-26"),
            ("238/2026/NĐ-CP", "2026-06-26"),
        ]:
            unit = _make_unit(doc_number=doc_number)
            cfg = DOC_CONFIG[doc_number]
            prefix = {"36/2024/QH15": "L36",
                      "168/2024/NĐ-CP": "ND168",
                      "238/2026/NĐ-CP": "ND238"}[doc_number]
            chunk = chunk_unit(unit, prefix, cfg, {})
            assert chunk["issue_date"] == expected_date, (
                f"{doc_number}: expected {expected_date!r}, got {chunk['issue_date']!r}"
            )

    def test_effective_date_configured(self):
        """effective_date manually configured must be preserved."""
        assert DOC_CONFIG["36/2024/QH15"]["effective_date"] == "2025-01-01"
        assert DOC_CONFIG["168/2024/NĐ-CP"]["effective_date"] == "2025-01-01"
        assert DOC_CONFIG["238/2026/NĐ-CP"]["effective_date"] == "2026-08-15"

    def test_effective_date_propagated_to_chunk(self):
        """effective_date must appear in every chunk produced from that document."""
        for doc_number, expected_date in [
            ("36/2024/QH15",   "2025-01-01"),
            ("168/2024/NĐ-CP", "2025-01-01"),
            ("238/2026/NĐ-CP", "2026-08-15"),
        ]:
            unit = _make_unit(doc_number=doc_number)
            cfg = DOC_CONFIG[doc_number]
            prefix = {"36/2024/QH15": "L36",
                      "168/2024/NĐ-CP": "ND168",
                      "238/2026/NĐ-CP": "ND238"}[doc_number]
            chunk = chunk_unit(unit, prefix, cfg, {})
            assert chunk["effective_date"] == expected_date, (
                f"{doc_number}: expected {expected_date!r}, got {chunk['effective_date']!r}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 26 — legal_status = "unknown" (không hard-code "in_force")
# ═══════════════════════════════════════════════════════════════════════════════

class TestLegalStatusNotHardcoded:
    def test_luat36_legal_status_unknown(self):
        """Luật 36 must use 'unknown', not the previously hard-coded 'in_force'."""
        assert DOC_CONFIG["36/2024/QH15"]["legal_status"] == "unknown"

    def test_nd168_legal_status_unknown(self):
        assert DOC_CONFIG["168/2024/NĐ-CP"]["legal_status"] == "unknown"

    def test_nd238_legal_status_unknown(self):
        assert DOC_CONFIG["238/2026/NĐ-CP"]["legal_status"] == "unknown"

    def test_legal_status_not_in_force(self):
        """No document in DOC_CONFIG may have legal_status = 'in_force'."""
        for doc_number, cfg in DOC_CONFIG.items():
            assert cfg["legal_status"] != "in_force", (
                f"{doc_number} still has legal_status='in_force'"
            )

    def test_legal_status_propagated_to_chunk(self):
        """legal_status field in every chunk must not be 'in_force'."""
        for doc_number in ["36/2024/QH15", "168/2024/NĐ-CP", "238/2026/NĐ-CP"]:
            unit = _make_unit(doc_number=doc_number)
            cfg = DOC_CONFIG[doc_number]
            prefix = {"36/2024/QH15": "L36",
                      "168/2024/NĐ-CP": "ND168",
                      "238/2026/NĐ-CP": "ND238"}[doc_number]
            chunk = chunk_unit(unit, prefix, cfg, {})
            assert chunk["legal_status"] != "in_force", (
                f"{doc_number}: chunk has legal_status='in_force'"
            )
            assert chunk["legal_status"] == "unknown"
