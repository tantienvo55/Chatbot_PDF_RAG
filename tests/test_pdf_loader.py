"""
Tests for src/ingestion/pdf_loader.py

Coverage (7 required + 1 bonus):
  1. load_pdf() on an existing PDF succeeds
  2. load_pdf() on a missing PDF raises FileNotFoundError
  3. Output has correct page count
  4. Every page has all required fields with correct types
  5. Both Luật 36 files share the same document_id ("36/2024/QH15")
  6. JSON output file is created on disk
  7. UTF-8 Vietnamese characters survive the JSON round-trip
  8. BONUS: get_document_id() mapping is correct for all 4 known stems

Notes:
- Tests use actual PDF files in data/raw/ (integration-style).
- tmp_path (pytest built-in fixture) is used for JSON output isolation.
- PyMuPDF is used independently to verify page counts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pymupdf
import pytest

# ── Make src importable from any working directory ────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.pdf_loader import (
    DOCUMENT_ID_MAP,
    DocumentRecord,
    get_document_id,
    load_pdf,
)

# ─── Paths ────────────────────────────────────────────────────────────────────
RAW_DIR = PROJECT_ROOT / "data" / "raw"

PDF_36_1 = RAW_DIR / "36-2024-qh15.pdf"
PDF_36_2 = RAW_DIR / "36-2024-qh15_tiep.pdf"
PDF_168  = RAW_DIR / "168_2024_ND-CP_619502.pdf"
PDF_238  = RAW_DIR / "238_2026_ND-CP_712521.pdf"

ALL_PDFS = [PDF_36_1, PDF_36_2, PDF_168, PDF_238]

# Required fields in every page record
REQUIRED_PAGE_FIELDS = {
    "source_file",
    "document_id",
    "page",
    "text",
    "extraction_method",
    "char_count",
}

VALID_METHODS = {"text", "ocr", "error"}

# ─── Helper ───────────────────────────────────────────────────────────────────

def pdf_page_count(pdf_path: Path) -> int:
    """Return real page count via PyMuPDF (independent of the loader)."""
    doc = pymupdf.open(str(pdf_path))
    n = len(doc)
    doc.close()
    return n


# ═══════════════════════════════════════════════════════════════════════════════
# Test 1 — Load existing PDF
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("pdf_path", ALL_PDFS)
def test_load_existing_pdf(pdf_path: Path, tmp_path: Path) -> None:
    """load_pdf() on each real PDF must return a non-empty DocumentRecord."""
    record = load_pdf(pdf_path, tmp_path)

    assert isinstance(record, DocumentRecord)
    assert record.source_file == pdf_path.name
    assert record.document_id != ""
    assert record.total_pages > 0
    assert len(record.pages) == record.total_pages


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2 — Load non-existing PDF
# ═══════════════════════════════════════════════════════════════════════════════

def test_load_nonexistent_pdf_raises(tmp_path: Path) -> None:
    """load_pdf() on a non-existing path must raise FileNotFoundError."""
    missing = RAW_DIR / "does_not_exist_at_all.pdf"
    with pytest.raises(FileNotFoundError):
        load_pdf(missing, tmp_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Test 3 — Correct page count
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("pdf_path", ALL_PDFS)
def test_page_count_matches_pdf(pdf_path: Path, tmp_path: Path) -> None:
    """total_pages and len(pages) must equal the actual PDF page count."""
    expected = pdf_page_count(pdf_path)
    record = load_pdf(pdf_path, tmp_path)

    assert record.total_pages == expected, (
        f"{pdf_path.name}: expected {expected} pages, got {record.total_pages}"
    )
    assert len(record.pages) == expected


# ═══════════════════════════════════════════════════════════════════════════════
# Test 4 — Every page has all required fields
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("pdf_path", ALL_PDFS)
def test_page_has_required_fields(pdf_path: Path, tmp_path: Path) -> None:
    """Each page dict must contain all required fields with correct types."""
    record = load_pdf(pdf_path, tmp_path)

    for page in record.pages:
        missing = REQUIRED_PAGE_FIELDS - set(page.keys())
        assert not missing, (
            f"{pdf_path.name} page {page.get('page')}: missing fields {missing}"
        )

        # Type checks
        assert isinstance(page["source_file"], str) and page["source_file"]
        assert isinstance(page["document_id"], str) and page["document_id"]
        assert isinstance(page["page"], int) and page["page"] >= 1
        assert isinstance(page["text"], str)           # may be empty on error pages
        assert page["extraction_method"] in VALID_METHODS, (
            f"Unexpected extraction_method: {page['extraction_method']!r}"
        )
        assert isinstance(page["char_count"], int) and page["char_count"] >= 0
        assert page["char_count"] == len(page["text"])


# ═══════════════════════════════════════════════════════════════════════════════
# Test 5 — Both Luật 36 parts share the same document_id
# ═══════════════════════════════════════════════════════════════════════════════

def test_luat36_same_document_id(tmp_path: Path) -> None:
    """36-2024-qh15.pdf and 36-2024-qh15_tiep.pdf must share document_id."""
    record1 = load_pdf(PDF_36_1, tmp_path)
    record2 = load_pdf(PDF_36_2, tmp_path)

    assert record1.document_id == "36/2024/QH15", (
        f"Part 1 got: {record1.document_id!r}"
    )
    assert record2.document_id == "36/2024/QH15", (
        f"Part 2 got: {record2.document_id!r}"
    )
    assert record1.document_id == record2.document_id


# ═══════════════════════════════════════════════════════════════════════════════
# Test 6 — JSON output file is created
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("pdf_path,expected_stem", [
    (PDF_36_1, "36-2024-qh15"),
    (PDF_36_2, "36-2024-qh15_tiep"),
    (PDF_168,  "168_2024_ND-CP_619502"),
    (PDF_238,  "238_2026_ND-CP_712521"),
])
def test_json_output_created(
    pdf_path: Path, expected_stem: str, tmp_path: Path
) -> None:
    """A <stem>.json file must exist in output_dir after load_pdf()."""
    load_pdf(pdf_path, tmp_path)
    output_file = tmp_path / f"{expected_stem}.json"
    assert output_file.exists(), f"Expected JSON not found: {output_file}"
    assert output_file.stat().st_size > 0, "JSON file is empty"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 7 — UTF-8 Vietnamese characters preserved
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("pdf_path,stem", [
    (PDF_168, "168_2024_ND-CP_619502"),
    (PDF_36_1, "36-2024-qh15"),
])
def test_vietnamese_utf8_preserved(
    pdf_path: Path, stem: str, tmp_path: Path
) -> None:
    """JSON must be valid UTF-8 and contain Vietnamese diacritics."""
    load_pdf(pdf_path, tmp_path)

    json_file = tmp_path / f"{stem}.json"
    raw_bytes = json_file.read_bytes()

    # Must decode cleanly as UTF-8
    content = raw_bytes.decode("utf-8")

    # Must not have Unicode escape sequences (ensure_ascii=False is set)
    escaped_chars = content.count("\\u")
    assert escaped_chars == 0, (
        f"Found {escaped_chars} Unicode escapes — ensure_ascii may be True"
    )

    # Must contain Vietnamese-specific characters somewhere in the text
    data = json.loads(content)
    all_text = " ".join(p["text"] for p in data["pages"] if p["text"])
    viet_chars = "àáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ"
    found = any(ch in all_text for ch in viet_chars)
    assert found, (
        "No Vietnamese diacritics found — possible encoding corruption\n"
        f"Sample (first 200 chars): {all_text[:200]!r}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 8 (BONUS) — document_id mapping correctness
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("stem,expected_id", [
    ("36-2024-qh15",          "36/2024/QH15"),
    ("36-2024-qh15_tiep",     "36/2024/QH15"),
    ("168_2024_ND-CP_619502", "168/2024/NĐ-CP"),
    ("238_2026_ND-CP_712521", "238/2026/NĐ-CP"),
])
def test_get_document_id_mapping(stem: str, expected_id: str) -> None:
    """get_document_id() must return the correct legal ID for all 4 stems."""
    assert get_document_id(stem) == expected_id


def test_get_document_id_unknown_stem() -> None:
    """Unknown stems must return the stem itself (with a warning, no crash)."""
    result = get_document_id("unknown_file_xyz")
    assert result == "unknown_file_xyz"
