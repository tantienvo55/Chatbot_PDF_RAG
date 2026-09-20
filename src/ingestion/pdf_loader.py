"""
PDF Loader with OCR fallback for Vietnamese legal documents.

Strategy:
  1. Extract text directly from each page using PyMuPDF.
  2. If extracted text is shorter than MIN_TEXT_CHARS, render the page to a
     PNG image and run Tesseract OCR (language: vie) via subprocess.
  3. Save per-document output as UTF-8 JSON to data/processed/raw_text/.

Usage (CLI):
    python -m src.ingestion.pdf_loader
    python src/ingestion/pdf_loader.py
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import pymupdf  # PyMuPDF >= 1.24 — use `import pymupdf` (fitz API deprecated)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Config constants — change here only, not scattered in logic
# ─────────────────────────────────────────────────────────────────────────────

MIN_TEXT_CHARS: int = 50        # pages with fewer chars trigger OCR fallback
OCR_DPI: int = 300              # render resolution for OCR
OCR_LANG: str = "vie"           # Tesseract language code
OCR_TIMEOUT_SEC: int = 120      # per-page OCR subprocess timeout

# Known Windows install path; also checked via PATH (shutil.which)
TESSERACT_KNOWN_PATH: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ─────────────────────────────────────────────────────────────────────────────
# Document ID mapping
# ─────────────────────────────────────────────────────────────────────────────

#: Maps PDF filename stem → canonical legal document ID.
#: 36-2024-qh15 and 36-2024-qh15_tiep are two parts of the same statute.
DOCUMENT_ID_MAP: dict[str, str] = {
    "36-2024-qh15":          "36/2024/QH15",
    "36-2024-qh15_tiep":     "36/2024/QH15",
    "168_2024_ND-CP_619502": "168/2024/NĐ-CP",
    "238_2026_ND-CP_712521": "238/2026/NĐ-CP",
}

# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

#: Valid extraction methods stored in each page record.
EXTRACTION_TEXT = "text"
EXTRACTION_OCR = "ocr"
EXTRACTION_ERROR = "error"


@dataclass
class PageRecord:
    """Per-page extraction result."""

    source_file: str          # PDF filename, e.g. "168_2024_ND-CP_619502.pdf"
    document_id: str          # Legal ID,  e.g. "168/2024/NĐ-CP"
    page: int                 # 1-indexed page number
    text: str                 # Extracted / OCR'd text
    extraction_method: str    # "text" | "ocr" | "error"
    char_count: int           # len(text)

    @classmethod
    def make_error(
        cls,
        source_file: str,
        document_id: str,
        page: int,
    ) -> "PageRecord":
        """Convenience constructor for pages that failed completely."""
        return cls(
            source_file=source_file,
            document_id=document_id,
            page=page,
            text="",
            extraction_method=EXTRACTION_ERROR,
            char_count=0,
        )


@dataclass
class DocumentRecord:
    """Full extraction result for one PDF file."""

    document_id: str
    source_file: str
    total_pages: int
    pages: list[dict] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Tesseract utilities
# ─────────────────────────────────────────────────────────────────────────────


def find_tesseract() -> Optional[str]:
    """
    Locate the Tesseract executable.

    Checks PATH first (covers session-level additions), then falls back to
    the known Windows install path.

    Returns:
        Absolute path string if found, else None.
    """
    tess = shutil.which("tesseract")
    if tess:
        return tess
    known = Path(TESSERACT_KNOWN_PATH)
    if known.exists():
        return str(known)
    return None


def ocr_pixmap(pixmap: pymupdf.Pixmap, tesseract_path: str) -> str:
    """
    Run Tesseract OCR on a rendered page pixmap.

    Saves the pixmap to a temporary PNG file, invokes Tesseract via
    subprocess (stdout mode), then cleans up the temp file.

    Args:
        pixmap: PyMuPDF Pixmap of the rendered page.
        tesseract_path: Absolute path to the tesseract executable.

    Returns:
        OCR'd text string (may be empty if Tesseract finds nothing).

    Raises:
        subprocess.TimeoutExpired: If OCR takes longer than OCR_TIMEOUT_SEC.
        OSError: If the temp file cannot be written.
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        pixmap.save(str(tmp_path))
        result = subprocess.run(
            [tesseract_path, str(tmp_path), "stdout", "-l", OCR_LANG],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=OCR_TIMEOUT_SEC,
        )
        if result.returncode != 0:
            logger.warning(
                "Tesseract returned code %d: %s",
                result.returncode,
                result.stderr.strip()[:200],
            )
            return ""
        return result.stdout.strip()
    finally:
        tmp_path.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────────────────


def get_document_id(pdf_stem: str) -> str:
    """
    Map a PDF filename stem to the canonical legal document ID.

    Args:
        pdf_stem: Filename without extension, e.g. "168_2024_ND-CP_619502".

    Returns:
        Legal document ID string.  Falls back to the stem itself with a warning
        if the stem is not in DOCUMENT_ID_MAP.
    """
    doc_id = DOCUMENT_ID_MAP.get(pdf_stem)
    if doc_id is None:
        logger.warning(
            "Unknown PDF stem '%s' — not in DOCUMENT_ID_MAP. "
            "Using stem as document_id.",
            pdf_stem,
        )
        return pdf_stem
    return doc_id


def _extract_page(
    page_obj: pymupdf.Page,
    page_1indexed: int,
    source_file: str,
    document_id: str,
    tesseract_path: Optional[str],
) -> PageRecord:
    """
    Extract text from a single page, with OCR fallback.

    Args:
        page_obj: PyMuPDF Page object.
        page_1indexed: 1-based page number (for human-readable output).
        source_file: PDF filename string (stored in the record).
        document_id: Legal document ID (stored in the record).
        tesseract_path: Path to Tesseract, or None if unavailable.

    Returns:
        PageRecord with extracted text and method tag.
    """
    # ── Step 1: Direct text extraction ──────────────────────────────────────
    try:
        raw_text = page_obj.get_text("text")
    except Exception as exc:
        logger.error("Page %d: PyMuPDF get_text() failed — %s", page_1indexed, exc)
        return PageRecord.make_error(source_file, document_id, page_1indexed)

    clean_text = raw_text.strip()

    # ── Step 2: Quality gate ─────────────────────────────────────────────────
    if len(clean_text) >= MIN_TEXT_CHARS:
        return PageRecord(
            source_file=source_file,
            document_id=document_id,
            page=page_1indexed,
            text=clean_text,
            extraction_method=EXTRACTION_TEXT,
            char_count=len(clean_text),
        )

    # ── Step 3: OCR fallback ─────────────────────────────────────────────────
    if tesseract_path is None:
        logger.warning(
            "Page %d: text too short (%d chars) — Tesseract unavailable. "
            "Keeping raw text as-is.",
            page_1indexed,
            len(clean_text),
        )
        return PageRecord(
            source_file=source_file,
            document_id=document_id,
            page=page_1indexed,
            text=clean_text,
            extraction_method=EXTRACTION_TEXT,
            char_count=len(clean_text),
        )

    try:
        mat = pymupdf.Matrix(OCR_DPI / 72, OCR_DPI / 72)
        pixmap = page_obj.get_pixmap(matrix=mat, colorspace=pymupdf.csRGB)
        ocr_text = ocr_pixmap(pixmap, tesseract_path)
        return PageRecord(
            source_file=source_file,
            document_id=document_id,
            page=page_1indexed,
            text=ocr_text,
            extraction_method=EXTRACTION_OCR,
            char_count=len(ocr_text),
        )
    except subprocess.TimeoutExpired:
        logger.error("Page %d: OCR timed out after %ds.", page_1indexed, OCR_TIMEOUT_SEC)
        return PageRecord.make_error(source_file, document_id, page_1indexed)
    except Exception as exc:
        logger.error("Page %d: OCR failed — %s", page_1indexed, exc)
        return PageRecord.make_error(source_file, document_id, page_1indexed)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def load_pdf(pdf_path: Path, output_dir: Path) -> DocumentRecord:
    """
    Load a single PDF, extract text page-by-page (OCR fallback where needed),
    and write a JSON file to output_dir.

    Args:
        pdf_path: Absolute or relative path to the PDF.
        output_dir: Directory where <stem>.json will be saved.

    Returns:
        DocumentRecord with all page records.

    Raises:
        FileNotFoundError: If pdf_path does not exist.
        ValueError: If the PDF cannot be opened by PyMuPDF.
        OSError: If the JSON output file cannot be written.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    stem = pdf_path.stem
    source_file = pdf_path.name
    document_id = get_document_id(stem)

    tesseract_path = find_tesseract()
    if tesseract_path is None:
        logger.warning(
            "Tesseract not found in PATH or at '%s'. "
            "OCR fallback will be unavailable.",
            TESSERACT_KNOWN_PATH,
        )

    t_start = time.perf_counter()

    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as exc:
        raise ValueError(f"Cannot open PDF '{source_file}': {exc}") from exc

    pages: list[dict] = []
    text_count = 0
    ocr_count = 0
    error_count = 0

    try:
        for page_idx in range(len(doc)):
            page_1indexed = page_idx + 1
            page_obj = doc[page_idx]

            record = _extract_page(
                page_obj=page_obj,
                page_1indexed=page_1indexed,
                source_file=source_file,
                document_id=document_id,
                tesseract_path=tesseract_path,
            )

            if record.extraction_method == EXTRACTION_TEXT:
                text_count += 1
            elif record.extraction_method == EXTRACTION_OCR:
                ocr_count += 1
            else:
                error_count += 1

            pages.append(asdict(record))
    finally:
        doc.close()

    elapsed = time.perf_counter() - t_start
    total_chars = sum(p["char_count"] for p in pages)

    # ── Console summary (UTF-8 safe on Windows) ──────────────────────────────
    sep = "─" * 56
    summary = (
        f"\n{sep}\n"
        f"  {source_file}\n"
        f"  Pages:      {len(pages)}\n"
        f"  Text pages: {text_count}\n"
        f"  OCR pages:  {ocr_count}\n"
        f"  Errors:     {error_count}\n"
        f"  Characters: {total_chars:,}\n"
        f"  Elapsed:    {elapsed:.2f}s\n"
        f"{sep}\n"
    )
    sys.stdout.buffer.write(summary.encode("utf-8", errors="replace"))
    sys.stdout.buffer.flush()

    document_record = DocumentRecord(
        document_id=document_id,
        source_file=source_file,
        total_pages=len(pages),
        pages=pages,
    )

    # ── Write JSON ───────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{stem}.json"
    try:
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(asdict(document_record), fh, ensure_ascii=False, indent=2)
        logger.info("Saved → %s", output_path)
    except OSError as exc:
        raise OSError(f"Failed to write JSON '{output_path}': {exc}") from exc

    return document_record


def load_all_pdfs(raw_dir: Path, output_dir: Path) -> list[DocumentRecord]:
    """
    Process every PDF file in raw_dir and save JSON output to output_dir.

    Files that raise FileNotFoundError or ValueError are logged and skipped;
    the batch continues.

    Args:
        raw_dir: Directory containing source PDF files.
        output_dir: Destination directory for JSON output.

    Returns:
        List of successfully processed DocumentRecord objects.
    """
    pdf_files = sorted(raw_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", raw_dir)
        return []

    records: list[DocumentRecord] = []
    for pdf_path in pdf_files:
        logger.info("Processing: %s", pdf_path.name)
        try:
            record = load_pdf(pdf_path, output_dir)
            records.append(record)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("Skipping %s — %s", pdf_path.name, exc)

    return records


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve project root = two levels above this file (src/ingestion/pdf_loader.py)
    project_root = Path(__file__).resolve().parents[2]
    raw_dir = project_root / "data" / "raw"
    output_dir = project_root / "data" / "processed" / "raw_text"

    if not raw_dir.exists():
        print(f"ERROR: {raw_dir} does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"Source : {raw_dir}")
    print(f"Output : {output_dir}")

    records = load_all_pdfs(raw_dir, output_dir)
    print(f"\nFinished. Processed {len(records)} / 4 PDF(s).")
