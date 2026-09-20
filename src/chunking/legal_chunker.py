"""
Legal Chunker for Vietnamese legal documents.

Converts flat LegalUnit records (output of legal_parser) into LegalChunk records
with enriched metadata, deterministic chunk_ids, and amendment relationship fields.

Chunking strategy:
  - One LegalUnit → one LegalChunk (parser already segments at the correct
    legal granularity: article / clause / point level).
  - content_with_context prepends structural headers so embedding models
    (e.g. BGE-M3) receive full legal context.
  - Amendment metadata is extracted from article_title patterns in NĐ 238.
  - Fragment merge: article-level chunks whose content is a mid-sentence
    fragment (OCR/PDF artifact) are merged into their preceding sibling
    within the same article, preserving all legal text.

Input : data/processed/parsed/  (JSON files from legal_parser)
Output: data/chunks/legal_chunks.json
        data/chunks/legal_chunks.jsonl

Usage (CLI):
    python -m src.chunking.legal_chunker
    python src/chunking/legal_chunker.py
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ─── Document Configuration ───────────────────────────────────────────────────

#: Static metadata per document group (keyed by doc_number).
#:
#: issue_date: extracted from the first page of the raw PDF text (the
#:   "Hà Nội, ngày DD tháng MM năm YYYY" line).  Only populated when the
#:   date appears unambiguously in the source document.
#:
#: effective_date: not present in the available corpus → null.
#:
#: legal_status: set to "unknown" because the corpus does not contain
#:   gazette (Công báo) revocation data needed to confirm "in_force".
#:   The amendment relationship (238 amends 168) is captured separately
#:   via amends_document.
DOC_CONFIG: dict[str, dict] = {
    "36/2024/QH15": {
        "document_type":   "law",
        "legal_status":    "unknown",   # cannot verify from corpus
        "issue_date":      "2024-06-27",  # "ngày 27 tháng 6 năm 2024" (raw p.1)
        "effective_date":  "2025-01-01",
        "amends_document": None,
    },
    "168/2024/NĐ-CP": {
        "document_type":   "decree",
        "legal_status":    "unknown",
        "issue_date":      "2024-12-26",  # "ngày 26 tháng 12 năm 2024" (raw p.1)
        "effective_date":  "2025-01-01",
        "amends_document": None,
    },
    "238/2026/NĐ-CP": {
        "document_type":   "amending_decree",
        "legal_status":    "unknown",
        "issue_date":      "2026-06-26",  # "ngày 26 tháng 6 năm 2026" (raw p.1)
        "effective_date":  "2026-08-15",
        "amends_document": "168/2024/NĐ-CP",
    },
}

#: Maps doc_number → short prefix for chunk_id generation.
CHUNK_ID_PREFIX: dict[str, str] = {
    "36/2024/QH15":   "L36",
    "168/2024/NĐ-CP": "ND168",
    "238/2026/NĐ-CP": "ND238",
}

# ─── Amendment Extraction Patterns ────────────────────────────────────────────

# Ordered list of regex patterns; first match wins.
#
# Captures:
#   group "diem"   — letter label of amended điểm (optional)
#   group "khoan"  — number label of amended khoản (optional)
#   group "dieu"   — number (+ optional letter suffix) of amended điều
#
# Vietnamese Unicode codepoints used:
#   điểm: đ=\u0111  i  ể=\u1ec3  m
#   khoản: k h o  ả=\u1ea3  n
#   Điều: Đ=\u0110 (or D)  i  ề=\u1ec1  u

_AMEND_PATTERNS: list[re.Pattern] = [
    # "điểm X khoản Y Điều Z"  (most specific first)
    re.compile(
        r"[\u0111d]i\u1ec3m\s+(?P<diem>[a-z\u0111])"
        r"\s+kho\u1ea3n\s+(?P<khoan>\d+[a-z\u0111]?)"
        r"\s+[\u0110D]i\u1ec1u\s+(?P<dieu>\d+[a-z\u0111]?)",
        re.UNICODE,
    ),
    # "khoản Y Điều Z"
    re.compile(
        r"kho\u1ea3n\s+(?P<khoan>\d+[a-z\u0111]?)"
        r"\s+[\u0110D]i\u1ec1u\s+(?P<dieu>\d+[a-z\u0111]?)",
        re.UNICODE,
    ),
    # "Điều Z" alone — Đ(\u0110) or D + i + ề(\u1ec1) + u
    re.compile(
        r"[\u0110D]i\u1ec1u\s+(?P<dieu>\d+[a-z\u0111]?)",
        re.UNICODE,
    ),
]


def extract_amendment_info(
    article_title: str,
    doc_number: str,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extract (amended_article, amended_clause, amended_point) from article_title.

    Only applies to amending decrees (doc_number in DOC_CONFIG with
    amends_document != None).  Returns (None, None, None) for all other docs.

    Args:
        article_title: The article_title field from the parsed unit.
        doc_number:    The document number of the *source* document.

    Returns:
        Tuple (amended_article, amended_clause, amended_point);
        each element is a string or None.
    """
    cfg = DOC_CONFIG.get(doc_number, {})
    if not cfg.get("amends_document"):
        return None, None, None

    if not article_title:
        return None, None, None

    for pat in _AMEND_PATTERNS:
        m = pat.search(article_title)
        if m:
            groups = m.groupdict()
            dieu  = groups.get("dieu")
            khoan = groups.get("khoan")
            diem  = groups.get("diem")

            amended_article = f"Điều {dieu}" if dieu else None
            amended_clause  = khoan if khoan else None
            amended_point   = f"{diem})" if diem else None
            return amended_article, amended_clause, amended_point

    return None, None, None


# ─── chunk_id Generation ──────────────────────────────────────────────────────

def _normalize_label(label: str) -> str:
    """
    Convert a Vietnamese structural label to an ASCII-safe chunk_id segment.

    Examples:
        "Điều 25"  → "DIEU25"
        "Điều 11a" → "DIEU11A"
        "1"        → "KHOAN1"
        "1a"       → "KHOAN1A"
        "a)"       → "DIEMA"
        "đ)"       → "DIEMD"   (đ → D)
    """
    s = label.strip()

    # Transliterate Vietnamese chars needed for matching:
    #   Đ (\u0110) / đ (\u0111) → D/d
    #   ề (\u1ec1) and variants  → e  (for "Điều")
    s_ascii = (
        s
        .replace("\u0110", "D")   # Đ
        .replace("\u0111", "d")   # đ
        .replace("\u1ec1", "e")   # ề
        .replace("\u1ec3", "e")   # ể
        .replace("\u1ebf", "e")   # ế
        .replace("\u1ec5", "e")   # ệ
    )

    # Article: "Điều N" or "Điều Na"  (after transliteration: "Dieu N")
    m = re.match(r"[Dd]ieu\s+(\d+[a-zA-Z]?)", s_ascii)
    if m:
        num = m.group(1).upper()
        return f"DIEU{num}"

    # Point: "a)", "d)", "b)", etc.  (đ already → d)
    m = re.match(r"([a-z])\)$", s_ascii)
    if m:
        letter = m.group(1).upper()
        return f"DIEM{letter}"

    # Clause: "1", "2", "1a", "9a"
    m = re.match(r"(\d+[a-zA-Z]?)$", s_ascii)
    if m:
        num = m.group(1).upper()
        return f"KHOAN{num}"

    # Fallback — strip non-alnum from ascii version
    clean = re.sub(r"[^A-Za-z0-9]", "", s_ascii).upper()
    return clean or "UNKNOWN"


def build_chunk_id(
    prefix: str,
    article: str,
    clause: Optional[str],
    point: Optional[str],
    counter: dict,
) -> str:
    """
    Build a deterministic, unique, human-readable chunk_id.

    Collision handling: if the same raw key appears more than once, appends
    ``_v2``, ``_v3``, … to subsequent occurrences.

    Args:
        prefix:  Short doc prefix, e.g. "L36", "ND168".
        article: Article label, e.g. "Điều 25".
        clause:  Clause label or None.
        point:   Point label or None.
        counter: Mutable dict used to track seen keys (pass the same dict
                 across all calls for one document batch).

    Returns:
        Unique chunk_id string.
    """
    parts = [prefix, _normalize_label(article)]
    if clause:
        parts.append(_normalize_label(clause))
    if point:
        parts.append(_normalize_label(point))

    raw_key = "_".join(parts)
    n = counter.get(raw_key, 0) + 1
    counter[raw_key] = n

    return raw_key if n == 1 else f"{raw_key}_v{n}"


# ─── Content with Context ─────────────────────────────────────────────────────

def build_content_with_context(unit: dict) -> str:
    """
    Build a context-enriched string for embedding.

    Prepends structural headers (Điều / Khoản / Điểm) so the embedding model
    receives the full legal position of each chunk without modifying the raw
    ``content`` field.

    Metadata and raw content remain separate; this string is stored in the
    dedicated ``content_with_context`` field.

    Args:
        unit: A LegalUnit dict (original parsed unit) or a LegalChunk dict.

    Returns:
        Multi-line string with structural headers followed by the raw content.
    """
    lines: list[str] = []

    article       = unit.get("article", "")
    article_title = unit.get("article_title", "")
    clause        = unit.get("clause")
    point         = unit.get("point")
    content       = unit.get("content", "")

    # Article header
    if article:
        header = article
        if article_title:
            header += f". {article_title}"
        lines.append(header)

    # Clause header (only if this is clause- or point-level)
    if clause:
        lines.append(f"Khoản {clause}.")

    # Point header
    if point:
        lines.append(f"{point}")

    # Raw content
    if content:
        lines.append(content)

    return "\n".join(lines)


# ─── Core Chunking Functions ──────────────────────────────────────────────────

def chunk_unit(
    unit: dict,
    prefix: str,
    cfg: dict,
    counter: dict,
) -> dict:
    """
    Convert a single LegalUnit dict into a LegalChunk dict.

    Args:
        unit:    A LegalUnit dict from the parsed JSON.
        prefix:  Short doc prefix for chunk_id (e.g. "ND168").
        cfg:     DOC_CONFIG entry for this doc_number.
        counter: Mutable dict for collision tracking in build_chunk_id.

    Returns:
        A LegalChunk dict with all required metadata fields.
    """
    doc_number    = unit.get("doc_number", "")
    article       = unit.get("article", "")
    clause        = unit.get("clause")      # str or None
    point         = unit.get("point")       # str or None
    article_title = unit.get("article_title", "")

    chunk_id = build_chunk_id(prefix, article, clause, point, counter)

    amended_article, amended_clause, amended_point = extract_amendment_info(
        article_title, doc_number
    )

    content_ctx = build_content_with_context(unit)

    return {
        "chunk_id":             chunk_id,
        "doc_id":               doc_number,
        "doc_title":            unit.get("doc_title", ""),
        "doc_number":           doc_number,
        "document_type":        cfg.get("document_type"),
        "issue_date":           cfg.get("issue_date"),
        "effective_date":       cfg.get("effective_date"),
        "legal_status":         cfg.get("legal_status"),
        "chapter":              unit.get("chapter", ""),
        "chapter_title":        unit.get("chapter_title", ""),
        "section":              unit.get("section", ""),
        "section_title":        unit.get("section_title", ""),
        "article":              article,
        "article_title":        article_title,
        "clause":               clause,
        "point":                point,
        "content":              unit.get("content", ""),
        "content_with_context": content_ctx,
        "source_file":          unit.get("source_file", ""),
        "start_page":           unit.get("start_page"),
        "end_page":             unit.get("end_page"),
        "amends_document":      cfg.get("amends_document"),
        "amended_article":      amended_article,
        "amended_clause":       amended_clause,
        "amended_point":        amended_point,
    }


def chunk_document(parsed_doc: dict, counter: dict) -> list[dict]:
    """
    Chunk all LegalUnit records from one parsed document.

    Args:
        parsed_doc: The full parsed JSON dict (keys: doc_number, doc_title,
                    stats, units).
        counter:    Mutable dict for cross-document chunk_id collision tracking.
                    Pass the same dict when chunking multiple documents together.

    Returns:
        List of LegalChunk dicts.
    """
    doc_number = parsed_doc.get("doc_number", "")
    cfg    = DOC_CONFIG.get(doc_number, {
        "document_type":   "unknown",
        "legal_status":    "unknown",
        "issue_date":      None,
        "effective_date":  None,
        "amends_document": None,
    })
    prefix = CHUNK_ID_PREFIX.get(doc_number, "DOC")

    chunks: list[dict] = []
    for unit in parsed_doc.get("units", []):
        chunk = chunk_unit(unit, prefix, cfg, counter)
        chunks.append(chunk)

    logger.info(
        "%s: %d units → %d chunks",
        doc_number, len(parsed_doc.get("units", [])), len(chunks)
    )
    return chunks


# ─── Fragment Detection & Merge ───────────────────────────────────────────────

# Vietnamese and ASCII uppercase chars (valid first chars of a proper sentence)
_VN_UPPER_RE = re.compile(
    r"^[A-ZĐÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊ"
    r"ÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ\"\u201c\u00ab\u2018]",
    re.UNICODE,
)

# Digits are also valid sentence starters in legal text (e.g. "12.000.000 đồng")
_DIGIT_START_RE = re.compile(r"^\d")


def is_sentence_fragment(content: str) -> bool:
    """
    Return True if *content* looks like a mid-sentence fragment caused by
    an OCR/PDF page-break cut.

    A fragment is identified when the content:
      1. Does NOT start with an uppercase Vietnamese/ASCII letter, digit,
         or opening quote — i.e. it starts mid-word or mid-phrase with a
         lowercase letter.
      2. Is short enough that it cannot stand alone as a complete legal clause
         (< 50 characters heuristic; genuine short điểm items almost always
         start with an uppercase context word or a digit).

    Args:
        content: The raw ``content`` string of a chunk.

    Returns:
        True if the content is a suspected fragment, False otherwise.
    """
    s = content.strip()
    if not s:
        return False
    # Must be short — only fragment-merge very short strings
    if len(s) >= 50:
        return False
    # If it starts with uppercase/digit/quote → legitimate short content
    if _VN_UPPER_RE.match(s) or _DIGIT_START_RE.match(s):
        return False
    # Starts with lowercase → fragment
    return True


def merge_fragment_chunks(
    chunks: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Detect article-level chunks whose content is a mid-sentence fragment
    and merge them into the nearest preceding sibling within the **same article**.

    Merge rules:
      - Only article-level fragments (clause=None, point=None) are candidates.
        Clause- and point-level short content is considered valid legal enumeration.
      - Merge target must be the immediately preceding chunk with the **same article**.
      - Never merge across different articles.
      - After merge: target chunk's ``content`` gains the fragment text
        (joined with a space), ``end_page`` is updated if necessary, and
        ``content_with_context`` is rebuilt.
      - The fragment chunk is removed from the output and logged.

    Args:
        chunks: Input list of LegalChunk dicts (may be modified in-place).

    Returns:
        Tuple (cleaned_chunks, merged_into, removed_fragments) where:
          cleaned_chunks   — final list after merges
          merged_into      — list of chunks that received a merge
          removed_fragments — list of chunk dicts that were absorbed
    """
    result: list[dict] = []
    merged_into: list[dict] = []
    removed: list[dict] = []

    for chunk in chunks:
        # Only article-level chunks (no clause, no point) can be fragments
        is_article_level = (chunk.get("clause") is None
                            and chunk.get("point") is None)

        if is_article_level and is_sentence_fragment(chunk["content"]):
            # Find the most recent chunk in result with the same article and doc_number
            target_idx = None
            for i in range(len(result) - 1, -1, -1):
                if (result[i]["article"] == chunk["article"]
                        and result[i]["doc_number"] == chunk["doc_number"]):
                    target_idx = i
                    break

            if target_idx is not None:
                # Merge: append fragment content to target
                target = result[target_idx]
                old_content = target["content"]
                target["content"] = old_content.rstrip() + " " + chunk["content"].strip()
                # Update end_page if fragment is on a later page
                if (chunk.get("end_page") or 0) > (target.get("end_page") or 0):
                    target["end_page"] = chunk["end_page"]
                # Save merged_from_chunk_ids for traceability
                target.setdefault("merged_from_chunk_ids", []).append(chunk["chunk_id"])
                # Rebuild context string
                target["content_with_context"] = build_content_with_context(target)

                logger.info(
                    "MERGE fragment '%s' (chunk_id=%s) → '%s' (chunk_id=%s)",
                    chunk["content"][:40],
                    chunk["chunk_id"],
                    chunk["article"],
                    target["chunk_id"],
                )
                merged_into.append(target)
                removed.append(chunk)
                continue   # do not add fragment to result

            else:
                # No same-article predecessor found — keep as-is with warning
                logger.warning(
                    "Fragment chunk %s has no same-article predecessor; keeping as-is.",
                    chunk["chunk_id"],
                )

        result.append(chunk)

    return result, merged_into, removed


# ─── Quality Audit ────────────────────────────────────────────────────────────

def audit_chunks(chunks: list[dict]) -> dict:
    """
    Full quality audit of a chunk list.

    Returns a dict with:
      - length_distribution: counts by bucket
      - short_chunks: list of chunk dicts with content < 30 chars
      - fragment_candidates: subset of short chunks that are article-level fragments
      - duplicate_content_groups: count + sample of duplicate content strings
    """
    buckets: dict[str, int] = {
        "<10": 0, "10-29": 0, "30-49": 0, "50-99": 0, ">=100": 0,
    }
    short_chunks: list[dict] = []
    fragment_candidates: list[dict] = []

    for c in chunks:
        n = len(c["content"])
        if n < 10:
            buckets["<10"] += 1
        elif n < 30:
            buckets["10-29"] += 1
        elif n < 50:
            buckets["30-49"] += 1
        elif n < 100:
            buckets["50-99"] += 1
        else:
            buckets[">=100"] += 1

        if n < 30:
            short_chunks.append(c)
            if (c.get("clause") is None
                    and c.get("point") is None
                    and is_sentence_fragment(c["content"])):
                fragment_candidates.append(c)

    # Duplicate content
    content_map: dict[str, list[str]] = {}
    for c in chunks:
        key = c["content"].strip()
        content_map.setdefault(key, []).append(c["chunk_id"])
    dup_groups = {k: v for k, v in content_map.items() if len(v) > 1}

    return {
        "length_distribution":    buckets,
        "short_chunks":           short_chunks,
        "fragment_candidates":    fragment_candidates,
        "duplicate_content_count": len(dup_groups),
        "duplicate_content_sample": dict(list(dup_groups.items())[:3]),
    }


def print_audit(chunks: list[dict], label: str = "") -> None:
    """Print full quality audit report to stdout."""
    report = audit_chunks(chunks)
    sep = "─" * 64
    tag = f" [{label}]" if label else ""
    _print(f"\n{sep}")
    _print(f"QUALITY AUDIT{tag}")
    _print(sep)

    _print("  Content length distribution:")
    for bucket, count in report["length_distribution"].items():
        pct = count / len(chunks) * 100 if chunks else 0
        _print(f"    {bucket:8s}: {count:5d}  ({pct:.1f}%)")

    _print(f"\n  Short chunks (<30 chars)   : {len(report['short_chunks'])}")
    _print(f"  Fragment candidates        : {len(report['fragment_candidates'])}")
    _print(f"  Duplicate content groups   : {report['duplicate_content_count']}")

    if report["fragment_candidates"]:
        _print("\n  Fragment candidates (article-level, lowercase start):")
        for c in report["fragment_candidates"][:10]:
            _print(f"    [{len(c['content']):3d}] {c['chunk_id']:<40}  {c['content']!r}")


# ─── Output Helpers ───────────────────────────────────────────────────────────

def _write_output(chunks: list[dict], output_dir: Path, backup: bool = True) -> None:
    """Write legal_chunks.json and legal_chunks.jsonl, optionally backing up first."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path  = output_dir / "legal_chunks.json"
    jsonl_path = output_dir / "legal_chunks.jsonl"

    if backup:
        for p in [json_path, jsonl_path]:
            if p.exists():
                bak = p.with_suffix(p.suffix + ".bak")
                shutil.copy2(str(p), str(bak))
                _print(f"Backup: {bak.name}")

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    _print(f"Saved JSON  : {json_path}")

    with jsonl_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    _print(f"Saved JSONL : {jsonl_path}")


def chunk_all(parsed_dir: Path, output_dir: Path, backup: bool = True) -> list[dict]:
    """
    Process all parsed JSON files in *parsed_dir*, apply fragment merge,
    and write cleaned chunk output.

    Writes:
        output_dir/legal_chunks.json   — full JSON array (UTF-8)
        output_dir/legal_chunks.jsonl  — one chunk per line (UTF-8)

    Backs up existing output files before overwriting (when backup=True).

    Args:
        parsed_dir: Directory containing parsed JSON files.
        output_dir: Destination directory for chunk output.
        backup:     If True, back up existing output files before overwriting.

    Returns:
        Flat list of all cleaned LegalChunk dicts.
    """
    json_files = sorted(parsed_dir.glob("*.json"))
    if not json_files:
        logger.warning("No parsed JSON files found in %s", parsed_dir)
        return []

    all_chunks: list[dict] = []
    counter: dict = {}   # shared across all documents for global uniqueness

    for jf in json_files:
        _print(f"Chunking: {jf.name} …")
        with jf.open(encoding="utf-8") as f:
            parsed_doc = json.load(f)
        doc_chunks = chunk_document(parsed_doc, counter)
        all_chunks.extend(doc_chunks)
        _print(f"  → {len(doc_chunks)} chunks")

    # ── Audit before cleanup ────────────────────────────────────────────────
    n_before = len(all_chunks)
    print_audit(all_chunks, label="BEFORE cleanup")

    # ── Fragment merge ──────────────────────────────────────────────────────
    cleaned, merged_list, removed_list = merge_fragment_chunks(all_chunks)

    _print(f"\n  Merge summary:")
    _print(f"    Chunks before  : {n_before}")
    _print(f"    Fragments merged: {len(removed_list)}")
    _print(f"    Chunks after   : {len(cleaned)}")
    if removed_list:
        _print("  Absorbed fragments:")
        for r in removed_list:
            _print(f"    {r['chunk_id']}: {r['content']!r}")

    # ── Audit after cleanup ─────────────────────────────────────────────────
    print_audit(cleaned, label="AFTER cleanup")

    _write_output(cleaned, output_dir, backup=backup)

    return cleaned


# ─── Statistics & Sampling ────────────────────────────────────────────────────

def compute_stats(chunks: list[dict]) -> dict:
    """
    Compute quality statistics over the full chunk list.

    Returns a dict with per-document and corpus-level metrics.
    """
    from collections import defaultdict

    per_doc: dict[str, dict] = defaultdict(lambda: {
        "total": 0,
        "by_article": 0,
        "by_clause": 0,
        "by_point": 0,
        "char_lengths": [],
        "articles": set(),
    })

    chunk_ids_seen: set = set()
    duplicate_ids: list = []
    empty_chunks: list = []
    missing_doc_number: list = []
    missing_article: list = []
    amendment_chunks: list = []

    for c in chunks:
        cid     = c.get("chunk_id", "")
        dn      = c.get("doc_number", "")
        art     = c.get("article", "")
        kh      = c.get("clause")
        pt      = c.get("point")
        content = c.get("content", "")

        if cid in chunk_ids_seen:
            duplicate_ids.append(cid)
        chunk_ids_seen.add(cid)

        if not content.strip():
            empty_chunks.append(cid)
        if not dn:
            missing_doc_number.append(cid)
        if not art:
            missing_article.append(cid)
        if c.get("amends_document"):
            amendment_chunks.append(cid)

        d = per_doc[dn]
        d["total"] += 1
        d["char_lengths"].append(len(content))
        d["articles"].add(art)

        if not kh and not pt:
            d["by_article"] += 1
        elif kh and not pt:
            d["by_clause"] += 1
        elif pt:
            d["by_point"] += 1

    per_doc_final = {}
    for dn, d in per_doc.items():
        lengths = d["char_lengths"] or [0]
        per_doc_final[dn] = {
            "articles":      len(d["articles"]),
            "total_chunks":  d["total"],
            "by_article":    d["by_article"],
            "by_clause":     d["by_clause"],
            "by_point":      d["by_point"],
            "avg_chars":     round(sum(lengths) / len(lengths), 1),
            "min_chars":     min(lengths),
            "max_chars":     max(lengths),
        }

    return {
        "total_chunks":         len(chunks),
        "duplicate_chunk_ids":  len(duplicate_ids),
        "empty_chunks":         len(empty_chunks),
        "missing_doc_number":   len(missing_doc_number),
        "missing_article":      len(missing_article),
        "amendment_chunks":     len(amendment_chunks),
        "per_document":         per_doc_final,
        "_duplicate_ids":       duplicate_ids,
        "_empty_ids":           empty_chunks,
    }


def print_stats(chunks: list[dict]) -> None:
    """Print human-readable statistics to stdout."""
    stats = compute_stats(chunks)
    sep = "─" * 64

    _print(f"\n{sep}")
    _print("CHUNKING STATISTICS")
    _print(sep)
    _print(f"  Total chunks        : {stats['total_chunks']}")
    _print(f"  Duplicate chunk_ids : {stats['duplicate_chunk_ids']}")
    _print(f"  Empty chunks        : {stats['empty_chunks']}")
    _print(f"  Missing doc_number  : {stats['missing_doc_number']}")
    _print(f"  Missing article     : {stats['missing_article']}")
    _print(f"  Amendment chunks    : {stats['amendment_chunks']}")

    _print(f"\n{'':4}{'Document':<25} {'Điều':>5} {'Total':>6} {'Art':>5} {'Kh':>5} {'Pt':>5} {'Avg':>6} {'Min':>5} {'Max':>5}")
    _print(f"{'':4}{'-'*24} {'-'*5} {'-'*6} {'-'*5} {'-'*5} {'-'*5} {'-'*6} {'-'*5} {'-'*5}")
    for dn, d in stats["per_document"].items():
        _print(
            f"{'':4}{dn:<25} {d['articles']:>5} {d['total_chunks']:>6} "
            f"{d['by_article']:>5} {d['by_clause']:>5} {d['by_point']:>5} "
            f"{d['avg_chars']:>6.0f} {d['min_chars']:>5} {d['max_chars']:>5}"
        )

    if stats["_duplicate_ids"]:
        _print(f"\n  ⚠ Duplicates: {stats['_duplicate_ids'][:5]}")
    if stats["_empty_ids"]:
        _print(f"  ⚠ Empty     : {stats['_empty_ids'][:5]}")


def print_sample_chunks(chunks: list[dict], n: int = 10) -> None:
    """
    Print n representative sample chunks covering all documents and levels.
    """
    sep = "─" * 64

    def _pick(predicate, exclude: set) -> Optional[dict]:
        for c in chunks:
            if c["chunk_id"] not in exclude and predicate(c):
                return c
        return None

    selected: list[dict] = []
    seen_ids: set = set()

    criteria = [
        ("Luật 36 — article level",
         lambda c: c["doc_number"] == "36/2024/QH15" and not c["clause"] and not c["point"]),
        ("Luật 36 — clause level",
         lambda c: c["doc_number"] == "36/2024/QH15" and c["clause"] and not c["point"]),
        ("Luật 36 — point level",
         lambda c: c["doc_number"] == "36/2024/QH15" and c["point"]),
        ("NĐ 168 — article level",
         lambda c: c["doc_number"] == "168/2024/NĐ-CP" and not c["clause"] and not c["point"]),
        ("NĐ 168 — clause level",
         lambda c: c["doc_number"] == "168/2024/NĐ-CP" and c["clause"] and not c["point"]),
        ("NĐ 168 — point level",
         lambda c: c["doc_number"] == "168/2024/NĐ-CP" and c["point"]),
        ("NĐ 238 — with amendment metadata",
         lambda c: c["doc_number"] == "238/2026/NĐ-CP" and c.get("amended_article")),
        ("NĐ 238 — with clause+point amendment",
         lambda c: (c["doc_number"] == "238/2026/NĐ-CP"
                    and c.get("amended_article")
                    and c.get("amended_clause")
                    and c.get("amended_point"))),
        ("NĐ 238 — clause level",
         lambda c: c["doc_number"] == "238/2026/NĐ-CP" and c["clause"] and not c["point"]),
        ("NĐ 238 — point level",
         lambda c: c["doc_number"] == "238/2026/NĐ-CP" and c["point"]),
    ]

    for label, pred in criteria:
        if len(selected) >= n:
            break
        c = _pick(pred, seen_ids)
        if c:
            selected.append((label, c))
            seen_ids.add(c["chunk_id"])

    _print(f"\n{sep}")
    _print(f"SAMPLE CHUNKS  (showing {len(selected)})")
    _print(sep)

    for label, c in selected:
        content_preview = c["content"][:180].replace("\n", " ")
        if len(c["content"]) > 180:
            content_preview += "…"
        _print(f"\n  [{label}]")
        _print(f"  chunk_id         : {c['chunk_id']}")
        _print(f"  doc_number       : {c['doc_number']}")
        _print(f"  issue_date       : {c['issue_date']}")
        _print(f"  legal_status     : {c['legal_status']}")
        _print(f"  article          : {c['article']}")
        _print(f"  clause           : {c['clause']}")
        _print(f"  point            : {c['point']}")
        _print(f"  start_page       : {c['start_page']}")
        _print(f"  end_page         : {c['end_page']}")
        _print(f"  amends_document  : {c['amends_document']}")
        _print(f"  amended_article  : {c['amended_article']}")
        _print(f"  amended_clause   : {c['amended_clause']}")
        _print(f"  amended_point    : {c['amended_point']}")
        _print(f"  content preview  : {content_preview!r}")


# ─── Utilities ────────────────────────────────────────────────────────────────

def _print(msg: str) -> None:
    """UTF-8-safe print for Windows consoles."""
    sys.stdout.buffer.write((msg + "\n").encode("utf-8", errors="replace"))
    sys.stdout.buffer.flush()


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s — %(message)s",
        datefmt="%H:%M:%S",
    )

    project_root = Path(__file__).resolve().parents[2]
    parsed_dir   = project_root / "data" / "processed" / "parsed"
    output_dir   = project_root / "data" / "chunks"

    if not parsed_dir.exists():
        _print(f"ERROR: {parsed_dir} not found")
        sys.exit(1)

    _print(f"Input : {parsed_dir}")
    _print(f"Output: {output_dir}\n")

    chunks = chunk_all(parsed_dir, output_dir, backup=True)

    print_stats(chunks)
    print_sample_chunks(chunks, n=10)

    sep = "─" * 64
    _print(f"\n{sep}")
    _print(f"DONE — {len(chunks)} chunks total")
    _print(sep)
