"""
Legal Parser for Vietnamese legal documents.

Parses raw text (output of pdf_loader) into flat structured legal unit records.

Hierarchy: Văn bản → Chương → Mục → Điều → Khoản → Điểm

Special handling:
- 36-2024-qh15.pdf + 36-2024-qh15_tiep.pdf merged as one logical document
- Cross-page continuity: article/clause spanning multiple pages is preserved
- False-positive suppression for:
    * Monetary amounts  (12.000.000 đồng)
    * Years             (2024., 2026.)
    * Page numbers      (76, 77 … from Công Báo)
    * Document references (168/2024/NĐ-CP)
    * Quoted amendments ("i) Buộc lắp đặt…)
"""

from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ─── Document Metadata ────────────────────────────────────────────────────────

#: Mapping: PDF filename stem → document metadata.
DOC_META: dict[str, dict] = {
    "36-2024-qh15": {
        "doc_number": "36/2024/QH15",
        "doc_title":  "Luật Trật tự, an toàn giao thông đường bộ",
        "group":      "36_2024_QH15",
        "order":      0,
    },
    "36-2024-qh15_tiep": {
        "doc_number": "36/2024/QH15",
        "doc_title":  "Luật Trật tự, an toàn giao thông đường bộ",
        "group":      "36_2024_QH15",
        "order":      1,
    },
    "168_2024_ND-CP_619502": {
        "doc_number": "168/2024/NĐ-CP",
        "doc_title":  (
            "Nghị định xử phạt vi phạm hành chính về trật tự, "
            "an toàn giao thông đường bộ; "
            "trừ điểm, phục hồi điểm giấy phép lái xe"
        ),
        "group":      "168_2024_ND-CP",
        "order":      0,
    },
    "238_2026_ND-CP_712521": {
        "doc_number": "238/2026/NĐ-CP",
        "doc_title":  (
            "Nghị định sửa đổi, bổ sung một số điều của "
            "Nghị định số 168/2024/NĐ-CP"
        ),
        "group":      "238_2026_ND-CP",
        "order":      0,
    },
}

#: Maps group key → output filename in parsed/ directory.
_GROUP_OUTPUT: dict[str, str] = {
    "36_2024_QH15":  "36_2024_QH15.json",
    "168_2024_ND-CP": "168_2024_ND-CP.json",
    "238_2026_ND-CP": "238_2026_ND-CP.json",
}

# ─── Regex Patterns ───────────────────────────────────────────────────────────

# Chương: "Chương I", "Chương II", "Chương 1"
_CHAP_RE = re.compile(
    r"^(Chương\s+(?:[IVXLCDMivxlcdm]+|\d+))\s*(.*)",
    re.UNICODE,
)

# Mục: "Mục 1", "Mục 2"
_SECT_RE = re.compile(r"^(Mục\s+\d+)\s*(.*)", re.UNICODE)

# Điều: "Điều 1.", "Điều 25a."
_ART_RE = re.compile(r"^(Điều\s+\d+[a-zđ]?)\.\s*(.*)", re.UNICODE)

# Khoản: "1.", "2.", "1a.", "9a." — at start of line
# Requires at least one whitespace after the period, then any non-whitespace.
# group(1) = number label, group(2) = first non-whitespace char after "N. "
_CLAUSE_RE = re.compile(r"^(\d{1,2}[a-zđ]?)\.\s+(\S)", re.UNICODE)

# Điểm: "a)", "b)", "đ)" — at start of line
# group(1) = letter, group(2) = first non-whitespace after "X) "
_POINT_RE = re.compile(r"^([a-zđ])\)\s+(\S)", re.UNICODE)

# Vietnamese uppercase characters (Khoản content must start with one of these or a quote)
_VN_UPPER: frozenset[str] = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "ĐÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆ"
    "ÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢ"
    "ÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ"
)

# Opening-quote characters that precede quoted amendment text (not structural điểm)
_OPEN_QUOTES: frozenset[str] = frozenset('"\u201c\u201e\u00ab\u2018\'')

# ─── Token Kinds ──────────────────────────────────────────────────────────────

K_CHAPTER = "CHAPTER"
K_SECTION = "SECTION"
K_ARTICLE = "ARTICLE"
K_CLAUSE  = "CLAUSE"
K_POINT   = "POINT"
K_TEXT    = "TEXT"
K_EMPTY   = "EMPTY"


# ─── LineToken ────────────────────────────────────────────────────────────────

@dataclass
class LineToken:
    """Result of classifying a single text line."""

    kind: str
    label: str = ""    # Structural label: "Chương I", "Điều 3", "1", "a)"
    extra: str = ""    # Title text or inline content on the same line
    raw: str = ""
    page: int = 0
    source_file: str = ""


def classify_line(
    line: str,
    page: int = 0,
    source_file: str = "",
) -> LineToken:
    """
    Classify a single line of Vietnamese legal text into a LineToken.

    Priority order: EMPTY → CHAPTER → SECTION → ARTICLE → CLAUSE → POINT → TEXT.

    False-positive prevention:
      - Khoản numbers 1900–2099 are treated as years and rejected.
      - Khoản content must begin with uppercase Vietnamese/ASCII or a quote.
      - Điểm lines that start with a quote character are rejected (quoted amendments).
      - Pure-digit lines (page numbers from Công Báo) fall through to TEXT.
    """
    s = line.strip()
    kw = dict(raw=s, page=page, source_file=source_file)

    if not s:
        return LineToken(K_EMPTY, **kw)

    # ── Chương ────────────────────────────────────────────────────────────────
    m = _CHAP_RE.match(s)
    if m:
        return LineToken(K_CHAPTER, label=m.group(1), extra=m.group(2).strip(), **kw)

    # ── Mục ──────────────────────────────────────────────────────────────────
    m = _SECT_RE.match(s)
    if m:
        return LineToken(K_SECTION, label=m.group(1), extra=m.group(2).strip(), **kw)

    # ── Điều ─────────────────────────────────────────────────────────────────
    m = _ART_RE.match(s)
    if m:
        return LineToken(K_ARTICLE, label=m.group(1), extra=m.group(2).strip(), **kw)

    # ── Khoản ────────────────────────────────────────────────────────────────
    m = _CLAUSE_RE.match(s)
    if m:
        # Extract numeric portion for year-check (strip any letter suffix)
        digits = re.sub(r"[a-zđ]", "", m.group(1))
        num = int(digits)
        first_char = m.group(2)          # first char of actual content
        is_year = 1900 <= num <= 2099    # suppress 2024., 2026., etc.
        good_start = (
            first_char in _VN_UPPER
            or first_char in _OPEN_QUOTES
            or first_char in "–-"
        )
        if not is_year and good_start:
            # rest = everything after "N. " (or "Na. ")
            prefix_len = len(m.group(1)) + 2   # label + ". "
            rest = s[prefix_len:].strip()
            return LineToken(K_CLAUSE, label=m.group(1), extra=rest, **kw)

    # ── Điểm ─────────────────────────────────────────────────────────────────
    # Reject lines that begin with a quote (those are quoted amendment text)
    if s[0] not in _OPEN_QUOTES:
        m = _POINT_RE.match(s)
        if m:
            rest = s[3:].strip()         # skip "a) "
            return LineToken(K_POINT, label=m.group(1) + ")", extra=rest, **kw)

    # ── TEXT / fallthrough ────────────────────────────────────────────────────
    return LineToken(K_TEXT, extra=s, **kw)


def _is_caps_title(s: str) -> bool:
    """
    Return True if s looks like an ALL-CAPS legal heading.

    Used to collect multi-line Chương/Mục titles that appear on lines
    following the Chương/Mục marker (e.g., 'NHỮNG QUY ĐỊNH CHUNG').
    """
    s = s.strip()
    return (
        len(s) >= 4
        and s == s.upper()
        and any(c.isalpha() for c in s)
    )


# ─── LegalUnit ───────────────────────────────────────────────────────────────

@dataclass
class LegalUnit:
    """A single parsed legal record (leaf-level unit)."""

    doc_title:     str
    doc_number:    str
    chapter:       str            # e.g. "Chương I"
    chapter_title: str            # e.g. "NHỮNG QUY ĐỊNH CHUNG"
    section:       str            # e.g. "Mục 1" or ""
    section_title: str
    article:       str            # e.g. "Điều 25"
    article_title: str
    clause:        Optional[str]  # e.g. "1", "1a", or None
    point:         Optional[str]  # e.g. "a)", "đ)", or None
    content:       str
    source_file:   str
    start_page:    int
    end_page:      int


# ─── Parser State ─────────────────────────────────────────────────────────────

_Buf = list   # list of (text: str, page: int, source_file: str)


@dataclass
class _State:
    doc_title: str
    doc_number: str

    chapter:       str = ""
    chapter_title: str = ""
    section:       str = ""
    section_title: str = ""
    article:       str = ""
    article_title: str = ""
    clause:        str = ""
    point:         str = ""

    article_buf: _Buf = field(default_factory=list)
    clause_buf:  _Buf = field(default_factory=list)
    point_buf:   _Buf = field(default_factory=list)

    _await_caps: str = ""    # "chapter" | "section" | "" — collect multi-line title

    records:      list = field(default_factory=list)
    unparsed:     int  = 0   # text lines before any article (preamble/headers)


# ─── Emit Helpers ─────────────────────────────────────────────────────────────

def _make_unit(
    st: _State,
    buf: _Buf,
    clause: Optional[str],
    point: Optional[str],
) -> Optional[LegalUnit]:
    """Build a LegalUnit from a content buffer; returns None if buffer is empty."""
    if not buf:
        return None
    content = " ".join(t for t, _, _ in buf).strip()
    if not content:
        return None
    return LegalUnit(
        doc_title=st.doc_title,
        doc_number=st.doc_number,
        chapter=st.chapter,
        chapter_title=st.chapter_title,
        section=st.section,
        section_title=st.section_title,
        article=st.article,
        article_title=st.article_title,
        clause=clause,
        point=point,
        content=content,
        source_file=buf[0][2],
        start_page=buf[0][1],
        end_page=buf[-1][1],
    )


def _emit(st: _State, buf: _Buf, clause: Optional[str], point: Optional[str]) -> None:
    unit = _make_unit(st, buf, clause, point)
    if unit:
        st.records.append(asdict(unit))


def _flush_point(st: _State) -> None:
    """Emit the current điểm record and reset point state."""
    _emit(st, st.point_buf, clause=st.clause or None, point=st.point or None)
    st.point = ""
    st.point_buf = []


def _flush_clause_intro(st: _State) -> None:
    """Emit pending clause intro (content before first điểm) and clear clause_buf."""
    _emit(st, st.clause_buf, clause=st.clause or None, point=None)
    st.clause_buf = []


def _flush_clause(st: _State) -> None:
    """Flush điểm then clause intro, reset clause state."""
    _flush_point(st)
    _flush_clause_intro(st)
    st.clause = ""


def _flush_article(st: _State) -> None:
    """Flush all pending clause/point content, then article-level text."""
    _flush_clause(st)
    _emit(st, st.article_buf, clause=None, point=None)
    st.article = ""
    st.article_buf = []


def _add_text(st: _State, text: str, page: int, src: str) -> None:
    """Route a plain text line to the deepest active buffer."""
    entry = (text, page, src)
    if st.point:
        st.point_buf.append(entry)
    elif st.clause:
        st.clause_buf.append(entry)
    elif st.article:
        st.article_buf.append(entry)
    else:
        st.unparsed += 1


# ─── Token Processor ──────────────────────────────────────────────────────────

def _on_token(st: _State, tok: LineToken) -> None:
    """Apply one LineToken to the parse state."""

    # Accumulate multi-line ALL-CAPS chapter/section titles
    if st._await_caps:
        if tok.kind == K_TEXT and _is_caps_title(tok.extra):
            if st._await_caps == "chapter":
                st.chapter_title = (st.chapter_title + " " + tok.extra).strip()
            elif st._await_caps == "section":
                st.section_title = (st.section_title + " " + tok.extra).strip()
            return          # consumed as title continuation
        elif tok.kind != K_EMPTY:
            st._await_caps = ""   # non-empty non-CAPS line ends title collection

    if tok.kind == K_EMPTY:
        return

    # ── Chương ────────────────────────────────────────────────────────────────
    if tok.kind == K_CHAPTER:
        _flush_article(st)
        st.chapter = tok.label
        st.chapter_title = tok.extra         # may be "" if title is on next line
        st.section = st.section_title = ""
        if not tok.extra:
            st._await_caps = "chapter"

    # ── Mục ──────────────────────────────────────────────────────────────────
    elif tok.kind == K_SECTION:
        _flush_article(st)
        st.section = tok.label
        st.section_title = tok.extra
        if not tok.extra:
            st._await_caps = "section"

    # ── Điều ─────────────────────────────────────────────────────────────────
    elif tok.kind == K_ARTICLE:
        _flush_article(st)
        st.article = tok.label
        st.article_title = tok.extra

    # ── Khoản ────────────────────────────────────────────────────────────────
    elif tok.kind == K_CLAUSE:
        # Flush previous clause completely, then start fresh
        _flush_clause(st)
        st.clause = tok.label
        if tok.extra:
            st.clause_buf.append((tok.extra, tok.page, tok.source_file))

    # ── Điểm ─────────────────────────────────────────────────────────────────
    elif tok.kind == K_POINT:
        # Emit clause intro BEFORE the first điểm of this clause
        if st.clause and st.clause_buf:
            _flush_clause_intro(st)
        # Emit previous điểm (if any) and start new one
        _flush_point(st)
        st.point = tok.label
        if tok.extra:
            st.point_buf.append((tok.extra, tok.page, tok.source_file))

    # ── TEXT ─────────────────────────────────────────────────────────────────
    else:
        _add_text(st, tok.extra, tok.page, tok.source_file)


# ─── Public API ───────────────────────────────────────────────────────────────

def parse_pages(
    pages: list[dict],
    doc_title: str,
    doc_number: str,
) -> list[dict]:
    """
    Parse a list of page records into flat structured legal unit records.

    Args:
        pages:      List of page dicts with keys: ``text``, ``page``, ``source_file``.
        doc_title:  Full document title string.
        doc_number: Official document number (e.g. "36/2024/QH15").

    Returns:
        List of serialised :class:`LegalUnit` dicts.
    """
    st = _State(doc_title=doc_title, doc_number=doc_number)

    for pr in pages:
        pg   = pr["page"]
        src  = pr["source_file"]
        text = pr["text"]
        for line in text.split("\n"):
            _on_token(st, classify_line(line, pg, src))

    _flush_article(st)

    logger.info(
        "%s: %d records, %d unparsed lines",
        doc_number,
        len(st.records),
        st.unparsed,
    )
    return st.records


def load_raw_json(path: Path) -> dict:
    """Load a raw-text JSON file (output of pdf_loader)."""
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def parse_all(raw_text_dir: Path, parsed_dir: Path) -> dict[str, dict]:
    """
    Parse all documents in *raw_text_dir* and write JSON to *parsed_dir*.

    The two Luật 36 files are merged into one logical document before parsing.

    Args:
        raw_text_dir: Directory containing raw text JSON files.
        parsed_dir:   Destination directory for parsed output.

    Returns:
        Dict mapping group name → statistics dict.
    """
    # Group stems by document group, preserving part-order
    groups: dict[str, list[tuple[int, str, Path]]] = {}
    for jf in sorted(raw_text_dir.glob("*.json")):
        stem = jf.stem
        if stem not in DOC_META:
            logger.warning("Unknown stem '%s' — skipping", stem)
            continue
        meta = DOC_META[stem]
        groups.setdefault(meta["group"], []).append((meta["order"], stem, jf))

    parsed_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    for group, items in sorted(groups.items()):
        items.sort(key=lambda x: x[0])          # sort by "order"
        meta0      = DOC_META[items[0][1]]
        doc_title  = meta0["doc_title"]
        doc_number = meta0["doc_number"]

        # Merge pages from all files in group
        all_pages: list[dict] = []
        for _, _, jf in items:
            all_pages.extend(load_raw_json(jf)["pages"])

        _print(
            f"Parsing '{group}': "
            f"{len(items)} file(s), {len(all_pages)} pages …"
        )
        units = parse_pages(all_pages, doc_title, doc_number)

        # Compute statistics
        seen_chapters = {u["chapter"] for u in units if u.get("chapter")}
        seen_articles = {u["article"] for u in units if u.get("article")}
        n_clauses = sum(1 for u in units if u.get("clause") and not u.get("point"))
        n_points  = sum(1 for u in units if u.get("point"))

        stats = {
            "chapters":      len(seen_chapters),
            "articles":      len(seen_articles),
            "clause_records": n_clauses,
            "point_records":  n_points,
            "total_records":  len(units),
        }
        results[group] = stats

        out_file = parsed_dir / _GROUP_OUTPUT[group]
        payload = {
            "doc_number": doc_number,
            "doc_title":  doc_title,
            "stats":      stats,
            "units":      units,
        }
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        logger.info("→ %s (%d records)", out_file.name, len(units))

    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _print(msg: str) -> None:
    sys.stdout.buffer.write((msg + "\n").encode("utf-8", errors="replace"))
    sys.stdout.buffer.flush()


def _show_examples(parsed_dir: Path) -> None:
    """Print 3 sample records for Điều / Khoản / Điểm from each output file."""
    for jf in sorted(parsed_dir.glob("*.json")):
        data = json.loads(jf.read_text(encoding="utf-8"))
        units = data["units"]
        _print(f"\n{'─'*60}")
        _print(f"  {jf.name}  ({data['doc_number']})")
        _print(f"{'─'*60}")

        dieu_ex   = [u for u in units if u.get("article") and not u.get("clause")][:3]
        khoan_ex  = [u for u in units if u.get("clause") and not u.get("point")][:3]
        diem_ex   = [u for u in units if u.get("point")][:3]

        for label, examples in [("Điều", dieu_ex), ("Khoản", khoan_ex), ("Điểm", diem_ex)]:
            _print(f"\n  [{label} examples]")
            for ex in examples:
                art   = ex.get("article", "")
                kh    = ex.get("clause", "")
                pt    = ex.get("point", "")
                pg    = f"p{ex['start_page']}"
                if ex['start_page'] != ex['end_page']:
                    pg += f"–{ex['end_page']}"
                snip  = ex["content"][:120].replace("\n", " ")
                _print(f"    {art} {kh} {pt} [{pg}]: {snip!r}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s — %(message)s",
        datefmt="%H:%M:%S",
    )

    project_root = Path(__file__).resolve().parents[2]
    raw_dir  = project_root / "data" / "processed" / "raw_text"
    out_dir  = project_root / "data" / "processed" / "parsed"

    if not raw_dir.exists():
        _print(f"ERROR: {raw_dir} not found")
        sys.exit(1)

    _print(f"Input : {raw_dir}")
    _print(f"Output: {out_dir}\n")

    results = parse_all(raw_dir, out_dir)

    sep = "─" * 60
    _print(f"\n{sep}\nPARSING COMPLETE\n{sep}")
    for group, s in results.items():
        _print(
            f"\n  {group}\n"
            f"  Chương : {s['chapters']}\n"
            f"  Điều   : {s['articles']}\n"
            f"  Khoản  : {s['clause_records']} records\n"
            f"  Điểm   : {s['point_records']} records\n"
            f"  Total  : {s['total_records']} records"
        )

    _print(f"\n{sep}\nSAMPLE RECORDS\n{sep}")
    _show_examples(out_dir)
