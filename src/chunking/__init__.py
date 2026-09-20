"""
src/chunking — Legal document chunking package.

Exports:
    chunk_all       — process all parsed JSON files → legal chunks
    chunk_document  — process one parsed document dict
    chunk_unit      — convert one LegalUnit dict to one LegalChunk dict
"""

from src.chunking.legal_chunker import chunk_all, chunk_document, chunk_unit

__all__ = ["chunk_all", "chunk_document", "chunk_unit"]
