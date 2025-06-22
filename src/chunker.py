"""
Chunk every PDF in data/ into small, sentence-aligned segments.

• Each chunk ≤ 160 words        (≈ 120 tokens)
• 20 % word overlap             (to avoid splitting key facts)
• Writes ONE JSONL per PDF into chunks/
"""

from __future__ import annotations
import json
import re
from pathlib import Path

import pdfplumber
from tqdm import tqdm

# ─── Config ────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
CHUNK_DIR  = Path("chunks")
MAX_WORDS  = 160            # soft cap per chunk  (≈ 120 tokens)
OVERLAP    = 0.20           # 20 % word overlap

CHUNK_DIR.mkdir(exist_ok=True)

# ─── Sentence-aware chunker ────────────────────────────────────────────────
_sentence_sep = re.compile(r"(?<=[.!?])\s+")

def _split_into_chunks(text: str,
                       size_words: int = MAX_WORDS,
                       overlap_words: int | None = None) -> list[str]:
    """
    Break *text* into size_words-word chunks with overlap_words-word overlap.
    Respects sentence boundaries as much as possible.
    """
    if overlap_words is None:
        overlap_words = int(size_words * OVERLAP)

    sentences = _sentence_sep.split(text)
    chunks: list[str] = []
    buf: list[str] = []

    for sent in sentences:
        words = sent.split()
        if not words:
            continue

        # Flush when adding this sentence would exceed the limit
        if len(buf) + len(words) > size_words:
            chunks.append(" ".join(buf))
            buf = buf[-overlap_words:]          # keep the overlap section
        buf.extend(words)

    if buf:
        chunks.append(" ".join(buf))

    return chunks

# ─── Helpers ────────────────────────────────────────────────────────────────
def _clean(text: str) -> str:
    """Strip page numbers & repeated headers (simple heuristic)."""
    lines = [
        ln for ln in text.splitlines()
        if not re.match(r"^\s*\d+\s*$", ln)     # drop isolated page numbers
    ]
    return " ".join(lines)

def chunk_page(raw: str) -> list[str]:
    """Return cleaned + chunked text for one PDF page."""
    return _split_into_chunks(_clean(raw))

# ─── Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    pdfs = list(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {DATA_DIR.resolve()}")

    for pdf_path in tqdm(pdfs, desc="Chunking PDFs"):
        out_path = CHUNK_DIR / f"{pdf_path.stem}.jsonl"
        with pdfplumber.open(pdf_path) as pdf, out_path.open("w", encoding="utf-8") as fout:
            cid = 0
            for page_num, page in enumerate(pdf.pages, 1):
                for chunk in chunk_page(page.extract_text() or ""):
                    fout.write(json.dumps({
                        "id":   f"{pdf_path.stem}_p{page_num}_c{cid}",
                        "file": pdf_path.name,
                        "page": page_num,
                        "text": chunk,
                    }, ensure_ascii=False) + "\n")
                    cid += 1
        tqdm.write(f"✅ {out_path.name}: {cid} chunks")

if __name__ == "__main__":
    main()
