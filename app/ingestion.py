"""PDF text extraction, section detection, cleaning, and chunking with PyMuPDF."""
import json
import re
from pathlib import Path
from typing import Any

import pymupdf

from .config import FULLTEXT_CACHE_DIR, MAX_PDF_PAGES
from .security import SecurityValidationError, sanitize_extracted_text

# Standard academic section heading keywords
SECTION_KEYWORDS = [
    "abstract",
    "introduction",
    "background",
    "related work",
    "prior art",
    "preliminaries",
    "problem formulation",
    "methodology",
    "method",
    "methods",
    "model",
    "proposed method",
    "system architecture",
    "approach",
    "implementation",
    "experiments",
    "experimental setup",
    "experimental results",
    "results",
    "evaluation",
    "discussion",
    "analysis",
    "ablation study",
    "case study",
    "limitations",
    "threats to validity",
    "conclusion",
    "conclusions",
    "future work",
    "ethical considerations",
    "broader impact",
    "acknowledgments",
    "references",
    "bibliography",
    "appendix",
]


def clean_paper_id_for_filesystem(paper_id: str) -> str:
    """Sanitize OpenAlex ID (e.g. https://openalex.org/W123 or W123) for safe folder names."""
    clean = re.sub(r"^https?://openalex\.org/", "", paper_id)
    return re.sub(r"[^A-Za-z0-9_\-]", "_", clean)


def is_heading_candidate(line: str) -> str | None:
    """
    Check if a line looks like an academic section heading.
    Returns normalized section title if detected, else None.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return None

    # Strip numbering like "1.", "1.1", "IV.", "Section 3"
    cleaned = re.sub(r"^(?:(?:section|sec\.)\s*)?(?:[0-9]+(?:\.[0-9]+)*|[IVXLCDM]+)[\.\:\s\-]+", "", stripped, flags=re.IGNORECASE).strip()
    cleaned_lower = cleaned.lower()

    for kw in SECTION_KEYWORDS:
        if cleaned_lower == kw or cleaned_lower.startswith(f"{kw} ") or cleaned_lower.startswith(f"{kw}:"):
            return kw.title()

    # Short title-cased or uppercase line without period at end
    if len(stripped.split()) <= 6 and not stripped.endswith(".") and not stripped.endswith(","):
        if stripped.isupper() or stripped.istitle():
            if len(stripped) >= 3:
                return stripped.title()

    return None


def extract_pdf_pages(pdf_bytes: bytes) -> tuple[list[dict[str, Any]], bool]:
    """
    Extract text page by page with PyMuPDF, strip headers/footers, detect sections.
    Returns (pages_data, has_text_layer).
    """
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise SecurityValidationError(f"Invalid or corrupted PDF file: {e}")

    page_count = doc.page_count
    if page_count > MAX_PDF_PAGES:
        doc.close()
        raise SecurityValidationError(f"PDF exceeds page limit ({page_count} > {MAX_PDF_PAGES} pages)")

    # First pass: collect top/bottom margin text for repeated header/footer detection
    top_margin_texts: dict[str, int] = {}
    bottom_margin_texts: dict[str, int] = {}
    raw_pages = []

    for page_idx in range(page_count):
        page = doc[page_idx]
        height = page.rect.height
        blocks = page.get_text("blocks")
        # Sort blocks by vertical position y0
        blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
        raw_pages.append((page_idx + 1, height, blocks))

        for b in blocks:
            if b[6] == 0:  # Text block
                text = b[4].strip()
                if not text:
                    continue
                y0, y1 = b[1], b[3]
                norm = re.sub(r"\d+", "#", text.lower())
                if y0 < 55:  # Top margin
                    top_margin_texts[norm] = top_margin_texts.get(norm, 0) + 1
                elif y1 > height - 55:  # Bottom margin
                    bottom_margin_texts[norm] = bottom_margin_texts.get(norm, 0) + 1

    doc.close()

    threshold = max(2, int(page_count * 0.4))
    repeated_headers = {k for k, v in top_margin_texts.items() if v >= threshold}
    repeated_footers = {k for k, v in bottom_margin_texts.items() if v >= threshold}

    pages_data = []
    total_text_length = 0
    current_section = "Introduction"
    is_references_mode = False

    for page_num, height, blocks in raw_pages:
        page_lines = []
        page_sections_detected = []

        for b in blocks:
            if b[6] != 0:
                continue
            block_text = sanitize_extracted_text(b[4])
            if not block_text:
                continue
            y0, y1 = b[1], b[3]
            norm = re.sub(r"\d+", "#", block_text.lower())

            # Skip repeated headers or footers
            if (y0 < 55 and norm in repeated_headers) or (y1 > height - 55 and norm in repeated_footers):
                continue

            # Skip standalone page numbers at very top or bottom
            if (y0 < 55 or y1 > height - 55) and block_text.strip().isdigit():
                continue

            # Check lines inside block for headings
            lines = block_text.splitlines()
            for line in lines:
                line_str = line.strip()
                if not line_str:
                    continue
                heading = is_heading_candidate(line_str)
                if heading:
                    current_section = heading
                    if heading.lower() in ("references", "bibliography"):
                        is_references_mode = True
                    page_sections_detected.append(heading)
                page_lines.append({
                    "text": line_str,
                    "section": current_section,
                    "is_references": is_references_mode,
                })
                total_text_length += len(line_str)

        pages_data.append({
            "page_num": page_num,
            "lines": page_lines,
            "sections": list(dict.fromkeys(page_sections_detected)),
        })

    has_text_layer = total_text_length >= 100
    return pages_data, has_text_layer


def chunk_extracted_pages(
    pages_data: list[dict[str, Any]],
    paper_label: str,
    paper_id: str,
    target_tokens: int = 700,
    overlap_tokens: int = 100,
) -> list[dict[str, Any]]:
    """
    Chunk paper text by section where headings were found, else ~700-token windows with 100-token overlap.
    Metadata per chunk: paper label, OpenAlex id, page start/end, section guess, chunk index.
    Tag references section and exclude from retrieval.
    """
    # 1 token roughly 0.75 words -> 700 tokens ~ 525 words, 100 tokens ~ 75 words
    target_words = int(target_tokens * 0.75)
    overlap_words = int(overlap_tokens * 0.75)

    # Flatten lines with page information
    all_elements = []
    for p in pages_data:
        p_num = p["page_num"]
        for line_item in p["lines"]:
            all_elements.append({
                "page": p_num,
                "text": line_item["text"],
                "section": line_item["section"],
                "is_references": line_item["is_references"],
            })

    if not all_elements:
        return []

    # Group elements into coherent sections
    sections: list[dict[str, Any]] = []
    curr_sec_name = all_elements[0]["section"]
    curr_sec_items = []
    curr_is_ref = all_elements[0]["is_references"]

    for el in all_elements:
        if el["section"] != curr_sec_name:
            if curr_sec_items:
                sections.append({
                    "name": curr_sec_name,
                    "items": curr_sec_items,
                    "is_references": curr_is_ref,
                })
            curr_sec_name = el["section"]
            curr_sec_items = [el]
            curr_is_ref = el["is_references"]
        else:
            curr_sec_items.append(el)

    if curr_sec_items:
        sections.append({
            "name": curr_sec_name,
            "items": curr_sec_items,
            "is_references": curr_is_ref,
        })

    chunks: list[dict[str, Any]] = []
    chunk_idx = 0

    for sec in sections:
        sec_name = sec["name"]
        sec_items = sec["items"]
        is_references = sec["is_references"]

        # Aggregate words with page mapping
        words_with_pages: list[tuple[str, int]] = []
        for item in sec_items:
            for w in item["text"].split():
                words_with_pages.append((w, item["page"]))

        if not words_with_pages:
            continue

        total_words = len(words_with_pages)
        if total_words <= target_words + overlap_words:
            # Single chunk for this section
            chunk_text = " ".join(w for w, _ in words_with_pages)
            page_start = words_with_pages[0][1]
            page_end = words_with_pages[-1][1]
            chunks.append({
                "chunk_id": f"{paper_label}_c{chunk_idx}",
                "paper_label": paper_label,
                "paper_id": paper_id,
                "page_start": page_start,
                "page_end": page_end,
                "section_guess": sec_name,
                "chunk_index": chunk_idx,
                "text": chunk_text,
                "is_references": is_references,
            })
            chunk_idx += 1
        else:
            # Windowing by ~700 tokens (~525 words) with ~100 tokens (~75 words) overlap
            step = max(1, target_words - overlap_words)
            for start in range(0, total_words, step):
                end = min(total_words, start + target_words)
                slice_words = words_with_pages[start:end]
                if not slice_words:
                    continue
                chunk_text = " ".join(w for w, _ in slice_words)
                page_start = slice_words[0][1]
                page_end = slice_words[-1][1]
                chunks.append({
                    "chunk_id": f"{paper_label}_c{chunk_idx}",
                    "paper_label": paper_label,
                    "paper_id": paper_id,
                    "page_start": page_start,
                    "page_end": page_end,
                    "section_guess": sec_name,
                    "chunk_index": chunk_idx,
                    "text": chunk_text,
                    "is_references": is_references,
                })
                chunk_idx += 1
                if end == total_words:
                    break

    return chunks


def get_cache_paths(paper_id: str) -> tuple[Path, Path, Path, Path]:
    """Return paths to paper's cache directory, raw.pdf, extracted.json, chunks.json."""
    clean_id = clean_paper_id_for_filesystem(paper_id)
    paper_dir = FULLTEXT_CACHE_DIR / clean_id
    paper_dir.mkdir(parents=True, exist_ok=True)
    return (
        paper_dir,
        paper_dir / "raw.pdf",
        paper_dir / "extracted.json",
        paper_dir / "chunks.json",
    )


def save_ingestion_cache(
    paper_id: str,
    pdf_bytes: bytes,
    pages_data: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    """Cache raw PDF, extracted text, and chunks by OpenAlex ID in data/fulltext/."""
    _, pdf_path, ext_path, chk_path = get_cache_paths(paper_id)
    if pdf_bytes:
        pdf_path.write_bytes(pdf_bytes)
    ext_payload = {
        "metadata": metadata,
        "pages": pages_data,
    }
    ext_path.write_text(json.dumps(ext_payload, indent=2), encoding="utf-8")
    chk_path.write_text(json.dumps(chunks, indent=2), encoding="utf-8")


def load_ingestion_cache(paper_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    """Load cached extraction and chunks if available."""
    _, _, ext_path, chk_path = get_cache_paths(paper_id)
    if ext_path.exists() and chk_path.exists():
        try:
            ext_data = json.loads(ext_path.read_text(encoding="utf-8"))
            chunks = json.loads(chk_path.read_text(encoding="utf-8"))
            return ext_data, chunks
        except Exception:
            return None
    return None
