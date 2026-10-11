# Study & Project Status (chatbot-fulltext)

- **Branch:** `chatbot-fulltext` (branched from `chatbot`)
- **Current Feature:** Full-Text RAG Literature Chatbot Demo (`/chat`)
- **Status:** Complete, tested, and validated.
- **Embedding Model Recorded:** `all-MiniLM-L6-v2` via `sentence-transformers` (selected for fast CPU inference: 0.02s per 10 sentences).
- **LLM Engine:** Groq API (`llama-3.3-70b-versatile` / `llama-3.1-8b-instant`) with tool use loop, SSE streaming, and $1.00 session cost cap.

## Full-Text RAG Implementation Checklist
- [x] Legal open-access PDF resolver (OpenAlex best OA -> open_access.oa_url -> other locations -> arXiv -> Semantic Scholar -> user upload).
- [x] Download security & SSRF guard (HTTPS only, loopback & RFC 1918 private IP blocked, 30 MB / 100 page caps, %PDF magic bytes verified).
- [x] PDF ingestion via PyMuPDF (page-by-page text, headers/footers stripped, section headings detected, references tagged & excluded, empty text layer scanned fallback).
- [x] Section-aware chunking (~700 tokens with 100 token overlap) with full metadata (paper label, OpenAlex ID, page range, section guess).
- [x] Session hybrid indexing: BM25 (`rank_bm25`) + Dense (`sentence-transformers` + `numpy`) with Reciprocal Rank Fusion ($k=60$).
- [x] Groq tool orchestration (`search_papers`, `get_paper`, `expand_citations`, `read_paper`, `search_paper_text`) with max 5 tool calls per turn.
- [x] Map-Reduce section summarizer for whole paper summaries without context overflow.
- [x] Server-enforced grounding rules (`[P# p.N]` citation validation against retrieved chunks, verbatim quote validation $\le 25$ words, absolute novelty claim sanitizer).
- [x] Vanilla JS UI at `/chat` (paper status badges, interactive `[P# p.N]` citation chips, slide-out passage drawer, per-paper PDF upload, and chat export).
- [x] Test suite: 17/17 pytest passing (`tests/test_fulltext.py`, `tests/test_leakage.py`, `tests/test_live_server_endpoints.py`), 0 ruff lint errors in `app/`.
- [x] Evaluation: 3 sample ideas evaluated end-to-end (64.4% OA coverage across top 15s, 8.95s mean ingestion time, 0 extraction failures).
- [x] Manual spot-check: 15 verification items created in `tests/spotcheck.md`.
- [x] Multi-source OA fallback & publisher 403 handling: checks arXiv, PubMed Central / Europe PMC, institutional repositories (Zenodo, HAL, .edu), OpenAlex, and Semantic Scholar; handles publisher Cloudflare/403 blocks gracefully.
- [x] In-app Full-Text & Abstract Reader: Modal with page-by-page extracted text, embedded PDF viewer, and rich abstract view with 1-click "Open PDF in Browser ↗".
- [x] Abstract auto-indexing: abstracts are indexed into session chunks so chat Q&A remains grounded and functional even if publisher blocks PDF downloads.
- [x] Removed cost meters and cost caps for frictionless user experience.
