# AI Assistance Disclosure Log

In accordance with academic journal and conference disclosure policies, this document logs all instances of AI generation for code, experimental workflows, and documentation in this repository.

## Log of AI Use

### 2026-10-11: Repository Archival & Initialization (Phase 0)
- **Role:** Antigravity Coding Agent (Gemini 3.8 Flash).
- **Assistance provided:**
  - Archival reorganization of previous link prediction study into `archive/novagap_eval/` and legacy app into `archive/old_novagap_app/`.
  - Integrity verification (SHA-256 computation and cross-verification of data caches).
  - Creation of archive and repository README files (`README.md`, `archive/novagap_eval/README.md`, `archive/old_novagap_app/README.md`).
- **Human oversight:** User reviewed inventory dry run, approved restructuring with explicit constraints, and ordered autonomous execution.

### 2026-10-11: Step 1 Topics & Skeleton (v2-lite)
- **Role:** Antigravity Coding Agent (Gemini 3.8 Flash).
- **Assistance provided:**
  - Created `v2-lite` branch.
  - Implemented `litsearch` package configuration, schemas, and strict temporal leakage guards (`litsearch/config.py`, `litsearch/leakage.py`).
  - Unit tests for leakage protection (`tests/test_leakage.py`).
  - Polled OpenAlex API to retrieve candidate Computer Science topics with 2023–2024 work counts (`has_abstract:true`).
- **Human oversight:** User mandated 3-gate structure and selected 4 topics at Gate 1.

### 2026-10-11: Full-Text RAG Chatbot Feature (`chatbot-fulltext`)
- **Role:** Antigravity Coding Agent.
- **Assistance provided:**
  - Branched `chatbot-fulltext` from `chatbot`.
  - Implemented legal open-access discovery (`OpenAlex`, `arXiv`, `Semantic Scholar`, and user upload fallback) in `app/sources.py`.
  - Implemented SSRF security guard, HTTPS enforcement, download size caps, and untrusted text sanitization in `app/security.py`.
  - Implemented PyMuPDF text extraction, repeated header/footer removal, section heading detection, and section-aware chunking in `app/ingestion.py`.
  - Implemented hybrid session indexing with BM25 (`rank_bm25`) and dense embeddings (`all-MiniLM-L6-v2`) via Reciprocal Rank Fusion ($k=60$) in `app/retrieval.py`.
  - Implemented Map-Reduce paper summarizer over section chunks in `app/summarizer.py`.
  - Implemented server-enforced citation validation (`[P# p.N]`), verbatim direct quote verification ($\le 25$ words), and claim sanitization in `app/grounding.py`.
  - Implemented Groq API tool use loop, SSE token streaming, per-session cost tracking ($1.00 cap), and rate limiting in `app/llm.py` and `app/service.py`.
  - Developed FastAPI web server (`app/server.py`) and single-page vanilla JS UI at `/chat` (`app/static/index.html`) featuring interactive citation chips, slide-out passage drawer, and chat export.
  - Built comprehensive test suite (`tests/test_fulltext.py`) passing 16/16 tests with 0 ruff errors.
  - Conducted end-to-end evaluation across 3 sample ideas (64.4% full-text coverage, 8.95s mean ingestion time, 0 extraction failures) and prepared 15 manual spot-check items in `tests/spotcheck.md`.
- **Human oversight:** User specified functional constraints, Groq API correction, and strict preservation of benchmark study code and preregistration files.

### 2026-10-12: Full-Text Reader Enhancement & Cost Removal (`chatbot-fulltext`)
- **Role:** Antigravity Coding Agent.
- **Assistance provided:**
  - Removed cost meter, cost caps, and cost reporting from UI and server endpoints.
  - Resolved publisher HTTP 403 / Cloudflare download blocks by enhancing multi-source legal OA candidate URL extraction with PMC/Europe PMC and open repository prioritization.
  - Implemented automatic abstract indexing fallback so all retrieved papers remain queryable and grounded in chat even when full text downloads are blocked by publisher paywalls.
  - Built an in-app Paper Reader modal with page-by-page extracted text, embedded PDF viewer, rich abstract view, and 1-click external browser link to original papers.
  - Upgraded PDF upload handler to auto-refresh session papers and immediately launch the reader view upon upload.
- **Human oversight:** User requested removal of cost display, reported 403 download error and upload flow friction, and verified resolved reading experience.

### 2026-10-12: Groq 429 Rate Limit Handling & Upload UX Streamlining (`chatbot-fulltext`)
- **Role:** Antigravity Coding Agent.
- **Assistance provided:**
  - Diagnosed Groq free tier 8,000 TPM limit root cause (two consecutive calls per turn with 1000 max_tokens requested 2,536 tokens against 5,554 used).
  - Reduced `LLM_MAX_TOKENS` from 1000 to 600 and tool selection calls to 250 tokens to lower token reservation footprint.
  - Implemented automatic retry with exponential backoff on 429 HTTP status (`_call_groq_with_retry`) matching Groq's ~0.67s sliding window reset.
  - Added automatic fallback to `qwen/qwen3.8-27b` if primary model is rate limited.
  - Streamlined UI by removing duplicate "Upload" buttons on individual paper cards, retaining a single clear global upload button in header with contextual upload in the reader modal.
- **Human oversight:** User reported 429 rate limit error during chat and asked why two upload file options were present.
