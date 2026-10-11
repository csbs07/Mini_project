# Literature Search & Relevance Engine

A research benchmark and production-quality literature search assistant investigating whether citation reference lists reliably serve as ground truth for literature search on research ideas.

---

## 1. Research Study

**"Cited Is Not Relevant: Comparing Reference Lists and Human Relevance Judgments for Short-Idea Literature Search in Computer Science"**

### Overview
Automated literature search and novelty assessment systems frequently evaluate retrieval by treating a paper's bibliography (reference list) as the ground truth. This pre-registered study empirically examines the gap between reference lists and blind human relevance judgments for short-idea literature retrieval across Computer Science topics in OpenAlex.

### Retrieval Systems Evaluated
- **S1 (BM25 Lexical Baseline):** BM25Okapi using title and abstract text.
- **S2 (SPECTER2 Dense Retrieval):** Dense semantic search using `allenai/specter2_base` with task adapters (`adhoc_query` and proximity adapters).
- **S3 (Citation-Network Expansion):** Hybrid retrieval seeding top hits from S2 and pulling citation-graph neighbors published prior to the target's cutoff.
- **S4 (Facet Decomposition + Reciprocal Rank Fusion):** LLM-guided facet extraction decomposing queries into constituent research aspects (task, setting, data type, constraints) merged via RRF ($k=60$).

### Core Methodological Safeguards
- **Strict Leakage Guard:** All candidate works and citation paths are strictly filtered to works published *prior* to each target paper's publication cutoff. No future citations, post-cutoff counts, or post-dated metadata are permitted during experiment runs.
- **Blind Adjudicated Annotation:** Labeling pools combine top retrieved hits from S1–S4 and target reference papers, deduplicated and shuffled with sources masked, evaluated on a 3-point relevance rubric (2 = directly relevant, 1 = background, 0 = unrelated).

---

## 2. Literature-Search Assistant Web Application

A production-quality web application built on the shared `litsearch/` retrieval foundation to assist researchers in finding and organizing relevant literature for new ideas:

- **Reading List & Prior Art:** Highlights closest prior work and explains relevance grounded strictly in retrieved titles and abstracts.
- **Aspect & Combination Evidence:** Maps existing papers matching individual facets and pairs of facets to show where research combinations already exist.
- **Ethical Safeguards:** **No match percentages and no automated novelty scores.** The tool assists literature discovery and organization; it does not issue automated novelty verdicts.
- **Export Formats:** BibTeX, RIS, and CSV export with verified OpenAlex DOIs.

---

## 3. Repository Structure

```
├── litsearch/                     # Core library: BM25, SPECTER2, citation expansion, facet fusion
├── app/
│   ├── backend/                   # FastAPI backend (REST API, safety validators, rate limiter)
│   └── frontend/                  # React + Vite + TypeScript + Tailwind UI
├── data/
│   └── raw/                       # Cached OpenAlex snapshots
├── prompts/                       # LLM prompt templates
├── results/                       # Empirical run outputs, annotation pools, evaluation tables
├── tests/                         # Unit tests and synthetic evaluation tests
├── archive/
│   ├── novagap_eval/              # Finished Study #2 (Link prediction & degree-matched evaluation)
│   │   ├── results/               # Tables 1–6 and Protocol B CSVs
│   │   ├── sample_caches.zip      # Archived OpenAlex random samples (SHA-256 verified)
│   │   └── superseded/            # Superseded pilot scripts and biased caches
│   └── old_novagap_app/           # Archived legacy prototype app and scripts
├── AI_USE.md                      # Journal AI disclosure log
└── MANIFEST.md                    # Parent commit hash and data snapshot provenance
```

---

## 4. Prior Studies & Archival Notice

The previous evaluation study on degree-matched vs. natural candidate evaluation in link prediction has concluded. All code, multi-seed replication tables, and data caches are preserved under [`archive/novagap_eval/`](archive/novagap_eval/) and tagged at `novagap-eval-final` (as well as `v1.0`).

---

## 5. Full-Text RAG Literature Assistant (`/chat`)

A demo literature assistant enabling conversational exploration of the **full text** of academic papers (beyond abstracts) using retrieval-augmented generation (RAG) powered by Groq and PyMuPDF.

### Architecture & Pipeline
1. **Open-Access Resolution (`app/sources.py`):**
   - Automatically searches for legal, open-access PDFs in priority order:
     1. OpenAlex `best_oa_location.pdf_url`
     2. OpenAlex `open_access.oa_url`
     3. Other indexed OA locations
     4. arXiv PDF repository (`arxiv.org/pdf/{id}.pdf`)
     5. Semantic Scholar `openAccessPdf.url`
   - If paywalled or no OA URL is available: falls back to `abstract_only` with a descriptive reason.
   - **Per-Paper Upload:** Users can upload their own PDF for any paper `P#` (status: `full_text (user upload)`).

2. **Download Safety & SSRF Guard (`app/security.py`):**
   - HTTPS only; host allowlist support via configuration.
   - Strict SSRF protection: DNS resolution blocks loopback, private RFC 1918, link-local, carrier-grade NAT, and reserved IP ranges.
   - Maximum 3 manual redirects; max 30 MB file size cap; 30 s timeout.
   - Magic bytes verification (`%PDF`) and content-type validation.
   - All extracted text is treated as untrusted data.

3. **Ingestion & Section Chunking (`app/ingestion.py`):**
   - Text extracted page-by-page using PyMuPDF (`fitz`), preserving 1-indexed page bounds.
   - Frequency- and coordinate-based stripping of repeated running headers and footers.
   - Automatic detection of standard academic section headings (`Introduction`, `Methods`, `Results`, `Discussion`, etc.).
   - References / Bibliography sections are automatically tagged and **excluded from retrieval by default**.
   - Scanned PDFs without a digital text layer (<100 characters) fall back to `abstract_only` with reason `no text layer`.
   - Chunking: grouped by section (~700-token windows with 100-token overlap).

4. **Hybrid Retrieval (`app/retrieval.py`):**
   - Dense semantic embeddings generated via `sentence-transformers` using `all-MiniLM-L6-v2` (fast CPU inference, ~0.02s per 10 items).
   - Lexical search using `rank_bm25.BM25Okapi`.
   - Reciprocal Rank Fusion (RRF $k=60$) combining BM25 and dense ranks.
   - Lazy on-demand ingestion (or capped "Read Top 5").

5. **Groq Tool Use & Map-Reduce Summarizer (`app/llm.py`, `app/summarizer.py`):**
   - Groq API integration (`llama-3.3-70b-versatile` / `llama-3.1-8b-instant`).
   - Tools: `search_papers`, `get_paper`, `expand_citations`, `read_paper`, `search_paper_text` (max 5 tool calls per turn).
   - Map-reduce summarizer over section chunks for full-paper overviews to prevent context overflow.
   - Streaming Server-Sent Events (SSE) with live progress statuses (`"Downloading P2..."`, `"Reading P2..."`, `"Searching passages..."`).

6. **Grounding & Server-Enforced Rules (`app/grounding.py`):**
   - Citations formatted strictly as `[P# p.N]`.
   - Server validates that `P#` exists in the pool, page $N$ is within paper bounds, and the chunk was retrieved in the current turn; invalid citations are stripped and flagged.
   - Direct quotes must be $\le 25$ words and appear verbatim in retrieved chunks; otherwise stripped and flagged.
   - Absolute novelty claims (`"nobody has done this"`, `"is novel"`) and automated percentage scores are rejected.

### Running Locally
```powershell
# In PowerShell:
.\run.ps1
# Opens server at http://127.0.0.1:8000/chat
```

### Honest Limitations
- **Open-Access Only:** Approximately 64% of top search results have directly downloadable open-access PDFs; paywalled papers require manual user upload.
- **PDF Layout Extraction:** Complex multi-column layouts, inline mathematical formulas, and tables can occasionally extract out of order.
- **No OCR:** Scanned papers lacking an embedded text layer fall back to abstract-only.
- **Not Exhaustive:** Retrieval is bounded by OpenAlex / arXiv coverage.
- **Not a Novelty Judgment:** The assistant retrieves and synthesizes literature; it does not issue automated novelty determinations.
