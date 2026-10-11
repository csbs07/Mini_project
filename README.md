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
