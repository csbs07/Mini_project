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
