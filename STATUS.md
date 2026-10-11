# Study & Project Status (v2-lite)

- **Branch:** `v2-lite`
- **Current Step:** STEP 1 (Topic Selection)
- **Gates Passed:** None (At GATE 1)
- **Current Gate:** GATE 1 — User chooses 4 Computer Science topics.
- **Next Action:** Wait for User selection of 4 topics at GATE 1, then proceed to STEP 2 (Corpus, Targets, Queries).

## Progress Checklist
- [x] Phase 0: Clean up and archive finished NovaGap evaluation study (`archive/novagap_eval/`, `archive/old_novagap_app/`).
- [x] Branch initialized: `v2-lite`.
- [x] Step 1: OpenAlex CS topics queried with 2023–2024 work counts (has_abstract:true).
- [ ] **GATE 1: User chooses 4 topics.** (AWAITING SELECTION)
- [ ] Step 2: Corpus, targets (40 targets: 10 per topic), query formulation, BM25 & SPECTER2 index, PREREG.md, tagging `prereg-v1`.
- [ ] **GATE 2: Review queries_draft.csv and confirm PREREG.md.**
- [ ] Step 3: Web App MVP (FastAPI + static HTML, live OpenAlex search, reranking, reference safety, export, tests, latency measure).
- [ ] Step 4: Run retrieval systems S1–S4, LLM Judge, generate `validation_sheet.csv` (60 items).
- [ ] **GATE 3: Human validation labeling (60 items).**
- [ ] Step 5: Statistical analysis (linear weighted kappa, Outcomes A–D, bootstrap, sensitivity tables, tag `results-v1`).
