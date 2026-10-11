"""
fetch_all_seeds.py — Ingest OpenAlex sample=10000 for seeds 43 and 44, and log metadata
"""

import sys, os, re, json, time, requests
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ALIAS_MAP = {
    "deep learning":           "deep learning",
    "neural network":          "deep learning",
    "neural networks":         "deep learning",
    "convolutional neural":    "deep learning",
    "convolutional network":   "deep learning",
    "natural language":        "natural language processing",
    "nlp":                     "natural language processing",
    "reinforcement learning":  "reinforcement learning",
    "rl":                      "reinforcement learning",
    "transformer":             "transformer model",
    "attention mechanism":     "transformer model",
    "self-attention":          "transformer model",
    "large language model":    "large language models",
    "llm":                     "large language models",
    "gpt":                     "large language models",
    "knowledge graph":         "knowledge graphs",
    "graph neural network":    "graph neural networks",
    "gnn":                     "graph neural networks",
    "federated learning":      "federated learning",
    "transfer learning":       "transfer learning",
}

def normalize_concept(raw: str) -> str:
    c = raw.lower().strip()
    c = re.sub(r"[^\w\s-]", "", c)
    c = re.sub(r"\s+", " ", c)
    for fragment, canonical in ALIAS_MAP.items():
        if fragment in c:
            return canonical
    return c

def fetch_page(year, seed, page):
    url = f"https://api.openalex.org/works?filter=publication_year:{year},concepts.id:c41008148&sample=10000&seed={seed}&per_page=200&page={page}&select=id,publication_year,cited_by_count,concepts"
    headers = {"User-Agent": "mailto:research@novagap.org"}
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=headers, timeout=20)
            if resp.status_code == 200:
                items = resp.json().get("results", [])
                papers = []
                for it in items:
                    raw_c = it.get("concepts", [])
                    cs = list({
                        normalize_concept(c["display_name"])
                        for c in raw_c
                        if c.get("level", 0) >= 2 and c.get("score", 0) > 0.3 and len(c.get("display_name", "")) > 2
                    })
                    if len(cs) >= 2:
                        papers.append({
                            "id": it["id"],
                            "year": it["publication_year"],
                            "citations": it.get("cited_by_count", 0),
                            "concepts": cs,
                        })
                return year, seed, page, papers
            elif resp.status_code == 429:
                time.sleep(1.0 + attempt)
        except Exception:
            time.sleep(0.5)
    return year, seed, page, []

def fetch_seed(seed, pages_per_year=50):
    out_file = Path(f"research/openalex_sample_seed{seed}.json")
    if out_file.exists():
        print(f"[Cache] Seed {seed} already cached at {out_file}, skipping fetch.")
        with open(out_file, "r", encoding="utf-8") as f:
            return len(json.load(f))

    years = list(range(2013, 2022))
    tasks = [(y, seed, p) for y in years for p in range(1, pages_per_year + 1)]
    print(f"Fetching seed {seed} ({len(tasks)} requests)...")
    t0 = time.time()
    by_task = {}
    completed = 0

    with ThreadPoolExecutor(max_workers=12) as executor:
        futs = {executor.submit(fetch_page, y, seed, p): (y, seed, p) for y, seed, p in tasks}
        for fut in as_completed(futs):
            y, s, p = futs[fut]
            try:
                ry, rs, rp, plist = fut.result()
                by_task[(ry, rp)] = plist
            except Exception:
                by_task[(y, p)] = []
            completed += 1
            if completed % 100 == 0 or completed == len(tasks):
                print(f"  Seed {seed}: {completed}/{len(tasks)} in {time.time() - t0:.1f}s")

    papers = []
    for y in years:
        for p in range(1, pages_per_year + 1):
            papers.extend(by_task.get((y, p), []))

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(papers, f)
    print(f"Saved {len(papers)} papers for seed {seed} in {time.time() - t0:.1f}s")
    return len(papers)

def main():
    # Ensure seed 42 file is copied/linked to seed42
    cache_42 = Path("research/openalex_sample_cache.json")
    seed_42_file = Path("research/openalex_sample_seed42.json")
    if cache_42.exists() and not seed_42_file.exists():
        with open(cache_42, "r", encoding="utf-8") as f:
            d = json.load(f)
        with open(seed_42_file, "w", encoding="utf-8") as f:
            json.dump(d, f)
        print(f"Initialized seed 42 from existing cache ({len(d)} papers).")

    counts = {}
    if seed_42_file.exists():
        with open(seed_42_file, "r", encoding="utf-8") as f:
            counts[42] = len(json.load(f))
    else:
        counts[42] = fetch_seed(42)

    counts[43] = fetch_seed(43)
    counts[44] = fetch_seed(44)

    metadata = {
        "fetch_date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "api_endpoint": "https://api.openalex.org/works",
        "filter_params": {
            "concept_filter": "concepts.id:c41008148",
            "concept_level_min": 2,
            "concept_score_min": 0.3,
            "publication_years": list(range(2013, 2022)),
            "per_page": 200,
            "pages_per_year": 50,
            "sample_size_per_year": 10000,
        },
        "query_template": "https://api.openalex.org/works?filter=publication_year:{year},concepts.id:c41008148&sample=10000&seed={seed}&per_page=200&page={page}&select=id,publication_year,cited_by_count,concepts",
        "seeds": [42, 43, 44],
        "cached_files": {
            "seed_42": "research/openalex_sample_seed42.json",
            "seed_43": "research/openalex_sample_seed43.json",
            "seed_44": "research/openalex_sample_seed44.json",
        },
        "paper_counts": counts,
    }

    meta_file = Path("research/openalex_fetch_metadata.json")
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata logged to {meta_file}")

if __name__ == "__main__":
    main()
