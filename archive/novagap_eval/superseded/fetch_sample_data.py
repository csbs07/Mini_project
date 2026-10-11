"""
fetch_sample_data.py — Ingest OpenAlex random sample papers for 2013-2021
Filter: publication_year:{year}, concepts.id:c41008148 (Computer Science)
sample: 10000, seed: 42
concepts: level >= 2, score > 0.3
"""

import re, json, time, requests
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

def fetch_page(year, page):
    url = f"https://api.openalex.org/works?filter=publication_year:{year},concepts.id:c41008148&sample=10000&seed=42&per_page=200&page={page}&select=id,publication_year,cited_by_count,concepts"
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
                return year, page, papers
            elif resp.status_code == 429:
                time.sleep(1.0 + attempt)
        except Exception:
            time.sleep(0.5)
    return year, page, []

def main():
    years = list(range(2013, 2022))
    pages_per_year = 50  # 50 pages * 200 = 10,000 papers per year
    tasks = [(y, p) for y in years for p in range(1, pages_per_year + 1)]
    print(f"Starting fetch for {len(tasks)} requests across years 2013-2021...")

    t0 = time.time()
    all_papers_by_year_page = {}
    completed = 0

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_task = {executor.submit(fetch_page, y, p): (y, p) for y, p in tasks}
        for future in as_completed(future_to_task):
            y, p = future_to_task[future]
            try:
                ry, rp, p_list = future.result()
                all_papers_by_year_page[(ry, rp)] = p_list
            except Exception as e:
                all_papers_by_year_page[(y, p)] = []
            completed += 1
            if completed % 50 == 0 or completed == len(tasks):
                print(f"Progress: {completed}/{len(tasks)} pages fetched in {time.time() - t0:.1f}s")

    # Assemble in stable order
    final_papers = []
    for y in years:
        for p in range(1, pages_per_year + 1):
            final_papers.extend(all_papers_by_year_page.get((y, p), []))

    out_path = Path("research/openalex_sample_cache.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_papers, f)

    print(f"Saved {len(final_papers)} valid papers to {out_path} in {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
