"""
dataset_builder.py — NovaGap-Lite
Pulls CS papers from OpenAlex (free, no API key needed),
normalizes concepts, builds temporal co-occurrence graph G_T,
and extracts post-cutoff pair counts for backtesting.
"""

import re
import time
import requests
import networkx as nx
from collections import defaultdict
from itertools import combinations
from typing import List, Dict, Tuple

# Known aliases to merge (keeps concept graph clean)
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
    """Lowercase, strip punctuation, resolve known aliases."""
    c = raw.lower().strip()
    c = re.sub(r"[^\w\s-]", "", c)
    c = re.sub(r"\s+", " ", c)
    # Alias resolution
    for fragment, canonical in ALIAS_MAP.items():
        if fragment in c:
            return canonical
    return c


def fetch_openalex_cs(
    from_year: int,
    to_year:   int,
    per_page:  int = 200,
    max_pages: int = 10,
) -> List[Dict]:
    """
    Pulls CS papers from OpenAlex public API (no key required).
    concept ID c41008148 = Computer Science
    Returns list of {id, year, citations, concepts}.
    """
    papers   = []
    base_url = "https://api.openalex.org/works"
    cs_id    = "c41008148"

    for page in range(1, max_pages + 1):
        params = {
            "filter":   f"concepts.id:{cs_id},publication_year:{from_year}-{to_year}",
            "per_page": per_page,
            "page":     page,
            "select":   "id,title,publication_year,concepts,cited_by_count",
        }
        try:
            resp = requests.get(base_url, params=params, timeout=30)
            if resp.status_code != 200:
                print(f"  [OpenAlex] HTTP {resp.status_code} on page {page}, stopping.")
                break

            items = resp.json().get("results", [])
            if not items:
                break

            for item in items:
                raw_concepts = item.get("concepts", [])
                concepts = list({
                    normalize_concept(c["display_name"])
                    for c in raw_concepts
                    if c.get("score", 0) > 0.3
                    and len(c.get("display_name", "")) > 2
                })
                if len(concepts) >= 2:
                    papers.append({
                        "id":        item["id"],
                        "year":      item["publication_year"],
                        "citations": item.get("cited_by_count", 0),
                        "concepts":  concepts,
                    })

            print(f"  [OpenAlex] page {page}/{max_pages}: +{len(items)} papers  (total so far: {len(papers)})")
            time.sleep(0.3)  # polite throttle

        except Exception as e:
            print(f"  [OpenAlex] error on page {page}: {e}")
            break

    return papers


def build_temporal_split(
    papers:            List[Dict],
    cutoff_T:          int,
    eval_window_years: int = 3,
) -> Tuple[nx.Graph, Dict[Tuple[str, str], int], Dict[Tuple[str, str], int]]:
    """
    Returns:
      G_train          — co-occurrence graph from papers with year < cutoff_T
      test_cooccur     — pair->count in [cutoff_T, cutoff_T+eval_window_years]
      test_citations   — pair->max citations of papers containing that pair in the window
    """
    G_train        = nx.Graph()
    test_cooccur   = defaultdict(int)
    test_citations = defaultdict(int)

    eval_max_year = cutoff_T + eval_window_years

    for p in papers:
        year     = p["year"]
        concepts = list(set(p["concepts"]))
        cites    = p["citations"]

        if year < cutoff_T:
            for c in concepts:
                if not G_train.has_node(c):
                    G_train.add_node(c, paper_count=0)
                G_train.nodes[c]["paper_count"] += 1

            for u, v in combinations(concepts, 2):
                pair = tuple(sorted([u, v]))
                if G_train.has_edge(pair[0], pair[1]):
                    G_train[pair[0]][pair[1]]["weight"] += 1
                else:
                    G_train.add_edge(pair[0], pair[1], weight=1)

        elif cutoff_T <= year <= eval_max_year:
            for u, v in combinations(concepts, 2):
                pair = tuple(sorted([u, v]))
                test_cooccur[pair]   += 1
                test_citations[pair]  = max(test_citations[pair], cites)

    return G_train, dict(test_cooccur), dict(test_citations)
