"""OpenAlex API client: search papers, get paper metadata, citation network expansion."""
import time
from typing import Any

import requests

from .config import OPENALEX_MAILTO


def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """Reconstruct continuous abstract string from OpenAlex abstract_inverted_index."""
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""
    pos_word = []
    for word, positions in inverted_index.items():
        for pos in positions:
            pos_word.append((pos, word))
    pos_word.sort(key=lambda x: x[0])
    return " ".join(w for _, w in pos_word)


def normalize_openalex_id(raw_id: str) -> str:
    """Normalize OpenAlex URL (https://openalex.org/W123) to short ID (W123)."""
    if not raw_id:
        return ""
    return raw_id.split("/")[-1].strip()


def parse_work_item(work: dict[str, Any], label: str | None = None) -> dict[str, Any]:
    """Parse raw OpenAlex work JSON into standardized paper dictionary."""
    work_id = normalize_openalex_id(work.get("id", ""))
    title = work.get("title") or "Untitled Work"
    year = work.get("publication_year")

    # Abstract reconstruction
    inv_abs = work.get("abstract_inverted_index")
    abstract = reconstruct_abstract(inv_abs) if inv_abs else ""

    # Authors
    authors = []
    for auth in work.get("authorships", []):
        name = auth.get("author", {}).get("display_name")
        if name:
            authors.append(name)

    # DOI & URLs
    doi = work.get("doi")
    landing_url = work.get("primary_location", {}).get("landing_page_url") or doi or f"https://openalex.org/{work_id}"

    return {
        "id": work_id,
        "label": label or "",
        "title": title,
        "year": year,
        "abstract": abstract,
        "authors": authors[:5],
        "doi": doi,
        "url": landing_url,
        "best_oa_location": work.get("best_oa_location"),
        "open_access": work.get("open_access", {}),
        "locations": work.get("locations", []),
        "ids": work.get("ids", {}),
        "referenced_works": [normalize_openalex_id(rw) for rw in work.get("referenced_works", [])],
        "full_text_status": "unprocessed",  # unprocessed | reading | full_text | abstract_only | full_text (user upload)
        "full_text_reason": None,
        "page_count": None,
        "sections": [],
    }


def search_openalex_papers(query: str, per_page: int = 15) -> list[dict[str, Any]]:
    """
    Search OpenAlex works matching user query or idea.
    Assigns sequential labels P1, P2, ... to results.
    """
    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per_page": per_page,
        "mailto": OPENALEX_MAILTO,
    }
    papers = []
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])
            for idx, item in enumerate(results):
                label = f"P{idx + 1}"
                parsed = parse_work_item(item, label=label)
                papers.append(parsed)
    except Exception as e:
        print(f"[OpenAlex Search Error] {e}")
    return papers


def get_openalex_paper(paper_id: str) -> dict[str, Any] | None:
    """Fetch a single work from OpenAlex by ID."""
    clean_id = normalize_openalex_id(paper_id)
    url = f"https://api.openalex.org/works/{clean_id}"
    params = {"mailto": OPENALEX_MAILTO}
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            return parse_work_item(resp.json())
    except Exception as e:
        print(f"[OpenAlex Get Error] {e}")
    return None


def expand_citations_openalex(paper_id: str, limit: int = 5) -> list[dict[str, Any]]:
    """Fetch referenced works for a paper to expand citations."""
    paper = get_openalex_paper(paper_id)
    if not paper or not paper.get("referenced_works"):
        return []

    ref_ids = paper["referenced_works"][:limit]
    results = []
    for r_id in ref_ids:
        ref_work = get_openalex_paper(r_id)
        if ref_work:
            results.append(ref_work)
        time.sleep(0.1)
    return results
