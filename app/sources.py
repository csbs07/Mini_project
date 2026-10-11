"""Legal open-access PDF resolution and source discovery."""
import re
from typing import Any

import requests


def extract_arxiv_id(paper: dict[str, Any]) -> str | None:
    """Extract an arXiv ID from paper metadata (ids, doi, url, or primary_location)."""
    # 1. Direct IDs dictionary
    ids = paper.get("ids", {})
    if isinstance(ids, dict):
        arxiv_id = ids.get("arxiv")
        if arxiv_id:
            # Clean arXiv prefix if present
            clean = re.sub(r"^https?://arxiv\.org/abs/", "", arxiv_id)
            clean = re.sub(r"^arxiv:", "", clean, flags=re.IGNORECASE)
            return clean.strip()

    # 2. Check DOI for arXiv: 10.48550/arXiv.XXXX.XXXXX
    doi = paper.get("doi") or ""
    match = re.search(r"10\.48550/arXiv\.([0-9]+\.[0-9]+(?:v[0-9]+)?)", doi, re.IGNORECASE)
    if match:
        return match.group(1)

    # 3. Check paper landing page URL or primary_location
    url = paper.get("url") or ""
    match = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]+\.[0-9]+(?:v[0-9]+)?)", url, re.IGNORECASE)
    if match:
        return match.group(1)

    prim = paper.get("primary_location") or {}
    landing = prim.get("landing_page_url") or ""
    match = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]+\.[0-9]+(?:v[0-9]+)?)", landing, re.IGNORECASE)
    if match:
        return match.group(1)

    return None


def fetch_semantic_scholar_oa_pdf(doi_or_title: str) -> str | None:
    """Query Semantic Scholar Graph API for openAccessPdf URL."""
    if not doi_or_title:
        return None
    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/{doi_or_title}?fields=openAccessPdf"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            oa_info = data.get("openAccessPdf")
            if oa_info and isinstance(oa_info, dict):
                pdf_url = oa_info.get("url")
                if pdf_url and pdf_url.startswith("https://"):
                    return pdf_url
    except Exception:
        pass
    return None


def find_pdf_source(paper: dict[str, Any]) -> tuple[str | None, str, str | None]:
    """
    Find a legal open access PDF URL in the required priority order:
    1. OpenAlex best_oa_location.pdf_url
    2. open_access.oa_url
    3. other OA locations
    4. arXiv PDF if an arXiv id exists
    5. Semantic Scholar openAccessPdf if available
    6. Paywalled or no OA copy: status 'abstract_only' with a reason.

    Returns:
        (pdf_url, status, reason)
        where status is 'oa_available' or 'abstract_only'
    """
    # 1. OpenAlex best_oa_location.pdf_url
    best_oa = paper.get("best_oa_location")
    if isinstance(best_oa, dict):
        pdf_url = best_oa.get("pdf_url")
        if pdf_url and pdf_url.startswith("https://"):
            return pdf_url, "oa_available", "openalex_best_oa"

    # 2. open_access.oa_url
    oa_info = paper.get("open_access", {})
    if isinstance(oa_info, dict):
        oa_url = oa_info.get("oa_url")
        if oa_url and oa_url.startswith("https://"):
            if oa_url.lower().endswith(".pdf") or "/pdf/" in oa_url.lower():
                return oa_url, "oa_available", "openalex_oa_url"

    # 3. Other OA locations in locations list
    locations = paper.get("locations") or []
    if isinstance(locations, list):
        for loc in locations:
            if isinstance(loc, dict) and loc.get("is_oa"):
                pdf_url = loc.get("pdf_url")
                if pdf_url and pdf_url.startswith("https://"):
                    return pdf_url, "oa_available", "openalex_location"

    # 4. arXiv PDF if an arXiv ID exists
    arxiv_id = extract_arxiv_id(paper)
    if arxiv_id:
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf", "oa_available", "arxiv"

    # 5. Semantic Scholar openAccessPdf if available
    doi = paper.get("doi")
    if doi:
        # Strip https://doi.org/
        clean_doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi)
        s2_url = fetch_semantic_scholar_oa_pdf(clean_doi)
        if s2_url:
            return s2_url, "oa_available", "semantic_scholar"

    # 6. Fallback: Paywalled or no OA copy
    is_oa = False
    if isinstance(oa_info, dict):
        is_oa = oa_info.get("is_oa", False)

    if not is_oa:
        return None, "abstract_only", "Paywalled (no legal open-access version found)"
    return None, "abstract_only", "Open-access landing page available, but direct PDF URL not indexed"
