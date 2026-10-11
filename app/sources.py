"""Legal open-access PDF resolution and multi-source fallback discovery."""
import re
from typing import Any

import requests


def extract_arxiv_id(paper: dict[str, Any]) -> str | None:
    """Extract an arXiv ID from paper metadata (ids, doi, url, or primary_location)."""
    ids = paper.get("ids", {})
    if isinstance(ids, dict):
        arxiv_id = ids.get("arxiv")
        if arxiv_id:
            clean = re.sub(r"^https?://arxiv\.org/abs/", "", arxiv_id)
            clean = re.sub(r"^arxiv:", "", clean, flags=re.IGNORECASE)
            return clean.strip()

    doi = paper.get("doi") or ""
    match = re.search(r"10\.48550/arXiv\.([0-9]+\.[0-9]+(?:v[0-9]+)?)", doi, re.IGNORECASE)
    if match:
        return match.group(1)

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


def normalize_pdf_url(url: str) -> str | None:
    """Ensure URL is https and valid."""
    if not url or not isinstance(url, str):
        return None
    cleaned = url.strip()
    if cleaned.startswith("http://"):
        cleaned = "https://" + cleaned[7:]
    if not cleaned.startswith("https://"):
        return None
    return cleaned


def extract_pmc_id(paper: dict[str, Any]) -> str | None:
    """Extract PMC ID from paper metadata or locations."""
    locations = paper.get("locations") or []
    if isinstance(locations, list):
        for loc in locations:
            loc_id = str(loc.get("id") or "")
            match = re.search(r"pubmedcentral\.nih\.gov:([0-9]+)", loc_id)
            if match:
                return match.group(1)
            landing = str(loc.get("landing_page_url") or "")
            match = re.search(r"ncbi\.nlm\.nih\.gov/pmc/articles/(?:PMC)?([0-9]+)", landing)
            if match:
                return match.group(1)

    ids = paper.get("ids", {})
    if isinstance(ids, dict):
        pmcid = ids.get("pmcid")
        if pmcid:
            clean = re.sub(r"^https?://(?:www\.)?ncbi\.nlm\.nih\.gov/pmc/articles/", "", pmcid)
            clean = re.sub(r"^PMC", "", clean, flags=re.IGNORECASE).strip()
            return clean

    return None


def get_candidate_pdf_urls(paper: dict[str, Any]) -> list[str]:
    """
    Collect all possible legal open-access PDF candidate URLs in priority order:
    1. arXiv direct PDF (if arXiv ID exists)
    2. BioRxiv / MedRxiv direct PDF
    3. Institutional / open repository locations (Zenodo, HAL, OSF, .edu, .ac.uk)
    4. PubMed Central mirrors
    5. OpenAlex best_oa_location.pdf_url
    6. Other OpenAlex locations with pdf_url
    7. Semantic Scholar openAccessPdf
    8. open_access.oa_url (if direct PDF)
    """
    candidates: list[str] = []

    # 1. arXiv direct PDF
    arxiv_id = extract_arxiv_id(paper)
    if arxiv_id:
        candidates.append(f"https://arxiv.org/pdf/{arxiv_id}.pdf")

    # 2. PubMed Central direct PDF if PMC ID exists
    pmc_num = extract_pmc_id(paper)
    if pmc_num:
        epmc_url = f"https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC{pmc_num}&blobtype=pdf"
        pmc_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_num}/pdf/"
        for u in [epmc_url, pmc_url]:
            if u not in candidates:
                candidates.append(u)

    # 3. Separate locations: prioritize open repositories before commercial publishers
    locations = paper.get("locations") or []
    repo_cands: list[str] = []
    publisher_cands: list[str] = []

    if isinstance(locations, list):
        for loc in locations:
            if isinstance(loc, dict):
                p_url = normalize_pdf_url(loc.get("pdf_url"))
                if not p_url:
                    continue
                p_url_lower = p_url.lower()
                # Check if from a commercial publisher with bot protection
                is_commercial = any(
                    dom in p_url_lower
                    for dom in [
                        "academic.oup.com",
                        "springer.com",
                        "sciencedirect.com",
                        "wiley.com",
                        "nature.com",
                        "tandfonline.com",
                        "ieee.org",
                    ]
                )
                if is_commercial:
                    if p_url not in publisher_cands:
                        publisher_cands.append(p_url)
                else:
                    if p_url not in repo_cands:
                        repo_cands.append(p_url)

    # Add open repository candidates first
    for r_url in repo_cands:
        if r_url not in candidates:
            candidates.append(r_url)

    # 4. OpenAlex best_oa_location.pdf_url
    best_oa = paper.get("best_oa_location")
    if isinstance(best_oa, dict):
        b_url = normalize_pdf_url(best_oa.get("pdf_url"))
        if b_url and b_url not in candidates:
            candidates.append(b_url)

    # 5. Publisher candidates
    for p_url in publisher_cands:
        if p_url not in candidates:
            candidates.append(p_url)

    # 6. Semantic Scholar openAccessPdf
    doi = paper.get("doi")
    if doi:
        clean_doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi)
        s2_url = normalize_pdf_url(fetch_semantic_scholar_oa_pdf(clean_doi))
        if s2_url and s2_url not in candidates:
            candidates.append(s2_url)

    # 7. open_access.oa_url if direct PDF
    oa_info = paper.get("open_access", {})
    if isinstance(oa_info, dict):
        oa_url = normalize_pdf_url(oa_info.get("oa_url"))
        if oa_url and (oa_url.lower().endswith(".pdf") or "/pdf/" in oa_url.lower()):
            if oa_url not in candidates:
                candidates.append(oa_url)

    return candidates


def find_pdf_source(paper: dict[str, Any]) -> tuple[str | None, str, str | None]:
    """Find a legal open access PDF URL in required priority order."""
    candidates = get_candidate_pdf_urls(paper)
    if candidates:
        return candidates[0], "oa_available", "open_access_found"

    oa_info = paper.get("open_access", {})
    is_oa = oa_info.get("is_oa", False) if isinstance(oa_info, dict) else False

    if not is_oa:
        return None, "abstract_only", "Paywalled (no legal open-access version found)"
    return None, "abstract_only", "Open-access landing page available, but direct PDF URL not indexed"
