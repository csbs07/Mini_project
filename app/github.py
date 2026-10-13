"""GitHub REST API integration with precision search queries and backend keyword-overlap relevance filtering."""
import re
import time
from typing import Any

import requests

from .config import GITHUB_RELEVANCE_THRESHOLD

GITHUB_SEARCH_URL = "https://api.github.com/search/repositories"
DEFAULT_USER_AGENT = "Research-Novelty-Verification-Assistant"
CACHE_TTL_SECONDS = 600  # 10 minutes cache to preserve rate limits

# Simple in-memory cache: query_string -> (timestamp, result_dict)
_SEARCH_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}

# Common narrative stop phrases to filter out
STOP_PHRASES = [
    r"\bin this paper\b",
    r"\bwe propose\b",
    r"\bour approach\b",
    r"\bour method\b",
    r"\bthis research\b",
    r"\bthis project\b",
    r"\bnovel framework\b",
    r"\bstate of the art\b",
    r"\baims to\b",
    r"\bwe introduce\b",
    r"\bwe evaluate\b",
    r"\bwe demonstrate\b",
    r"\bwe investigate\b",
    r"\bthe goal is\b",
]

COMMON_STOP_WORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "using", "used", "via", "into",
    "through", "during", "before", "after", "above", "below", "between", "under",
    "we", "our", "us", "it", "its", "they", "them", "their", "this", "that", "these",
    "those", "which", "what", "where", "when", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "than", "too", "very",
    "can", "will", "just", "should", "now", "paper", "study", "research", "proposal",
    "idea", "project", "work", "novel", "novelty", "system", "new", "based",
}

# Generic computer science words that should NEVER be used alone in search queries or for relevance scoring
GENERIC_WORDS = {
    "model", "models", "data", "learning", "network", "networks", "system", "systems",
    "approach", "approaches", "paper", "algorithm", "algorithms", "deep", "neural",
    "framework", "frameworks", "analysis", "tool", "tools", "code", "method", "methods",
    "application", "applications", "research", "study", "using", "based", "via", "task",
    "tasks", "technique", "techniques", "process", "processes", "pipeline", "pipelines",
    "program", "programs", "software", "implementation", "implementations",
}


def extract_research_facets(text: str) -> dict[str, list[str]]:
    """
    Extract or categorize research keywords into standard facets:
    - problem: the research problem or application domain
    - method: the proposed model, architecture, or algorithm
    - dataset: benchmark or dataset names
    - evaluation: metrics or evaluation protocols
    """
    clean_text = text.strip()
    facets: dict[str, list[str]] = {
        "problem": [],
        "method": [],
        "dataset": [],
        "evaluation": [],
        "general_keywords": [],
    }

    if not clean_text:
        return facets

    # 1. Check for explicit facet headings in text (e.g. "Problem: ... Method: ...")
    problem_match = re.search(r"(?:problem|task|domain|objective)\s*:\s*([^;\n\.]+)", clean_text, re.IGNORECASE)
    method_match = re.search(r"(?:method|model|approach|architecture|algorithm)\s*:\s*([^;\n\.]+)", clean_text, re.IGNORECASE)
    dataset_match = re.search(r"(?:dataset|benchmark|corpus|data)\s*:\s*([^;\n\.]+)", clean_text, re.IGNORECASE)
    eval_match = re.search(r"(?:evaluation|metric|protocol)\s*:\s*([^;\n\.]+)", clean_text, re.IGNORECASE)

    if problem_match:
        facets["problem"].extend([k.strip() for k in re.split(r"[,/]", problem_match.group(1)) if k.strip()])
    if method_match:
        facets["method"].extend([k.strip() for k in re.split(r"[,/]", method_match.group(1)) if k.strip()])
    if dataset_match:
        facets["dataset"].extend([k.strip() for k in re.split(r"[,/]", dataset_match.group(1)) if k.strip()])
    if eval_match:
        facets["evaluation"].extend([k.strip() for k in re.split(r"[,/]", eval_match.group(1)) if k.strip()])

    # 2. Extract technical terms / general keywords by cleaning narrative words
    filtered = clean_text
    for pat in STOP_PHRASES:
        filtered = re.sub(pat, " ", filtered, flags=re.IGNORECASE)

    # Detect known datasets/benchmarks
    known_datasets = [
        "imagenet", "cifar", "cifar-10", "cifar-100", "mnist", "coco", "squad",
        "glue", "superglue", "mmlu", "gsm8k", "human-eval", "planetoid", "cora",
        "citeseer", "pubmed", "openalex", "wikitext", "common crawl", "pile",
    ]
    for d in known_datasets:
        if re.search(r"\b" + re.escape(d) + r"\b", filtered, re.IGNORECASE):
            if d not in [x.lower() for x in facets["dataset"]]:
                facets["dataset"].append(d)

    # Detect common AI/CS methods
    known_methods = [
        "diffusion model", "diffusion models", "transformer", "transformers",
        "graph neural network", "gnn", "gnns", "reinforcement learning", "rlhf",
        "dpo", "lora", "qlora", "retrieval-augmented generation", "rag",
        "contrastive learning", "variational autoencoder", "vae", "gan",
        "gans", "bm25", "specter", "mamba", "state space model", "ssm",
        "siamese network", "random walk", "reciprocal rank fusion",
    ]
    for m in known_methods:
        if re.search(r"\b" + re.escape(m) + r"\b", filtered, re.IGNORECASE):
            if m not in [x.lower() for x in facets["method"]]:
                facets["method"].append(m)

    # Detect common evaluation metrics
    known_evals = [
        "accuracy", "f1-score", "f1 score", "precision", "recall", "auc", "roc-auc",
        "bleu", "rouge", "perplexity", "hit@10", "ndcg", "mrr", "map", "loss",
    ]
    for e in known_evals:
        if re.search(r"\b" + re.escape(e) + r"\b", filtered, re.IGNORECASE):
            if e not in [x.lower() for x in facets["evaluation"]]:
                facets["evaluation"].append(e)

    # General keywords: alphanumeric words of length >= 3 not in stop words or generic words
    words = re.findall(r"[A-Za-z0-9_\-\+]+", filtered)
    meaningful = [
        w for w in words
        if len(w) >= 3
        and w.lower() not in COMMON_STOP_WORDS
        and w.lower() not in GENERIC_WORDS
        and not w.isdigit()
    ]
    # Deduplicate while preserving order
    seen = set()
    deduped_keywords = []
    for w in meaningful:
        wl = w.lower()
        if wl not in seen:
            seen.add(wl)
            deduped_keywords.append(w)

    facets["general_keywords"] = deduped_keywords[:10]
    return facets


def build_github_search_queries(raw_text: str, facets: dict[str, list[str]]) -> list[str]:
    """
    Construct high-precision, concise search queries with exact phrases and qualifiers.
    - Avoids generic standalone words (e.g. 'model', 'data').
    - Puts multi-word technical concepts in quotes (e.g. '"diffusion model"').
    - Targets 'in:name,description,topics' to avoid noisy full-README matches.
    - Limits to at most 2 queries to protect the unauthenticated rate limit.
    """
    queries = []

    methods = [m.strip() for m in facets.get("method", []) if m.strip()]
    problems = [p.strip() for p in facets.get("problem", []) if p.strip()]
    datasets = [d.strip() for d in facets.get("dataset", []) if d.strip()]
    keywords = [k.strip() for k in facets.get("general_keywords", []) if k.strip() and k.lower() not in GENERIC_WORDS]

    # Helper to quote multi-word terms
    def quote_term(t: str) -> str:
        clean = t.replace('"', '').strip()
        if " " in clean:
            return f'"{clean}"'
        return clean

    # Priority 1: method phrase + problem phrase (e.g. '"diffusion models" "single-cell RNA"')
    if methods and problems:
        m_q = quote_term(methods[0])
        p_q = quote_term(problems[0])
        queries.append(f"{m_q} {p_q}".strip())

    # Priority 2: method phrase + dataset (e.g. '"diffusion model" "cora"')
    if methods and datasets and (not queries or datasets[0] not in queries[0]):
        m_q = quote_term(methods[0])
        d_q = quote_term(datasets[0])
        queries.append(f"{m_q} {d_q}".strip())

    # Priority 3: specific problem phrase alone if no method, or method alone if multi-word
    if not queries and problems:
        queries.append(quote_term(problems[0]))
    elif not queries and methods and " " in methods[0]:
        queries.append(quote_term(methods[0]))

    # Priority 4: Top specific non-generic keywords
    if not queries and keywords:
        top_k = " ".join(keywords[:3])
        if top_k:
            queries.append(top_k)

    # Fallback: if still empty, extract clean non-generic words from text
    if not queries:
        raw_clean = [w for w in raw_text.split() if w.lower() not in COMMON_STOP_WORDS and w.lower() not in GENERIC_WORDS]
        if raw_clean:
            queries.append(" ".join(raw_clean[:3]))
        else:
            queries.append(raw_text.strip()[:30])

    # Append qualifiers 'in:name,description,topics' and ensure length <= 120
    formatted_queries = []
    for q in queries:
        clean_q = q.strip()
        if clean_q and clean_q not in formatted_queries:
            # Add qualifier
            full_q = f"{clean_q} in:name,description,topics"
            formatted_queries.append(full_q[:120])

    return formatted_queries[:2]


def calculate_repository_relevance(
    item: dict[str, Any],
    facets: dict[str, list[str]],
    raw_query: str,
) -> tuple[float, list[str]]:
    """
    Score repository relevance strictly by keyword and phrase overlap with problem and method.
    NEVER uses stars, forks, or counts.
    Returns:
    - score (float): higher indicates stronger conceptual alignment
    - matched_keywords (list[str]): exact matching concepts for user display
    """
    repo_name = (item.get("name") or "").lower()
    repo_full_name = (item.get("full_name") or "").lower()
    repo_desc = (item.get("description") or "").lower()
    repo_topics = [str(t).lower() for t in (item.get("topics") or [])]
    combined_text = f"{repo_name} {repo_full_name} {repo_desc} {' '.join(repo_topics)}"

    score = 0.0
    matched: list[str] = []
    seen_matched: set[str] = set()

    # Collect core multi-word concepts & specific terms
    core_phrases = []
    core_terms = []

    for category in ["problem", "method", "dataset"]:
        for term in facets.get(category, []):
            t_clean = term.strip().lower()
            if not t_clean or t_clean in GENERIC_WORDS:
                continue
            if " " in t_clean or "-" in t_clean:
                core_phrases.append(t_clean)
            else:
                core_terms.append(t_clean)

    # Also collect clean non-generic general keywords
    for kw in facets.get("general_keywords", []):
        kw_clean = kw.strip().lower()
        if kw_clean and kw_clean not in GENERIC_WORDS and len(kw_clean) >= 3:
            if kw_clean not in core_terms:
                core_terms.append(kw_clean)

    # 1. Exact multi-word core phrase overlap (weight: 3.0 pts)
    for phrase in core_phrases:
        # Check phrase match in text or in topics
        phrase_hyphen = phrase.replace(" ", "-")
        phrase_nospace = phrase.replace(" ", "")
        if (
            phrase in combined_text
            or phrase_hyphen in combined_text
            or phrase_nospace in combined_text
        ):
            score += 3.0
            if phrase not in seen_matched:
                seen_matched.add(phrase)
                matched.append(phrase)

    # 2. Topic tag exact matches (weight: 2.0 pts)
    for topic in repo_topics:
        for phrase in core_phrases:
            if phrase.replace(" ", "") in topic.replace("-", "") or phrase.replace(" ", "-") == topic:
                score += 2.0
                if phrase not in seen_matched:
                    seen_matched.add(phrase)
                    matched.append(f"topic:{topic}")
        for term in core_terms:
            if term == topic or term in topic.split("-"):
                score += 1.5
                if term not in seen_matched:
                    seen_matched.add(term)
                    matched.append(f"topic:{topic}")

    # 3. Name matches (weight: 2.0 pts for phrase, 1.5 pts for specific term)
    for phrase in core_phrases:
        if phrase.replace(" ", "-") in repo_name or phrase.replace(" ", "") in repo_name:
            score += 2.0
    for term in core_terms:
        if term in repo_name.split("-") or term in repo_name.split("_"):
            score += 1.5
            if term not in seen_matched:
                seen_matched.add(term)
                matched.append(term)

    # 4. Description specific technical keyword matches (weight: 1.0 pts)
    for term in core_terms:
        if re.search(r"\b" + re.escape(term) + r"\b", repo_desc):
            score += 1.0
            if term not in seen_matched:
                seen_matched.add(term)
                matched.append(term)

    # Core match guardrail:
    # A repository MUST have matched at least one core problem, method, or dataset term/phrase.
    # Standalone generic matches or zero matches receive 0.0.
    has_core_match = False
    methods_lower = [m.lower() for m in facets.get("method", [])]
    problems_lower = [p.lower() for p in facets.get("problem", [])]
    datasets_lower = [d.lower() for d in facets.get("dataset", [])]

    for m in matched:
        clean_m = m.replace("topic:", "").strip()
        if (
            any(clean_m in meth or meth in clean_m for meth in methods_lower)
            or any(clean_m in prob or prob in clean_m for prob in problems_lower)
            or any(clean_m in dat or dat in clean_m for dat in datasets_lower)
            or any(clean_m in ph for ph in core_phrases)
        ):
            has_core_match = True
            break

    if not has_core_match:
        score = 0.0

    return score, matched


def search_github_repositories(
    query: str,
    max_results: int | None = None,
    min_threshold: float | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """
    Search GitHub public repositories using unauthenticated GitHub REST API
    and enforce a strict backend keyword-overlap relevance filter.

    - Queries are focused on name, description, and topics (no full README).
    - Repositories are filtered by GITHUB_RELEVANCE_THRESHOLD.
    - Irrelevant repos are rejected (returns 0 repos if none pass).
    - Sorted strictly by relevance_score descending (best first).
    - Forks are filtered out unless they are the only match.
    - Archived repos are marked as archived: True.
    - Every repo includes matched_keywords.
    """
    effective_threshold = (
        min_threshold if min_threshold is not None else GITHUB_RELEVANCE_THRESHOLD
    )
    cache_key = f"{query.strip().lower()}::thr_{effective_threshold}"
    now = time.time()

    # Check in-memory cache
    if use_cache and cache_key in _SEARCH_CACHE:
        cached_time, cached_data = _SEARCH_CACHE[cache_key]
        if now - cached_time < CACHE_TTL_SECONDS:
            return cached_data

    facets = extract_research_facets(query)
    search_queries = build_github_search_queries(query, facets)

    raw_candidates: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    is_rate_limited = False
    error_message: str | None = None

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": DEFAULT_USER_AGENT,
    }

    # Execute search queries (capped at 2 to conserve the unauthenticated 10 req/min rate limit)
    for q in search_queries[:2]:
        params = {
            "q": q,
            "sort": "stars",
            "order": "desc",
            "per_page": 20,
        }

        try:
            resp = requests.get(GITHUB_SEARCH_URL, params=params, headers=headers, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("items", [])
                for item in items:
                    repo_id = item.get("id")
                    if repo_id in seen_ids:
                        continue
                    seen_ids.add(repo_id)
                    raw_candidates.append(item)

            elif resp.status_code in (403, 429):
                is_rate_limited = True
                reset_header = resp.headers.get("X-RateLimit-Reset")
                retry_after = ""
                if reset_header:
                    try:
                        seconds_left = max(0, int(reset_header) - int(time.time()))
                        retry_after = f" Resets in ~{seconds_left}s."
                    except Exception:
                        pass
                error_message = (
                    f"GitHub API rate limit reached (HTTP {resp.status_code})."
                    f"{retry_after} Public unauthenticated requests are limited to 10 searches/min."
                )
                break
            else:
                error_message = f"GitHub API returned HTTP {resp.status_code}: {resp.text[:100]}"
                break

        except requests.Timeout:
            error_message = "GitHub API connection timed out. Please try again."
            break
        except requests.RequestException as e:
            error_message = f"GitHub API network connection error: {e}"
            break
        except Exception as e:
            error_message = f"Unexpected error while searching GitHub: {e}"
            break

    # --------------------------------------------------------------------------
    # Relevance Filtering: Score each candidate by keyword overlap
    # --------------------------------------------------------------------------
    qualified_repos: list[dict[str, Any]] = []

    for item in raw_candidates:
        score, matched_keywords = calculate_repository_relevance(item, facets, query)

        # Filter strictly by threshold
        if score < effective_threshold:
            continue

        desc = item.get("description") or "No description provided."
        updated_at = item.get("updated_at", "")
        if updated_at and "T" in updated_at:
            updated_at = updated_at.split("T")[0]

        qualified_repos.append({
            "id": item.get("id"),
            "name": item.get("name", "Unnamed Repo"),
            "full_name": item.get("full_name", item.get("name", "")),
            "url": item.get("html_url", ""),
            "description": desc[:350],
            "language": item.get("language") or "Not specified",
            "stars": item.get("stargazers_count", 0),
            "forks": item.get("forks_count", 0),
            "updated_at": updated_at,
            "topics": (item.get("topics") or [])[:5],
            "is_fork": bool(item.get("fork", False)),
            "archived": bool(item.get("archived", False)),
            "relevance_score": round(score, 2),
            "matched_keywords": matched_keywords,
        })

    # Fork removal: if non-forks exist, remove forks. Only retain forks if they are the only match.
    non_forks = [r for r in qualified_repos if not r["is_fork"]]
    if non_forks:
        final_repos = non_forks
    else:
        final_repos = qualified_repos

    # Sort strictly by relevance score descending (best matches first)
    final_repos.sort(key=lambda r: r["relevance_score"], reverse=True)

    # If max_results is explicitly provided, optionally slice
    if max_results is not None and max_results > 0:
        final_repos = final_repos[:max_results]

    disclaimer = (
        "Supporting Evidence Notice: Discovered GitHub repositories reflect public code implementations, "
        "which provide evidence of prior software tooling or algorithmic implementations. "
        "They do not constitute automated proof that a research idea is novel or non-novel. "
        "A lack of matching repositories does not imply scientific novelty, and GitHub does not cover all published research."
    )

    empty_reason = None
    if len(final_repos) == 0 and not is_rate_limited and error_message is None:
        empty_reason = (
            "No closely related repositories found on GitHub. "
            "This does not prove your idea is novel, because GitHub does not cover all research papers."
        )

    result = {
        "query": query,
        "facets": facets,
        "searched_queries": search_queries,
        "repositories": final_repos,
        "count": len(final_repos),
        "relevance_threshold": effective_threshold,
        "rate_limited": is_rate_limited,
        "error": error_message,
        "empty_message": empty_reason,
        "disclaimer": disclaimer,
    }

    # Cache successful searches
    if not is_rate_limited and error_message is None:
        _SEARCH_CACHE[cache_key] = (now, result)

    return result
