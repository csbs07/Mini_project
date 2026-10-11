"""
Multi-source retrieval: arXiv + GitHub + curated sample fallback.
"""
import time
import requests
from typing import List, Dict, Any, Optional
import arxiv
from .models import RetrievedItem


SAMPLE_PAPERS = [
    {
        "title": "Can LLMs Generate Novel Research Ideas? A Large-Scale Human Study with 100+ NLP Researchers",
        "abstract": "We perform the first head-to-head comparison between expert NLP researchers and an LLM ideation agent. LLM-generated ideas are judged as more novel than human expert ideas while being slightly weaker on feasibility.",
        "year": 2024,
        "url": "https://arxiv.org/abs/2409.04109",
        "source": "sample"
    },
    {
        "title": "Literature-Grounded Novelty Assessment of Scientific Ideas",
        "abstract": "We propose the Idea Novelty Checker, an LLM-based RAG framework that uses two-stage retrieve-then-rerank and expert-labeled examples to evaluate novelty of scientific ideas with literature grounding.",
        "year": 2025,
        "url": "https://arxiv.org/abs/2506.22026",
        "source": "sample"
    },
    {
        "title": "Automated Novelty Evaluation of Academic Paper: A Collaborative Approach Integrating Human and LLM Knowledge",
        "abstract": "We integrate human peer-review knowledge and LLM summaries of methodology sections to train models that predict method novelty of academic papers.",
        "year": 2025,
        "url": "https://arxiv.org/abs/2507.11330",
        "source": "sample"
    },
    {
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "abstract": "RAG combines a pre-trained retriever with a pre-trained generator to improve knowledge-intensive tasks and reduce hallucination.",
        "year": 2020,
        "url": "https://arxiv.org/abs/2005.11401",
        "source": "sample"
    },
    {
        "title": "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
        "abstract": "We present Sentence-BERT, a modification of BERT that uses siamese networks to derive semantically meaningful sentence embeddings that can be compared using cosine similarity.",
        "year": 2019,
        "url": "https://arxiv.org/abs/1908.10084",
        "source": "sample"
    },
    {
        "title": "SciMON: Scientific Inspiration Machines Optimized for Novelty",
        "abstract": "SciMON generates novel scientific ideas by retrieving and combining concepts from literature while optimizing for novelty.",
        "year": 2024,
        "url": "https://arxiv.org/abs/2305.14647",
        "source": "sample"
    },
]

SAMPLE_REPOS = [
    {
        "title": "awesome-ai-auto-research",
        "description": "A curated list of AI systems and papers for automated research idea generation, literature review, and scientific discovery agents.",
        "url": "https://github.com/worldbench/awesome-ai-auto-research",
        "stars": 1200,
        "language": "Markdown",
        "source": "sample"
    },
    {
        "title": "idea_novelty_checker",
        "description": "Implementation related to literature-grounded novelty assessment of scientific ideas using RAG and re-ranking.",
        "url": "https://github.com/simra-shahid/idea_novelty_checker",
        "stars": 85,
        "language": "Python",
        "source": "sample"
    },
    {
        "title": "sentence-transformers",
        "description": "State-of-the-art sentence, text and image embeddings. Used for semantic similarity and retrieval.",
        "url": "https://github.com/UKPLab/sentence-transformers",
        "stars": 15000,
        "language": "Python",
        "source": "sample"
    },
    {
        "title": "langchain",
        "description": "Framework for developing applications powered by language models, including RAG pipelines for literature.",
        "url": "https://github.com/langchain-ai/langchain",
        "stars": 90000,
        "language": "Python",
        "source": "sample"
    },
]


def search_arxiv(query: str, max_results: int = 8) -> List[RetrievedItem]:
    items = []
    try:
        client = arxiv.Client(page_size=max_results, delay_seconds=3.0, num_retries=2)
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        for r in client.results(search):
            abstract = (r.summary or "")[:1200]
            items.append(RetrievedItem(
                source="arxiv",
                title=r.title,
                url=r.entry_id,
                abstract_or_description=abstract,
                year=r.published.year if r.published else None,
                extra={"authors": [a.name for a in r.authors[:5]]}
            ))
            time.sleep(0.3)
    except Exception as e:
        print(f"[arXiv] retrieval error: {e}")
    return items


def search_github(query: str, max_results: int = 6) -> List[RetrievedItem]:
    items = []
    try:
        # Unauthenticated GitHub Search (rate limited ~10/min)
        url = "https://api.github.com/search/repositories"
        params = {
            "q": f"{query} in:name,description,readme",
            "sort": "stars",
            "order": "desc",
            "per_page": max_results
        }
        headers = {"Accept": "application/vnd.github.v3+json", "User-Agent": "NovaGap-Demo"}
        resp = requests.get(url, params=params, headers=headers, timeout=12)
        if resp.status_code == 200:
            data = resp.json()
            for repo in data.get("items", [])[:max_results]:
                desc = repo.get("description") or ""
                items.append(RetrievedItem(
                    source="github",
                    title=repo.get("full_name", repo.get("name", "")),
                    url=repo.get("html_url"),
                    abstract_or_description=desc[:800],
                    year=None,
                    extra={
                        "stars": repo.get("stargazers_count", 0),
                        "language": repo.get("language"),
                        "forks": repo.get("forks_count", 0)
                    }
                ))
        else:
            print(f"[GitHub] status {resp.status_code}")
    except Exception as e:
        print(f"[GitHub] retrieval error: {e}")
    return items


def get_sample_papers() -> List[RetrievedItem]:
    return [
        RetrievedItem(
            source="sample",
            title=p["title"],
            url=p["url"],
            abstract_or_description=p["abstract"],
            year=p["year"],
            extra={}
        ) for p in SAMPLE_PAPERS
    ]


def get_sample_repos() -> List[RetrievedItem]:
    return [
        RetrievedItem(
            source="sample",
            title=r["title"],
            url=r["url"],
            abstract_or_description=r["description"],
            year=None,
            extra={"stars": r.get("stars"), "language": r.get("language")}
        ) for r in SAMPLE_REPOS
    ]


def multi_source_retrieve(title: str, abstract: str, use_live: bool = True) -> Dict[str, List[RetrievedItem]]:
    """
    Retrieve from arXiv + GitHub. Falls back to curated samples if live fails or use_live=False.
    """
    query = f"{title} {abstract[:200]}"
    short_query = title

    papers: List[RetrievedItem] = []
    repos: List[RetrievedItem] = []

    if use_live:
        papers = search_arxiv(short_query, max_results=6)
        time.sleep(1)
        repos = search_github(short_query, max_results=5)

    # Always enrich with high-quality samples so demo is never empty
    sample_papers = get_sample_papers()
    sample_repos = get_sample_repos()

    # Prefer live results; pad with samples
    if len(papers) < 3:
        papers = papers + sample_papers
    else:
        papers = papers + sample_papers[:2]

    if len(repos) < 2:
        repos = repos + sample_repos
    else:
        repos = repos + sample_repos[:2]

    return {"papers": papers[:10], "repos": repos[:8]}
