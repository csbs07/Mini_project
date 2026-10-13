"""Tests for GitHub REST API integration: precision query building, relevance scoring, fork/archive handling, and empty states."""
from unittest.mock import MagicMock, patch
import uuid

import pytest
from fastapi.testclient import TestClient

from app.github import (
    build_github_search_queries,
    calculate_repository_relevance,
    extract_research_facets,
    search_github_repositories,
)
from app.server import app


@pytest.fixture
def client():
    uid = uuid.uuid4().hex[:8]
    c = TestClient(app)
    c.post("/api/auth/signup", json={
        "username": f"gh_user_{uid}",
        "email": f"gh_{uid}@example.com",
        "password": "Password123!",
    })
    return c


def test_extract_research_facets_structured():
    """Test facet extraction from structured research proposal text."""
    text = (
        "Problem: single-cell RNA sequencing denoising. "
        "Method: diffusion models and transformers. "
        "Dataset: Planetoid and Cora. "
        "Evaluation: F1-score and hit@10."
    )
    facets = extract_research_facets(text)
    assert any("single-cell" in p.lower() for p in facets["problem"])
    assert any("diffusion" in m.lower() for m in facets["method"])
    assert any("planetoid" in d.lower() or "cora" in d.lower() for d in facets["dataset"])
    assert any("f1" in e.lower() or "hit@10" in e.lower() for e in facets["evaluation"])


def test_extract_research_facets_narrative():
    """Test extracting facets from natural narrative proposal without explicit labels."""
    proposal = (
        "In this paper we propose using graph neural networks and contrastive learning "
        "to solve single-cell RNA velocity estimation evaluated on accuracy."
    )
    facets = extract_research_facets(proposal)
    assert "we propose" not in [w.lower() for w in facets["general_keywords"]]
    assert any("graph neural network" in m.lower() or "contrastive learning" in m.lower() for m in facets["method"])


def test_build_github_search_queries_precision():
    """Queries should use quotes for multi-word phrases and include qualifiers."""
    proposal = (
        "Problem: single-cell RNA sequencing. "
        "Method: diffusion models. "
        "Dataset: Cora."
    )
    facets = extract_research_facets(proposal)
    queries = build_github_search_queries(proposal, facets)
    assert len(queries) > 0
    for q in queries:
        assert len(q) <= 120
        assert "in:name,description,topics" in q
        # Multi-word phrase should be quoted
        assert '"' in q


def test_relevance_scoring_and_unrelated_rejection():
    """High-star unrelated repositories matching only generic words must be rejected."""
    query = "Problem: single-cell RNA sequencing. Method: diffusion models."
    facets = extract_research_facets(query)

    # 1. Unrelated popular repo that only has generic words
    unrelated_repo = {
        "id": 1,
        "name": "awesome-deep-learning",
        "full_name": "ai/awesome-deep-learning",
        "description": "A curated list of deep learning models and data pipelines",
        "topics": ["deep-learning", "machine-learning"],
        "stargazers_count": 95000,
        "fork": False,
        "archived": False,
    }
    score_unrelated, matched_unrelated = calculate_repository_relevance(unrelated_repo, facets, query)
    assert score_unrelated < 2.0
    assert len(matched_unrelated) == 0

    # 2. Highly relevant repo with specific core problem and method match
    relevant_repo = {
        "id": 2,
        "name": "sc-diffusion-rna",
        "full_name": "lab/sc-diffusion-rna",
        "description": "Diffusion models for single-cell RNA sequencing denoising",
        "topics": ["diffusion-models", "single-cell", "rna-seq"],
        "stargazers_count": 35,
        "fork": False,
        "archived": False,
    }
    score_relevant, matched_relevant = calculate_repository_relevance(relevant_repo, facets, query)
    assert score_relevant >= 2.5
    assert any("diffusion" in m.lower() for m in matched_relevant)
    assert any("single-cell" in m.lower() or "rna" in m.lower() for m in matched_relevant)


@patch("app.github.requests.get")
def test_search_github_repositories_filters_irrelevant(mock_get):
    """search_github_repositories should only return relevant repos above threshold."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "items": [
            {
                "id": 101,
                "name": "huge-unrelated-repo",
                "full_name": "bigtech/huge-unrelated-repo",
                "html_url": "https://github.com/bigtech/huge-unrelated-repo",
                "description": "General machine learning model library and system",
                "language": "Python",
                "stargazers_count": 80000,
                "forks_count": 10000,
                "updated_at": "2024-05-10T12:00:00Z",
                "topics": ["machine-learning"],
                "fork": False,
                "archived": False,
            },
            {
                "id": 102,
                "name": "sc-rna-diffusion",
                "full_name": "researcher/sc-rna-diffusion",
                "html_url": "https://github.com/researcher/sc-rna-diffusion",
                "description": "Implementation of diffusion models for single-cell RNA sequencing",
                "language": "Python",
                "stargazers_count": 42,
                "forks_count": 5,
                "updated_at": "2024-04-01T10:00:00Z",
                "topics": ["diffusion-models", "single-cell-rna"],
                "fork": False,
                "archived": False,
            },
        ]
    }
    mock_get.return_value = mock_response

    res = search_github_repositories("diffusion models for single-cell RNA sequencing", use_cache=False)
    # The unrelated repo must be filtered out completely
    assert res["count"] == 1
    assert res["repositories"][0]["full_name"] == "researcher/sc-rna-diffusion"
    assert len(res["repositories"][0]["matched_keywords"]) > 0


@patch("app.github.requests.get")
def test_search_github_repositories_empty_honest_message(mock_get):
    """When no repository passes the filter, honest message is displayed."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "items": [
            {
                "id": 201,
                "name": "random-web-tool",
                "full_name": "user/random-web-tool",
                "html_url": "https://github.com/user/random-web-tool",
                "description": "A web scraper for website data",
                "language": "JavaScript",
                "stargazers_count": 100,
                "forks_count": 2,
                "updated_at": "2023-01-01T00:00:00Z",
                "topics": ["web"],
                "fork": False,
                "archived": False,
            }
        ]
    }
    mock_get.return_value = mock_response

    res = search_github_repositories("quantum topological error correction codes", use_cache=False)
    assert res["count"] == 0
    assert len(res["repositories"]) == 0
    assert "No closely related repositories found on GitHub" in res["empty_message"]
    assert "does not prove your idea is novel" in res["empty_message"]


@patch("app.github.requests.get")
def test_fork_and_archive_handling(mock_get):
    """Forks are dropped if non-fork matches exist, but kept if only match; archived repos are flagged."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "items": [
            {
                "id": 301,
                "name": "original-sc-diffusion",
                "full_name": "origin/original-sc-diffusion",
                "html_url": "https://github.com/origin/original-sc-diffusion",
                "description": "Diffusion models for single-cell RNA sequencing",
                "language": "Python",
                "stargazers_count": 50,
                "forks_count": 10,
                "updated_at": "2024-01-01T00:00:00Z",
                "topics": ["diffusion-models"],
                "fork": False,
                "archived": True,
            },
            {
                "id": 302,
                "name": "forked-sc-diffusion",
                "full_name": "user/forked-sc-diffusion",
                "html_url": "https://github.com/user/forked-sc-diffusion",
                "description": "Diffusion models for single-cell RNA sequencing fork",
                "language": "Python",
                "stargazers_count": 1,
                "forks_count": 0,
                "updated_at": "2024-01-02T00:00:00Z",
                "topics": ["diffusion-models"],
                "fork": True,
                "archived": False,
            },
        ]
    }
    mock_get.return_value = mock_response

    res = search_github_repositories("diffusion models for single-cell RNA sequencing", use_cache=False)
    assert res["count"] == 1
    # Fork dropped because non-fork existed
    assert res["repositories"][0]["full_name"] == "origin/original-sc-diffusion"
    # Archived tag preserved
    assert res["repositories"][0]["archived"] is True


@patch("app.github.requests.get")
def test_search_github_repositories_rate_limited(mock_get):
    """Rate limit 403 / 429 returns friendly error message."""
    mock_response = MagicMock()
    mock_response.status_code = 403
    mock_response.headers = {"X-RateLimit-Reset": "1800000000"}
    mock_get.return_value = mock_response

    res = search_github_repositories("single-cell rna", use_cache=False)
    assert res["rate_limited"] is True
    assert "rate limit reached" in res["error"].lower()
    assert res["count"] == 0


def test_api_github_search_route(client):
    """Test the POST /api/github/search endpoint."""
    with patch("app.server.search_github_repositories") as mock_search:
        mock_search.return_value = {
            "query": "graph link prediction",
            "facets": {"problem": ["link prediction"], "method": ["gnn"], "dataset": [], "evaluation": []},
            "searched_queries": ['"gnn" "link prediction" in:name,description,topics'],
            "repositories": [
                {
                    "id": 1,
                    "name": "gnn-link-pred",
                    "full_name": "ai/gnn-link-pred",
                    "url": "https://github.com/ai/gnn-link-pred",
                    "description": "Link prediction with GNNs",
                    "language": "Python",
                    "stars": 200,
                    "forks": 30,
                    "updated_at": "2024-02-15",
                    "topics": ["gnn"],
                    "matched_keywords": ["gnn", "link prediction"],
                    "archived": False,
                }
            ],
            "count": 1,
            "rate_limited": False,
            "error": None,
            "disclaimer": "Supporting evidence notice",
        }

        resp = client.post("/api/github/search", json={"query": "graph link prediction", "max_results": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["repositories"][0]["full_name"] == "ai/gnn-link-pred"
        assert data["rate_limited"] is False


def test_api_github_search_empty_query(client):
    """Test that empty queries return 400."""
    resp = client.post("/api/github/search", json={"query": "   "})
    assert resp.status_code == 400
