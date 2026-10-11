from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class ProjectInput(BaseModel):
    title: str = Field(..., min_length=5, max_length=300)
    abstract: str = Field(..., min_length=20, max_length=3000)
    keywords: Optional[List[str]] = None


class RetrievedItem(BaseModel):
    source: str  # "arxiv" | "github" | "sample"
    title: str
    url: Optional[str] = None
    abstract_or_description: str
    year: Optional[int] = None
    similarity: float = 0.0
    extra: Dict[str, Any] = {}


class Gap(BaseModel):
    description: str
    evidence: List[str]
    severity: str  # "high" | "medium" | "low"


class ImprovedIdea(BaseModel):
    title: str
    description: str
    how_it_fills_gaps: List[str]
    estimated_feasibility: str  # "high" | "medium" | "low"


class NoveltyReport(BaseModel):
    input_title: str
    novelty_score: float  # 0-100
    similarity_score: float  # 0-100 (inverse of novelty)
    interpretation: str
    top_similar_papers: List[RetrievedItem]
    top_similar_repos: List[RetrievedItem]
    research_gaps: List[Gap]
    improved_ideas: List[ImprovedIdea]
    literature_survey_summary: str
    references: List[str]
    method_notes: str
