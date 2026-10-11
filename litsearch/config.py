# litsearch package configuration and schemas
from dataclasses import dataclass, field
from datetime import date
from typing import Optional, List, Dict, Any
import os

EXPERIMENT_MODE: bool = os.getenv("EXPERIMENT_MODE", "true").lower() in ("true", "1", "yes")
DEFAULT_MODEL: str = os.getenv("LLM_MODEL", "claude-haiku-5-5")
SPECTER2_MODEL_NAME: str = "allenai/specter2_base"
# Pinned SPECTER2 base model revision/commit hash for strict reproducibility
SPECTER2_COMMIT_HASH: str = "e1136c1e550c60ba4cb9b8e96bfad2f91df0f023"

# Adapter names per model card
SPECTER2_PROXIMITY_ADAPTER: str = "allenai/specter2"
SPECTER2_ADHOC_QUERY_ADAPTER: str = "allenai/specter2_adhoc_query"

@dataclass
class TargetPaper:
    target_id: str
    title: str
    abstract: str
    publication_date: str  # YYYY-MM-DD
    publication_year: int
    topic_id: str
    references: List[str] = field(default_factory=list)

@dataclass
class CandidateWork:
    work_id: str
    title: str
    abstract: str
    publication_date: str  # YYYY-MM-DD
    publication_year: int
    cited_by_count: Optional[int] = None
    references: List[str] = field(default_factory=list)
    doi: Optional[str] = None
    venue: Optional[str] = None
    authors: List[str] = field(default_factory=list)
