"""Temporal leakage guard for pre-registered empirical study.

Rule:
When experiment_mode is ON:
1. Any candidate signal dated after the target's publication date raises TemporalLeakageError.
2. Any usage of post-cutoff citation counts or post-dated signals raises TemporalLeakageError.
"""
from typing import Optional
from datetime import datetime, date

class TemporalLeakageError(RuntimeError):
    """Raised when an experimental pipeline accesses or uses post-cutoff temporal signals."""
    pass

def parse_date(date_str: str) -> date:
    """Parse YYYY-MM-DD or YYYY string into a date object."""
    if not date_str:
        raise ValueError("Empty date string cannot be evaluated for temporal cutoff.")
    if len(date_str) == 4:
        return datetime.strptime(f"{date_str}-12-31", "%Y-%m-%d").date()
    return datetime.strptime(date_str[:10], "%Y-%m-%d").date()

def assert_no_leakage(
    target_publication_date: str,
    candidate_publication_date: str,
    candidate_id: Optional[str] = None,
    allow_same_day: bool = False,
    experiment_mode: bool = True
) -> None:
    """Assert candidate was published strictly before target publication date."""
    if not experiment_mode:
        return
        
    t_date = parse_date(target_publication_date)
    c_date = parse_date(candidate_publication_date)
    
    if allow_same_day:
        if c_date > t_date:
            raise TemporalLeakageError(
                f"LEAKAGE DETECTED: Candidate {candidate_id or ''} (dated {c_date}) "
                f"was published after target cutoff ({t_date})."
            )
    else:
        if c_date >= t_date:
            raise TemporalLeakageError(
                f"LEAKAGE DETECTED: Candidate {candidate_id or ''} (dated {c_date}) "
                f"was published on or after target cutoff ({t_date}). Candidate must precede target."
            )

def assert_no_citation_count_signal(
    candidate: dict,
    experiment_mode: bool = True
) -> None:
    """Assert that citation count signals are never used for ranking/filtering in experiment mode."""
    if not experiment_mode:
        return
    if "cited_by_count" in candidate and candidate.get("used_for_ranking", False):
        raise TemporalLeakageError(
            "LEAKAGE DETECTED: Citation counts (cited_by_count) cannot be used as a ranking or filtering signal in experiment mode."
        )

def filter_candidates_pre_cutoff(
    target_pub_date: str,
    candidates: list,
    experiment_mode: bool = True
) -> list:
    """Filter candidates strictly published before target publication date, asserting no future signal."""
    if not experiment_mode:
        return candidates

    t_date = parse_date(target_pub_date)
    filtered = []
    for c in candidates:
        pub_date_str = c.get("publication_date") or f"{c.get('publication_year')}-12-31"
        c_date = parse_date(pub_date_str)
        if c_date < t_date:
            # Mask citation counts in experiment mode so they cannot accidentally be consumed
            sanitized = dict(c)
            sanitized["cited_by_count"] = None
            filtered.append(sanitized)
    return filtered
