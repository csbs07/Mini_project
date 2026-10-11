"""litsearch: shared literature search and evaluation package."""
__version__ = "0.1.0"

from .config import (
    EXPERIMENT_MODE,
    SPECTER2_MODEL_NAME,
    SPECTER2_COMMIT_HASH,
    SPECTER2_PROXIMITY_ADAPTER,
    SPECTER2_ADHOC_QUERY_ADAPTER,
)
from .leakage import (
    TemporalLeakageError,
    assert_no_leakage,
    assert_no_citation_count_signal,
    filter_candidates_pre_cutoff,
)

__all__ = [
    "EXPERIMENT_MODE",
    "SPECTER2_MODEL_NAME",
    "SPECTER2_COMMIT_HASH",
    "SPECTER2_PROXIMITY_ADAPTER",
    "SPECTER2_ADHOC_QUERY_ADAPTER",
    "TemporalLeakageError",
    "assert_no_leakage",
    "assert_no_citation_count_signal",
    "filter_candidates_pre_cutoff",
]
