"""Unit tests for litsearch leakage guard and temporal constraints."""
import pytest
from datetime import date
from litsearch.leakage import (
    TemporalLeakageError,
    assert_no_leakage,
    assert_no_citation_count_signal,
    filter_candidates_pre_cutoff,
    parse_date,
)

def test_parse_date():
    assert parse_date("2024-03-15") == date(2024, 3, 15)
    assert parse_date("2023") == date(2023, 12, 31)

def test_assert_no_leakage_passes_for_strictly_earlier():
    # Candidate published 2023-01-15, Target published 2023-06-20 -> No error
    assert_no_leakage(
        target_publication_date="2023-06-20",
        candidate_publication_date="2023-01-15",
        experiment_mode=True
    )

def test_assert_no_leakage_raises_for_future_date():
    # Candidate published 2023-08-01, Target published 2023-06-20 -> Must raise TemporalLeakageError
    with pytest.raises(TemporalLeakageError, match="LEAKAGE DETECTED"):
        assert_no_leakage(
            target_publication_date="2023-06-20",
            candidate_publication_date="2023-08-01",
            candidate_id="W999",
            experiment_mode=True
        )

def test_assert_no_leakage_raises_for_same_day_when_strict():
    with pytest.raises(TemporalLeakageError, match="LEAKAGE DETECTED"):
        assert_no_leakage(
            target_publication_date="2023-06-20",
            candidate_publication_date="2023-06-20",
            allow_same_day=False,
            experiment_mode=True
        )

def test_assert_no_citation_count_signal():
    # If a signal flag indicates cited_by_count was used for ranking in experiment mode, raise error
    with pytest.raises(TemporalLeakageError, match="Citation counts"):
        assert_no_citation_count_signal(
            {"id": "W1", "cited_by_count": 42, "used_for_ranking": True},
            experiment_mode=True
        )

def test_filter_candidates_pre_cutoff():
    target_date = "2023-07-01"
    candidates = [
        {"work_id": "W1", "publication_date": "2022-05-10", "cited_by_count": 50},
        {"work_id": "W2", "publication_date": "2023-06-30", "cited_by_count": 12},
        {"work_id": "W3", "publication_date": "2023-07-01", "cited_by_count": 8},
        {"work_id": "W4", "publication_date": "2024-01-15", "cited_by_count": 99},
    ]
    filtered = filter_candidates_pre_cutoff(target_date, candidates, experiment_mode=True)
    assert len(filtered) == 2
    assert [c["work_id"] for c in filtered] == ["W1", "W2"]
    # Citation counts must be masked in experiment mode
    assert all(c["cited_by_count"] is None for c in filtered)
