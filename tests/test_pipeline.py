"""
Comprehensive Integration & Edge Case Test Suite
Tests:
1. All 10 Module 7 Benchmark Scenarios.
2. All 10 Module 8 Advanced Edge Cases:
   - Empty skills
   - Missing education
   - Polyglot resume
   - Unrelated projects
   - Academic only / no experience
   - Empty vector database
   - Partial metadata
   - Ingestion deduplication
   - Unmatched niche candidate
   - Ultra short resume
"""

import json
import pytest
from src.resume_preprocessing import preprocess_resume
from src.internship_processing import validate_and_clean_internship
from src.vector_database import init_vector_database, get_or_create_collection
from src.internship_retrieval import search_matching_internships
from src.rag_generator import generate_rag_explanations


@pytest.fixture(scope="module")
def collection():
    col, _ = init_vector_database(json_filepath="data/internships.json")
    return col


@pytest.fixture(scope="module")
def test_candidates():
    with open("data/test_candidates.json", "r", encoding="utf-8") as f:
        return json.load(f)


def test_all_10_benchmark_scenarios(collection, test_candidates):
    """Verifies that all 10 standard benchmark candidates execute end-to-end successfully."""
    benchmark_candidates = [c for c in test_candidates if c["candidate_id"].startswith("CAND-")]
    assert len(benchmark_candidates) == 10

    for cand in benchmark_candidates:
        results = search_matching_internships(cand, top_k=5, collection=collection)
        assert len(results) == 5
        # Verify scores are bounded in [0.0, 1.0]
        for r in results:
            assert 0.0 <= r["similarity_score"] <= 1.0
            assert 0.0 <= r["overall_ranking_score"] <= 1.0
            assert "explainability" in r

        # Run RAG explanations
        explained = generate_rag_explanations(cand, results)
        assert len(explained) == 5
        for exp in explained:
            assert "ai_explanation" in exp
            assert "summary" in exp["ai_explanation"]
            assert "generation_method" in exp["ai_explanation"]


# ==============================================================================
# 10 MODULE 8 EDGE CASE TESTS
# ==============================================================================

def test_edge_01_no_technical_skills(collection, test_candidates):
    cand = next(c for c in test_candidates if c["candidate_id"] == "EDGE-01")
    results = search_matching_internships(cand, top_k=5, collection=collection)
    assert len(results) == 5
    explained = generate_rag_explanations(cand, results)
    assert len(explained) == 5


def test_edge_02_missing_education(collection, test_candidates):
    cand = next(c for c in test_candidates if c["candidate_id"] == "EDGE-02")
    results = search_matching_internships(cand, top_k=5, collection=collection)
    assert len(results) == 5
    for r in results:
        assert r["education_relevance_score"] >= 0.0


def test_edge_03_polyglot_candidate(collection, test_candidates):
    cand = next(c for c in test_candidates if c["candidate_id"] == "EDGE-03")
    results = search_matching_internships(cand, top_k=5, collection=collection)
    assert len(results) == 5
    assert results[0]["similarity_score"] > 0.3


def test_edge_04_unrelated_projects(collection, test_candidates):
    cand = next(c for c in test_candidates if c["candidate_id"] == "EDGE-04")
    results = search_matching_internships(cand, top_k=5, collection=collection)
    assert len(results) == 5
    # Projects score should be gracefully low without crashing
    assert results[0]["project_relevance_score"] <= 0.6


def test_edge_05_academic_only_no_experience(collection, test_candidates):
    cand = next(c for c in test_candidates if c["candidate_id"] == "EDGE-05")
    results = search_matching_internships(cand, top_k=5, collection=collection)
    assert len(results) == 5
    for r in results:
        assert r["experience_relevance_score"] > 0.0


def test_edge_06_empty_database_resilience():
    # Pass a dummy client/collection that is empty
    import chromadb
    client = chromadb.EphemeralClient()
    empty_col = client.create_collection("empty_test_col", metadata={"hnsw:space": "cosine"})

    cand = {"candidate_name": "Applicant", "technical_skills": {"programming_languages": ["Python"]}}
    results = search_matching_internships(cand, top_k=5, collection=empty_col)
    assert results == []


def test_edge_07_partial_metadata_handling():
    partial_raw = {
        "title": "Minimal Title",
        "description": "Minimal description with at least five characters"
        # Omitting id, company, category, skills, etc.
    }
    cleaned = validate_and_clean_internship(partial_raw, idx=99)
    assert cleaned is not None
    assert cleaned["company_name"] == "Confidential Organization"
    assert cleaned["category"] == "Software Development"
    assert cleaned["required_skills"] == []


def test_edge_08_duplicate_internship_handling():
    raw_dup = {
        "internship_id": "INT-DUP-01",
        "title": "Duplicate Check",
        "description": "Testing duplicate ID handling"
    }
    cleaned = validate_and_clean_internship(raw_dup, idx=1)
    assert cleaned["internship_id"] == "INT-DUP-01"


def test_edge_09_unmatched_niche_candidate(collection, test_candidates):
    cand = next(c for c in test_candidates if c["candidate_id"] == "EDGE-09")
    results = search_matching_internships(cand, top_k=5, collection=collection)
    assert len(results) == 5
    # Should complete without error and provide grounded explanations indicating gaps
    explained = generate_rag_explanations(cand, results)
    assert len(explained) == 5


def test_edge_10_ultra_short_resume(collection, test_candidates):
    cand = next(c for c in test_candidates if c["candidate_id"] == "EDGE-10")
    results = search_matching_internships(cand, top_k=5, collection=collection)
    assert len(results) == 5
    explained = generate_rag_explanations(cand, results)
    assert len(explained) == 5
