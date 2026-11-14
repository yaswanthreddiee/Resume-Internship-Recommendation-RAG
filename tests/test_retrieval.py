"""
Unit and Integration tests for Module 5: Internship Retrieval and Matching
"""

import pytest
from src.internship_retrieval import (
    calculate_skill_overlap,
    calculate_project_relevance,
    calculate_education_relevance,
    calculate_experience_relevance,
    calculate_overall_ranking_score,
    apply_metadata_filters,
    search_matching_internships
)
from src.vector_database import init_vector_database, get_or_create_collection


@pytest.fixture(scope="module")
def initialized_db():
    collection, count = init_vector_database(json_filepath="data/internships.json")
    return collection


def test_calculate_skill_overlap():
    cand_skills = ["Python", "Django", "PostgreSQL", "Docker"]
    req_skills = ["Python", "Django", "PostgreSQL"]
    pref_skills = ["Docker", "Redis"]

    res = calculate_skill_overlap(cand_skills, req_skills, pref_skills)
    assert res["matched_required_skills"] == ["Python", "Django", "PostgreSQL"]
    assert res["missing_required_skills"] == []
    assert res["matched_preferred_skills"] == ["Docker"]
    assert res["required_match_ratio"] == 1.0
    assert res["technical_skill_score"] >= 0.85


def test_calculate_skill_overlap_with_missing():
    cand_skills = ["Python"]
    req_skills = ["Python", "Django", "PostgreSQL"]
    pref_skills = ["Redis"]

    res = calculate_skill_overlap(cand_skills, req_skills, pref_skills)
    assert res["matched_required_skills"] == ["Python"]
    assert "Django" in res["missing_required_skills"]
    assert "PostgreSQL" in res["missing_required_skills"]
    assert res["required_match_ratio"] == pytest.approx(1/3, 0.01)


def test_metadata_filtering():
    sample_internship = {
        "internship_id": "INT-AI-001",
        "title": "Computer Vision Intern",
        "category": "Artificial Intelligence",
        "location": "Bengaluru, India",
        "work_mode": "Hybrid",
        "duration": "6 months",
        "required_skills": ["Python", "PyTorch"],
        "preferred_skills": ["OpenCV"]
    }

    # Match category
    assert apply_metadata_filters(sample_internship, {"category": "Artificial Intelligence"}) is True
    assert apply_metadata_filters(sample_internship, {"category": "Backend Development"}) is False

    # Match work mode
    assert apply_metadata_filters(sample_internship, {"work_mode": "Hybrid"}) is True
    assert apply_metadata_filters(sample_internship, {"work_mode": "Remote"}) is False

    # Match skill
    assert apply_metadata_filters(sample_internship, {"skill": "PyTorch"}) is True
    assert apply_metadata_filters(sample_internship, {"skill": "Django"}) is False


def test_search_matching_internships_top5(initialized_db):
    candidate = {
        "candidate_id": "TEST-CAND",
        "candidate_name": "Test Python Dev",
        "education": {
            "degree": "B.Tech",
            "specialization": "Computer Science",
            "college": "State University",
            "graduation_year": 2025
        },
        "technical_skills": {
            "programming_languages": ["Python", "SQL"],
            "frameworks": ["Django", "Django REST Framework"],
            "tools": ["PostgreSQL", "Docker", "Git"]
        },
        "projects": [
            {
                "title": "REST API in Django",
                "description": "Built scalable backend services with PostgreSQL",
                "technologies": ["Python", "Django", "PostgreSQL"]
            }
        ]
    }

    results = search_matching_internships(
        candidate,
        top_k=5,
        collection=initialized_db,
        use_hybrid=True,
        use_reranking=True
    )

    assert len(results) == 5
    top_result = results[0]
    assert "internship_id" in top_result
    assert "similarity_score" in top_result
    assert "overall_ranking_score" in top_result
    assert "explainability" in top_result
    # Similarity should be positive
    assert top_result["similarity_score"] > 0.0
    # Top result should be related to Backend or Python
    assert top_result["category"] in ["Backend Development", "Python Development", "Full Stack Development"]
