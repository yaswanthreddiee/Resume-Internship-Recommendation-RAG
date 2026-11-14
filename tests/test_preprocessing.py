"""
Unit tests for Module 1: Resume Preprocessing Pipeline
"""

import pytest
from src.resume_preprocessing import (
    preprocess_resume,
    normalize_skill,
    clean_skill_list,
    get_all_skills_flat
)


def test_normalize_skill():
    assert normalize_skill("python3") == "Python"
    assert normalize_skill("drf") == "Django REST Framework"
    assert normalize_skill("react.js") == "React"
    assert normalize_skill("postgres") == "PostgreSQL"
    assert normalize_skill("tf") == "TensorFlow"
    assert normalize_skill("UNKNOWN_SKILL") == "UNKNOWN_SKILL"
    assert normalize_skill("") == ""
    assert normalize_skill(None) == ""


def test_clean_skill_list_deduplication():
    raw = ["Python", "python", "py", "Django", "django", "PostgreSQL", "postgres", "Docker"]
    cleaned = clean_skill_list(raw)
    assert cleaned == ["Python", "Django", "PostgreSQL", "Docker"]


def test_preprocess_resume_standard():
    sample_resume = {
        "candidate_id": "CAND-01",
        "candidate_name": "Aarav Sharma",
        "email": "aarav@example.com",
        "education": {
            "degree": "B.Tech",
            "specialization": "Computer Science",
            "college": "NITK",
            "graduation_year": "2025"
        },
        "technical_skills": {
            "programming_languages": ["py", "python3", "sql"],
            "frameworks": ["django", "drf"],
            "tools": ["postgres", "docker", "git"]
        },
        "projects": [
            {
                "title": "E-Commerce Micro-Backend",
                "description": "Built REST APIs with Django and PostgreSQL",
                "technologies": ["Python", "Django", "PostgreSQL"]
            }
        ],
        "internship_experience": [
            {
                "role": "Backend Intern",
                "company": "Tech Corp",
                "duration": "3 months",
                "description": "Maintained endpoints"
            }
        ],
        "certifications": ["AWS Certified Cloud Practitioner"],
        "other_information": "Backend enthusiast"
    }

    cleaned_data, embedding_text = preprocess_resume(sample_resume)

    assert cleaned_data["candidate_name"] == "Aarav Sharma"
    assert cleaned_data["education"]["graduation_year"] == 2025
    assert cleaned_data["technical_skills"]["programming_languages"] == ["Python", "SQL"]
    assert cleaned_data["technical_skills"]["frameworks"] == ["Django", "Django REST Framework"]
    assert cleaned_data["technical_skills"]["tools"] == ["PostgreSQL", "Docker", "Git"]
    assert "Aarav Sharma" in embedding_text
    assert "Django" in embedding_text
    assert "PostgreSQL" in embedding_text


def test_preprocess_resume_null_and_empty_edge_cases():
    # Empty dictionary
    cleaned_empty, emb_empty = preprocess_resume({})
    assert cleaned_empty["candidate_name"] == "Anonymous Candidate"
    assert cleaned_empty["technical_skills"]["programming_languages"] == []
    assert len(emb_empty) > 0

    # Resume with None fields
    null_resume = {
        "candidate_name": None,
        "education": None,
        "technical_skills": None,
        "projects": None
    }
    cleaned_null, emb_null = preprocess_resume(null_resume)
    assert cleaned_null["candidate_name"] == "Anonymous Candidate"
    assert cleaned_null["education"]["degree"] == ""
    assert cleaned_null["education"]["graduation_year"] is None
    assert cleaned_null["technical_skills"]["programming_languages"] == []


def test_get_all_skills_flat():
    candidate = {
        "technical_skills": {
            "programming_languages": ["Python", "C++"],
            "frameworks": ["PyTorch"],
            "tools": ["Git", "Linux"]
        }
    }
    flat = get_all_skills_flat(candidate)
    assert flat == ["Python", "C++", "PyTorch", "Git", "Linux"]
