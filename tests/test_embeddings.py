"""
Unit tests for Module 3: Embedding Generator
"""

import pytest
import numpy as np
from src.embedding_generator import (
    EmbeddingManager,
    get_embedding_manager,
    generate_candidate_embedding,
    generate_internship_embeddings,
    compute_cosine_similarity
)


def test_embedding_manager_singleton():
    mgr1 = get_embedding_manager()
    mgr2 = get_embedding_manager()
    assert mgr1 is mgr2
    assert mgr1.model is not None


def test_generate_candidate_embedding():
    candidate = {
        "candidate_id": "TEST-01",
        "candidate_name": "Test Python Dev",
        "technical_skills": {
            "programming_languages": ["Python"],
            "frameworks": ["Django"],
            "tools": ["PostgreSQL"]
        },
        "projects": [
            {
                "title": "Django App",
                "description": "Built web app in Django",
                "technologies": ["Python", "Django"]
            }
        ]
    }
    emb = generate_candidate_embedding(candidate)
    assert isinstance(emb, list)
    assert len(emb) == 384
    # Check normalized vector magnitude ~ 1.0
    norm = np.linalg.norm(np.array(emb))
    assert abs(norm - 1.0) < 1e-3


def test_generate_internship_embeddings():
    sample_records = [
        {
            "internship_id": "TEST-INT-01",
            "title": "Python Backend Intern",
            "category": "Backend Development",
            "company_name": "Test Corp",
            "description": "Backend API development using Django and PostgreSQL",
            "required_skills": ["Python", "Django", "PostgreSQL"],
            "preferred_skills": ["Docker"],
            "education_requirements": "B.Tech",
            "experience_requirements": "Fresher",
            "location": "Bengaluru",
            "work_mode": "Remote",
            "duration": "6 months",
            "other_requirements": "",
            "is_synthetic": True,
            "provenance": "Unit Test"
        }
    ]
    records, embs = generate_internship_embeddings(sample_records, batch_size=1)
    assert len(records) == 1
    assert len(embs) == 1
    assert len(embs[0]) == 384


def test_compute_cosine_similarity():
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    v3 = [0.0, 1.0, 0.0]
    v_zero = [0.0, 0.0, 0.0]

    assert abs(compute_cosine_similarity(v1, v2) - 1.0) < 1e-5
    assert abs(compute_cosine_similarity(v1, v3) - 0.0) < 1e-5
    assert compute_cosine_similarity(v1, v_zero) == 0.0
