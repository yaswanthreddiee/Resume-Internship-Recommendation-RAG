"""
Module 5: Candidate-to-Internship Semantic Matching and Hybrid Retrieval Engine
Implements:
1. Dense semantic search via ChromaDB (cosine similarity).
2. Keyword skill-overlap scoring with normalized skill mapping.
3. Hybrid dense-sparse retrieval combining vector and lexical signals.
4. Multi-factor Weighted Matching Engine:
   - Technical Skills (40%)
   - Project Relevance (25%)
   - Education Relevance (15%)
   - Experience Relevance (15%)
   - Certifications (5%)
5. Second-stage re-ranking across top candidate pool.
6. Optional metadata filtering (Category, Location, Work Mode, Duration).
7. Comprehensive explainability feature breakdown.
"""

import logging
import math
import re
import time
from typing import Dict, List, Any, Optional, Tuple, Union

from src.resume_preprocessing import (
    preprocess_resume,
    get_all_skills_flat,
    clean_skill_list,
    normalize_skill
)
from src.embedding_generator import (
    generate_candidate_embedding,
    compute_cosine_similarity
)
from src.vector_database import (
    get_or_create_collection,
    deserialize_metadata_from_chroma
)

logger = logging.getLogger(__name__)

# Default weights for Hybrid Search
DEFAULT_SEMANTIC_WEIGHT = 0.65
DEFAULT_KEYWORD_WEIGHT = 0.35

# Default weights for Multi-Factor Weighted Matching Engine
WEIGHT_TECHNICAL_SKILLS = 0.40
WEIGHT_PROJECT_RELEVANCE = 0.25
WEIGHT_EDUCATION = 0.15
WEIGHT_EXPERIENCE = 0.15
WEIGHT_CERTIFICATIONS = 0.05


def calculate_skill_overlap(
    candidate_skills: List[str],
    required_skills: List[str],
    preferred_skills: List[str]
) -> Dict[str, Any]:
    """
    Computes precise technical skill match breakdown between candidate and internship.
    Distinguishes required skills from preferred skills.
    """
    cand_set = {s.lower() for s in candidate_skills}
    req_set = {s.lower() for s in required_skills}
    pref_set = {s.lower() for s in preferred_skills}

    matched_req = [s for s in required_skills if s.lower() in cand_set]
    missing_req = [s for s in required_skills if s.lower() not in cand_set]
    matched_pref = [s for s in preferred_skills if s.lower() in cand_set]

    # Required ratio: if required skills list is empty, default to 1.0 (no strict barrier)
    req_ratio = len(matched_req) / max(1, len(required_skills)) if required_skills else 1.0
    pref_ratio = len(matched_pref) / max(1, len(preferred_skills)) if preferred_skills else 0.5

    # Composite skill score: 75% required, 25% preferred bonus
    skill_score = min(1.0, (0.75 * req_ratio) + (0.25 * pref_ratio))

    return {
        "matched_required_skills": matched_req,
        "missing_required_skills": missing_req,
        "matched_preferred_skills": matched_pref,
        "required_match_ratio": round(req_ratio, 4),
        "preferred_match_ratio": round(pref_ratio, 4),
        "technical_skill_score": round(skill_score, 4)
    }


def calculate_project_relevance(candidate_projects: List[Dict[str, Any]], internship: Dict[str, Any]) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Evaluates relevance of candidate projects against internship title, description, and skills.
    Returns project relevance score in [0.0, 1.0] and list of relevant matching projects.
    """
    if not candidate_projects:
        # Absence of projects is not an automatic complete penalty if candidate has skills
        return 0.2, []

    int_text = f"{internship.get('title', '')} {internship.get('category', '')} {internship.get('description', '')}".lower()
    int_skills = {s.lower() for s in internship.get("required_skills", []) + internship.get("preferred_skills", [])}

    matching_projects = []
    total_proj_score = 0.0

    for proj in candidate_projects:
        p_title = proj.get("title", "")
        p_desc = proj.get("description", "")
        p_techs = [s.lower() for s in proj.get("technologies", [])]

        # Calculate technology overlap
        tech_overlap = len(set(p_techs).intersection(int_skills))
        # Keyword context match
        text_match = sum(1 for w in int_skills if w in p_desc.lower() or w in p_title.lower())

        relevance = 0.0
        if tech_overlap > 0:
            relevance += min(0.6, 0.25 * tech_overlap)
        if text_match > 0:
            relevance += min(0.4, 0.15 * text_match)

        # Baseline credit for having documented practical project
        relevance = min(1.0, relevance + 0.2)

        if relevance >= 0.3:
            matching_projects.append({
                "title": p_title,
                "relevance_score": round(relevance, 2),
                "matched_technologies": [t for t in proj.get("technologies", []) if t.lower() in int_skills]
            })
        total_proj_score += relevance

    avg_score = total_proj_score / max(1, len(candidate_projects))
    final_score = min(1.0, max(0.2, avg_score))
    return round(final_score, 4), matching_projects


def calculate_education_relevance(candidate_edu: Dict[str, Any], internship: Dict[str, Any]) -> float:
    """
    Assesses degree and quantitative background alignment.
    Does not disqualify non-traditional candidates; provides progressive matching.
    """
    degree = candidate_edu.get("degree", "").lower()
    spec = candidate_edu.get("specialization", "").lower()
    int_edu = internship.get("education_requirements", "").lower()

    if not degree and not spec:
        return 0.5  # Neutral fallback for unlisted education

    # High match for CS / IT / Engineering / Data Science quantitative backgrounds
    score = 0.6
    tech_disciplines = ["computer", "data", "information", "artificial", "software", "statistics", "mathematics", "robotics", "electronics", "ece", "it", "cs"]
    if any(d in spec for d in tech_disciplines) or any(d in degree for d in tech_disciplines):
        score += 0.3

    # Check for direct degree alignment (B.Tech, M.Tech, MCA, etc.)
    degrees_kw = ["b.tech", "b.e", "m.tech", "mca", "bca", "m.sc", "b.sc", "ms", "bs"]
    if any(dk in degree for dk in degrees_kw) and any(dk in int_edu for dk in degrees_kw):
        score += 0.1

    return min(1.0, round(score, 4))


def calculate_experience_relevance(candidate: Dict[str, Any], internship: Dict[str, Any]) -> float:
    """
    Evaluates candidate's internship or project history against required experience level.
    """
    internships = candidate.get("internship_experience", [])
    work_exp = candidate.get("work_experience", [])
    int_exp_req = internship.get("experience_requirements", "").lower()

    # If internship welcomes freshers/entry level, everyone gets strong baseline
    is_fresher_friendly = "fresher" in int_exp_req or "entry" in int_exp_req or "0-1" in int_exp_req or not int_exp_req

    total_exps = len(internships) + len(work_exp)
    if total_exps > 0:
        base = 0.85 if is_fresher_friendly else 0.75
        bonus = min(0.15, 0.05 * total_exps)
        return min(1.0, round(base + bonus, 4))
    else:
        # No formal experience listed
        return 0.70 if is_fresher_friendly else 0.40


def calculate_certifications_relevance(candidate_certs: List[str], internship: Dict[str, Any]) -> float:
    """
    Evaluates whether candidate certifications match internship domain.
    """
    if not candidate_certs:
        return 0.4  # Neutral fallback (5% total weight)

    int_text = f"{internship.get('title', '')} {internship.get('category', '')} {internship.get('description', '')}".lower()
    matches = 0
    for cert in candidate_certs:
        cert_lower = cert.lower()
        if any(w in cert_lower for w in ["aws", "cloud", "developer", "data", "machine learning", "deep learning", "python", "java", "sql", "meta", "google"]):
            matches += 1

    return min(1.0, round(0.5 + (0.25 * matches), 4))


def calculate_overall_ranking_score(
    semantic_similarity: float,
    skill_score: float,
    project_score: float,
    education_score: float,
    experience_score: float,
    certifications_score: float,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Calculates composite ranking score across multi-factor criteria.
    Default experimental weights:
    - Technical Skills: 40%
    - Project Relevance: 25%
    - Education Relevance: 15%
    - Experience Relevance: 15%
    - Certifications: 5%
    """
    w_skill = weights.get("skills", WEIGHT_TECHNICAL_SKILLS) if weights else WEIGHT_TECHNICAL_SKILLS
    w_proj = weights.get("projects", WEIGHT_PROJECT_RELEVANCE) if weights else WEIGHT_PROJECT_RELEVANCE
    w_edu = weights.get("education", WEIGHT_EDUCATION) if weights else WEIGHT_EDUCATION
    w_exp = weights.get("experience", WEIGHT_EXPERIENCE) if weights else WEIGHT_EXPERIENCE
    w_cert = weights.get("certifications", WEIGHT_CERTIFICATIONS) if weights else WEIGHT_CERTIFICATIONS

    ranking_score = (
        (w_skill * skill_score) +
        (w_proj * project_score) +
        (w_edu * education_score) +
        (w_exp * experience_score) +
        (w_cert * certifications_score)
    )
    return round(float(ranking_score), 4)


def apply_metadata_filters(internship: Dict[str, Any], filters: Optional[Dict[str, Any]] = None) -> bool:
    """
    Evaluates optional metadata filters against an internship record.
    Returns True if the record passes all specified filters, False otherwise.
    Filters:
    - category: string or list
    - location: string or substring
    - work_mode: Remote / Hybrid / On-site
    - min_duration / max_duration: duration filter
    - required_skills: skill must be present in required/preferred skills
    """
    if not filters:
        return True

    # 1. Category Filter
    if "category" in filters and filters["category"]:
        filter_cat = filters["category"]
        if isinstance(filter_cat, list):
            if internship.get("category") not in filter_cat:
                return False
        elif isinstance(filter_cat, str) and filter_cat.strip() and filter_cat.lower() != "all":
            if internship.get("category", "").lower() != filter_cat.strip().lower():
                return False

    # 2. Work Mode Filter
    if "work_mode" in filters and filters["work_mode"]:
        filter_mode = filters["work_mode"]
        if isinstance(filter_mode, str) and filter_mode.strip() and filter_mode.lower() != "all":
            if internship.get("work_mode", "").lower() != filter_mode.strip().lower():
                return False

    # 3. Location Filter
    if "location" in filters and filters["location"]:
        loc_filter = filters["location"]
        if isinstance(loc_filter, str) and loc_filter.strip() and loc_filter.lower() != "all":
            if loc_filter.strip().lower() not in internship.get("location", "").lower():
                return False

    # 4. Specific Required Skill Filter
    if "skill" in filters and filters["skill"]:
        target_skill = normalize_skill(filters["skill"]).lower()
        int_skills = [s.lower() for s in internship.get("required_skills", []) + internship.get("preferred_skills", [])]
        if target_skill not in int_skills:
            return False

    return True


def search_matching_internships(
    candidate_data: Union[Dict[str, Any], str],
    top_k: int = 5,
    collection=None,
    filters: Optional[Dict[str, Any]] = None,
    use_hybrid: bool = True,
    use_reranking: bool = True,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
    weights_override: Optional[Dict[str, float]] = None
) -> List[Dict[str, Any]]:
    """
    Core semantic candidate-to-internship matching function.

    Workflow:
    1. Receive candidate resume information.
    2. Clean and preprocess candidate data.
    3. Generate 384-dimensional dense candidate embedding.
    4. Query persistent ChromaDB with cosine similarity.
    5. Retrieve candidate pool (up to 20 candidates for hybrid re-ranking).
    6. Apply optional metadata filters.
    7. Calculate technical skill overlap, project relevance, education, and experience signals.
    8. Calculate separate scores for semantic similarity and multi-factor ranking.
    9. Perform second-stage re-ranking.
    10. Return Top-K structured results with detailed explainability metadata.

    Args:
        candidate_data: Raw resume dictionary or JSON string.
        top_k: Number of recommendations to retrieve (e.g. 5 or 10).
        collection: Optional ChromaDB collection reference.
        filters: Optional metadata filters.
        use_hybrid: Whether to combine dense semantic and keyword signals.
        use_reranking: Whether to apply multi-factor re-ranking.
        semantic_weight: Weight for dense vector similarity.
        keyword_weight: Weight for lexical skill overlap.
        weights_override: Optional custom weights for ranking engine.

    Returns:
        List of structured internship recommendation dictionaries.
    """
    start_t = time.perf_counter()

    # 1. Clean candidate data
    cleaned_cand, emb_text = preprocess_resume(candidate_data)
    cand_skills_flat = get_all_skills_flat(cleaned_cand)

    # 2. Generate candidate embedding
    cand_embedding = generate_candidate_embedding(cleaned_cand)

    # 3. Connect to ChromaDB
    if collection is None:
        collection = get_or_create_collection()

    total_in_db = collection.count()
    if total_in_db == 0:
        logger.warning("Internship collection is empty. Returning 0 matches.")
        return []

    # Initial candidate pool size
    initial_top_k = min(total_in_db, max(20, top_k * 2))

    # 4. Dense Retrieval via ChromaDB
    query_results = collection.query(
        query_embeddings=[cand_embedding],
        n_results=initial_top_k,
        include=["metadatas", "documents", "distances"]
    )

    ids = query_results.get("ids", [[]])[0]
    metas = query_results.get("metadatas", [[]])[0]
    distances = query_results.get("distances", [[]])[0]
    docs = query_results.get("documents", [[]])[0]

    retrieved_items: List[Dict[str, Any]] = []

    for i in range(len(ids)):
        internship_id = ids[i]
        meta = metas[i] if i < len(metas) else {}
        dist = distances[i] if i < len(distances) else 1.0
        doc = docs[i] if i < len(docs) else ""

        # Reconstruct native internship dictionary
        internship_rec = deserialize_metadata_from_chroma(meta)
        internship_rec["internship_id"] = internship_id
        internship_rec["embedding_text"] = doc

        # Calculate semantic cosine similarity
        # In ChromaDB cosine space, cosine_distance = 1.0 - cosine_similarity
        # Therefore cosine_similarity = 1.0 - distance
        raw_sim = 1.0 - float(dist)
        # Clamp to [0.0, 1.0]
        semantic_similarity = round(max(0.0, min(1.0, raw_sim)), 4)

        # 5. Apply optional metadata filters
        if not apply_metadata_filters(internship_rec, filters):
            continue

        # 6. Detailed multi-factor analysis
        req_skills = internship_rec.get("required_skills", [])
        pref_skills = internship_rec.get("preferred_skills", [])

        # Skill overlap
        skill_analysis = calculate_skill_overlap(cand_skills_flat, req_skills, pref_skills)
        technical_skill_score = skill_analysis["technical_skill_score"]

        # Project relevance
        proj_score, matched_projects = calculate_project_relevance(
            cleaned_cand.get("projects", []),
            internship_rec
        )

        # Education relevance
        edu_score = calculate_education_relevance(
            cleaned_cand.get("education", {}),
            internship_rec
        )

        # Experience relevance
        exp_score = calculate_experience_relevance(
            cleaned_cand,
            internship_rec
        )

        # Certifications relevance
        cert_score = calculate_certifications_relevance(
            cleaned_cand.get("certifications", []),
            internship_rec
        )

        # Hybrid Score calculation (dense + keyword)
        if use_hybrid:
            hybrid_score = round(
                (semantic_weight * semantic_similarity) + (keyword_weight * technical_skill_score),
                4
            )
        else:
            hybrid_score = semantic_similarity

        # Multi-Factor Weighted Ranking Score
        ranking_score = calculate_overall_ranking_score(
            semantic_similarity=semantic_similarity,
            skill_score=technical_skill_score,
            project_score=proj_score,
            education_score=edu_score,
            experience_score=exp_score,
            certifications_score=cert_score,
            weights=weights_override
        )

        # Build recommendation record
        item = {
            "internship_id": internship_id,
            "title": internship_rec.get("title", ""),
            "company_name": internship_rec.get("company_name", ""),
            "category": internship_rec.get("category", ""),
            "description": internship_rec.get("description", ""),
            "required_skills": req_skills,
            "preferred_skills": pref_skills,
            "education_requirements": internship_rec.get("education_requirements", ""),
            "experience_requirements": internship_rec.get("experience_requirements", ""),
            "location": internship_rec.get("location", ""),
            "work_mode": internship_rec.get("work_mode", ""),
            "duration": internship_rec.get("duration", ""),
            "other_requirements": internship_rec.get("other_requirements", ""),
            "is_synthetic": internship_rec.get("is_synthetic", True),
            "provenance": internship_rec.get("provenance", ""),

            # Distinct score outputs
            "similarity_score": semantic_similarity,       # Pure Cosine Vector Similarity
            "hybrid_retrieval_score": hybrid_score,        # Dense + Keyword Search Score
            "technical_skill_score": technical_skill_score, # Required + Preferred Skill Overlap
            "project_relevance_score": proj_score,         # Project Alignment Score
            "education_relevance_score": edu_score,        # Academic Background Score
            "experience_relevance_score": exp_score,       # Prior Experience Alignment
            "certifications_score": cert_score,            # Certifications Score
            "overall_ranking_score": ranking_score,        # Multi-factor Weighted Overall Score

            # Explainability payload
            "explainability": {
                "matched_required_skills": skill_analysis["matched_required_skills"],
                "missing_required_skills": skill_analysis["missing_required_skills"],
                "matched_preferred_skills": skill_analysis["matched_preferred_skills"],
                "relevant_projects": matched_projects,
                "education_alignment": f"Matches {internship_rec.get('education_requirements', '')}",
                "experience_alignment": f"Evaluated against {internship_rec.get('experience_requirements', '')}"
            }
        }
        retrieved_items.append(item)

    # 7. Sorting / Re-ranking
    if use_reranking:
        # Sort by overall multi-factor ranking score descending, with similarity as secondary key
        retrieved_items.sort(key=lambda x: (x["overall_ranking_score"], x["similarity_score"]), reverse=True)
    elif use_hybrid:
        retrieved_items.sort(key=lambda x: x["hybrid_retrieval_score"], reverse=True)
    else:
        retrieved_items.sort(key=lambda x: x["similarity_score"], reverse=True)

    elapsed_ms = (time.perf_counter() - start_t) * 1000.0
    logger.info(f"Retrieved and ranked {len(retrieved_items)} matching internships in {elapsed_ms:.1f}ms.")

    final_results = retrieved_items[:top_k]
    for rank, res in enumerate(final_results, start=1):
        res["rank"] = rank

    return final_results
