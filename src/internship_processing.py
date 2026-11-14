"""
Module 2: Internship Data Processing and Ingestion Pipeline
Handles loading, validation, sanitization, schema enforcement,
metadata formatting, and embedding text generation for internship opportunities.
"""

import json
import logging
import os
import re
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field, field_validator

from src.resume_preprocessing import clean_string, clean_skill_list

logger = logging.getLogger(__name__)


class InternshipRecord(BaseModel):
    """Pydantic model enforcing validation on internship records."""
    internship_id: str = Field(..., description="Unique alphanumeric identifier for the internship")
    title: str = Field(..., min_length=2, description="Job title / role")
    company_name: str = Field(default="Company Confidential")
    category: str = Field(..., description="Primary internship category/domain")
    description: str = Field(..., min_length=5, description="Full description of the internship")
    required_skills: List[str] = Field(default_factory=list)
    preferred_skills: List[str] = Field(default_factory=list)
    education_requirements: str = Field(default="Any relevant degree")
    experience_requirements: str = Field(default="Fresher / Entry Level")
    location: str = Field(default="India")
    work_mode: str = Field(default="Remote")
    duration: str = Field(default="6 months")
    other_requirements: str = Field(default="")
    is_synthetic: bool = Field(default=True)
    provenance: str = Field(default="Synthetic Benchmark")

    @field_validator("work_mode", mode="before")
    @classmethod
    def normalize_work_mode(cls, v: Any) -> str:
        s = str(v).strip().capitalize() if v else "Remote"
        if "remote" in s.lower():
            return "Remote"
        elif "hybrid" in s.lower():
            return "Hybrid"
        elif "site" in s.lower() or "office" in s.lower():
            return "On-site"
        return s or "Remote"


def prepare_internship_embedding_text(record: Dict[str, Any]) -> str:
    """
    Builds structured, information-dense embedding text for an internship.
    Captures role title, company, category, required & preferred skills,
    description, and educational requirements.
    """
    title = clean_string(record.get("title", ""))
    company = clean_string(record.get("company_name", ""))
    category = clean_string(record.get("category", ""))
    desc = clean_string(record.get("description", ""))
    req_skills = clean_skill_list(record.get("required_skills", []))
    pref_skills = clean_skill_list(record.get("preferred_skills", []))
    edu = clean_string(record.get("education_requirements", ""))
    exp = clean_string(record.get("experience_requirements", ""))
    loc = clean_string(record.get("location", ""))
    mode = clean_string(record.get("work_mode", ""))
    dur = clean_string(record.get("duration", ""))
    other = clean_string(record.get("other_requirements", ""))

    parts = [
        f"Internship Opportunity: {title} in {category} at {company}.",
        f"Work Arrangement: {mode} ({loc}), Duration: {dur}." if (mode or loc or dur) else "",
        f"Job Overview and Responsibilities: {desc}." if desc else "",
        f"Required Core Technical Skills: {', '.join(req_skills)}." if req_skills else "",
        f"Preferred Additional Qualifications: {', '.join(pref_skills)}." if pref_skills else "",
        f"Education Criteria: {edu}." if edu else "",
        f"Experience Level: {exp}." if exp else "",
        f"Key Requirements: {other}." if other else ""
    ]

    cleaned_parts = [p.strip() for p in parts if p.strip()]
    return " ".join(cleaned_parts)


def validate_and_clean_internship(raw: Dict[str, Any], idx: int = 0) -> Optional[Dict[str, Any]]:
    """
    Validates a raw internship dictionary against quality criteria.
    Deduplicates skills, handles missing values, and assigns stable fallback IDs.
    Returns None if essential fields (title or description) cannot be salvaged.
    """
    if not isinstance(raw, dict):
        logger.warning(f"Skipping record #{idx}: not a dictionary.")
        return None

    title = clean_string(raw.get("title") or raw.get("internship_title") or "")
    desc = clean_string(raw.get("description") or raw.get("internship_description") or "")
    if not title or not desc:
        logger.warning(f"Record #{idx} missing mandatory title or description. Skipping.")
        return None

    internship_id = clean_string(raw.get("internship_id") or f"INT-GEN-{idx+1:03d}")
    company = clean_string(raw.get("company_name") or raw.get("company") or "Confidential Organization")
    category = clean_string(raw.get("category") or "Software Development")

    req_skills = clean_skill_list(raw.get("required_skills", []))
    pref_skills = clean_skill_list(raw.get("preferred_skills", []))

    # Normalize work mode
    work_mode_raw = clean_string(raw.get("work_mode", "Remote")).capitalize()
    if "remote" in work_mode_raw.lower():
        work_mode = "Remote"
    elif "hybrid" in work_mode_raw.lower():
        work_mode = "Hybrid"
    elif "on-site" in work_mode_raw.lower() or "onsite" in work_mode_raw.lower():
        work_mode = "On-site"
    else:
        work_mode = work_mode_raw or "Remote"

    cleaned: Dict[str, Any] = {
        "internship_id": internship_id,
        "title": title,
        "company_name": company,
        "category": category,
        "description": desc,
        "required_skills": req_skills,
        "preferred_skills": pref_skills,
        "education_requirements": clean_string(raw.get("education_requirements") or "Any quantitative degree"),
        "experience_requirements": clean_string(raw.get("experience_requirements") or "Fresher / Entry level"),
        "location": clean_string(raw.get("location") or "India"),
        "work_mode": work_mode,
        "duration": clean_string(raw.get("duration") or "6 months"),
        "other_requirements": clean_string(raw.get("other_requirements") or ""),
        "is_synthetic": bool(raw.get("is_synthetic", True)),
        "provenance": clean_string(raw.get("provenance") or "Infosys Synthetic Benchmark")
    }

    # Generate embedding text
    cleaned["embedding_text"] = prepare_internship_embedding_text(cleaned)

    return cleaned


def load_and_preprocess_internships(json_filepath: str = "data/internships.json") -> List[Dict[str, Any]]:
    """
    Loads, sanitizes, and deduplicates internship records from a JSON file.
    Guarantees unique internship IDs.
    """
    if not os.path.exists(json_filepath):
        logger.error(f"Internship dataset not found at {json_filepath}")
        return []

    with open(json_filepath, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {json_filepath}: {e}")
            return []

    if not isinstance(data, list):
        logger.error("Expected list of internship records in JSON file.")
        return []

    cleaned_records: List[Dict[str, Any]] = []
    seen_ids = set()

    for idx, raw in enumerate(data):
        rec = validate_and_clean_internship(raw, idx)
        if rec:
            internship_id = rec["internship_id"]
            if internship_id in seen_ids:
                # Disambiguate duplicate ID
                new_id = f"{internship_id}-DUP{idx}"
                logger.info(f"Duplicate internship ID {internship_id} detected. Renamed to {new_id}")
                rec["internship_id"] = new_id
                internship_id = new_id

            seen_ids.add(internship_id)
            cleaned_records.append(rec)

    logger.info(f"Loaded and validated {len(cleaned_records)} unique internship postings.")
    return cleaned_records
