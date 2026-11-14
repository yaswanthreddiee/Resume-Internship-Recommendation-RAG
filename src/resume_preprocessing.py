"""
Module 1: Resume Preprocessing Pipeline
Provides functions to clean, normalize, validate, and structure candidate resume data,
and generate high-quality embedding-ready text.
"""

import json
import re
from typing import Dict, List, Any, Tuple, Union, Optional
from pydantic import BaseModel, Field, field_validator


# Canonical skill normalization map
SKILL_NORMALIZATION_MAP: Dict[str, str] = {
    # Programming Languages
    "python": "Python",
    "py": "Python",
    "python3": "Python",
    "javascript": "JavaScript",
    "js": "JavaScript",
    "typescript": "TypeScript",
    "ts": "TypeScript",
    "java": "Java",
    "c++": "C++",
    "cpp": "C++",
    "c": "C",
    "c#": "C#",
    "csharp": "C#",
    "go": "Go",
    "golang": "Go",
    "rust": "Rust",
    "r": "R",
    "ruby": "Ruby",
    "php": "PHP",
    "sql": "SQL",
    "html": "HTML5",
    "html5": "HTML5",
    "css": "CSS3",
    "css3": "CSS3",
    "dart": "Dart",
    "kotlin": "Kotlin",
    "swift": "Swift",
    "matlab": "MATLAB",
    "verilog": "Verilog",
    "vhdl": "VHDL",

    # Frameworks & Libraries
    "django": "Django",
    "drf": "Django REST Framework",
    "django rest framework": "Django REST Framework",
    "django rest": "Django REST Framework",
    "fastapi": "FastAPI",
    "flask": "Flask",
    "react": "React",
    "reactjs": "React",
    "react.js": "React",
    "next": "Next.js",
    "nextjs": "Next.js",
    "next.js": "Next.js",
    "vue": "Vue.js",
    "vuejs": "Vue.js",
    "vue.js": "Vue.js",
    "angular": "Angular",
    "angularjs": "Angular",
    "spring": "Spring Boot",
    "spring boot": "Spring Boot",
    "springboot": "Spring Boot",
    "spring data jpa": "Spring Data JPA",
    "hibernate": "Hibernate",
    "express": "Express.js",
    "expressjs": "Express.js",
    "express.js": "Express.js",
    "node": "Node.js",
    "nodejs": "Node.js",
    "node.js": "Node.js",
    "pytorch": "PyTorch",
    "torch": "PyTorch",
    "tensorflow": "TensorFlow",
    "tf": "TensorFlow",
    "keras": "Keras",
    "scikit-learn": "Scikit-Learn",
    "scikit learn": "Scikit-Learn",
    "sklearn": "Scikit-Learn",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "langchain": "LangChain",
    "llamaindex": "LlamaIndex",
    "llama-index": "LlamaIndex",
    "huggingface": "HuggingFace",
    "hugging face": "HuggingFace",
    "transformers": "Transformers",
    "opencv": "OpenCV",
    "cv2": "OpenCV",
    "pandas": "Pandas",
    "numpy": "NumPy",
    "scipy": "SciPy",
    "statsmodels": "Statsmodels",
    "tailwind": "Tailwind CSS",
    "tailwindcss": "Tailwind CSS",
    "bootstrap": "Bootstrap",
    "redux": "Redux",
    "redux toolkit": "Redux Toolkit",
    "celery": "Celery",
    "pydantic": "Pydantic",
    "sqlalchemy": "SQLAlchemy",

    # Tools & Platforms
    "git": "Git",
    "github": "GitHub",
    "gitlab": "GitLab",
    "docker": "Docker",
    "kubernetes": "Kubernetes",
    "k8s": "Kubernetes",
    "linux": "Linux",
    "bash": "Bash",
    "aws": "AWS",
    "amazon web services": "AWS",
    "gcp": "Google Cloud",
    "google cloud platform": "Google Cloud",
    "azure": "Azure",
    "postgresql": "PostgreSQL",
    "postgres": "PostgreSQL",
    "mysql": "MySQL",
    "mongodb": "MongoDB",
    "redis": "Redis",
    "chromadb": "ChromaDB",
    "chroma": "ChromaDB",
    "postman": "Postman",
    "tableau": "Tableau",
    "powerbi": "PowerBI",
    "power bi": "PowerBI",
    "excel": "Excel",
    "ms excel": "Excel",
    "jira": "Jira",
    "figma": "Figma",
    "pytest": "Pytest",
    "junit": "JUnit",
    "maven": "Maven",
    "gradle": "Gradle",
    "kafka": "Kafka",
    "apache kafka": "Kafka"
}


def normalize_skill(skill: Any) -> str:
    """Normalizes a single skill string against canonical vocabulary."""
    if not skill or not isinstance(skill, str):
        return ""
    cleaned = skill.strip()
    key = cleaned.lower()
    return SKILL_NORMALIZATION_MAP.get(key, cleaned)


def clean_string(val: Any) -> str:
    """Clean arbitrary values into a safe trimmed string without null characters."""
    if val is None:
        return ""
    s = str(val).strip()
    # Remove control characters
    s = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', s)
    return s


def clean_skill_list(skills: Any) -> List[str]:
    """Cleans, normalizes, and deduplicates a list of skills while preserving order."""
    if not skills:
        return []
    if isinstance(skills, str):
        # Allow comma/semicolon/pipe separated strings
        raw_list = [item.strip() for item in re.split(r'[,|;\n]+', skills) if item.strip()]
    elif isinstance(skills, (list, tuple, set)):
        raw_list = [clean_string(s) for s in skills if clean_string(s)]
    else:
        return []

    seen = set()
    normalized_list = []
    for s in raw_list:
        norm = normalize_skill(s)
        if norm and norm.lower() not in seen:
            seen.add(norm.lower())
            normalized_list.append(norm)
    return normalized_list


def clean_education_data(edu: Any) -> Dict[str, Any]:
    """Clean and validate education dictionary."""
    if not isinstance(edu, dict):
        return {
            "degree": "",
            "specialization": "",
            "college": "",
            "graduation_year": None
        }

    grad_year = edu.get("graduation_year")
    try:
        if grad_year is not None and str(grad_year).strip():
            grad_year = int(str(grad_year).strip())
        else:
            grad_year = None
    except (ValueError, TypeError):
        grad_year = None

    return {
        "degree": clean_string(edu.get("degree")),
        "specialization": clean_string(edu.get("specialization")),
        "college": clean_string(edu.get("college")),
        "graduation_year": grad_year
    }


def clean_projects_data(projects: Any) -> List[Dict[str, Any]]:
    """Clean and validate project list."""
    if not isinstance(projects, list):
        return []
    cleaned_projects = []
    for p in projects:
        if isinstance(p, dict):
            cleaned_projects.append({
                "title": clean_string(p.get("title")),
                "description": clean_string(p.get("description")),
                "technologies": clean_skill_list(p.get("technologies", []))
            })
        elif isinstance(p, str) and p.strip():
            cleaned_projects.append({
                "title": p.strip(),
                "description": "",
                "technologies": []
            })
    return cleaned_projects


def clean_experience_data(experiences: Any) -> List[Dict[str, Any]]:
    """Clean and validate internship or work experience list."""
    if not isinstance(experiences, list):
        return []
    cleaned_exp = []
    for exp in experiences:
        if isinstance(exp, dict):
            cleaned_exp.append({
                "role": clean_string(exp.get("role")),
                "company": clean_string(exp.get("company")),
                "duration": clean_string(exp.get("duration")),
                "description": clean_string(exp.get("description"))
            })
        elif isinstance(exp, str) and exp.strip():
            cleaned_exp.append({
                "role": exp.strip(),
                "company": "",
                "duration": "",
                "description": ""
            })
    return cleaned_exp


def prepare_embedding_text(candidate: Dict[str, Any]) -> str:
    """
    Constructs an informative, structured text representation of the candidate
    optimized for dense vector semantic similarity with internship descriptions.
    """
    parts = []

    # 1. Headline / Candidate Identity
    name = candidate.get("candidate_name", "").strip()
    if name:
        parts.append(f"Candidate Profile: {name}.")

    # 2. Education Details
    edu = candidate.get("education", {})
    degree = edu.get("degree", "").strip()
    spec = edu.get("specialization", "").strip()
    college = edu.get("college", "").strip()
    year = edu.get("graduation_year")

    edu_str = []
    if degree:
        edu_str.append(degree)
    if spec:
        edu_str.append(f"in {spec}")
    if college:
        edu_str.append(f"from {college}")
    if year:
        edu_str.append(f"(Class of {year})")

    if edu_str:
        parts.append(f"Education: {' '.join(edu_str)}.")

    # 3. Technical Skills Breakdown
    skills = candidate.get("technical_skills", {})
    langs = skills.get("programming_languages", [])
    frameworks = skills.get("frameworks", [])
    tools = skills.get("tools", [])

    all_skills_combined = clean_skill_list(langs + frameworks + tools)
    if all_skills_combined:
        skill_parts = []
        if langs:
            skill_parts.append(f"Programming Languages: {', '.join(langs)}")
        if frameworks:
            skill_parts.append(f"Frameworks and Libraries: {', '.join(frameworks)}")
        if tools:
            skill_parts.append(f"Tools and Databases: {', '.join(tools)}")
        parts.append(f"Technical Competencies: {'; '.join(skill_parts)}.")

    # 4. Projects Breakdown
    projects = candidate.get("projects", [])
    if projects:
        proj_texts = []
        for p in projects:
            p_title = p.get("title", "").strip()
            p_desc = p.get("description", "").strip()
            p_techs = p.get("technologies", [])
            tech_clause = f" using {', '.join(p_techs)}" if p_techs else ""
            if p_title and p_desc:
                proj_texts.append(f"{p_title}: {p_desc}{tech_clause}")
            elif p_title:
                proj_texts.append(f"{p_title}{tech_clause}")
            elif p_desc:
                proj_texts.append(f"{p_desc}{tech_clause}")
        if proj_texts:
            parts.append(f"Projects and Practical Work: {' | '.join(proj_texts)}.")

    # 5. Internship and Work Experience
    internships = candidate.get("internship_experience", [])
    work_exp = candidate.get("work_experience", [])
    exp_entries = internships + work_exp
    if exp_entries:
        exp_texts = []
        for e in exp_entries:
            role = e.get("role", "").strip()
            company = e.get("company", "").strip()
            dur = e.get("duration", "").strip()
            desc = e.get("description", "").strip()
            entry = f"{role}"
            if company:
                entry += f" at {company}"
            if dur:
                entry += f" ({dur})"
            if desc:
                entry += f": {desc}"
            exp_texts.append(entry)
        if exp_texts:
            parts.append(f"Prior Experience: {' | '.join(exp_texts)}.")

    # 6. Certifications
    certs = candidate.get("certifications", [])
    if certs:
        cleaned_certs = [c.strip() for c in certs if isinstance(c, str) and c.strip()]
        if cleaned_certs:
            parts.append(f"Certifications and Honors: {', '.join(cleaned_certs)}.")

    # 7. Other Relevant Info
    other = candidate.get("other_information", "").strip()
    if other:
        parts.append(f"Additional Notes: {other}")

    # Fallback if profile is completely empty
    full_text = " ".join(parts).strip()
    if not full_text:
        full_text = "Empty candidate profile with no listed skills or project details."

    return full_text


def preprocess_resume(resume_data: Union[Dict[str, Any], str]) -> Tuple[Dict[str, Any], str]:
    """
    Core reusable preprocessing function for candidate resume data.

    Accepts raw resume data as a dictionary or JSON string.
    Performs data cleaning, skill normalization, deduplication, null handling,
    and returns a tuple of (cleaned_data_dict, embedding_ready_text).
    """
    if isinstance(resume_data, str):
        try:
            raw = json.loads(resume_data)
        except json.JSONDecodeError as e:
            # Fallback for plain text input
            raw = {
                "candidate_name": "Applicant",
                "other_information": resume_data.strip()
            }
    elif isinstance(resume_data, dict):
        raw = resume_data
    else:
        raw = {}

    # Extract & clean candidate metadata
    candidate_id = clean_string(raw.get("candidate_id", "CAND-USER"))
    name = clean_string(raw.get("candidate_name") or raw.get("name") or "Anonymous Candidate")
    email = clean_string(raw.get("email", ""))

    # Extract & clean skills
    raw_skills = raw.get("technical_skills", {})
    if isinstance(raw_skills, dict):
        langs = clean_skill_list(raw_skills.get("programming_languages", []))
        frameworks = clean_skill_list(raw_skills.get("frameworks", []))
        tools = clean_skill_list(raw_skills.get("tools", []))
    elif isinstance(raw_skills, list):
        # Flat skills list
        langs = []
        frameworks = []
        tools = clean_skill_list(raw_skills)
    else:
        langs, frameworks, tools = [], [], []

    # Collect any extra top-level skills lists if present
    if "skills" in raw and isinstance(raw["skills"], list):
        extra = clean_skill_list(raw["skills"])
        for item in extra:
            if item not in langs and item not in frameworks and item not in tools:
                tools.append(item)

    cleaned_skills = {
        "programming_languages": clean_skill_list(langs),
        "frameworks": clean_skill_list(frameworks),
        "tools": clean_skill_list(tools)
    }

    # Clean education
    cleaned_edu = clean_education_data(raw.get("education", {}))

    # Clean projects
    cleaned_projects = clean_projects_data(raw.get("projects", []))

    # Clean experiences
    cleaned_internships = clean_experience_data(raw.get("internship_experience", []))
    cleaned_work_exp = clean_experience_data(raw.get("work_experience", []))

    # Clean certifications
    raw_certs = raw.get("certifications", [])
    if isinstance(raw_certs, list):
        cleaned_certs = [clean_string(c) for c in raw_certs if clean_string(c)]
    elif isinstance(raw_certs, str) and raw_certs.strip():
        cleaned_certs = [s.strip() for s in raw_certs.split(",") if s.strip()]
    else:
        cleaned_certs = []

    cleaned_other = clean_string(raw.get("other_information", ""))

    cleaned_data: Dict[str, Any] = {
        "candidate_id": candidate_id,
        "candidate_name": name,
        "email": email,
        "education": cleaned_edu,
        "technical_skills": cleaned_skills,
        "projects": cleaned_projects,
        "internship_experience": cleaned_internships,
        "work_experience": cleaned_work_exp,
        "certifications": cleaned_certs,
        "other_information": cleaned_other
    }

    # Generate rich embedding-ready text
    embedding_text = prepare_embedding_text(cleaned_data)
    cleaned_data["embedding_text"] = embedding_text

    return cleaned_data, embedding_text


def get_all_skills_flat(cleaned_candidate: Dict[str, Any]) -> List[str]:
    """Helper to retrieve all normalized technical skills as a flat deduplicated list."""
    skills_dict = cleaned_candidate.get("technical_skills", {})
    all_s = []
    for k in ["programming_languages", "frameworks", "tools"]:
        all_s.extend(skills_dict.get(k, []))
    return clean_skill_list(all_s)
