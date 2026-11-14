"""
Module 6: Retrieval-Augmented Generation (RAG) & Explanation Engine
Provides:
1. Grounded Prompt Engineering with prompt injection guardrails.
2. API-based LLM integration with Google Gemini (gemini-1.5-flash).
3. Robust deterministic fallback explanation engine when API keys are absent.
4. Strict factual adherence: never invents companies, salaries, or eligibility guarantees.
"""

import json
import logging
import os
import re
from typing import Dict, List, Any, Optional, Union
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()


def sanitize_for_prompt(text: Any, max_len: int = 1500) -> str:
    """
    Sanitizes untrusted text to prevent prompt injection or delimiter breakout.
    Replaces prompt delimiter tags and truncates excessive lengths.
    """
    if text is None:
        return ""
    s = str(text)
    # Neutralize XML/tag injection attempts
    s = s.replace("<candidate_profile>", "").replace("</candidate_profile>", "")
    s = s.replace("<internship>", "").replace("</internship>", "")
    s = s.replace("<instructions>", "").replace("</instructions>", "")
    s = s.replace("<system>", "").replace("</system>", "")
    # Remove control characters
    s = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', s)
    return s[:max_len].strip()


def build_deterministic_explanation(candidate: Dict[str, Any], internship: Dict[str, Any]) -> Dict[str, Any]:
    """
    Constructs a deterministic, fully grounded analytical explanation
    based strictly on retrieved data and candidate resume features.
    Used when LLM API keys are unavailable or as an offline fallback.
    """
    int_title = internship.get("title", "Internship Opportunity")
    int_company = internship.get("company_name", "Organization")
    category = internship.get("category", "Technology")
    sim_score = internship.get("similarity_score", 0.0)
    ranking_score = internship.get("overall_ranking_score", 0.0)

    explain_data = internship.get("explainability", {})
    matched_req = explain_data.get("matched_required_skills", [])
    missing_req = explain_data.get("missing_required_skills", [])
    matched_pref = explain_data.get("matched_preferred_skills", [])
    relevant_projs = explain_data.get("relevant_projects", [])

    # 1. Match rationale
    if matched_req:
        match_reason = (
            f"This {category} opportunity matches the candidate due to strong alignment in "
            f"core skills ({', '.join(matched_req[:4])}) with a semantic similarity of {sim_score:.1%} "
            f"and an overall ranking score of {ranking_score:.1%}."
        )
    else:
        match_reason = (
            f"This role was retrieved with a semantic similarity of {sim_score:.1%}. "
            f"While direct required skill overlap is limited, the candidate's general domain background shows potential alignment."
        )

    # 2. Projects relevance summary
    if relevant_projs:
        p_names = [p["title"] for p in relevant_projs if "title" in p]
        proj_summary = f"Demonstrated practical hands-on capability in: {', '.join(p_names)}."
    else:
        proj_summary = "No directly matching projects were identified in the candidate's portfolio for this specific domain."

    # 3. Actionable skill gap / improvement recommendations
    if missing_req:
        skill_gaps = (
            f"To increase competitiveness for this role, the candidate should acquire or document experience with: "
            f"{', '.join(missing_req)}."
        )
    else:
        skill_gaps = (
            "The candidate satisfies all listed required skills. Recommending deep-dive practice into preferred tools "
            f"such as: {', '.join(internship.get('preferred_skills', [])[:3]) or 'advanced production deployments'}."
        )

    # 4. Objective conclusion
    conclusion = (
        "Note: This assessment is based strictly on automated semantic retrieval and profile parsing. "
        "It does not represent a hiring decision, salary estimate, or guarantee of interview selection."
    )

    return {
        "summary": match_reason,
        "matched_skills": matched_req,
        "matched_preferred_skills": matched_pref,
        "missing_skills": missing_req,
        "project_relevance": proj_summary,
        "suggested_improvements": skill_gaps,
        "disclaimer": conclusion,
        "generation_method": "Grounded Deterministic Engine (Fallback - Factual Rule-Based)"
    }


def call_gemini_api(prompt: str, api_key: str, model_name: str = GEMINI_MODEL) -> Optional[str]:
    """Calls Google Gemini Generative AI SDK with exception handling."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.2, "max_output_tokens": 1200}
        )
        if response and response.text:
            return response.text.strip()
    except Exception as e:
        logger.warning(f"Gemini API call failed: {e}. Falling back to deterministic engine.")
    return None


def generate_rag_explanations(
    candidate: Dict[str, Any],
    retrieved_internships: List[Dict[str, Any]],
    api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Generates grounded AI explanations for each retrieved internship recommendation.
    Prefers Gemini LLM if GEMINI_API_KEY is available; otherwise engages the deterministic fallback.

    Args:
        candidate: Preprocessed candidate profile.
        retrieved_internships: List of retrieved internship items from search_matching_internships.
        api_key: Optional Gemini API key override.

    Returns:
        Updated list of internships with attached 'ai_explanation' dictionary.
    """
    active_key = (api_key or GEMINI_API_KEY).strip()
    use_llm = bool(active_key and active_key != "your_gemini_api_key_here")

    # If LLM key is available, attempt grounded synthesis
    if use_llm:
        # Construct grounded prompt
        cand_summary = {
            "name": sanitize_for_prompt(candidate.get("candidate_name")),
            "education": candidate.get("education"),
            "technical_skills": candidate.get("technical_skills"),
            "projects": [
                {"title": sanitize_for_prompt(p.get("title")), "technologies": p.get("technologies", [])}
                for p in candidate.get("projects", [])
            ],
            "certifications": candidate.get("certifications", [])
        }

        internship_summaries = []
        for item in retrieved_internships:
            internship_summaries.append({
                "internship_id": item.get("internship_id"),
                "title": sanitize_for_prompt(item.get("title")),
                "company_name": sanitize_for_prompt(item.get("company_name")),
                "category": sanitize_for_prompt(item.get("category")),
                "similarity_score": item.get("similarity_score"),
                "ranking_score": item.get("overall_ranking_score"),
                "required_skills": item.get("required_skills"),
                "preferred_skills": item.get("preferred_skills"),
                "education_requirements": sanitize_for_prompt(item.get("education_requirements")),
                "experience_requirements": sanitize_for_prompt(item.get("experience_requirements"))
            })

        system_instruction = (
            "You are a strict, objective AI Career Recommendation Evaluator for Infosys. "
            "You must generate an accurate, grounded explanation for each retrieved internship based ONLY on the supplied data.\n"
            "STRICT RULES:\n"
            "1. Use ONLY the supplied candidate data and retrieved internship data.\n"
            "2. Never invent company names, salaries, or eligibility guarantees.\n"
            "3. Explicitly list matched skills and missing skills.\n"
            "4. Keep explanations grounded and professional.\n"
            "5. Return a JSON array where each object contains: 'internship_id', 'summary', 'matched_skills', 'missing_skills', 'project_relevance', 'suggested_improvements'.\n"
        )

        user_prompt = (
            f"{system_instruction}\n\n"
            f"<candidate_profile>\n{json.dumps(cand_summary, indent=2)}\n</candidate_profile>\n\n"
            f"<retrieved_internships>\n{json.dumps(internship_summaries, indent=2)}\n</retrieved_internships>\n\n"
            "Generate JSON output array now:"
        )

        llm_output = call_gemini_api(user_prompt, active_key)

        if llm_output:
            try:
                # Extract JSON array from LLM response
                clean_json_str = llm_output
                if "```json" in clean_json_str:
                    clean_json_str = clean_json_str.split("```json")[1].split("```")[0].strip()
                elif "```" in clean_json_str:
                    clean_json_str = clean_json_str.split("```")[1].split("```")[0].strip()

                parsed_exps = json.loads(clean_json_str)
                if isinstance(parsed_exps, list):
                    exp_by_id = {item.get("internship_id"): item for item in parsed_exps if isinstance(item, dict)}
                    for int_item in retrieved_internships:
                        iid = int_item.get("internship_id")
                        if iid in exp_by_id:
                            llm_exp = exp_by_id[iid]
                            llm_exp["generation_method"] = f"Gemini LLM ({GEMINI_MODEL})"
                            llm_exp["disclaimer"] = "Automated advisory explanation grounded on retrieved context."
                            int_item["ai_explanation"] = llm_exp
                        else:
                            int_item["ai_explanation"] = build_deterministic_explanation(candidate, int_item)
                    return retrieved_internships
            except Exception as parse_err:
                logger.warning(f"Failed to parse LLM JSON: {parse_err}. Reverting to deterministic generator.")

    # Fallback to deterministic grounded explanation for each item
    for int_item in retrieved_internships:
        int_item["ai_explanation"] = build_deterministic_explanation(candidate, int_item)

    return retrieved_internships
