"""
Main CLI Application: Resume-Based Internship Recommendation System (RAG)
Provides an end-to-end command-line workflow for candidate ingestion,
semantic search, hybrid re-ranking, and grounded RAG explanation.
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, Any, Optional

from src.resume_preprocessing import preprocess_resume
from src.internship_processing import load_and_preprocess_internships
from src.vector_database import init_vector_database, get_or_create_collection
from src.internship_retrieval import search_matching_internships
from src.rag_generator import generate_rag_explanations
from src.evaluation import run_benchmark_evaluation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("main")


def print_banner():
    banner = r"""
========================================================================================
     RESUME-BASED INTERNSHIP RECOMMENDATION SYSTEM USING RAG
     Infosys Internship Project Benchmark | Advanced Semantic & Hybrid Matching
========================================================================================
    """
    print(banner)


def format_score_bar(score: float, width: int = 15) -> str:
    """Renders a simple ASCII progress bar for scores between 0.0 and 1.0."""
    filled = int(round(score * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {score*100:5.1f}%"


def display_results_terminal(candidate: Dict[str, Any], recommendations: list):
    print("\n" + "="*90)
    print(f"CANDIDATE PROFILE: {candidate.get('candidate_name', 'Applicant')}")
    edu = candidate.get("education", {})
    print(f"Education: {edu.get('degree', '')} in {edu.get('specialization', '')} from {edu.get('college', '')} ({edu.get('graduation_year', 'N/A')})")
    skills = candidate.get("technical_skills", {})
    all_s = skills.get("programming_languages", []) + skills.get("frameworks", []) + skills.get("tools", [])
    print(f"Skills   : {', '.join(all_s) if all_s else 'None listed'}")
    print("="*90)

    if not recommendations:
        print("\nNo matching internships found matching the criteria.")
        return

    print(f"\nTOP {len(recommendations)} RETRIEVED & RANKED INTERNSHIP OPPORTUNITIES:\n")

    for item in recommendations:
        rank = item.get("rank", 1)
        iid = item.get("internship_id")
        title = item.get("title")
        company = item.get("company_name")
        cat = item.get("category")
        loc = item.get("location")
        mode = item.get("work_mode")
        dur = item.get("duration")

        sim_score = item.get("similarity_score", 0.0)
        rank_score = item.get("overall_ranking_score", 0.0)
        skill_score = item.get("technical_skill_score", 0.0)
        proj_score = item.get("project_relevance_score", 0.0)

        ai_exp = item.get("ai_explanation", {})

        print(f"Rank #{rank} | [{iid}] {title}")
        print(f"Company: {company} | Category: {cat} | Location: {loc} ({mode}) | Duration: {dur}")
        print("-" * 90)
        print(f"  • Semantic Cosine Similarity : {format_score_bar(sim_score)}")
        print(f"  • Technical Skill Overlap    : {format_score_bar(skill_score)}")
        print(f"  • Project Relevance Score    : {format_score_bar(proj_score)}")
        print(f"  • Overall Multi-Factor Rank  : {format_score_bar(rank_score)} (Primary Sorting Signal)")
        print("-" * 90)

        explain = item.get("explainability", {})
        matched_req = explain.get("matched_required_skills", [])
        missing_req = explain.get("missing_required_skills", [])
        matched_pref = explain.get("matched_preferred_skills", [])

        print(f"  [+] Matched Required Skills : {', '.join(matched_req) if matched_req else 'None'}")
        if missing_req:
            print(f"  [-] Missing Required Skills : {', '.join(missing_req)}")
        if matched_pref:
            print(f"  [*] Matched Preferred Skills: {', '.join(matched_pref)}")

        print("\n  [RAG AI MATCH ANALYSIS]")
        print(f"  Summary      : {ai_exp.get('summary', 'N/A')}")
        print(f"  Projects     : {ai_exp.get('project_relevance', 'N/A')}")
        print(f"  Improvements : {ai_exp.get('suggested_improvements', 'N/A')}")
        print(f"  Engine       : {ai_exp.get('generation_method', 'N/A')}")
        print("="*90 + "\n")


def run_pipeline(
    candidate_source: str,
    top_k: int = 5,
    mode: str = "hybrid_rerank",
    category_filter: Optional[str] = None,
    work_mode_filter: Optional[str] = None,
    location_filter: Optional[str] = None
):
    print_banner()

    # 1. Initialize Vector Database
    print("[1/5] Checking and Initializing ChromaDB Vector Store...")
    collection, count = init_vector_database(json_filepath="data/internships.json")
    print(f"      Connected to collection '{collection.name}' ({count} records indexed with cosine metric).")

    # 2. Load Candidate Data
    print(f"[2/5] Loading Candidate Information from: {candidate_source}...")
    candidate_raw = None

    if os.path.exists(candidate_source):
        with open(candidate_source, "r", encoding="utf-8") as f:
            candidate_raw = json.load(f)
    else:
        # Check if candidate_source matches an ID in test_candidates.json
        if os.path.exists("data/test_candidates.json"):
            with open("data/test_candidates.json", "r", encoding="utf-8") as f:
                all_cands = json.load(f)
                matched = [c for c in all_cands if c.get("candidate_id", "").lower() == candidate_source.lower()]
                if matched:
                    candidate_raw = matched[0]

    if candidate_raw is None:
        print(f"Error: Could not locate candidate file or ID '{candidate_source}'.")
        print("Defaulting to sample candidate CAND-01 (Python + Django Developer).")
        with open("data/test_candidates.json", "r", encoding="utf-8") as f:
            candidate_raw = json.load(f)[0]

    # 3. Preprocess Candidate Resume
    print("[3/5] Cleaning and Preprocessing Resume Data...")
    cleaned_cand, emb_text = preprocess_resume(candidate_raw)
    print(f"      Cleaned candidate: '{cleaned_cand.get('candidate_name')}'. Normalized skills and generated embedding text.")

    # 4. Search and Retrieval
    print(f"[4/5] Executing Semantic & Hybrid Search (Top-{top_k}, Mode: {mode})...")
    filters = {}
    if category_filter and category_filter != "all":
        filters["category"] = category_filter
    if work_mode_filter and work_mode_filter != "all":
        filters["work_mode"] = work_mode_filter
    if location_filter and location_filter != "all":
        filters["location"] = location_filter

    use_hybrid = mode in ["hybrid", "hybrid_rerank"]
    use_rerank = mode == "hybrid_rerank"

    results = search_matching_internships(
        candidate_data=cleaned_cand,
        top_k=top_k,
        collection=collection,
        filters=filters if filters else None,
        use_hybrid=use_hybrid,
        use_reranking=use_rerank
    )
    print(f"      Retrieved {len(results)} matching opportunities from vector space.")

    # 5. RAG AI Explanation Generation
    print("[5/5] Generating Grounded RAG AI Analysis & Skill Gap Explanations...")
    final_recommendations = generate_rag_explanations(cleaned_cand, results)

    # 6. Display Structured Output
    display_results_terminal(cleaned_cand, final_recommendations)


def main():
    parser = argparse.ArgumentParser(description="Resume-Based Internship Recommendation System (RAG)")
    parser.add_argument("--candidate", "-c", default="CAND-01", help="Candidate ID (e.g. CAND-01, CAND-05) or path to candidate JSON file")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of internship recommendations to return (default: 5)")
    parser.add_argument("--mode", "-m", choices=["hybrid_rerank", "hybrid", "dense", "keyword"], default="hybrid_rerank", help="Retrieval algorithm mode")
    parser.add_argument("--category", help="Optional filter by internship category")
    parser.add_argument("--work-mode", choices=["Remote", "Hybrid", "On-site", "all"], help="Optional filter by work mode")
    parser.add_argument("--location", help="Optional filter by location substring")
    parser.add_argument("--evaluate", action="store_true", help="Execute complete benchmark evaluation suite and generate metrics report")

    args = parser.parse_args()

    if args.evaluate:
        print_banner()
        run_benchmark_evaluation(top_k=args.top_k)
    else:
        run_pipeline(
            candidate_source=args.candidate,
            top_k=args.top_k,
            mode=args.mode,
            category_filter=args.category,
            work_mode_filter=args.work_mode,
            location_filter=args.location
        )


if __name__ == "__main__":
    main()
