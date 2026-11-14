"""
Module 7 & 9: Testing, Benchmark Evaluation, and Metrics Engine
Calculates standard Information Retrieval evaluation metrics:
- Precision@5
- Recall@5
- Mean Reciprocal Rank (MRR)
- Hit Rate@5
- Average Retrieval Latency (ms)

Performs comparative benchmark across:
1. Dense Vector Search (Pure Semantic)
2. Keyword Search (Lexical Overlap)
3. Hybrid Search (Dense + Lexical)
4. Hybrid Search with Re-ranking (Full Pipeline)

Saves empirical results to data/evaluation_results.json.
"""

import json
import logging
import os
import time
from typing import Dict, List, Any, Tuple, Optional
import numpy as np

from src.resume_preprocessing import preprocess_resume
from src.vector_database import init_vector_database, get_or_create_collection
from src.internship_retrieval import search_matching_internships
from src.rag_generator import generate_rag_explanations

logger = logging.getLogger(__name__)

# Ground-truth relevance category mapping for benchmark candidates
BENCHMARK_RELEVANCE_MAP: Dict[str, List[str]] = {
    "CAND-01": ["Backend Development", "Python Development"],
    "CAND-02": ["Machine Learning", "Artificial Intelligence", "Data Science"],
    "CAND-03": ["Data Science", "Data Analytics"],
    "CAND-04": ["Frontend Development", "Full Stack Development"],
    "CAND-05": ["Generative AI", "Artificial Intelligence", "Machine Learning"],
    "CAND-06": ["Java Development", "Backend Development"],
    "CAND-07": ["Python Development", "Backend Development", "Artificial Intelligence"],
    "CAND-08": ["Artificial Intelligence", "Machine Learning"],
    "CAND-09": ["Data Analytics", "Data Science"],
    "CAND-10": ["Full Stack Development", "Backend Development", "Frontend Development"]
}


def is_item_relevant(item: Dict[str, Any], relevant_categories: List[str]) -> bool:
    """Evaluates whether an internship item is considered relevant under ground-truth labels."""
    item_cat = item.get("category", "")
    return item_cat in relevant_categories


def compute_retrieval_metrics(
    retrieved_items: List[Dict[str, Any]],
    relevant_categories: List[str],
    total_relevant_in_db: int,
    k: int = 5
) -> Dict[str, float]:
    """
    Computes Precision@K, Recall@K, MRR, and Hit Rate@K for a single query.
    """
    top_items = retrieved_items[:k]
    if not top_items:
        return {
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "hit_rate_at_k": 0.0
        }

    relevant_retrieved = 0
    first_relevant_rank = None

    for rank, item in enumerate(top_items, start=1):
        if is_item_relevant(item, relevant_categories):
            relevant_retrieved += 1
            if first_relevant_rank is None:
                first_relevant_rank = rank

    precision = relevant_retrieved / float(k)
    recall = relevant_retrieved / float(max(1, total_relevant_in_db))
    hit_rate = 1.0 if relevant_retrieved > 0 else 0.0
    mrr = (1.0 / first_relevant_rank) if first_relevant_rank is not None else 0.0

    return {
        "precision_at_k": round(precision, 4),
        "recall_at_k": round(recall, 4),
        "mrr": round(mrr, 4),
        "hit_rate_at_k": round(hit_rate, 4)
    }


def evaluate_single_candidate(
    candidate: Dict[str, Any],
    collection,
    mode: str = "hybrid_rerank",
    top_k: int = 5,
    all_internships: Optional[List[Dict[str, Any]]] = None
) -> Tuple[Dict[str, Any], Dict[str, float], float]:
    """
    Runs retrieval for a single candidate under a specific search mode and calculates metrics.
    Modes:
    - 'dense': Pure cosine vector search
    - 'keyword': Lexical skill matching only
    - 'hybrid': Dense + keyword without re-ranking
    - 'hybrid_rerank': Dense + keyword with multi-factor re-ranking
    """
    cand_id = candidate.get("candidate_id", "CAND-00")
    relevant_cats = BENCHMARK_RELEVANCE_MAP.get(cand_id, [candidate.get("target_category", "")])

    # Count total relevant in DB
    if all_internships:
        total_rel = sum(1 for item in all_internships if item.get("category") in relevant_cats)
    else:
        total_rel = 6  # 6 records per category standard

    start_t = time.perf_counter()

    if mode == "dense":
        results = search_matching_internships(
            candidate,
            top_k=top_k,
            collection=collection,
            use_hybrid=False,
            use_reranking=False
        )
    elif mode == "keyword":
        # Weight 100% on keyword
        results = search_matching_internships(
            candidate,
            top_k=top_k,
            collection=collection,
            use_hybrid=True,
            use_reranking=False,
            semantic_weight=0.0,
            keyword_weight=1.0
        )
    elif mode == "hybrid":
        results = search_matching_internships(
            candidate,
            top_k=top_k,
            collection=collection,
            use_hybrid=True,
            use_reranking=False,
            semantic_weight=0.65,
            keyword_weight=0.35
        )
    else:  # 'hybrid_rerank'
        results = search_matching_internships(
            candidate,
            top_k=top_k,
            collection=collection,
            use_hybrid=True,
            use_reranking=True
        )

    latency_ms = (time.perf_counter() - start_t) * 1000.0
    metrics = compute_retrieval_metrics(results, relevant_cats, total_rel, k=top_k)

    # Attach relevance indicator to results
    for r in results:
        r["is_relevant_ground_truth"] = is_item_relevant(r, relevant_cats)

    query_report = {
        "candidate_id": cand_id,
        "candidate_name": candidate.get("candidate_name"),
        "target_category": candidate.get("target_category"),
        "scenario_description": candidate.get("scenario_description"),
        "relevant_categories": relevant_cats,
        "latency_ms": round(latency_ms, 2),
        "metrics": metrics,
        "retrieved_results": [
            {
                "rank": r.get("rank"),
                "internship_id": r.get("internship_id"),
                "title": r.get("title"),
                "company_name": r.get("company_name"),
                "category": r.get("category"),
                "similarity_score": r.get("similarity_score"),
                "overall_ranking_score": r.get("overall_ranking_score"),
                "is_relevant": r.get("is_relevant_ground_truth")
            }
            for r in results
        ]
    }

    return query_report, metrics, latency_ms


def run_benchmark_evaluation(
    candidates_path: str = "data/test_candidates.json",
    internships_path: str = "data/internships.json",
    top_k: int = 5,
    save_results_path: str = "data/evaluation_results.json"
) -> Dict[str, Any]:
    """
    Executes comprehensive benchmark evaluation across all 10 candidate scenarios
    and compares Dense, Keyword, Hybrid, and Hybrid+Re-ranking approaches.
    """
    logger.info("Initializing vector database for evaluation...")
    collection, _ = init_vector_database(json_filepath=internships_path)

    # Load candidates
    with open(candidates_path, "r", encoding="utf-8") as f:
        all_candidates = json.load(f)

    # Filter to the primary 10 benchmark candidates (CAND-01 to CAND-10)
    benchmark_candidates = [c for c in all_candidates if c["candidate_id"].startswith("CAND-")]

    # Load all internships for exact recall denominator
    with open(internships_path, "r", encoding="utf-8") as f:
        all_internships = json.load(f)

    modes = ["dense", "keyword", "hybrid", "hybrid_rerank"]
    mode_labels = {
        "dense": "Dense Vector Search (Cosine Similarity)",
        "keyword": "Keyword Search (Lexical Overlap)",
        "hybrid": "Hybrid Search (Dense + Keyword)",
        "hybrid_rerank": "Hybrid Search with Re-ranking (Full Pipeline)"
    }

    comparison_results: Dict[str, Any] = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "k_metric": top_k,
        "num_candidates_tested": len(benchmark_candidates),
        "total_internships_in_database": len(all_internships),
        "method_comparison": {},
        "candidate_detailed_runs": {}
    }

    print("\n" + "="*80)
    print("RESUME-BASED INTERNSHIP RECOMMENDATION RAG SYSTEM - BENCHMARK EVALUATION")
    print(f"Testing {len(benchmark_candidates)} Candidates | K={top_k} | Database: {len(all_internships)} Internships")
    print("="*80)

    for mode in modes:
        mode_metrics: List[Dict[str, float]] = []
        mode_latencies: List[float] = []
        query_runs: List[Dict[str, Any]] = []

        for cand in benchmark_candidates:
            report, metrics, lat = evaluate_single_candidate(
                candidate=cand,
                collection=collection,
                mode=mode,
                top_k=top_k,
                all_internships=all_internships
            )
            mode_metrics.append(metrics)
            mode_latencies.append(lat)
            query_runs.append(report)

        # Aggregate averages
        avg_precision = float(np.mean([m["precision_at_k"] for m in mode_metrics]))
        avg_recall = float(np.mean([m["recall_at_k"] for m in mode_metrics]))
        avg_mrr = float(np.mean([m["mrr"] for m in mode_metrics]))
        avg_hit_rate = float(np.mean([m["hit_rate_at_k"] for m in mode_metrics]))
        avg_latency = float(np.mean(mode_latencies))

        comparison_results["method_comparison"][mode] = {
            "method_name": mode_labels[mode],
            f"precision_at_{top_k}": round(avg_precision, 4),
            f"recall_at_{top_k}": round(avg_recall, 4),
            "mean_reciprocal_rank_mrr": round(avg_mrr, 4),
            f"hit_rate_at_{top_k}": round(avg_hit_rate, 4),
            "avg_latency_ms": round(avg_latency, 2)
        }
        comparison_results["candidate_detailed_runs"][mode] = query_runs

        print(f"\nConfiguration: {mode_labels[mode]}")
        print(f"  Precision@{top_k} : {avg_precision*100:.2f}%")
        print(f"  Recall@{top_k}    : {avg_recall*100:.2f}%")
        print(f"  MRR          : {avg_mrr:.4f}")
        print(f"  Hit Rate@{top_k} : {avg_hit_rate*100:.2f}%")
        print(f"  Avg Latency  : {avg_latency:.2f} ms")

    # Save to disk
    os.makedirs(os.path.dirname(save_results_path), exist_ok=True)
    with open(save_results_path, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)

    print("\n" + "="*80)
    print(f"Evaluation report successfully saved to: {save_results_path}")
    print("="*80 + "\n")

    return comparison_results


if __name__ == "__main__":
    run_benchmark_evaluation()
