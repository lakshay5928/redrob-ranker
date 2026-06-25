"""
Multi-Signal Scoring Engine v3 — Tuned for Maximum NDCG.

Key changes from v2:
- Education score now included in profile score
- Company quality weighted separately
- Production signal integrated into tech depth
- Tighter behavioral multiplier floor (0.80)
"""

import numpy as np
from typing import Any, Dict, List, Tuple

from config import ScoringWeights, STAGE1_TOP_K, STAGE2_TOP_K, FINAL_TOP_K
from features import (
    aggregate_text, compute_behavioral_v2, behavioral_multiplier_v2,
    classify_title_v2, classify_company_v2, analyze_career_v2,
    score_experience_v2, score_location_v2, score_education_v2,
    match_jd_required_skills, match_jd_context_keywords,
    compute_tech_depth_v2, extract_all_features_v2, stage1_prefilter_v2,
)
from embeddings import compute_similarity_scores
from honeypot_detector import compute_honeypot_penalty


def _apply_boosts_and_penalties(scored: list) -> list:
    """
    Post-scoring boosts/penalties for sharper differentiation.
    Applied AFTER initial scoring — makes top candidates stand out more.
    """
    result = []
    for cand, score, feats in scored:
        boost = 1.0
        ca = feats.get("career_analysis", {})

        # Tier 5 title + product company + production signals = strong boost
        if (feats.get("title_tier", 0) >= 5 and
                feats.get("company_type", "") == "product_strong" and
                ca.get("production_score", 0) > 0.5):
            boost *= 1.08

        # Strong semantic match + high skill coverage
        if (feats.get("semantic_similarity", 0) > 0.72 and
                feats.get("jd_skill_match_score", 0) > 0.5):
            boost *= 1.05

        # Active candidate ready to join quickly
        if (feats.get("open_to_work", False) and
                feats.get("notice_days", 999) <= 30 and
                feats.get("response_rate", 0) > 0.7):
            boost *= 1.03

        # Ideal experience band: 5-7 years
        years = feats.get("years", 0)
        if 5.0 <= years <= 7.0:
            boost *= 1.02

        # Consulting-only hard penalty (JD disqualifier — extra enforcement)
        if ca.get("is_consulting_only", False):
            boost *= 0.60

        # Job hopper hard penalty
        if ca.get("job_hopper", False):
            boost *= 0.88

        # No ML title at all in career
        if ca.get("ml_title_count", 0) == 0 and feats.get("title_tier", 0) < 3:
            boost *= 0.75

        final_score = min(score * boost, 0.9999)  # Cap at <1.0
        result.append((cand, final_score, feats))

    return result


def stage2_deepscore_v2(
    candidates: List[Dict],
    jd_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
) -> List[Tuple[Dict, float, Dict[str, Any]]]:
    """
    Stage 2: Deep scoring with embeddings + features.
    Returns: (candidate, final_score, features)
    """
    similarities = compute_similarity_scores(candidate_embeddings, jd_embedding)

    scored = []
    weights = ScoringWeights()

    for i, cand in enumerate(candidates):
        features = extract_all_features_v2(cand)

        # Honeypot penalty — hard filter
        hp = compute_honeypot_penalty(cand)
        if hp == 0.0:
            continue

        # Semantic similarity from embeddings
        sem = float(similarities[i])

        # Profile score (weighted combination)
        profile_score = (
            features["career_score"]     * weights.title_career +
            sem                          * weights.semantic_similarity +
            features["technical_depth"]  * weights.technical_depth +
            features["experience_score"] * weights.experience_quality +
            features["company_score"]    * weights.company_quality +
            features["location_score"]   * weights.location_match
        )

        # Behavioral multiplier (floor 0.80 — profile first)
        beh_mult = features["behavioral_multiplier"]

        # Final score
        final = profile_score * beh_mult * hp

        features["semantic_similarity"] = sem
        features["honeypot_penalty"] = hp
        features["profile_score"] = profile_score

        scored.append((cand, final, features))

    # Apply post-scoring boosts/penalties
    scored = _apply_boosts_and_penalties(scored)

    # Sort by score desc, then candidate_id asc for tie-breaking (validation requirement)
    scored.sort(key=lambda x: (-x[1], x[0].get("candidate_id", "")))
    return scored[:STAGE2_TOP_K]


def rank_candidates_v2(candidates: List[Dict], jd_embedding: np.ndarray) -> List[Tuple[Dict, float, Dict[str, Any]]]:
    """Full v3 ranking pipeline."""
    import time
    print(f"[Pipeline] Starting with {len(candidates)} candidates")

    # Stage 1
    print("[Stage 1] Fast pre-filtering...")
    t0 = time.time()
    stage1 = stage1_prefilter_v2(candidates)
    print(f"[Stage 1] {len(stage1)} candidates kept in {time.time()-t0:.1f}s")

    # Embeddings
    stage1_cands = [c for c, _, _ in stage1]
    print(f"[Embeddings] Computing for {len(stage1_cands)} candidates...")
    from embeddings import compute_embeddings_batch
    t0 = time.time()
    texts = [aggregate_text(c) for c in stage1_cands]
    embs = compute_embeddings_batch(texts, batch_size=64)
    print(f"[Embeddings] Done in {time.time()-t0:.1f}s")

    # Stage 2
    print("[Stage 2] Deep scoring...")
    t0 = time.time()
    stage2 = stage2_deepscore_v2(stage1_cands, jd_embedding, embs)
    print(f"[Stage 2] {len(stage2)} scored in {time.time()-t0:.1f}s")

    # Stage 3: Final ranking — score desc, candidate_id asc for ties
    print("[Stage 3] Final ranking...")
    stage2.sort(key=lambda x: (-x[1], x[0].get("candidate_id", "")))
    final = stage2[:FINAL_TOP_K]

    print(f"[Done] Top {len(final)} selected")
    return final
