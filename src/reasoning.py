"""
Reasoning Generator v3.

Generates specific, honest, varied 1-2 sentence justifications.
v3 improvements:
- Mentions production signals explicitly
- Clearer skill callouts
- More natural language
"""

from typing import Any, Dict, List, Tuple


def generate_reasoning(candidate: Dict, features: Dict, rank: int) -> str:
    """Generate unique, specific reasoning for a candidate."""
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    title = profile.get("current_title", "")
    company = profile.get("current_company", "")
    years = profile.get("years_of_experience", 0)
    location = profile.get("location", "")
    country = profile.get("country", "")

    tier = features.get("title_tier", 2)
    career_score = features.get("career_score", 0)
    tech_depth = features.get("technical_depth", 0)
    sem_sim = features.get("semantic_similarity", 0)
    matched_skills = features.get("matched_jd_skills", 0)
    ca = features.get("career_analysis", {})
    notice = features.get("notice_days", 90)
    otw = features.get("open_to_work", False)
    rr = features.get("response_rate", 0)
    beh_mult = features.get("behavioral_multiplier", 1.0)
    prod_score = ca.get("production_score", 0)

    strengths = []
    concerns = []

    # Strengths
    if tier >= 5:
        strengths.append(f"title '{title}' is a direct JD match for Senior AI/ML Engineering")
    elif tier >= 4:
        strengths.append(f"{title} role with strong ML/AI background")
    elif career_score > 0.55:
        strengths.append("solid AI/ML career trajectory")

    if ca.get("product_ratio", 0) > 0.6:
        strengths.append("predominantly product company experience")
    elif ca.get("product_ratio", 0) > 0.3:
        strengths.append("mixed product/consulting background with product exposure")

    if prod_score > 0.6:
        strengths.append("clear evidence of shipping production ML systems")
    elif prod_score > 0.3:
        strengths.append("some production deployment signals")

    if sem_sim > 0.75:
        strengths.append("high semantic alignment with JD requirements")
    elif sem_sim > 0.60:
        strengths.append("good semantic match with JD")

    if matched_skills >= 8:
        strengths.append(f"{matched_skills} of the required JD skills matched (embeddings, vector DBs, eval)")
    elif matched_skills >= 5:
        strengths.append(f"{matched_skills} JD-required technical skills")
    elif matched_skills >= 3:
        strengths.append(f"{matched_skills} relevant skills from JD requirements")

    if 5 <= years <= 7:
        strengths.append(f"{years:.0f} years is the ideal range (JD wants 5-9)")
    elif 7 < years <= 9:
        strengths.append(f"{years:.0f} years of experience")

    if features.get("location_score", 0) >= 0.9:
        strengths.append(f"based in {location} — preferred JD location")
    elif features.get("location_score", 0) >= 0.6:
        strengths.append(f"India-based, open to Pune/Noida")

    if rr > 0.75:
        strengths.append(f"strong recruiter response rate ({rr:.0%})")
    if otw and notice <= 30:
        strengths.append(f"actively open to work with {notice}-day notice")
    elif otw:
        strengths.append("actively open to work")
    elif notice <= 30:
        strengths.append(f"short {notice}-day notice period")

    if ca.get("has_progression", False):
        strengths.append("clear career progression to senior roles")

    # Concerns
    if tier <= 2:
        concerns.append(f"title '{title}' is not aligned with AI Engineering")

    if ca.get("is_consulting_only", False):
        concerns.append("consulting-only background (JD disqualifier)")
    elif ca.get("consulting_ratio", 0) > 0.5:
        concerns.append("heavy consulting background")

    if ca.get("job_hopper", False):
        concerns.append("frequent job changes (title-chaser risk)")

    if years < 5:
        concerns.append(f"{years:.1f} years below ideal 5-9 range")
    elif years > 12:
        concerns.append(f"{years:.1f} years — may be overqualified")

    if beh_mult < 0.85:
        if rr < 0.2:
            concerns.append(f"very low response rate ({rr:.0%})")
        else:
            concerns.append("low behavioral engagement signals")

    if notice > 90:
        concerns.append(f"long {notice}-day notice period")

    if features.get("location_score", 0) < 0.4:
        concerns.append(f"located in {location}, {country} — outside preferred zone")

    # Construct reasoning by rank
    if rank <= 15:
        if strengths:
            s_text = "; ".join(strengths[:3])
            if concerns and rank <= 10:
                reasoning = f"{s_text}. Note: {concerns[0]}."
            else:
                reasoning = f"{s_text}."
        else:
            reasoning = f"Strong overall fit: {years:.0f} years as {title} at {company}."

    elif rank <= 50:
        if strengths and concerns:
            reasoning = f"{strengths[0]}; however, {concerns[0]}."
        elif strengths:
            reasoning = f"{strengths[0]}; moderate overall fit."
        elif concerns:
            reasoning = f"Concerns: {concerns[0]}."
        else:
            reasoning = f"{title} with {years:.1f} years; partial JD alignment."

    else:
        if concerns:
            c_text = "; ".join(concerns[:2])
            if strengths:
                reasoning = f"{c_text}. Positives: {strengths[0]}."
            else:
                reasoning = f"{c_text}."
        else:
            reasoning = f"Lower-ranked — {years:.1f} years as {title}, partial alignment."

    reasoning = reasoning.strip()
    if not reasoning.endswith("."):
        reasoning += "."
    return reasoning[0].upper() + reasoning[1:]


def generate_all_reasonings(ranked):
    """Generate reasoning for all top-K candidates."""
    return [generate_reasoning(cand, feats, rank)
            for rank, (cand, score, feats) in enumerate(ranked, start=1)]
