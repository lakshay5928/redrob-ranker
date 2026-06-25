"""
Feature Extraction Engine v3 — Tuned for NDCG@10 and NDCG@50.

Key improvements over v2:
- Production signal detection in career descriptions
- Better company classification (startup scoring improved)
- Stronger penalty for consulting-only + no product experience
- Career description quality scoring
- Improved JD skill matching with aliases
- Experience progression check (junior → senior ML)
- Tighter behavioral multiplier floor (0.80)
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Set, Tuple

from config import (
    CONSULTING_COMPANIES,
    FINAL_TOP_K,
    IDEAL_YEARS_MAX,
    IDEAL_YEARS_MIN,
    IDEAL_YEARS_TARGET,
    INDIA_COUNTRY,
    JD_CONTEXT_KEYWORDS,
    JD_NICE_TO_HAVE_SKILLS,
    JD_REQUIRED_SKILLS,
    PREFERRED_LOCATIONS,
    PRODUCTION_KEYWORDS,
    STAGE1_TOP_K,
    STARTUP_INDICATORS,
    STRONG_PRODUCT_COMPANIES,
    TIER1_TITLES,
    TIER2_TITLES,
    TIER3_TITLES,
    TIER4_TITLES,
    TIER5_TITLES,
    TOO_JUNIOR,
    VERY_HIGH_YEARS,
)


# ============================================================================
# TEXT PREPROCESSING
# ============================================================================

def normalize(text: str) -> str:
    return text.lower().strip() if text else ""


def normalize_company(name: str) -> str:
    name = normalize(name)
    name = re.sub(r'\s+(pvt\.?\s+ltd\.?|ltd\.?|inc\.?|corp\.?|corporation|limited|llc|llp)$', '', name)
    return name.strip()


# ============================================================================
# TITLE CLASSIFICATION (Tier-based)
# ============================================================================

def classify_title_v2(title: str) -> Tuple[int, float, str]:
    """
    Classify title into tiers (1-5).
    Tier 5=1.0, Tier 4=0.80, Tier 3=0.55, Tier 2=0.30, Tier 1=0.10
    """
    title_norm = normalize(title)

    # Direct exact matches
    if title_norm in TIER5_TITLES:
        return 5, 1.0, "tier5"
    if title_norm in TIER4_TITLES:
        return 4, 0.80, "tier4"
    if title_norm in TIER3_TITLES:
        return 3, 0.55, "tier3"
    if title_norm in TIER2_TITLES:
        return 2, 0.30, "tier2"
    if title_norm in TIER1_TITLES:
        return 1, 0.10, "tier1"

    # Partial/substring matches
    for t in TIER5_TITLES:
        if t in title_norm or title_norm in t:
            return 5, 1.0, "tier5"
    for t in TIER4_TITLES:
        if t in title_norm or title_norm in t:
            return 4, 0.80, "tier4"
    for t in TIER3_TITLES:
        if t in title_norm or title_norm in t:
            return 3, 0.55, "tier3"
    for t in TIER1_TITLES:
        if t in title_norm or title_norm in t:
            return 1, 0.10, "tier1"

    # Keyword-based detection
    strong_ml_kws = ["ai ", " ai", "ml ", " ml", "machine learning", "deep learning",
                     "nlp", "natural language", "search", "ranking", "retrieval",
                     "recommendation", "algorithm", "scientist", "research engineer"]
    if any(kw in title_norm for kw in strong_ml_kws):
        return 4, 0.70, "keyword_ml"

    moderate_kws = ["data", "analytics", "intelligence"]
    if any(kw in title_norm for kw in moderate_kws):
        return 3, 0.45, "keyword_data"

    return 2, 0.25, "unknown"


# ============================================================================
# COMPANY CLASSIFICATION
# ============================================================================

def classify_company_v2(company: str) -> Tuple[str, float]:
    """
    Classify company with score.
    Strong product: 1.0, Good product/startup: 0.75-0.85, Unknown: 0.45, Consulting: 0.0
    """
    name_norm = normalize_company(company)
    if not name_norm:
        return "unknown", 0.4

    # Strong product companies
    for pc in STRONG_PRODUCT_COMPANIES:
        if pc in name_norm or name_norm in pc:
            return "product_strong", 1.0

    # Consulting companies (JD explicitly disqualifies)
    for cc in CONSULTING_COMPANIES:
        if cc in name_norm or name_norm in cc:
            return "consulting", 0.0

    # Startup indicators — count how many match
    indicator_count = sum(1 for ind in STARTUP_INDICATORS if ind in name_norm)
    if indicator_count >= 2:
        return "product_good", 0.85
    if indicator_count >= 1:
        return "product_good", 0.75

    # Short name without spaces often a startup/product company
    if len(name_norm) <= 10 and " " not in name_norm:
        return "product_good", 0.65

    return "unknown", 0.45


# ============================================================================
# EXPLICIT JD SKILL MATCHING (v3 — with aliases)
# ============================================================================

# Skill aliases for fuzzy matching
SKILL_ALIASES: Dict[str, List[str]] = {
    "sentence-transformers": ["sentence_transformers", "sbert", "bi-encoder"],
    "elasticsearch": ["elastic search", "es", "elastic"],
    "opensearch": ["open search"],
    "vector database": ["vector db", "vector store", "vectordb"],
    "llm": ["large language model", "gpt", "claude", "gemini", "llama", "mistral"],
    "rag": ["retrieval augmented generation", "retrieval-augmented"],
    "fine-tuning": ["finetuning", "fine tuning", "finetune"],
    "lora": ["low-rank adaptation", "low rank adaptation"],
    "ndcg": ["normalized discounted cumulative gain"],
    "mrr": ["mean reciprocal rank"],
    "map": ["mean average precision"],
    "a/b test": ["ab test", "a/b testing", "ab testing", "split test"],
    "machine learning": ["ml", "machine-learning"],
    "deep learning": ["dl", "deep-learning"],
    "information retrieval": ["ir ", "search relevance"],
    "hybrid search": ["hybrid retrieval", "bm25+dense"],
    "recommendation": ["recommender", "recsys", "rec sys"],
}


def match_jd_required_skills(skills: List[Dict]) -> Tuple[int, int, float]:
    """
    Count JD-required skills with alias matching.
    Returns: (matched_count, total_jd_skills, match_score)
    """
    if not skills:
        return 0, len(JD_REQUIRED_SKILLS), 0.0

    candidate_text = set()
    for skill in skills:
        name = normalize(skill.get("name", ""))
        candidate_text.add(name)
        candidate_text.add(name.replace(" ", "").replace("-", ""))
        # Add individual words
        for word in name.split():
            if len(word) > 2:
                candidate_text.add(word)

    matched = 0
    for jd_skill in JD_REQUIRED_SKILLS:
        jd_norm = normalize(jd_skill)
        found = False

        # Direct match
        for cs in candidate_text:
            if jd_norm in cs or cs in jd_norm:
                found = True
                break

        # Alias match
        if not found:
            aliases = SKILL_ALIASES.get(jd_norm, [])
            for alias in aliases:
                if any(alias in cs or cs in alias for cs in candidate_text):
                    found = True
                    break

        if found:
            matched += 1

    score = min(matched / max(len(JD_REQUIRED_SKILLS) * 0.5, 1), 1.0)  # Normalize: 50% match = 1.0
    return matched, len(JD_REQUIRED_SKILLS), score


def match_jd_context_keywords(text: str) -> Tuple[int, float]:
    """Count JD context keywords in text."""
    text_norm = normalize(text)
    matched = sum(1 for kw in JD_CONTEXT_KEYWORDS if kw in text_norm)
    score = min(matched / max(len(JD_CONTEXT_KEYWORDS) * 0.3, 1), 1.0)  # 30% = max
    return matched, score


def score_production_signals(career_history: List[Dict]) -> float:
    """
    Score production/shipping signals in career descriptions.
    JD explicitly wants people who have SHIPPED things to real users.
    """
    if not career_history:
        return 0.0

    prod_hits = 0
    total_desc_len = 0

    for role in career_history:
        desc = normalize(role.get("description", ""))
        total_desc_len += len(desc)
        prod_hits += sum(1 for kw in PRODUCTION_KEYWORDS if kw in desc)

    if total_desc_len < 50:
        return 0.3  # No descriptions — neutral

    # Normalize by number of roles
    score = min(prod_hits / (len(career_history) * 2.0), 1.0)
    return score


# ============================================================================
# CAREER TRAJECTORY ANALYSIS (v3)
# ============================================================================

def analyze_career_v2(career_history: List[Dict]) -> Dict[str, Any]:
    """Career analysis with production signals and progression."""
    if not career_history:
        return {
            "product_ratio": 0.0,
            "consulting_ratio": 0.0,
            "avg_tenure_months": 0,
            "num_companies": 0,
            "has_progression": False,
            "ml_title_count": 0,
            "total_roles": 0,
            "title_stability": 0.0,
            "is_consulting_only": True,
            "has_recent_ml_role": False,
            "industries": set(),
            "production_score": 0.0,
            "job_hopper": False,
        }

    companies = set()
    consulting_months = 0
    product_months = 0
    total_months = 0
    ml_title_count = 0
    titles = []
    industries = set()
    short_tenures = 0

    for role in career_history:
        company = role.get("company", "")
        title = role.get("title", "")
        duration = role.get("duration_months", 0)
        industry = role.get("industry", "")

        companies.add(company)
        industries.add(industry)
        titles.append(title)
        total_months += duration

        if duration < 12 and duration > 0:
            short_tenures += 1

        comp_type, _ = classify_company_v2(company)
        if comp_type == "consulting":
            consulting_months += duration
        elif comp_type in ("product_strong", "product_good"):
            product_months += duration

        tier, score, _ = classify_title_v2(title)
        if tier >= 4:
            ml_title_count += 1

    num_companies = len(companies)
    avg_tenure = total_months / num_companies if num_companies > 0 else 0

    # Title stability
    if avg_tenure >= 30:
        title_stability = 1.0
    elif avg_tenure >= 24:
        title_stability = 0.9
    elif avg_tenure >= 18:
        title_stability = 0.7
    elif avg_tenure >= 12:
        title_stability = 0.5
    else:
        title_stability = 0.2

    # Job hopper: > 40% roles < 1 year
    job_hopper = short_tenures / max(len(career_history), 1) > 0.4

    # Recent ML role (current or last position)
    current_or_last = career_history[0] if career_history else {}
    recent_tier, _, _ = classify_title_v2(current_or_last.get("title", ""))
    has_recent_ml = recent_tier >= 4

    is_consulting_only = (
        consulting_months > 0 and product_months == 0 and
        consulting_months >= total_months * 0.8
    )

    product_ratio = product_months / total_months if total_months > 0 else 0
    consulting_ratio = consulting_months / total_months if total_months > 0 else 0

    # Production signals from descriptions
    production_score = score_production_signals(career_history)

    return {
        "product_ratio": product_ratio,
        "consulting_ratio": consulting_ratio,
        "avg_tenure_months": avg_tenure,
        "num_companies": num_companies,
        "has_progression": check_progression(titles),
        "ml_title_count": ml_title_count,
        "total_roles": len(titles),
        "title_stability": title_stability,
        "is_consulting_only": is_consulting_only,
        "has_recent_ml_role": has_recent_ml,
        "industries": industries,
        "production_score": production_score,
        "job_hopper": job_hopper,
    }


def check_progression(titles: List[str]) -> bool:
    """Check for upward career progression."""
    if not titles or len(titles) < 2:
        return False
    all_text = " ".join(titles).lower()
    if any(kw in all_text for kw in ["senior", "staff", "principal", "lead", "head of", "director"]):
        return True
    return False


# ============================================================================
# EXPERIENCE SCORING
# ============================================================================

def score_experience_v2(years: float) -> float:
    """Score experience with ideal range 5-9 years."""
    if years < 1:
        return 0.0
    if years < 3:
        return 0.15
    if years < 4:
        return 0.35
    if years < 5:
        return 0.60
    if years <= 7:
        return 1.0   # Sweet spot: 5-7 years
    if years <= 9:
        return 0.90  # Still good: 7-9 years
    if years <= 12:
        return 0.70  # Slightly senior
    if years <= 15:
        return 0.50  # Overqualified
    return 0.30      # Too senior


# ============================================================================
# EDUCATION SCORING
# ============================================================================

def score_education_v2(education: List[Dict]) -> float:
    """Score education — tier, field, institution."""
    if not education:
        return 0.3

    best_score = 0.0

    for edu in education:
        degree = normalize(edu.get("degree", ""))
        field = normalize(edu.get("field_of_study", ""))
        institution = normalize(edu.get("institution", ""))

        # Degree level
        if any(kw in degree for kw in ["phd", "ph.d", "doctorate"]):
            deg_score = 1.0
        elif any(kw in degree for kw in ["master", "m.tech", "m.e.", "ms ", "msc"]):
            deg_score = 0.85
        elif any(kw in degree for kw in ["bachelor", "b.tech", "b.e.", "bs ", "bsc"]):
            deg_score = 0.7
        else:
            deg_score = 0.5

        # Field relevance
        if any(kw in field for kw in ["computer science", "machine learning", "artificial intelligence",
                                       "data science", "information technology", "statistics"]):
            field_score = 1.0
        elif any(kw in field for kw in ["engineering", "mathematics", "physics", "electronics"]):
            field_score = 0.8
        else:
            field_score = 0.5

        # Tier institutions (IITs, IISc, NITs, IIITs)
        if any(kw in institution for kw in ["iit", "iisc", "bits", "iiit"]):
            inst_score = 1.0
        elif any(kw in institution for kw in ["nit ", "vit", "dtu", "nsit", "pes university",
                                               "manipal", "srm", "amrita"]):
            inst_score = 0.8
        else:
            inst_score = 0.6

        edu_score = deg_score * 0.5 + field_score * 0.35 + inst_score * 0.15
        best_score = max(best_score, edu_score)

    return best_score


# ============================================================================
# LOCATION SCORING
# ============================================================================

def score_location_v2(location: str, country: str) -> float:
    """Score location — Pune/Noida preferred, India OK."""
    loc_norm = normalize(location)
    country_norm = normalize(country)

    if any(city in loc_norm for city in ["pune", "noida", "gurgaon", "gurugram"]):
        return 1.0
    if any(city in loc_norm for city in PREFERRED_LOCATIONS):
        return 0.85
    if country_norm == INDIA_COUNTRY or "india" in loc_norm:
        return 0.6
    # Outside India
    return 0.2


# ============================================================================
# BEHAVIORAL SIGNALS (v3 — tighter range)
# ============================================================================

def compute_behavioral_v2(signals: Dict) -> Dict[str, float]:
    """Compute behavioral signals — profile is primary, behavioral secondary."""
    result = {}

    # Response rate
    rr = signals.get("recruiter_response_rate", 0)
    result["response_rate"] = rr

    # Recency
    last_active = signals.get("last_active_date", "")
    if last_active:
        try:
            days = (datetime(2026, 6, 23) - datetime.strptime(last_active, "%Y-%m-%d")).days
            if days < 7:   result["recency"] = 1.0
            elif days < 30: result["recency"] = 0.92
            elif days < 60: result["recency"] = 0.80
            elif days < 90: result["recency"] = 0.65
            elif days < 180: result["recency"] = 0.45
            else:           result["recency"] = 0.25
        except:
            result["recency"] = 0.5
    else:
        result["recency"] = 0.45

    # Open to work
    result["open_to_work"] = 1.0 if signals.get("open_to_work_flag", False) else 0.5

    # Profile completeness
    result["completeness"] = signals.get("profile_completeness_score", 50) / 100.0

    # Interview completion
    result["interview"] = signals.get("interview_completion_rate", 0.5)

    # Notice period
    notice = signals.get("notice_period_days", 90)
    if notice <= 15:    result["notice"] = 1.0
    elif notice <= 30:  result["notice"] = 0.95
    elif notice <= 60:  result["notice"] = 0.80
    elif notice <= 90:  result["notice"] = 0.65
    elif notice <= 120: result["notice"] = 0.45
    else:               result["notice"] = 0.25

    # Offer acceptance
    oar = signals.get("offer_acceptance_rate", -1)
    result["offer_acceptance"] = min(oar * 1.5, 1.0) if oar >= 0 else 0.5

    # GitHub
    gh = signals.get("github_activity_score", -1)
    result["github"] = min(gh / 50.0, 1.0) if gh >= 0 else 0.3

    # Verification
    ve = signals.get("verified_email", False)
    vp = signals.get("verified_phone", False)
    result["verification"] = (0.5 if ve else 0) + (0.5 if vp else 0)

    # Saved by recruiters
    saves = signals.get("saved_by_recruiters_30d", 0)
    result["recruiter_saves"] = min(saves / 10.0, 1.0)

    return result


def behavioral_multiplier_v2(behavioral: Dict[str, float]) -> float:
    """
    Compute behavioral multiplier — range [0.80, 1.0].
    Behavioral adjusts but does NOT dominate.
    """
    weights = {
        "response_rate":    0.25,
        "recency":          0.20,
        "open_to_work":     0.15,
        "notice":           0.12,
        "interview":        0.10,
        "completeness":     0.08,
        "recruiter_saves":  0.05,
        "github":           0.03,
        "verification":     0.02,
    }

    score = sum(behavioral.get(k, 0) * w for k, w in weights.items())
    # Map to [0.80, 1.0] — floor at 0.80 to not punish strong profiles
    return 0.80 + 0.20 * max(0, min(1, score))


# ============================================================================
# TEXT AGGREGATION FOR EMBEDDINGS
# ============================================================================

def aggregate_text(candidate: Dict) -> str:
    """
    Aggregate candidate text for embedding — enriched with career descriptions.
    """
    parts = []
    p = candidate.get("profile", {})

    # Profile
    parts.append(p.get("current_title", ""))
    parts.append(p.get("headline", ""))
    parts.append(p.get("summary", ""))

    # Career — title + description (most important)
    for role in candidate.get("career_history", []):
        parts.append(role.get("title", ""))
        parts.append(role.get("company", ""))
        desc = role.get("description", "")
        if desc:
            parts.append(desc)  # Full description for semantic richness

    # Skills
    for skill in candidate.get("skills", []):
        parts.append(skill.get("name", ""))

    # Education
    for edu in candidate.get("education", []):
        parts.append(edu.get("field_of_study", ""))

    return " ".join(x for x in parts if x)


# ============================================================================
# COMPLETE FEATURE EXTRACTION (v3)
# ============================================================================

def match_jd_nice_skills(skills: List[Dict]) -> float:
    """Score nice-to-have JD skills — bonus signal."""
    if not skills:
        return 0.0
    candidate_text = set()
    for skill in skills:
        name = normalize(skill.get("name", ""))
        candidate_text.add(name)
        for word in name.split():
            if len(word) > 2:
                candidate_text.add(word)
    matched = sum(1 for jd_skill in JD_NICE_TO_HAVE_SKILLS
                  if any(normalize(jd_skill) in cs or cs in normalize(jd_skill)
                         for cs in candidate_text))
    return min(matched / max(len(JD_NICE_TO_HAVE_SKILLS) * 0.4, 1), 1.0)


def compute_tech_depth_v2(skills: List[Dict], career_analysis: Dict,
                          skill_match_score: float, context_score: float) -> float:
    """Compute technical depth — JD skill matching + production signals."""
    if skills:
        avg_prof = sum({"beginner": 0.25, "intermediate": 0.5,
                        "advanced": 0.75, "expert": 1.0}.get(
                        s.get("proficiency", "beginner").lower(), 0.25)
                      for s in skills) / len(skills)
        avg_duration = sum(s.get("duration_months", 0) for s in skills) / len(skills)
        duration_score = min(avg_duration / 24.0, 1.0)

        # Keyword stuffing: expert/advanced with near-zero duration
        expert_zero_duration = sum(1 for s in skills
                                   if s.get("proficiency") in ["expert", "advanced"]
                                   and s.get("duration_months", 0) < 3)
        stuffing_penalty = min(expert_zero_duration * 0.1, 0.4)

        skill_quality = avg_prof * 0.3 + duration_score * 0.3 - stuffing_penalty * 0.4
    else:
        skill_quality = 0.0

    ml_score = min(career_analysis.get("ml_title_count", 0) / 2.0, 1.0) * 0.4
    prod_score = career_analysis.get("production_score", 0.0)
    nice_score = match_jd_nice_skills(skills)

    return (
        skill_match_score * 0.32 +
        context_score     * 0.20 +
        skill_quality     * 0.18 +
        ml_score          * 0.12 +
        prod_score        * 0.13 +
        nice_score        * 0.05
    )


def extract_all_features_v2(candidate: Dict) -> Dict[str, Any]:
    """Extract complete feature vector."""
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    education = candidate.get("education", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})

    years = profile.get("years_of_experience", 0)
    title = profile.get("current_title", "")
    company = profile.get("current_company", "")
    location = profile.get("location", "")
    country = profile.get("country", "")
    summary = profile.get("summary", "")
    headline = profile.get("headline", "")

    # 1. Title
    tier, title_score, title_cat = classify_title_v2(title)

    # 2. Career analysis
    career_analysis = analyze_career_v2(career)

    # Career score
    career_score = (
        title_score * 0.38 +
        career_analysis["product_ratio"] * 0.28 +
        (0.0 if career_analysis["is_consulting_only"] else 0.55) * 0.14 +
        career_analysis["title_stability"] * 0.10 +
        (1.0 if career_analysis["has_recent_ml_role"] else 0.0) * 0.06 +
        (1.0 if career_analysis["has_progression"] else 0.0) * 0.04
    )
    # Job hopper penalty (JD explicitly says "title-chasers switching every 1.5 years")
    if career_analysis["job_hopper"]:
        career_score *= 0.80

    # Hard consulting-only penalty — JD explicit disqualifier
    if career_analysis["is_consulting_only"]:
        career_score *= 0.50

    # Boost for recent ML role + production signals together
    if career_analysis["has_recent_ml_role"] and career_analysis.get("production_score", 0) > 0.4:
        career_score = min(career_score * 1.10, 1.0)

    # 3. Company quality
    company_type, company_score = classify_company_v2(company)

    # 4. JD Skill matching
    matched_skills, total_jd_skills, skill_match_score = match_jd_required_skills(skills)

    # 5. Context keywords in summary + all career descriptions
    all_text = summary + " " + headline
    for role in career:
        all_text += " " + role.get("description", "")
    _, context_score = match_jd_context_keywords(all_text)

    # 6. Technical depth
    tech_depth = compute_tech_depth_v2(skills, career_analysis, skill_match_score, context_score)

    # 7. Experience
    exp_score = score_experience_v2(years)

    # 8. Location
    loc_score = score_location_v2(location, country)

    # 9. Education
    edu_score = score_education_v2(education)

    # 10. Behavioral
    behavioral = compute_behavioral_v2(signals)
    beh_multiplier = behavioral_multiplier_v2(behavioral)

    return {
        "candidate_id": candidate.get("candidate_id", ""),
        "title_tier": tier,
        "title_score": title_score,
        "title_category": title_cat,
        "career_score": career_score,
        "career_analysis": career_analysis,
        "company_type": company_type,
        "company_score": company_score,
        "matched_jd_skills": matched_skills,
        "jd_skill_match_score": skill_match_score,
        "context_score": context_score,
        "technical_depth": tech_depth,
        "experience_score": exp_score,
        "location_score": loc_score,
        "education_score": edu_score,
        "behavioral": behavioral,
        "behavioral_multiplier": beh_multiplier,
        "years": years,
        "title": title,
        "company": company,
        "location": location,
        "country": country,
        "notice_days": signals.get("notice_period_days", 90),
        "open_to_work": signals.get("open_to_work_flag", False),
        "response_rate": signals.get("recruiter_response_rate", 0),
    }


# ============================================================================
# STAGE 1: FAST PRE-FILTER (v3 — higher recall)
# ============================================================================

def stage1_prefilter_v2(candidates: List[Dict]) -> List[Tuple[Dict, float, int]]:
    """
    Stage 1: Fast heuristic pre-filter.
    Higher recall (25K) — better to keep borderline candidates for stage 2.
    """
    scored = []

    for idx, cand in enumerate(candidates):
        profile = cand.get("profile", {})
        years = profile.get("years_of_experience", 0)
        title = profile.get("current_title", "")
        company = profile.get("current_company", "")
        signals = cand.get("redrob_signals", {})

        # Hard bounds — slightly relaxed
        if years < 1.0 or years > 28:
            continue

        # Title check — keep more candidates
        tier, title_score, _ = classify_title_v2(title)
        if tier == 1:
            # Tier 1 only kept if strong ML skills
            skills = cand.get("skills", [])
            core_ml = sum(1 for s in skills
                         if any(kw in normalize(s.get("name", ""))
                               for kw in ["machine learning", "deep learning", "nlp",
                                         "tensorflow", "pytorch", "neural", "transformer",
                                         "embedding", "retrieval", "ranking"]))
            if core_ml < 3:
                continue

        company_type, company_score = classify_company_v2(company)

        # Quick score — includes context keywords
        skills_list = cand.get("skills", [])
        skill_count_score = min(len(skills_list) / 12.0, 1.0)

        quick = (
            title_score * 0.35 +
            score_experience_v2(years) * 0.20 +
            company_score * 0.20 +
            skill_count_score * 0.10 +
            signals.get("recruiter_response_rate", 0) * 0.10 +
            (1.0 if signals.get("open_to_work_flag", False) else 0.3) * 0.05
        )

        scored.append((cand, quick, idx))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:STAGE1_TOP_K]
