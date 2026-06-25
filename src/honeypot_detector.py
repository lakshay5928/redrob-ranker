"""
Honeypot & Impossible Profile Detector v2.

Identifies candidates with subtly impossible profiles.
Ground truth forces these to Tier 0 — >10% in top 100 = disqualification.
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Tuple

from features import classify_title_v2, normalize

# Known company founding years for timeline validation
FOUNDING_YEARS = {
    "google": 1998, "microsoft": 1975, "amazon": 1994, "meta": 2004,
    "facebook": 2004, "apple": 1976, "netflix": 1997, "uber": 2009,
    "airbnb": 2008, "stripe": 2010, "spotify": 2006, "twitter": 2006,
    "linkedin": 2003, "nvidia": 1993, "openai": 2015, "anthropic": 2021,
    "salesforce": 1999, "adobe": 1982, "flipkart": 2007, "swiggy": 2014,
    "zomato": 2008, "ola": 2010, "paytm": 2010, "phonepe": 2015,
    "razorpay": 2014, "freshworks": 2010, "zoho": 1996, "postman": 2014,
    "tcs": 1968, "infosys": 1981, "wipro": 1945, "accenture": 1989,
    "cognizant": 1994, "capgemini": 1967, "mindtree": 1999,
    "tech mahindra": 1986, "hcl": 1976, "redrob": 2021,
    "byju": 2011, "unacademy": 2015, "cred": 2018, "groww": 2016,
    "zerodha": 2010, "meesho": 2015, "sharechat": 2015,
}


def detect_honeypot(candidate: Dict) -> Tuple[bool, float, List[str]]:
    """Returns: (is_honeypot, confidence, reasons)"""
    reasons = []
    confidences = []
    
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    education = candidate.get("education", [])
    years = profile.get("years_of_experience", 0)
    
    # Check 1: Timeline consistency
    issue, conf, rs = _check_timeline(career, years)
    if issue:
        confidences.append(conf)
        reasons.extend(rs)
    
    # Check 2: Company founding years
    issue, conf, rs = _check_founding_years(career)
    if issue:
        confidences.append(conf)
        reasons.extend(rs)
    
    # Check 3: Expert skills with zero duration
    issue, conf, rs = _check_skill_consistency(skills)
    if issue:
        confidences.append(conf)
        reasons.extend(rs)
    
    # Check 4: Title-description mismatch
    issue, conf, rs = _check_title_mismatch(career)
    if issue:
        confidences.append(conf)
        reasons.extend(rs)
    
    # Check 5: Keyword stuffing
    issue, conf, rs = _check_keyword_stuffing(candidate, skills)
    if issue:
        confidences.append(conf)
        reasons.extend(rs)
    
    # Check 6: Education-experience mismatch
    issue, conf, rs = _check_edu_exp(education, years)
    if issue:
        confidences.append(conf)
        reasons.extend(rs)
    
    # Check 7: Impossible career jumps
    issue, conf, rs = _check_career_jumps(career)
    if issue:
        confidences.append(conf)
        reasons.extend(rs)
    
    # Aggregate
    if not confidences:
        return False, 0.0, reasons
    
    max_conf = max(confidences)
    flag_count = len(confidences)
    
    if max_conf >= 0.8:
        return True, max_conf, reasons
    elif flag_count >= 2 and sum(confidences) / flag_count >= 0.5:
        return True, sum(confidences) / flag_count, reasons
    elif flag_count >= 3:
        return True, 0.5, reasons
    
    return False, 0.0, reasons


def compute_honeypot_penalty(candidate: Dict) -> float:
    """Returns 0.0 if honeypot (exclude), 0.3 if suspicious, 1.0 if clean."""
    is_hp, conf, _ = detect_honeypot(candidate)
    if is_hp and conf >= 0.5:
        return 0.0
    elif is_hp:
        return 0.3
    return 1.0


# ============================================================================
# Individual Checks
# ============================================================================

def _check_timeline(career: List[Dict], total_years: float) -> Tuple[bool, float, List[str]]:
    reasons = []
    months = sum(r.get("duration_months", 0) for r in career)
    from_history = months / 12.0
    discrepancy = abs(from_history - total_years)
    
    if discrepancy > 3:
        reasons.append(f"Timeline: {from_history:.1f}y in history vs {total_years:.1f}y claimed")
        return True, min(discrepancy / 5.0, 1.0), reasons
    
    # Overlaps
    dates = []
    for r in career:
        s, e = r.get("start_date", ""), r.get("end_date") or "2030-01-01"
        if s:
            try:
                sd = datetime.strptime(s, "%Y-%m-%d")
                ed = datetime.strptime(e, "%Y-%m-%d") if e else datetime.now()
                dates.append((sd, ed, r.get("company", "")))
            except:
                pass
    
    for i in range(len(dates)):
        for j in range(i + 1, len(dates)):
            s1, e1, c1 = dates[i]
            s2, e2, c2 = dates[j]
            if s1 < e2 and s2 < e1:
                overlap = (min(e1, e2) - max(s1, s2)).days / 30.0
                if overlap > 3:
                    reasons.append(f"Overlap: {c1} and {c2} by {overlap:.0f} months")
                    return True, min(overlap / 12.0, 1.0), reasons
    
    return False, 0.0, reasons


def _check_founding_years(career: List[Dict]) -> Tuple[bool, float, List[str]]:
    reasons = []
    max_conf = 0.0
    
    for r in career:
        company = normalize(r.get("company", ""))
        duration = r.get("duration_months", 0)
        start = r.get("start_date", "")
        if not company or not start:
            continue
        
        for known, founded in FOUNDING_YEARS.items():
            if known in company or company in known:
                try:
                    sy = datetime.strptime(start, "%Y-%m-%d").year
                    age = sy - founded
                    if age < -1:
                        reasons.append(f"Joined {company} in {sy}, founded {founded}")
                        return True, 1.0, reasons
                    yrs = duration / 12.0
                    if yrs > age + 1 and age >= 0:
                        conf = min(0.9, yrs / 10)
                        max_conf = max(max_conf, conf)
                        reasons.append(f"{yrs:.1f}y at {company} but only {age}y old")
                except:
                    pass
    
    if reasons:
        return True, max_conf, reasons
    return False, 0.0, reasons


def _check_skill_consistency(skills: List[Dict]) -> Tuple[bool, float, List[str]]:
    reasons = []
    count = 0
    for s in skills:
        name = s.get("name", "")
        prof = s.get("proficiency", "")
        dur = s.get("duration_months", 0)
        endo = s.get("endorsements", 0)
        
        if prof in ["expert", "advanced"] and dur < 3:
            count += 1
            reasons.append(f"'{name}' as {prof} with {dur}mo exp")
        if prof == "expert" and endo == 0 and dur < 12:
            count += 1
    
    if count >= 3:
        return True, min(count / 5.0, 1.0), reasons
    return False, count * 0.2, reasons


def _check_title_mismatch(career: List[Dict]) -> Tuple[bool, float, List[str]]:
    reasons = []
    mismatch = 0
    
    for r in career:
        title = normalize(r.get("title", ""))
        desc = normalize(r.get("description", ""))
        if not title or not desc:
            continue
        
        tier, score, _ = classify_title_v2(title)
        if score >= 0.5:
            ml_terms = ["machine learning", "ml", "ai ", "model", "algorithm",
                       "prediction", "classification", "neural", "deep learning",
                       "training", "inference", "embedding", "vector", "retrieval",
                       "ranking", "recommendation", "nlp", "feature", "pipeline", "deploy"]
            has_ml = any(t in desc for t in ml_terms)
            
            if not has_ml:
                non_ml = ["accounting", "hr ", "sales", "marketing", "customer support",
                         "graphic design", "civil engineering", "mechanical engineering"]
                if any(ind in desc for ind in non_ml):
                    mismatch += 1
                    reasons.append(f"'{r.get('title', '')}' describes non-ML work")
    
    if mismatch >= 2:
        return True, min(mismatch / 3.0, 1.0), reasons
    return False, mismatch * 0.3, reasons


def _check_keyword_stuffing(candidate: Dict, skills: List[Dict]) -> Tuple[bool, float, List[str]]:
    reasons = []
    indicators = 0
    
    expert_count = sum(1 for s in skills if s.get("proficiency") == "expert")
    if expert_count > 8:
        indicators += 1
        reasons.append(f"{expert_count} expert-level skills")
    
    total_endo = sum(s.get("endorsements", 0) for s in skills)
    if len(skills) > 15 and total_endo < len(skills):
        indicators += 1
        reasons.append(f"{len(skills)} skills, {total_endo} endorsements")
    
    title = normalize(candidate.get("profile", {}).get("current_title", ""))
    tier, score, _ = classify_title_v2(title)
    if score < 0.3 and len(skills) > 10:
        tech = sum(1 for s in skills if any(kw in normalize(s.get("name", ""))
                   for kw in ["python", "tensorflow", "pytorch", "ml", "ai",
                             "neural", "deep learning", "nlp", "vector"]))
        if tech > 5:
            indicators += 1
            reasons.append(f"Non-tech '{title}' with {tech} advanced skills")
    
    zero_endo = sum(1 for s in skills if s.get("endorsements", 0) == 0)
    if zero_endo > len(skills) * 0.7 and len(skills) > 8:
        indicators += 1
        reasons.append(f"{zero_endo}/{len(skills)} skills with 0 endorsements")
    
    if indicators >= 2:
        return True, min(indicators / 3.0, 1.0), reasons
    return False, indicators * 0.3, reasons


def _check_edu_exp(education: List[Dict], total_years: float) -> Tuple[bool, float, List[str]]:
    reasons = []
    if not education:
        return False, 0.0, reasons
    
    latest = max((e.get("end_year", 0) for e in education), default=0)
    if latest == 0:
        return False, 0.0, reasons
    
    since = 2026 - latest
    if total_years > since + 2:
        reasons.append(f"{total_years:.1f}y experience but graduated {latest}")
        return True, min((total_years - since) / 5.0, 1.0), reasons
    
    return False, 0.0, reasons


def _check_career_jumps(career: List[Dict]) -> Tuple[bool, float, List[str]]:
    reasons = []
    if len(career) < 2:
        return False, 0.0, reasons
    
    titles = [normalize(r.get("title", "")) for r in reversed(career) if r.get("title")]
    if len(titles) < 2:
        return False, 0.0, reasons
    
    nontech = ["accountant", "hr manager", "sales executive", "marketing manager",
               "customer support", "content writer", "graphic designer",
               "civil engineer", "mechanical engineer", "operations manager"]
    tech = ["ai engineer", "ml engineer", "machine learning engineer",
            "data scientist", "software engineer", "nlp engineer"]
    
    for i in range(len(titles) - 1):
        prev_nt = any(nt in titles[i] for nt in nontech)
        curr_t = any(t in titles[i + 1] for t in tech)
        if prev_nt and curr_t and i + 1 == len(titles) - 1:
            reasons.append(f"Jump from '{titles[i]}' to '{titles[i+1]}'")
            return True, 0.7, reasons
    
    return False, 0.0, reasons
