# Redrob Hackathon — Intelligent Candidate Ranker

**Team:** MindSync - Members-Lakshay Verma (Lead),Khushi Sharma
**Challenge:** Intelligent Candidate Discovery & Ranking  
**Role:** Senior AI Engineer — Founding Team

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the ranker (single command)
python src/rank.py --candidates ./candidates.jsonl.gz --out ./team_xxx.csv

# 3. Validate the submission
python validate_submission.py team_xxx.csv
```

---

## Architecture

### 3-Stage Ranking Pipeline

```
100K Candidates
    |
    v
[Stage 1: Fast Heuristic Pre-Filter]  ~10 seconds
    - Hard disqualifiers (title, experience bounds)
    - Lightweight scoring (title + experience + company type)
    - Reduces to ~15K candidates
    |
    v
[Embedding Computation]  ~90 seconds
    - all-MiniLM-L6-v2 sentence transformer (CPU, 384-dim)
    - Semantic text similarity with JD
    |
    v
[Stage 2: Deep Multi-Signal Scoring]  ~30 seconds
    - Profile scoring (title, career, technical depth, experience, location, education)
    - Semantic similarity from embeddings
    - Honeypot detection (career timeline, skill consistency, keyword stuffing)
    - Behavioral signal multiplier
    - Reduces to ~500 candidates
    |
    v
[Stage 3: Final Ranking]  ~10 seconds
    - Top 100 selection with tie-breaking
    - Specific reasoning generation
    |
    v
[Submission CSV]
```

**Total runtime:** ~2-3 minutes on CPU (within 5-minute constraint)

---

## Key Design Decisions

### 1. JD Understanding
The JD explicitly warns against keyword matching. Our system:
- **Title + Career Trajectory** (28% weight): Current title and company type matter more than listed skills. A "Data Scientist" at a product company with retrieval experience scores higher than an "AI Engineer" title at a consulting firm.
- **Semantic Similarity** (22% weight): Embeddings capture the *meaning* of career descriptions, not just keyword overlap.
- **Technical Depth** (20% weight): Evaluates actual skill quality (proficiency × duration × endorsements) and penalizes keyword stuffing.

### 2. Behavioral Signals as Multiplier
Instead of additive scoring, behavioral signals act as a **multiplier** (0.1 to 1.0):
- A candidate with perfect profile but 5% response rate and 6-month inactivity gets heavily penalized
- This matches the JD's emphasis: "a perfect-on-paper candidate who hasn't logged in for 6 months is not actually available"

### 3. Honeypot Detection
Multi-check system identifies impossible profiles:
- Career timeline validation (overlapping employment, experience vs graduation year)
- Company founding year checks (8 years at a 3-year-old company)
- Expert skills with zero duration
- Title-description mismatches
- Keyword stuffing detection

### 4. Reasoning Generation
Dynamic reasoning constructed from actual candidate data:
- Specific facts (years, title, company, skills)
- JD connection (why this candidate fits)
- Honest concerns (notice period, location, gaps)
- Rank-appropriate tone (top ranks highlight strengths, lower ranks note concerns)

---

## Repository Structure

```
.
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── submission_metadata.yaml      # Filled metadata template
├── src/
│   ├── rank.py                   # Main entry point
│   ├── config.py                 # JD understanding + scoring weights
│   ├── features.py               # Feature extraction engine
│   ├── embeddings.py             # Text embedding module
│   ├── honeypot_detector.py      # Honeypot/impossible profile detection
│   ├── scorer.py                 # Multi-signal scoring engine
│   └── reasoning.py              # Reasoning generator
└── validate_submission.py        # Format validator (from hackathon bundle)
```

---

## Reproduction

### Prerequisites
- Python 3.10+
- 16 GB RAM
- CPU only (no GPU needed)
- No network during ranking

### Steps
```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the ranker
python src/rank.py --candidates ./candidates.jsonl.gz --out ./team_xxx.csv
```

---

## Compute Environment

- **Platform:** [e.g., MacBook Pro M3 / AWS EC2 / Local Linux]
- **CPU:** [e.g., 8 cores]
- **RAM:** 16 GB
- **Python:** 3.11
- **Runtime:** ~2-3 minutes for 100K candidates

---

## Methodology Summary

Multi-stage ranker combining semantic embeddings with structured heuristic scoring. Stage 1 uses fast title/experience/company-type heuristics to reduce 100K to 15K candidates. Stage 2 computes all-MiniLM-L6-v2 embeddings for semantic JD similarity, then applies a 6-component profile score (title/career, semantic similarity, technical depth, experience, location, education) with a multiplicative behavioral-signal modifier. Honeypot detection runs 7 independent checks (timeline consistency, company founding years, skill-proficiency mismatches, title-description alignment, keyword stuffing, education-experience consistency, suspicious career jumps). The title/career component is the decisive signal against keyword-stuffer traps; an endorsement-and-duration trust multiplier on skills catches lazy keyword stuffing. Reasoning is dynamically generated from candidate-specific facts with rank-appropriate tone. Runtime is ~2-3 minutes for 100K candidates on CPU.

---

## AI Tools Used

- **Claude:** Architecture discussion, code structure, and debugging
- **GitHub Copilot:** Autocomplete for boilerplate

No candidate data was fed to any LLM API during ranking. All ranking is done locally.

---

## Notes

- The ranking code does NOT make any external API calls
- All embeddings are computed locally using sentence-transformers
- The system is fully reproducible with a single command
- Random seed is fixed (42) for reproducibility

---

## Author

- Lakshay Verma (Team Lead)
- Khushi Sharma