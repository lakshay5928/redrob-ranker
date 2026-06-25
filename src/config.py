"""
JD Configuration & Scoring Weights for Redrob Hackathon Ranker v3.

Key changes from v2:
- Better embedding model (bge-base for better retrieval quality)
- Refined scoring weights tuned for NDCG@10 and NDCG@50
- Expanded skill taxonomy
- Better company detection
- Production vs research distinction boosted
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

# ============================================================================
# EMBEDDING MODEL
# ============================================================================
# BGE-base gives better quality than bge-small at acceptable speed
# Fallback to bge-small if memory/time is tight
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM = 768  # bge-base uses 768-dim

# ============================================================================
# JD TEXT (for embedding similarity) - Enriched for better retrieval
# ============================================================================

JD_TEXT = """Senior AI Engineer Founding Team. Redrob AI Series A AI-native talent intelligence platform.
Pune Noida India Hybrid. Open to relocation from Tier-1 Indian cities. Full-time. 5 to 9 years experience.

Deep technical depth in modern ML systems: embeddings, retrieval, ranking, LLMs, fine-tuning, combined with scrappy product-engineering attitude. 
Own the intelligence layer: ranking, retrieval, and matching systems that decide what recruiters see when they search for candidates and what candidates see when they search for roles.

Must-have skills and experience:
Production experience with embeddings-based retrieval systems using sentence-transformers, OpenAI embeddings, BGE, E5 deployed to real users. 
Experience with embedding drift, index refresh, retrieval-quality regression in production.
Production experience with vector databases or hybrid search infrastructure: Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS.
Strong Python programming with code quality focus.
Hands-on experience designing evaluation frameworks for ranking systems including NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation.

Nice-to-have skills:
LLM fine-tuning experience with LoRA, QLoRA, PEFT.
Learning-to-rank models using XGBoost or neural approaches.
HR-tech, recruiting tech, or marketplace products exposure.
Distributed systems or large-scale inference optimization.
Open-source contributions in AI/ML.

Ideal candidate profile:
6 to 8 years total experience, of which 4 to 5 are in applied ML AI roles at product companies not pure services.
Has shipped at least one end-to-end ranking, search, or recommendation system to real users at meaningful scale.
Strong opinions about retrieval hybrid vs dense, evaluation offline vs online, and LLM integration when to fine-tune vs prompt.
Located in or willing to relocate to Noida or Pune India.

Explicit disqualifiers:
Pure research environments without production deployment.
Recent under 12 months AI experience of LangChain plus OpenAI calls without pre-LLM-era ML production experience.
Senior engineer who has not written production code in last 18 months.
Title-chasers switching companies every 1.5 years.
People who have only worked at consulting firms: TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini.
Primary expertise is computer vision, speech, or robotics without significant NLP or information retrieval exposure.
Work entirely on closed-source proprietary systems for 5+ years without external validation.
"""

# ============================================================================
# JD KEY REQUIREMENTS (for explicit matching) — EXPANDED
# ============================================================================

JD_REQUIRED_SKILLS: List[str] = [
    # Core retrieval/search
    "embeddings", "retrieval", "ranking", "vector search", "vector database", "semantic search",
    "information retrieval", "hybrid search", "dense retrieval", "sparse retrieval",
    # Vector DBs
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch", "faiss",
    "chroma", "pgvector", "redis vector", "typesense",
    # Embedding models
    "sentence-transformers", "sentence transformers", "bge", "e5", "openai embeddings",
    "bi-encoder", "cross-encoder", "text embeddings",
    # Evaluation
    "ndcg", "mrr", "map", "mean average precision", "evaluation framework", "a/b test",
    "offline evaluation", "online evaluation", "retrieval quality",
    # LLMs
    "llm", "large language model", "fine-tuning", "lora", "peft", "qlora",
    "rag", "retrieval augmented generation", "instruction tuning",
    # ML/DL
    "python", "pytorch", "tensorflow", "transformers", "huggingface",
    "machine learning", "deep learning", "neural network",
    # Ranking
    "learning to rank", "ltr", "xgboost", "lightgbm", "recommendation system",
    "candidate retrieval", "two-stage ranking",
]

JD_NICE_TO_HAVE_SKILLS: List[str] = [
    "distributed systems", "inference optimization", "model serving", "triton", "onnx",
    "hr-tech", "recruiting tech", "talent intelligence", "marketplace",
    "open source", "github contributions",
    "spark", "kafka", "airflow", "kubernetes", "docker",
    "bm25", "solr", "lucene",
]

JD_CONTEXT_KEYWORDS: List[str] = [
    # Production signals
    "production", "deployed", "shipped", "launched", "live system", "real users",
    "scale", "end-to-end", "at scale", "millions", "billion",
    # Product signals
    "product company", "startup", "series", "funded", "growth stage",
    # Technical depth
    "architecture", "designed", "built", "led", "owned",
    "benchmark", "latency", "throughput", "precision@",
    # Search/retrieval specific
    "recruiter", "candidate matching", "talent", "search relevance",
    "index refresh", "embedding drift", "retrieval quality regression",
    "a/b testing", "offline eval", "feedback loop",
    "two-tower", "bi-encoder", "reranking", "reranker",
]

# Strong production keywords for career descriptions
PRODUCTION_KEYWORDS: List[str] = [
    "deployed", "production", "shipped", "launched", "built and deployed",
    "real users", "live", "serving", "inference", "at scale",
    "improved", "increased", "reduced", "optimized",
]

# ============================================================================
# TITLE CLASSIFICATION (Granular with Seniority)
# ============================================================================

# Tier 5: Perfect match titles - AI/ML Engineering at Senior+ level
TIER5_TITLES: Set[str] = {
    "senior ai engineer", "staff ai engineer", "principal ai engineer",
    "senior ml engineer", "senior machine learning engineer",
    "staff ml engineer", "staff machine learning engineer",
    "principal ml engineer", "principal machine learning engineer",
    "lead ai engineer", "lead ml engineer", "lead machine learning engineer",
    "applied scientist", "senior applied scientist", "staff applied scientist",
    "principal applied scientist", "research engineer", "senior research engineer",
    "ml scientist", "ai scientist", "senior ml scientist", "senior ai scientist",
    "senior nlp engineer", "staff nlp engineer", "principal nlp engineer",
    "senior search engineer", "senior recommendation engineer",
    "senior ranking engineer", "senior deep learning engineer",
    "ai architect", "ml architect", "machine learning architect",
    "founding ai engineer", "founding ml engineer",
    # Additional titles common in India
    "senior data scientist", "lead data scientist",
    "applied ml engineer", "senior applied ml engineer",
    "ai/ml engineer", "senior ai/ml engineer",
    "nlp scientist", "senior nlp scientist",
    "information retrieval engineer", "search relevance engineer",
}

# Tier 4: Strong match
TIER4_TITLES: Set[str] = {
    "ai engineer", "ml engineer", "machine learning engineer",
    "nlp engineer", "search engineer", "recommendation engineer",
    "ranking engineer", "deep learning engineer",
    "data scientist", "research scientist",
    "algorithm engineer", "senior algorithm engineer",
    "ai researcher", "ml researcher",
    "ai developer", "ml developer",
    "computer vision engineer",
    "software engineer machine learning", "software engineer ml",
    "engineer ii ml", "engineer iii ml",
    "mlops engineer", "senior mlops engineer",
}

# Tier 3: Moderate match - Senior SWE with ML, or ML-adjacent
TIER3_TITLES: Set[str] = {
    "software engineer", "senior software engineer", "staff software engineer",
    "principal software engineer",
    "backend engineer", "senior backend engineer",
    "platform engineer", "senior platform engineer",
    "infrastructure engineer",
    "data engineer", "senior data engineer",
    "analytics engineer",
    "full stack engineer", "senior full stack engineer",
}

# Tier 2: Weak match
TIER2_TITLES: Set[str] = {
    "backend developer", "software developer",
    "data analyst", "senior data analyst",
    "solutions architect", "technical architect",
}

# Tier 1: Very weak
TIER1_TITLES: Set[str] = {
    "marketing manager", "product manager", "project manager", "program manager",
    "operations manager", "hr manager", "human resources manager",
    "accountant", "financial analyst", "business analyst",
    "content writer", "technical writer",
    "graphic designer", "ui designer", "ux designer",
    "sales executive", "sales manager", "business development",
    "customer support", "customer success",
    "civil engineer", "mechanical engineer", "electrical engineer",
    "quality assurance", "qa engineer", "test engineer",
    "network engineer", "system administrator", "database administrator",
    "consultant", "analyst", "associate", "trainee", "intern",
    "devops engineer", "cloud engineer", "security engineer",
}

# ============================================================================
# COMPANY CLASSIFICATION
# ============================================================================

CONSULTING_COMPANIES: Set[str] = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "mindtree", "tech mahindra", "hcl", "genpact", "ibm consulting",
    "deloitte", "ernst", "kpmg", "pwc", "ey",
    "lti", "ltimindtree", "mphasis", "persistent",
    "birlasoft", "hexaware", "zensar", "cyient",
    "cgi", "ntt data", "softtek",
    "virtusa", "syntel", "collabera",
    # Additional
    "mastech", "niit technologies", "rackspace", "dxc", "unisys",
    "kforce", "stefanini", "softchoice", "solugenix",
}

STRONG_PRODUCT_COMPANIES: Set[str] = {
    # Global tech
    "google", "microsoft", "amazon", "meta", "facebook", "apple", "netflix",
    "uber", "airbnb", "stripe", "spotify", "twitter", "linkedin",
    "nvidia", "openai", "anthropic", "cohere", "huggingface",
    "salesforce", "adobe", "intuit", "square", "paypal", "shopify",
    "databricks", "snowflake", "elastic", "confluent", "mongodb",
    "atlassian", "github", "gitlab", "hashicorp", "twilio",
    # Indian product
    "flipkart", "swiggy", "zomato", "ola", "paytm", "phonepe",
    "razorpay", "freshworks", "zoho", "postman", "redrob",
    "instahyre", "unstop", "naukri", "foundit", "apna",
    "cred", "groww", "zerodha", "upstox",
    "meesho", "sharechat", "moj", "dailyhunt",
    "lenskart", "nykaa", "urban company", "vedantu",
    "byju", "unacademy", "whitehat jr",
    "dream11", "mpl", "games24x7",
    "indiamart", "justdial",
    "browserstack", "clevertap", "chargebee", "setu",
    "darwinbox", "leadsquared", "capillary", "mediamath",
    "hasura", "dgraph", "sarvam", "krutrim",
    # Mid-tier product
    "walmart labs", "target", "booking", "expedia", "tripadvisor",
    "quora", "medium", "canva", "figma", "notion",
    "mixpanel", "amplitude", "segment", "braze",
}

STARTUP_INDICATORS: Set[str] = {
    "ai", "ml", "deep", "neural", "cognitive", "brain",
    "intelligence", "bot", "automation", "predictive",
    "analytics", "data", "search", "recommend",
    "labs", "ventures", "innovations", "technologies",
    "tech", "systems", "solutions",
}

# ============================================================================
# EXPERIENCE PARAMETERS
# ============================================================================

IDEAL_YEARS_MIN = 5.0
IDEAL_YEARS_MAX = 9.0
IDEAL_YEARS_TARGET = 6.5
VERY_HIGH_YEARS = 15.0
TOO_JUNIOR = 3.0

# ============================================================================
# LOCATION PARAMETERS
# ============================================================================

PREFERRED_LOCATIONS: Set[str] = {
    "pune", "noida", "gurgaon", "gurugram", "hyderabad",
    "mumbai", "bangalore", "bengaluru", "chennai", "delhi",
    "faridabad", "ghaziabad", "greater noida", "new delhi",
    "navi mumbai", "thane",
}

INDIA_COUNTRY = "india"

# ============================================================================
# STAGE PARAMETERS
# ============================================================================

STAGE1_TOP_K = 25000   # Increased recall for stage 1 (was 20K)
STAGE2_TOP_K = 500
FINAL_TOP_K = 100

# ============================================================================
# SCORING WEIGHTS (v3 — tuned for NDCG@10 & NDCG@50)
# ============================================================================

@dataclass
class ScoringWeights:
    """
    v3.1 weights: career + semantic co-equal, tech depth strong.
    Education removed from weighted sum (only used for tie-break).
    """
    title_career: float = 0.30        # Title match + career trajectory (boosted)
    semantic_similarity: float = 0.25 # Embedding similarity (bge-base quality)
    technical_depth: float = 0.24     # Skills + production signals (boosted)
    experience_quality: float = 0.11  # Years (JD: 5-9)
    company_quality: float = 0.07     # Product vs consulting
    location_match: float = 0.03      # India preferred

# Behavioral multiplier range — kept tight (profile first)
BEHAVIORAL_MULTIPLIER_RANGE = (0.80, 1.0)  # Tighter — behavioral is secondary
