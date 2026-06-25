"""
Embedding Module v3 — BGE-base with query prefix for better retrieval.

Key changes from v2:
- BGE-base-en-v1.5 (768-dim) instead of BGE-small (384-dim)
- BGE requires "Represent this sentence: " prefix for passages
- Query prefix for JD embedding
- Fallback to bge-small if bge-base is unavailable
"""

import numpy as np
from typing import List

_model = None
_model_name = None


def _load_model():
    global _model, _model_name
    if _model is not None:
        return _model

    from config import EMBEDDING_MODEL
    from sentence_transformers import SentenceTransformer
    import torch

    # Try bge-base first, fallback to bge-small
    models_to_try = [
        EMBEDDING_MODEL,
        "BAAI/bge-small-en-v1.5",
        "all-MiniLM-L6-v2",
    ]

    for model_name in models_to_try:
        try:
            print(f"[Embeddings] Loading {model_name}...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _model = SentenceTransformer(model_name, device=device)
            _model_name = model_name
            print(f"[Embeddings] Loaded {model_name} on {device}")
            return _model
        except Exception as e:
            print(f"[Embeddings] Failed to load {model_name}: {e}")
            continue

    raise RuntimeError("Could not load any embedding model!")


def _get_bge_prefix(is_query: bool = False) -> str:
    """BGE models use different prefixes for queries vs passages."""
    if _model_name and "bge" in _model_name.lower():
        return "Represent this sentence for searching relevant passages: " if is_query else ""
    return ""


def compute_embedding(text: str, is_query: bool = True) -> np.ndarray:
    """Compute single embedding (for JD — treated as query)."""
    model = _load_model()
    prefix = _get_bge_prefix(is_query)
    full_text = prefix + text if prefix else text
    emb = model.encode(full_text, normalize_embeddings=True, show_progress_bar=False)
    return np.array(emb)


def compute_embeddings_batch(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """Compute embeddings for candidate passages (not queries)."""
    model = _load_model()
    prefix = _get_bge_prefix(is_query=False)

    if prefix:
        texts = [prefix + t for t in texts]

    embs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return np.array(embs)


def compute_similarity_scores(
    candidate_embeddings: np.ndarray,
    jd_embedding: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarities (already normalized, so dot product)."""
    # Both are L2-normalized, so cosine sim = dot product
    sims = candidate_embeddings @ jd_embedding
    # Shift from [-1,1] to [0,1] range
    sims = (sims + 1.0) / 2.0
    return sims
