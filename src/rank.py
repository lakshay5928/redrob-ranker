#!/usr/bin/env python3
"""
Redrob Hackathon — Intelligent Candidate Ranker v3 (Score-Optimized).

Key improvements over v2:
- BGE-base-en-v1.5 embeddings (768-dim, better retrieval quality)
- BGE query prefix for JD embedding
- Production signal detection in career descriptions
- Expanded skill taxonomy with aliases
- Tighter behavioral multiplier floor (0.80)
- Higher stage-1 recall (25K candidates)

Usage:
    python src/rank.py --candidates ./candidates.jsonl.gz --out ./team_xxx.csv
"""

import argparse
import csv
import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import JD_TEXT, FINAL_TOP_K
from embeddings import compute_embedding
from reasoning import generate_all_reasonings
from scorer import rank_candidates_v2


def load_candidates(path: str):
    cands = []
    opener = gzip.open if path.endswith(".gz") else open
    mode = "rt" if path.endswith(".gz") else "r"
    with opener(path, mode, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cands.append(json.loads(line))
    return cands


def write_submission(ranked, out_path, reasonings):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, ((cand, score, feats), reasoning) in enumerate(zip(ranked, reasonings), 1):
            writer.writerow([cand.get("candidate_id", ""), rank, f"{score:.4f}", reasoning])
    print(f"[Submission] Wrote {len(ranked)} rows to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker v3")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl[.gz]")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    start = time.time()
    print("=" * 60)
    print("Redrob Hackathon — Candidate Ranker v3 (Score-Optimized)")
    print("=" * 60)

    # Load
    t0 = time.time()
    print("\n[Step 1] Loading candidates...")
    cands = load_candidates(args.candidates)
    print(f"[Step 1] Loaded {len(cands)} candidates in {time.time()-t0:.1f}s")

    # JD embedding (with query prefix for BGE)
    t0 = time.time()
    print("\n[Step 2] Computing JD embedding...")
    from config import EMBEDDING_MODEL
    print(f"[Step 2] Using model: {EMBEDDING_MODEL}")
    jd_emb = compute_embedding(JD_TEXT, is_query=True)
    print(f"[Step 2] Done in {time.time()-t0:.1f}s")

    # Rank
    t0 = time.time()
    print("\n[Step 3] Running ranking pipeline v3...")
    ranked = rank_candidates_v2(cands, jd_emb)
    print(f"[Step 3] Done in {time.time()-t0:.1f}s")

    # Reasoning
    t0 = time.time()
    print("\n[Step 4] Generating reasoning...")
    reasonings = generate_all_reasonings(ranked)
    print(f"[Step 4] Done in {time.time()-t0:.1f}s")

    # Write
    t0 = time.time()
    print("\n[Step 5] Writing submission...")
    write_submission(ranked, args.out, reasonings)
    print(f"[Step 5] Done in {time.time()-t0:.1f}s")

    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"Total time: {elapsed:.1f}s")
    print(f"Output: {args.out}")
    if ranked:
        top_profile = ranked[0][0].get("profile", {})
        print(f"Top candidate: {top_profile.get('current_title', 'N/A')} | score: {ranked[0][1]:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
