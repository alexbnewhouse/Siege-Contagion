"""02b – Dictionary-Based Siege Scoring for /pol/.

Applies the same Siege lexicon from 02_siege_lexicon.py to the
pre-filtered /pol/ posts. Produces siege_keyword_score, siege_binary,
and word_count columns.

This is separated from the IM lexicon stage because:
  - /pol/ data lives in a different parquet file
  - no need for embedding-based context adjustment on the first pass
    (the pre-filter already ensures relevance)
  - output goes to pol_siege_scores.parquet (not siege_scores.parquet)

The scoring uses the same SIEGE_DICTIONARY and compute_siege_keyword_score
function, ensuring score comparability across platforms.
"""

from __future__ import annotations

import multiprocessing
import os

import polars as pl

from utils import DATA_PROCESSED

# Import the shared scoring function from the IM lexicon module
from importlib import import_module

_N_WORKERS = min(multiprocessing.cpu_count(), int(os.environ.get("SIEGE_WORKERS", "16")))


def main():
    print("=" * 60)
    print("PHASE 1a-pol: Dictionary-Based Siege Scoring (/pol/)")
    print("=" * 60)

    lex = import_module("02_siege_lexicon")

    pol_path = DATA_PROCESSED / "pol_posts.parquet"
    if not pol_path.exists():
        print(f"  ✗ {pol_path} not found. Run 01b_preprocess_pol first.")
        return

    pol = pl.read_parquet(pol_path)
    print(f"  /pol/ posts: {pol.height:,}")

    # ── Score posts ───────────────────────────────────────────────────
    print("\n  Scoring /pol/ posts with Siege lexicon…")
    texts = pol["text"].to_list()

    with multiprocessing.Pool(_N_WORKERS) as pool:
        results = pool.map(lex.compute_siege_keyword_score, texts, chunksize=1024)

    pol = pol.with_columns([
        pl.Series("siege_keyword_count", [r["keyword_count"] for r in results], dtype=pl.Int32),
        pl.Series("siege_keyword_score", [r["keyword_score"] for r in results], dtype=pl.Float64),
        pl.Series("siege_keyword_density", [r["keyword_density"] for r in results], dtype=pl.Float64),
        pl.Series("siege_binary", [1 if r["keyword_score"] > 0 else 0 for r in results], dtype=pl.Int8),
        pl.Series("siege_keyword_context_score", [r["keyword_context_score"] for r in results], dtype=pl.Float64),
        pl.Series("word_count", [len((t or "").split()) for t in texts], dtype=pl.Int32),
    ])

    siege_posts = pol.filter(pl.col("siege_binary") == 1)
    print(f"  Posts with siege terms: {siege_posts.height:,} / {pol.height:,} "
          f"({100 * siege_posts.height / pol.height:.1f}%)")

    # ── Build unified format ─────────────────────────────────────────
    score_cols = [
        "siege_keyword_count", "siege_keyword_score",
        "siege_keyword_density", "siege_binary",
        "siege_keyword_context_score", "word_count",
    ]

    pol_unified = pol.select([
        pl.col("post_id"),
        pl.col("author_id"),
        pl.col("date"),
        pl.col("text"),
        pl.col("thread_id"),
        pl.lit("pol").alias("channel"),
        pl.lit("pol").alias("platform"),
        *[pl.col(c) for c in score_cols],
    ])

    out_path = DATA_PROCESSED / "pol_siege_scores.parquet"
    pol_unified.write_parquet(out_path)
    print(f"\n  Saved {out_path.name} ({pol_unified.height:,} rows)")

    # ── Summary stats ─────────────────────────────────────────────────
    mean_score = pol_unified["siege_keyword_score"].mean()
    max_score = pol_unified["siege_keyword_score"].max()
    print(f"  Mean siege score: {mean_score:.4f}")
    print(f"  Max siege score:  {max_score:.2f}")

    print("\n✓ /pol/ dictionary scoring complete.")


if __name__ == "__main__":
    main()
