"""03 – Embedding-Based Siege Similarity Scoring.

Uses sentence-transformers to compute semantic similarity to a Siege reference
corpus centroid, providing a continuous siege-proximity measure.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from utils import DATA_PROCESSED, ZEIGER_MEMBER_ID

# Lazy import to avoid circular; used only in main()
_apply_embedding_boost = None

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 512
MAX_SEQ_LEN = 256  # tokens; truncate long posts


def build_siege_reference_corpus(
    forum_posts: pl.DataFrame,
    search_tags: pl.DataFrame,
    forums_topics: pl.DataFrame,
) -> list[str]:
    """Construct the Siege reference corpus from:
    (a) Zeiger posts with siege dictionary terms
    (b) Posts in siege-tagged threads
    """
    # (a) Zeiger posts with siege terms
    zeiger_siege = forum_posts.filter(
        (pl.col("author_id") == ZEIGER_MEMBER_ID)
        & (pl.col("siege_binary") == 1)
    )
    print(f"  Zeiger siege posts: {zeiger_siege.height}")

    # (b) Posts in siege-tagged topics
    siege_tags = search_tags.filter(
        pl.col("index_tag").str.to_lowercase().str.contains(
            r"siege|james.mason|siegeculture"
        )
    )
    siege_topic_ids = siege_tags["index_id"].unique().to_list()

    # Cross-reference: index_id may map to topic_firstpost in forums_topics
    siege_topic_posts = forum_posts.filter(
        pl.col("topic_id").is_in(siege_topic_ids)
    )
    print(f"  Posts in siege-tagged topics: {siege_topic_posts.height}")

    # Combine and deduplicate
    combined = pl.concat([zeiger_siege, siege_topic_posts]).unique(subset=["pid"])
    texts = combined["text"].drop_nulls().to_list()
    texts = [t for t in texts if len(t.strip()) > 20]
    print(f"  Total reference corpus size: {len(texts)}")
    return texts


def compute_embeddings(
    model: SentenceTransformer,
    texts: list[str],
    desc: str = "Encoding",
) -> np.ndarray:
    """Compute sentence embeddings in batches with progress."""
    print(f"  {desc}: {len(texts)} texts…")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings


def main():
    print("=" * 60)
    print("PHASE 1b: Embedding-Based Siege Scoring")
    print("=" * 60)

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"  Using device: {device}")

    model = SentenceTransformer(MODEL_NAME, device=device)

    # ── Load data ─────────────────────────────────────────────────────
    fp = pl.read_parquet(DATA_PROCESSED / "forum_posts.parquet")
    dm = pl.read_parquet(DATA_PROCESSED / "dm_posts.parquet")
    tags = pl.read_parquet(DATA_PROCESSED / "core_search_index_tags.parquet")
    topics = pl.read_parquet(DATA_PROCESSED / "forums_topics.parquet")

    # ── Build reference corpus ────────────────────────────────────────
    print("\nBuilding Siege reference corpus…")
    ref_texts = build_siege_reference_corpus(fp, tags, topics)

    if len(ref_texts) == 0:
        print("  ⚠ Empty reference corpus — cannot compute embeddings.")
        return

    # ── Compute reference centroid ────────────────────────────────────
    print("\nComputing reference centroid…")
    ref_embeddings = compute_embeddings(model, ref_texts, "Reference corpus")
    centroid = ref_embeddings.mean(axis=0, keepdims=True)
    centroid = centroid / np.linalg.norm(centroid)  # normalise

    # ── Score forum posts ─────────────────────────────────────────────
    print("\nScoring forum posts…")
    fp_texts = fp["text"].fill_null("").to_list()
    fp_embeddings = compute_embeddings(model, fp_texts, "Forum posts")
    fp_similarities = cosine_similarity(fp_embeddings, centroid).flatten()
    fp = fp.with_columns(
        pl.Series("siege_similarity", fp_similarities, dtype=pl.Float64)
    )

    # Apply embedding-boosted context adjustment
    from importlib import import_module
    _lex = import_module("02_siege_lexicon")
    _boost = _lex.apply_embedding_boost

    if "siege_keyword_context_score" in fp.columns:
        print("  Applying embedding boost to context-dependent terms…")
        fp_adj = _boost(
            fp["siege_keyword_score"].to_numpy(),
            fp["siege_keyword_context_score"].to_numpy(),
            fp_similarities,
        )
        fp = fp.with_columns(
            pl.Series("siege_keyword_score_adjusted", fp_adj, dtype=pl.Float64)
        )

    fp.write_parquet(DATA_PROCESSED / "forum_posts.parquet")
    print(f"  Forum mean similarity: {fp_similarities.mean():.4f}")
    print(f"  Forum max similarity:  {fp_similarities.max():.4f}")

    # ── Score DM posts ────────────────────────────────────────────────
    print("\nScoring DM posts…")
    dm_texts = dm["text"].fill_null("").to_list()
    dm_embeddings = compute_embeddings(model, dm_texts, "DM posts")
    dm_similarities = cosine_similarity(dm_embeddings, centroid).flatten()
    dm = dm.with_columns(
        pl.Series("siege_similarity", dm_similarities, dtype=pl.Float64)
    )

    if "siege_keyword_context_score" in dm.columns:
        print("  Applying embedding boost to context-dependent terms…")
        dm_adj = _boost(
            dm["siege_keyword_score"].to_numpy(),
            dm["siege_keyword_context_score"].to_numpy(),
            dm_similarities,
        )
        dm = dm.with_columns(
            pl.Series("siege_keyword_score_adjusted", dm_adj, dtype=pl.Float64)
        )

    dm.write_parquet(DATA_PROCESSED / "dm_posts.parquet")
    print(f"  DM mean similarity: {dm_similarities.mean():.4f}")
    print(f"  DM max similarity:  {dm_similarities.max():.4f}")

    # ── Build unified siege_scores.parquet ─────────────────────────────
    print("\nBuilding unified siege_scores…")
    score_cols = [
        "siege_keyword_count", "siege_keyword_score",
        "siege_keyword_density", "siege_binary", "siege_similarity",
        "siege_keyword_context_score", "siege_keyword_score_adjusted",
        "word_count",
    ]

    fp_unified = fp.select([
        pl.col("pid").alias("post_id"),
        pl.col("author_id"),
        pl.col("post_date").alias("date"),
        pl.col("text"),
        pl.lit("forum").alias("channel"),
        *[pl.col(c) for c in score_cols],
    ])

    dm_unified = dm.select([
        pl.col("msg_id").alias("post_id"),
        pl.col("msg_author_id").alias("author_id"),
        pl.col("msg_date").alias("date"),
        pl.col("text"),
        pl.lit("dm").alias("channel"),
        *[pl.col(c) for c in score_cols],
    ])

    unified = pl.concat([fp_unified, dm_unified])
    unified.write_parquet(DATA_PROCESSED / "siege_scores.parquet")
    print(f"  Saved siege_scores.parquet ({unified.height:,} rows)")

    # Save centroid for reproducibility
    np.save(DATA_PROCESSED / "siege_centroid.npy", centroid)

    print("\n✓ Embedding scoring complete.")


if __name__ == "__main__":
    main()
