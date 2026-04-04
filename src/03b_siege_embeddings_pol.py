"""03b – Embedding-Based Siege Scoring for /pol/.

Computes per-post semantic similarity to the Siege reference centroid
(built from Iron March data in step 03) for /pol/ posts. This provides
a continuous siege-proximity measure that complements dictionary scoring.

Design decisions:
  - Uses the SAME centroid as IM (not a /pol/-specific centroid), so
    scores are directly comparable across platforms.
  - The centroid captures what Siege discourse sounds like on IM; high
    similarity on /pol/ means a post resembles IM Siege language.
  - Applies the same embedding boost to context-dependent terms.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from utils import DATA_PROCESSED

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 512


def main():
    print("=" * 60)
    print("PHASE 1b-pol: Embedding-Based Siege Scoring (/pol/)")
    print("=" * 60)

    # ── Check prerequisites ───────────────────────────────────────────
    centroid_path = DATA_PROCESSED / "siege_centroid.npy"
    pol_path = DATA_PROCESSED / "pol_siege_scores.parquet"

    if not centroid_path.exists():
        print(f"  ✗ {centroid_path} not found. Run 03_siege_embeddings first.")
        return
    if not pol_path.exists():
        print(f"  ✗ {pol_path} not found. Run 02b_siege_lexicon_pol first.")
        return

    # ── Load centroid ─────────────────────────────────────────────────
    centroid = np.load(centroid_path)
    if centroid.ndim == 1:
        centroid = centroid.reshape(1, -1)
    print(f"  Centroid shape: {centroid.shape}")

    # ── Load model ────────────────────────────────────────────────────
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"  Device: {device}")
    model = SentenceTransformer(MODEL_NAME, device=device)

    # ── Load /pol/ data ───────────────────────────────────────────────
    pol = pl.read_parquet(pol_path)
    print(f"  /pol/ posts: {pol.height:,}")

    # ── Compute embeddings ────────────────────────────────────────────
    print("\n  Computing embeddings…")
    texts = pol["text"].fill_null("").to_list()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # ── Compute similarities ──────────────────────────────────────────
    print("  Computing cosine similarities…")
    similarities = cosine_similarity(embeddings, centroid).flatten()

    pol = pol.with_columns(
        pl.Series("siege_similarity", similarities, dtype=pl.Float64)
    )

    # ── Apply embedding boost ─────────────────────────────────────────
    if "siege_keyword_context_score" in pol.columns:
        print("  Applying embedding boost to context-dependent terms…")
        from importlib import import_module
        _lex = import_module("02_siege_lexicon")

        adjusted = _lex.apply_embedding_boost(
            pol["siege_keyword_score"].to_numpy(),
            pol["siege_keyword_context_score"].to_numpy(),
            similarities,
        )
        pol = pol.with_columns(
            pl.Series("siege_keyword_score_adjusted", adjusted, dtype=pl.Float64)
        )

    # ── Save ──────────────────────────────────────────────────────────
    pol.write_parquet(pol_path)

    print(f"\n  Mean similarity: {similarities.mean():.4f}")
    print(f"  Max similarity:  {similarities.max():.4f}")
    print(f"  Posts with sim > 0.3: "
          f"{(similarities > 0.3).sum():,} "
          f"({(similarities > 0.3).mean() * 100:.1f}%)")

    print(f"\n✓ /pol/ embedding scoring complete. Updated {pol_path.name}")


if __name__ == "__main__":
    main()
