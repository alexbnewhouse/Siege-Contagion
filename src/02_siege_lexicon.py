"""02 – Dictionary-Based Siegist Scoring.

Builds a weighted keyword dictionary and computes per-post siege scores.
"""

from __future__ import annotations

import re

import polars as pl

from utils import DATA_PROCESSED


# ── Siege keyword dictionary ─────────────────────────────────────────
# Weight > 0 → positive indicator; Weight < 0 → counter-indicator.
# Patterns are compiled as case-insensitive word-boundary regexes.

SIEGE_DICTIONARY: list[tuple[str, float]] = [
    # Core Siege terms
    (r"\bsiege\b", 1.0),
    (r"\bjames\s+mason\b", 3.0),
    (r"\buniversal\s+order\b", 3.0),
    (r"\batomwaffen\b", 3.0),
    (r"\bnslf\b", 2.0),
    (r"\bnational\s+socialist\s+liberation\s+front\b", 3.0),
    (r"\btotal\s+attack\b", 2.0),
    (r"\bthe\s+system\b", 0.5),
    (r"\bleaderless\s+resistance\b", 2.5),
    (r"\blone\s+wolf\b", 2.0),

    # Accelerationist rhetoric
    (r"\baccelerat(?:e|ion|ionism|ionist)\b", 2.0),
    (r"\bcollapse\b", 1.0),
    (r"\bboogaloo\b", 2.0),
    (r"\brace\s+war\b", 2.0),
    (r"\bday\s+of\s+the\s+rope\b", 2.5),
    (r"\brahowa\b", 2.0),
    (r"\bdotr\b", 2.0),
    (r"\barmed\s+struggle\b", 2.0),
    (r"\binsurrection\b", 1.5),

    # Mason-specific concepts
    (r"\bcharles\s+manson\b", 2.0),
    (r"\bmanson\b", 1.5),
    (r"\batwa\b", 2.0),
    (r"\bhelter\s+skelter\b", 2.0),

    # Siege Culture markers
    (r"\bread\s+siege\b", 3.0),
    (r"\bsiege\s*culture\b", 3.0),
    (r"\bsiegepill(?:ed)?\b", 3.0),
    (r"\bironpill\b", 2.0),

    # Counter-indicators (non-Mason usage of "siege")
    (r"\bmedieval\s+siege\b", -2.0),
    (r"\bsiege\s+of\s+\w+\b", -1.5),
    (r"\brainbow\s+six\b", -3.0),
    (r"\bsiege\s+engine\b", -2.0),
    (r"\bcastle\s+siege\b", -2.0),
    (r"\bsiege\s+warfare\b", -2.0),
]

# Pre-compile patterns
_COMPILED_DICT: list[tuple[re.Pattern, float]] = [
    (re.compile(pattern, re.IGNORECASE), weight)
    for pattern, weight in SIEGE_DICTIONARY
]


def compute_siege_keyword_score(text: str | None) -> dict:
    """Compute dictionary-based siege scores for a single text.

    Returns dict with keys: keyword_count, keyword_score, keyword_density.
    """
    if not text or not isinstance(text, str):
        return {"keyword_count": 0, "keyword_score": 0.0, "keyword_density": 0.0}

    total_count = 0
    total_score = 0.0

    for pattern, weight in _COMPILED_DICT:
        matches = pattern.findall(text)
        n = len(matches)
        if n > 0:
            total_count += n
            total_score += n * weight

    words = len(text.split())
    density = total_score / words if words > 0 else 0.0

    return {
        "keyword_count": total_count,
        "keyword_score": total_score,
        "keyword_density": density,
    }


def score_dataframe(df: pl.DataFrame, text_col: str = "text") -> pl.DataFrame:
    """Add siege keyword columns to a Polars DataFrame."""
    scores = df[text_col].to_list()
    results = [compute_siege_keyword_score(t) for t in scores]

    keyword_counts = [r["keyword_count"] for r in results]
    keyword_scores = [r["keyword_score"] for r in results]
    keyword_densities = [r["keyword_density"] for r in results]

    df = df.with_columns([
        pl.Series("siege_keyword_count", keyword_counts, dtype=pl.Int32),
        pl.Series("siege_keyword_score", keyword_scores, dtype=pl.Float64),
        pl.Series("siege_keyword_density", keyword_densities, dtype=pl.Float64),
        pl.Series(
            "siege_binary",
            [1 if s > 0 else 0 for s in keyword_scores],
            dtype=pl.Int8,
        ),
    ])
    return df


def main():
    print("=" * 60)
    print("PHASE 1a: Dictionary-Based Siege Scoring")
    print("=" * 60)

    # ── Forum posts ───────────────────────────────────────────────────
    print("\nScoring forum posts…")
    fp = pl.read_parquet(DATA_PROCESSED / "forum_posts.parquet")
    fp = score_dataframe(fp, "text")

    siege_posts = fp.filter(pl.col("siege_binary") == 1)
    print(f"  Forum posts with siege terms: {siege_posts.height:,} / {fp.height:,} "
          f"({100 * siege_posts.height / fp.height:.1f}%)")
    fp.write_parquet(DATA_PROCESSED / "forum_posts.parquet")

    # ── DM posts ──────────────────────────────────────────────────────
    print("\nScoring DM posts…")
    dm = pl.read_parquet(DATA_PROCESSED / "dm_posts.parquet")
    dm = score_dataframe(dm, "text")

    siege_dms = dm.filter(pl.col("siege_binary") == 1)
    print(f"  DM posts with siege terms: {siege_dms.height:,} / {dm.height:,} "
          f"({100 * siege_dms.height / dm.height:.1f}%)")
    dm.write_parquet(DATA_PROCESSED / "dm_posts.parquet")

    print("\n✓ Dictionary scoring complete.")


if __name__ == "__main__":
    main()
