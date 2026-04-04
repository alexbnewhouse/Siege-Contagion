"""02 – Dictionary-Based Siegist Scoring.

Builds a weighted keyword dictionary and computes per-post siege scores.
Includes embedding-boosted context adjustment for ambiguous terms.
"""

from __future__ import annotations

import multiprocessing
import os
import re

import numpy as np
import polars as pl

from utils import DATA_PROCESSED

_N_WORKERS = min(multiprocessing.cpu_count(), int(os.environ.get("SIEGE_WORKERS", "16")))


# ── Siege keyword dictionary ─────────────────────────────────────────
# Weight > 0 → positive indicator; Weight < 0 → counter-indicator.
# Patterns are compiled as case-insensitive word-boundary regexes.
#
# Tier 1 – Unambiguous Siege/Mason markers (weight 2.5–3.0)
# Tier 2 – Siege-adjacent ideology & figures (weight 1.5–2.5)
# Tier 3 – Context-dependent terms needing embedding boost (weight 0.3–1.0)
#
# Research basis: Siege (Mason 1992), ICCT reports (Johnson & Feldman 2021;
# Ware 2019), SPLC Hatewatch, CTC Sentinel (Upchurch 2021), Wikipedia
# articles on Siege, James Mason, Joseph Tommasi, Atomwaffen Division,
# NSLF, militant accelerationism, and corpus-level hit-rate analysis of
# the Iron March dataset.

SIEGE_DICTIONARY: list[tuple[str, float]] = [
    # ── Tier 1: Unambiguous Siege markers ─────────────────────────────
    (r"\bsiege\b", 1.0),
    (r"\bjames\s+mason\b", 3.0),
    (r"\buniversal\s+order\b", 3.0),
    (r"\batomwaffen\b", 3.0),
    (r"\bnslf\b", 2.0),
    (r"\bnational\s+socialist\s+liberation\s+front\b", 3.0),
    (r"\bread\s+siege\b", 3.0),
    (r"\bsiege\s*culture\b", 3.0),
    (r"\bsiegepill(?:ed)?\b", 3.0),
    # Tommasi — co-creator of Siege, NSLF founder (62 corpus hits)
    (r"\bjoseph\s+tommasi\b", 3.0),
    (r"\btommasi\b", 2.5),
    # Tommasi's signature slogan (7 corpus hits, extremely specific)
    (r"\bpolitical\s+terror\b", 3.0),
    # Siege Culture aesthetic (skull mask network — CTC Sentinel 2021)
    (r"\bskull\s*mask\b", 2.5),
    # Direct Siege book language ("Jew-Capitalist System")
    (r"\bjew[- ]?capitalist\b", 2.5),
    # Siege-specific derogatory for system collaborators
    (r"\bsystem\s+pig\b", 2.5),
    # Post-AWD Siege Culture organisations
    (r"\bamerican\s+futurist\b", 2.5),
    (r"\bantipodean\s+resistance\b", 2.5),

    # ── Tier 2: Siege-adjacent ideology & figures ─────────────────────
    (r"\btotal\s+attack\b", 2.0),
    (r"\bleaderless\s+resistance\b", 2.5),
    (r"\blone\s+wolf\b", 2.0),
    (r"\bironpill\b", 2.0),
    # Accelerationist rhetoric
    (r"\baccelerat(?:e|ion|ionism|ionist)\b", 2.0),
    (r"\bboogaloo\b", 2.0),
    (r"\brace\s+war\b", 2.0),
    (r"\bday\s+of\s+the\s+rope\b", 2.5),
    (r"\brahowa\b", 2.0),
    (r"\bdotr\b", 2.0),
    (r"\barmed\s+struggle\b", 2.0),
    # Mason-specific concepts
    (r"\bcharles\s+manson\b", 2.0),
    (r"\bmanson\b", 1.5),
    (r"\batwa\b", 2.0),
    (r"\bhelter\s+skelter\b", 2.0),
    # William Luther Pierce — wrote Turner Diaries, key Siege influence
    (r"\bwilliam\s+(?:luther\s+)?pierce\b", 2.0),
    (r"\bturner\s+diar(?:y|ies)\b", 2.0),
    # Order of Nine Angles — closely tied to AWD/Siege Culture
    (r"\bo9a\b", 2.0),
    (r"\border\s+of\s+nine\s+angles?\b", 2.5),
    # Savitri Devi — lionised in Siege book (207 corpus hits)
    (r"\bsavitri\s+devi\b", 2.0),
    # Accelerationist meme term ("Right Wing Death Squads")
    (r"\brwds\b", 2.0),
    # Siege/accelerationist targets celebrated as "saints"
    (r"\bdylann\s+roof\b", 1.5),
    (r"\bbreivik\b", 1.5),
    # Core Siege vocabulary
    (r"\bwhite\s+revolution\b", 2.0),
    (r"\bwhite\s+jihad\b", 2.0),
    (r"\brace\s+trait(?:or|ors?)\b", 1.5),
    (r"\bethnostate\b", 1.5),
    (r"\banti[- ]?system\b", 1.5),
    (r"\bpropaganda\s+of\s+the\s+deed\b", 2.5),
    (r"\bzionist\s+occupation\s+government\b", 2.5),

    # ── Tier 3: Context-dependent (see CONTEXT_DEPENDENT_PATTERNS) ────
    (r"\bthe\s+system\b", 0.3),
    (r"\bcollapse\b", 0.3),
    (r"\binsurrection\b", 1.0),
    (r"\bzog\b", 0.5),
    (r"\btotal\s+war\b", 0.3),
    (r"\bblack\s+sun\b", 0.5),
    (r"\bguerrilla\b", 0.3),

    # ── Counter-indicators (non-Mason / non-ideological usage) ────────
    (r"\bmedieval\s+siege\b", -2.0),
    (r"\bsiege\s+of\s+\w+\b", -1.5),
    (r"\brainbow\s+six\b", -3.0),
    (r"\bsiege\s+engine\b", -2.0),
    (r"\bcastle\s+siege\b", -2.0),
    (r"\bsiege\s+warfare\b", -2.0),
    # Total War video game franchise
    (r"\btotal\s+war\s*(?:warhammer|rome|shogun|attila|medieval|empire|napoleon|troy)\b", -2.0),
]

# Patterns whose score contribution is context-dependent.  When embedding
# similarity is available (from step 03), their weight is scaled:
#   sim >= 0.4 → 2.0×      (post is firmly in Siege semantic space)
#   sim >= 0.3 → 1.5×
#   sim <  0.15 → 0.25×    (post is far from Siege context)
#   otherwise  → 1.0×      (no change)
CONTEXT_DEPENDENT_PATTERNS: set[str] = {
    r"\bthe\s+system\b",
    r"\bcollapse\b",
    r"\binsurrection\b",
    r"\bzog\b",
    r"\btotal\s+war\b",
    r"\bblack\s+sun\b",
    r"\bguerrilla\b",
}

# Pre-compile patterns
_COMPILED_DICT: list[tuple[re.Pattern, float, bool]] = [
    (
        re.compile(pattern, re.IGNORECASE),
        weight,
        pattern in CONTEXT_DEPENDENT_PATTERNS,
    )
    for pattern, weight in SIEGE_DICTIONARY
]


def compute_siege_keyword_score(text: str | None) -> dict:
    """Compute dictionary-based siege scores for a single text.

    Returns dict with keys: keyword_count, keyword_score, keyword_density,
    keyword_context_score.
    """
    if not text or not isinstance(text, str):
        return {
            "keyword_count": 0,
            "keyword_score": 0.0,
            "keyword_density": 0.0,
            "keyword_context_score": 0.0,
        }

    total_count = 0
    total_score = 0.0
    context_score = 0.0

    for pattern, weight, is_context_dep in _COMPILED_DICT:
        matches = pattern.findall(text)
        n = len(matches)
        if n > 0:
            total_count += n
            contribution = n * weight
            total_score += contribution
            if is_context_dep:
                context_score += contribution

    words = len(text.split())
    density = total_score / words if words > 0 else 0.0

    return {
        "keyword_count": total_count,
        "keyword_score": total_score,
        "keyword_density": density,
        "keyword_context_score": context_score,
    }


def embedding_boost_factor(similarity: float) -> float:
    """Return the multiplicative boost for context-dependent term scores."""
    if similarity >= 0.4:
        return 2.0
    if similarity >= 0.3:
        return 1.5
    if similarity < 0.15:
        return 0.25
    return 1.0


def apply_embedding_boost(
    keyword_scores: np.ndarray | list[float],
    context_scores: np.ndarray | list[float],
    similarities: np.ndarray | list[float],
) -> np.ndarray:
    """Produce adjusted keyword scores using embedding similarity.

    adjusted = (keyword_score − context_score) + context_score × boost(sim)
    """
    kw = np.asarray(keyword_scores, dtype=np.float64)
    ctx = np.asarray(context_scores, dtype=np.float64)
    sims = np.asarray(similarities, dtype=np.float64)

    boosts = np.vectorize(embedding_boost_factor)(sims)
    return kw - ctx + ctx * boosts


def score_dataframe(df: pl.DataFrame, text_col: str = "text") -> pl.DataFrame:
    """Add siege keyword columns to a Polars DataFrame."""
    scores = df[text_col].to_list()

    print(f"    Scoring {len(scores):,} texts with {_N_WORKERS} workers…")
    with multiprocessing.Pool(_N_WORKERS) as pool:
        results = pool.map(compute_siege_keyword_score, scores, chunksize=1024)

    keyword_counts = [r["keyword_count"] for r in results]
    keyword_scores = [r["keyword_score"] for r in results]
    keyword_densities = [r["keyword_density"] for r in results]
    keyword_context_scores = [r["keyword_context_score"] for r in results]

    df = df.with_columns([
        pl.Series("siege_keyword_count", keyword_counts, dtype=pl.Int32),
        pl.Series("siege_keyword_score", keyword_scores, dtype=pl.Float64),
        pl.Series("siege_keyword_density", keyword_densities, dtype=pl.Float64),
        pl.Series(
            "siege_binary",
            [1 if s > 0 else 0 for s in keyword_scores],
            dtype=pl.Int8,
        ),
        pl.Series("siege_keyword_context_score", keyword_context_scores, dtype=pl.Float64),
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
