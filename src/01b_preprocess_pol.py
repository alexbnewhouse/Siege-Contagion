"""01b – Preprocessing: /pol/ text cleaning and schema normalisation.

Cleans 4chan HTML markup from /pol/ posts and normalises the schema
for downstream analysis stages.

4chan post markup conventions
-----------------------------
- ``<br>`` → newline
- ``<span class="quote">&gt;TEXT</span>`` → greentext
- ``<a href="..." class="quotelink">&gt;&gt;NUM</a>`` → cross-post refs
- ``<s>TEXT</s>`` → spoiler tags
- ``<wbr>`` → word break hints (inserted in long URLs)
- HTML entities: ``&gt;``, ``&lt;``, ``&amp;``, ``&#039;``, ``&quot;``

Output schema (matches Iron March ``forum_posts.parquet`` structure)
-------------------------------------------------------------------
- ``post_id``     (Int64)  – post number (``num``)
- ``author_id``   (Utf8)   – poster_hash or "anon" (no persistent IDs on /pol/)
- ``date``        (Datetime)
- ``text``        (Utf8)   – cleaned post text (greentext removed)
- ``text_full``   (Utf8)   – full text including greentext
- ``word_count``  (Int32)
- ``thread_id``   (Int64)  – thread_num
- ``op``          (Int8)
- ``trip``        (Utf8)   – tripcode if present
- ``title``       (Utf8)   – OP subject line
- ``platform``    (Utf8)   – literal "pol"
- ``poster_country`` (Utf8)
"""

from __future__ import annotations

import html
import multiprocessing
import os
import re

import polars as pl

from utils import DATA_PROCESSED

_N_WORKERS = min(multiprocessing.cpu_count(), int(os.environ.get("SIEGE_WORKERS", "16")))


# ── 4chan markup cleaning ─────────────────────────────────────────────

# Pre-compiled patterns for performance
_QUOTELINK_RE = re.compile(
    r'<a[^>]*class="quotelink"[^>]*>&gt;&gt;\d+</a>', re.IGNORECASE
)
_GREENTEXT_RE = re.compile(
    r'<span class="quote">[^<]*</span>', re.IGNORECASE
)
_DEADLINK_RE = re.compile(
    r'<span class="deadlink">[^<]*</span>', re.IGNORECASE
)
_SPOILER_RE = re.compile(r"<s>(.*?)</s>", re.IGNORECASE | re.DOTALL)
_WBR_RE = re.compile(r"<wbr\s*/?>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"[ \t]+")
_MULTILINE_RE = re.compile(r"\n{3,}")


def strip_4chan_html(raw: str | None, *, keep_greentext: bool = False) -> str:
    """Strip 4chan HTML markup from a post's comment field.

    Parameters
    ----------
    raw : str | None
        Raw HTML comment from the Asagi dump.
    keep_greentext : bool
        If False (default), remove greentext quotes (lines starting
        with '>'). If True, keep them as plain text.

    Returns
    -------
    str
        Cleaned plain-text post content.
    """
    if raw is None or not isinstance(raw, str) or raw.strip() == "":
        return ""

    text = raw

    # 1. Remove cross-post references (>>12345)
    text = _QUOTELINK_RE.sub("", text)

    # 2. Handle greentext
    if keep_greentext:
        # Convert greentext spans to plain > prefixed text
        text = re.sub(
            r'<span class="quote">(.*?)</span>',
            r"\1",
            text,
            flags=re.IGNORECASE,
        )
    else:
        text = _GREENTEXT_RE.sub("", text)

    # 3. Remove dead links
    text = _DEADLINK_RE.sub("", text)

    # 4. Reveal spoilers (keep text, remove tags)
    text = _SPOILER_RE.sub(r"\1", text)

    # 5. Remove <wbr> tags
    text = _WBR_RE.sub("", text)

    # 6. Convert <br> to newlines
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)

    # 7. Remove all remaining HTML tags
    text = _TAG_RE.sub("", text)

    # 8. Decode HTML entities
    text = html.unescape(text)

    # 9. Normalise whitespace (preserve meaningful newlines)
    text = _WHITESPACE_RE.sub(" ", text)
    text = _MULTILINE_RE.sub("\n\n", text)
    text = text.strip()

    return text


def strip_4chan_full(raw: str | None) -> str:
    """Strip all HTML but keep greentext as plain text."""
    return strip_4chan_html(raw, keep_greentext=True)


def strip_4chan_nogreen(raw: str | None) -> str:
    """Strip all HTML and remove greentext (quoted text)."""
    return strip_4chan_html(raw, keep_greentext=False)


# ── Schema normalisation ─────────────────────────────────────────────

def normalise_pol_schema(df: pl.DataFrame) -> pl.DataFrame:
    """Rename /pol/ columns to match the unified cross-platform schema.

    - ``num`` → ``post_id``
    - ``thread_num`` → ``thread_id``
    - ``poster_hash`` → ``author_id`` (Utf8; NULL → "anon")
    """
    df = df.rename({
        "num": "post_id",
        "thread_num": "thread_id",
    })

    # Author ID: use poster_hash if available, else "anon"
    if "poster_hash" in df.columns:
        df = df.with_columns(
            pl.coalesce(pl.col("poster_hash"), pl.lit("anon"))
            .alias("author_id")
        )
    else:
        df = df.with_columns(pl.lit("anon").alias("author_id"))

    return df


def main():
    """Run /pol/ preprocessing pipeline."""
    print("=" * 60)
    print("PHASE 0b-2: /pol/ Preprocessing")
    print("=" * 60)

    # ── Load raw /pol/ posts ──────────────────────────────────────────
    raw_path = DATA_PROCESSED / "pol_posts_raw.parquet"
    if not raw_path.exists():
        print(f"  ✗ {raw_path} not found. Run 00b_ingest_pol first.")
        return

    df = pl.read_parquet(raw_path)
    print(f"  Loaded {df.height:,} raw /pol/ posts")

    # ── HTML stripping ────────────────────────────────────────────────
    raw_html = df["comment"].to_list()
    print(f"  Stripping 4chan HTML with {_N_WORKERS} workers…")

    with multiprocessing.Pool(_N_WORKERS) as pool:
        texts = pool.map(strip_4chan_nogreen, raw_html, chunksize=2048)
        texts_full = pool.map(strip_4chan_full, raw_html, chunksize=2048)

    df = df.with_columns([
        pl.Series("text", texts, dtype=pl.Utf8),
        pl.Series("text_full", texts_full, dtype=pl.Utf8),
    ])

    # ── Word count ────────────────────────────────────────────────────
    df = df.with_columns(
        pl.col("text").str.split(" ").list.len().alias("word_count")
    )

    # ── Filter empty posts ────────────────────────────────────────────
    before = df.height
    df = df.filter(pl.col("word_count") > 0)
    print(f"  Dropped {before - df.height:,} empty posts → {df.height:,} remaining")

    # ── Schema normalisation ──────────────────────────────────────────
    df = normalise_pol_schema(df)

    # Ensure platform column
    if "platform" not in df.columns:
        df = df.with_columns(pl.lit("pol").alias("platform"))

    # ── Save ──────────────────────────────────────────────────────────
    outpath = DATA_PROCESSED / "pol_posts.parquet"
    df.write_parquet(outpath)
    print(f"\n  Saved: {outpath} ({df.height:,} rows)")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n  Date range: {df['date'].min()} → {df['date'].max()}")
    print(f"  Mean word count: {df['word_count'].mean():.1f}")
    print(f"  Unique threads: {df['thread_id'].n_unique():,}")
    tripped = df.filter(
        pl.col("trip").is_not_null() & (pl.col("trip") != "")
    ).height
    print(f"  Posts with tripcodes: {tripped:,} ({tripped/df.height*100:.1f}%)")

    print("\n✓ /pol/ preprocessing complete.")


if __name__ == "__main__":
    main()
