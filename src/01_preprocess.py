"""01 – Preprocessing: HTML stripping, text cleaning, member reconciliation,
and treatment date identification."""

from __future__ import annotations

import re

import polars as pl
from bs4 import BeautifulSoup

from utils import DATA_PROCESSED, ZEIGER_MEMBER_ID, load_parquet


# ── HTML / text cleaning ─────────────────────────────────────────────

def strip_html(html: str | None) -> str:
    """Strip HTML tags from a post, extracting plain text.

    Handles nested blockquotes by separating quoted vs. original text.
    Returns only the *original* text (non-quoted portion).
    """
    if html is None or not isinstance(html, str) or html.strip() == "":
        return ""
    try:
        soup = BeautifulSoup(html, "html.parser")
        # Remove blockquotes (quoted text from other users)
        for bq in soup.find_all("blockquote"):
            bq.decompose()
        # Remove data-ipsquote wrappers (IPB-style quotes)
        for div in soup.find_all("div", attrs={"data-ipsquote": True}):
            div.decompose()
        text = soup.get_text(separator=" ")
    except Exception:
        text = re.sub(r"<[^>]+>", " ", html)

    # Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_html_full(html: str | None) -> str:
    """Strip ALL HTML, including quotes, for full-text extraction."""
    if html is None or not isinstance(html, str) or html.strip() == "":
        return ""
    try:
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator=" ")
    except Exception:
        text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Member reconciliation ────────────────────────────────────────────

def reconcile_members() -> pl.DataFrame:
    """Merge core_members and orig_members, preferring core values."""
    core = load_parquet("core_members.parquet").select([
        "member_id", "name", "joined", "member_posts",
        "pp_reputation_points", "member_group_id",
    ])
    try:
        orig = load_parquet("orig_members.parquet").select([
            "member_id", "name", "joined",
        ]).rename({"name": "name_orig", "joined": "joined_orig"})
    except (FileNotFoundError, Exception):
        return core

    merged = core.join(orig, on="member_id", how="full", coalesce=True)

    # Fill core nulls from orig where available
    merged = merged.with_columns([
        pl.coalesce(pl.col("name"), pl.col("name_orig")).alias("name"),
        pl.coalesce(pl.col("joined"), pl.col("joined_orig")).alias("joined"),
    ]).drop(["name_orig", "joined_orig"])

    return merged


# ── Treatment date identification ────────────────────────────────────

def find_treatment_dates(
    forum_posts: pl.DataFrame,
    forums_topics: pl.DataFrame,
    search_tags: pl.DataFrame,
) -> dict:
    """Empirically identify Zeiger's Siege publication date(s)."""
    results = {"T0": None, "editions": []}

    # 1. Find Siege-tagged topics
    siege_tags = search_tags.filter(
        pl.col("index_tag").str.to_lowercase().str.contains(
            r"siege|james.mason|siegeculture"
        )
    )
    siege_index_ids = siege_tags["index_id"].unique().to_list()
    print(f"  Siege-tagged index IDs: {len(siege_index_ids)}")

    # 2. Find Zeiger topics with "siege" in title
    zeiger_siege_topics = forums_topics.filter(
        (pl.col("starter_id") == ZEIGER_MEMBER_ID)
        & pl.col("title").str.to_lowercase().str.contains("siege")
    ).sort("start_date")
    print(f"  Zeiger topics with 'siege' in title: {zeiger_siege_topics.height}")
    if zeiger_siege_topics.height > 0:
        for row in zeiger_siege_topics.iter_rows(named=True):
            print(f"    tid={row['tid']}: '{row['title']}' ({row['start_date']})")

    # 3. Find Zeiger posts mentioning Siege
    zeiger_siege_posts = forum_posts.filter(
        (pl.col("author_id") == ZEIGER_MEMBER_ID)
        & pl.col("text").str.to_lowercase().str.contains(r"siege")
    ).sort("post_date")
    print(f"  Zeiger posts mentioning 'siege': {zeiger_siege_posts.height}")

    # 4. Find new_topic posts by Zeiger mentioning siege
    if "new_topic" in forum_posts.columns:
        zeiger_new_siege = forum_posts.filter(
            (pl.col("author_id") == ZEIGER_MEMBER_ID)
            & (pl.col("new_topic") == True)
            & pl.col("text").str.to_lowercase().str.contains(r"siege")
        ).sort("post_date")
    else:
        zeiger_new_siege = pl.DataFrame()

    # Combine all candidate dates
    candidate_dates = []
    if zeiger_siege_topics.height > 0:
        for row in zeiger_siege_topics.iter_rows(named=True):
            if row["start_date"] is not None:
                candidate_dates.append(row["start_date"])

    if zeiger_new_siege.height > 0:
        for row in zeiger_new_siege.iter_rows(named=True):
            if row["post_date"] is not None:
                candidate_dates.append(row["post_date"])

    if candidate_dates:
        candidate_dates.sort()
        results["T0"] = candidate_dates[0]
        results["editions"] = candidate_dates
        print(f"\n  ✓ Treatment date T0 = {results['T0']}")
        for i, d in enumerate(candidate_dates):
            print(f"    T{i} = {d}")
    else:
        # Fallback: use the earliest Zeiger post mentioning siege
        if zeiger_siege_posts.height > 0:
            results["T0"] = zeiger_siege_posts["post_date"][0]
            print(f"\n  ✓ Treatment date T0 (fallback) = {results['T0']}")
        else:
            print("  ⚠ Could not identify treatment date from data!")

    return results


# ── Main preprocessing pipeline ──────────────────────────────────────

def main():
    print("=" * 60)
    print("PHASE 0b: Preprocessing")
    print("=" * 60)

    # ── Forum posts ───────────────────────────────────────────────────
    print("\nProcessing forum posts…")
    fp = load_parquet("forums_posts.parquet")
    print(f"  Raw forum posts: {fp.height:,}")

    fp = fp.with_columns([
        pl.col("post").map_elements(strip_html, return_dtype=pl.Utf8).alias("text"),
        pl.col("post").map_elements(strip_html_full, return_dtype=pl.Utf8).alias("text_full"),
    ])
    fp = fp.with_columns(
        pl.col("text").str.split(" ").list.len().alias("word_count")
    )
    fp.write_parquet(DATA_PROCESSED / "forum_posts.parquet")
    print(f"  Saved forum_posts.parquet ({fp.height:,} rows)")

    # ── DM posts ──────────────────────────────────────────────────────
    print("\nProcessing DM posts…")
    dm = load_parquet("core_message_posts.parquet")
    print(f"  Raw DM posts: {dm.height:,}")

    dm = dm.with_columns([
        pl.col("msg_post").map_elements(strip_html, return_dtype=pl.Utf8).alias("text"),
        pl.col("msg_post").map_elements(strip_html_full, return_dtype=pl.Utf8).alias("text_full"),
    ])
    dm = dm.with_columns(
        pl.col("text").str.split(" ").list.len().alias("word_count")
    )
    dm.write_parquet(DATA_PROCESSED / "dm_posts.parquet")
    print(f"  Saved dm_posts.parquet ({dm.height:,} rows)")

    # ── Members ───────────────────────────────────────────────────────
    print("\nReconciling members…")
    members = reconcile_members()
    members.write_parquet(DATA_PROCESSED / "members.parquet")
    print(f"  Saved members.parquet ({members.height:,} rows)")

    # ── Treatment date ────────────────────────────────────────────────
    print("\nIdentifying treatment date…")
    topics = load_parquet("forums_topics.parquet")
    tags = load_parquet("core_search_index_tags.parquet")
    treatment = find_treatment_dates(fp, topics, tags)

    # Save treatment info
    import json
    import datetime

    def _serialize(obj):
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        return str(obj)

    with open(DATA_PROCESSED / "treatment_dates.json", "w") as f:
        json.dump(treatment, f, default=_serialize, indent=2)
    print(f"  Saved treatment_dates.json")

    print("\n✓ Preprocessing complete.")


if __name__ == "__main__":
    main()
