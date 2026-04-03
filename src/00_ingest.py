"""00 – Data Ingestion: Load Iron March CSV exports into Polars DataFrames
and save as Parquet in data/processed/."""

from __future__ import annotations

import polars as pl

from utils import DATA_CSV, DATA_PROCESSED, ZEIGER_MEMBER_ID


# ── CSV → table mapping ──────────────────────────────────────────────

TABLE_MAP: dict[str, str] = {
    # Forum data
    "forums_posts": "im_forums_dfs__forums_posts.csv",
    "forums_topics": "im_forums_dfs__forums_topics.csv",
    "forums_forums": "im_forums_dfs__forums_forums.csv",
    # Core tables
    "core_members": "im_core_dfs__core_members.csv",
    "core_message_posts": "im_core_dfs__core_message_posts.csv",
    "core_message_topics": "im_core_dfs__core_message_topics.csv",
    "core_message_topic_user_map": "im_core_dfs__core_message_topic_user_map.csv",
    "core_reputation_index": "im_core_dfs__core_reputation_index.csv",
    "core_follow": "im_core_dfs__core_follow.csv",
    "core_search_index_tags": "im_core_dfs__core_search_index_tags.csv",
    "core_pfields_content": "im_core_dfs__core_pfields_content.csv",
    "core_member_history": "im_core_dfs__core_member_history.csv",
    # Original DB tables
    "orig_members": "im_orig_dfs__orig_members.csv",
    "orig_posts": "im_orig_dfs__orig_posts.csv",
    "orig_topics": "im_orig_dfs__orig_topics.csv",
    "orig_message_posts": "im_orig_dfs__orig_message_posts.csv",
    "orig_message_topics": "im_orig_dfs__orig_message_topics.csv",
    "orig_message_topic_user_map": "im_orig_dfs__orig_message_topic_user_map.csv",
    "orig_reputation_index": "im_orig_dfs__orig_reputation_index.csv",
    "orig_pfields_content": "im_orig_dfs__orig_pfields_content.csv",
    "orig_dnames_change": "im_orig_dfs__orig_dnames_change.csv",
    # Combo tables (pre-reconciled)
    "combo_members": "combo_members_df.csv",
    "combo_messages": "combo_messages_df.csv",
}


def _safe_int_cols(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Cast float-encoded integer columns to Int64, handling NaN."""
    for col in cols:
        if col in df.columns:
            dtype = df[col].dtype
            if dtype in (pl.Float64, pl.Float32):
                df = df.with_columns(
                    pl.col(col).cast(pl.Int64, strict=False).alias(col)
                )
    return df


def _parse_unix_ts(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Convert timestamp columns to Datetime.

    Handles Unix-timestamp seconds (Int64/Float64), ISO strings, and
    common date-time string formats from R's write.csv().
    """
    for col in cols:
        if col in df.columns:
            dtype = df[col].dtype
            if dtype in (pl.Float64, pl.Int64, pl.Float32, pl.Int32):
                df = df.with_columns(
                    pl.from_epoch(pl.col(col).cast(pl.Int64, strict=False), time_unit="s")
                    .alias(col)
                )
            elif dtype == pl.Utf8:
                # Try several formats; first success wins
                parsed = None
                for fmt in (
                    "%Y-%m-%dT%H:%M:%S%.f",   # ISO with fractional
                    "%Y-%m-%dT%H:%M:%S",       # ISO
                    "%Y-%m-%d %H:%M:%S",       # R default
                    "%Y-%m-%d",                 # date-only
                ):
                    attempt = pl.col(col).str.to_datetime(format=fmt, strict=False)
                    if parsed is None:
                        parsed = attempt
                    else:
                        parsed = pl.coalesce(parsed, attempt)
                df = df.with_columns(parsed.alias(col))
    return df


def load_csv(table_name: str) -> pl.DataFrame:
    """Load a single CSV from DATA_CSV directory."""
    csv_file = TABLE_MAP[table_name]
    path = DATA_CSV / csv_file
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    print(f"  Loading {table_name} from {csv_file}…")
    return pl.read_csv(path, infer_schema_length=10000, ignore_errors=True)


def ingest_all() -> dict[str, pl.DataFrame]:
    """Load and type-coerce all tables, returning a dict of DataFrames."""
    tables: dict[str, pl.DataFrame] = {}

    for name in TABLE_MAP:
        try:
            tables[name] = load_csv(name)
            print(f"    → {name}: {tables[name].shape}")
        except FileNotFoundError as e:
            print(f"    ⚠ Skipping {name}: {e}")

    # ── Type coercions ────────────────────────────────────────────────
    id_cols_map = {
        "forums_posts": ["pid", "author_id", "topic_id"],
        "forums_topics": ["tid", "starter_id", "forum_id", "last_poster_id", "topic_firstpost"],
        "core_members": ["member_id", "member_group_id"],
        "core_message_posts": ["msg_id", "msg_topic_id", "msg_author_id"],
        "core_message_topics": ["mt_id", "mt_starter_id"],
        "core_message_topic_user_map": ["map_id", "map_user_id", "map_topic_id"],
        "core_reputation_index": ["id", "member_id", "type_id", "member_received", "item_id"],
        "core_follow": ["follow_id", "follow_member_id", "follow_rel_id"],
        "core_pfields_content": ["member_id"],
        "orig_members": ["member_id"],
        "combo_members": ["member_id"],
    }

    ts_cols_map = {
        "forums_posts": ["post_date", "edit_time"],
        "forums_topics": ["start_date", "last_post"],
        "core_members": ["joined", "last_visit", "last_activity"],
        "core_message_posts": ["msg_date"],
        "core_message_topics": ["mt_date"],
        "core_reputation_index": ["rep_date"],
        "core_follow": ["follow_added"],
        "orig_members": ["joined"],
        "combo_members": ["joined"],
    }

    for tbl, cols in id_cols_map.items():
        if tbl in tables:
            tables[tbl] = _safe_int_cols(tables[tbl], cols)

    for tbl, cols in ts_cols_map.items():
        if tbl in tables:
            tables[tbl] = _parse_unix_ts(tables[tbl], cols)

    # ── Fallbacks: use orig_* when core_* is missing ──────────────────
    _FALLBACKS = {
        "core_message_posts": "orig_message_posts",
        "core_message_topics": "orig_message_topics",
        "core_message_topic_user_map": "orig_message_topic_user_map",
        "core_pfields_content": "orig_pfields_content",
    }
    for core_name, orig_name in _FALLBACKS.items():
        if core_name not in tables and orig_name in tables:
            tables[core_name] = tables[orig_name]
            print(f"    ↳ Using {orig_name} as fallback for {core_name}")

    # Reputation: derive member_received from post author lookups
    if "core_reputation_index" not in tables and "orig_reputation_index" in tables:
        rep = tables["orig_reputation_index"]
        rep = _safe_int_cols(rep, ["id", "member_id", "type_id"])
        rep = _parse_unix_ts(rep, ["rep_date"])
        # type_id refers to a post id — join with forum posts to get post author
        if "forums_posts" in tables:
            post_authors = tables["forums_posts"].select([
                pl.col("pid").alias("type_id"),
                pl.col("author_id").alias("member_received"),
            ]).unique(subset=["type_id"])
            rep = rep.join(post_authors, on="type_id", how="left")
            rep = _safe_int_cols(rep, ["member_received"])
        else:
            rep = rep.with_columns(pl.lit(None).cast(pl.Int64).alias("member_received"))
        rep = rep.with_columns(pl.col("type_id").alias("item_id"))
        tables["core_reputation_index"] = rep
        print(f"    ↳ Built core_reputation_index from orig_reputation_index "
              f"({rep.height:,} rows, {rep.filter(pl.col('member_received').is_not_null()).height:,} with receiver)")

    # Tags: map orig_core_tags → core_search_index_tags schema
    if "core_search_index_tags" not in tables:
        tags_csv = DATA_CSV / "im_orig_dfs__orig_core_tags.csv"
        if tags_csv.exists():
            tags = pl.read_csv(tags_csv, infer_schema_length=10000, ignore_errors=True)
            tags = tags.rename({"tag_meta_id": "index_id", "tag_text": "index_tag"})
            tags = _safe_int_cols(tags, ["index_id"])
            tables["core_search_index_tags"] = tags
            print(f"    ↳ Built core_search_index_tags from orig_core_tags ({tags.height:,} rows)")

    return tables


def verify_zeiger(tables: dict[str, pl.DataFrame]):
    """Verify Zeiger's identity in the members table."""
    members = tables.get("core_members")
    if members is None:
        print("  ⚠ core_members not loaded — skipping Zeiger verification")
        return

    zeiger = members.filter(pl.col("member_id") == ZEIGER_MEMBER_ID)
    if zeiger.height == 0:
        print(f"  ⚠ member_id {ZEIGER_MEMBER_ID} not found in core_members!")
        return

    name = zeiger["name"][0]
    print(f"  ✓ Zeiger verified: member_id={ZEIGER_MEMBER_ID}, name='{name}'")

    # Check name history
    history = tables.get("core_member_history")
    if history is not None:
        zeiger_names = history.filter(
            pl.col("log_member") == ZEIGER_MEMBER_ID
        )
        if zeiger_names.height > 0:
            print(f"    Name history entries: {zeiger_names.height}")

    # Also check orig_members
    orig = tables.get("orig_members")
    if orig is not None:
        zeiger_orig = orig.filter(pl.col("member_id") == ZEIGER_MEMBER_ID)
        if zeiger_orig.height > 0:
            print(f"    Also in orig_members: name='{zeiger_orig['name'][0]}'")


def data_inventory(tables: dict[str, pl.DataFrame]):
    """Print a data inventory report."""
    print("\n" + "=" * 60)
    print("DATA INVENTORY REPORT")
    print("=" * 60)

    for name, df in sorted(tables.items()):
        print(f"\n  {name}: {df.shape[0]:,} rows × {df.shape[1]} cols")

    # Date ranges for posts
    if "forums_posts" in tables:
        fp = tables["forums_posts"].filter(pl.col("post_date").is_not_null())
        if fp.height > 0:
            print(f"\n  Forum posts date range: {fp['post_date'].min()} → {fp['post_date'].max()}")

    if "core_message_posts" in tables:
        mp = tables["core_message_posts"].filter(pl.col("msg_date").is_not_null())
        if mp.height > 0:
            print(f"  DM posts date range:    {mp['msg_date'].min()} → {mp['msg_date'].max()}")

    # Active users by year
    if "forums_posts" in tables:
        fp = tables["forums_posts"].filter(pl.col("post_date").is_not_null())
        by_year = (
            fp.with_columns(pl.col("post_date").dt.year().alias("year"))
            .group_by("year")
            .agg(pl.col("author_id").n_unique().alias("unique_authors"))
            .sort("year")
        )
        print("\n  Unique active forum posters by year:")
        for row in by_year.iter_rows(named=True):
            print(f"    {row['year']}: {row['unique_authors']}")

    # Zeiger's posting activity
    if "forums_posts" in tables:
        fp = tables["forums_posts"]
        zeiger_posts = fp.filter(pl.col("author_id") == ZEIGER_MEMBER_ID)
        print(f"\n  Zeiger's forum posts: {zeiger_posts.height}")
        if zeiger_posts.height > 0:
            zeiger_by_year = (
                zeiger_posts.filter(pl.col("post_date").is_not_null())
                .with_columns(pl.col("post_date").dt.year().alias("year"))
                .group_by("year")
                .agg(pl.len().alias("count"))
                .sort("year")
            )
            for row in zeiger_by_year.iter_rows(named=True):
                print(f"    {row['year']}: {row['count']}")


def save_tables(tables: dict[str, pl.DataFrame]):
    """Save key tables as Parquet."""
    key_tables = [
        "forums_posts", "forums_topics", "forums_forums",
        "core_members", "core_message_posts", "core_message_topics",
        "core_message_topic_user_map", "core_reputation_index",
        "core_follow", "core_search_index_tags", "core_pfields_content",
        "core_member_history", "orig_members", "combo_members",
    ]
    for name in key_tables:
        if name in tables:
            path = DATA_PROCESSED / f"{name}.parquet"
            tables[name].write_parquet(path)
            print(f"  Saved {name}.parquet ({tables[name].shape[0]:,} rows)")


def main():
    print("=" * 60)
    print("PHASE 0a: Data Ingestion")
    print("=" * 60)

    tables = ingest_all()
    verify_zeiger(tables)
    data_inventory(tables)
    save_tables(tables)

    print("\n✓ Ingestion complete.")


if __name__ == "__main__":
    main()
