"""00b – Data Ingestion: 4plebs /pol/ Archive.

Ingest the 4plebs /pol/ CSV dump (Asagi MySQL schema) into Polars
DataFrames. The archive is a single headerless CSV inside a tar.gz.

The Asagi schema is the standard format for 4chan archival databases
(Hine et al. 2017; Papasavva et al. 2020). Column order is fixed:

    num, subnum, thread_num, op, timestamp, timestamp_expired,
    preview_orig, preview_w, preview_h, media_filename,
    media_w, media_h, media_size, media_hash, media_orig,
    spoiler, deleted, capcode, email, name,
    trip, title, comment, sticky, locked,
    poster_hash, poster_country, exif

Strategy
--------
Full /pol/ has ~200M+ posts; loading everything is impractical.
We use a **two-pass streaming approach**:

  Pass 1 – Scan for Siege-relevant posts using a lightweight regex
           pre-filter on the raw ``comment`` field. Also collect weekly
           aggregate post counts for all posts (needed as denominator
           in prevalence calculations).

  Pass 2 – (Done in 01b_preprocess_pol.py) Text cleaning on the
           filtered subset only.

Output
------
- ``data/processed/pol_posts_raw.parquet``   – Siege-relevant posts (raw HTML)
- ``data/processed/pol_weekly_totals.parquet`` – Weekly post counts for all /pol/
"""

from __future__ import annotations

import csv
import datetime
import gzip
import io
import re
import tarfile
from collections import defaultdict
from pathlib import Path

import polars as pl

from utils import DATA_PROCESSED, PROJECT_ROOT

# ── Path to the /pol/ archive ─────────────────────────────────────────
POL_ARCHIVE = PROJECT_ROOT / "data" / "pol" / "pol.csv.tar.gz"

# ── Asagi column schema (headerless CSV) ──────────────────────────────
ASAGI_COLUMNS = [
    "num", "subnum", "thread_num", "op", "timestamp", "timestamp_expired",
    "preview_orig", "preview_w", "preview_h", "media_filename",
    "media_w", "media_h", "media_size", "media_hash", "media_orig",
    "spoiler", "deleted", "capcode", "email", "name",
    "trip", "title", "comment", "sticky", "locked",
    "poster_hash", "poster_country", "exif",
]

# Columns we actually keep (drop media metadata to save memory)
KEEP_COLUMNS = [
    "num", "subnum", "thread_num", "op", "timestamp",
    "name", "trip", "title", "comment",
    "sticky", "locked", "poster_hash", "poster_country",
]

# ── Lightweight Siege pre-filter ──────────────────────────────────────
# This captures Tier 1 + Tier 2 terms plus broader context keywords.
# False positives are acceptable here — lexicon scoring happens later.
_PREFILTER_TERMS = [
    r"siege", r"james\s*mason", r"universal\s*order", r"atomwaffen",
    r"nslf", r"read\s*siege", r"siege\s*culture", r"siegepill",
    r"tommasi", r"skull\s*mask", r"accelerat", r"boogaloo",
    r"race\s*war", r"day\s*of\s*the\s*rope", r"rahowa", r"dotr",
    r"leaderless\s*resistance", r"lone\s*wolf", r"helter\s*skelter",
    r"charles\s*manson", r"turner\s*diar", r"o9a",
    r"order\s*of\s*nine\s*angles?", r"savitri\s*devi",
    r"rwds", r"white\s*revolution", r"white\s*jihad",
    r"ethnostate", r"propaganda\s*of\s*the\s*deed",
    r"zionist\s*occupation", r"iron\s*march", r"ironmarch",
    r"political\s*terror", r"system\s*pig", r"american\s*futurist",
    r"antipodean\s*resistance", r"dylann\s*roof", r"breivik",
    r"armed\s*struggle", r"total\s*attack",
]
PREFILTER_RE = re.compile("|".join(_PREFILTER_TERMS), re.IGNORECASE)

# ── NULL handling ─────────────────────────────────────────────────────
_MYSQL_NULL = "\\N"


def _clean_value(val: str) -> str | None:
    """Convert MySQL NULL representation to Python None."""
    if val == _MYSQL_NULL:
        return None
    return val


def _safe_int(val: str | None) -> int | None:
    """Parse an integer, returning None for nulls/errors."""
    if val is None or val == _MYSQL_NULL:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _unix_to_iso(val: str | None) -> str | None:
    """Convert Unix timestamp string to ISO format."""
    if val is None or val == _MYSQL_NULL or val == "0":
        return None
    try:
        ts = int(val)
        return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).isoformat()
    except (ValueError, TypeError, OSError):
        return None


def stream_pol_archive(
    archive_path: Path = POL_ARCHIVE,
    *,
    siege_filter: bool = True,
    progress_interval: int = 5_000_000,
) -> tuple[list[dict], dict[str, int]]:
    """Stream-read the /pol/ CSV from tar.gz, applying optional Siege filter.

    Parameters
    ----------
    archive_path : Path
        Path to pol.csv.tar.gz
    siege_filter : bool
        If True, only retain posts matching the Siege pre-filter regex.
        Always collects weekly totals regardless.
    progress_interval : int
        Print progress every N rows.

    Returns
    -------
    filtered_rows : list[dict]
        Posts matching the Siege pre-filter (or all posts if filter=False).
    weekly_totals : dict[str, int]
        Mapping of ISO week string (YYYY-WNN) → total post count.
    """
    print(f"  Opening archive: {archive_path}")
    print(f"  Siege pre-filter: {'ON' if siege_filter else 'OFF'}")

    filtered_rows: list[dict] = []
    weekly_totals: defaultdict[str, int] = defaultdict(int)
    total_read = 0
    total_matched = 0

    with tarfile.open(archive_path, "r:gz") as tar:
        # Find the CSV file inside the archive
        members = tar.getmembers()
        csv_member = None
        for m in members:
            if m.name.endswith(".csv"):
                csv_member = m
                break
        if csv_member is None:
            raise FileNotFoundError("No .csv file found inside the tar.gz archive")

        print(f"  Extracting: {csv_member.name}")

        fileobj = tar.extractfile(csv_member)
        if fileobj is None:
            raise IOError(f"Cannot extract {csv_member.name}")

        # Wrap in TextIOWrapper for csv.reader
        text_stream = io.TextIOWrapper(fileobj, encoding="utf-8", errors="replace")

        reader = csv.reader(text_stream, quotechar='"', escapechar="\\")

        for row in reader:
            total_read += 1

            # Skip malformed rows
            if len(row) < len(ASAGI_COLUMNS):
                continue

            # Trim to expected number of columns (some rows have trailing fields)
            row = row[: len(ASAGI_COLUMNS)]

            # Extract timestamp for weekly totals
            ts_val = _safe_int(row[4])
            if ts_val and ts_val > 0:
                try:
                    dt = datetime.datetime.fromtimestamp(ts_val, tz=datetime.timezone.utc)
                    week_key = dt.strftime("%G-W%V")
                    weekly_totals[week_key] += 1
                except (OSError, ValueError):
                    pass

            # Apply Siege pre-filter on the comment field
            comment = row[22] if len(row) > 22 else ""
            title = row[21] if len(row) > 21 else ""
            search_text = f"{title or ''} {comment or ''}"

            if siege_filter and not PREFILTER_RE.search(search_text):
                if total_read % progress_interval == 0:
                    print(f"    … {total_read:,} rows scanned, "
                          f"{total_matched:,} matched "
                          f"({len(weekly_totals):,} weeks)")
                continue

            total_matched += 1

            # Build record from kept columns
            record = {
                "num": _safe_int(row[0]),
                "subnum": _safe_int(row[1]),
                "thread_num": _safe_int(row[2]),
                "op": _safe_int(row[3]),
                "timestamp": _safe_int(row[4]),
                "name": _clean_value(row[19]),
                "trip": _clean_value(row[20]),
                "title": _clean_value(row[21]),
                "comment": _clean_value(row[22]),
                "sticky": _safe_int(row[23]),
                "locked": _safe_int(row[24]),
                "poster_hash": _clean_value(row[25]),
                "poster_country": _clean_value(row[26]),
            }
            filtered_rows.append(record)

            if total_read % progress_interval == 0:
                print(f"    … {total_read:,} rows scanned, "
                      f"{total_matched:,} matched "
                      f"({len(weekly_totals):,} weeks)")

    print(f"\n  Scan complete: {total_read:,} total rows")
    print(f"  Siege-relevant posts: {total_matched:,} "
          f"({total_matched / max(total_read, 1) * 100:.2f}%)")
    print(f"  Weekly bins: {len(weekly_totals):,}")

    return filtered_rows, dict(weekly_totals)


def rows_to_dataframe(rows: list[dict]) -> pl.DataFrame:
    """Convert list of dicts to a typed Polars DataFrame."""
    if not rows:
        return pl.DataFrame(schema={
            "num": pl.Int64, "subnum": pl.Int64, "thread_num": pl.Int64,
            "op": pl.Int8, "timestamp": pl.Int64,
            "name": pl.Utf8, "trip": pl.Utf8, "title": pl.Utf8,
            "comment": pl.Utf8, "sticky": pl.Int8, "locked": pl.Int8,
            "poster_hash": pl.Utf8, "poster_country": pl.Utf8,
        })

    df = pl.DataFrame(rows)

    # Type coercion
    int64_cols = ["num", "subnum", "thread_num", "timestamp"]
    int8_cols = ["op", "sticky", "locked"]

    for col in int64_cols:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Int64, strict=False))
    for col in int8_cols:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Int8, strict=False))

    # Parse timestamps to datetime
    df = df.with_columns(
        pl.from_epoch(pl.col("timestamp").cast(pl.Int64, strict=False), time_unit="s")
        .alias("date")
    )

    # Add platform marker
    df = df.with_columns(pl.lit("pol").alias("platform"))

    return df


def weekly_totals_to_dataframe(weekly_totals: dict[str, int]) -> pl.DataFrame:
    """Convert weekly totals dict to a sorted Polars DataFrame."""
    weeks = sorted(weekly_totals.keys())
    counts = [weekly_totals[w] for w in weeks]

    df = pl.DataFrame({
        "week_iso": weeks,
        "total_posts": counts,
    })

    # Parse ISO week to a Monday date
    df = df.with_columns(
        pl.col("week_iso").str.strptime(pl.Date, "%G-W%V", strict=False)
        .dt.offset_by("0d")  # ensure Monday
        .alias("week_start")
    )

    return df.sort("week_iso")


def main():
    """Run the /pol/ ingest pipeline."""
    print("=" * 60)
    print("PHASE 0b: /pol/ Data Ingestion (4plebs Archive)")
    print("=" * 60)

    if not POL_ARCHIVE.exists():
        print(f"  ✗ Archive not found: {POL_ARCHIVE}")
        print("  Place pol.csv.tar.gz in data/pol/ and re-run.")
        return

    print(f"\n  Archive: {POL_ARCHIVE}")
    print(f"  Size: {POL_ARCHIVE.stat().st_size / (1024**3):.1f} GB\n")

    # ── Stream and filter ─────────────────────────────────────────────
    rows, weekly_totals = stream_pol_archive(POL_ARCHIVE, siege_filter=True)

    # ── Build DataFrames ──────────────────────────────────────────────
    print("\nBuilding DataFrames…")
    df = rows_to_dataframe(rows)
    print(f"  Siege-filtered posts: {df.height:,} rows × {df.width} cols")

    if df.height > 0:
        print(f"  Date range: {df['date'].min()} → {df['date'].max()}")
        print(f"  Unique threads: {df['thread_num'].n_unique():,}")
        print(f"  With tripcodes: {df.filter(pl.col('trip').is_not_null()).height:,}")

    # ── Save ──────────────────────────────────────────────────────────
    outpath = DATA_PROCESSED / "pol_posts_raw.parquet"
    df.write_parquet(outpath)
    print(f"\n  Saved: {outpath}")

    wt = weekly_totals_to_dataframe(weekly_totals)
    wt_path = DATA_PROCESSED / "pol_weekly_totals.parquet"
    wt.write_parquet(wt_path)
    print(f"  Saved: {wt_path} ({wt.height:,} weeks)")

    # ── Summary stats ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  /pol/ INGEST SUMMARY")
    print("=" * 60)
    total_all = sum(weekly_totals.values())
    print(f"  Total /pol/ posts scanned:     {total_all:,}")
    print(f"  Siege-relevant posts retained: {df.height:,}")
    if total_all > 0:
        print(f"  Retention rate:                {df.height / total_all * 100:.3f}%")
    print(f"  Archive coverage:              {wt['week_iso'].min()} → {wt['week_iso'].max()}")

    print("\n✓ /pol/ ingest complete.")


if __name__ == "__main__":
    main()
