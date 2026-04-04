"""19 – Cross-Platform Content Bridges (H14).

Detects shared content (URLs, phrases, images) appearing on both
Iron March and /pol/, with temporal ordering to determine which
platform originated the material.

Key hypotheses
--------------
- H14a: Do Siege-relevant URLs/terms propagate from IM → /pol/?
- H14b: Is there a measurable lag between IM appearance and /pol/ echo?
- H14c: Are there bridging users (matching tripcodes, shared language
         fingerprints) who may facilitate cross-platform diffusion?

Methodology
-----------
We identify shared content signatures across the two platforms:
  1. URL overlap analysis (shared links in Siege-flagged posts)
  2. N-gram fingerprinting (rare distinctive phrases appearing in both)
  3. Temporal priority analysis (which platform had it first?)

This follows the cross-platform diffusion methodology of Zannettou et al.
(2017) "The Web Centipede" and Hine et al. (2017) "Kek, Cucks, and God
Emperor Trump", adapted for far-right extremist content tracking.
"""

from __future__ import annotations

import datetime
import json
import re
from collections import Counter, defaultdict

import numpy as np
import polars as pl

from utils import (
    DATA_PROCESSED, RESULTS_DIR,
    setup_plot_style, save_figure, FIGSIZE_WIDE, CB_PALETTE, ZEIGER_MEMBER_ID,
)
import matplotlib.pyplot as plt


# ── URL extraction ────────────────────────────────────────────────────
_URL_PATTERN = re.compile(
    r'https?://[^\s<>"\')\]]+',
    re.IGNORECASE,
)

# Skip obvious noise domains
_NOISE_DOMAINS = frozenset({
    "imgur.com", "i.imgur.com", "youtube.com", "youtu.be",
    "twitter.com", "facebook.com", "google.com", "wikipedia.org",
    "reddit.com", "boards.4chan.org", "4chan.org",
    "archive.org", "web.archive.org",
})


def _extract_domain(url: str) -> str | None:
    """Extract domain from URL."""
    match = re.match(r'https?://(?:www\.)?([^/]+)', url)
    return match.group(1).lower() if match else None


def extract_urls(text: str | None) -> list[str]:
    """Extract non-noise URLs from text."""
    if not text:
        return []
    urls = _URL_PATTERN.findall(text)
    result = []
    for url in urls:
        domain = _extract_domain(url)
        if domain and domain not in _NOISE_DOMAINS:
            result.append(url.rstrip(".,;:!?)>"))
    return result


# ── N-gram fingerprinting ─────────────────────────────────────────────
def extract_ngrams(text: str | None, n: int = 4) -> list[str]:
    """Extract word-level n-grams from text."""
    if not text:
        return []
    words = text.lower().split()
    # Only consider posts with enough words
    if len(words) < n:
        return []
    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]


def find_shared_urls(
    im_posts: pl.DataFrame,
    pol_posts: pl.DataFrame,
) -> dict:
    """Find URLs shared between IM and /pol/ Siege-flagged posts.

    For each shared URL, record which platform posted it first.
    """
    print("  Extracting URLs from IM posts…")
    im_urls: dict[str, datetime.datetime] = {}
    for row in im_posts.iter_rows(named=True):
        text = row.get("text", "")
        date = row.get("date")
        if not date:
            continue
        for url in extract_urls(text):
            if url not in im_urls or date < im_urls[url]:
                im_urls[url] = date

    print(f"    IM unique URLs: {len(im_urls):,}")

    print("  Extracting URLs from /pol/ posts…")
    pol_urls: dict[str, datetime.datetime] = {}
    for row in pol_posts.iter_rows(named=True):
        text = row.get("text", "")
        date = row.get("date")
        if not date:
            continue
        for url in extract_urls(text):
            if url not in pol_urls or date < pol_urls[url]:
                pol_urls[url] = date

    print(f"    /pol/ unique URLs: {len(pol_urls):,}")

    # Find overlap
    shared = set(im_urls.keys()) & set(pol_urls.keys())
    print(f"    Shared URLs: {len(shared):,}")

    im_first = 0
    pol_first = 0
    simultaneous = 0
    url_records = []

    for url in shared:
        im_date = im_urls[url]
        pol_date = pol_urls[url]

        # Determine temporal priority
        # Allow ±1 day window for "simultaneous"
        diff = (pol_date - im_date).total_seconds() / 86400
        if diff > 1:
            source = "im"
            im_first += 1
        elif diff < -1:
            source = "pol"
            pol_first += 1
        else:
            source = "simultaneous"
            simultaneous += 1

        url_records.append({
            "url": url,
            "domain": _extract_domain(url),
            "im_first_seen": str(im_date),
            "pol_first_seen": str(pol_date),
            "lag_days": round(diff, 1),
            "source_platform": source,
        })

    # Sort by lag
    url_records.sort(key=lambda r: r["lag_days"])

    return {
        "total_im_urls": len(im_urls),
        "total_pol_urls": len(pol_urls),
        "shared_urls": len(shared),
        "im_first": im_first,
        "pol_first": pol_first,
        "simultaneous": simultaneous,
        "url_records": url_records[:200],  # Cap stored examples
    }


def find_shared_ngrams(
    im_posts: pl.DataFrame,
    pol_posts: pl.DataFrame,
    n: int = 4,
    min_frequency: int = 3,
) -> dict:
    """Find distinctive n-grams shared between platforms.

    We focus on rare-ish n-grams that appear on both platforms —
    common English 4-grams are excluded by frequency filtering.
    """
    print(f"  Computing {n}-gram fingerprints…")

    # Build n-gram frequency counts per platform
    im_ngrams: Counter[str] = Counter()
    for text in im_posts["text"].drop_nulls().to_list():
        im_ngrams.update(extract_ngrams(text, n))

    pol_ngrams: Counter[str] = Counter()
    for text in pol_posts["text"].drop_nulls().to_list():
        pol_ngrams.update(extract_ngrams(text, n))

    print(f"    IM unique {n}-grams: {len(im_ngrams):,}")
    print(f"    /pol/ unique {n}-grams: {len(pol_ngrams):,}")

    # Find shared n-grams with minimum frequency on both platforms
    shared_ngrams = {}
    for ngram in im_ngrams:
        if ngram in pol_ngrams:
            im_count = im_ngrams[ngram]
            pol_count = pol_ngrams[ngram]
            if im_count >= min_frequency and pol_count >= min_frequency:
                shared_ngrams[ngram] = {
                    "im_count": im_count,
                    "pol_count": pol_count,
                    "total": im_count + pol_count,
                }

    print(f"    Shared {n}-grams (freq≥{min_frequency}): {len(shared_ngrams):,}")

    # Rank by total frequency
    ranked = sorted(shared_ngrams.items(), key=lambda x: x[1]["total"], reverse=True)

    return {
        "n": n,
        "min_frequency": min_frequency,
        "im_unique_ngrams": len(im_ngrams),
        "pol_unique_ngrams": len(pol_ngrams),
        "shared_ngram_count": len(shared_ngrams),
        "top_shared_ngrams": [
            {"ngram": k, **v} for k, v in ranked[:100]
        ],
    }


def temporal_priority_analysis(
    im_posts: pl.DataFrame,
    pol_posts: pl.DataFrame,
) -> dict:
    """Analyse which platform leads in Siege term adoption.

    For each Siege lexicon term that appears on both platforms,
    determine which platform used it first.
    """
    from importlib import import_module
    _lex = import_module("02_siege_lexicon")

    results = []
    for pattern_str, weight in _lex.SIEGE_DICTIONARY:
        if weight < 1.0:
            continue  # Skip low-weight / counter-indicators

        pat = re.compile(pattern_str, re.IGNORECASE)

        # Find earliest IM usage
        im_first = None
        for row in im_posts.sort("date").iter_rows(named=True):
            text = row.get("text") or ""
            if pat.search(text):
                im_first = row["date"]
                break

        # Find earliest /pol/ usage
        pol_first = None
        for row in pol_posts.sort("date").iter_rows(named=True):
            text = row.get("text") or ""
            if pat.search(text):
                pol_first = row["date"]
                break

        if im_first and pol_first:
            diff_days = (pol_first - im_first).total_seconds() / 86400
            results.append({
                "term": pattern_str,
                "weight": weight,
                "im_first_date": str(im_first),
                "pol_first_date": str(pol_first),
                "lag_days": round(diff_days, 1),
                "source": "im" if diff_days > 0 else "pol",
            })

    im_led = sum(1 for r in results if r["source"] == "im")
    pol_led = sum(1 for r in results if r["source"] == "pol")

    return {
        "terms_on_both": len(results),
        "im_led": im_led,
        "pol_led": pol_led,
        "term_results": results,
    }


def find_ironmarch_mentions_on_pol(pol_posts: pl.DataFrame) -> dict:
    """Find /pol/ posts explicitly mentioning Iron March.

    This directly tests whether /pol/ users are aware of and
    discussing Iron March content.
    """
    im_pattern = re.compile(
        r"iron\s*march|ironmarch\.org|iron-march",
        re.IGNORECASE,
    )

    mentions = []
    for row in pol_posts.iter_rows(named=True):
        text = row.get("text") or ""
        if im_pattern.search(text):
            mentions.append({
                "post_id": row.get("post_id"),
                "date": str(row.get("date")),
                "thread_id": row.get("thread_id"),
            })

    # Weekly counts
    if mentions:
        dates = [m["date"] for m in mentions if m["date"] != "None"]
        weekly = Counter()
        for d in dates:
            try:
                dt = datetime.datetime.fromisoformat(d)
                week_key = dt.strftime("%G-W%V")
                weekly[week_key] += 1
            except (ValueError, TypeError):
                pass
        weekly_sorted = sorted(weekly.items())
    else:
        weekly_sorted = []

    return {
        "total_mentions": len(mentions),
        "weekly_counts": dict(weekly_sorted[:200]),
        "example_posts": mentions[:50],
    }


def plot_url_lag_distribution(url_result: dict, filename: str):
    """Plot the distribution of URL appearance lags."""
    setup_plot_style()
    records = url_result.get("url_records", [])
    if not records:
        return

    lags = [r["lag_days"] for r in records if abs(r["lag_days"]) < 365]
    if not lags:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.hist(lags, bins=50, color=CB_PALETTE[0], alpha=0.7, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Simultaneous")
    ax.axvline(np.median(lags), color=CB_PALETTE[2], linestyle="-",
               linewidth=2, label=f"Median: {np.median(lags):.0f} days")

    ax.set_xlabel("Lag (days; positive = IM first)")
    ax.set_ylabel("Count")
    ax.set_title("URL Appearance Lag: Iron March → /pol/")
    ax.legend()

    save_figure(fig, filename)


def plot_im_mentions_timeline(mentions_result: dict, filename: str):
    """Plot timeline of Iron March mentions on /pol/."""
    setup_plot_style()
    weekly = mentions_result.get("weekly_counts", {})
    if not weekly:
        return

    weeks = sorted(weekly.keys())
    counts = [weekly[w] for w in weeks]

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.bar(range(len(weeks)), counts, color=CB_PALETTE[0], alpha=0.7)

    # Show every Nth label
    step = max(1, len(weeks) // 20)
    ax.set_xticks(range(0, len(weeks), step))
    ax.set_xticklabels([weeks[i] for i in range(0, len(weeks), step)],
                       rotation=45, ha="right")

    ax.set_ylabel("Mentions per week")
    ax.set_title("Iron March Mentions on /pol/ Over Time")

    save_figure(fig, filename)


def main():
    """Run cross-platform content bridge analysis."""
    print("=" * 60)
    print("PHASE 10: Cross-Platform Content Bridges (H14)")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────
    im_path = DATA_PROCESSED / "siege_scores.parquet"
    pol_path = DATA_PROCESSED / "pol_siege_scores.parquet"

    if not im_path.exists():
        print(f"  ✗ {im_path} not found.")
        return
    if not pol_path.exists():
        print(f"  ✗ {pol_path} not found. Run lexicon scoring on /pol/ first.")
        return

    im_scores = pl.read_parquet(im_path)
    im_forum = im_scores.filter(
        (pl.col("channel") == "forum")
        & (pl.col("author_id") != ZEIGER_MEMBER_ID)
    )
    pol_scores = pl.read_parquet(pol_path)

    print(f"  IM posts: {im_forum.height:,}")
    print(f"  /pol/ posts: {pol_scores.height:,}")

    results = {}

    # ── 1. URL overlap analysis ───────────────────────────────────────
    print("\n▸ URL overlap analysis")
    url_result = find_shared_urls(im_forum, pol_scores)
    results["url_analysis"] = {
        k: v for k, v in url_result.items() if k != "url_records"
    }
    results["url_analysis"]["n_url_records"] = len(url_result.get("url_records", []))

    if url_result.get("url_records"):
        plot_url_lag_distribution(url_result, "cross_platform_url_lags")

    # ── 2. N-gram fingerprint analysis ────────────────────────────────
    print("\n▸ N-gram fingerprint analysis")
    ngram_result = find_shared_ngrams(im_forum, pol_scores, n=4, min_frequency=3)
    results["ngram_analysis"] = {
        k: v for k, v in ngram_result.items() if k != "top_shared_ngrams"
    }

    # ── 3. Temporal priority of Siege terms ───────────────────────────
    print("\n▸ Temporal priority analysis")
    temporal = temporal_priority_analysis(im_forum, pol_scores)
    results["temporal_priority"] = {
        k: v for k, v in temporal.items() if k != "term_results"
    }

    # ── 4. Iron March mentions on /pol/ ───────────────────────────────
    print("\n▸ Iron March mentions on /pol/")
    im_mentions = find_ironmarch_mentions_on_pol(pol_scores)
    results["ironmarch_mentions"] = {
        "total_mentions": im_mentions["total_mentions"],
        "n_weeks_with_mentions": len(im_mentions["weekly_counts"]),
    }

    if im_mentions["total_mentions"] > 0:
        plot_im_mentions_timeline(im_mentions, "pol_ironmarch_mentions")

    # ── Save ──────────────────────────────────────────────────────────
    out_path = RESULTS_DIR / "cross_platform_bridges_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ Bridge results saved to {out_path}")


if __name__ == "__main__":
    main()
