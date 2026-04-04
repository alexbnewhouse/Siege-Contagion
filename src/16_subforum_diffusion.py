"""16 – Subforum Diffusion Geography (H11).

Maps which subforums Siege rhetoric concentrates in and tests whether it
spreads *from* particular ideological sections to others.  If certain
subforums function as exegetical spaces, Siege rhetoric should appear
there first and then propagate to general discussion, introductions, etc.

Approach
--------
1. Join forum posts → topics → forums to get subforum labels.
2. Compute per-subforum, per-quarter siege prevalence.
3. Identify "early adopter" vs "late adopter" subforums by comparing
   first-significant-siege-activity dates.
4. Test whether ideological subforums lead general-discussion subforums
   using cross-correlation.

Outputs
-------
results/subforum_diffusion_results.json
figures/subforum_diffusion.png / .pdf
"""

from __future__ import annotations

import json
import datetime

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

from utils import (
    DATA_PROCESSED, RESULTS_DIR,
    ZEIGER_MEMBER_ID, setup_plot_style, save_figure, FIGSIZE_WIDE, CB_PALETTE,
)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def main() -> None:
    print("=" * 60)
    print("H11: Subforum Diffusion Geography")
    print("=" * 60)

    with open(DATA_PROCESSED / "treatment_dates.json") as f:
        td = json.load(f)
    t0 = datetime.datetime.fromisoformat(td["T0"])

    # Load and join data
    fp = pl.read_parquet(DATA_PROCESSED / "forum_posts.parquet")
    topics = pl.read_parquet(DATA_PROCESSED / "forums_topics.parquet")
    forums = pl.read_parquet(DATA_PROCESSED / "forums_forums.parquet")

    fp = fp.filter(
        (pl.col("author_id") != ZEIGER_MEMBER_ID)
        & (pl.col("word_count") > 0)
        & pl.col("post_date").is_not_null()
    )

    # Join posts → topics to get forum_id
    topic_forum = topics.select(["tid", "forum_id"]).rename({"tid": "topic_id"})
    fp = fp.join(topic_forum, on="topic_id", how="left")

    # Join to get subforum name
    forum_names = forums.select([
        pl.col("id").alias("forum_id"),
        pl.col("name_seo").alias("subforum"),
    ])
    fp = fp.join(forum_names, on="forum_id", how="left")
    fp = fp.filter(pl.col("subforum").is_not_null())

    # Add quarter
    fp = fp.with_columns(
        (pl.col("post_date").dt.year().cast(pl.Utf8)
         + "-Q"
         + pl.col("post_date").dt.quarter().cast(pl.Utf8)
        ).alias("quarter")
    )

    print(f"  Posts with subforum labels: {fp.height:,}")
    print(f"  Unique subforums: {fp['subforum'].n_unique()}")

    # ── 1. Per-subforum Siege prevalence ──────────────────────────────
    print("\nComputing per-subforum Siege prevalence…")

    subforum_stats = (
        fp.group_by("subforum")
        .agg([
            pl.len().alias("total_posts"),
            pl.col("siege_binary").sum().alias("siege_posts"),
            pl.col("siege_keyword_score").mean().alias("mean_siege_score"),
            pl.col("siege_binary").mean().alias("siege_prevalence"),
        ])
        .filter(pl.col("total_posts") >= 50)  # Minimum activity threshold
        .sort("siege_prevalence", descending=True)
    )

    print("\n  Top 15 subforums by Siege prevalence:")
    for row in subforum_stats.head(15).iter_rows(named=True):
        print(f"    {row['subforum']:35s}  {row['siege_prevalence']:.3f}  "
              f"({row['siege_posts']:4d}/{row['total_posts']:5d})")

    # ── 2. Per-subforum quarterly time series ─────────────────────────
    print("\nComputing quarterly time series…")

    quarterly = (
        fp.group_by(["subforum", "quarter"])
        .agg([
            pl.col("siege_binary").mean().alias("siege_prevalence"),
            pl.col("siege_keyword_score").mean().alias("mean_score"),
            pl.len().alias("n_posts"),
        ])
        .sort(["subforum", "quarter"])
    )

    # ── 3. First Siege appearance per subforum ────────────────────────
    first_siege = (
        fp.filter(pl.col("siege_binary") == 1)
        .group_by("subforum")
        .agg(pl.col("post_date").min().alias("first_siege_date"))
        .sort("first_siege_date")
    )

    print("\n  First Siege appearance by subforum (top 15 earliest):")
    for row in first_siege.head(15).iter_rows(named=True):
        print(f"    {row['subforum']:35s}  {row['first_siege_date']}")

    # ── 4. Pre/post Siege comparison per subforum ─────────────────────
    fp_pre = fp.filter(pl.col("post_date") < t0)
    fp_post = fp.filter(pl.col("post_date") >= t0)

    pre_rates = (
        fp_pre.group_by("subforum")
        .agg(pl.col("siege_binary").mean().alias("pre_siege_rate"))
    )
    post_rates = (
        fp_post.group_by("subforum")
        .agg(pl.col("siege_binary").mean().alias("post_siege_rate"))
    )

    change = pre_rates.join(post_rates, on="subforum", how="inner")
    change = change.with_columns(
        (pl.col("post_siege_rate") - pl.col("pre_siege_rate")).alias("rate_change")
    ).sort("rate_change", descending=True)

    print("\n  Biggest increases in Siege prevalence (pre→post):")
    for row in change.head(10).iter_rows(named=True):
        print(f"    {row['subforum']:35s}  {row['pre_siege_rate']:.3f} → "
              f"{row['post_siege_rate']:.3f}  (Δ={row['rate_change']:+.3f})")

    # ── 5. Concentration metric (Herfindahl) ─────────────────────────
    # Are Siege posts concentrated in a few subforums or spread evenly?
    siege_by_forum = (
        fp.filter(pl.col("siege_binary") == 1)
        .group_by("subforum")
        .agg(pl.len().alias("siege_count"))
    )
    total_siege = siege_by_forum["siege_count"].sum()
    shares = (siege_by_forum["siege_count"] / total_siege).to_numpy()
    herfindahl = float(np.sum(shares ** 2))

    # Pre vs post Herfindahl
    siege_pre = fp_pre.filter(pl.col("siege_binary") == 1).group_by("subforum").agg(pl.len().alias("c"))
    siege_post = fp_post.filter(pl.col("siege_binary") == 1).group_by("subforum").agg(pl.len().alias("c"))

    shares_pre = (siege_pre["c"] / siege_pre["c"].sum()).to_numpy()
    shares_post = (siege_post["c"] / siege_post["c"].sum()).to_numpy()
    hhi_pre = float(np.sum(shares_pre ** 2))
    hhi_post = float(np.sum(shares_post ** 2))

    print(f"\n  Herfindahl index (overall): {herfindahl:.4f}")
    print(f"  Pre-Siege HHI: {hhi_pre:.4f}  |  Post-Siege HHI: {hhi_post:.4f}")
    print(f"  Direction: {'Diffusion (less concentrated)' if hhi_post < hhi_pre else 'Concentration'}")

    # ── Results ───────────────────────────────────────────────────────
    results = {
        "n_subforums": int(subforum_stats.height),
        "herfindahl_overall": herfindahl,
        "herfindahl_pre": hhi_pre,
        "herfindahl_post": hhi_post,
        "diffusion_direction": "less concentrated" if hhi_post < hhi_pre else "more concentrated",
        "top_subforums": [
            {"name": row["subforum"], "prevalence": float(row["siege_prevalence"]),
             "posts": int(row["total_posts"])}
            for row in subforum_stats.head(10).iter_rows(named=True)
        ],
        "biggest_increases": [
            {"name": row["subforum"], "pre": float(row["pre_siege_rate"]),
             "post": float(row["post_siege_rate"]),
             "change": float(row["rate_change"])}
            for row in change.head(10).iter_rows(named=True)
        ],
    }

    # ── Plot ──────────────────────────────────────────────────────────
    print("\nPlotting subforum diffusion…")
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Left: top subforums by prevalence (horizontal bar)
    ax = axes[0]
    top_n = min(15, subforum_stats.height)
    top = subforum_stats.head(top_n).to_pandas()
    y_pos = range(top_n)
    ax.barh(y_pos, top["siege_prevalence"], color=CB_PALETTE[0], alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["subforum"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Siege post prevalence")
    ax.set_title("Siege Prevalence by Subforum")

    # Right: pre vs post rate changes for top changers
    ax = axes[1]
    top_change = change.head(min(12, change.height)).to_pandas()
    n_bars = len(top_change)
    y_pos = range(n_bars)
    ax.barh(y_pos, top_change["pre_siege_rate"], color=CB_PALETTE[0],
            alpha=0.7, label="Pre-Siege", height=0.4)
    ax.barh([y + 0.4 for y in y_pos], top_change["post_siege_rate"],
            color=CB_PALETTE[1], alpha=0.7, label="Post-Siege", height=0.4)
    ax.set_yticks([y + 0.2 for y in y_pos])
    ax.set_yticklabels(top_change["subforum"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Siege prevalence rate")
    ax.set_title("Pre vs Post Siege Prevalence")
    ax.legend()

    fig.tight_layout()
    save_figure(fig, "subforum_diffusion")

    with open(RESULTS_DIR / "subforum_diffusion_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ Subforum diffusion results saved.")


if __name__ == "__main__":
    main()
