"""25 – Country-Level Correlation (H18).

Tests whether /pol/ posts originating from countries with known Iron March
membership concentrations show higher Siege rhetoric prevalence.  IM was
disproportionately Anglo-American (US, UK, AU, CA) and Nordic — if those
same geographic pockets show elevated /pol/ Siege scores, that is
consistent with a shared recruitment pool or geographic diffusion channel.

Methodology
-----------
- Group /pol/ Siege-filtered posts by ``poster_country`` flag.
- Compare mean scores, prevalence, and volume in IM-heavy vs other.
- Mann–Whitney U test between IM-cluster countries and the rest.
- Weekly time-series overlay for top countries.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
from scipy import stats as sp_stats

from utils import (
    DATA_PROCESSED, RESULTS_DIR,
    setup_plot_style, save_figure, FIGSIZE_WIDE, CB_PALETTE,
)
import matplotlib.pyplot as plt

# Countries over-represented in Iron March membership rolls
IM_HEAVY_COUNTRIES: set[str] = {"US", "GB", "CA", "AU", "SE", "FI", "NO", "DE"}


def country_summary(pol: pl.DataFrame) -> pl.DataFrame:
    """Per-country aggregate of Siege metrics."""
    return (
        pol.filter(
            pl.col("poster_country").is_not_null()
            & (pl.col("poster_country") != "")
        )
        .group_by("poster_country")
        .agg([
            pl.len().alias("n_posts"),
            pl.col("siege_keyword_score").mean().alias("mean_score"),
            pl.col("siege_binary").mean().alias("prevalence"),
        ])
        .sort("n_posts", descending=True)
    )


def im_vs_rest_test(summary: pl.DataFrame) -> dict:
    """Mann–Whitney U comparing IM-heavy countries to the rest."""
    summary = summary.with_columns(
        pl.col("poster_country")
        .is_in(list(IM_HEAVY_COUNTRIES))
        .alias("im_cluster")
    )

    im_scores = summary.filter(pl.col("im_cluster"))["mean_score"].to_numpy()
    rest_scores = summary.filter(~pl.col("im_cluster"))["mean_score"].to_numpy()

    if len(im_scores) < 2 or len(rest_scores) < 2:
        return {"error": "insufficient countries with data"}

    u_stat, p_val = sp_stats.mannwhitneyu(im_scores, rest_scores,
                                           alternative="greater")

    return {
        "im_cluster_n": int(len(im_scores)),
        "rest_n": int(len(rest_scores)),
        "im_mean_score": float(im_scores.mean()),
        "rest_mean_score": float(rest_scores.mean()),
        "u_statistic": float(u_stat),
        "p_value": float(p_val),
        "significant_05": bool(p_val < 0.05),
    }


def weekly_by_cluster(pol: pl.DataFrame) -> pl.DataFrame:
    """Weekly time series split by IM-cluster vs rest."""
    return (
        pol.filter(
            pl.col("date").is_not_null()
            & pl.col("poster_country").is_not_null()
        )
        .with_columns([
            pl.col("date").dt.truncate("1w").alias("week"),
            pl.col("poster_country")
            .is_in(list(IM_HEAVY_COUNTRIES))
            .alias("im_cluster"),
        ])
        .group_by(["week", "im_cluster"])
        .agg([
            pl.col("siege_keyword_score").mean().alias("mean_score"),
            pl.len().alias("n_posts"),
        ])
        .sort("week")
    )


def plot_country_comparison(summary: pl.DataFrame, test_result: dict,
                            filename: str):
    """Bar chart of top countries coloured by IM-cluster membership."""
    setup_plot_style()
    top = summary.head(20).to_pandas()

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    colours = [
        CB_PALETTE[0] if c in IM_HEAVY_COUNTRIES else CB_PALETTE[2]
        for c in top["poster_country"]
    ]
    ax.barh(top["poster_country"][::-1], top["mean_score"][::-1],
            color=colours[::-1], alpha=0.8)
    ax.set_xlabel("Mean Siege Keyword Score")
    ax.set_title("Mean Siege Score by /pol/ Poster Country\n"
                 f"(blue = IM-heavy countries; U p={test_result.get('p_value', 1):.3f})")
    fig.tight_layout()
    save_figure(fig, filename)


def plot_cluster_timeseries(weekly: pl.DataFrame, filename: str):
    """Overlay time series of IM-cluster vs rest."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    for cluster_val, label, color in [
        (True, "IM-cluster countries", CB_PALETTE[0]),
        (False, "Other countries", CB_PALETTE[2]),
    ]:
        sub = weekly.filter(pl.col("im_cluster") == cluster_val).to_pandas()
        if len(sub) == 0:
            continue
        ax.plot(sub["week"], sub["mean_score"], label=label,
                color=color, alpha=0.7, linewidth=1)

    ax.set_xlabel("Date")
    ax.set_ylabel("Mean Siege Score")
    ax.set_title("Weekly Siege Score: IM-Cluster vs Other Countries")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    save_figure(fig, filename)


def main():
    print("=" * 60)
    print("H18: Country-Level Correlation")
    print("=" * 60)

    pol_path = DATA_PROCESSED / "pol_siege_scores.parquet"
    if not pol_path.exists():
        print("  ✗ pol_siege_scores.parquet not found.")
        return

    pol = pl.read_parquet(pol_path)
    print(f"  /pol/ posts: {pol.height:,}")

    if "poster_country" not in pol.columns:
        # poster_country may have been dropped during scoring;
        # recover it from the preprocessed parquet
        raw_path = DATA_PROCESSED / "pol_posts.parquet"
        if raw_path.exists():
            raw = pl.read_parquet(raw_path, columns=["post_id", "poster_country"])
            pol = pol.join(raw, on="post_id", how="left")
            print("  ⟳ Joined poster_country from pol_posts.parquet")
        else:
            print("  ✗ poster_country column not found and pol_posts.parquet missing.")
            return

    if "poster_country" not in pol.columns:
        print("  ✗ poster_country column not available.")
        return

    has_country = pol.filter(
        pl.col("poster_country").is_not_null()
        & (pl.col("poster_country") != "")
    ).height
    print(f"  Posts with country flag: {has_country:,} "
          f"({100 * has_country / pol.height:.1f}%)")

    # Per-country summary
    summary = country_summary(pol)
    print(f"  Unique countries: {summary.height}")
    print("\n  Top 10 countries by post count:")
    for row in summary.head(10).iter_rows(named=True):
        flag = "★" if row["poster_country"] in IM_HEAVY_COUNTRIES else " "
        print(f"    {flag} {row['poster_country']:>4s}: "
              f"{row['n_posts']:>8,} posts, "
              f"mean={row['mean_score']:.4f}, "
              f"prev={row['prevalence']:.4f}")

    # Statistical test
    print("\n  IM-cluster vs rest Mann–Whitney U:")
    test_result = im_vs_rest_test(summary)
    for k, v in test_result.items():
        print(f"    {k}: {v}")

    # Weekly time series by cluster
    weekly = weekly_by_cluster(pol)

    # Plots
    plot_country_comparison(summary, test_result, "country_comparison")
    plot_cluster_timeseries(weekly, "country_timeseries")

    # Save
    results = {
        "n_countries": summary.height,
        "posts_with_country": has_country,
        "im_cluster_countries": sorted(IM_HEAVY_COUNTRIES),
        "top_20_countries": summary.head(20).to_dicts(),
        "im_vs_rest_test": test_result,
    }

    with open(RESULTS_DIR / "country_correlation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\n✓ Country correlation results saved.")


if __name__ == "__main__":
    main()
