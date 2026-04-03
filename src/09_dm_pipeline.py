"""09 – Private-to-Public Pipeline (H5).

Tests whether siegist rhetoric appears in DMs before public forum posts
for the same users.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
from scipy import stats

from utils import (
    DATA_PROCESSED, RESULTS_DIR,
    ZEIGER_MEMBER_ID, setup_plot_style, save_figure, FIGSIZE_WIDE, CB_PALETTE,
)
import matplotlib.pyplot as plt


def main():
    print("=" * 60)
    print("PHASE 6: Private-to-Public Pipeline (H5)")
    print("=" * 60)

    scores = pl.read_parquet(DATA_PROCESSED / "siege_scores.parquet")
    scores = scores.filter(
        (pl.col("author_id") != ZEIGER_MEMBER_ID)
        & (pl.col("siege_binary") == 1)
        & pl.col("date").is_not_null()
    )

    # Split by channel
    forum_siege = scores.filter(pl.col("channel") == "forum")
    dm_siege = scores.filter(pl.col("channel") == "dm")

    # First siege date per user per channel
    first_forum = (
        forum_siege.group_by("author_id")
        .agg(pl.col("date").min().alias("first_forum_siege_date"))
    )
    first_dm = (
        dm_siege.group_by("author_id")
        .agg(pl.col("date").min().alias("first_dm_siege_date"))
    )

    # Users with siege in both channels
    both = first_forum.join(first_dm, on="author_id", how="inner")
    print(f"  Users with siege in both DM and forum: {both.height}")

    results = {}

    if both.height < 3:
        results["error"] = f"Only {both.height} users with siege in both channels"
        print(f"  ⚠ Insufficient data for pipeline analysis")
    else:
        # Compute lag: positive means DMs precede forum
        both = both.with_columns(
            (pl.col("first_forum_siege_date") - pl.col("first_dm_siege_date"))
            .dt.total_days()
            .alias("dm_lead_days")
        )

        lags = both["dm_lead_days"].to_numpy().astype(float)
        lags = lags[~np.isnan(lags)]

        mean_lag = float(np.mean(lags))
        median_lag = float(np.median(lags))

        # One-sample t-test: is mean lag > 0?
        t_stat, t_p = stats.ttest_1samp(lags, 0.0)
        # Wilcoxon signed-rank test
        try:
            w_stat, w_p = stats.wilcoxon(lags, alternative="greater")
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")

        dm_first_count = int(np.sum(lags > 0))
        forum_first_count = int(np.sum(lags < 0))
        same_count = int(np.sum(lags == 0))

        results = {
            "n_users": int(len(lags)),
            "mean_dm_lead_days": mean_lag,
            "median_dm_lead_days": median_lag,
            "dm_first_pct": float(dm_first_count / len(lags) * 100),
            "forum_first_pct": float(forum_first_count / len(lags) * 100),
            "same_pct": float(same_count / len(lags) * 100),
            "t_stat": float(t_stat),
            "t_pvalue": float(t_p),
            "wilcoxon_stat": float(w_stat),
            "wilcoxon_pvalue": float(w_p),
        }

        print(f"  Mean DM lead: {mean_lag:.1f} days")
        print(f"  Median DM lead: {median_lag:.1f} days")
        print(f"  DM first: {dm_first_count}/{len(lags)} ({results['dm_first_pct']:.1f}%)")
        print(f"  Forum first: {forum_first_count}/{len(lags)} ({results['forum_first_pct']:.1f}%)")
        print(f"  t-test: t={t_stat:.3f}, p={t_p:.4f}")

        # ── Plot lag distribution ─────────────────────────────────────
        setup_plot_style()
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        ax.hist(lags, bins=30, color=CB_PALETTE[0], edgecolor="white", alpha=0.8)
        ax.axvline(0, color="red", linestyle="--", linewidth=2, alpha=0.7,
                   label="Simultaneous")
        ax.axvline(mean_lag, color=CB_PALETTE[2], linestyle="-", linewidth=2,
                   label=f"Mean = {mean_lag:.0f} days")
        ax.set_xlabel("DM → Forum lag (days; positive = DMs first)")
        ax.set_ylabel("Number of users")
        ax.set_title("Private-to-Public Pipeline: DM Lead Over Forum")
        ax.legend()
        save_figure(fig, "dm_to_forum_lag")

    with open(RESULTS_DIR / "pipeline_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ Pipeline analysis results saved.")


if __name__ == "__main__":
    main()
