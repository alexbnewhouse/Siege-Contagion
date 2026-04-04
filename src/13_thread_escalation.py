"""13 – Within-Thread Escalation (H8).

Tests whether Siege rhetoric *escalates* within forum threads – that is,
later replies in a Siege-mentioning thread score higher than the opening
posts.  If exegesis is a collective process (group interpretation), we
expect within-thread amplification rather than simple individual expression.

Approach
--------
1. Identify "Siege threads" (threads containing >= 1 Siege-flagged post).
2. For each Siege thread, order posts chronologically and assign a
   normalised within-thread position (0 = first, 1 = last).
3. Regress siege_keyword_score on within-thread position, controlling for
   post length and thread fixed effects.
4. Test whether the position coefficient is positive (escalation).

Outputs
-------
results/thread_escalation_results.json
figures/thread_escalation.png / .pdf
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
from scipy import stats

from utils import (
    DATA_PROCESSED, RESULTS_DIR,
    ZEIGER_MEMBER_ID, setup_plot_style, save_figure, FIGSIZE_WIDE, CB_PALETTE,
)
import matplotlib.pyplot as plt


def build_thread_position_df() -> pl.DataFrame:
    """Add within-thread position to forum posts in Siege threads."""
    fp = pl.read_parquet(DATA_PROCESSED / "forum_posts.parquet")
    fp = fp.filter(pl.col("author_id") != ZEIGER_MEMBER_ID)
    fp = fp.filter(pl.col("word_count") > 0)

    # Identify Siege threads (threads with at least 1 siege post)
    siege_threads = (
        fp.filter(pl.col("siege_binary") == 1)
        .select("topic_id")
        .unique()
    )

    # Keep only posts in Siege threads
    df = fp.join(siege_threads, on="topic_id", how="inner")

    # Within-thread post ordering by date
    df = df.sort(["topic_id", "post_date"])
    df = df.with_columns(
        pl.col("pid")
        .cum_count()
        .over("topic_id")
        .alias("post_position_1idx")
    )
    # Thread size
    df = df.with_columns(
        pl.col("pid").count().over("topic_id").alias("thread_size")
    )
    # Filter to threads with >=3 posts (need room for escalation)
    df = df.filter(pl.col("thread_size") >= 3)

    # Normalised position [0, 1]
    df = df.with_columns(
        ((pl.col("post_position_1idx") - 1) / (pl.col("thread_size") - 1))
        .alias("position_norm")
    )

    return df


def main() -> None:
    print("=" * 60)
    print("H8: Within-Thread Escalation of Siege Rhetoric")
    print("=" * 60)

    df = build_thread_position_df()
    n_threads = df["topic_id"].n_unique()
    print(f"  Siege threads (≥3 posts): {n_threads:,}")
    print(f"  Posts in Siege threads: {df.height:,}")

    # ── OLS: siege_score ~ position + log(word_count) ─────────────────
    print("\nFitting position regression…")
    pdf = df.select([
        "siege_keyword_score", "position_norm", "word_count", "topic_id",
    ]).to_pandas()
    pdf["log_word_count"] = np.log1p(pdf["word_count"])

    # Simple model
    X = pdf[["position_norm", "log_word_count"]].copy()
    X = sm.add_constant(X)
    y = pdf["siege_keyword_score"]
    ols_simple = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": pdf["topic_id"]})

    print(f"  position_norm coef: {ols_simple.params['position_norm']:.4f} "
          f"(p={ols_simple.pvalues['position_norm']:.4f})")
    print(f"  R²: {ols_simple.rsquared:.4f}")

    # ── Binned analysis: mean score by position quintile ──────────────
    df_binned = df.with_columns(
        (pl.col("position_norm") * 4).cast(pl.Int32).clip(0, 3).alias("quartile")
    )
    quartile_means = (
        df_binned.group_by("quartile")
        .agg([
            pl.col("siege_keyword_score").mean().alias("mean_score"),
            pl.col("siege_keyword_score").std().alias("std_score"),
            pl.len().alias("n"),
        ])
        .sort("quartile")
    )
    print("\n  Mean siege score by within-thread position quartile:")
    for row in quartile_means.iter_rows(named=True):
        labels = ["Q1 (start)", "Q2", "Q3", "Q4 (end)"]
        print(f"    {labels[row['quartile']]}: {row['mean_score']:.4f} "
              f"(n={row['n']:,})")

    # ── Paired first-vs-last test within threads ──────────────────────
    first_posts = df.sort(["topic_id", "post_date"]).group_by("topic_id").first()
    last_posts = df.sort(["topic_id", "post_date"]).group_by("topic_id").last()

    paired = first_posts.select([
        pl.col("topic_id"),
        pl.col("siege_keyword_score").alias("first_score"),
    ]).join(
        last_posts.select([
            pl.col("topic_id"),
            pl.col("siege_keyword_score").alias("last_score"),
        ]),
        on="topic_id",
    )
    diffs = (paired["last_score"] - paired["first_score"]).to_numpy()
    t_stat, t_p = stats.ttest_1samp(diffs, 0)
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(diffs, alternative="two-sided")

    print(f"\n  First vs. last post in thread:")
    print(f"    Mean first: {paired['first_score'].mean():.4f}")
    print(f"    Mean last:  {paired['last_score'].mean():.4f}")
    print(f"    Mean diff:  {diffs.mean():.4f}")
    print(f"    t-test: t={t_stat:.3f}, p={t_p:.4f}")
    print(f"    Wilcoxon: p={wilcoxon_p:.4f}")

    # ── Results ───────────────────────────────────────────────────────
    results = {
        "n_siege_threads": int(n_threads),
        "n_posts_in_siege_threads": int(df.height),
        "regression": {
            "position_coef": float(ols_simple.params["position_norm"]),
            "position_se": float(ols_simple.bse["position_norm"]),
            "position_p": float(ols_simple.pvalues["position_norm"]),
            "r_squared": float(ols_simple.rsquared),
        },
        "quartile_means": {
            f"Q{row['quartile']+1}": float(row["mean_score"])
            for row in quartile_means.iter_rows(named=True)
        },
        "first_vs_last": {
            "mean_first": float(paired["first_score"].mean()),
            "mean_last": float(paired["last_score"].mean()),
            "mean_diff": float(diffs.mean()),
            "t_stat": float(t_stat),
            "t_p": float(t_p),
            "wilcoxon_p": float(wilcoxon_p),
        },
    }

    # ── Plot ──────────────────────────────────────────────────────────
    print("\nPlotting thread escalation…")
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # Left: mean score by position quartile
    ax = axes[0]
    qm = quartile_means.to_pandas()
    labels = ["Q1\n(start)", "Q2", "Q3", "Q4\n(end)"]
    ax.bar(range(4), qm["mean_score"], color=CB_PALETTE[0], alpha=0.8,
           yerr=qm["std_score"] / np.sqrt(qm["n"]))
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean siege keyword score")
    ax.set_title("Siege Score by Thread Position")

    # Right: distribution of last-first differences
    ax = axes[1]
    ax.hist(diffs, bins=50, color=CB_PALETTE[1], alpha=0.8, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="No change")
    ax.axvline(diffs.mean(), color=CB_PALETTE[2], linestyle="-", linewidth=1.5,
               label=f"Mean = {diffs.mean():.3f}")
    ax.set_xlabel("Last − First post score")
    ax.set_ylabel("Number of threads")
    ax.set_title("Within-Thread Score Change")
    ax.legend()

    fig.tight_layout()
    save_figure(fig, "thread_escalation")

    with open(RESULTS_DIR / "thread_escalation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ Thread escalation results saved.")


if __name__ == "__main__":
    main()
