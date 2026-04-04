"""20 – Within-Thread Escalation on /pol/ (H8-pol).

Adapts the H8 within-thread escalation test for /pol/ threads.
Tests whether Siege rhetoric escalates within /pol/ threads that
mention Siege-related terms — evidence that /pol/'s anonymous
discussion structure also produces collective amplification.

On /pol/, thread_id replaces topic_id, and author_id is the
poster_hash (or "anon" for posts without a poster ID).

Outputs
-------
results/pol_thread_escalation_results.json
figures/pol_thread_escalation.png / .pdf
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
    setup_plot_style, save_figure, FIGSIZE_WIDE, CB_PALETTE,
)
import matplotlib.pyplot as plt


def build_thread_position_df_pol() -> pl.DataFrame:
    """Build within-thread position data for /pol/ Siege threads."""
    pol = pl.read_parquet(DATA_PROCESSED / "pol_siege_scores.parquet")

    # Filter to posts with actual content
    pol = pol.filter(
        pl.col("word_count") > 0
    )

    # Identify Siege threads (threads with at least 1 siege post)
    siege_threads = (
        pol.filter(pl.col("siege_binary") == 1)
        .select("thread_id")
        .unique()
    )

    # Keep only posts in Siege threads
    df = pol.join(siege_threads, on="thread_id", how="inner")

    # Within-thread ordering by date
    df = df.sort(["thread_id", "date"])
    df = df.with_columns(
        pl.col("post_id")
        .cum_count()
        .over("thread_id")
        .alias("post_position_1idx")
    )
    # Thread size
    df = df.with_columns(
        pl.col("post_id").count().over("thread_id").alias("thread_size")
    )
    # Filter to threads with >=3 posts
    df = df.filter(pl.col("thread_size") >= 3)

    # Normalised position [0, 1]
    df = df.with_columns(
        ((pl.col("post_position_1idx") - 1) / (pl.col("thread_size") - 1))
        .alias("position_norm")
    )

    return df


def main() -> None:
    print("=" * 60)
    print("H8-pol: Within-Thread Escalation on /pol/")
    print("=" * 60)

    pol_path = DATA_PROCESSED / "pol_siege_scores.parquet"
    if not pol_path.exists():
        print(f"  ✗ {pol_path} not found.")
        return

    df = build_thread_position_df_pol()
    n_threads = df["thread_id"].n_unique()
    print(f"  /pol/ Siege threads (≥3 posts): {n_threads:,}")
    print(f"  Posts in Siege threads: {df.height:,}")

    if n_threads < 10:
        print("  ⚠ Too few threads for meaningful analysis.")
        return

    # ── OLS: siege_score ~ position + log(word_count) ─────────────────
    print("\nFitting position regression…")
    pdf = df.select([
        "siege_keyword_score", "position_norm", "word_count", "thread_id",
    ]).to_pandas()
    pdf["log_word_count"] = np.log1p(pdf["word_count"])

    X = pdf[["position_norm", "log_word_count"]].copy()
    X = sm.add_constant(X)
    y = pdf["siege_keyword_score"]
    ols = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": pdf["thread_id"]})

    print(f"  position_norm coef: {ols.params['position_norm']:.4f} "
          f"(p={ols.pvalues['position_norm']:.4f})")
    print(f"  R²: {ols.rsquared:.4f}")

    # ── Binned analysis ───────────────────────────────────────────────
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

    print("\n  Mean siege score by within-thread position:")
    for row in quartile_means.iter_rows(named=True):
        labels = ["Q1 (start)", "Q2", "Q3", "Q4 (end)"]
        print(f"    {labels[row['quartile']]}: {row['mean_score']:.4f} "
              f"(n={row['n']:,})")

    # ── First vs. last post ───────────────────────────────────────────
    first_posts = df.sort(["thread_id", "date"]).group_by("thread_id").first()
    last_posts = df.sort(["thread_id", "date"]).group_by("thread_id").last()

    paired = first_posts.select([
        pl.col("thread_id"),
        pl.col("siege_keyword_score").alias("first_score"),
    ]).join(
        last_posts.select([
            pl.col("thread_id"),
            pl.col("siege_keyword_score").alias("last_score"),
        ]),
        on="thread_id",
    )
    diffs = (paired["last_score"] - paired["first_score"]).to_numpy()
    t_stat, t_p = stats.ttest_1samp(diffs, 0)

    # Wilcoxon requires non-zero diffs
    nonzero_diffs = diffs[diffs != 0]
    if len(nonzero_diffs) > 10:
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(nonzero_diffs, alternative="two-sided")
    else:
        wilcoxon_stat, wilcoxon_p = float("nan"), float("nan")

    print(f"\n  First vs. last post:")
    print(f"    Mean first: {paired['first_score'].mean():.4f}")
    print(f"    Mean last:  {paired['last_score'].mean():.4f}")
    print(f"    Mean diff:  {diffs.mean():.4f}")
    print(f"    t-test: t={t_stat:.3f}, p={t_p:.4f}")

    # ── Results ───────────────────────────────────────────────────────
    results = {
        "platform": "pol",
        "n_siege_threads": int(n_threads),
        "n_posts_in_siege_threads": int(df.height),
        "regression": {
            "position_coef": float(ols.params["position_norm"]),
            "position_se": float(ols.bse["position_norm"]),
            "position_p": float(ols.pvalues["position_norm"]),
            "r_squared": float(ols.rsquared),
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
    print("\nPlotting…")
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    ax = axes[0]
    qm = quartile_means.to_pandas()
    labels = ["Q1\n(start)", "Q2", "Q3", "Q4\n(end)"]
    ax.bar(range(4), qm["mean_score"], color=CB_PALETTE[2], alpha=0.8,
           yerr=qm["std_score"] / np.sqrt(qm["n"]))
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean siege keyword score")
    ax.set_title("/pol/ Siege Score by Thread Position")

    ax = axes[1]
    ax.hist(diffs, bins=50, color=CB_PALETTE[3], alpha=0.8, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="No change")
    ax.axvline(diffs.mean(), color=CB_PALETTE[2], linestyle="-", linewidth=1.5,
               label=f"Mean = {diffs.mean():.3f}")
    ax.set_xlabel("Last − First post score")
    ax.set_ylabel("Number of threads")
    ax.set_title("/pol/ Within-Thread Score Change")
    ax.legend()

    fig.tight_layout()
    save_figure(fig, "pol_thread_escalation")

    # ── Save ──────────────────────────────────────────────────────────
    with open(RESULTS_DIR / "pol_thread_escalation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ /pol/ thread escalation results saved.")


if __name__ == "__main__":
    main()
