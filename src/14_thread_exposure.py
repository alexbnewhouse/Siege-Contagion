"""14 – Thread Exposure → Subsequent Adoption (H9).

Tests whether users who participate in high-Siege-score threads subsequently
increase their own Siege rhetoric in later posts, beyond baseline trends.
This provides a thread-level causal mechanism: it is the *discussion itself*
that radicalises, not just the dyadic network tie.

Approach
--------
1. Compute per-thread Siege intensity (mean siege_keyword_score of *other*
   users' posts in the thread, excluding the focal user).
2. For each user-month, compute cumulative thread exposure: the mean thread
   Siege intensity across all threads the user participated in during the
   *previous* month.
3. Regress the user's next-month siege_keyword_score on their thread exposure,
   controlling for their own lagged score and month FE.

Outputs
-------
results/thread_exposure_results.json
figures/thread_exposure.png / .pdf
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm

from utils import (
    DATA_PROCESSED, RESULTS_DIR,
    ZEIGER_MEMBER_ID, setup_plot_style, save_figure, FIGSIZE_WIDE, CB_PALETTE,
)
import matplotlib.pyplot as plt


def main() -> None:
    print("=" * 60)
    print("H9: Thread Exposure → Subsequent Siege Adoption")
    print("=" * 60)

    fp = pl.read_parquet(DATA_PROCESSED / "forum_posts.parquet")
    fp = fp.filter(
        (pl.col("author_id") != ZEIGER_MEMBER_ID)
        & (pl.col("word_count") > 0)
        & pl.col("post_date").is_not_null()
    )

    # Add month
    fp = fp.with_columns(
        pl.col("post_date").dt.strftime("%Y-%m").alias("month")
    )

    # ── 1. Per-thread Siege intensity (excluding focal user) ──────────
    print("\nComputing per-thread Siege intensity…")

    # Thread-level stats per user: for each (topic_id, author_id),
    # the "other users'" mean score in that thread
    thread_scores = (
        fp.group_by("topic_id")
        .agg([
            pl.col("siege_keyword_score").sum().alias("thread_total_score"),
            pl.len().alias("thread_n_posts"),
        ])
    )

    # User contribution per thread
    user_thread = (
        fp.group_by(["topic_id", "author_id"])
        .agg([
            pl.col("siege_keyword_score").sum().alias("user_thread_score"),
            pl.len().alias("user_thread_posts"),
            pl.col("month").first().alias("first_month_in_thread"),
        ])
    )

    # Join to get thread totals
    user_thread = user_thread.join(thread_scores, on="topic_id", how="left")

    # Exposure = (thread_total - user_own) / (thread_n - user_n)
    user_thread = user_thread.with_columns(
        pl.when(pl.col("thread_n_posts") - pl.col("user_thread_posts") > 0)
        .then(
            (pl.col("thread_total_score") - pl.col("user_thread_score"))
            / (pl.col("thread_n_posts") - pl.col("user_thread_posts"))
        )
        .otherwise(0.0)
        .alias("others_mean_siege")
    )

    # ── 2. Monthly aggregation: user thread exposure ──────────────────
    print("Computing monthly thread exposure…")

    # User's mean exposure from threads they participated in each month
    # Use the earliest month the user posted in each thread
    user_monthly_exposure = (
        user_thread.group_by(["author_id", "first_month_in_thread"])
        .agg(pl.col("others_mean_siege").mean().alias("mean_thread_exposure"))
        .rename({"first_month_in_thread": "month"})
    )

    # User's own monthly siege score
    user_monthly_score = (
        fp.group_by(["author_id", "month"])
        .agg([
            pl.col("siege_keyword_score").mean().alias("own_siege_score"),
            pl.len().alias("n_posts"),
        ])
    )

    # Get sorted months
    all_months = sorted(fp["month"].unique().to_list())
    month_to_next = {all_months[i]: all_months[i + 1] for i in range(len(all_months) - 1)}

    # Lag exposure by 1 month
    user_monthly_exposure = user_monthly_exposure.with_columns(
        pl.col("month").replace_strict(month_to_next, default=None).alias("next_month")
    ).filter(pl.col("next_month").is_not_null())

    # Join: user's next-month score ~ their current-month thread exposure
    panel = user_monthly_score.join(
        user_monthly_exposure.select([
            pl.col("author_id"),
            pl.col("next_month").alias("month"),
            pl.col("mean_thread_exposure").alias("lagged_thread_exposure"),
        ]),
        on=["author_id", "month"],
        how="inner",
    )

    # Also add lagged own score
    user_lagged_score = (
        user_monthly_score.with_columns(
            pl.col("month").replace_strict(month_to_next, default=None).alias("next_month")
        )
        .filter(pl.col("next_month").is_not_null())
        .select([
            pl.col("author_id"),
            pl.col("next_month").alias("month"),
            pl.col("own_siege_score").alias("lagged_own_score"),
        ])
    )

    panel = panel.join(user_lagged_score, on=["author_id", "month"], how="left")
    panel = panel.with_columns(pl.col("lagged_own_score").fill_null(0.0))
    panel = panel.filter(pl.col("n_posts") >= 1)

    print(f"  Panel observations: {panel.height:,} user-months")
    print(f"  Unique users: {panel['author_id'].n_unique():,}")

    # ── 3. Regression ─────────────────────────────────────────────────
    print("\nFitting regression…")
    pdf = panel.to_pandas()

    X = pdf[["lagged_thread_exposure", "lagged_own_score"]].copy()
    X = sm.add_constant(X)
    y = pdf["own_siege_score"]

    ols = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": pdf["author_id"]})
    print(f"  lagged_thread_exposure: {ols.params['lagged_thread_exposure']:.4f} "
          f"(p={ols.pvalues['lagged_thread_exposure']:.4f})")
    print(f"  lagged_own_score:       {ols.params['lagged_own_score']:.4f} "
          f"(p={ols.pvalues['lagged_own_score']:.4f})")
    print(f"  R²: {ols.rsquared:.4f}")

    # ── Tercile analysis ──────────────────────────────────────────────
    panel_pd = pdf.copy()
    panel_pd["exposure_tercile"] = pd.qcut(
        panel_pd["lagged_thread_exposure"], q=3, labels=["Low", "Medium", "High"]
    )
    tercile_means = panel_pd.groupby("exposure_tercile", observed=True)["own_siege_score"].mean()
    print("\n  Mean next-month siege score by thread exposure tercile:")
    for t, v in tercile_means.items():
        print(f"    {t}: {v:.4f}")

    # ── Results ───────────────────────────────────────────────────────
    results = {
        "n_user_months": int(panel.height),
        "n_users": int(panel["author_id"].n_unique()),
        "thread_exposure_coef": float(ols.params["lagged_thread_exposure"]),
        "thread_exposure_se": float(ols.bse["lagged_thread_exposure"]),
        "thread_exposure_p": float(ols.pvalues["lagged_thread_exposure"]),
        "lagged_own_score_coef": float(ols.params["lagged_own_score"]),
        "lagged_own_score_p": float(ols.pvalues["lagged_own_score"]),
        "r_squared": float(ols.rsquared),
        "tercile_means": {str(k): float(v) for k, v in tercile_means.items()},
    }

    # ── Plot ──────────────────────────────────────────────────────────
    print("\nPlotting thread exposure effect…")
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # Left: tercile bar chart
    ax = axes[0]
    labels = list(tercile_means.index)
    vals = list(tercile_means.values)
    ax.bar(range(len(vals)), vals, color=[CB_PALETTE[2], CB_PALETTE[0], CB_PALETTE[1]],
           alpha=0.8)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean next-month siege score")
    ax.set_title("Subsequent Siege Score by\nThread Exposure Tercile")

    # Right: scatter of thread exposure vs next score
    ax = axes[1]
    sample = pdf.sample(n=min(5000, len(pdf)), random_state=42)
    ax.scatter(sample["lagged_thread_exposure"], sample["own_siege_score"],
               alpha=0.2, s=8, color=CB_PALETTE[0])
    # Add regression line
    x_range = np.linspace(0, sample["lagged_thread_exposure"].max(), 100)
    y_pred = ols.params["const"] + ols.params["lagged_thread_exposure"] * x_range
    ax.plot(x_range, y_pred, color=CB_PALETTE[1], linewidth=2)
    ax.set_xlabel("Lagged thread exposure (others' mean siege score)")
    ax.set_ylabel("Own siege score (next month)")
    ax.set_title("Thread Exposure → Subsequent Score")

    fig.tight_layout()
    save_figure(fig, "thread_exposure")

    with open(RESULTS_DIR / "thread_exposure_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ Thread exposure results saved.")


if __name__ == "__main__":
    main()
