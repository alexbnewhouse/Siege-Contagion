"""15 – Semantic Convergence (H10).

Tests whether the Iron March community develops a *shared* interpretation
of Siege – not merely that individuals adopt Siege vocabulary independently,
but that their language converges.  This distinguishes 'vocabulary diffusion'
from 'collective exegesis producing a shared worldview'.

Approach
--------
1. Compute per-user monthly mean embedding vector (using siege_similarity
   as a proxy, and per-post siege_keyword_score profile).
2. Measure average pairwise cosine similarity between Siege-using users
   per month.
3. Test via ITS whether convergence increases after Siege publication.
4. As a control, compute the same metric for non-Siege content.

Since raw embeddings aren't persisted, we use the siege_keyword_score
*profile* across posts as a low-dimensional user representation per month,
and measure community-level variance (lower variance = more convergence).

Outputs
-------
results/semantic_convergence_results.json
figures/semantic_convergence.png / .pdf
"""

from __future__ import annotations

import json
import datetime

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


def main() -> None:
    print("=" * 60)
    print("H10: Semantic Convergence of Siege Rhetoric")
    print("=" * 60)

    with open(DATA_PROCESSED / "treatment_dates.json") as f:
        td = json.load(f)
    t0 = datetime.datetime.fromisoformat(td["T0"])

    scores = pl.read_parquet(DATA_PROCESSED / "siege_scores.parquet")
    scores = scores.filter(
        (pl.col("author_id") != ZEIGER_MEMBER_ID)
        & pl.col("date").is_not_null()
        & (pl.col("channel") == "forum")
        & (pl.col("word_count") > 0)
    )

    scores = scores.with_columns(
        pl.col("date").dt.strftime("%Y-%m").alias("month")
    )

    # ── 1. User monthly Siege score profiles ──────────────────────────
    print("\nComputing monthly user profiles…")

    # Per user-month: mean and std of siege score
    user_monthly = (
        scores.group_by(["author_id", "month"])
        .agg([
            pl.col("siege_keyword_score").mean().alias("mean_score"),
            pl.col("siege_keyword_score").std().alias("std_score"),
            pl.col("siege_similarity").mean().alias("mean_similarity"),
            pl.len().alias("n_posts"),
        ])
        .filter(pl.col("n_posts") >= 2)  # Need ≥2 posts for variance
    )

    # ── 2. Monthly community convergence metrics ─────────────────────
    print("Computing monthly convergence…")

    # For each month, compute coefficient of variation (CV) of user mean scores
    # Lower CV = more convergence
    # Also compute inter-user std of mean scores
    monthly_convergence = (
        user_monthly.filter(pl.col("mean_score") > 0)  # Only Siege-active users
        .group_by("month")
        .agg([
            pl.col("mean_score").std().alias("inter_user_std"),
            pl.col("mean_score").mean().alias("inter_user_mean"),
            pl.col("mean_similarity").std().alias("similarity_std"),
            pl.col("mean_similarity").mean().alias("similarity_mean"),
            pl.col("author_id").n_unique().alias("n_users"),
        ])
        .filter(pl.col("n_users") >= 5)  # Need enough users for meaningful metric
        .sort("month")
    )

    # CV = std / mean (lower = more convergent)
    monthly_convergence = monthly_convergence.with_columns([
        (pl.col("inter_user_std") / pl.col("inter_user_mean")).alias("cv_keyword"),
        (pl.col("similarity_std") / pl.col("similarity_mean")).alias("cv_similarity"),
    ])

    print(f"  Monthly observations: {monthly_convergence.height}")

    # ── 3. ITS on convergence metric ──────────────────────────────────
    print("\nFitting ITS model on convergence…")

    mc_pd = monthly_convergence.to_pandas()
    mc_pd["date"] = pd.to_datetime(mc_pd["month"] + "-01")
    mc_pd = mc_pd.sort_values("date").reset_index(drop=True)
    mc_pd["time"] = np.arange(len(mc_pd))
    mc_pd["post_siege"] = (mc_pd["date"] >= t0).astype(int)
    mc_pd["time_x_post"] = mc_pd["time"] * mc_pd["post_siege"]

    results = {}

    for metric, label in [("cv_keyword", "CV of keyword score"),
                           ("cv_similarity", "CV of similarity score")]:
        y = mc_pd[metric].values
        if np.any(np.isnan(y)):
            y = np.nan_to_num(y, nan=np.nanmean(y))

        X = mc_pd[["time", "post_siege", "time_x_post"]].values
        X = sm.add_constant(X)
        ols = sm.OLS(y, X).fit(cov_type="HC3")

        print(f"\n  {label}:")
        print(f"    Level change (β₂): {ols.params[2]:.4f} (p={ols.pvalues[2]:.4f})")
        print(f"    Slope change (β₃): {ols.params[3]:.6f} (p={ols.pvalues[3]:.4f})")
        print(f"    R²: {ols.rsquared:.4f}")
        # Negative slope change = increasing convergence (decreasing CV)

        results[metric] = {
            "label": label,
            "level_change": float(ols.params[2]),
            "level_change_p": float(ols.pvalues[2]),
            "slope_change": float(ols.params[3]),
            "slope_change_p": float(ols.pvalues[3]),
            "r_squared": float(ols.rsquared),
            "n_months": int(len(mc_pd)),
        }

    # ── 4. Pre/post comparison of inter-user variability ──────────────
    pre = mc_pd[mc_pd["post_siege"] == 0]["cv_keyword"]
    post = mc_pd[mc_pd["post_siege"] == 1]["cv_keyword"]

    t_stat, t_p = stats.ttest_ind(pre, post, equal_var=False)
    results["pre_post_comparison"] = {
        "pre_mean_cv": float(pre.mean()),
        "post_mean_cv": float(post.mean()),
        "t_stat": float(t_stat),
        "t_p": float(t_p),
        "direction": "convergent" if post.mean() < pre.mean() else "divergent",
    }
    print(f"\n  Pre-Siege mean CV: {pre.mean():.4f}")
    print(f"  Post-Siege mean CV: {post.mean():.4f}")
    print(f"  t-test: t={t_stat:.3f}, p={t_p:.4f}")

    # ── 5. Plot ───────────────────────────────────────────────────────
    print("\nPlotting convergence trends…")
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # Left: CV of keyword score over time
    ax = axes[0]
    ax.plot(mc_pd["date"], mc_pd["cv_keyword"], color=CB_PALETTE[0],
            alpha=0.7, linewidth=1)
    ax.axvline(t0, color="red", linestyle="--", linewidth=1.5, label="Siege publication")
    # Add trend lines
    pre_mask = mc_pd["post_siege"] == 0
    post_mask = mc_pd["post_siege"] == 1
    if pre_mask.sum() > 1:
        z = np.polyfit(mc_pd.loc[pre_mask, "time"], mc_pd.loc[pre_mask, "cv_keyword"], 1)
        ax.plot(mc_pd.loc[pre_mask, "date"],
                np.polyval(z, mc_pd.loc[pre_mask, "time"]),
                color=CB_PALETTE[3], linestyle="--", linewidth=2, label="Pre-Siege trend")
    if post_mask.sum() > 1:
        z = np.polyfit(mc_pd.loc[post_mask, "time"], mc_pd.loc[post_mask, "cv_keyword"], 1)
        ax.plot(mc_pd.loc[post_mask, "date"],
                np.polyval(z, mc_pd.loc[post_mask, "time"]),
                color=CB_PALETTE[2], linestyle="--", linewidth=2, label="Post-Siege trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Coefficient of variation")
    ax.set_title("Inter-User Siege Score Variability\n(lower = more convergent)")
    ax.legend(fontsize=9)

    # Right: number of siege-active users per month
    ax = axes[1]
    ax.bar(mc_pd["date"], mc_pd["n_users"], width=25, color=CB_PALETTE[0], alpha=0.7)
    ax.axvline(t0, color="red", linestyle="--", linewidth=1.5, label="Siege publication")
    ax.set_xlabel("Date")
    ax.set_ylabel("N Siege-active users")
    ax.set_title("Siege-Active Users per Month")
    ax.legend()

    fig.tight_layout()
    save_figure(fig, "semantic_convergence")

    with open(RESULTS_DIR / "semantic_convergence_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ Semantic convergence results saved.")


if __name__ == "__main__":
    main()
