"""21 – Semantic Convergence on /pol/ (H10-pol).

Adapts the H10 semantic convergence test for /pol/. Because /pol/ is
anonymous, "users" are identified by poster_hash (per-thread IDs
assigned by 4chan). This limits the analysis to within-thread identity
consistency, but tests whether /pol/ discussants converge on shared
Siege interpretations within individual threads.

Approach
--------
1. Group posts by month and poster_hash (excluding "anon" posts
   without a poster ID).
2. Compute inter-user std of siege scores per month.
3. Test via ITS whether variance decreases after Siege publication
   (using IM's T0 as treatment date).

Outputs
-------
results/pol_semantic_convergence_results.json
figures/pol_semantic_convergence.png / .pdf
"""

from __future__ import annotations

import datetime
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


def load_treatment_date() -> datetime.datetime:
    """Load treatment date T0."""
    with open(DATA_PROCESSED / "treatment_dates.json") as f:
        td = json.load(f)
    t0 = td["T0"]
    if isinstance(t0, str):
        t0 = datetime.datetime.fromisoformat(t0)
    return t0


def main() -> None:
    print("=" * 60)
    print("H10-pol: Semantic Convergence on /pol/")
    print("=" * 60)

    t0 = load_treatment_date()

    pol_path = DATA_PROCESSED / "pol_siege_scores.parquet"
    if not pol_path.exists():
        print(f"  ✗ {pol_path} not found.")
        return

    pol = pl.read_parquet(pol_path)

    # Filter: non-anon posters with content
    pol = pol.filter(
        pl.col("author_id").is_not_null()
        & (pl.col("author_id") != "anon")
        & pl.col("date").is_not_null()
        & (pl.col("word_count") > 0)
    )

    print(f"  /pol/ posts with poster ID: {pol.height:,}")
    print(f"  Unique poster hashes: {pol['author_id'].n_unique():,}")

    pol = pol.with_columns(
        pl.col("date").dt.strftime("%Y-%m").alias("month")
    )

    # ── 1. Per-user monthly siege profiles ────────────────────────────
    print("\nComputing monthly user profiles…")
    user_monthly = (
        pol.group_by(["author_id", "month"])
        .agg([
            pl.col("siege_keyword_score").mean().alias("mean_score"),
            pl.col("siege_keyword_score").std().alias("std_score"),
            pl.len().alias("n_posts"),
        ])
        .filter(pl.col("n_posts") >= 2)
    )

    # ── 2. Monthly convergence metrics ────────────────────────────────
    print("Computing monthly convergence…")
    monthly_convergence = (
        user_monthly.filter(pl.col("mean_score") > 0)
        .group_by("month")
        .agg([
            pl.col("mean_score").std().alias("inter_user_std"),
            pl.col("mean_score").mean().alias("inter_user_mean"),
            pl.col("author_id").n_unique().alias("n_users"),
        ])
        .filter(pl.col("n_users") >= 5)
        .sort("month")
    )

    monthly_convergence = monthly_convergence.with_columns(
        (pl.col("inter_user_std") / pl.col("inter_user_mean")).alias("cv_keyword"),
    )

    print(f"  Monthly observations: {monthly_convergence.height}")

    if monthly_convergence.height < 10:
        print("  ⚠ Too few monthly observations for ITS.")
        return

    # ── 3. ITS on convergence ─────────────────────────────────────────
    print("\nFitting ITS model…")
    mc_pd = monthly_convergence.to_pandas()
    mc_pd["date_dt"] = pd.to_datetime(mc_pd["month"] + "-01")
    mc_pd = mc_pd.sort_values("date_dt").reset_index(drop=True)
    mc_pd["time"] = np.arange(len(mc_pd))
    mc_pd["post_siege"] = (mc_pd["date_dt"] >= t0).astype(int)
    mc_pd["time_x_post"] = mc_pd["time"] * mc_pd["post_siege"]

    y = mc_pd["cv_keyword"].values
    if np.any(np.isnan(y)):
        y = np.nan_to_num(y, nan=np.nanmean(y))

    X = mc_pd[["time", "post_siege", "time_x_post"]].values
    X = sm.add_constant(X)
    ols = sm.OLS(y, X).fit(cov_type="HC3")

    print(f"  Level change (β₂): {ols.params[2]:.4f} (p={ols.pvalues[2]:.4f})")
    print(f"  Slope change (β₃): {ols.params[3]:.6f} (p={ols.pvalues[3]:.4f})")
    print(f"  R²: {ols.rsquared:.4f}")

    # ── 4. Pre/post comparison ────────────────────────────────────────
    pre = mc_pd[mc_pd["post_siege"] == 0]["cv_keyword"]
    post = mc_pd[mc_pd["post_siege"] == 1]["cv_keyword"]

    if len(pre) > 1 and len(post) > 1:
        t_stat, t_p = stats.ttest_ind(pre, post, equal_var=False)
    else:
        t_stat, t_p = float("nan"), float("nan")

    direction = "convergent" if post.mean() < pre.mean() else "divergent"
    print(f"\n  Pre-Siege mean CV: {pre.mean():.4f}")
    print(f"  Post-Siege mean CV: {post.mean():.4f}")
    print(f"  Direction: {direction}")

    # ── Results ───────────────────────────────────────────────────────
    results = {
        "platform": "pol",
        "n_months": int(len(mc_pd)),
        "its_model": {
            "level_change": float(ols.params[2]),
            "level_change_p": float(ols.pvalues[2]),
            "slope_change": float(ols.params[3]),
            "slope_change_p": float(ols.pvalues[3]),
            "r_squared": float(ols.rsquared),
        },
        "pre_post_comparison": {
            "pre_mean_cv": float(pre.mean()),
            "post_mean_cv": float(post.mean()),
            "t_stat": float(t_stat),
            "t_p": float(t_p),
            "direction": direction,
        },
    }

    # ── Plot ──────────────────────────────────────────────────────────
    print("\nPlotting…")
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    ax = axes[0]
    ax.plot(mc_pd["date_dt"], mc_pd["cv_keyword"], color=CB_PALETTE[2],
            alpha=0.7, linewidth=1)
    ax.axvline(t0, color="red", linestyle="--", linewidth=1.5,
               label="Siege publication")

    pre_mask = mc_pd["post_siege"] == 0
    post_mask = mc_pd["post_siege"] == 1
    if pre_mask.sum() > 1:
        z = np.polyfit(mc_pd.loc[pre_mask, "time"], mc_pd.loc[pre_mask, "cv_keyword"], 1)
        ax.plot(mc_pd.loc[pre_mask, "date_dt"],
                np.polyval(z, mc_pd.loc[pre_mask, "time"]),
                color=CB_PALETTE[3], linestyle="--", linewidth=2,
                label="Pre-Siege trend")
    if post_mask.sum() > 1:
        z = np.polyfit(mc_pd.loc[post_mask, "time"], mc_pd.loc[post_mask, "cv_keyword"], 1)
        ax.plot(mc_pd.loc[post_mask, "date_dt"],
                np.polyval(z, mc_pd.loc[post_mask, "time"]),
                color=CB_PALETTE[4], linestyle="--", linewidth=2,
                label="Post-Siege trend")

    ax.set_xlabel("Date")
    ax.set_ylabel("Coefficient of Variation")
    ax.set_title("/pol/ Inter-User Siege Score Variability")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.bar(mc_pd["date_dt"], mc_pd["n_users"], width=25,
           color=CB_PALETTE[2], alpha=0.7)
    ax.axvline(t0, color="red", linestyle="--", linewidth=1.5,
               label="Siege publication")
    ax.set_xlabel("Date")
    ax.set_ylabel("N Siege-active posters")
    ax.set_title("/pol/ Siege-Active Posters per Month")
    ax.legend()

    fig.tight_layout()
    save_figure(fig, "pol_semantic_convergence")

    # ── Save ──────────────────────────────────────────────────────────
    with open(RESULTS_DIR / "pol_semantic_convergence_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ /pol/ semantic convergence results saved.")


if __name__ == "__main__":
    main()
