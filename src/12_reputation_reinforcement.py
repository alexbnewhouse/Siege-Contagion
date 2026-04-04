"""12 – Reputation Reinforcement of Siege Rhetoric (H7).

Tests whether forum posts containing Siege rhetoric receive more reputation
points (likes/reactions) than non-Siege posts, controlling for post length,
author effects, and time period.  This operationalises the 'canonisation'
mechanism: the community doesn't just absorb Siege, it *rewards* Siege-
aligned speech.

Outputs
-------
results/reputation_reinforcement_results.json
figures/reputation_reinforcement.png / .pdf
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


def build_post_reputation_df() -> pl.DataFrame:
    """Join forum posts with reputation counts."""
    fp = pl.read_parquet(DATA_PROCESSED / "forum_posts.parquet")
    rep = pl.read_parquet(DATA_PROCESSED / "core_reputation_index.parquet")

    # Keep only forum-post reputation events
    forum_rep = rep.filter(pl.col("type") == "pid")
    rep_counts = (
        forum_rep.group_by("type_id")
        .agg(pl.len().alias("rep_count"))
        .rename({"type_id": "pid"})
        .with_columns(pl.col("pid").cast(pl.Int64))
    )

    # Join to posts
    df = fp.join(rep_counts, on="pid", how="left")
    df = df.with_columns(pl.col("rep_count").fill_null(0))
    return df


def main() -> None:
    print("=" * 60)
    print("H7: Reputation Reinforcement of Siege Rhetoric")
    print("=" * 60)

    df = build_post_reputation_df()

    # Exclude Zeiger
    df = df.filter(pl.col("author_id") != ZEIGER_MEMBER_ID)
    # Require non-empty posts
    df = df.filter(pl.col("word_count") > 0)

    # Add month variable for time FE
    df = df.with_columns(
        pl.col("post_date").dt.strftime("%Y-%m").alias("month")
    )

    print(f"  Posts: {df.height:,}  |  With reps: {df.filter(pl.col('rep_count')>0).height:,}")
    print(f"  Mean rep count: {df['rep_count'].mean():.2f}  |  Max: {df['rep_count'].max()}")

    # ── Negative Binomial regression ──────────────────────────────────
    # Model: rep_count ~ siege_binary + log(word_count) + month FE
    print("\nFitting negative binomial model…")

    pdf = df.select([
        "rep_count", "siege_binary", "word_count", "siege_keyword_score", "month"
    ]).to_pandas()

    pdf["log_word_count"] = np.log1p(pdf["word_count"])

    # Simple model without FE first (for robustness)
    X_simple = pdf[["siege_binary", "log_word_count"]].copy()
    X_simple = sm.add_constant(X_simple)
    y = pdf["rep_count"]

    try:
        nb_simple = sm.GLM(y, X_simple, family=sm.families.NegativeBinomial()).fit()
        print("\n  Simple model (no FE):")
        print(f"    siege_binary coef: {nb_simple.params['siege_binary']:.4f} "
              f"(p={nb_simple.pvalues['siege_binary']:.4f})")
        print(f"    IRR: {np.exp(nb_simple.params['siege_binary']):.4f}")
    except Exception as e:
        print(f"  ⚠ Simple NB failed: {e}")
        nb_simple = None

    # Full model with month FE (dummies)
    print("\n  Fitting model with month fixed effects…")
    month_dummies = pd.get_dummies(pdf["month"], prefix="m", drop_first=True, dtype=float)
    X_full = pd.concat([
        pdf[["siege_binary", "log_word_count"]],
        month_dummies,
    ], axis=1)
    X_full = sm.add_constant(X_full)

    try:
        nb_full = sm.GLM(y, X_full, family=sm.families.NegativeBinomial()).fit()
        print(f"    siege_binary coef: {nb_full.params['siege_binary']:.4f} "
              f"(p={nb_full.pvalues['siege_binary']:.4f})")
        print(f"    IRR: {np.exp(nb_full.params['siege_binary']):.4f}")
    except Exception as e:
        print(f"  ⚠ Full NB failed: {e}")
        nb_full = None

    # ── Non-parametric comparison ─────────────────────────────────────
    siege_reps = df.filter(pl.col("siege_binary") == 1)["rep_count"].to_numpy()
    non_siege_reps = df.filter(pl.col("siege_binary") == 0)["rep_count"].to_numpy()
    mwu_stat, mwu_p = stats.mannwhitneyu(siege_reps, non_siege_reps, alternative="two-sided")

    print(f"\n  Mann-Whitney U: stat={mwu_stat:.0f}, p={mwu_p:.4f}")
    print(f"  Siege posts mean rep: {siege_reps.mean():.3f}  |  Non-siege: {non_siege_reps.mean():.3f}")

    # ── Results dict ──────────────────────────────────────────────────
    results = {
        "n_posts": int(df.height),
        "n_siege_posts": int((df["siege_binary"] == 1).sum()),
        "mean_rep_siege": float(siege_reps.mean()),
        "mean_rep_non_siege": float(non_siege_reps.mean()),
        "mann_whitney_U": float(mwu_stat),
        "mann_whitney_p": float(mwu_p),
    }

    if nb_simple is not None:
        results["simple_model"] = {
            "siege_coef": float(nb_simple.params["siege_binary"]),
            "siege_irr": float(np.exp(nb_simple.params["siege_binary"])),
            "siege_p": float(nb_simple.pvalues["siege_binary"]),
            "siege_se": float(nb_simple.bse["siege_binary"]),
        }

    if nb_full is not None:
        results["month_fe_model"] = {
            "siege_coef": float(nb_full.params["siege_binary"]),
            "siege_irr": float(np.exp(nb_full.params["siege_binary"])),
            "siege_p": float(nb_full.pvalues["siege_binary"]),
            "siege_se": float(nb_full.bse["siege_binary"]),
        }

    # ── Plot: Distribution of reputation by siege status ──────────────
    print("\nPlotting reputation distributions…")
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # Left: box plot
    ax = axes[0]
    siege_data = [non_siege_reps, siege_reps]
    bp = ax.boxplot(siege_data, labels=["Non-Siege", "Siege"], widths=0.5,
                    showfliers=False, patch_artist=True)
    bp["boxes"][0].set_facecolor(CB_PALETTE[0])
    bp["boxes"][1].set_facecolor(CB_PALETTE[1])
    ax.set_ylabel("Reputation points received")
    ax.set_title("Reputation by Siege Status")

    # Right: mean over time
    ax = axes[1]
    monthly = (
        df.group_by(["month", "siege_binary"])
        .agg(pl.col("rep_count").mean().alias("mean_rep"))
        .sort("month")
    )
    for label, val, color in [("Siege posts", 1, CB_PALETTE[1]),
                               ("Non-Siege posts", 0, CB_PALETTE[0])]:
        sub = monthly.filter(pl.col("siege_binary") == val).to_pandas()
        ax.plot(range(len(sub)), sub["mean_rep"], color=color, label=label, alpha=0.8)
    ax.set_xlabel("Month (index)")
    ax.set_ylabel("Mean reputation per post")
    ax.set_title("Mean Reputation Over Time")
    ax.legend()

    fig.tight_layout()
    save_figure(fig, "reputation_reinforcement")

    # Save results
    with open(RESULTS_DIR / "reputation_reinforcement_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ Reputation reinforcement results saved.")


if __name__ == "__main__":
    main()
