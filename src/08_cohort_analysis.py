"""08 – Cohort-Stratified Adoption Analysis (H4).

Tests whether pre-Siege members show a conversion effect vs. self-selected
newcomers driving adoption.
"""

from __future__ import annotations

import json
import datetime

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
    print("PHASE 5: Cohort-Stratified Adoption (H4)")
    print("=" * 60)

    with open(DATA_PROCESSED / "treatment_dates.json") as f:
        td = json.load(f)
    t0 = datetime.datetime.fromisoformat(td["T0"])
    print(f"  T0 = {t0}")

    scores = pl.read_parquet(DATA_PROCESSED / "siege_scores.parquet")
    members = pl.read_parquet(DATA_PROCESSED / "members.parquet")

    # Exclude Zeiger
    scores = scores.filter(pl.col("author_id") != ZEIGER_MEMBER_ID)

    # Merge join dates
    member_join = members.select(["member_id", "joined"]).rename(
        {"member_id": "author_id"}
    )
    scores_m = scores.join(member_join, on="author_id", how="left")

    # Split into cohorts
    pre_siege_users = scores_m.filter(
        pl.col("joined").is_not_null() & (pl.col("joined") < t0)
    )["author_id"].unique()
    post_siege_users = scores_m.filter(
        pl.col("joined").is_not_null() & (pl.col("joined") >= t0)
    )["author_id"].unique()
    print(f"  Pre-Siege joiners: {pre_siege_users.len()}")
    print(f"  Post-Siege joiners: {post_siege_users.len()}")

    results = {}

    # ── Within-user DiD for pre-Siege joiners ─────────────────────────
    print("\nPre-Siege joiners: within-user difference-in-differences…")
    pre_users_list = pre_siege_users.to_list()
    pre_scores = scores_m.filter(pl.col("author_id").is_in(pre_users_list))

    pre_before = (
        pre_scores.filter(pl.col("date") < t0)
        .group_by("author_id")
        .agg(pl.col("siege_keyword_score").mean().alias("score_before"))
    )
    pre_after = (
        pre_scores.filter(pl.col("date") >= t0)
        .group_by("author_id")
        .agg(pl.col("siege_keyword_score").mean().alias("score_after"))
    )

    did = pre_before.join(pre_after, on="author_id", how="inner")
    did = did.with_columns(
        (pl.col("score_after") - pl.col("score_before")).alias("diff")
    )

    if did.height > 0:
        diffs = did["diff"].to_numpy()
        t_stat, p_val = stats.ttest_1samp(diffs, 0.0)
        try:
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(diffs, alternative="two-sided")
        except ValueError:
            wilcoxon_stat, wilcoxon_p = float("nan"), float("nan")

        results["pre_siege_did"] = {
            "n_users": int(did.height),
            "mean_diff": float(np.mean(diffs)),
            "median_diff": float(np.median(diffs)),
            "t_stat": float(t_stat),
            "t_pvalue": float(p_val),
            "wilcoxon_stat": float(wilcoxon_stat),
            "wilcoxon_pvalue": float(wilcoxon_p),
        }
        print(f"  Users with both pre/post data: {did.height}")
        print(f"  Mean score change: {np.mean(diffs):.4f}")
        print(f"  t-test: t={t_stat:.3f}, p={p_val:.4f}")
        print(f"  Wilcoxon: W={wilcoxon_stat:.1f}, p={wilcoxon_p:.4f}")
    else:
        results["pre_siege_did"] = {"error": "no users with both pre and post data"}

    # ── Entry siege scores: pre vs post joiners ───────────────────────
    print("\nEntry-level siege scores comparison…")

    def first_month_scores(user_list, label):
        user_scores = scores_m.filter(pl.col("author_id").is_in(user_list))
        first_posts = (
            user_scores
            .sort("date")
            .group_by("author_id")
            .head(10)  # first 10 posts per user
        )
        return first_posts["siege_keyword_score"].to_numpy()

    pre_entry = first_month_scores(pre_users_list, "pre")
    post_entry = first_month_scores(post_siege_users.to_list(), "post")

    if len(pre_entry) > 0 and len(post_entry) > 0:
        mw_stat, mw_p = stats.mannwhitneyu(pre_entry, post_entry, alternative="two-sided")
        results["entry_comparison"] = {
            "pre_n": int(len(pre_entry)),
            "post_n": int(len(post_entry)),
            "pre_mean": float(np.mean(pre_entry)),
            "post_mean": float(np.mean(post_entry)),
            "mann_whitney_u": float(mw_stat),
            "mann_whitney_p": float(mw_p),
        }
        print(f"  Pre-joiners first posts mean: {np.mean(pre_entry):.4f}")
        print(f"  Post-joiners first posts mean: {np.mean(post_entry):.4f}")
        print(f"  Mann-Whitney U: {mw_stat:.1f}, p={mw_p:.4f}")

    # ── Plot cohort adoption curves ───────────────────────────────────
    print("\nPlotting cohort adoption curves…")
    setup_plot_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    for cohort_name, user_list, color in [
        ("Pre-Siege joiners", pre_users_list, CB_PALETTE[0]),
        ("Post-Siege joiners", post_siege_users.to_list(), CB_PALETTE[2]),
    ]:
        cohort_scores = (
            scores_m.filter(
                pl.col("author_id").is_in(user_list)
                & pl.col("date").is_not_null()
            )
            .with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
            .group_by("month")
            .agg(pl.col("siege_keyword_score").mean().alias("mean_score"))
            .sort("month")
        )
        if cohort_scores.height > 0:
            cpdf = cohort_scores.to_pandas()
            ax.plot(cpdf["month"], cpdf["mean_score"], color=color,
                    linewidth=1.5, label=cohort_name)

    ax.axvline(t0, color="red", linestyle="--", linewidth=2, alpha=0.7,
               label="Siege publication")
    ax.set_xlabel("Date")
    ax.set_ylabel("Mean siege keyword score")
    ax.set_title("Siege Rhetoric Adoption by Cohort")
    ax.legend()
    fig.autofmt_xdate()
    save_figure(fig, "cohort_adoption")

    # Save
    with open(RESULTS_DIR / "cohort_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ Cohort analysis results saved.")


if __name__ == "__main__":
    main()
