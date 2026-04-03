"""04 – Interrupted Time Series Analysis (H1).

Tests whether Siege Culture rhetoric increased after Zeiger published Siege,
using both dictionary and embedding measures.
"""

from __future__ import annotations

import json
import datetime

import numpy as np
import polars as pl
import statsmodels.api as sm
from scipy import stats
import ruptures as rpt

from utils import (
    DATA_PROCESSED, RESULTS_DIR,
    ZEIGER_MEMBER_ID, setup_plot_style, save_figure, FIGSIZE_WIDE, CB_PALETTE,
)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def load_treatment_date() -> datetime.datetime:
    with open(DATA_PROCESSED / "treatment_dates.json") as f:
        td = json.load(f)
    t0 = td["T0"]
    if isinstance(t0, str):
        t0 = datetime.datetime.fromisoformat(t0)
    return t0


def build_weekly_series(
    df: pl.DataFrame,
    t0: datetime.datetime,
    measure_col: str,
    exclude_zeiger: bool = True,
) -> pl.DataFrame:
    """Aggregate siege measure to weekly time series."""
    if exclude_zeiger:
        df = df.filter(pl.col("author_id") != ZEIGER_MEMBER_ID)

    df = df.filter(pl.col("date").is_not_null())

    weekly = (
        df.with_columns(
            pl.col("date").dt.truncate("1w").alias("week")
        )
        .group_by("week")
        .agg([
            pl.col(measure_col).mean().alias("mean_score"),
            pl.col(measure_col).sum().alias("total_score"),
            pl.len().alias("post_count"),
            pl.col("author_id").n_unique().alias("unique_users"),
            pl.col("siege_binary").sum().alias("siege_post_count"),
        ])
        .sort("week")
    )

    # Add ITS variables
    t0_aware = t0
    weekly = weekly.with_columns([
        (pl.col("week") >= t0_aware).cast(pl.Int8).alias("post_treatment"),
        pl.col("week").diff().dt.total_days().fill_null(7).cum_sum().alias("time_index"),
    ])
    # Centre time on T0
    t0_idx = weekly.filter(pl.col("post_treatment") == 1)
    if t0_idx.height > 0:
        shift = t0_idx["time_index"][0]
        weekly = weekly.with_columns(
            (pl.col("time_index") - shift).alias("time_centered")
        )
    else:
        weekly = weekly.with_columns(
            pl.col("time_index").alias("time_centered")
        )

    weekly = weekly.with_columns(
        (pl.col("time_centered") * pl.col("post_treatment")).alias("time_x_post")
    )

    return weekly


def run_its_regression(weekly: pl.DataFrame, label: str) -> dict:
    """Run ITS regression with Newey-West standard errors."""
    pdf = weekly.to_pandas().dropna(subset=["mean_score", "time_centered", "post_treatment", "time_x_post"])

    if len(pdf) < 10:
        print(f"  ⚠ {label}: insufficient data ({len(pdf)} weeks)")
        return {"label": label, "error": "insufficient data"}

    y = pdf["mean_score"].values
    X = pdf[["time_centered", "post_treatment", "time_x_post"]].values
    X = sm.add_constant(X)

    try:
        model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    except Exception as e:
        print(f"  ⚠ {label}: regression failed: {e}")
        return {"label": label, "error": str(e)}

    result = {
        "label": label,
        "n_weeks": int(len(pdf)),
        "intercept": float(model.params[0]),
        "b_time": float(model.params[1]),
        "b_post_treatment": float(model.params[2]),
        "b_time_x_post": float(model.params[3]),
        "p_time": float(model.pvalues[1]),
        "p_post_treatment": float(model.pvalues[2]),
        "p_time_x_post": float(model.pvalues[3]),
        "r_squared": float(model.rsquared),
        "se_post_treatment": float(model.bse[2]),
        "se_time_x_post": float(model.bse[3]),
    }

    print(f"\n  {label}:")
    print(f"    β₂ (level change)    = {result['b_post_treatment']:.4f} "
          f"(p={result['p_post_treatment']:.4f})")
    print(f"    β₃ (slope change)    = {result['b_time_x_post']:.4f} "
          f"(p={result['p_time_x_post']:.4f})")
    print(f"    R² = {result['r_squared']:.4f}")

    return result


def run_changepoint_detection(weekly: pl.DataFrame, measure_col: str, label: str) -> dict:
    """Run Bayesian change-point detection using PELT algorithm."""
    values = weekly[measure_col].drop_nulls().to_numpy()
    if len(values) < 20:
        return {"label": label, "error": "insufficient data"}

    try:
        algo = rpt.Pelt(model="rbf", min_size=4).fit(values)
        result = algo.predict(pen=1.0)
        weeks = weekly["week"].to_list()
        changepoints = []
        for idx in result[:-1]:  # last is always len(values)
            if idx < len(weeks):
                cp_date = weeks[idx]
                changepoints.append(str(cp_date))
        return {"label": label, "changepoints": changepoints}
    except Exception as e:
        return {"label": label, "error": str(e)}


def plot_its(
    weekly: pl.DataFrame,
    t0: datetime.datetime,
    measure_col: str,
    label: str,
    filename: str,
):
    """Plot ITS time series with regression lines."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    pdf = weekly.to_pandas().dropna(subset=[measure_col])
    ax.plot(pdf["week"], pdf[measure_col], color=CB_PALETTE[0], alpha=0.6, linewidth=1)

    # Pre/post regression lines
    for phase, color, ls in [("pre", CB_PALETTE[1], "--"), ("post", CB_PALETTE[2], "--")]:
        if phase == "pre":
            mask = pdf["week"] < t0
        else:
            mask = pdf["week"] >= t0
        subset = pdf[mask]
        if len(subset) >= 2:
            x_num = (subset["week"] - subset["week"].min()).dt.days.values.astype(float)
            coeffs = np.polyfit(x_num, subset[measure_col].values, 1)
            ax.plot(subset["week"], np.polyval(coeffs, x_num),
                    color=color, linestyle=ls, linewidth=2,
                    label=f"{phase}-Siege trend")

    ax.axvline(t0, color="red", linestyle="--", linewidth=2, alpha=0.7, label="Siege publication")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Weekly mean {label}")
    ax.set_title(f"Interrupted Time Series: {label}")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    save_figure(fig, filename)


def plot_cumulative_adoption(
    scores: pl.DataFrame,
    t0: datetime.datetime,
    filename: str,
):
    """Plot cumulative share of users who used ≥1 siege term over time."""
    setup_plot_style()

    siege_users = (
        scores.filter(
            (pl.col("siege_binary") == 1)
            & (pl.col("author_id") != ZEIGER_MEMBER_ID)
            & pl.col("date").is_not_null()
        )
        .group_by("author_id")
        .agg(pl.col("date").min().alias("first_siege_date"))
        .sort("first_siege_date")
    )

    if siege_users.height == 0:
        print("  ⚠ No siege adopters found for cumulative plot")
        return

    total_users = scores.filter(pl.col("author_id") != ZEIGER_MEMBER_ID)["author_id"].n_unique()

    pdf = siege_users.to_pandas()
    pdf["cumulative"] = range(1, len(pdf) + 1)
    pdf["cumulative_pct"] = pdf["cumulative"] / total_users * 100

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(pdf["first_siege_date"], pdf["cumulative_pct"], color=CB_PALETTE[0], linewidth=2)
    ax.axvline(t0, color="red", linestyle="--", linewidth=2, alpha=0.7, label="Siege publication")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative % of users who used siege rhetoric")
    ax.set_title("Cumulative Siege Rhetoric Adoption")
    ax.legend()
    fig.autofmt_xdate()
    save_figure(fig, filename)


def plot_zeiger_vs_community(
    scores: pl.DataFrame,
    t0: datetime.datetime,
    filename: str,
):
    """Overlay Zeiger's posting intensity on community siege score."""
    setup_plot_style()

    zeiger = (
        scores.filter(
            (pl.col("author_id") == ZEIGER_MEMBER_ID)
            & pl.col("date").is_not_null()
        )
        .with_columns(pl.col("date").dt.truncate("1w").alias("week"))
        .group_by("week")
        .agg(pl.col("siege_keyword_score").mean().alias("zeiger_score"))
        .sort("week")
    )

    community = (
        scores.filter(
            (pl.col("author_id") != ZEIGER_MEMBER_ID)
            & pl.col("date").is_not_null()
        )
        .with_columns(pl.col("date").dt.truncate("1w").alias("week"))
        .group_by("week")
        .agg(pl.col("siege_keyword_score").mean().alias("community_score"))
        .sort("week")
    )

    fig, ax1 = plt.subplots(figsize=FIGSIZE_WIDE)
    zpdf = zeiger.to_pandas()
    cpdf = community.to_pandas()

    ax1.plot(cpdf["week"], cpdf["community_score"], color=CB_PALETTE[0],
             linewidth=1.5, label="Community siege score")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Community mean siege score", color=CB_PALETTE[0])

    ax2 = ax1.twinx()
    ax2.plot(zpdf["week"], zpdf["zeiger_score"], color=CB_PALETTE[2],
             linewidth=1.5, alpha=0.7, label="Zeiger siege score")
    ax2.set_ylabel("Zeiger mean siege score", color=CB_PALETTE[2])

    ax1.axvline(t0, color="red", linestyle="--", linewidth=2, alpha=0.7, label="Siege publication")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.set_title("Zeiger vs. Community Siege Rhetoric Over Time")
    fig.autofmt_xdate()
    save_figure(fig, filename)


def main():
    print("=" * 60)
    print("PHASE 2: Interrupted Time Series Analysis (H1)")
    print("=" * 60)

    t0 = load_treatment_date()
    print(f"  Treatment date T0 = {t0}")

    scores = pl.read_parquet(DATA_PROCESSED / "siege_scores.parquet")
    forum = scores.filter(pl.col("channel") == "forum")
    dm = scores.filter(pl.col("channel") == "dm")

    all_results = {}

    # Run ITS for each combination of channel × measure
    for channel_name, channel_df in [("forum", forum), ("dm", dm), ("all", scores)]:
        for measure, measure_label in [
            ("siege_keyword_score", "keyword_score"),
            ("siege_similarity", "similarity"),
        ]:
            if measure not in channel_df.columns:
                continue
            label = f"{channel_name}_{measure_label}"
            print(f"\n── {label} ──")

            weekly = build_weekly_series(channel_df, t0, measure)
            result = run_its_regression(weekly, label)
            all_results[label] = result

            # Change-point detection
            cp = run_changepoint_detection(weekly, "mean_score", label)
            all_results[f"{label}_changepoints"] = cp
            if "changepoints" in cp:
                print(f"  Change points: {cp['changepoints']}")

            # Plot
            plot_its(weekly, t0, "mean_score", measure_label, f"its_{label}")

    # ── Cumulative adoption ───────────────────────────────────────────
    print("\nPlotting cumulative adoption…")
    plot_cumulative_adoption(scores, t0, "cumulative_adoption")

    # ── Zeiger vs community ───────────────────────────────────────────
    print("\nPlotting Zeiger vs. community…")
    plot_zeiger_vs_community(scores, t0, "zeiger_vs_community")

    # ── Save results ──────────────────────────────────────────────────
    with open(RESULTS_DIR / "its_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✓ ITS results saved to results/its_results.json")


if __name__ == "__main__":
    main()
