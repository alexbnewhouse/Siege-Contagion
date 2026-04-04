"""17 – Cross-Platform Interrupted Time Series (H12).

Tests whether /pol/ shows a structural break in Siege rhetoric
following the same T0 (Zeiger's June 2015 publication on Iron March).
This is the most direct test of cross-platform contagion: if the
treatment event was IM-internal, /pol/ should not show a break at T0
unless rhetoric propagated across platforms.

Additional tests
----------------
- Independent changepoint detection on /pol/ series (does /pol/ have
  its *own* break dates?)
- Comparison of effect sizes between platforms.
"""

from __future__ import annotations

import datetime
import json

import numpy as np
import polars as pl
import statsmodels.api as sm
import ruptures as rpt

from utils import (
    DATA_PROCESSED, RESULTS_DIR, FIGURES_DIR,
    setup_plot_style, save_figure, FIGSIZE_WIDE, CB_PALETTE,
)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def load_treatment_date() -> datetime.datetime:
    """Load the Iron March treatment date T0."""
    with open(DATA_PROCESSED / "treatment_dates.json") as f:
        td = json.load(f)
    t0 = td["T0"]
    if isinstance(t0, str):
        t0 = datetime.datetime.fromisoformat(t0)
    return t0


def build_weekly_series_pol(
    df: pl.DataFrame,
    t0: datetime.datetime,
    measure_col: str,
) -> pl.DataFrame:
    """Aggregate /pol/ siege measure to weekly time series with ITS variables."""
    df = df.filter(pl.col("date").is_not_null())

    weekly = (
        df.with_columns(pl.col("date").dt.truncate("1w").alias("week"))
        .group_by("week")
        .agg([
            pl.col(measure_col).mean().alias("mean_score"),
            pl.col(measure_col).sum().alias("total_score"),
            pl.len().alias("post_count"),
            pl.col("siege_binary").sum().alias("siege_post_count"),
        ])
        .sort("week")
    )

    # ITS variables
    weekly = weekly.with_columns([
        (pl.col("week") >= t0).cast(pl.Int8).alias("post_treatment"),
        pl.col("week").diff().dt.total_days().fill_null(7).cum_sum().alias("time_index"),
    ])

    t0_idx = weekly.filter(pl.col("post_treatment") == 1)
    if t0_idx.height > 0:
        shift = t0_idx["time_index"][0]
        weekly = weekly.with_columns(
            (pl.col("time_index") - shift).alias("time_centered")
        )
    else:
        weekly = weekly.with_columns(pl.col("time_index").alias("time_centered"))

    weekly = weekly.with_columns(
        (pl.col("time_centered") * pl.col("post_treatment")).alias("time_x_post")
    )

    return weekly


def build_weekly_prevalence(
    siege_weekly: pl.DataFrame,
    all_weekly: pl.DataFrame,
) -> pl.DataFrame:
    """Merge Siege post counts with total /pol/ post counts to compute prevalence.

    This corrects for the natural growth of /pol/ over time — testing
    whether the *share* of Siege rhetoric increases, not just the count.
    """
    prevalence = siege_weekly.select([
        "week", "siege_post_count", "post_count", "mean_score",
        "post_treatment", "time_centered", "time_x_post",
    ]).join(
        all_weekly.select(["week_start", "total_posts"]).rename({"week_start": "week"}),
        on="week",
        how="left",
    )

    # Prevalence = siege posts / total posts on /pol/ that week
    prevalence = prevalence.with_columns(
        (pl.col("siege_post_count") / pl.col("total_posts").cast(pl.Float64).clip(1, None))
        .alias("siege_prevalence")
    )

    return prevalence


def run_its_regression(weekly: pl.DataFrame, label: str, y_col: str = "mean_score") -> dict:
    """Run ITS regression with Newey-West HAC standard errors."""
    pdf = weekly.to_pandas().dropna(
        subset=[y_col, "time_centered", "post_treatment", "time_x_post"]
    )

    if len(pdf) < 10:
        print(f"  ⚠ {label}: insufficient data ({len(pdf)} weeks)")
        return {"label": label, "error": "insufficient data"}

    y = pdf[y_col].values
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
    print(f"    β₂ (level change)    = {result['b_post_treatment']:.6f} "
          f"(p={result['p_post_treatment']:.4f})")
    print(f"    β₃ (slope change)    = {result['b_time_x_post']:.6f} "
          f"(p={result['p_time_x_post']:.4f})")
    print(f"    R² = {result['r_squared']:.4f}")

    return result


def run_changepoint_detection(weekly: pl.DataFrame, measure_col: str, label: str) -> dict:
    """Run PELT changepoint detection on /pol/ series."""
    values = weekly[measure_col].drop_nulls().to_numpy()
    if len(values) < 20:
        return {"label": label, "error": "insufficient data"}

    try:
        algo = rpt.Pelt(model="rbf", min_size=4).fit(values)
        result = algo.predict(pen=1.0)
        weeks = weekly["week"].to_list()
        changepoints = []
        for idx in result[:-1]:
            if idx < len(weeks):
                changepoints.append(str(weeks[idx]))
        return {"label": label, "changepoints": changepoints}
    except Exception as e:
        return {"label": label, "error": str(e)}


def plot_cross_platform_its(
    im_weekly: pl.DataFrame,
    pol_weekly: pl.DataFrame,
    t0: datetime.datetime,
    filename: str,
):
    """Plot both platforms' ITS on a single figure for visual comparison."""
    setup_plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    for ax, weekly, title, color in [
        (axes[0], im_weekly, "Iron March", CB_PALETTE[0]),
        (axes[1], pol_weekly, "/pol/", CB_PALETTE[2]),
    ]:
        pdf = weekly.to_pandas().dropna(subset=["mean_score"])
        ax.plot(pdf["week"], pdf["mean_score"], color=color, alpha=0.6, linewidth=1)

        # Trend lines
        for phase, lc, ls in [("pre", CB_PALETTE[1], "--"), ("post", CB_PALETTE[3], "--")]:
            mask = pdf["week"] < t0 if phase == "pre" else pdf["week"] >= t0
            subset = pdf[mask]
            if len(subset) >= 2:
                x_num = (subset["week"] - subset["week"].min()).dt.days.values.astype(float)
                coeffs = np.polyfit(x_num, subset["mean_score"].values, 1)
                ax.plot(subset["week"], np.polyval(coeffs, x_num),
                        color=lc, linestyle=ls, linewidth=2, label=f"{phase}-Siege trend")

        ax.axvline(t0, color="red", linestyle="--", linewidth=2, alpha=0.7,
                   label="Siege publication (IM)")
        ax.set_ylabel("Mean Siege Score")
        ax.set_title(f"{title}: Weekly Siege Rhetoric")
        ax.legend(loc="upper left")

    axes[1].set_xlabel("Date")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()

    save_figure(fig, filename)


def main():
    """Run cross-platform ITS analysis."""
    print("=" * 60)
    print("PHASE 8: Cross-Platform ITS (H12)")
    print("=" * 60)

    t0 = load_treatment_date()
    print(f"  Treatment date T0 = {t0}")

    results = {"treatment_date": str(t0)}

    # ── Load Iron March scores ────────────────────────────────────────
    im_scores_path = DATA_PROCESSED / "siege_scores.parquet"
    if im_scores_path.exists():
        im_scores = pl.read_parquet(im_scores_path)
        im_scores = im_scores.filter(pl.col("channel") == "forum")
        print(f"  Iron March forum posts: {im_scores.height:,}")
    else:
        print("  ⚠ Iron March scores not found; skipping IM series")
        im_scores = None

    # ── Load /pol/ scores ─────────────────────────────────────────────
    pol_scores_path = DATA_PROCESSED / "pol_siege_scores.parquet"
    if not pol_scores_path.exists():
        print(f"  ✗ {pol_scores_path} not found. Run lexicon + embedding on /pol/ first.")
        return

    pol = pl.read_parquet(pol_scores_path)
    print(f"  /pol/ scored posts: {pol.height:,}")

    # ── Build weekly series for /pol/ ─────────────────────────────────
    print("\n  Building /pol/ weekly series…")
    pol_weekly = build_weekly_series_pol(pol, t0, "siege_keyword_score")
    print(f"  /pol/ weeks: {pol_weekly.height}")

    # ── ITS on /pol/ with IM treatment date ───────────────────────────
    print("\n  ITS: /pol/ keyword score (IM T0 as treatment)")
    results["pol_keyword_its"] = run_its_regression(pol_weekly, "pol_keyword_score")

    # Similarity-based ITS if available
    if "siege_similarity" in pol.columns:
        pol_weekly_sim = build_weekly_series_pol(pol, t0, "siege_similarity")
        print("\n  ITS: /pol/ embedding similarity (IM T0 as treatment)")
        results["pol_similarity_its"] = run_its_regression(
            pol_weekly_sim, "pol_similarity", y_col="mean_score"
        )

    # ── Prevalence-based ITS ──────────────────────────────────────────
    wt_path = DATA_PROCESSED / "pol_weekly_totals.parquet"
    if wt_path.exists():
        all_weekly = pl.read_parquet(wt_path)
        prevalence = build_weekly_prevalence(pol_weekly, all_weekly)
        print("\n  ITS: /pol/ Siege prevalence (siege_posts / total_posts)")
        results["pol_prevalence_its"] = run_its_regression(
            prevalence, "pol_prevalence", y_col="siege_prevalence"
        )

    # ── Changepoint detection on /pol/ ────────────────────────────────
    print("\n  Changepoint detection on /pol/ series…")
    results["pol_changepoints"] = run_changepoint_detection(
        pol_weekly, "mean_score", "pol_keyword_changepoints"
    )
    if "changepoints" in results["pol_changepoints"]:
        for cp in results["pol_changepoints"]["changepoints"]:
            print(f"    Changepoint: {cp}")

    # ── Cross-platform comparison plot ────────────────────────────────
    if im_scores is not None:
        print("\n  Generating cross-platform ITS plot…")
        # Build IM weekly series
        from importlib import import_module
        its_mod = import_module("04_its_analysis")
        im_weekly = its_mod.build_weekly_series(im_scores, t0, "siege_keyword_score")
        plot_cross_platform_its(im_weekly, pol_weekly, t0, "cross_platform_its")
        results["im_weeks"] = im_weekly.height
        results["pol_weeks"] = pol_weekly.height

    # ── Save results ──────────────────────────────────────────────────
    with open(RESULTS_DIR / "cross_platform_its_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✓ Cross-platform ITS results saved.")


if __name__ == "__main__":
    main()
