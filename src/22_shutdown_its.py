"""22 – Post-Shutdown Migration ITS (H15).

Tests whether /pol/ shows a structural break in Siege rhetoric after
Iron March was shut down (November 2017).  IM's closure provides a
cleaner natural experiment than the Siege publication date — it *forces*
users off-platform, so any surge on /pol/ immediately after suggests
displaced IM users migrated there.

Treatment date
--------------
T_shutdown ≈ 2017-11-21 (last known IM post: 2017-11-21 03:41:24 UTC).

Additional analyses
-------------------
- Compare effect size at T_shutdown vs T0 (Siege publication).
- Separate ITS for raw score, prevalence, and volume.
- Changepoint detection near the shutdown window.
"""

from __future__ import annotations

import datetime
import json

import numpy as np
import polars as pl
import statsmodels.api as sm

from utils import (
    DATA_PROCESSED, RESULTS_DIR, FIGURES_DIR,
    setup_plot_style, save_figure, FIGSIZE_WIDE, CB_PALETTE,
)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Treatment date: IM shutdown ───────────────────────────────────────
T_SHUTDOWN = datetime.datetime(2017, 11, 21, tzinfo=datetime.timezone.utc)


def build_weekly_series(
    df: pl.DataFrame,
    t0: datetime.datetime,
    measure_col: str,
) -> pl.DataFrame:
    """Aggregate to weekly time series with ITS regressors centred on *t0*."""
    df = df.filter(pl.col("date").is_not_null())

    weekly = (
        df.with_columns(pl.col("date").dt.truncate("1w").alias("week"))
        .group_by("week")
        .agg([
            pl.col(measure_col).mean().alias("mean_score"),
            pl.len().alias("post_count"),
            pl.col("siege_binary").sum().alias("siege_post_count"),
        ])
        .sort("week")
    )

    # Align timezone: compare t0 against week column's tz
    week_dtype = weekly["week"].dtype
    if hasattr(week_dtype, "time_zone") and week_dtype.time_zone:  # type: ignore[union-attr]
        # Column is tz-aware — ensure t0 also has tz
        import datetime as _dt
        if t0.tzinfo is None:
            t0_cmp = t0.replace(tzinfo=_dt.timezone.utc)
        else:
            t0_cmp = t0
    else:
        # Column is tz-naive — strip tz from t0
        t0_cmp = t0.replace(tzinfo=None) if t0.tzinfo else t0

    weekly = weekly.with_columns([
        (pl.col("week") >= t0_cmp).cast(pl.Int8).alias("post_treatment"),
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


def run_its(weekly: pl.DataFrame, label: str, y_col: str = "mean_score") -> dict:
    """Run ITS regression with Newey-West HAC SEs."""
    pdf = weekly.to_pandas().dropna(
        subset=[y_col, "time_centered", "post_treatment", "time_x_post"]
    )
    if len(pdf) < 10:
        print(f"  ⚠ {label}: insufficient data ({len(pdf)} weeks)")
        return {"label": label, "error": "insufficient data"}

    y = pdf[y_col].values
    X = sm.add_constant(
        pdf[["time_centered", "post_treatment", "time_x_post"]].values
    )
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})

    result = {
        "label": label,
        "n_weeks": int(len(pdf)),
        "intercept": float(model.params[0]),
        "b_time": float(model.params[1]),
        "b_level": float(model.params[2]),
        "b_slope": float(model.params[3]),
        "p_time": float(model.pvalues[1]),
        "p_level": float(model.pvalues[2]),
        "p_slope": float(model.pvalues[3]),
        "r_squared": float(model.rsquared),
    }

    print(f"\n  {label}:")
    print(f"    β₂ (level) = {result['b_level']:.6f}  (p={result['p_level']:.4f})")
    print(f"    β₃ (slope) = {result['b_slope']:.6f}  (p={result['p_slope']:.4f})")
    print(f"    R² = {result['r_squared']:.4f}")
    return result


def build_prevalence(
    siege_weekly: pl.DataFrame,
    all_weekly: pl.DataFrame,
) -> pl.DataFrame:
    """Join siege weekly counts with total /pol/ weekly counts."""
    return (
        siege_weekly.select([
            "week", "siege_post_count", "post_count", "mean_score",
            "post_treatment", "time_centered", "time_x_post",
        ])
        .with_columns(pl.col("week").cast(pl.Date))
        .join(
            all_weekly.select(["week_start", "total_posts"])
            .rename({"week_start": "week"}),
            on="week",
            how="left",
        )
        .with_columns(
            (pl.col("siege_post_count")
             / pl.col("total_posts").cast(pl.Float64).clip(1, None))
            .alias("siege_prevalence")
        )
    )


def plot_shutdown_its(
    weekly: pl.DataFrame,
    t0_siege: datetime.datetime,
    t_shutdown: datetime.datetime,
    filename: str,
):
    """Dual-panel plot: keyword score & post volume around shutdown."""
    setup_plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    pdf = weekly.to_pandas().dropna(subset=["mean_score"])

    # Panel 1 – mean score
    axes[0].plot(pdf["week"], pdf["mean_score"],
                 color=CB_PALETTE[0], alpha=0.6, linewidth=1)
    axes[0].axvline(t_shutdown, color="red", linestyle="--", linewidth=2,
                    label="IM shutdown")
    axes[0].axvline(t0_siege, color="orange", linestyle=":", linewidth=1.5,
                    alpha=0.6, label="Siege publication")
    axes[0].set_ylabel("Mean Siege Score")
    axes[0].set_title("/pol/ Siege Rhetoric: Post-Shutdown ITS")
    axes[0].legend()

    # Panel 2 – post volume
    axes[1].bar(pdf["week"], pdf["post_count"],
                color=CB_PALETTE[2], alpha=0.7, width=5)
    axes[1].axvline(t_shutdown, color="red", linestyle="--", linewidth=2)
    axes[1].axvline(t0_siege, color="orange", linestyle=":", linewidth=1.5,
                    alpha=0.6)
    axes[1].set_ylabel("Siege-relevant Posts / Week")
    axes[1].set_xlabel("Date")

    fig.autofmt_xdate()
    fig.tight_layout()
    save_figure(fig, filename)


def main():
    print("=" * 60)
    print("H15: Post-Shutdown Migration ITS")
    print("=" * 60)

    results: dict = {"t_shutdown": str(T_SHUTDOWN)}

    # Load /pol/ scores
    pol_path = DATA_PROCESSED / "pol_siege_scores.parquet"
    if not pol_path.exists():
        print("  ✗ pol_siege_scores.parquet not found.")
        return
    pol = pl.read_parquet(pol_path)
    print(f"  /pol/ posts: {pol.height:,}")

    # Load T0 for comparison
    td_path = DATA_PROCESSED / "treatment_dates.json"
    if td_path.exists():
        with open(td_path) as f:
            t0_siege = datetime.datetime.fromisoformat(json.load(f)["T0"])
    else:
        t0_siege = datetime.datetime(2015, 6, 3, tzinfo=datetime.timezone.utc)
    results["t0_siege"] = str(t0_siege)

    # ---------- ITS at T_shutdown ----------
    print("\n  ITS with T_shutdown as treatment:")
    ws = build_weekly_series(pol, T_SHUTDOWN, "siege_keyword_score")
    results["keyword_its"] = run_its(ws, "pol_keyword_shutdown")

    # Volume ITS
    results["volume_its"] = run_its(ws, "pol_volume_shutdown", y_col="post_count")

    # Prevalence ITS
    wt_path = DATA_PROCESSED / "pol_weekly_totals.parquet"
    if wt_path.exists():
        all_wt = pl.read_parquet(wt_path)
        prev = build_prevalence(ws, all_wt)
        results["prevalence_its"] = run_its(
            prev, "pol_prevalence_shutdown", y_col="siege_prevalence"
        )

    # Similarity ITS if available
    if "siege_similarity" in pol.columns:
        ws_sim = build_weekly_series(pol, T_SHUTDOWN, "siege_similarity")
        results["similarity_its"] = run_its(ws_sim, "pol_similarity_shutdown")

    # ---------- Compare T0 vs T_shutdown effect sizes ----------
    ws_t0 = build_weekly_series(pol, t0_siege, "siege_keyword_score")
    results["keyword_its_t0"] = run_its(ws_t0, "pol_keyword_t0_comparison")

    # Compare β₂ magnitudes
    b_shut = results.get("keyword_its", {}).get("b_level", 0)
    b_t0 = results.get("keyword_its_t0", {}).get("b_level", 0)
    results["effect_comparison"] = {
        "b_level_shutdown": b_shut,
        "b_level_t0": b_t0,
        "ratio": abs(b_shut / b_t0) if b_t0 != 0 else None,
        "larger": "shutdown" if abs(b_shut) > abs(b_t0) else "t0",
    }
    print(f"\n  Effect comparison: |β₂_shutdown|={abs(b_shut):.4f} "
          f"vs |β₂_T0|={abs(b_t0):.4f}")

    # ---------- Plot ----------
    plot_shutdown_its(ws, t0_siege, T_SHUTDOWN, "shutdown_its")

    # ---------- Save ----------
    with open(RESULTS_DIR / "shutdown_its_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\n✓ Shutdown ITS results saved.")


if __name__ == "__main__":
    main()
