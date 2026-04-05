"""31 – Apocalypticism Interrupted Time Series (ITS) Analysis.

Tests whether apocalyptic rhetoric on /pol/ increases in the immediate
aftermath of mass-casualty violent events.  For each event in the
mass-casualty catalogue (stage 29), we:

  1. Build a daily time series of apocalypticism measures.
  2. Fit a standard ITS regression:
       Y_t = β₀ + β₁·time + β₂·post_event + β₃·(time × post_event) + ε_t
     with Newey-West HAC standard errors to account for autocorrelation.
  3. Also run a *stacked* (pooled) ITS across all events using event
     fixed effects.
  4. Stratify by ideology to test heterogeneous treatment effects.

Output
------
``results/apocalypticism_its_results.json``
  Per-event and pooled regression results.

``figures/apoc_its_*.{png,pdf}``
  Per-event and pooled ITS visualisation.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np
import polars as pl
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from utils import (
    DATA_PROCESSED, RESULTS_DIR, FIGURES_DIR,
    setup_plot_style, save_figure, FIGSIZE_WIDE, CB_PALETTE,
)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Analysis parameters ───────────────────────────────────────────────
WINDOW_PRE_DAYS = 30      # days before event
WINDOW_POST_DAYS = 30     # days after event
MIN_DAILY_POSTS = 5       # minimum posts per day to include
NEWEY_WEST_LAGS = 4       # HAC lag truncation
MEASURES = ["apoc_lr_prob", "apoc_similarity", "apoc_combined"]


def build_daily_series(
    df: pl.DataFrame,
    measure_col: str,
) -> pl.DataFrame:
    """Aggregate apocalypticism measure to daily time series."""
    df = df.filter(pl.col("date").is_not_null())

    daily = (
        df.with_columns(
            pl.col("date").dt.date().alias("day")
        )
        .group_by("day")
        .agg([
            pl.col(measure_col).mean().alias("mean_score"),
            pl.col(measure_col).median().alias("median_score"),
            pl.col(measure_col).sum().alias("total_score"),
            pl.len().alias("post_count"),
            pl.col("apoc_binary").sum().alias("apoc_post_count"),
        ])
        .sort("day")
    )

    # Prevalence: fraction of posts that are apocalyptic
    daily = daily.with_columns(
        (pl.col("apoc_post_count").cast(pl.Float64) / pl.col("post_count")).alias("apoc_prevalence")
    )

    return daily


def build_event_window(
    daily: pl.DataFrame,
    event_date: datetime.date,
    pre_days: int = WINDOW_PRE_DAYS,
    post_days: int = WINDOW_POST_DAYS,
) -> pl.DataFrame | None:
    """Extract a window around an event and add ITS regressors."""
    if event_date is None:
        return None
    window_start = event_date - datetime.timedelta(days=pre_days)
    window_end = event_date + datetime.timedelta(days=post_days)

    window = daily.filter(
        (pl.col("day") >= window_start) & (pl.col("day") <= window_end)
    ).sort("day")

    if window.height < 10:
        return None

    window = window.with_columns([
        ((pl.col("day") - event_date).dt.total_days()).alias("time_centered"),
        (pl.col("day") >= event_date).cast(pl.Int8).alias("post_event"),
    ])
    window = window.with_columns(
        (pl.col("time_centered") * pl.col("post_event")).alias("time_x_post")
    )

    return window


def run_its_regression(
    window: pl.DataFrame,
    label: str,
    y_col: str = "mean_score",
) -> dict:
    """Run ITS regression on an event window with Newey-West HAC SEs."""
    pdf = window.to_pandas().dropna(
        subset=[y_col, "time_centered", "post_event", "time_x_post"]
    )

    # Filter days with too few posts
    pdf = pdf[pdf["post_count"] >= MIN_DAILY_POSTS]

    if len(pdf) < 10:
        return {"label": label, "error": "insufficient data", "n_days": len(pdf)}

    y = pdf[y_col].values
    X = pdf[["time_centered", "post_event", "time_x_post"]].values
    X = sm.add_constant(X)

    try:
        model = sm.OLS(y, X).fit(
            cov_type="HAC", cov_kwds={"maxlags": NEWEY_WEST_LAGS}
        )
    except Exception as e:
        return {"label": label, "error": str(e)}

    result = {
        "label": label,
        "n_days": int(len(pdf)),
        "y_col": y_col,
        "intercept": float(model.params[0]),
        "b_time": float(model.params[1]),
        "b_level": float(model.params[2]),
        "b_slope": float(model.params[3]),
        "se_level": float(model.bse[2]),
        "se_slope": float(model.bse[3]),
        "p_time": float(model.pvalues[1]),
        "p_level": float(model.pvalues[2]),
        "p_slope": float(model.pvalues[3]),
        "r_squared": float(model.rsquared),
        "ci_level_lo": float(model.conf_int()[2, 0]),
        "ci_level_hi": float(model.conf_int()[2, 1]),
        "ci_slope_lo": float(model.conf_int()[3, 0]),
        "ci_slope_hi": float(model.conf_int()[3, 1]),
    }

    return result


def run_pooled_its(
    daily: pl.DataFrame,
    events: pl.DataFrame,
    measure_col: str,
    pre_days: int = WINDOW_PRE_DAYS,
    post_days: int = WINDOW_POST_DAYS,
) -> dict:
    """Run stacked/pooled ITS across all events with event fixed effects.

    Each event contributes a window.  Overlapping windows are resolved by
    assigning each day to the *nearest* event.  The regression includes
    event indicator dummies.
    """
    panels = []
    event_dates = events["event_date"].to_list()
    event_names = events["event_name"].to_list()

    for i, (edate, ename) in enumerate(zip(event_dates, event_names)):
        window = build_event_window(daily, edate, pre_days, post_days)
        if window is None:
            continue
        window = window.with_columns([
            pl.lit(ename).alias("event_name"),
            pl.lit(i).alias("event_id"),
        ])
        panels.append(window)

    if not panels:
        return {"label": "pooled", "error": "no valid event windows"}

    stacked = pl.concat(panels)
    pdf = stacked.to_pandas().dropna(
        subset=["mean_score", "time_centered", "post_event", "time_x_post"]
    )
    pdf = pdf[pdf["post_count"] >= MIN_DAILY_POSTS]

    if len(pdf) < 20:
        return {"label": "pooled", "error": "insufficient pooled data"}

    # Event fixed effects
    event_dummies = pd.get_dummies(pdf["event_id"], prefix="ev", drop_first=True, dtype=float)

    y = pdf["mean_score"].values
    X_core = pdf[["time_centered", "post_event", "time_x_post"]].values
    X = np.column_stack([X_core, event_dummies.values])
    X = sm.add_constant(X)

    try:
        model = sm.OLS(y, X).fit(
            cov_type="HC1"  # Robust SEs for pooled
        )
    except Exception as e:
        return {"label": "pooled", "error": str(e)}

    result = {
        "label": f"pooled_{measure_col}",
        "n_obs": int(len(pdf)),
        "n_events": len(panels),
        "b_time": float(model.params[1]),
        "b_level": float(model.params[2]),
        "b_slope": float(model.params[3]),
        "se_level": float(model.bse[2]),
        "se_slope": float(model.bse[3]),
        "p_time": float(model.pvalues[1]),
        "p_level": float(model.pvalues[2]),
        "p_slope": float(model.pvalues[3]),
        "r_squared": float(model.rsquared),
        "ci_level_lo": float(model.conf_int()[2, 0]),
        "ci_level_hi": float(model.conf_int()[2, 1]),
    }

    return result


def run_stratified_its(
    daily: pl.DataFrame,
    events: pl.DataFrame,
    measure_col: str,
) -> dict:
    """Run pooled ITS stratified by event ideology."""
    results = {}
    for ideology in events["ideology"].unique().to_list():
        subset = events.filter(pl.col("ideology") == ideology)
        if subset.height < 2:
            results[ideology] = {"label": ideology, "error": "too few events",
                                 "n_events": subset.height}
            continue
        r = run_pooled_its(daily, subset, measure_col)
        r["ideology"] = ideology
        r["n_events_in_stratum"] = subset.height
        results[ideology] = r
    return results


def run_category_comparison(
    daily: pl.DataFrame,
    events: pl.DataFrame,
    measure_col: str,
) -> dict:
    """Compare ITS effects for mass violence vs non-violence events.

    This is a falsification test: if apocalyptic rhetoric responds
    ONLY to mass violence and NOT to non-violence discontinuities,
    the causal specificity of the violence effect is strengthened.
    """
    results = {}
    violence = events.filter(pl.col("event_category") == "mass_violence")
    nonviolence = events.filter(pl.col("event_category") != "mass_violence")

    for label, subset in [("mass_violence", violence), ("nonviolence", nonviolence)]:
        if subset.height < 2:
            results[label] = {"label": label, "error": "too few events",
                              "n_events": subset.height}
            continue
        r = run_pooled_its(daily, subset, measure_col)
        r["event_category"] = label
        r["n_events_category"] = subset.height
        results[label] = r

    # Also run by specific non-violence subcategory
    if "event_category" in events.columns:
        for cat in events["event_category"].unique().to_list():
            if cat == "mass_violence":
                continue
            subset = events.filter(pl.col("event_category") == cat)
            if subset.height < 2:
                results[cat] = {"label": cat, "error": "too few events",
                                "n_events": subset.height}
                continue
            r = run_pooled_its(daily, subset, measure_col)
            r["event_category"] = cat
            r["n_events_category"] = subset.height
            results[cat] = r

    return results


def plot_event_its(
    window: pl.DataFrame,
    event_date: datetime.date,
    event_name: str,
    y_col: str,
    filename: str,
):
    """Plot ITS time series for a single event."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    pdf = window.to_pandas().dropna(subset=[y_col])
    ax.scatter(pdf["day"], pdf[y_col],
               color=CB_PALETTE[0], alpha=0.5, s=20, zorder=3)

    # Pre/post regression lines
    event_ts = pd.Timestamp(event_date)
    for phase, color, ls in [("pre", CB_PALETTE[1], "--"), ("post", CB_PALETTE[2], "--")]:
        if phase == "pre":
            mask = pdf["day"] < event_ts
        else:
            mask = pdf["day"] >= event_ts
        subset = pdf[mask]
        if len(subset) >= 2:
            x_num = (subset["day"] - subset["day"].min()).dt.days.values.astype(float)
            if x_num.std() > 0:
                coeffs = np.polyfit(x_num, subset[y_col].values, 1)
                ax.plot(subset["day"], np.polyval(coeffs, x_num),
                        color=color, linestyle=ls, linewidth=2,
                        label=f"{phase}-event trend")

    ax.axvline(pd.Timestamp(event_date), color="red", linestyle="--",
               linewidth=2, alpha=0.7, label=event_name)
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Daily mean {y_col}")
    ax.set_title(f"ITS: {event_name}")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()

    save_figure(fig, filename)


def plot_pooled_forest(
    per_event_results: list[dict],
    filename: str,
):
    """Forest plot of per-event β₂ (level change) with 95% CI."""
    setup_plot_style()

    # Filter to events with valid results
    valid = [r for r in per_event_results if "b_level" in r and "error" not in r]
    if not valid:
        print("  ⚠ No valid results for forest plot")
        return

    valid.sort(key=lambda r: r["b_level"])

    labels = [r["label"] for r in valid]
    betas = [r["b_level"] for r in valid]
    ci_lo = [r.get("ci_level_lo", r["b_level"] - 1.96 * r.get("se_level", 0)) for r in valid]
    ci_hi = [r.get("ci_level_hi", r["b_level"] + 1.96 * r.get("se_level", 0)) for r in valid]

    fig, ax = plt.subplots(figsize=(10, max(6, len(valid) * 0.4)))
    y_pos = range(len(valid))

    ax.errorbar(betas, y_pos,
                xerr=[np.array(betas) - np.array(ci_lo),
                      np.array(ci_hi) - np.array(betas)],
                fmt="o", color=CB_PALETTE[0], capsize=3, markersize=5)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("β₂ (Level change in apocalypticism)")
    ax.set_title("Forest Plot: Post-Event Level Change in Apocalyptic Rhetoric")

    fig.tight_layout()
    save_figure(fig, filename)


def plot_average_response(
    daily: pl.DataFrame,
    events: pl.DataFrame,
    measure_col: str,
    filename: str,
    pre_days: int = WINDOW_PRE_DAYS,
    post_days: int = WINDOW_POST_DAYS,
):
    """Plot average apocalypticism trajectory centred on events."""
    setup_plot_style()

    all_trajectories = []
    for edate in events["event_date"].to_list():
        window = build_event_window(daily, edate, pre_days, post_days)
        if window is None:
            continue
        traj = window.select(["time_centered", "mean_score"]).to_pandas()
        all_trajectories.append(traj)

    if not all_trajectories:
        print("  ⚠ No valid windows for average trajectory")
        return

    combined = pd.concat(all_trajectories)
    avg = combined.groupby("time_centered")["mean_score"].agg(["mean", "sem"]).reset_index()

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(avg["time_centered"], avg["mean"], color=CB_PALETTE[0], linewidth=2)
    ax.fill_between(
        avg["time_centered"],
        avg["mean"] - 1.96 * avg["sem"],
        avg["mean"] + 1.96 * avg["sem"],
        alpha=0.2, color=CB_PALETTE[0],
    )
    ax.axvline(0, color="red", linestyle="--", linewidth=2, alpha=0.7, label="Event day")
    ax.set_xlabel("Days relative to event")
    ax.set_ylabel(f"Mean {measure_col}")
    ax.set_title("Average Apocalyptic Rhetoric Response to Mass-Casualty Events")
    ax.legend()

    save_figure(fig, filename)


def main():
    print("=" * 60)
    print("STAGE 31: Apocalypticism ITS Analysis")
    print("=" * 60)

    results: dict = {}

    # ── Load data ─────────────────────────────────────────────────────
    apoc_path = DATA_PROCESSED / "pol_apocalypticism_scores.parquet"
    events_path = DATA_PROCESSED / "mass_casualty_events.parquet"

    if not apoc_path.exists():
        print(f"  ✗ {apoc_path.name} not found. Run stage 30 first.")
        return
    if not events_path.exists():
        print(f"  ✗ {events_path.name} not found. Run stage 29 first.")
        return

    pol = pl.read_parquet(apoc_path)
    events = pl.read_parquet(events_path)
    print(f"  /pol/ posts: {pol.height:,}")
    print(f"  Events: {events.height}")

    # ── Determine /pol/ data range ────────────────────────────────────
    pol_date_min = pol["date"].min()
    pol_date_max = pol["date"].max()
    print(f"  /pol/ date range: {pol_date_min} to {pol_date_max}")

    # Filter events to /pol/ data range (with buffer for window)
    buffer = datetime.timedelta(days=WINDOW_PRE_DAYS)
    if pol_date_min is not None and pol_date_max is not None:
        pol_min_date = pol_date_min.date() if hasattr(pol_date_min, "date") else pol_date_min
        pol_max_date = pol_date_max.date() if hasattr(pol_date_max, "date") else pol_date_max
        events = events.filter(
            (pl.col("event_date") >= (pol_min_date + buffer))
            & (pl.col("event_date") <= (pol_max_date - buffer))
        )
        print(f"  Events in /pol/ range: {events.height}")

    results["n_posts"] = pol.height
    results["n_events"] = events.height
    results["window_pre_days"] = WINDOW_PRE_DAYS
    results["window_post_days"] = WINDOW_POST_DAYS

    # ── Build daily series for primary measure ────────────────────────
    primary_measure = "apoc_combined"
    daily = build_daily_series(pol, primary_measure)
    print(f"  Daily series: {daily.height} days")

    # ── Per-event ITS ─────────────────────────────────────────────────
    print("\n  Running per-event ITS…")
    per_event_results = []
    for row in events.iter_rows(named=True):
        event_date = row["event_date"]
        event_name = row["event_name"]

        window = build_event_window(daily, event_date)
        if window is None:
            per_event_results.append({
                "label": event_name,
                "event_date": str(event_date),
                "ideology": row["ideology"],
                "killed": row["killed"],
                "error": "insufficient data in window",
            })
            continue

        r = run_its_regression(window, event_name)
        r["event_date"] = str(event_date)
        r["ideology"] = row["ideology"]
        r["killed"] = row["killed"]
        r["online_nexus"] = row["online_nexus"]
        per_event_results.append(r)

        # Print significant results
        if "error" not in r:
            sig = ""
            if r["p_level"] < 0.001:
                sig = "***"
            elif r["p_level"] < 0.01:
                sig = "**"
            elif r["p_level"] < 0.05:
                sig = "*"
            elif r["p_level"] < 0.10:
                sig = "†"
            print(f"    {event_name}: β₂={r['b_level']:.4f} "
                  f"(p={r['p_level']:.4f}) {sig}")

    results["per_event"] = per_event_results

    # Count significant results
    sig_events = [r for r in per_event_results
                  if "error" not in r and r.get("p_level", 1) < 0.05]
    total_valid = [r for r in per_event_results if "error" not in r]
    print(f"\n  Significant at α=0.05: {len(sig_events)}/{len(total_valid)} events")

    # ── Pooled ITS ────────────────────────────────────────────────────
    print("\n  Running pooled ITS (all events)…")
    pooled = run_pooled_its(daily, events, primary_measure)
    results["pooled"] = pooled
    if "error" not in pooled:
        print(f"    Pooled β₂ (level) = {pooled['b_level']:.6f} "
              f"(p={pooled['p_level']:.4f})")
        print(f"    Pooled β₃ (slope) = {pooled['b_slope']:.6f} "
              f"(p={pooled['p_slope']:.4f})")

    # ── Stratified ITS by ideology ────────────────────────────────────
    print("\n  Running stratified ITS by ideology…")
    violence_events = events.filter(pl.col("event_category") == "mass_violence") if "event_category" in events.columns else events
    stratified = run_stratified_its(daily, violence_events, primary_measure)
    results["stratified"] = stratified
    for ideology, r in stratified.items():
        if "error" not in r:
            print(f"    {ideology}: β₂={r['b_level']:.6f} (p={r['p_level']:.4f})")
        else:
            print(f"    {ideology}: {r['error']}")

    # ── Category comparison (falsification) ───────────────────────────
    if "event_category" in events.columns:
        print("\n  Running category comparison (violence vs non-violence)…")
        cat_results = run_category_comparison(daily, events, primary_measure)
        results["category_comparison"] = cat_results
        for cat, r in cat_results.items():
            if "error" not in r:
                print(f"    {cat}: β₂={r['b_level']:.6f} (p={r['p_level']:.4f})")
            else:
                print(f"    {cat}: {r['error']}")

    # ── Run additional measures ───────────────────────────────────────
    print("\n  Running ITS for additional measures…")
    for measure in MEASURES:
        if measure == primary_measure:
            continue
        daily_m = build_daily_series(pol, measure)
        pooled_m = run_pooled_its(daily_m, events, measure)
        results[f"pooled_{measure}"] = pooled_m
        if "error" not in pooled_m:
            print(f"    {measure}: β₂={pooled_m['b_level']:.6f} "
                  f"(p={pooled_m['p_level']:.4f})")

    # Prevalence ITS
    print("\n  Running prevalence ITS…")
    daily_prev = build_daily_series(pol, primary_measure)  # prevalence is computed inside
    pooled_prev = run_pooled_its(daily_prev, events, primary_measure)
    results["pooled_prevalence"] = pooled_prev

    # ── Plots ─────────────────────────────────────────────────────────
    print("\n  Generating plots…")

    # Average response trajectory
    plot_average_response(daily, events, primary_measure, "apoc_its_average_response")

    # Forest plot
    plot_pooled_forest(per_event_results, "apoc_its_forest_plot")

    # Individual event plots for top events (by casualties)
    top_events = events.sort("killed", descending=True).head(10)
    for row in top_events.iter_rows(named=True):
        edate = row["event_date"]
        ename = row["event_name"]
        safe_name = ename.lower().replace(" ", "_").replace("(", "").replace(")", "")[:40]
        window = build_event_window(daily, edate)
        if window is not None:
            plot_event_its(window, edate, ename, "mean_score",
                           f"apoc_its_{safe_name}")

    # ── Save results ──────────────────────────────────────────────────
    out_path = RESULTS_DIR / "apocalypticism_its_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✓ Apocalypticism ITS analysis complete. Saved to {out_path.name}")


if __name__ == "__main__":
    main()
