"""32 – Apocalypticism Robustness Checks.

Comprehensive robustness battery for the apocalypticism ITS findings:

  1. **Placebo tests** – Run ITS on randomly drawn pseudo-event dates
     to construct a null distribution of β₂ and test whether observed
     effects are distinguishable from noise.
  2. **Bandwidth sensitivity** – Vary the pre/post window (7, 14, 21,
     30, 45, 60 days) and confirm results are stable.
  3. **Dose-response** – Test whether larger attacks (more casualties)
     produce bigger apocalyptic spikes.
  4. **Online-nexus heterogeneity** – Compare events where the
     perpetrator had known online presence vs. no online nexus.
  5. **Lag structure** – Test delayed effects (1, 3, 7, 14 day lags).
  6. **Weekday / weekend controls** – Add day-of-week fixed effects.
  7. **Multiple-comparison correction** – Apply Benjamini-Hochberg FDR.
  8. **Autoregressive control** – AR(1) model for autocorrelation check.

Output
------
``results/apocalypticism_robustness_results.json``
``figures/apoc_robustness_*.{png,pdf}``
"""

from __future__ import annotations

import datetime
import json
import random

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

# Import from the ITS module
import importlib
_its = importlib.import_module("31_apocalypticism_its")
build_daily_series = _its.build_daily_series
build_event_window = _its.build_event_window
run_its_regression = _its.run_its_regression
run_pooled_its = _its.run_pooled_its
WINDOW_PRE_DAYS = _its.WINDOW_PRE_DAYS
WINDOW_POST_DAYS = _its.WINDOW_POST_DAYS
MIN_DAILY_POSTS = _its.MIN_DAILY_POSTS

N_PLACEBO = 500           # number of placebo iterations
RANDOM_SEED = 42          # reproducibility
BANDWIDTHS = [7, 14, 21, 30, 45, 60]  # window sizes to test
LAG_DAYS = [0, 1, 3, 7, 14]  # treatment delay in days


# ══════════════════════════════════════════════════════════════════════
# 1. Placebo test
# ══════════════════════════════════════════════════════════════════════

def run_placebo_test(
    daily: pl.DataFrame,
    n_events: int,
    measure_col: str = "mean_score",
    n_iter: int = N_PLACEBO,
    seed: int = RANDOM_SEED,
) -> dict:
    """Draw random pseudo-event dates and estimate null β₂ distribution."""
    rng = random.Random(seed)

    all_days = daily["day"].to_list()
    # Exclude edge days (need full window)
    buffer = max(WINDOW_PRE_DAYS, WINDOW_POST_DAYS)
    eligible_days = all_days[buffer:-buffer]

    if len(eligible_days) < n_events:
        return {"error": "insufficient eligible days for placebo test"}

    null_betas_level = []
    null_betas_slope = []

    for _ in range(n_iter):
        pseudo_dates = rng.sample(eligible_days, n_events)
        pseudo_events = pl.DataFrame({
            "event_date": pseudo_dates,
            "event_name": [f"placebo_{i}" for i in range(n_events)],
        })
        r = run_pooled_its(daily, pseudo_events, measure_col)
        if "error" not in r and "b_level" in r:
            null_betas_level.append(r["b_level"])
            null_betas_slope.append(r["b_slope"])

    return {
        "n_iter": n_iter,
        "n_valid": len(null_betas_level),
        "null_betas_level": null_betas_level,
        "null_betas_slope": null_betas_slope,
        "mean_null_level": float(np.mean(null_betas_level)) if null_betas_level else None,
        "std_null_level": float(np.std(null_betas_level)) if null_betas_level else None,
        "mean_null_slope": float(np.mean(null_betas_slope)) if null_betas_slope else None,
        "std_null_slope": float(np.std(null_betas_slope)) if null_betas_slope else None,
    }


def compute_placebo_pvalue(
    observed_beta: float,
    null_distribution: list[float],
) -> float:
    """Two-sided placebo p-value: fraction of null |β| >= |observed β|."""
    if not null_distribution:
        return float("nan")
    abs_obs = abs(observed_beta)
    n_extreme = sum(1 for b in null_distribution if abs(b) >= abs_obs)
    return n_extreme / len(null_distribution)


# ══════════════════════════════════════════════════════════════════════
# 2. Bandwidth sensitivity
# ══════════════════════════════════════════════════════════════════════

def run_bandwidth_sensitivity(
    daily: pl.DataFrame,
    events: pl.DataFrame,
    measure_col: str,
    bandwidths: list[int] = BANDWIDTHS,
) -> list[dict]:
    """Run pooled ITS at different window widths."""
    results = []
    for bw in bandwidths:
        r = run_pooled_its(daily, events, measure_col,
                           pre_days=bw, post_days=bw)
        r["bandwidth_days"] = bw
        results.append(r)
    return results


# ══════════════════════════════════════════════════════════════════════
# 3. Dose-response
# ══════════════════════════════════════════════════════════════════════

def run_dose_response(
    daily: pl.DataFrame,
    events: pl.DataFrame,
    measure_col: str,
) -> dict:
    """Test whether larger attacks produce bigger apocalyptic spikes.

    Adds log(casualties) as a moderator interacted with post_event.
    """
    panels = []
    event_dates = events["event_date"].to_list()
    event_names = events["event_name"].to_list()
    event_killed = events["killed"].to_list()

    for i, (edate, ename, killed) in enumerate(zip(event_dates, event_names, event_killed)):
        window = build_event_window(daily, edate)
        if window is None:
            continue
        window = window.with_columns([
            pl.lit(ename).alias("event_name"),
            pl.lit(i).alias("event_id"),
            pl.lit(float(np.log1p(killed))).alias("log_killed"),
        ])
        panels.append(window)

    if not panels:
        return {"error": "no valid panels"}

    stacked = pl.concat(panels)
    pdf = stacked.to_pandas().dropna(
        subset=["mean_score", "time_centered", "post_event", "log_killed"]
    )
    pdf = pdf[pdf["post_count"] >= MIN_DAILY_POSTS]

    if len(pdf) < 20:
        return {"error": "insufficient data"}

    # Need variation in log_killed for dose-response interaction
    if pdf["log_killed"].nunique() < 2:
        return {"error": "no variation in casualties (need multiple events)"}

    # Interaction: post_event × log_killed
    pdf["post_x_logkilled"] = pdf["post_event"] * pdf["log_killed"]

    y = pdf["mean_score"].values
    X = pdf[["time_centered", "post_event", "time_x_post",
             "log_killed", "post_x_logkilled"]].values
    X = sm.add_constant(X)

    try:
        model = sm.OLS(y, X).fit(cov_type="HC1")
    except Exception as e:
        return {"error": str(e)}

    return {
        "n_obs": int(len(pdf)),
        "n_events": len(panels),
        "b_level": float(model.params[2]),
        "b_slope": float(model.params[3]),
        "b_log_killed": float(model.params[4]),
        "b_post_x_logkilled": float(model.params[5]),
        "p_level": float(model.pvalues[2]),
        "p_slope": float(model.pvalues[3]),
        "p_log_killed": float(model.pvalues[4]),
        "p_post_x_logkilled": float(model.pvalues[5]),
        "r_squared": float(model.rsquared),
    }


# ══════════════════════════════════════════════════════════════════════
# 4. Online nexus heterogeneity
# ══════════════════════════════════════════════════════════════════════

def run_online_nexus_comparison(
    daily: pl.DataFrame,
    events: pl.DataFrame,
    measure_col: str,
) -> dict:
    """Compare ITS effects for events with vs. without online nexus."""
    online = events.filter(pl.col("online_nexus"))
    offline = events.filter(~pl.col("online_nexus"))

    r_on = run_pooled_its(daily, online, measure_col) if online.height >= 2 else {"error": "too few"}
    r_off = run_pooled_its(daily, offline, measure_col) if offline.height >= 2 else {"error": "too few"}

    r_on["group"] = "online_nexus"
    r_on["n_events_group"] = online.height
    r_off["group"] = "no_online_nexus"
    r_off["n_events_group"] = offline.height

    return {"online_nexus": r_on, "no_online_nexus": r_off}


# ══════════════════════════════════════════════════════════════════════
# 5. Lag structure
# ══════════════════════════════════════════════════════════════════════

def run_lag_analysis(
    daily: pl.DataFrame,
    events: pl.DataFrame,
    measure_col: str,
    lags: list[int] = LAG_DAYS,
) -> list[dict]:
    """Shift treatment onset by k days and re-estimate ITS."""
    results = []
    for lag in lags:
        # Shift event dates forward by `lag` days
        shifted = events.with_columns(
            (pl.col("event_date") + datetime.timedelta(days=lag)).alias("event_date")
        )
        r = run_pooled_its(daily, shifted, measure_col)
        r["lag_days"] = lag
        results.append(r)
    return results


# ══════════════════════════════════════════════════════════════════════
# 6. Day-of-week controls
# ══════════════════════════════════════════════════════════════════════

def run_dow_controlled_its(
    daily: pl.DataFrame,
    events: pl.DataFrame,
    measure_col: str,
) -> dict:
    """Pooled ITS with day-of-week fixed effects."""
    panels = []
    event_dates = events["event_date"].to_list()
    event_names = events["event_name"].to_list()

    for i, (edate, ename) in enumerate(zip(event_dates, event_names)):
        window = build_event_window(daily, edate)
        if window is None:
            continue
        window = window.with_columns([
            pl.lit(i).alias("event_id"),
        ])
        panels.append(window)

    if not panels:
        return {"error": "no valid panels"}

    stacked = pl.concat(panels)
    pdf = stacked.to_pandas().dropna(
        subset=["mean_score", "time_centered", "post_event", "time_x_post"]
    )
    pdf = pdf[pdf["post_count"] >= MIN_DAILY_POSTS]

    if len(pdf) < 20:
        return {"error": "insufficient data"}

    # Day-of-week dummies (0=Monday)
    pdf["dow"] = pd.to_datetime(pdf["day"]).dt.dayofweek
    dow_dummies = pd.get_dummies(pdf["dow"], prefix="dow", drop_first=True, dtype=float)

    y = pdf["mean_score"].values
    X_core = pdf[["time_centered", "post_event", "time_x_post"]].values
    X = np.column_stack([X_core, dow_dummies.values])
    X = sm.add_constant(X)

    try:
        model = sm.OLS(y, X).fit(cov_type="HC1")
    except Exception as e:
        return {"error": str(e)}

    return {
        "label": "pooled_dow_controlled",
        "n_obs": int(len(pdf)),
        "b_level": float(model.params[2]),
        "b_slope": float(model.params[3]),
        "p_level": float(model.pvalues[2]),
        "p_slope": float(model.pvalues[3]),
        "r_squared": float(model.rsquared),
    }


# ══════════════════════════════════════════════════════════════════════
# 7. Benjamini-Hochberg FDR
# ══════════════════════════════════════════════════════════════════════

def benjamini_hochberg(pvalues: list[float], alpha: float = 0.05) -> list[dict]:
    """Apply BH FDR correction to a list of p-values.

    Returns list of dicts with original p, rank, BH-adjusted p, and
    whether it passes the threshold.
    """
    n = len(pvalues)
    if n == 0:
        return []

    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    results = [None] * n

    prev_adj = 1.0
    for rank_idx in range(n - 1, -1, -1):
        orig_idx, p = indexed[rank_idx]
        rank = rank_idx + 1
        adj_p = min(prev_adj, (n / rank) * p)
        adj_p = min(adj_p, 1.0)
        prev_adj = adj_p
        results[orig_idx] = {
            "original_index": orig_idx,
            "p_original": p,
            "rank": rank,
            "p_adjusted": adj_p,
            "significant": adj_p < alpha,
        }

    return results


# ══════════════════════════════════════════════════════════════════════
# 8. AR(1) model
# ══════════════════════════════════════════════════════════════════════

def run_ar1_its(
    daily: pl.DataFrame,
    events: pl.DataFrame,
    measure_col: str,
) -> dict:
    """Pooled ITS with an AR(1) lag of the dependent variable."""
    panels = []
    event_dates = events["event_date"].to_list()
    event_names = events["event_name"].to_list()

    for i, (edate, ename) in enumerate(zip(event_dates, event_names)):
        window = build_event_window(daily, edate)
        if window is None:
            continue
        window = window.with_columns([
            pl.lit(i).alias("event_id"),
            pl.col("mean_score").shift(1).alias("y_lag1"),
        ])
        panels.append(window)

    if not panels:
        return {"error": "no valid panels"}

    stacked = pl.concat(panels)
    pdf = stacked.to_pandas().dropna(
        subset=["mean_score", "time_centered", "post_event", "time_x_post", "y_lag1"]
    )
    pdf = pdf[pdf["post_count"] >= MIN_DAILY_POSTS]

    if len(pdf) < 20:
        return {"error": "insufficient data"}

    y = pdf["mean_score"].values
    X = pdf[["time_centered", "post_event", "time_x_post", "y_lag1"]].values
    X = sm.add_constant(X)

    try:
        model = sm.OLS(y, X).fit(cov_type="HC1")
    except Exception as e:
        return {"error": str(e)}

    return {
        "label": "pooled_ar1",
        "n_obs": int(len(pdf)),
        "b_level": float(model.params[2]),
        "b_slope": float(model.params[3]),
        "b_y_lag1": float(model.params[4]),
        "p_level": float(model.pvalues[2]),
        "p_slope": float(model.pvalues[3]),
        "p_y_lag1": float(model.pvalues[4]),
        "r_squared": float(model.rsquared),
    }


# ══════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════

def plot_placebo_histogram(
    observed_beta: float,
    null_distribution: list[float],
    filename: str,
    label: str = "β₂",
):
    """Histogram of null β₂ with observed value marked."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    ax.hist(null_distribution, bins=50, color=CB_PALETTE[0],
            alpha=0.7, edgecolor="white", density=True)
    ax.axvline(observed_beta, color="red", linestyle="--", linewidth=2,
               label=f"Observed {label} = {observed_beta:.4f}")

    p = compute_placebo_pvalue(observed_beta, null_distribution)
    ax.set_title(f"Placebo Test: {label} (p={p:.4f})")
    ax.set_xlabel(label)
    ax.set_ylabel("Density")
    ax.legend()

    save_figure(fig, filename)


def plot_bandwidth_sensitivity(
    bw_results: list[dict],
    filename: str,
):
    """Plot β₂ and p-values across bandwidths."""
    setup_plot_style()
    valid = [r for r in bw_results if "error" not in r]
    if not valid:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    bws = [r["bandwidth_days"] for r in valid]
    betas = [r["b_level"] for r in valid]
    pvals = [r["p_level"] for r in valid]

    ax1.plot(bws, betas, "o-", color=CB_PALETTE[0], linewidth=2, markersize=8)
    ax1.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax1.set_xlabel("Window bandwidth (days)")
    ax1.set_ylabel("β₂ (level change)")
    ax1.set_title("Level Change Across Bandwidths")

    ax2.plot(bws, pvals, "o-", color=CB_PALETTE[2], linewidth=2, markersize=8)
    ax2.axhline(0.05, color="red", linestyle="--", linewidth=1, label="α = 0.05")
    ax2.set_xlabel("Window bandwidth (days)")
    ax2.set_ylabel("p-value")
    ax2.set_title("Statistical Significance Across Bandwidths")
    ax2.legend()

    fig.suptitle("Bandwidth Sensitivity Analysis")
    fig.tight_layout()
    save_figure(fig, filename)


def plot_lag_analysis(
    lag_results: list[dict],
    filename: str,
):
    """Plot β₂ across different treatment lags."""
    setup_plot_style()
    valid = [r for r in lag_results if "error" not in r]
    if not valid:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    lags = [r["lag_days"] for r in valid]
    betas = [r["b_level"] for r in valid]
    pvals = [r["p_level"] for r in valid]

    colors = [CB_PALETTE[2] if p < 0.05 else CB_PALETTE[0] for p in pvals]
    ax.bar(lags, betas, color=colors, alpha=0.8, width=0.8)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Treatment lag (days)")
    ax.set_ylabel("β₂ (level change)")
    ax.set_title("Lag Structure Analysis")

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=CB_PALETTE[2], alpha=0.8, label="p < 0.05"),
        Patch(facecolor=CB_PALETTE[0], alpha=0.8, label="p ≥ 0.05"),
    ]
    ax.legend(handles=legend_elements)

    save_figure(fig, filename)


def main():
    print("=" * 60)
    print("STAGE 32: Apocalypticism Robustness Checks")
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

    primary_measure = "apoc_combined"
    daily = build_daily_series(pol, primary_measure)
    print(f"  Daily series: {daily.height} days")

    # Filter events to data range
    pol_date_min = pol["date"].min()
    pol_date_max = pol["date"].max()
    if pol_date_min is not None and pol_date_max is not None:
        buffer = datetime.timedelta(days=60)  # max bandwidth
        pol_min_date = pol_date_min.date() if hasattr(pol_date_min, "date") else pol_date_min
        pol_max_date = pol_date_max.date() if hasattr(pol_date_max, "date") else pol_date_max
        events = events.filter(
            (pl.col("event_date") >= (pol_min_date + buffer))
            & (pl.col("event_date") <= (pol_max_date - buffer))
        )
    print(f"  Events in range: {events.height}")

    # Filter to mass-violence events for robustness checks
    # (non-violence events are used in ITS category comparison, not here)
    if "event_category" in events.columns:
        violence_events = events.filter(pl.col("event_category") == "mass_violence")
        print(f"  Mass-violence events (for robustness): {violence_events.height}")
    else:
        violence_events = events

    # Need the observed pooled beta for placebo comparison
    its_results_path = RESULTS_DIR / "apocalypticism_its_results.json"
    observed_beta_level = None
    if its_results_path.exists():
        with open(its_results_path) as f:
            prior = json.load(f)
        pooled = prior.get("pooled", {})
        if "b_level" in pooled:
            observed_beta_level = pooled["b_level"]

    # ── 1. Placebo test ───────────────────────────────────────────────
    print("\n  1. Running placebo test…")
    placebo = run_placebo_test(daily, n_events=violence_events.height)
    if observed_beta_level is not None and placebo.get("null_betas_level"):
        placebo["observed_beta_level"] = observed_beta_level
        placebo["placebo_p_level"] = compute_placebo_pvalue(
            observed_beta_level, placebo["null_betas_level"]
        )
        print(f"    Placebo p-value (level): {placebo['placebo_p_level']:.4f}")

        # Plot
        plot_placebo_histogram(
            observed_beta_level,
            placebo["null_betas_level"],
            "apoc_robustness_placebo"
        )
    # Don't save the full null distributions (too large for JSON)
    placebo_summary = {k: v for k, v in placebo.items()
                       if k not in ("null_betas_level", "null_betas_slope")}
    results["placebo"] = placebo_summary

    # ── 2. Bandwidth sensitivity ──────────────────────────────────────
    print("\n  2. Running bandwidth sensitivity…")
    bw = run_bandwidth_sensitivity(daily, violence_events, primary_measure)
    results["bandwidth_sensitivity"] = bw
    for r in bw:
        if "error" not in r:
            print(f"    BW={r['bandwidth_days']}d: β₂={r['b_level']:.6f} "
                  f"(p={r['p_level']:.4f})")
    plot_bandwidth_sensitivity(bw, "apoc_robustness_bandwidth")

    # ── 3. Dose-response ──────────────────────────────────────────────
    print("\n  3. Running dose-response analysis…")
    dose = run_dose_response(daily, violence_events, primary_measure)
    results["dose_response"] = dose
    if "error" not in dose:
        print(f"    β(post×log_killed) = {dose['b_post_x_logkilled']:.6f} "
              f"(p={dose['p_post_x_logkilled']:.4f})")

    # ── 4. Online nexus comparison ────────────────────────────────────
    print("\n  4. Running online nexus comparison…")
    nexus = run_online_nexus_comparison(daily, violence_events, primary_measure)
    results["online_nexus"] = nexus
    for group, r in nexus.items():
        if "error" not in r:
            print(f"    {group}: β₂={r['b_level']:.6f} (p={r['p_level']:.4f})")

    # ── 5. Lag structure ──────────────────────────────────────────────
    print("\n  5. Running lag structure analysis…")
    lags = run_lag_analysis(daily, violence_events, primary_measure)
    results["lag_structure"] = lags
    for r in lags:
        if "error" not in r:
            print(f"    Lag={r['lag_days']}d: β₂={r['b_level']:.6f} "
                  f"(p={r['p_level']:.4f})")
    plot_lag_analysis(lags, "apoc_robustness_lags")

    # ── 6. Day-of-week controls ───────────────────────────────────────
    print("\n  6. Running day-of-week controlled ITS…")
    dow = run_dow_controlled_its(daily, violence_events, primary_measure)
    results["dow_controlled"] = dow
    if "error" not in dow:
        print(f"    DoW-controlled: β₂={dow['b_level']:.6f} (p={dow['p_level']:.4f})")

    # ── 7. BH FDR correction ─────────────────────────────────────────
    print("\n  7. Applying Benjamini-Hochberg FDR correction…")
    if its_results_path.exists():
        with open(its_results_path) as f:
            prior = json.load(f)
        per_event = prior.get("per_event", [])
        pvals = [r["p_level"] for r in per_event if "p_level" in r and "error" not in r]
        labels = [r["label"] for r in per_event if "p_level" in r and "error" not in r]
        if pvals:
            bh = benjamini_hochberg(pvals)
            n_sig_raw = sum(1 for p in pvals if p < 0.05)
            n_sig_bh = sum(1 for r in bh if r["significant"])
            results["bh_fdr"] = {
                "n_tests": len(pvals),
                "n_significant_raw": n_sig_raw,
                "n_significant_bh": n_sig_bh,
                "per_event_bh": [
                    {"label": labels[r["original_index"]],
                     "p_original": r["p_original"],
                     "p_adjusted": r["p_adjusted"],
                     "significant_bh": r["significant"]}
                    for r in bh
                ],
            }
            print(f"    Raw significant: {n_sig_raw}/{len(pvals)}")
            print(f"    BH-corrected significant: {n_sig_bh}/{len(pvals)}")
    else:
        results["bh_fdr"] = {"error": "no per-event results to correct"}

    # ── 8. AR(1) model ────────────────────────────────────────────────
    print("\n  8. Running AR(1) controlled ITS…")
    ar1 = run_ar1_its(daily, violence_events, primary_measure)
    results["ar1_controlled"] = ar1
    if "error" not in ar1:
        print(f"    AR(1): β₂={ar1['b_level']:.6f} (p={ar1['p_level']:.4f}), "
              f"ρ={ar1['b_y_lag1']:.4f} (p={ar1['p_y_lag1']:.4f})")

    # ── Summary table ─────────────────────────────────────────────────
    print("\n  ── Robustness Summary ──")
    checks_passed = 0
    checks_total = 0

    # Placebo
    if placebo_summary.get("placebo_p_level") is not None:
        checks_total += 1
        if placebo_summary["placebo_p_level"] < 0.05:
            checks_passed += 1
            print("    ✓ Placebo test: observed > null (p < 0.05)")
        else:
            print(f"    ✗ Placebo test: not significant "
                  f"(p={placebo_summary['placebo_p_level']:.4f})")

    # Bandwidth stability
    valid_bw = [r for r in bw if "error" not in r and r.get("p_level", 1) < 0.05]
    checks_total += 1
    if len(valid_bw) >= len(bw) // 2:
        checks_passed += 1
        print(f"    ✓ Bandwidth stability: {len(valid_bw)}/{len(bw)} significant")
    else:
        print(f"    ✗ Bandwidth stability: {len(valid_bw)}/{len(bw)} significant")

    # Dose-response direction
    if "error" not in dose and dose.get("b_post_x_logkilled", 0) > 0:
        checks_total += 1
        checks_passed += 1
        print("    ✓ Dose-response: positive coefficient (more deaths → more apocalypticism)")
    elif "error" not in dose:
        checks_total += 1
        print("    ✗ Dose-response: non-positive coefficient")

    # DoW invariance
    if "error" not in dow and dow.get("p_level", 1) < 0.05:
        checks_total += 1
        checks_passed += 1
        print("    ✓ Day-of-week controls: result survives")
    elif "error" not in dow:
        checks_total += 1
        print(f"    ✗ Day-of-week controls: not significant (p={dow.get('p_level', 'N/A')})")

    # AR(1)
    if "error" not in ar1 and ar1.get("p_level", 1) < 0.05:
        checks_total += 1
        checks_passed += 1
        print("    ✓ AR(1) control: result survives")
    elif "error" not in ar1:
        checks_total += 1
        print(f"    ✗ AR(1) control: not significant (p={ar1.get('p_level', 'N/A')})")

    results["summary"] = {
        "checks_passed": checks_passed,
        "checks_total": checks_total,
    }
    print(f"\n  Overall: {checks_passed}/{checks_total} robustness checks passed")

    # ── Save ──────────────────────────────────────────────────────────
    out_path = RESULTS_DIR / "apocalypticism_robustness_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✓ Robustness checks complete. Saved to {out_path.name}")


if __name__ == "__main__":
    main()
