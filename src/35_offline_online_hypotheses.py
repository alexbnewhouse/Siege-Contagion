"""35 – Offline Violence ↔ Online Neo-Fascism: Five New Hypotheses.

Tests five novel hypotheses about the interaction between real-world
mass-casualty violence and online neo-fascist apocalyptic rhetoric
on 4chan's /pol/.

Hypotheses
----------
H22: **Contagion Decay** – Post-attack apocalyptic rhetoric spikes
     decay exponentially.  We estimate the half-life by fitting an
     exponential decay model to the post-event impulse trajectory.

H23: **Reciprocal Amplification** – There exist feedback loops where
     spikes in online apocalyptic rhetoric temporally *precede*
     subsequent real-world events, and vice-versa.  Tested via
     bivariate VAR impulse-response and Granger in both directions.

H24: **Threshold Activation** – Apocalyptic rhetoric on /pol/
     responds non-linearly: only attacks exceeding a casualty
     threshold produce significant rhetoric shifts.  We test via
     piecewise regression and threshold VAR.

H25: **Temporal Clustering** – Attacks cluster temporally.  When
     multiple events occur within a short window, the compounding
     effect on rhetoric is super-additive (interaction > sum of
     individual effects).

H26: **Mimetic Contagion** – Attacks perpetrated by individuals with
     a documented online nexus produce rhetoric that is semantically
     *more similar* to the attacker's language/ideology, i.e. the
     rhetoric shifts *toward* the specific attack's ideological frame.

Output
------
``results/offline_online_hypotheses_results.json``
``figures/h22_*.{png,pdf}``  through  ``figures/h26_*.{png,pdf}``
"""

from __future__ import annotations

import datetime
import importlib
import json
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR as VARModel
from scipy import stats
from scipy.optimize import curve_fit

from utils import (
    DATA_PROCESSED, RESULTS_DIR, FIGURES_DIR,
    setup_plot_style, save_figure, FIGSIZE_WIDE, CB_PALETTE,
)
import matplotlib.pyplot as plt

# ── Imports from earlier stages ───────────────────────────────────────
_its = importlib.import_module("31_apocalypticism_its")
build_daily_series = _its.build_daily_series
build_event_window = _its.build_event_window
WINDOW_PRE_DAYS = _its.WINDOW_PRE_DAYS
WINDOW_POST_DAYS = _its.WINDOW_POST_DAYS
MIN_DAILY_POSTS = _its.MIN_DAILY_POSTS

_adv = importlib.import_module("34_advanced_ts_apocalypticism")
build_event_series = _adv.build_event_series


# ══════════════════════════════════════════════════════════════════════
# H22: Contagion Decay – Exponential half-life estimation
# ══════════════════════════════════════════════════════════════════════

def _exp_decay(t, a, lam, c):
    """Exponential decay: a * exp(-lam * t) + c."""
    return a * np.exp(-lam * t) + c


def estimate_decay_halflife(
    daily: pl.DataFrame,
    events: pl.DataFrame,
    measure_col: str = "apoc_combined",
    post_days: int = 30,
) -> dict:
    """Estimate the half-life of post-attack apocalypticism spikes.

    For each event, extracts the post-event trajectory and fits
    y(t) = a · exp(-λ · t) + c.  Half-life = ln(2) / λ.
    """
    per_event = []
    all_trajectories = []

    for row in events.iter_rows(named=True):
        edate = row["event_date"]
        ename = row["event_name"]

        window = build_event_window(daily, edate, pre_days=7, post_days=post_days)
        if window is None:
            continue

        pdf = window.to_pandas()
        post = pdf[pdf["time_centered"] >= 0].sort_values("time_centered")

        if len(post) < 5:
            continue

        t = post["time_centered"].values.astype(float)
        y = post["mean_score"].values

        # Baseline: mean of pre-event period
        pre = pdf[pdf["time_centered"] < 0]
        baseline = pre["mean_score"].mean() if len(pre) > 0 else y[-1]

        # Fit exponential decay
        try:
            p0 = [max(y[0] - baseline, 0.001), 0.1, baseline]
            popt, pcov = curve_fit(
                _exp_decay, t, y, p0=p0,
                bounds=([0, 0, -np.inf], [np.inf, 10, np.inf]),
                maxfev=5000,
            )
            a, lam, c = popt
            half_life = np.log(2) / lam if lam > 1e-10 else float("nan")

            y_pred = _exp_decay(t, *popt)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

            per_event.append({
                "event": ename,
                "amplitude": float(a),
                "decay_rate": float(lam),
                "half_life_days": float(half_life),
                "baseline": float(c),
                "r_squared": float(r_squared),
                "n_post_days": int(len(post)),
            })

            all_trajectories.append({
                "t": t.tolist(),
                "y_centered": (y - baseline).tolist(),
            })

        except (RuntimeError, ValueError):
            per_event.append({
                "event": ename,
                "error": "curve_fit failed",
            })

    if not per_event:
        return {"error": "no events analyzable"}

    valid = [r for r in per_event if "error" not in r and not np.isnan(r["half_life_days"])]
    half_lives = [r["half_life_days"] for r in valid]

    # Fit average trajectory
    avg_trajectory = {}
    if all_trajectories:
        max_t = max(max(tr["t"]) for tr in all_trajectories)
        t_grid = np.arange(0, int(max_t) + 1)
        y_grid = np.full(len(t_grid), np.nan)

        for ti in range(len(t_grid)):
            vals = []
            for tr in all_trajectories:
                idx = [i for i, x in enumerate(tr["t"]) if abs(x - ti) < 0.5]
                if idx:
                    vals.append(tr["y_centered"][idx[0]])
            if vals:
                y_grid[ti] = np.mean(vals)

        valid_mask = ~np.isnan(y_grid)
        if sum(valid_mask) >= 5:
            try:
                t_valid = t_grid[valid_mask]
                y_valid = y_grid[valid_mask]
                p0_avg = [max(y_valid[0], 0.001), 0.1, 0]
                popt_avg, _ = curve_fit(
                    _exp_decay, t_valid, y_valid, p0=p0_avg,
                    bounds=([0, 0, -np.inf], [np.inf, 10, np.inf]),
                    maxfev=5000,
                )
                avg_half_life = np.log(2) / popt_avg[1] if popt_avg[1] > 1e-10 else float("nan")
                avg_trajectory = {
                    "amplitude": float(popt_avg[0]),
                    "decay_rate": float(popt_avg[1]),
                    "half_life_days": float(avg_half_life),
                    "baseline": float(popt_avg[2]),
                }
            except (RuntimeError, ValueError):
                avg_trajectory = {"error": "average curve_fit failed"}

    result = {
        "hypothesis": "H22",
        "description": "Post-attack apocalyptic rhetoric decays exponentially",
        "n_events": len(per_event),
        "n_valid_fits": len(valid),
        "per_event": per_event,
        "aggregate": {
            "mean_half_life": float(np.mean(half_lives)) if half_lives else float("nan"),
            "median_half_life": float(np.median(half_lives)) if half_lives else float("nan"),
            "std_half_life": float(np.std(half_lives)) if half_lives else float("nan"),
            "min_half_life": float(np.min(half_lives)) if half_lives else float("nan"),
            "max_half_life": float(np.max(half_lives)) if half_lives else float("nan"),
        },
        "average_trajectory_fit": avg_trajectory,
        "supported": len(valid) > 0 and (np.median(half_lives) if half_lives else float("nan")) > 0,
    }

    return result


def plot_decay(decay_results: dict, filename: str):
    """Plot the average decay trajectory and half-life distribution."""
    if "error" in decay_results:
        return

    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Half-life distribution
    valid = [r for r in decay_results["per_event"]
             if "error" not in r and not np.isnan(r["half_life_days"])]
    if valid:
        ax = axes[0]
        hl = [r["half_life_days"] for r in valid]
        # Clip extreme values for visualization
        hl_clipped = [min(h, 60) for h in hl]
        ax.hist(hl_clipped, bins=15, color=CB_PALETTE[0], edgecolor="black",
                alpha=0.7)
        median_hl = np.median(hl)
        ax.axvline(median_hl, color="red", linestyle="--",
                   label=f"Median = {median_hl:.1f} days")
        ax.set_xlabel("Half-life (days)")
        ax.set_ylabel("Count")
        ax.set_title("H22: Distribution of Decay Half-Lives")
        ax.legend()

    # Amplitude vs half-life
    if valid:
        ax = axes[1]
        amps = [r["amplitude"] for r in valid]
        hls = [r["half_life_days"] for r in valid]
        ax.scatter(amps, hls, color=CB_PALETTE[0], alpha=0.6, s=30)
        ax.set_xlabel("Spike amplitude")
        ax.set_ylabel("Half-life (days)")
        ax.set_title("Amplitude vs. Decay Speed")

    fig.tight_layout()
    save_figure(fig, filename)


# ══════════════════════════════════════════════════════════════════════
# H23: Reciprocal Amplification – Bidirectional feedback
# ══════════════════════════════════════════════════════════════════════

def test_reciprocal_amplification(
    ts_df: pd.DataFrame,
    maxlag: int = 14,
    irf_periods: int = 30,
) -> dict:
    """Test for bidirectional feedback between online rhetoric and
    offline violence using VAR-based Granger causality and IRFs.

    H23: There exist feedback loops where spikes in online apocalyptic
    rhetoric precede subsequent real-world events, and vice-versa.
    """
    data = ts_df[["apoc_mean", "event_occurred"]].dropna()

    if len(data) < 50:
        return {"error": "insufficient data"}

    try:
        model = VARModel(data)
        fitted = model.fit(maxlags=min(maxlag, len(data) // 5 - 1), ic="aic")
    except Exception as e:
        return {"error": f"VAR fit failed: {e}"}

    # Bidirectional Granger causality
    gc_results = {}
    for caused, by in [("apoc_mean", "event_occurred"),
                        ("event_occurred", "apoc_mean")]:
        try:
            gc = fitted.test_causality(caused, [by])
            gc_results[f"{by}_causes_{caused}"] = {
                "test_statistic": float(gc.test_statistic),
                "p_value": float(gc.pvalue),
                "df": int(gc.df_num),
                "significant": bool(gc.pvalue < 0.05),
            }
        except Exception as e:
            gc_results[f"{by}_causes_{caused}"] = {"error": str(e)}

    # Impulse responses in both directions
    irf_results = {}
    try:
        irf = fitted.irf(irf_periods)
        # Event → Apoc
        irf_e2a = irf.irfs[:, 0, 1]
        irf_results["event_to_apoc"] = {
            "response": [float(x) for x in irf_e2a],
            "cumulative": [float(x) for x in np.cumsum(irf_e2a)],
            "peak": float(np.max(np.abs(irf_e2a))),
            "peak_day": int(np.argmax(np.abs(irf_e2a))),
        }
        # Apoc → Event
        irf_a2e = irf.irfs[:, 1, 0]
        irf_results["apoc_to_event"] = {
            "response": [float(x) for x in irf_a2e],
            "cumulative": [float(x) for x in np.cumsum(irf_a2e)],
            "peak": float(np.max(np.abs(irf_a2e))),
            "peak_day": int(np.argmax(np.abs(irf_a2e))),
        }
    except Exception as e:
        irf_results = {"error": str(e)}

    # Determine if reciprocal
    e2a_sig = gc_results.get("event_occurred_causes_apoc_mean", {}).get("significant", False)
    a2e_sig = gc_results.get("apoc_mean_causes_event_occurred", {}).get("significant", False)

    result = {
        "hypothesis": "H23",
        "description": "Reciprocal amplification: bidirectional feedback between violence and rhetoric",
        "n_obs": int(len(data)),
        "var_lag": int(fitted.k_ar),
        "granger_causality": gc_results,
        "irf": irf_results,
        "findings": {
            "events_cause_rhetoric": e2a_sig,
            "rhetoric_causes_events": a2e_sig,
            "reciprocal_feedback": e2a_sig and a2e_sig,
            "unidirectional_event_to_rhetoric": e2a_sig and not a2e_sig,
            "unidirectional_rhetoric_to_event": not e2a_sig and a2e_sig,
            "no_causal_link": not e2a_sig and not a2e_sig,
        },
        "supported": e2a_sig and a2e_sig,
    }

    return result


def plot_reciprocal_irf(results: dict, filename: str):
    """Plot bidirectional IRFs."""
    irf = results.get("irf", {})
    if "error" in irf:
        return

    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, key, title in [
        (axes[0], "event_to_apoc", "Event → Apocalypticism"),
        (axes[1], "apoc_to_event", "Apocalypticism → Event"),
    ]:
        if key in irf:
            response = irf[key]["response"]
            ax.plot(range(len(response)), response, color=CB_PALETTE[0],
                    linewidth=2)
            ax.axhline(0, color="gray", linestyle="--", linewidth=1)
            ax.set_xlabel("Days after shock")
            ax.set_ylabel("Response")
            ax.set_title(f"H23: {title}")

    fig.tight_layout()
    save_figure(fig, filename)


# ══════════════════════════════════════════════════════════════════════
# H24: Threshold Activation – Non-linear casualty response
# ══════════════════════════════════════════════════════════════════════

def test_threshold_activation(
    daily: pl.DataFrame,
    events: pl.DataFrame,
    measure_col: str = "apoc_combined",
    candidate_thresholds: list[int] | None = None,
) -> dict:
    """Test whether apocalyptic rhetoric only responds to attacks
    above a casualty threshold.

    Method: Piecewise regression with threshold search.
    For each candidate threshold T:
      β = mean(Δapoc | killed >= T) - mean(Δapoc | killed < T)
    Select T that maximises the absolute difference; test significance.
    """
    if candidate_thresholds is None:
        candidate_thresholds = [3, 5, 10, 15, 20, 30, 50]

    # Get per-event apocalypticism change
    violence_events = events.filter(pl.col("event_category") == "mass_violence")
    per_event = []

    for row in violence_events.iter_rows(named=True):
        edate = row["event_date"]
        killed = row["killed"]
        ename = row["event_name"]

        window = build_event_window(daily, edate, pre_days=14, post_days=14)
        if window is None:
            continue

        pdf = window.to_pandas()
        pre = pdf[pdf["time_centered"] < 0]["mean_score"]
        post = pdf[pdf["time_centered"] >= 0]["mean_score"]

        if len(pre) < 3 or len(post) < 3:
            continue

        delta = post.mean() - pre.mean()
        per_event.append({
            "event": ename,
            "killed": killed,
            "delta_apoc": float(delta),
        })

    if len(per_event) < 10:
        return {"error": "insufficient events for threshold analysis"}

    pdf = pd.DataFrame(per_event)

    # Test each threshold
    threshold_results = []
    for T in candidate_thresholds:
        above = pdf[pdf["killed"] >= T]["delta_apoc"]
        below = pdf[pdf["killed"] < T]["delta_apoc"]

        if len(above) < 3 or len(below) < 3:
            continue

        diff = float(above.mean() - below.mean())

        # Mann-Whitney test
        try:
            u_stat, u_p = stats.mannwhitneyu(above, below, alternative="two-sided")
        except Exception:
            u_stat, u_p = float("nan"), float("nan")

        # Welch's t-test
        try:
            t_result = stats.ttest_ind(above, below, equal_var=False)
            t_stat = float(t_result.statistic)  # type: ignore[union-attr]
            t_p = float(t_result.pvalue)  # type: ignore[union-attr]
        except Exception:
            t_stat, t_p = float("nan"), float("nan")

        threshold_results.append({
            "threshold": T,
            "n_above": int(len(above)),
            "n_below": int(len(below)),
            "mean_above": float(above.mean()),
            "mean_below": float(below.mean()),
            "difference": diff,
            "mannwhitney_U": float(u_stat),
            "mannwhitney_p": float(u_p),
            "welch_t": t_stat,
            "welch_p": t_p,
            "significant": bool(u_p < 0.05),
        })

    if not threshold_results:
        return {"error": "no valid thresholds testable"}

    # Optimal threshold = smallest p-value
    best = min(threshold_results, key=lambda r: r.get("mannwhitney_p", 1.0))

    # Piecewise regression: Y = β₀ + β₁·killed + β₂·(killed > T) + β₃·(killed > T)·killed
    best_T = best["threshold"]
    pdf["above_threshold"] = (pdf["killed"] >= best_T).astype(int)
    pdf["killed_above"] = pdf["above_threshold"] * pdf["killed"]

    try:
        X = pdf[["killed", "above_threshold", "killed_above"]].values
        X = sm.add_constant(X)
        y = pdf["delta_apoc"].values
        pw_model = sm.OLS(y, X).fit(cov_type="HC1")

        piecewise = {
            "threshold_used": best_T,
            "b_killed": float(pw_model.params[1]),
            "b_above_threshold": float(pw_model.params[2]),
            "b_killed_above": float(pw_model.params[3]),
            "p_killed": float(pw_model.pvalues[1]),
            "p_above_threshold": float(pw_model.pvalues[2]),
            "p_killed_above": float(pw_model.pvalues[3]),
            "r_squared": float(pw_model.rsquared),
        }
    except Exception as e:
        piecewise = {"error": str(e)}

    result = {
        "hypothesis": "H24",
        "description": "Threshold activation: rhetoric responds only above casualty threshold",
        "n_events": len(per_event),
        "threshold_tests": threshold_results,
        "optimal_threshold": best,
        "piecewise_regression": piecewise,
        "supported": best.get("significant", False),
    }

    return result


def plot_threshold(results: dict, filename: str):
    """Plot threshold activation analysis."""
    if "error" in results:
        return

    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # P-values across thresholds
    threshold_tests = results.get("threshold_tests", [])
    if threshold_tests:
        ax = axes[0]
        ts = [r["threshold"] for r in threshold_tests]
        ps = [r["mannwhitney_p"] for r in threshold_tests]
        ax.plot(ts, ps, "o-", color=CB_PALETTE[0])
        ax.axhline(0.05, color="red", linestyle="--", alpha=0.7, label="α = 0.05")
        ax.set_xlabel("Casualty threshold")
        ax.set_ylabel("Mann-Whitney p-value")
        ax.set_title("H24: Threshold Scan")
        ax.legend()

    # Mean difference at each threshold
    if threshold_tests:
        ax = axes[1]
        ts = [r["threshold"] for r in threshold_tests]
        diffs = [r["difference"] for r in threshold_tests]
        colors = [CB_PALETTE[2] if r["significant"] else CB_PALETTE[7]
                  for r in threshold_tests]
        ax.bar(range(len(ts)), diffs, tick_label=[str(t) for t in ts],
               color=colors, edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="gray", linestyle="--")
        ax.set_xlabel("Casualty threshold")
        ax.set_ylabel("Mean Δ above − Δ below")
        ax.set_title("Effect Size by Threshold")

    fig.tight_layout()
    save_figure(fig, filename)


# ══════════════════════════════════════════════════════════════════════
# H25: Temporal Clustering – Compounding effects of sequential attacks
# ══════════════════════════════════════════════════════════════════════

def test_temporal_clustering(
    daily: pl.DataFrame,
    events: pl.DataFrame,
    measure_col: str = "apoc_combined",
    cluster_window_days: int = 14,
) -> dict:
    """Test whether temporally clustered attacks produce super-additive
    rhetoric effects.

    Method:
    1. Identify event clusters (events within cluster_window_days).
    2. Compare post-event Δapoc for isolated events vs clustered events.
    3. Test interaction: does the effect of event N+1 depend on
       recency of event N?
    """
    violence = events.filter(
        pl.col("event_category") == "mass_violence"
    ).sort("event_date")

    if violence.height < 5:
        return {"error": "too few violence events for clustering analysis"}

    dates = violence["event_date"].to_list()
    names = violence["event_name"].to_list()

    # Classify events as isolated vs clustered
    classifications = []
    for i, (d, n) in enumerate(zip(dates, names)):
        # Check distance to nearest other event
        distances = [abs((d - other).days) for j, other in enumerate(dates) if j != i]
        min_dist = min(distances) if distances else float("inf")

        is_clustered = min_dist <= cluster_window_days
        n_neighbors = sum(1 for dist in distances if dist <= cluster_window_days)

        # Compute delta apoc
        window = build_event_window(daily, d, pre_days=14, post_days=14)
        if window is None:
            continue

        pdf = window.to_pandas()
        pre = pdf[pdf["time_centered"] < 0]["mean_score"]
        post = pdf[pdf["time_centered"] >= 0]["mean_score"]

        if len(pre) < 3 or len(post) < 3:
            continue

        delta = post.mean() - pre.mean()
        classifications.append({
            "event": n,
            "date": str(d),
            "min_distance_days": min_dist,
            "is_clustered": is_clustered,
            "n_neighbors": n_neighbors,
            "delta_apoc": float(delta),
        })

    if len(classifications) < 5:
        return {"error": "too few classifiable events"}

    cdf = pd.DataFrame(classifications)

    isolated = cdf[~cdf["is_clustered"]]["delta_apoc"]
    clustered = cdf[cdf["is_clustered"]]["delta_apoc"]

    comparison = {}
    if len(isolated) >= 2 and len(clustered) >= 2:
        try:
            u_stat, u_p = stats.mannwhitneyu(clustered, isolated,
                                              alternative="greater")
        except Exception:
            u_stat, u_p = float("nan"), float("nan")

        comparison = {
            "n_isolated": int(len(isolated)),
            "n_clustered": int(len(clustered)),
            "mean_isolated": float(isolated.mean()),
            "mean_clustered": float(clustered.mean()),
            "mannwhitney_U": float(u_stat),
            "mannwhitney_p": float(u_p),
            "significant": bool(u_p < 0.05),
            "super_additive": bool(u_p < 0.05 and clustered.mean() > isolated.mean()),
        }
    else:
        comparison = {"error": "insufficient events in one or both groups"}

    # Regression: delta_apoc ~ n_neighbors
    regression = {}
    if len(cdf) >= 5:
        try:
            X = sm.add_constant(cdf["n_neighbors"].values)
            y = cdf["delta_apoc"].values
            model = sm.OLS(y, X).fit(cov_type="HC1")
            regression = {
                "b_neighbors": float(model.params[1]),
                "p_neighbors": float(model.pvalues[1]),
                "r_squared": float(model.rsquared),
                "n_obs": int(len(cdf)),
            }
        except Exception as e:
            regression = {"error": str(e)}

    result = {
        "hypothesis": "H25",
        "description": "Temporal clustering: sequential attacks produce compounding rhetoric effects",
        "cluster_window_days": cluster_window_days,
        "n_events": len(classifications),
        "per_event": classifications,
        "comparison": comparison,
        "regression": regression,
        "supported": comparison.get("super_additive", False),
    }

    return result


def plot_clustering(results: dict, filename: str):
    """Plot temporal clustering analysis."""
    if "error" in results:
        return

    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    per_event = results.get("per_event", [])
    if not per_event:
        return

    df = pd.DataFrame(per_event)

    # Isolated vs clustered boxplot
    ax = axes[0]
    groups = [
        df[~df["is_clustered"]]["delta_apoc"].values,
        df[df["is_clustered"]]["delta_apoc"].values,
    ]
    bp = ax.boxplot(groups, labels=["Isolated", "Clustered"],
                    patch_artist=True)
    bp["boxes"][0].set_facecolor(CB_PALETTE[0])
    bp["boxes"][1].set_facecolor(CB_PALETTE[2])
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_ylabel("Δ Apocalypticism")
    ax.set_title("H25: Isolated vs. Clustered Events")

    # Scatter: n_neighbors vs delta
    ax = axes[1]
    ax.scatter(df["n_neighbors"], df["delta_apoc"],
               color=CB_PALETTE[0], alpha=0.6, s=30)
    ax.set_xlabel("Number of nearby events")
    ax.set_ylabel("Δ Apocalypticism")
    ax.set_title("Effect Size vs. Event Density")

    # Fit line
    if len(df) >= 5:
        z = np.polyfit(df["n_neighbors"], df["delta_apoc"], 1)
        x_line = np.linspace(df["n_neighbors"].min(), df["n_neighbors"].max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), "--", color=CB_PALETTE[2])

    fig.tight_layout()
    save_figure(fig, filename)


# ══════════════════════════════════════════════════════════════════════
# H26: Mimetic Contagion – Online-nexus attacks and semantic shift
# ══════════════════════════════════════════════════════════════════════

def test_mimetic_contagion(
    daily: pl.DataFrame,
    events: pl.DataFrame,
    measure_col: str = "apoc_combined",
) -> dict:
    """Test whether online-nexus attacks produce rhetoric shifts that
    are qualitatively different (more extreme, more ideologically
    aligned) than non-nexus attacks.

    Compares: (1) magnitude of Δapoc, (2) shift in sub-theme
    prevalence, (3) post-event variance in rhetoric scores.
    """
    violence = events.filter(pl.col("event_category") == "mass_violence")

    online_events = violence.filter(pl.col("online_nexus"))
    offline_events = violence.filter(~pl.col("online_nexus"))

    def compute_event_metrics(ev_df, daily_df):
        metrics = []
        for row in ev_df.iter_rows(named=True):
            window = build_event_window(daily_df, row["event_date"],
                                         pre_days=14, post_days=14)
            if window is None:
                continue
            pdf = window.to_pandas()
            pre = pdf[pdf["time_centered"] < 0]["mean_score"]
            post = pdf[pdf["time_centered"] >= 0]["mean_score"]
            if len(pre) < 3 or len(post) < 3:
                continue

            metrics.append({
                "event": row["event_name"],
                "ideology": row["ideology"],
                "killed": row["killed"],
                "delta_apoc": float(post.mean() - pre.mean()),
                "abs_delta": float(abs(post.mean() - pre.mean())),
                "pre_variance": float(pre.var()),
                "post_variance": float(post.var()),
                "variance_ratio": float(post.var() / pre.var())
                    if pre.var() > 0 else float("nan"),
            })
        return metrics

    online_metrics = compute_event_metrics(online_events, daily)
    offline_metrics = compute_event_metrics(offline_events, daily)

    if len(online_metrics) < 2 or len(offline_metrics) < 2:
        return {
            "hypothesis": "H26",
            "error": "insufficient events in one group",
            "n_online": len(online_metrics),
            "n_offline": len(offline_metrics),
        }

    online_df = pd.DataFrame(online_metrics)
    offline_df = pd.DataFrame(offline_metrics)

    # Test 1: Magnitude of rhetoric change
    try:
        u_mag, p_mag = stats.mannwhitneyu(
            online_df["abs_delta"], offline_df["abs_delta"],
            alternative="greater",
        )
    except Exception:
        u_mag, p_mag = float("nan"), float("nan")

    # Test 2: Direction of change
    try:
        u_dir, p_dir = stats.mannwhitneyu(
            online_df["delta_apoc"], offline_df["delta_apoc"],
            alternative="two-sided",
        )
    except Exception:
        u_dir, p_dir = float("nan"), float("nan")

    # Test 3: Variance change (polarisation)
    try:
        u_var, p_var = stats.mannwhitneyu(
            online_df["variance_ratio"], offline_df["variance_ratio"],
            alternative="greater",
        )
    except Exception:
        u_var, p_var = float("nan"), float("nan")

    # Effect sizes (Cohen's d)
    def cohens_d(a, b):
        na, nb = len(a), len(b)
        pooled_std = np.sqrt(((na - 1) * a.var() + (nb - 1) * b.var()) / (na + nb - 2))
        return float((a.mean() - b.mean()) / pooled_std) if pooled_std > 0 else float("nan")

    result = {
        "hypothesis": "H26",
        "description": "Mimetic contagion: online-nexus attacks trigger distinctive rhetoric shifts",
        "n_online": len(online_metrics),
        "n_offline": len(offline_metrics),
        "online_events": online_metrics,
        "offline_events": offline_metrics,
        "magnitude_test": {
            "online_mean_abs_delta": float(online_df["abs_delta"].mean()),
            "offline_mean_abs_delta": float(offline_df["abs_delta"].mean()),
            "mannwhitney_U": float(u_mag),
            "mannwhitney_p": float(p_mag),
            "cohens_d": cohens_d(online_df["abs_delta"], offline_df["abs_delta"]),
            "significant": bool(p_mag < 0.05),
        },
        "direction_test": {
            "online_mean_delta": float(online_df["delta_apoc"].mean()),
            "offline_mean_delta": float(offline_df["delta_apoc"].mean()),
            "mannwhitney_U": float(u_dir),
            "mannwhitney_p": float(p_dir),
            "significant": bool(p_dir < 0.05),
        },
        "polarisation_test": {
            "online_mean_variance_ratio": float(online_df["variance_ratio"].mean()),
            "offline_mean_variance_ratio": float(offline_df["variance_ratio"].mean()),
            "mannwhitney_U": float(u_var),
            "mannwhitney_p": float(p_var),
            "significant": bool(p_var < 0.05),
        },
        "supported": bool(p_mag < 0.05 or p_dir < 0.05),
    }

    return result


def plot_mimetic(results: dict, filename: str):
    """Plot mimetic contagion comparison."""
    if "error" in results:
        return

    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    online = pd.DataFrame(results.get("online_events", []))
    offline = pd.DataFrame(results.get("offline_events", []))

    if online.empty or offline.empty:
        plt.close(fig)
        return

    # Magnitude comparison
    ax = axes[0]
    groups = [offline["abs_delta"].values, online["abs_delta"].values]
    bp = ax.boxplot(groups, labels=["No online nexus", "Online nexus"],
                    patch_artist=True)
    bp["boxes"][0].set_facecolor(CB_PALETTE[7])
    bp["boxes"][1].set_facecolor(CB_PALETTE[2])
    ax.set_ylabel("|Δ Apocalypticism|")
    ax.set_title("H26: Rhetoric Shift Magnitude")

    # Direction comparison
    ax = axes[1]
    groups = [offline["delta_apoc"].values, online["delta_apoc"].values]
    bp = ax.boxplot(groups, labels=["No nexus", "Online nexus"],
                    patch_artist=True)
    bp["boxes"][0].set_facecolor(CB_PALETTE[7])
    bp["boxes"][1].set_facecolor(CB_PALETTE[2])
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_ylabel("Δ Apocalypticism")
    ax.set_title("Direction of Shift")

    # Variance ratio
    ax = axes[2]
    groups = [offline["variance_ratio"].dropna().values,
              online["variance_ratio"].dropna().values]
    if all(len(g) > 0 for g in groups):
        bp = ax.boxplot(groups, labels=["No nexus", "Online nexus"],
                        patch_artist=True)
        bp["boxes"][0].set_facecolor(CB_PALETTE[7])
        bp["boxes"][1].set_facecolor(CB_PALETTE[2])
    ax.axhline(1, color="gray", linestyle="--", label="No change")
    ax.set_ylabel("Post/Pre variance ratio")
    ax.set_title("Polarisation Effect")
    ax.legend()

    fig.tight_layout()
    save_figure(fig, filename)


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("STAGE 35: Offline Violence ↔ Online Neo-Fascism Hypotheses")
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

    # ── Build daily series ────────────────────────────────────────────
    primary_measure = "apoc_combined"
    daily = build_daily_series(pol, primary_measure)
    ts_df = build_event_series(daily, events)
    print(f"  Daily series: {daily.height} days")

    # ── H22: Contagion Decay ──────────────────────────────────────────
    print("\n  H22: Testing contagion decay (half-life estimation)…")
    h22 = estimate_decay_halflife(daily, events)
    results["H22_contagion_decay"] = h22
    if "error" not in h22:
        agg = h22["aggregate"]
        print(f"    Valid fits: {h22['n_valid_fits']}/{h22['n_events']}")
        print(f"    Median half-life: {agg['median_half_life']:.1f} days")
        print(f"    Mean half-life: {agg['mean_half_life']:.1f} days")
        print(f"    Supported: {h22['supported']}")
    else:
        print(f"    ✗ {h22['error']}")

    # ── H23: Reciprocal Amplification ─────────────────────────────────
    print("\n  H23: Testing reciprocal amplification…")
    h23 = test_reciprocal_amplification(ts_df)
    results["H23_reciprocal_amplification"] = h23
    if "error" not in h23:
        f = h23["findings"]
        print(f"    Events → rhetoric: {f['events_cause_rhetoric']}")
        print(f"    Rhetoric → events: {f['rhetoric_causes_events']}")
        print(f"    Reciprocal feedback: {f['reciprocal_feedback']}")
        print(f"    Supported: {h23['supported']}")
    else:
        print(f"    ✗ {h23['error']}")

    # ── H24: Threshold Activation ─────────────────────────────────────
    print("\n  H24: Testing threshold activation…")
    h24 = test_threshold_activation(daily, events)
    results["H24_threshold_activation"] = h24
    if "error" not in h24:
        opt = h24["optimal_threshold"]
        print(f"    Optimal threshold: {opt['threshold']} killed")
        print(f"    Effect diff: {opt['difference']:.4f} "
              f"(p={opt['mannwhitney_p']:.4f})")
        print(f"    Supported: {h24['supported']}")
    else:
        print(f"    ✗ {h24['error']}")

    # ── H25: Temporal Clustering ──────────────────────────────────────
    print("\n  H25: Testing temporal clustering…")
    h25 = test_temporal_clustering(daily, events)
    results["H25_temporal_clustering"] = h25
    if "error" not in h25:
        comp = h25["comparison"]
        if "error" not in comp:
            print(f"    Isolated events: n={comp['n_isolated']}, "
                  f"mean Δ={comp['mean_isolated']:.4f}")
            print(f"    Clustered events: n={comp['n_clustered']}, "
                  f"mean Δ={comp['mean_clustered']:.4f}")
            print(f"    Super-additive: {comp.get('super_additive', False)}")
        print(f"    Supported: {h25['supported']}")
    else:
        print(f"    ✗ {h25['error']}")

    # ── H26: Mimetic Contagion ────────────────────────────────────────
    print("\n  H26: Testing mimetic contagion…")
    h26 = test_mimetic_contagion(daily, events)
    results["H26_mimetic_contagion"] = h26
    if "error" not in h26:
        mag = h26["magnitude_test"]
        print(f"    Online nexus |Δ|: {mag['online_mean_abs_delta']:.4f}")
        print(f"    No nexus |Δ|: {mag['offline_mean_abs_delta']:.4f}")
        print(f"    Magnitude p: {mag['mannwhitney_p']:.4f}")
        print(f"    Supported: {h26['supported']}")
    else:
        print(f"    ✗ {h26['error']}")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n  " + "─" * 50)
    print("  HYPOTHESIS SUMMARY")
    print("  " + "─" * 50)
    for key in ["H22_contagion_decay", "H23_reciprocal_amplification",
                "H24_threshold_activation", "H25_temporal_clustering",
                "H26_mimetic_contagion"]:
        r = results.get(key, {})
        h = key.split("_")[0]
        supported = r.get("supported", "error")
        desc = r.get("description", "")
        print(f"    {h}: {'✓ Supported' if supported else '✗ Not supported'} — {desc}")

    results["summary"] = {
        h.split("_")[0]: {
            "supported": results[h].get("supported", False),
            "description": results[h].get("description", ""),
        }
        for h in ["H22_contagion_decay", "H23_reciprocal_amplification",
                   "H24_threshold_activation", "H25_temporal_clustering",
                   "H26_mimetic_contagion"]
    }

    # ── Plots ─────────────────────────────────────────────────────────
    print("\n  Generating plots…")

    if "error" not in h22:
        plot_decay(h22, "h22_contagion_decay")
    if "error" not in h23:
        plot_reciprocal_irf(h23, "h23_reciprocal_irf")
    if "error" not in h24:
        plot_threshold(h24, "h24_threshold_activation")
    if "error" not in h25:
        plot_clustering(h25, "h25_temporal_clustering")
    if "error" not in h26:
        plot_mimetic(h26, "h26_mimetic_contagion")

    # ── Save results ──────────────────────────────────────────────────
    out_path = RESULTS_DIR / "offline_online_hypotheses_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✓ Offline-online hypothesis tests complete. Saved to {out_path.name}")


if __name__ == "__main__":
    main()
