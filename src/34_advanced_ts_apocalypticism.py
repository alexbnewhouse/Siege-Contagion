"""34 – Advanced Time-Series Analyses of Apocalypticism.

Extends the ITS findings from Stage 31 with additional time-series
methods to triangulate the relationship between mass-casualty events
and apocalyptic rhetoric on /pol/.

Methods
-------
1. **Vector Autoregression (VAR)** – Jointly models apocalyptic rhetoric
   and a binary event indicator as an endogenous system, estimating
   impulse response functions (IRF) and forecast error variance
   decomposition (FEVD).

2. **Autoregressive Distributed Lag (ARDL)** – Tests long-run
   cointegration between event intensity (daily casualty counts) and
   apocalyptic rhetoric via the Pesaran bounds test.  Estimates both
   short-run dynamics and the error-correction term.

3. **Bayesian Structural Time Series (BSTS)** – Uses a local-level or
   local-linear-trend state-space model with event intervention
   components to estimate the causal impact of mass-casualty events,
   with posterior credible intervals.

4. **Local Projections (Jordà, 2005)** – Non-parametric impulse
   responses estimated via sequential OLS at horizons h = 0, …, 30,
   providing robust estimates that don't require correct VAR lag
   specification.

5. **Method Comparison** – Formal comparison matrix of all methods
   (ITS, VAR, ARDL, BSTS, LP) on direction, magnitude, significance,
   and robustness.

Output
------
``results/advanced_ts_results.json``
``figures/adv_ts_*.{png,pdf}``
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
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from scipy import stats

from utils import (
    DATA_PROCESSED, RESULTS_DIR, FIGURES_DIR,
    setup_plot_style, save_figure, FIGSIZE_WIDE, CB_PALETTE,
)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Import from Stage 31 ─────────────────────────────────────────────
_its = importlib.import_module("31_apocalypticism_its")
build_daily_series = _its.build_daily_series
WINDOW_PRE_DAYS = _its.WINDOW_PRE_DAYS
WINDOW_POST_DAYS = _its.WINDOW_POST_DAYS
MIN_DAILY_POSTS = _its.MIN_DAILY_POSTS


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def build_event_series(
    daily: pl.DataFrame,
    events: pl.DataFrame,
) -> pd.DataFrame:
    """Build aligned daily DataFrame with apocalypticism + event indicators.

    Returns a pandas DataFrame indexed by date with columns:
      - apoc_mean: daily mean apocalypticism score
      - apoc_prevalence: daily apocalypticism prevalence
      - event_occurred: binary indicator (1 if event on that day)
      - event_casualties: total casualties on that day (0 if none)
      - post_count: number of /pol/ posts
    """
    pdf = daily.to_pandas()
    pdf["day"] = pd.to_datetime(pdf["day"])
    pdf = pdf.set_index("day").sort_index()

    # Fill missing days
    full_idx = pd.date_range(pdf.index.min(), pdf.index.max(), freq="D")
    pdf = pdf.reindex(full_idx)
    pdf["post_count"] = pdf["post_count"].fillna(0)
    pdf["mean_score"] = pdf["mean_score"].interpolate(method="linear")
    pdf["apoc_prevalence"] = pdf["apoc_prevalence"].interpolate(method="linear")

    # Event indicators
    event_dates = set()
    event_casualties = {}
    for row in events.iter_rows(named=True):
        ed = row["event_date"]
        if isinstance(ed, datetime.date) and not isinstance(ed, datetime.datetime):
            ed = datetime.datetime.combine(ed, datetime.time())
        ed = pd.Timestamp(ed)
        event_dates.add(ed)
        event_casualties[ed] = event_casualties.get(ed, 0) + row.get("killed", 0) + row.get("injured", 0)

    pdf["event_occurred"] = pdf.index.isin(event_dates).astype(int)
    pdf["event_casualties"] = pdf.index.map(
        lambda d: event_casualties.get(d, 0)
    ).astype(float)

    pdf = pdf.rename(columns={"mean_score": "apoc_mean"})
    pdf.index.name = "date"

    # Drop rows with NaN apocalypticism
    pdf = pdf.dropna(subset=["apoc_mean"])

    return pdf[["apoc_mean", "apoc_prevalence", "event_occurred",
                "event_casualties", "post_count"]]


def stationarity_tests(series: pd.Series, name: str) -> dict:
    """Run ADF test on a series and return results."""
    clean = series.dropna()
    if len(clean) < 20:
        return {"name": name, "error": "insufficient data"}
    try:
        adf_result = adfuller(clean, autolag="AIC")  # type: ignore[assignment]
        # adfuller returns (adf_stat, pvalue, usedlag, nobs, critical_values, icbest)
        adf_stat = float(adf_result[0])
        p_val = float(adf_result[1])
        lags = int(adf_result[2])  # type: ignore[arg-type]
        nobs = int(adf_result[3])  # type: ignore[arg-type]
        crit: dict = adf_result[4]  # type: ignore[assignment]
        return {
            "name": name,
            "adf_statistic": adf_stat,
            "p_value": p_val,
            "lags_used": lags,
            "n_obs": nobs,
            "critical_values": {k: float(v) for k, v in crit.items()},
            "stationary_at_5pct": bool(p_val < 0.05),
        }
    except Exception as e:
        return {"name": name, "error": str(e)}


# ══════════════════════════════════════════════════════════════════════
# 1. Vector Autoregression (VAR)
# ══════════════════════════════════════════════════════════════════════

def run_var_analysis(
    ts_df: pd.DataFrame,
    maxlag: int = 14,
    irf_periods: int = 30,
) -> dict:
    """Fit VAR model on (apoc_mean, event_occurred) and compute IRF/FEVD.

    Parameters
    ----------
    ts_df : pd.DataFrame
        Daily time series with 'apoc_mean' and 'event_occurred'.
    maxlag : int
        Maximum lag order for model selection.
    irf_periods : int
        Number of periods for impulse response functions.

    Returns
    -------
    dict with VAR results, Granger tests, IRF, and FEVD.
    """
    data = ts_df[["apoc_mean", "event_occurred"]].dropna()

    if len(data) < 50:
        return {"error": "insufficient data for VAR"}

    # Stationarity
    stationarity = {
        "apoc_mean": stationarity_tests(data["apoc_mean"], "apoc_mean"),
        "event_occurred": stationarity_tests(data["event_occurred"], "event_occurred"),
    }

    try:
        model = VARModel(data)
        # Select optimal lag via AIC
        lag_order = model.select_order(maxlags=min(maxlag, len(data) // 5 - 1))
        selected_lag = lag_order.aic
        if selected_lag < 1:
            selected_lag = 1

        fitted = model.fit(maxlags=selected_lag, ic="aic")
    except Exception as e:
        return {"error": f"VAR fitting failed: {e}", "stationarity": stationarity}

    # Granger causality (within VAR framework)
    granger = {}
    try:
        gc_event_to_apoc = fitted.test_causality("apoc_mean", ["event_occurred"])
        granger["event_causes_apoc"] = {
            "test_statistic": float(gc_event_to_apoc.test_statistic),
            "p_value": float(gc_event_to_apoc.pvalue),
            "df": int(gc_event_to_apoc.df[0]),
            "significant_at_05": bool(gc_event_to_apoc.pvalue < 0.05),
        }
    except Exception as e:
        granger["event_causes_apoc"] = {"error": str(e)}

    try:
        gc_apoc_to_event = fitted.test_causality("event_occurred", ["apoc_mean"])
        granger["apoc_causes_event"] = {
            "test_statistic": float(gc_apoc_to_event.test_statistic),
            "p_value": float(gc_apoc_to_event.pvalue),
            "df": int(gc_apoc_to_event.df[0]),
            "significant_at_05": bool(gc_apoc_to_event.pvalue < 0.05),
        }
    except Exception as e:
        granger["apoc_causes_event"] = {"error": str(e)}

    # Impulse Response Functions
    irf_result = {}
    try:
        irf = fitted.irf(irf_periods)
        # Response of apoc_mean to event_occurred shock
        irf_apoc = irf.irfs[:, 0, 1]  # response of var 0 to shock in var 1

        irf_result = {
            "response_of_apoc_to_event": [float(x) for x in irf_apoc],
            "cumulative_response": [float(x) for x in np.cumsum(irf_apoc)],
            "peak_response": float(np.max(np.abs(irf_apoc))),
            "peak_period": int(np.argmax(np.abs(irf_apoc))),
        }

        # Monte Carlo confidence intervals
        try:
            mc_lower, mc_upper = irf.errband_mc(repl=500, signif=0.05)
            irf_result["ci_lower"] = [float(x) for x in mc_lower[:, 0, 1]]
            irf_result["ci_upper"] = [float(x) for x in mc_upper[:, 0, 1]]
        except Exception:
            pass  # CIs are optional
    except Exception as e:
        irf_result = {"error": str(e)}

    # Forecast Error Variance Decomposition
    fevd_result = {}
    try:
        fevd = fitted.fevd(irf_periods)
        # % of apoc_mean forecast variance explained by event_occurred
        fevd_decomp = fevd.decomp[0]  # for apoc_mean
        fevd_result = {
            "pct_apoc_explained_by_event": [float(x) for x in fevd_decomp[:, 1]],
            "pct_at_horizon_1": float(fevd_decomp[0, 1]),
            "pct_at_horizon_7": float(fevd_decomp[min(6, len(fevd_decomp) - 1), 1]),
            "pct_at_horizon_14": float(fevd_decomp[min(13, len(fevd_decomp) - 1), 1]),
            "pct_at_horizon_30": float(fevd_decomp[min(29, len(fevd_decomp) - 1), 1]),
        }
    except Exception as e:
        fevd_result = {"error": str(e)}

    result = {
        "method": "VAR",
        "n_obs": int(len(data)),
        "selected_lag": int(selected_lag),
        "lag_selection_criteria": {
            "aic": int(lag_order.aic),
            "bic": int(lag_order.bic),
            "hqic": int(lag_order.hqic),
        },
        "stationarity": stationarity,
        "granger_causality": granger,
        "irf": irf_result,
        "fevd": fevd_result,
        "aic": float(fitted.aic),
        "bic": float(fitted.bic),
    }

    return result


def plot_irf(irf_data: dict, filename: str):
    """Plot impulse response function."""
    if "error" in irf_data or "response_of_apoc_to_event" not in irf_data:
        return

    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    response = irf_data["response_of_apoc_to_event"]
    horizons = range(len(response))

    # IRF
    ax = axes[0]
    ax.plot(horizons, response, color=CB_PALETTE[0], linewidth=2)
    if "ci_lower" in irf_data:
        ax.fill_between(horizons, irf_data["ci_lower"], irf_data["ci_upper"],
                        alpha=0.2, color=CB_PALETTE[0])
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Days after shock")
    ax.set_ylabel("Response of apocalypticism")
    ax.set_title("IRF: Apocalypticism ← Event Shock")

    # Cumulative IRF
    ax = axes[1]
    cum = irf_data["cumulative_response"]
    ax.plot(horizons, cum, color=CB_PALETTE[2], linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Days after shock")
    ax.set_ylabel("Cumulative response")
    ax.set_title("Cumulative IRF")

    fig.tight_layout()
    save_figure(fig, filename)


def plot_fevd(fevd_data: dict, filename: str):
    """Plot forecast error variance decomposition."""
    if "error" in fevd_data or "pct_apoc_explained_by_event" not in fevd_data:
        return

    setup_plot_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    pcts = fevd_data["pct_apoc_explained_by_event"]
    horizons = range(1, len(pcts) + 1)

    ax.bar(horizons, [p * 100 for p in pcts], color=CB_PALETTE[0], alpha=0.7)
    ax.set_xlabel("Forecast horizon (days)")
    ax.set_ylabel("% of apocalypticism variance\nexplained by events")
    ax.set_title("FEVD: Share of Apocalypticism Variance Explained by Mass-Casualty Events")

    save_figure(fig, filename)


# ══════════════════════════════════════════════════════════════════════
# 2. Autoregressive Distributed Lag (ARDL)
# ══════════════════════════════════════════════════════════════════════

def run_ardl_analysis(
    ts_df: pd.DataFrame,
    max_ar_lag: int = 7,
    max_dl_lag: int = 7,
) -> dict:
    """Fit ARDL model and test for long-run cointegration.

    Uses the Pesaran bounds test approach:
      Y_t = α + Σᵢ φᵢ Y_{t-i} + Σⱼ θⱼ X_{t-j} + ε_t

    Then estimates the error-correction form (ECM) to separate
    short-run dynamics from long-run equilibrium.
    """
    data = ts_df[["apoc_mean", "event_casualties"]].dropna()

    if len(data) < 50:
        return {"error": "insufficient data for ARDL"}

    y = data["apoc_mean"].values
    x = data["event_casualties"].values

    # Select lag orders via AIC
    best_aic = np.inf
    best_p, best_q = 1, 1
    best_model = None

    for p in range(1, max_ar_lag + 1):
        for q in range(0, max_dl_lag + 1):
            try:
                endog = y[max(p, q):]
                exog_cols = []
                for i in range(1, p + 1):
                    exog_cols.append(y[max(p, q) - i: len(y) - i])
                for j in range(0, q + 1):
                    exog_cols.append(x[max(p, q) - j: len(x) - j])

                X_mat = np.column_stack(exog_cols)
                X_mat = sm.add_constant(X_mat)

                model = sm.OLS(endog, X_mat).fit()
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_p, best_q = p, q
                    best_model = model
            except Exception:
                continue

    if best_model is None:
        return {"error": "ARDL model fitting failed"}

    # Extract coefficients
    n_ar = best_p
    n_dl = best_q + 1  # includes lag 0

    ar_coeffs = [float(best_model.params[1 + i]) for i in range(n_ar)]
    dl_coeffs = [float(best_model.params[1 + n_ar + j]) for j in range(n_dl)]

    # Long-run multiplier: Σθⱼ / (1 - Σφᵢ)
    sum_ar = sum(ar_coeffs)
    sum_dl = sum(dl_coeffs)
    long_run_multiplier = sum_dl / (1 - sum_ar) if abs(1 - sum_ar) > 1e-10 else float("nan")

    # Bounds test: F-test on joint significance of lagged levels
    # (simplified — tests whether lagged level terms are jointly significant)
    try:
        # Reconstruct ECM form
        dy = np.diff(np.asarray(y, dtype=float))
        endog_ecm = dy[max(best_p, best_q):]

        ecm_cols = []
        # Lagged level of Y
        ecm_cols.append(np.asarray(y[max(best_p, best_q): -1], dtype=float))
        # Lagged level of X
        ecm_cols.append(np.asarray(x[max(best_p, best_q): -1], dtype=float))
        # Lagged differences of Y
        for i in range(1, best_p):
            ecm_cols.append(dy[max(best_p, best_q) - i: len(dy) - i])
        # Current and lagged differences of X
        dx = np.diff(np.asarray(x, dtype=float))
        for j in range(0, best_q):
            ecm_cols.append(dx[max(best_p, best_q) - j: len(dx) - j])

        if len(ecm_cols) > 0 and all(len(c) == len(endog_ecm) for c in ecm_cols):
            X_ecm = np.column_stack(ecm_cols)
            X_ecm = sm.add_constant(X_ecm)

            ecm_model = sm.OLS(endog_ecm, X_ecm).fit()

            # Error correction coefficient (speed of adjustment)
            ec_coeff = float(ecm_model.params[1])
            ec_pval = float(ecm_model.pvalues[1])

            # Bounds test F-statistic (joint test on lagged levels)
            r_matrix = np.zeros((2, X_ecm.shape[1]))
            r_matrix[0, 1] = 1  # coefficient on lagged Y level
            r_matrix[1, 2] = 1  # coefficient on lagged X level
            try:
                f_test = ecm_model.f_test(r_matrix)
                bounds_f = float(f_test.fvalue.item() if hasattr(f_test.fvalue, 'item') else f_test.fvalue)
                bounds_p = float(f_test.pvalue.item() if hasattr(f_test.pvalue, 'item') else f_test.pvalue)
            except Exception:
                bounds_f = float("nan")
                bounds_p = float("nan")

            ecm_result = {
                "ec_coefficient": ec_coeff,
                "ec_p_value": ec_pval,
                "ec_significant": bool(ec_pval < 0.05 and ec_coeff < 0),
                "bounds_F": bounds_f,
                "bounds_p": bounds_p,
                "ecm_r_squared": float(ecm_model.rsquared),
            }
        else:
            ecm_result = {"error": "ECM construction failed (dimension mismatch)"}
    except Exception as e:
        ecm_result = {"error": f"ECM failed: {e}"}

    result = {
        "method": "ARDL",
        "n_obs": int(len(data)),
        "ar_order": int(best_p),
        "dl_order": int(best_q),
        "aic": float(best_aic),
        "r_squared": float(best_model.rsquared),
        "ar_coefficients": ar_coeffs,
        "dl_coefficients": dl_coeffs,
        "long_run_multiplier": float(long_run_multiplier),
        "ecm": ecm_result,
    }

    return result


# ══════════════════════════════════════════════════════════════════════
# 3. Bayesian Structural Time Series (BSTS)
# ══════════════════════════════════════════════════════════════════════

def run_bsts_analysis(
    ts_df: pd.DataFrame,
    events: pl.DataFrame,
    n_post_days: int = 30,
) -> dict:
    """Bayesian structural time series causal impact estimation.

    Uses a local-level state-space model with the Kalman filter.
    For each event, estimates the counterfactual (what would have
    happened without the event) and compares to observed.

    This is a simplified BSTS using statsmodels UnobservedComponents
    (since CausalImpact requires R/rpy2).
    """
    from statsmodels.tsa.statespace.structural import UnobservedComponents

    data = ts_df[["apoc_mean"]].dropna()

    if len(data) < 60:
        return {"error": "insufficient data for BSTS"}

    event_dates = []
    for row in events.iter_rows(named=True):
        ed = row["event_date"]
        if isinstance(ed, datetime.date) and not isinstance(ed, datetime.datetime):
            ed = datetime.datetime.combine(ed, datetime.time())
        event_dates.append(pd.Timestamp(ed))

    # Filter to events within data range
    event_dates = [d for d in event_dates if d in data.index]

    if not event_dates:
        # Try matching to nearest date
        all_dates = data.index
        matched = []
        for ed in events["event_date"].to_list():
            ed_ts = pd.Timestamp(ed)
            diffs = abs(all_dates - ed_ts)  # type: ignore[operator]
            nearest = all_dates[diffs.argmin()]
            if abs((nearest - ed_ts).days) <= 1:  # type: ignore[operator]
                matched.append(nearest)
        event_dates = matched

    if not event_dates:
        return {"error": "no events found within data range"}

    # Fit local-level + stochastic trend model on pre-event period
    # Use earliest event for primary analysis
    event_dates_sorted = sorted(event_dates)

    per_event_bsts = []
    for event_ts in event_dates_sorted[:20]:  # Limit to 20 events
        try:
            # Pre-event data for model training
            pre_start = event_ts - pd.Timedelta(days=WINDOW_PRE_DAYS * 2)
            pre_data = data.loc[pre_start:event_ts - pd.Timedelta(days=1), "apoc_mean"]
            post_data = data.loc[event_ts:event_ts + pd.Timedelta(days=n_post_days), "apoc_mean"]

            if len(pre_data) < 20 or len(post_data) < 5:
                continue

            # Fit structural model on pre-event data
            model = UnobservedComponents(
                pre_data.values,
                level="local linear trend",  # type: ignore[arg-type]
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = model.fit(disp=False, maxiter=200)

            # Forecast counterfactual
            forecast = fitted.get_forecast(steps=len(post_data))  # type: ignore[union-attr]
            predicted_mean = forecast.predicted_mean
            predicted_ci = forecast.conf_int(alpha=0.05)

            # Causal impact = observed - predicted
            observed = post_data.values
            pointwise_impact = observed - predicted_mean
            cumulative_impact = float(np.sum(pointwise_impact))
            avg_impact = float(np.mean(pointwise_impact))

            # Posterior probability that effect is positive
            # (fraction of post-period where observed > predicted)
            prob_positive = float(np.mean(pointwise_impact > 0))

            per_event_bsts.append({
                "event_date": str(event_ts.date()),
                "n_pre": int(len(pre_data)),
                "n_post": int(len(post_data)),
                "avg_observed": float(np.mean(observed)),  # type: ignore[arg-type]
                "avg_predicted": float(np.mean(predicted_mean)),
                "avg_impact": avg_impact,
                "cumulative_impact": cumulative_impact,
                "prob_positive_effect": prob_positive,
                "relative_effect": float(avg_impact / np.mean(predicted_mean))
                    if np.mean(predicted_mean) != 0 else float("nan"),
            })

        except Exception:
            continue

    if not per_event_bsts:
        return {"error": "BSTS failed for all events"}

    # Aggregate across events
    avg_impacts = [r["avg_impact"] for r in per_event_bsts]
    cum_impacts = [r["cumulative_impact"] for r in per_event_bsts]

    # One-sample t-test: is the average impact significantly different from 0?
    if len(avg_impacts) >= 3:
        t_result = stats.ttest_1samp(avg_impacts, 0)
        t_stat_val = float(t_result.statistic)  # type: ignore[union-attr]
        t_pval_val = float(t_result.pvalue)  # type: ignore[union-attr]
    else:
        t_stat_val, t_pval_val = float("nan"), float("nan")

    result = {
        "method": "BSTS",
        "n_events_analyzed": len(per_event_bsts),
        "per_event": per_event_bsts,
        "aggregate": {
            "mean_impact": float(np.mean(avg_impacts)),
            "median_impact": float(np.median(avg_impacts)),
            "std_impact": float(np.std(avg_impacts)),
            "mean_cumulative_impact": float(np.mean(cum_impacts)),
            "t_statistic": t_stat_val,
            "t_p_value": t_pval_val,
            "significant_at_05": bool(t_pval_val < 0.05) if not np.isnan(t_pval_val) else False,
            "pct_positive": float(np.mean([r["prob_positive_effect"] for r in per_event_bsts])),
        },
    }

    return result


# ══════════════════════════════════════════════════════════════════════
# 4. Local Projections (Jordà, 2005)
# ══════════════════════════════════════════════════════════════════════

def run_local_projections(
    ts_df: pd.DataFrame,
    max_horizon: int = 30,
    n_lags: int = 7,
) -> dict:
    """Estimate impulse responses via local projections.

    For each horizon h = 0, …, max_horizon:
      Y_{t+h} = α_h + β_h · event_t + Σ γ_{h,k} Y_{t-k} + ε_{t+h}

    β_h gives the response at horizon h to an event at time t.
    Uses Newey-West HAC standard errors with bandwidth = h + 1.
    """
    data = ts_df[["apoc_mean", "event_occurred"]].dropna()

    if len(data) < 50:
        return {"error": "insufficient data for local projections"}

    y = data["apoc_mean"].values
    event = data["event_occurred"].values

    results_by_horizon = []

    for h in range(max_horizon + 1):
        # Build regression for horizon h
        # Y_{t+h} = α + β·event_t + Σ γ_k·Y_{t-k}
        max_offset = max(n_lags, h)
        if max_offset >= len(y) - 1:
            break

        y_fwd = y[max_offset + h:]
        event_t = event[max_offset: len(y) - h]
        n = min(len(y_fwd), len(event_t))
        y_fwd = y_fwd[:n]
        event_t = event_t[:n]

        # Lagged controls
        controls = []
        for k in range(1, n_lags + 1):
            lag = y[max_offset - k: max_offset - k + n]
            controls.append(lag)

        if not controls:
            continue

        try:
            X = np.column_stack([np.asarray(event_t)] + [np.asarray(c) for c in controls])
            X = sm.add_constant(X)

            # Newey-West with bandwidth h+1
            nw_lags = max(1, h + 1)
            model = sm.OLS(y_fwd, X).fit(
                cov_type="HAC", cov_kwds={"maxlags": nw_lags}
            )

            results_by_horizon.append({
                "horizon": h,
                "beta": float(model.params[1]),
                "se": float(model.bse[1]),
                "t_stat": float(model.tvalues[1]),
                "p_value": float(model.pvalues[1]),
                "ci_lo": float(model.conf_int()[1, 0]),
                "ci_hi": float(model.conf_int()[1, 1]),
                "n_obs": int(n),
            })
        except Exception:
            continue

    if not results_by_horizon:
        return {"error": "no valid horizons estimated"}

    # Find peak effect
    betas = [r["beta"] for r in results_by_horizon]
    abs_betas = [abs(b) for b in betas]
    peak_idx = int(np.argmax(abs_betas))

    result = {
        "method": "LocalProjections",
        "n_horizons": len(results_by_horizon),
        "n_lags_controlled": n_lags,
        "horizons": results_by_horizon,
        "peak_horizon": results_by_horizon[peak_idx]["horizon"],
        "peak_beta": betas[peak_idx],
        "peak_p_value": results_by_horizon[peak_idx]["p_value"],
        "significant_horizons": [r["horizon"] for r in results_by_horizon
                                  if r["p_value"] < 0.05],
        "n_significant": sum(1 for r in results_by_horizon if r["p_value"] < 0.05),
    }

    return result


def plot_local_projections(lp_results: dict, filename: str):
    """Plot local projection impulse response function."""
    if "error" in lp_results or "horizons" not in lp_results:
        return

    setup_plot_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    horizons = [r["horizon"] for r in lp_results["horizons"]]
    betas = [r["beta"] for r in lp_results["horizons"]]
    ci_lo = [r["ci_lo"] for r in lp_results["horizons"]]
    ci_hi = [r["ci_hi"] for r in lp_results["horizons"]]

    ax.plot(horizons, betas, color=CB_PALETTE[0], linewidth=2, marker="o",
            markersize=3)
    ax.fill_between(horizons, ci_lo, ci_hi, alpha=0.2, color=CB_PALETTE[0])
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Horizon (days)")
    ax.set_ylabel("Response of apocalypticism (β_h)")
    ax.set_title("Local Projections: Apocalypticism Response to Mass-Casualty Events")

    # Mark significant horizons
    sig_h = [r["horizon"] for r in lp_results["horizons"] if r["p_value"] < 0.05]
    sig_b = [r["beta"] for r in lp_results["horizons"] if r["p_value"] < 0.05]
    if sig_h:
        ax.scatter(sig_h, sig_b, color=CB_PALETTE[2], zorder=5, s=40,
                   label="p < 0.05")
        ax.legend()

    save_figure(fig, filename)


# ══════════════════════════════════════════════════════════════════════
# 5. Method Comparison
# ══════════════════════════════════════════════════════════════════════

def build_method_comparison(
    its_results: dict,
    var_results: dict,
    ardl_results: dict,
    bsts_results: dict,
    lp_results: dict,
) -> dict:
    """Build a comparison matrix across all methods."""
    methods = {}

    # ITS
    if "error" not in its_results:
        b = its_results.get("b_level", float("nan"))
        p = its_results.get("p_level", float("nan"))
        methods["ITS"] = {
            "direction": "positive" if b > 0 else "negative",
            "effect_size": float(b),
            "p_value": float(p),
            "significant": bool(p < 0.05) if not np.isnan(p) else False,
            "n_obs": its_results.get("n_obs", its_results.get("n_days", None)),
        }

    # VAR
    if "error" not in var_results:
        irf = var_results.get("irf", {})
        peak = irf.get("peak_response", float("nan"))
        gc = var_results.get("granger_causality", {}).get("event_causes_apoc", {})
        methods["VAR"] = {
            "direction": "positive" if peak > 0 else "negative",
            "effect_size": float(peak),
            "p_value": gc.get("p_value", float("nan")),
            "significant": gc.get("significant_at_05", False),
            "n_obs": var_results.get("n_obs"),
            "selected_lag": var_results.get("selected_lag"),
        }

    # ARDL
    if "error" not in ardl_results:
        lrm = ardl_results.get("long_run_multiplier", float("nan"))
        ecm = ardl_results.get("ecm", {})
        methods["ARDL"] = {
            "direction": "positive" if lrm > 0 else "negative",
            "effect_size": float(lrm),
            "p_value": ecm.get("ec_p_value", float("nan")),
            "significant": ecm.get("ec_significant", False),
            "n_obs": ardl_results.get("n_obs"),
            "ar_order": ardl_results.get("ar_order"),
            "dl_order": ardl_results.get("dl_order"),
        }

    # BSTS
    if "error" not in bsts_results:
        agg = bsts_results.get("aggregate", {})
        methods["BSTS"] = {
            "direction": "positive" if agg.get("mean_impact", 0) > 0 else "negative",
            "effect_size": agg.get("mean_impact", float("nan")),
            "p_value": agg.get("t_p_value", float("nan")),
            "significant": agg.get("significant_at_05", False),
            "n_events": bsts_results.get("n_events_analyzed"),
        }

    # Local Projections
    if "error" not in lp_results:
        peak_b = lp_results.get("peak_beta", float("nan"))
        peak_p = lp_results.get("peak_p_value", float("nan"))
        methods["LocalProjections"] = {
            "direction": "positive" if peak_b > 0 else "negative",
            "effect_size": float(peak_b),
            "p_value": float(peak_p),
            "significant": bool(peak_p < 0.05) if not np.isnan(peak_p) else False,
            "n_significant_horizons": lp_results.get("n_significant"),
            "peak_horizon": lp_results.get("peak_horizon"),
        }

    # Consensus
    directions = [m["direction"] for m in methods.values()
                  if not np.isnan(m["effect_size"])]
    sig_count = sum(1 for m in methods.values() if m.get("significant", False))

    consensus = {
        "n_methods": len(methods),
        "n_significant": sig_count,
        "direction_agreement": len(set(directions)) == 1 if directions else False,
        "consensus_direction": directions[0] if len(set(directions)) == 1 else "mixed",
        "conclusion": _derive_conclusion(methods),
    }

    return {"methods": methods, "consensus": consensus}


def _derive_conclusion(methods: dict) -> str:
    """Derive a summary conclusion from method comparison."""
    if not methods:
        return "No methods produced valid results."

    sig_methods = [k for k, v in methods.items() if v.get("significant")]
    directions = [v["direction"] for v in methods.values()
                  if not np.isnan(v["effect_size"])]

    if not sig_methods:
        return ("No method finds a significant effect of mass-casualty events "
                "on apocalyptic rhetoric. The null hypothesis of no effect "
                "cannot be rejected.")

    if len(set(directions)) == 1:
        d = directions[0]
        return (f"Methods converge on a {d} direction. "
                f"{len(sig_methods)}/{len(methods)} methods reach "
                f"significance: {', '.join(sig_methods)}.")

    return (f"Methods disagree on direction. {len(sig_methods)}/"
            f"{len(methods)} significant: {', '.join(sig_methods)}. "
            "Results are inconclusive.")


def plot_method_comparison(comparison: dict, filename: str):
    """Bar chart comparing effect sizes across methods."""
    methods = comparison.get("methods", {})
    if not methods:
        return

    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    names = list(methods.keys())
    effects = [methods[n]["effect_size"] for n in names]
    colors = [CB_PALETTE[2] if methods[n].get("significant") else CB_PALETTE[7]
              for n in names]

    bars = ax.barh(names, effects, color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Effect size")
    ax.set_title("Method Comparison: Effect of Mass-Casualty Events on Apocalypticism")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=CB_PALETTE[2], label="Significant (p < 0.05)"),
        Patch(facecolor=CB_PALETTE[7], label="Not significant"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    fig.tight_layout()
    save_figure(fig, filename)


# ══════════════════════════════════════════════════════════════════════
# Stratified analyses (violence vs non-violence)
# ══════════════════════════════════════════════════════════════════════

def run_stratified_var(
    ts_df: pd.DataFrame,
    events: pl.DataFrame,
) -> dict:
    """Run VAR separately for mass-violence and non-violence events."""
    results = {}
    for category in ["mass_violence", "nonviolence"]:
        if category == "mass_violence":
            subset = events.filter(pl.col("event_category") == "mass_violence")
        else:
            subset = events.filter(pl.col("event_category") != "mass_violence")

        if subset.height < 3:
            results[category] = {"error": f"too few events ({subset.height})"}
            continue

        sub_ts = build_event_series(
            pl.DataFrame({
                "day": pd.Series(ts_df.index).dt.date.tolist(),
                "mean_score": ts_df["apoc_mean"].values,
                "apoc_prevalence": ts_df["apoc_prevalence"].values,
                "post_count": ts_df["post_count"].values,
                "apoc_post_count": (ts_df["apoc_prevalence"] * ts_df["post_count"]).values,
            }).with_columns(pl.col("day").cast(pl.Date)),
            subset,
        )

        r = run_var_analysis(sub_ts)
        r["category"] = category
        r["n_events"] = subset.height
        results[category] = r

    return results


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("STAGE 34: Advanced Time-Series Analyses of Apocalypticism")
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
    print(f"  Daily series: {daily.height} days")

    # ── Build aligned time series ─────────────────────────────────────
    ts_df = build_event_series(daily, events)
    print(f"  Aligned time series: {len(ts_df)} days")
    print(f"  Event days: {ts_df['event_occurred'].sum():.0f}")

    results["n_posts"] = pol.height
    results["n_events"] = events.height
    results["n_days"] = len(ts_df)

    # ── Stationarity tests ────────────────────────────────────────────
    print("\n  Stationarity tests…")
    stationarity = {
        "apoc_mean": stationarity_tests(ts_df["apoc_mean"], "apoc_mean"),
        "event_occurred": stationarity_tests(ts_df["event_occurred"], "event_occurred"),
        "event_casualties": stationarity_tests(ts_df["event_casualties"], "event_casualties"),
    }
    results["stationarity"] = stationarity
    for name, s in stationarity.items():
        if "error" not in s:
            print(f"    {name}: ADF={s['adf_statistic']:.3f}, "
                  f"p={s['p_value']:.4f} "
                  f"({'stationary' if s['stationary_at_5pct'] else 'non-stationary'})")

    # ── 1. VAR ────────────────────────────────────────────────────────
    print("\n  Running VAR analysis…")
    var_results = run_var_analysis(ts_df)
    results["var"] = var_results
    if "error" not in var_results:
        gc = var_results["granger_causality"]
        if "error" not in gc.get("event_causes_apoc", {}):
            print(f"    Granger: events → apoc: "
                  f"p={gc['event_causes_apoc']['p_value']:.4f}")
        if "error" not in gc.get("apoc_causes_event", {}):
            print(f"    Granger: apoc → events: "
                  f"p={gc['apoc_causes_event']['p_value']:.4f}")
        irf = var_results.get("irf", {})
        if "error" not in irf:
            print(f"    IRF peak: {irf['peak_response']:.6f} "
                  f"at day {irf['peak_period']}")
    else:
        print(f"    ✗ {var_results['error']}")

    # ── 2. ARDL ───────────────────────────────────────────────────────
    print("\n  Running ARDL analysis…")
    ardl_results = run_ardl_analysis(ts_df)
    results["ardl"] = ardl_results
    if "error" not in ardl_results:
        print(f"    ARDL({ardl_results['ar_order']},{ardl_results['dl_order']})")
        print(f"    Long-run multiplier: {ardl_results['long_run_multiplier']:.6f}")
        ecm = ardl_results.get("ecm", {})
        if "error" not in ecm:
            print(f"    EC coefficient: {ecm['ec_coefficient']:.6f} "
                  f"(p={ecm['ec_p_value']:.4f})")
    else:
        print(f"    ✗ {ardl_results['error']}")

    # ── 3. BSTS ───────────────────────────────────────────────────────
    print("\n  Running BSTS analysis…")
    bsts_results = run_bsts_analysis(ts_df, events)
    results["bsts"] = bsts_results
    if "error" not in bsts_results:
        agg = bsts_results["aggregate"]
        print(f"    Events analyzed: {bsts_results['n_events_analyzed']}")
        print(f"    Mean causal impact: {agg['mean_impact']:.6f} "
              f"(p={agg['t_p_value']:.4f})")
        print(f"    Pct positive effect: {agg['pct_positive']:.1%}")
    else:
        print(f"    ✗ {bsts_results['error']}")

    # ── 4. Local Projections ──────────────────────────────────────────
    print("\n  Running local projections…")
    lp_results = run_local_projections(ts_df)
    results["local_projections"] = lp_results
    if "error" not in lp_results:
        print(f"    Horizons estimated: {lp_results['n_horizons']}")
        print(f"    Peak β: {lp_results['peak_beta']:.6f} "
              f"at h={lp_results['peak_horizon']} "
              f"(p={lp_results['peak_p_value']:.4f})")
        print(f"    Significant horizons: {lp_results['n_significant']}/{lp_results['n_horizons']}")
    else:
        print(f"    ✗ {lp_results['error']}")

    # ── 5. Method comparison ──────────────────────────────────────────
    print("\n  Building method comparison…")
    # Load ITS pooled results
    its_path = RESULTS_DIR / "apocalypticism_its_results.json"
    its_pooled = {}
    if its_path.exists():
        with open(its_path) as f:
            its_data = json.load(f)
        its_pooled = its_data.get("pooled", {})

    comparison = build_method_comparison(
        its_pooled, var_results, ardl_results, bsts_results, lp_results,
    )
    results["method_comparison"] = comparison
    print(f"    Consensus: {comparison['consensus']['conclusion']}")

    # ── 6. Stratified VAR ─────────────────────────────────────────────
    print("\n  Running stratified VAR (violence vs non-violence)…")
    strat_var = run_stratified_var(ts_df, events)
    results["stratified_var"] = strat_var
    for cat, r in strat_var.items():
        if "error" not in r:
            gc = r.get("granger_causality", {}).get("event_causes_apoc", {})
            p = gc.get("p_value", float("nan"))
            print(f"    {cat}: Granger p={p:.4f}")
        else:
            print(f"    {cat}: {r['error']}")

    # ── Plots ─────────────────────────────────────────────────────────
    print("\n  Generating plots…")

    if "error" not in var_results:
        plot_irf(var_results.get("irf", {}), "adv_ts_var_irf")
        plot_fevd(var_results.get("fevd", {}), "adv_ts_var_fevd")

    if "error" not in lp_results:
        plot_local_projections(lp_results, "adv_ts_local_projections")

    plot_method_comparison(comparison, "adv_ts_method_comparison")

    # ── Save results ──────────────────────────────────────────────────
    out_path = RESULTS_DIR / "advanced_ts_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✓ Advanced time-series analysis complete. Saved to {out_path.name}")

    # ══════════════════════════════════════════════════════════════════
    # Per-category disaggregated advanced TS
    # ══════════════════════════════════════════════════════════════════
    _s30 = importlib.import_module("30_pol_apocalypticism")
    APOC_CATEGORIES = _s30.APOC_CATEGORIES

    if "apoc_category" in pol.columns:
        print("\n" + "=" * 60)
        print("  Per-Category Disaggregated Advanced TS")
        print("=" * 60)

        cat_ts: dict = {}
        for cat in APOC_CATEGORIES:
            print(f"\n  ── Category: {cat} ──")
            cat_pol = pol.filter(
                (pl.col("apoc_binary") == 1) & (pl.col("apoc_category") == cat)
            )
            if cat_pol.height < 50:
                print(f"    Skipping (only {cat_pol.height} posts)")
                cat_ts[cat] = {"error": "insufficient posts",
                                "n_posts": cat_pol.height}
                continue

            cat_daily = build_daily_series(cat_pol, primary_measure)
            cat_ts_df = build_event_series(cat_daily, events)
            print(f"    Posts: {cat_pol.height:,}, Days: {len(cat_ts_df)}")

            # VAR
            cat_var = run_var_analysis(cat_ts_df)
            if "error" not in cat_var:
                gc = cat_var.get("granger_causality", {}).get(
                    "event_causes_apoc", {}
                )
                print(f"    VAR Granger p={gc.get('p_value', 'N/A')}")

            # ARDL
            cat_ardl = run_ardl_analysis(cat_ts_df)
            if "error" not in cat_ardl:
                print(f"    ARDL LR multiplier: "
                      f"{cat_ardl.get('long_run_multiplier', 'N/A')}")

            # Local Projections
            cat_lp = run_local_projections(cat_ts_df)
            if "error" not in cat_lp:
                print(f"    LP peak β={cat_lp['peak_beta']:.6f} "
                      f"at h={cat_lp['peak_horizon']}")

            cat_ts[cat] = {
                "n_posts": cat_pol.height,
                "var": cat_var,
                "ardl": cat_ardl,
                "local_projections": cat_lp,
            }

        results["per_category"] = cat_ts
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n✓ Per-category advanced TS complete. Updated {out_path.name}")


if __name__ == "__main__":
    main()
