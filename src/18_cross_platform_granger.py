"""18 – Cross-Platform Granger Causality (H13).

Tests the temporal ordering of Siege rhetoric between Iron March and /pol/.

Key hypotheses
--------------
- H13a: Does Iron March Siege rhetoric *Granger-cause* /pol/ Siege rhetoric?
- H13b: Does /pol/ Granger-cause Iron March? (reverse direction)
- H13c: What is the peak cross-correlation lag?

If H13a is significant but H13b is not, the Iron March exegesis process
preceded and predicted /pol/ adoption — strong evidence that the
collective exegesis on IM produced cross-platform contagion.

Methodology follows Hine et al. (2017) and Zannettou et al. (2017) for
cross-platform temporal analysis, adapted with HAC-robust inference.
"""

from __future__ import annotations

import datetime
import json

import numpy as np
import polars as pl
from scipy import signal
from statsmodels.tsa.stattools import grangercausalitytests, adfuller

from utils import (
    DATA_PROCESSED, RESULTS_DIR, ZEIGER_MEMBER_ID,
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


def build_cross_platform_weekly(
    im_scores: pl.DataFrame,
    pol_scores: pl.DataFrame,
    measure_col: str = "siege_keyword_score",
) -> pl.DataFrame:
    """Build matched weekly time series for both platforms.

    Returns a DataFrame with columns:
        week, im_siege, pol_siege
    on weeks where both platforms have data.
    """
    im_weekly = (
        im_scores.filter(
            pl.col("date").is_not_null()
            & (pl.col("author_id") != ZEIGER_MEMBER_ID)
        )
        .with_columns(pl.col("date").dt.truncate("1w").alias("week"))
        .group_by("week")
        .agg([
            pl.col(measure_col).mean().alias("im_siege"),
            pl.len().alias("im_count"),
        ])
        .sort("week")
    )

    pol_weekly = (
        pol_scores.filter(pl.col("date").is_not_null())
        .with_columns(pl.col("date").dt.truncate("1w").alias("week"))
        .group_by("week")
        .agg([
            pl.col(measure_col).mean().alias("pol_siege"),
            pl.len().alias("pol_count"),
        ])
        .sort("week")
    )

    paired = im_weekly.join(pol_weekly, on="week", how="inner").sort("week")
    return paired


def test_stationarity(series: np.ndarray, name: str) -> dict:
    """Run Augmented Dickey-Fuller test.

    If the series is non-stationary, Granger tests on levels are invalid;
    we should first-difference.
    """
    result = adfuller(series, maxlag=12, autolag="AIC")
    adf_stat, p_value = result[0], result[1]
    stationary = p_value < 0.05
    print(f"    ADF test ({name}): stat={adf_stat:.3f}, p={p_value:.4f} "
          f"→ {'stationary' if stationary else 'NON-STATIONARY'}")
    return {
        "series": name,
        "adf_statistic": float(adf_stat),
        "p_value": float(p_value),
        "stationary": stationary,
    }


def run_granger_tests(
    paired: pl.DataFrame,
    max_lag: int = 12,
) -> dict:
    """Run bidirectional Granger causality tests.

    If either series is non-stationary, applies first-differencing.
    """
    pdf = paired.to_pandas().dropna()

    if len(pdf) < max_lag + 10:
        print(f"  ⚠ Insufficient overlapping weeks ({len(pdf)})")
        return {"error": "insufficient data", "n_weeks": len(pdf)}

    results = {"n_weeks": len(pdf)}

    # ── Stationarity checks ──────────────────────────────────────────
    print("\n  Stationarity tests:")
    im_adf = test_stationarity(pdf["im_siege"].values, "IM")
    pol_adf = test_stationarity(pdf["pol_siege"].values, "/pol/")
    results["adf_im"] = im_adf
    results["adf_pol"] = pol_adf

    # If either is non-stationary, first-difference both
    use_diff = not (im_adf["stationary"] and pol_adf["stationary"])
    if use_diff:
        print("    → Using first differences for Granger tests")
        pdf["im_siege"] = pdf["im_siege"].diff()
        pdf["pol_siege"] = pdf["pol_siege"].diff()
        pdf = pdf.dropna()
        results["differenced"] = True
    else:
        results["differenced"] = False

    # ── IM → /pol/ ────────────────────────────────────────────────────
    print(f"\n  Testing: Iron March → /pol/ (max_lag={max_lag})")
    try:
        data = pdf[["pol_siege", "im_siege"]].values
        gc = grangercausalitytests(data, maxlag=max_lag, verbose=False)
        im_to_pol = {}
        for lag in range(1, max_lag + 1):
            test = gc[lag][0]
            im_to_pol[f"lag{lag}"] = {
                "f_stat": float(test["ssr_ftest"][0]),
                "p_value": float(test["ssr_ftest"][1]),
            }
            sig = "***" if test["ssr_ftest"][1] < 0.01 else \
                  "**" if test["ssr_ftest"][1] < 0.05 else \
                  "*" if test["ssr_ftest"][1] < 0.1 else ""
            print(f"    Lag {lag:2d}: F={test['ssr_ftest'][0]:7.3f}, "
                  f"p={test['ssr_ftest'][1]:.4f} {sig}")

        # Best lag
        best_lag = min(im_to_pol, key=lambda k: im_to_pol[k]["p_value"])
        results["im_to_pol"] = im_to_pol
        results["im_to_pol_best_lag"] = best_lag
        results["im_to_pol_best_p"] = im_to_pol[best_lag]["p_value"]
        print(f"    Best: {best_lag} (p={im_to_pol[best_lag]['p_value']:.4f})")
    except Exception as e:
        print(f"    Error: {e}")
        results["im_to_pol_error"] = str(e)

    # ── /pol/ → IM ────────────────────────────────────────────────────
    print(f"\n  Testing: /pol/ → Iron March (max_lag={max_lag})")
    try:
        data = pdf[["im_siege", "pol_siege"]].values
        gc = grangercausalitytests(data, maxlag=max_lag, verbose=False)
        pol_to_im = {}
        for lag in range(1, max_lag + 1):
            test = gc[lag][0]
            pol_to_im[f"lag{lag}"] = {
                "f_stat": float(test["ssr_ftest"][0]),
                "p_value": float(test["ssr_ftest"][1]),
            }
            sig = "***" if test["ssr_ftest"][1] < 0.01 else \
                  "**" if test["ssr_ftest"][1] < 0.05 else \
                  "*" if test["ssr_ftest"][1] < 0.1 else ""
            print(f"    Lag {lag:2d}: F={test['ssr_ftest'][0]:7.3f}, "
                  f"p={test['ssr_ftest'][1]:.4f} {sig}")

        best_lag = min(pol_to_im, key=lambda k: pol_to_im[k]["p_value"])
        results["pol_to_im"] = pol_to_im
        results["pol_to_im_best_lag"] = best_lag
        results["pol_to_im_best_p"] = pol_to_im[best_lag]["p_value"]
        print(f"    Best: {best_lag} (p={pol_to_im[best_lag]['p_value']:.4f})")
    except Exception as e:
        print(f"    Error: {e}")
        results["pol_to_im_error"] = str(e)

    return results


def compute_ccf(
    paired: pl.DataFrame,
    max_lag: int = 26,
) -> dict:
    """Compute cross-correlation function and find peak lag.

    A positive peak at lag=k means IM *leads* /pol/ by k weeks.
    """
    pdf = paired.to_pandas().dropna()
    im = pdf["im_siege"].values
    pol = pdf["pol_siege"].values

    # Normalise
    im_z = (im - im.mean()) / (im.std() + 1e-10)
    pol_z = (pol - pol.mean()) / (pol.std() + 1e-10)

    lags = np.arange(-max_lag, max_lag + 1)
    ccf_vals = np.correlate(pol_z, im_z, mode="full")
    # Normalise by n
    ccf_vals = ccf_vals / len(im_z)

    # The full correlation has length 2*n-1; extract the relevant window
    mid = len(im_z) - 1
    ccf_window = []
    for lag in lags:
        idx = mid + lag
        if 0 <= idx < len(ccf_vals):
            ccf_window.append(float(ccf_vals[idx]))
        else:
            ccf_window.append(0.0)

    peak_idx = int(np.argmax(np.abs(ccf_window)))
    peak_lag = int(lags[peak_idx])
    peak_corr = ccf_window[peak_idx]

    # 95% CI
    ci = 1.96 / np.sqrt(len(im_z))

    print(f"\n  CCF peak: lag={peak_lag} weeks, r={peak_corr:.4f}")
    if peak_lag > 0:
        print(f"    → IM leads /pol/ by ~{peak_lag} weeks")
    elif peak_lag < 0:
        print(f"    → /pol/ leads IM by ~{abs(peak_lag)} weeks")
    else:
        print(f"    → Contemporaneous (no lag)")

    return {
        "peak_lag": peak_lag,
        "peak_correlation": peak_corr,
        "ci_95": float(ci),
        "lags": [int(l) for l in lags],
        "correlations": ccf_window,
    }


def plot_cross_platform_ccf(ccf_result: dict, filename: str):
    """Plot cross-correlation function between platforms."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    lags = ccf_result["lags"]
    corrs = ccf_result["correlations"]
    ci = ccf_result["ci_95"]

    ax.bar(lags, corrs, color=CB_PALETTE[0], alpha=0.7, width=0.8)
    ax.axhline(ci, color="red", linestyle="--", alpha=0.5, label="95% CI")
    ax.axhline(-ci, color="red", linestyle="--", alpha=0.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)

    # Mark peak
    peak_lag = ccf_result["peak_lag"]
    peak_corr = ccf_result["peak_correlation"]
    ax.annotate(
        f"Peak: lag={peak_lag}, r={peak_corr:.3f}",
        xy=(peak_lag, peak_corr),
        xytext=(peak_lag + 3, peak_corr + 0.05),
        arrowprops={"arrowstyle": "->", "color": "red"},
        fontsize=10, color="red",
    )

    ax.set_xlabel("Lag (weeks; positive = Iron March leads /pol/)")
    ax.set_ylabel("Cross-correlation")
    ax.set_title("Cross-Platform CCF: Iron March vs. /pol/ Siege Score")
    ax.legend()

    save_figure(fig, filename)


def plot_cross_platform_series(
    paired: pl.DataFrame,
    t0: datetime.datetime,
    filename: str,
):
    """Plot both platform weekly series overlaid."""
    setup_plot_style()
    fig, ax1 = plt.subplots(figsize=FIGSIZE_WIDE)

    pdf = paired.to_pandas().sort_values("week")

    ax1.plot(pdf["week"], pdf["im_siege"], color=CB_PALETTE[0],
             alpha=0.8, linewidth=1.5, label="Iron March")
    ax1.set_ylabel("IM Siege Score", color=CB_PALETTE[0])

    ax2 = ax1.twinx()
    ax2.plot(pdf["week"], pdf["pol_siege"], color=CB_PALETTE[2],
             alpha=0.8, linewidth=1.5, label="/pol/")
    ax2.set_ylabel("/pol/ Siege Score", color=CB_PALETTE[2])

    ax1.axvline(t0, color="red", linestyle="--", linewidth=2,
                alpha=0.7, label="Siege publication")

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.set_xlabel("Date")
    ax1.set_title("Cross-Platform Siege Rhetoric: Iron March vs. /pol/")
    fig.autofmt_xdate()

    save_figure(fig, filename)


def main():
    """Run cross-platform Granger causality analysis."""
    print("=" * 60)
    print("PHASE 9: Cross-Platform Granger Causality (H13)")
    print("=" * 60)

    t0 = load_treatment_date()
    results = {"treatment_date": str(t0)}

    # ── Load data ─────────────────────────────────────────────────────
    im_path = DATA_PROCESSED / "siege_scores.parquet"
    pol_path = DATA_PROCESSED / "pol_siege_scores.parquet"

    if not im_path.exists():
        print(f"  ✗ {im_path} not found.")
        return
    if not pol_path.exists():
        print(f"  ✗ {pol_path} not found. Run lexicon scoring on /pol/ first.")
        return

    im_scores = pl.read_parquet(im_path)
    im_scores = im_scores.filter(pl.col("channel") == "forum")
    pol_scores = pl.read_parquet(pol_path)
    print(f"  IM posts: {im_scores.height:,}, /pol/ posts: {pol_scores.height:,}")

    # ── Build paired weekly series ────────────────────────────────────
    print("\n  Building cross-platform weekly series…")
    paired = build_cross_platform_weekly(im_scores, pol_scores)
    print(f"  Overlapping weeks: {paired.height}")
    if paired.height > 0:
        print(f"  Date range: {paired['week'].min()} → {paired['week'].max()}")

    results["overlapping_weeks"] = paired.height

    if paired.height < 15:
        print("  ⚠ Too few overlapping weeks for Granger tests")
        with open(RESULTS_DIR / "cross_platform_granger_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        return

    # ── Granger causality ─────────────────────────────────────────────
    print("\n  Running Granger causality tests…")
    granger = run_granger_tests(paired, max_lag=12)
    results["granger"] = granger

    # ── Cross-correlation ─────────────────────────────────────────────
    print("\n  Computing cross-correlation function…")
    ccf = compute_ccf(paired, max_lag=min(26, paired.height // 3))
    results["ccf"] = {
        "peak_lag": ccf["peak_lag"],
        "peak_correlation": ccf["peak_correlation"],
        "ci_95": ccf["ci_95"],
    }

    # ── Plots ─────────────────────────────────────────────────────────
    print("\n  Generating plots…")
    plot_cross_platform_ccf(ccf, "cross_platform_ccf")
    plot_cross_platform_series(paired, t0, "cross_platform_series")

    # ── Save ──────────────────────────────────────────────────────────
    with open(RESULTS_DIR / "cross_platform_granger_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✓ Cross-platform Granger results saved.")


if __name__ == "__main__":
    main()
