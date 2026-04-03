"""07 – Granger Causality Analysis (H3).

Tests whether Zeiger's rhetoric leads the community's, or vice versa.
"""

from __future__ import annotations

import json
import datetime

import numpy as np
import polars as pl
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests

from utils import (
    DATA_PROCESSED, RESULTS_DIR,
    ZEIGER_MEMBER_ID, setup_plot_style, save_figure, FIGSIZE_WIDE, CB_PALETTE,
)
import matplotlib.pyplot as plt


def build_weekly_pair(scores: pl.DataFrame) -> pl.DataFrame:
    """Build matched weekly time series of Zeiger vs. community scores."""
    df = scores.filter(pl.col("date").is_not_null())

    zeiger = (
        df.filter(pl.col("author_id") == ZEIGER_MEMBER_ID)
        .with_columns(pl.col("date").dt.truncate("1w").alias("week"))
        .group_by("week")
        .agg(pl.col("siege_keyword_score").mean().alias("zeiger_siege"))
        .sort("week")
    )

    community = (
        df.filter(pl.col("author_id") != ZEIGER_MEMBER_ID)
        .with_columns(pl.col("date").dt.truncate("1w").alias("week"))
        .group_by("week")
        .agg(pl.col("siege_keyword_score").mean().alias("community_siege"))
        .sort("week")
    )

    # Merge on week
    paired = zeiger.join(community, on="week", how="inner").sort("week")
    return paired


def run_granger_tests(paired: pl.DataFrame, max_lag: int = 8) -> dict:
    """Run pairwise Granger causality tests."""
    pdf = paired.to_pandas().dropna()

    if len(pdf) < max_lag + 5:
        print(f"  ⚠ Insufficient data for Granger tests ({len(pdf)} obs)")
        return {"error": "insufficient data"}

    results = {}

    # Zeiger → Community
    print("\n  Testing: Zeiger → Community")
    try:
        data_zc = pdf[["community_siege", "zeiger_siege"]].values
        gc_zc = grangercausalitytests(data_zc, maxlag=max_lag, verbose=False)
        for lag in range(1, max_lag + 1):
            test_result = gc_zc[lag][0]
            results[f"zeiger_to_community_lag{lag}"] = {
                "f_stat": float(test_result["ssr_ftest"][0]),
                "p_value": float(test_result["ssr_ftest"][1]),
            }
            sig = "***" if test_result["ssr_ftest"][1] < 0.01 else \
                  "**" if test_result["ssr_ftest"][1] < 0.05 else \
                  "*" if test_result["ssr_ftest"][1] < 0.1 else ""
            print(f"    Lag {lag}: F={test_result['ssr_ftest'][0]:.3f}, "
                  f"p={test_result['ssr_ftest'][1]:.4f} {sig}")
    except Exception as e:
        print(f"    Error: {e}")
        results["zeiger_to_community_error"] = str(e)

    # Community → Zeiger
    print("\n  Testing: Community → Zeiger")
    try:
        data_cz = pdf[["zeiger_siege", "community_siege"]].values
        gc_cz = grangercausalitytests(data_cz, maxlag=max_lag, verbose=False)
        for lag in range(1, max_lag + 1):
            test_result = gc_cz[lag][0]
            results[f"community_to_zeiger_lag{lag}"] = {
                "f_stat": float(test_result["ssr_ftest"][0]),
                "p_value": float(test_result["ssr_ftest"][1]),
            }
            sig = "***" if test_result["ssr_ftest"][1] < 0.01 else \
                  "**" if test_result["ssr_ftest"][1] < 0.05 else \
                  "*" if test_result["ssr_ftest"][1] < 0.1 else ""
            print(f"    Lag {lag}: F={test_result['ssr_ftest'][0]:.3f}, "
                  f"p={test_result['ssr_ftest'][1]:.4f} {sig}")
    except Exception as e:
        print(f"    Error: {e}")
        results["community_to_zeiger_error"] = str(e)

    return results


def plot_ccf(paired: pl.DataFrame, filename: str):
    """Plot cross-correlation function between Zeiger and community."""
    setup_plot_style()

    pdf = paired.to_pandas().dropna()
    z = pdf["zeiger_siege"].values
    c = pdf["community_siege"].values

    # Normalize
    z = (z - z.mean()) / (z.std() + 1e-10)
    c = (c - c.mean()) / (c.std() + 1e-10)

    max_lag = min(20, len(z) // 4)
    lags = range(-max_lag, max_lag + 1)
    ccf_values = []
    for lag in lags:
        if lag >= 0:
            ccf_values.append(np.corrcoef(z[lag:], c[:len(c) - lag])[0, 1] if lag < len(z) else 0.0)
        else:
            ccf_values.append(np.corrcoef(z[:len(z) + lag], c[-lag:])[0, 1] if -lag < len(c) else 0.0)

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.bar(list(lags), ccf_values, color=CB_PALETTE[0], alpha=0.7)

    # Significance bounds (approximate)
    n = len(z)
    ci = 1.96 / np.sqrt(n)
    ax.axhline(ci, color="red", linestyle="--", alpha=0.5)
    ax.axhline(-ci, color="red", linestyle="--", alpha=0.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)

    ax.set_xlabel("Lag (weeks; positive = Community leads)")
    ax.set_ylabel("Cross-correlation")
    ax.set_title("Cross-Correlation: Zeiger vs. Community Siege Score")

    save_figure(fig, filename)


def main():
    print("=" * 60)
    print("PHASE 4: Granger Causality (H3)")
    print("=" * 60)

    scores = pl.read_parquet(DATA_PROCESSED / "siege_scores.parquet")

    print("\nBuilding weekly paired series…")
    paired = build_weekly_pair(scores)
    print(f"  Paired weeks: {paired.height}")

    print("\nRunning Granger causality tests…")
    results = run_granger_tests(paired)

    print("\nPlotting CCF…")
    plot_ccf(paired, "ccf_zeiger_community")

    with open(RESULTS_DIR / "granger_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✓ Granger causality results saved.")


if __name__ == "__main__":
    main()
