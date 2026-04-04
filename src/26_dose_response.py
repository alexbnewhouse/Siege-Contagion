"""26 – Dose-Response at Multiple Lags (H19).

Non-parametric test: does higher IM Siege activity in week *t* predict
elevated /pol/ Siege rhetoric at lag weeks *t+1, t+2, … t+8*?

This avoids the stationarity assumptions of Granger/VAR models by
binning IM weeks into quartiles by Siege intensity and comparing the
*distribution* of /pol/ scores across those bins at each lag.

Key outputs
-----------
- Heatmap of mean /pol/ score by IM-quartile × lag.
- Kruskal–Wallis test at each lag (non-parametric ANOVA).
- Spearman rank-correlation between IM dose and lagged /pol/ response.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
from scipy import stats as sp_stats

from utils import (
    DATA_PROCESSED, RESULTS_DIR, ZEIGER_MEMBER_ID,
    setup_plot_style, save_figure, FIGSIZE_WIDE, CB_PALETTE,
)
import matplotlib.pyplot as plt


MAX_LAG = 8  # weeks


def build_weekly_scores(
    im: pl.DataFrame, pol: pl.DataFrame,
) -> pl.DataFrame:
    """Return joined weekly mean siege scores for both platforms."""
    im_w = (
        im.filter(
            pl.col("date").is_not_null()
            & (pl.col("author_id") != ZEIGER_MEMBER_ID)
        )
        .with_columns(pl.col("date").dt.truncate("1w").alias("week"))
        .group_by("week")
        .agg(pl.col("siege_keyword_score").mean().alias("im_score"))
        .sort("week")
    )

    pol_w = (
        pol.filter(pl.col("date").is_not_null())
        .with_columns(pl.col("date").dt.truncate("1w").alias("week"))
        .group_by("week")
        .agg(pl.col("siege_keyword_score").mean().alias("pol_score"))
        .sort("week")
    )

    return im_w.join(pol_w, on="week", how="inner").sort("week")


def dose_response_analysis(
    weekly: pl.DataFrame, max_lag: int = MAX_LAG,
) -> dict:
    """Bin IM weeks by quartile and test /pol/ response at each lag."""
    im_scores = weekly["im_score"].to_numpy()
    pol_scores = weekly["pol_score"].to_numpy()
    n = len(im_scores)

    # Compute IM dose quartiles
    quartile_edges = np.percentile(im_scores, [25, 50, 75])
    quartiles = np.digitize(im_scores, quartile_edges)  # 0-3

    results_by_lag: dict = {}

    for lag in range(1, max_lag + 1):
        if lag >= n:
            break

        # Match IM dose at t with /pol/ response at t+lag
        dose = quartiles[:n - lag]
        response = pol_scores[lag:]

        # Group responses by dose quartile
        groups = {q: response[dose == q] for q in range(4)}
        groups = {q: g for q, g in groups.items() if len(g) > 0}

        if len(groups) < 2:
            results_by_lag[f"lag_{lag}"] = {"error": "insufficient groups"}
            continue

        # Kruskal-Wallis test
        group_arrays = [g for g in groups.values()]
        try:
            h_stat, kw_p = sp_stats.kruskal(*group_arrays)
        except ValueError:
            h_stat, kw_p = 0.0, 1.0

        # Spearman correlation: dose rank vs response
        rho, sp_p = sp_stats.spearmanr(dose, response)

        # Mean response by quartile
        quartile_means = {
            f"Q{q + 1}": float(g.mean()) for q, g in sorted(groups.items())
        }
        quartile_ns = {
            f"Q{q + 1}": int(len(g)) for q, g in sorted(groups.items())
        }

        results_by_lag[f"lag_{lag}"] = {
            "kruskal_h": float(h_stat),
            "kruskal_p": float(kw_p),
            "spearman_rho": float(rho),
            "spearman_p": float(sp_p),
            "quartile_means": quartile_means,
            "quartile_ns": quartile_ns,
            "significant_kw_05": bool(kw_p < 0.05),
            "significant_sp_05": bool(sp_p < 0.05),
        }

    return results_by_lag


def plot_dose_response_heatmap(lag_results: dict, filename: str):
    """Heatmap of /pol/ mean score by IM quartile × lag."""
    setup_plot_style()

    lags = sorted(
        [k for k in lag_results if k.startswith("lag_") and "error" not in lag_results[k]],
        key=lambda x: int(x.split("_")[1]),
    )
    if not lags:
        return

    quartiles = ["Q1", "Q2", "Q3", "Q4"]
    matrix = np.full((len(quartiles), len(lags)), np.nan)

    for j, lag_key in enumerate(lags):
        means = lag_results[lag_key].get("quartile_means", {})
        for i, q in enumerate(quartiles):
            if q in means:
                matrix[i, j] = means[q]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(range(len(lags)))
    ax.set_xticklabels([f"t+{k.split('_')[1]}" for k in lags])
    ax.set_yticks(range(len(quartiles)))
    ax.set_yticklabels(quartiles)
    ax.set_xlabel("Lag (weeks)")
    ax.set_ylabel("IM Dose Quartile (Q1=lowest)")
    ax.set_title("Dose-Response: /pol/ Mean Siege Score by IM Activity Quartile")

    # Add significance markers
    for j, lag_key in enumerate(lags):
        p = lag_results[lag_key].get("kruskal_p", 1.0)
        if p < 0.01:
            marker = "***"
        elif p < 0.05:
            marker = "**"
        elif p < 0.1:
            marker = "*"
        else:
            marker = ""
        if marker:
            ax.text(j, -0.5, marker, ha="center", fontsize=11, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Mean /pol/ Siege Score")
    fig.tight_layout()
    save_figure(fig, filename)


def plot_spearman_by_lag(lag_results: dict, filename: str):
    """Line plot of Spearman rho by lag with significance band."""
    setup_plot_style()
    lags_int = []
    rhos = []
    sigs = []

    for key in sorted(lag_results.keys(),
                      key=lambda x: int(x.split("_")[1]) if x.startswith("lag_") else 999):
        if not key.startswith("lag_") or "error" in lag_results[key]:
            continue
        lag_num = int(key.split("_")[1])
        lags_int.append(lag_num)
        rhos.append(lag_results[key]["spearman_rho"])
        sigs.append(lag_results[key]["spearman_p"] < 0.05)

    if not lags_int:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colours = [CB_PALETTE[0] if s else "gray" for s in sigs]
    ax.bar(lags_int, rhos, color=colours, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Lag (weeks)")
    ax.set_ylabel("Spearman ρ")
    ax.set_title("IM Dose → /pol/ Response: Spearman Correlation by Lag")
    ax.set_xticks(lags_int)

    # Legend proxy
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=CB_PALETTE[0], label="p < 0.05"),
        Patch(color="gray", label="p ≥ 0.05"),
    ])

    fig.tight_layout()
    save_figure(fig, filename)


def main():
    print("=" * 60)
    print("H19: Dose-Response at Multiple Lags")
    print("=" * 60)

    im_path = DATA_PROCESSED / "siege_scores.parquet"
    pol_path = DATA_PROCESSED / "pol_siege_scores.parquet"
    if not im_path.exists() or not pol_path.exists():
        print("  ✗ Missing scored data.")
        return

    im = pl.read_parquet(im_path).filter(pl.col("channel") == "forum")
    pol = pl.read_parquet(pol_path)
    print(f"  IM posts: {im.height:,}  |  /pol/ posts: {pol.height:,}")

    weekly = build_weekly_scores(im, pol)
    print(f"  Overlapping weeks: {weekly.height}")

    if weekly.height < 15:
        print("  ⚠ Insufficient overlapping weeks.")
        return

    print("\n  Running dose-response analysis…")
    lag_results = dose_response_analysis(weekly)

    # Print summary
    for lag_key in sorted(lag_results.keys(),
                          key=lambda x: int(x.split("_")[1]) if x.startswith("lag_") else 999):
        r = lag_results[lag_key]
        if "error" in r:
            continue
        s_kw = "★" if r["significant_kw_05"] else " "
        s_sp = "★" if r["significant_sp_05"] else " "
        print(f"    {lag_key}: KW H={r['kruskal_h']:.2f} p={r['kruskal_p']:.3f} {s_kw}"
              f"  |  ρ={r['spearman_rho']:.3f} p={r['spearman_p']:.3f} {s_sp}")

    # Plots
    plot_dose_response_heatmap(lag_results, "dose_response_heatmap")
    plot_spearman_by_lag(lag_results, "dose_response_spearman")

    # Save
    results = {
        "n_overlapping_weeks": weekly.height,
        "max_lag": MAX_LAG,
        "lag_results": lag_results,
    }

    with open(RESULTS_DIR / "dose_response_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\n✓ Dose-response results saved.")


if __name__ == "__main__":
    main()
