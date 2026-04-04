"""23 – Vocabulary Adoption Lag Curves (H16).

For each Siege term that appears on both platforms, compute the delay
between first Iron March appearance and first /pol/ appearance.  Then
test whether this lag *shortens* over time (accelerating cross-platform
diffusion) or remains constant.

Key analyses
------------
- Per-term lag (days) between IM and /pol/ first usage.
- OLS: lag ~ IM_first_date  (does later IM adoption predict shorter /pol/ lag?)
- Survival/CDF of adoption lags.
"""

from __future__ import annotations

import datetime
import json
import re

import numpy as np
import polars as pl
from scipy import stats

from utils import (
    DATA_PROCESSED, RESULTS_DIR,
    setup_plot_style, save_figure, FIGSIZE_WIDE, CB_PALETTE,
)
import matplotlib.pyplot as plt


def compute_term_lags(
    im_posts: pl.DataFrame,
    pol_posts: pl.DataFrame,
) -> list[dict]:
    """For each Siege dictionary term, find earliest use on each platform."""
    from importlib import import_module
    lex = import_module("02_siege_lexicon")

    im_sorted = im_posts.filter(pl.col("date").is_not_null()).sort("date")
    pol_sorted = pol_posts.filter(pl.col("date").is_not_null()).sort("date")

    im_texts = list(zip(
        im_sorted["text"].to_list(),
        im_sorted["date"].to_list(),
    ))
    pol_texts = list(zip(
        pol_sorted["text"].to_list(),
        pol_sorted["date"].to_list(),
    ))

    results = []
    for pattern_str, weight in lex.SIEGE_DICTIONARY:
        if weight < 0.5:
            continue
        pat = re.compile(pattern_str, re.IGNORECASE)

        im_first = None
        for text, date in im_texts:
            if text and pat.search(text):
                im_first = date
                break

        pol_first = None
        for text, date in pol_texts:
            if text and pat.search(text):
                pol_first = date
                break

        if im_first and pol_first:
            lag_days = (pol_first - im_first).total_seconds() / 86400
            results.append({
                "term": pattern_str,
                "weight": weight,
                "im_first": im_first,
                "pol_first": pol_first,
                "lag_days": lag_days,
                "im_first_str": str(im_first),
                "pol_first_str": str(pol_first),
            })

    results.sort(key=lambda r: r["lag_days"])
    return results


def test_lag_acceleration(term_lags: list[dict]) -> dict:
    """Test whether later-adopted IM terms have shorter transfer lags to /pol/.

    OLS: lag_days ~ im_first_ordinal
    Negative slope → acceleration (later terms transfer faster).
    """
    if len(term_lags) < 5:
        return {"error": "insufficient terms"}

    im_dates = np.array([
        r["im_first"].timestamp() / 86400 for r in term_lags
    ])
    lags = np.array([r["lag_days"] for r in term_lags])

    slope, intercept, r_value, p_value, std_err = stats.linregress(im_dates, lags)

    result = {
        "n_terms": len(term_lags),
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_value ** 2),
        "p_value": float(p_value),
        "interpretation": "accelerating" if slope < 0 and p_value < 0.05
                          else "decelerating" if slope > 0 and p_value < 0.05
                          else "no trend",
    }

    print(f"  Lag acceleration OLS: slope={slope:.4f}, p={p_value:.4f}, "
          f"R²={r_value**2:.4f}")
    print(f"  Interpretation: {result['interpretation']}")
    return result


def plot_lag_curves(term_lags: list[dict], filename: str):
    """Plot lag CDF and scatter of IM adoption date vs transfer lag."""
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    lags = [r["lag_days"] for r in term_lags]

    # Panel 1 — CDF
    sorted_lags = np.sort(lags)
    cdf = np.arange(1, len(sorted_lags) + 1) / len(sorted_lags)
    axes[0].step(sorted_lags, cdf, color=CB_PALETTE[0], linewidth=2)
    axes[0].axvline(0, color="red", linestyle="--", alpha=0.5,
                    label="Simultaneous")
    median_lag = float(np.median(lags))
    axes[0].axvline(median_lag, color=CB_PALETTE[2], linestyle=":",
                    label=f"Median: {median_lag:.0f} days")
    axes[0].set_xlabel("Lag (days; positive = IM first)")
    axes[0].set_ylabel("Cumulative proportion")
    axes[0].set_title("CDF of IM → /pol/ Term Adoption Lags")
    axes[0].legend()

    # Panel 2 — scatter: IM first-use date vs lag
    im_dates = [r["im_first"] for r in term_lags]
    axes[1].scatter(im_dates, lags, alpha=0.6, color=CB_PALETTE[0], s=30)
    # Trend line
    x_num = np.array([(d - im_dates[0]).total_seconds() / 86400
                      for d in im_dates])
    if len(x_num) >= 2:
        coeffs = np.polyfit(x_num, lags, 1)
        x_fit = np.linspace(x_num.min(), x_num.max(), 100)
        axes[1].plot(
            [im_dates[0] + datetime.timedelta(days=float(d)) for d in x_fit],
            np.polyval(coeffs, x_fit),
            color=CB_PALETTE[3], linewidth=2, linestyle="--",
            label=f"slope={coeffs[0]:.2f} d/d",
        )
    axes[1].set_xlabel("IM First Appearance Date")
    axes[1].set_ylabel("Transfer Lag (days)")
    axes[1].set_title("Does Transfer Speed Up?")
    axes[1].legend()
    fig.autofmt_xdate()

    fig.tight_layout()
    save_figure(fig, filename)


def main():
    print("=" * 60)
    print("H16: Vocabulary Adoption Lag Curves")
    print("=" * 60)

    im_path = DATA_PROCESSED / "siege_scores.parquet"
    pol_path = DATA_PROCESSED / "pol_siege_scores.parquet"
    if not im_path.exists() or not pol_path.exists():
        print("  ✗ Missing scored data.")
        return

    im = pl.read_parquet(im_path).filter(pl.col("channel") == "forum")
    pol = pl.read_parquet(pol_path)
    print(f"  IM posts: {im.height:,}  |  /pol/ posts: {pol.height:,}")

    # Compute per-term lags
    print("\n  Computing per-term adoption lags…")
    term_lags = compute_term_lags(im, pol)
    print(f"  Terms on both platforms: {len(term_lags)}")

    if not term_lags:
        print("  ✗ No shared terms found.")
        return

    im_led = sum(1 for r in term_lags if r["lag_days"] > 0)
    pol_led = sum(1 for r in term_lags if r["lag_days"] < 0)
    median_lag = float(np.median([r["lag_days"] for r in term_lags]))
    print(f"  IM first: {im_led}  |  /pol/ first: {pol_led}")
    print(f"  Median lag: {median_lag:.1f} days")

    # Acceleration test
    print("\n  Testing lag acceleration…")
    accel = test_lag_acceleration(term_lags)

    # Results
    results = {
        "n_terms": len(term_lags),
        "im_led": im_led,
        "pol_led": pol_led,
        "median_lag_days": median_lag,
        "mean_lag_days": float(np.mean([r["lag_days"] for r in term_lags])),
        "acceleration_test": accel,
        "term_lags": [
            {k: v for k, v in r.items() if k not in ("im_first", "pol_first")}
            for r in term_lags
        ],
    }

    plot_lag_curves(term_lags, "vocab_adoption_lags")

    with open(RESULTS_DIR / "vocab_adoption_lag_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\n✓ Vocabulary adoption lag results saved.")


if __name__ == "__main__":
    main()
