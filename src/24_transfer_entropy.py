"""24 – Transfer Entropy (H17).

Non-linear, model-free test for directed information flow between
Iron March and /pol/ Siege rhetoric time series.  Transfer entropy
captures non-linear dependencies that Granger causality (H13) misses
because Granger is restricted to linear VAR models.

Definition
----------
TE(X→Y) = H(Y_t | Y_{t-1:t-k}) − H(Y_t | Y_{t-1:t-k}, X_{t-1:t-k})

If TE(IM→pol) > TE(pol→IM) and is statistically significant (via
shuffle surrogates), this provides evidence for directed information
transfer from IM to /pol/.

Implementation uses binning-based estimator (Schreiber 2000) with
shuffle-surrogate significance testing.
"""

from __future__ import annotations

import datetime
import json

import numpy as np
import polars as pl

from utils import (
    DATA_PROCESSED, RESULTS_DIR, ZEIGER_MEMBER_ID,
    setup_plot_style, save_figure, FIGSIZE_WIDE, CB_PALETTE,
)
import matplotlib.pyplot as plt


def _discretise(x: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Discretise continuous series into equal-frequency bins."""
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(x, percentiles)
    # Ensure unique edges
    edges = np.unique(edges)
    return np.digitize(x, edges[1:-1])


def _transfer_entropy(source: np.ndarray, target: np.ndarray,
                      lag: int = 1, n_bins: int = 5) -> float:
    """Compute transfer entropy TE(source→target) using binned estimator.

    TE = Σ p(y_t, y_{t-lag}, x_{t-lag}) * log[ p(y_t|y_{t-lag},x_{t-lag})
                                                 / p(y_t|y_{t-lag}) ]
    """
    src = _discretise(source, n_bins)
    tgt = _discretise(target, n_bins)

    n = len(src)
    if n <= lag:
        return 0.0

    # Build joint counts
    yt = tgt[lag:]
    yt_prev = tgt[:-lag] if lag > 0 else tgt
    xt_prev = src[:-lag] if lag > 0 else src

    # Ensure same length
    min_len = min(len(yt), len(yt_prev), len(xt_prev))
    yt = yt[:min_len]
    yt_prev = yt_prev[:min_len]
    xt_prev = xt_prev[:min_len]

    N = len(yt)
    if N < 10:
        return 0.0

    # Count tables using integer hashing
    max_val = max(yt.max(), yt_prev.max(), xt_prev.max()) + 1

    # p(yt, yt_prev, xt_prev)
    triple_key = yt * max_val * max_val + yt_prev * max_val + xt_prev
    triple_vals, triple_counts = np.unique(triple_key, return_counts=True)
    p_triple = dict(zip(triple_vals, triple_counts / N))

    # p(yt, yt_prev)
    pair_key_yy = yt * max_val + yt_prev
    pair_vals_yy, pair_counts_yy = np.unique(pair_key_yy, return_counts=True)
    p_yy = dict(zip(pair_vals_yy, pair_counts_yy / N))

    # p(yt_prev, xt_prev)
    pair_key_yx = yt_prev * max_val + xt_prev
    pair_vals_yx, pair_counts_yx = np.unique(pair_key_yx, return_counts=True)
    p_yx = dict(zip(pair_vals_yx, pair_counts_yx / N))

    # p(yt_prev)
    prev_vals, prev_counts = np.unique(yt_prev, return_counts=True)
    p_prev = dict(zip(prev_vals, prev_counts / N))

    te = 0.0
    for tk, p_joint in p_triple.items():
        y_cur = tk // (max_val * max_val)
        y_prev = (tk // max_val) % max_val
        x_prev = tk % max_val

        p_yy_val = p_yy.get(y_cur * max_val + y_prev, 1e-10)
        p_yx_val = p_yx.get(y_prev * max_val + x_prev, 1e-10)
        p_y_val = p_prev.get(y_prev, 1e-10)

        # p(yt|yt_prev,xt_prev) = p(yt,yt_prev,xt_prev) / p(yt_prev,xt_prev)
        cond_joint = p_joint / p_yx_val
        # p(yt|yt_prev) = p(yt,yt_prev) / p(yt_prev)
        cond_marg = p_yy_val / p_y_val

        if cond_joint > 0 and cond_marg > 0:
            te += p_joint * np.log2(cond_joint / cond_marg)

    return float(te)


def transfer_entropy_with_significance(
    source: np.ndarray,
    target: np.ndarray,
    lag: int = 1,
    n_bins: int = 5,
    n_surrogates: int = 200,
    seed: int = 42,
) -> dict:
    """Compute TE with shuffle-surrogate significance test."""
    te_obs = _transfer_entropy(source, target, lag=lag, n_bins=n_bins)

    rng = np.random.RandomState(seed)
    te_surrogates = np.empty(n_surrogates)
    for i in range(n_surrogates):
        shuffled = rng.permutation(source)
        te_surrogates[i] = _transfer_entropy(shuffled, target,
                                             lag=lag, n_bins=n_bins)

    p_value = float(np.mean(te_surrogates >= te_obs))
    z_score = float(
        (te_obs - te_surrogates.mean()) / (te_surrogates.std() + 1e-10)
    )

    return {
        "te": float(te_obs),
        "surrogate_mean": float(te_surrogates.mean()),
        "surrogate_std": float(te_surrogates.std()),
        "p_value": p_value,
        "z_score": z_score,
        "significant_05": p_value < 0.05,
    }


def build_weekly_pair(
    im_scores: pl.DataFrame,
    pol_scores: pl.DataFrame,
    measure_col: str = "siege_keyword_score",
) -> tuple[np.ndarray, np.ndarray]:
    """Build matched weekly series and return aligned numpy arrays."""
    im_weekly = (
        im_scores.filter(
            pl.col("date").is_not_null()
            & (pl.col("author_id") != ZEIGER_MEMBER_ID)
        )
        .with_columns(pl.col("date").dt.truncate("1w").alias("week"))
        .group_by("week")
        .agg(pl.col(measure_col).mean().alias("im_siege"))
        .sort("week")
    )

    pol_weekly = (
        pol_scores.filter(pl.col("date").is_not_null())
        .with_columns(pl.col("date").dt.truncate("1w").alias("week"))
        .group_by("week")
        .agg(pl.col(measure_col).mean().alias("pol_siege"))
        .sort("week")
    )

    paired = im_weekly.join(pol_weekly, on="week", how="inner").sort("week")
    return (
        paired["im_siege"].to_numpy(),
        paired["pol_siege"].to_numpy(),
    )


def plot_transfer_entropy(te_results: dict, filename: str):
    """Bar chart comparing TE in both directions with significance."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    labels = ["IM → /pol/", "/pol/ → IM"]
    te_vals = [
        te_results["im_to_pol"]["te"],
        te_results["pol_to_im"]["te"],
    ]
    surr_means = [
        te_results["im_to_pol"]["surrogate_mean"],
        te_results["pol_to_im"]["surrogate_mean"],
    ]
    surr_stds = [
        te_results["im_to_pol"]["surrogate_std"],
        te_results["pol_to_im"]["surrogate_std"],
    ]
    p_vals = [
        te_results["im_to_pol"]["p_value"],
        te_results["pol_to_im"]["p_value"],
    ]

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width / 2, te_vals, width, label="Observed TE",
           color=[CB_PALETTE[0], CB_PALETTE[2]], alpha=0.8)
    ax.bar(x + width / 2, surr_means, width, label="Surrogate mean",
           color="gray", alpha=0.5, yerr=surr_stds)

    # Significance markers
    for i, p in enumerate(p_vals):
        if p < 0.01:
            marker = "***"
        elif p < 0.05:
            marker = "**"
        elif p < 0.1:
            marker = "*"
        else:
            marker = "n.s."
        ax.text(x[i], max(te_vals[i], surr_means[i]) + 0.005,
                f"p={p:.3f} {marker}", ha="center", fontsize=10)

    ax.set_ylabel("Transfer Entropy (bits)")
    ax.set_title("Transfer Entropy: Information Flow Between Platforms")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, filename)


def main():
    print("=" * 60)
    print("H17: Transfer Entropy")
    print("=" * 60)

    im_path = DATA_PROCESSED / "siege_scores.parquet"
    pol_path = DATA_PROCESSED / "pol_siege_scores.parquet"
    if not im_path.exists() or not pol_path.exists():
        print("  ✗ Missing scored data.")
        return

    im = pl.read_parquet(im_path).filter(pl.col("channel") == "forum")
    pol = pl.read_parquet(pol_path)
    print(f"  IM posts: {im.height:,}  |  /pol/ posts: {pol.height:,}")

    im_arr, pol_arr = build_weekly_pair(im, pol)
    print(f"  Overlapping weeks: {len(im_arr)}")

    if len(im_arr) < 20:
        print("  ⚠ Insufficient overlapping weeks.")
        return

    results: dict = {"n_weeks": len(im_arr)}

    # Run TE in both directions at multiple lags
    for lag in [1, 2, 4]:
        print(f"\n  Lag = {lag} week(s):")
        key_prefix = f"lag{lag}"

        print(f"    TE(IM → /pol/)…")
        te_im_pol = transfer_entropy_with_significance(
            im_arr, pol_arr, lag=lag, n_bins=5, n_surrogates=200
        )
        print(f"      TE = {te_im_pol['te']:.4f}  "
              f"(surr: {te_im_pol['surrogate_mean']:.4f} ± "
              f"{te_im_pol['surrogate_std']:.4f},  p={te_im_pol['p_value']:.3f})")

        print(f"    TE(/pol/ → IM)…")
        te_pol_im = transfer_entropy_with_significance(
            pol_arr, im_arr, lag=lag, n_bins=5, n_surrogates=200
        )
        print(f"      TE = {te_pol_im['te']:.4f}  "
              f"(surr: {te_pol_im['surrogate_mean']:.4f} ± "
              f"{te_pol_im['surrogate_std']:.4f},  p={te_pol_im['p_value']:.3f})")

        results[f"{key_prefix}_im_to_pol"] = te_im_pol
        results[f"{key_prefix}_pol_to_im"] = te_pol_im

        net = te_im_pol["te"] - te_pol_im["te"]
        results[f"{key_prefix}_net_flow"] = float(net)
        direction = "IM → /pol/" if net > 0 else "/pol/ → IM"
        print(f"    Net TE = {net:.4f}  ({direction})")

    # Summary at lag=1
    results["im_to_pol"] = results.get("lag1_im_to_pol", {})
    results["pol_to_im"] = results.get("lag1_pol_to_im", {})

    # Plot
    if "im_to_pol" in results and "pol_to_im" in results:
        plot_transfer_entropy(results, "transfer_entropy")

    with open(RESULTS_DIR / "transfer_entropy_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\n✓ Transfer entropy results saved.")


if __name__ == "__main__":
    main()
