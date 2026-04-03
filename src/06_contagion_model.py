"""06 – Social Contagion Model (H2).

Tests whether exposure to siegist rhetoric through network ties predicts
subsequent adoption, using panel regression with user and time fixed effects.
"""

from __future__ import annotations

import json
import datetime
import multiprocessing
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm

from utils import DATA_PROCESSED, NETWORKS_DIR, RESULTS_DIR, ZEIGER_MEMBER_ID

_N_WORKERS = min(multiprocessing.cpu_count(), int(os.environ.get("SIEGE_WORKERS", "16")))


def load_network_neighbors(name: str, directed: bool = False) -> dict[int, set[int]]:
    """Load network edgelist and return adjacency dict."""
    path = NETWORKS_DIR / f"{name}_edgelist.parquet"
    if not path.exists():
        print(f"  ⚠ {name} edgelist not found")
        return {}

    edges = pl.read_parquet(path)
    neighbors: dict[int, set[int]] = defaultdict(set)
    for row in edges.iter_rows(named=True):
        s, t = int(row["source"]), int(row["target"])
        neighbors[s].add(t)
        if not directed:
            neighbors[t].add(s)
    return neighbors


def compute_network_exposure(
    user_monthly_scores: dict[tuple[int, str], float],
    neighbors: dict[int, set[int]],
    users: set[int],
    months: list[str],
) -> dict[tuple[int, str], float]:
    """Compute lagged network exposure for each user-month.

    exposure_i_t = (1/|N_i|) * Σ_{j ∈ N_i} siege_score_j_{t-1}
    """
    exposure: dict[tuple[int, str], float] = {}

    for t_idx, month in enumerate(months):
        if t_idx == 0:
            continue  # no lag for first month
        prev_month = months[t_idx - 1]

        for user in users:
            nbrs = neighbors.get(user, set())
            if not nbrs:
                exposure[(user, month)] = 0.0
                continue

            nbr_scores = []
            for nbr in nbrs:
                score = user_monthly_scores.get((nbr, prev_month), 0.0)
                nbr_scores.append(score)

            exposure[(user, month)] = float(np.mean(nbr_scores)) if nbr_scores else 0.0

    return exposure


def build_panel(scores: pl.DataFrame, treatment_date: datetime.datetime) -> pd.DataFrame:
    """Build user-month panel dataset with siege scores and controls."""
    df = scores.filter(
        pl.col("date").is_not_null()
        & pl.col("author_id").is_not_null()
    ).with_columns(
        pl.col("date").dt.strftime("%Y-%m").alias("month")
    )

    # Aggregate to user-month
    panel = (
        df.group_by(["author_id", "month"])
        .agg([
            pl.col("siege_keyword_score").mean().alias("siege_score"),
            pl.col("siege_similarity").mean().alias("siege_sim"),
            pl.col("siege_binary").max().alias("siege_any"),
            pl.len().alias("post_count"),
        ])
        .sort(["author_id", "month"])
    )

    # Post-treatment indicator
    t0_month = treatment_date.strftime("%Y-%m")
    panel = panel.with_columns(
        (pl.col("month") >= t0_month).cast(pl.Int8).alias("post_treatment")
    )

    return panel.to_pandas()


def run_panel_regression(panel: pd.DataFrame, label: str) -> dict:
    """Run panel regression with two-way (user + time) fixed effects via demeaning."""
    df = panel.copy()

    # Need sufficient variation
    if df["siege_score"].std() < 1e-10:
        return {"label": label, "error": "no variation in outcome"}

    # Create exposure columns if they exist
    exposure_cols = [c for c in df.columns if c.endswith("_exposure")]
    control_cols = ["log_post_count", "post_treatment"]

    df["log_post_count"] = np.log1p(df["post_count"])

    # Two-way within-transformation: subtract user means, month means, add grand mean
    all_cols = ["siege_score"] + exposure_cols + control_cols
    for col in all_cols:
        if col in df.columns:
            user_mean = df.groupby("author_id")[col].transform("mean")
            month_mean = df.groupby("month")[col].transform("mean")
            grand_mean = df[col].mean()
            df[f"{col}_dm"] = df[col] - user_mean - month_mean + grand_mean

    # Build regression
    y = df["siege_score_dm"].values

    feature_cols_dm = [f"{c}_dm" for c in exposure_cols + control_cols if f"{c}_dm" in df.columns]
    if not feature_cols_dm:
        return {"label": label, "error": "no features"}

    X = df[feature_cols_dm].values
    X = sm.add_constant(X)

    try:
        model = sm.OLS(y, X).fit(cov_type="HC1")
    except Exception as e:
        return {"label": label, "error": str(e)}

    result = {"label": label, "n_obs": int(len(df)), "r_squared": float(model.rsquared)}
    for i, col in enumerate(["const"] + feature_cols_dm):
        result[f"b_{col}"] = float(model.params[i])
        result[f"p_{col}"] = float(model.pvalues[i])
        result[f"se_{col}"] = float(model.bse[i])

    print(f"\n  {label} (n={len(df)}, R²={model.rsquared:.4f}):")
    for col in feature_cols_dm:
        idx = feature_cols_dm.index(col) + 1
        print(f"    {col}: β={model.params[idx]:.4f} (p={model.pvalues[idx]:.4f})")

    return result


def _run_single_permutation(args: tuple) -> dict[str, float]:
    """Run a single permutation (top-level function for pickling)."""
    perm_idx, seed, panel_values, panel_columns, panel_dtypes, neighbors_dict_serializable, user_monthly_scores, users_list, months = args

    panel = pd.DataFrame(panel_values, columns=panel_columns)
    # Restore dtypes lost during numpy serialization
    for col, dtype in panel_dtypes.items():
        if col in panel.columns:
            try:
                panel[col] = panel[col].astype(dtype)
            except (ValueError, TypeError):
                pass
    users = set(users_list)

    # Reconstruct neighbors as dict[int, set[int]]
    neighbors_dict = {name: {int(k): set(int(v) for v in vs) for k, vs in nbrs.items()}
                      for name, nbrs in neighbors_dict_serializable.items()}

    rng = np.random.default_rng(seed)
    perm_panel = panel.copy()

    for net_name, nbrs in neighbors_dict.items():
        all_nodes = list(nbrs.keys())
        perm_mapping = dict(zip(all_nodes, rng.permutation(all_nodes)))
        perm_nbrs = {perm_mapping.get(k, k): {perm_mapping.get(v, v) for v in vs}
                     for k, vs in nbrs.items()}

        exposure = compute_network_exposure(user_monthly_scores, perm_nbrs, users, months)
        col = f"{net_name}_exposure"
        if col in perm_panel.columns:
            perm_panel[col] = perm_panel.apply(
                lambda row, _exp=exposure: _exp.get(
                    (int(row["author_id"]), row["month"]), 0.0
                ),
                axis=1,
            )

    perm_result = run_panel_regression(perm_panel, f"perm_{perm_idx}")
    return {k: v for k, v in perm_result.items() if k.startswith("b_") and "exposure" in k}


def run_permutation_test(
    panel: pd.DataFrame,
    neighbors_dict: dict[str, dict[int, set[int]]],
    user_monthly_scores: dict[tuple[int, str], float],
    users: set[int],
    months: list[str],
    n_perms: int = 200,
) -> dict:
    """Permutation test: shuffle network edges and re-run regression."""
    print(f"\n  Running permutation test ({n_perms} permutations, {_N_WORKERS} workers)…")

    # Get observed coefficients
    obs_result = run_panel_regression(panel, "observed")
    obs_betas = {k: v for k, v in obs_result.items() if k.startswith("b_") and "exposure" in k}

    perm_betas: dict[str, list[float]] = {k: [] for k in obs_betas}

    # Serialize neighbors for pickling (convert sets to lists)
    nbrs_serializable = {
        name: {k: list(vs) for k, vs in nbrs.items()}
        for name, nbrs in neighbors_dict.items()
    }

    # Pre-generate seeds for reproducibility
    master_rng = np.random.default_rng(42)
    seeds = master_rng.integers(0, 2**31, size=n_perms).tolist()

    panel_values = panel.values
    panel_columns = panel.columns.tolist()
    panel_dtypes = {col: str(panel[col].dtype) for col in panel.columns}
    users_list = list(users)

    args_list = [
        (i, seeds[i], panel_values, panel_columns, panel_dtypes, nbrs_serializable,
         user_monthly_scores, users_list, months)
        for i in range(n_perms)
    ]

    with ProcessPoolExecutor(max_workers=_N_WORKERS) as executor:
        results = list(executor.map(_run_single_permutation, args_list, chunksize=4))

    for perm_result in results:
        for k in perm_betas:
            perm_betas[k].append(perm_result.get(k, 0.0))

    # Compute permutation p-values
    perm_pvalues = {}
    for k, obs_val in obs_betas.items():
        perm_dist = np.array(perm_betas[k])
        p = float(np.mean(np.abs(perm_dist) >= np.abs(obs_val)))
        perm_pvalues[k.replace("b_", "perm_p_")] = p
        print(f"    {k}: obs={obs_val:.4f}, perm_p={p:.4f}")

    return perm_pvalues


def main():
    print("=" * 60)
    print("PHASE 3b: Social Contagion Model (H2)")
    print("=" * 60)

    # Load treatment date
    with open(DATA_PROCESSED / "treatment_dates.json") as f:
        td = json.load(f)
    t0 = datetime.datetime.fromisoformat(td["T0"])

    # Load scores
    scores = pl.read_parquet(DATA_PROCESSED / "siege_scores.parquet")
    scores = scores.filter(pl.col("author_id") != ZEIGER_MEMBER_ID)

    # Build panel
    print("\nBuilding user-month panel…")
    panel = build_panel(scores, t0)
    print(f"  Panel size: {len(panel)} user-months")

    # Load networks
    print("\nLoading networks…")
    networks = {}
    for name, directed in [("dm", False), ("forum", False), ("reputation", True)]:
        networks[name] = load_network_neighbors(name, directed=directed)

    # Compute user-month scores lookup
    df_monthly = scores.filter(
        pl.col("date").is_not_null()
    ).with_columns(
        pl.col("date").dt.strftime("%Y-%m").alias("month")
    )
    user_monthly = (
        df_monthly.group_by(["author_id", "month"])
        .agg(pl.col("siege_keyword_score").mean().alias("score"))
    )
    user_monthly_scores = {
        (int(r["author_id"]), r["month"]): float(r["score"])
        for r in user_monthly.iter_rows(named=True)
    }
    users = set(panel["author_id"].unique())
    months = sorted(panel["month"].unique())

    # Compute network exposure for each network
    print("\nComputing network exposure…")
    for net_name, nbrs in networks.items():
        exposure = compute_network_exposure(user_monthly_scores, nbrs, users, months)
        panel[f"{net_name}_exposure"] = panel.apply(
            lambda row: exposure.get((int(row["author_id"]), row["month"]), 0.0),
            axis=1,
        )
        print(f"  {net_name}: mean exposure = {panel[f'{net_name}_exposure'].mean():.4f}")

    # Run panel regression
    print("\nRunning panel regression…")
    result = run_panel_regression(panel, "contagion_main")

    # Run permutation test (reduced iterations for speed)
    perm_results = run_permutation_test(
        panel, networks, user_monthly_scores, users, months, n_perms=200
    )
    result.update(perm_results)

    # Save results
    with open(RESULTS_DIR / "contagion_results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n✓ Contagion model results saved.")


if __name__ == "__main__":
    main()
