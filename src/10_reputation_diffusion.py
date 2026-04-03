"""10 – Reputation-Mediated Diffusion (H6).

Tests whether high-status users drive faster adoption of Siege rhetoric,
using Cox proportional hazards.
"""

from __future__ import annotations

import json
import datetime

import numpy as np
import networkx as nx
import polars as pl
from lifelines import CoxPHFitter, KaplanMeierFitter

from utils import (
    DATA_PROCESSED, NETWORKS_DIR, RESULTS_DIR,
    ZEIGER_MEMBER_ID, setup_plot_style, save_figure, FIGSIZE_WIDE, CB_PALETTE,
)
import matplotlib.pyplot as plt


def main():
    print("=" * 60)
    print("PHASE 7: Reputation-Mediated Diffusion (H6)")
    print("=" * 60)

    with open(DATA_PROCESSED / "treatment_dates.json") as f:
        td = json.load(f)
    t0 = datetime.datetime.fromisoformat(td["T0"])

    # Load data
    scores = pl.read_parquet(DATA_PROCESSED / "siege_scores.parquet")
    scores = scores.filter(pl.col("author_id") != ZEIGER_MEMBER_ID)
    members = pl.read_parquet(DATA_PROCESSED / "members.parquet")

    # ── Build forum co-participation network for centrality ───────────
    print("\nComputing network centrality…")
    forum_edges = pl.read_parquet(NETWORKS_DIR / "forum_edgelist.parquet")
    G = nx.Graph()
    for row in forum_edges.iter_rows(named=True):
        G.add_edge(int(row["source"]), int(row["target"]), weight=row["weight"])

    print(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    degree_cent = nx.degree_centrality(G)
    betweenness_cent = nx.betweenness_centrality(G, k=min(500, G.number_of_nodes()))

    centrality_df = pl.DataFrame({
        "author_id": list(degree_cent.keys()),
        "degree_centrality": list(degree_cent.values()),
        "betweenness_centrality": [betweenness_cent.get(n, 0.0) for n in degree_cent.keys()],
    })

    # ── Time-to-first-siege-post (survival analysis) ──────────────────
    print("\nPreparing survival data…")

    # First post date per user
    first_post = (
        scores.filter(pl.col("date").is_not_null())
        .group_by("author_id")
        .agg(pl.col("date").min().alias("first_post_date"))
    )

    # First siege post per user
    first_siege = (
        scores.filter(
            (pl.col("siege_binary") == 1) & pl.col("date").is_not_null()
        )
        .group_by("author_id")
        .agg(pl.col("date").min().alias("first_siege_date"))
    )

    # All users who ever posted
    all_users = first_post.join(first_siege, on="author_id", how="left")

    # Compute duration: from first post to first siege post (or censoring)
    max_date = scores.filter(pl.col("date").is_not_null())["date"].max()

    all_users = all_users.with_columns([
        pl.when(pl.col("first_siege_date").is_not_null())
        .then(
            (pl.col("first_siege_date") - pl.col("first_post_date")).dt.total_days()
        )
        .otherwise(
            (pl.lit(max_date) - pl.col("first_post_date")).dt.total_days()
        )
        .alias("duration_days"),
        pl.col("first_siege_date").is_not_null().cast(pl.Int8).alias("event"),
    ])

    # Remove zero/negative durations
    all_users = all_users.filter(pl.col("duration_days") > 0)

    # Merge centrality
    survival = all_users.join(centrality_df, on="author_id", how="left")
    survival = survival.with_columns([
        pl.col("degree_centrality").fill_null(0.0),
        pl.col("betweenness_centrality").fill_null(0.0),
    ])

    # Merge member info
    mem_join = members.select(["member_id", "pp_reputation_points"]).rename(
        {"member_id": "author_id"}
    )
    survival = survival.join(mem_join, on="author_id", how="left")
    survival = survival.with_columns(
        pl.col("pp_reputation_points").fill_null(0).cast(pl.Float64).alias("reputation")
    )

    print(f"  Survival dataset: {survival.height} users, "
          f"{survival.filter(pl.col('event') == 1).height} events")

    # ── Cox proportional hazards ──────────────────────────────────────
    print("\nFitting Cox PH model…")
    surv_pdf = survival.select([
        "duration_days", "event", "degree_centrality",
        "betweenness_centrality", "reputation",
    ]).to_pandas()

    # Standardize covariates
    for col in ["degree_centrality", "betweenness_centrality", "reputation"]:
        std = surv_pdf[col].std()
        if std > 0:
            surv_pdf[col] = (surv_pdf[col] - surv_pdf[col].mean()) / std

    results = {}
    try:
        cph = CoxPHFitter()
        cph.fit(
            surv_pdf,
            duration_col="duration_days",
            event_col="event",
        )
        cph.print_summary()

        for var in ["degree_centrality", "betweenness_centrality", "reputation"]:
            s = cph.summary.loc[var]
            results[var] = {
                "coef": float(s["coef"]),
                "exp_coef": float(s["exp(coef)"]),
                "se": float(s["se(coef)"]),
                "z": float(s["z"]),
                "p": float(s["p"]),
                "ci_lower": float(s["coef lower 95%"]),
                "ci_upper": float(s["coef upper 95%"]),
            }
    except Exception as e:
        print(f"  ⚠ Cox PH failed: {e}")
        results["error"] = str(e)

    # ── Plot: Survival curves by centrality group ─────────────────────
    print("\nPlotting survival curves by centrality…")
    setup_plot_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    median_cent = survival["degree_centrality"].median()
    for label, filt, color in [
        ("High centrality", pl.col("degree_centrality") >= median_cent, CB_PALETTE[0]),
        ("Low centrality", pl.col("degree_centrality") < median_cent, CB_PALETTE[2]),
    ]:
        subset = survival.filter(filt).to_pandas()
        kmf = KaplanMeierFitter()
        kmf.fit(subset["duration_days"], event_observed=subset["event"], label=label)
        kmf.plot_survival_function(ax=ax, color=color, linewidth=1.5)

    ax.set_xlabel("Days since first post")
    ax.set_ylabel("Survival probability (no siege rhetoric)")
    ax.set_title("Time to First Siege Rhetoric by Network Centrality")
    ax.legend()
    save_figure(fig, "survival_centrality")

    # Save
    with open(RESULTS_DIR / "reputation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ Reputation diffusion results saved.")


if __name__ == "__main__":
    main()
