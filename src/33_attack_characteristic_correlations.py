"""33 – Attack-Characteristic Correlations with Apocalypticism.

Tests whether characteristics of mass-casualty events predict the
magnitude and direction of post-attack apocalyptic rhetoric on /pol/.

Analyses
--------
1. **Severity correlations** – Pearson/Spearman of per-event β₂ with
   killed, injured, and total_casualties.
2. **Ideology group comparison** – Kruskal-Wallis + pairwise Mann-Whitney
   comparing β₂ across ideology categories.
3. **Domestic vs. international** – Mann-Whitney test comparing β₂ for
   US-domestic events vs. international events.
4. **Online nexus** – Mann-Whitney comparing β₂ for events where
   perpetrator had known online presence vs. not.
5. **Geographic heterogeneity** – β₂ by country, grouped by region.
6. **Multiple regression** – OLS of β₂ on log_killed, domestic,
   online_nexus, and ideology dummies.

Output
------
``results/attack_correlations_results.json``
``figures/apoc_corr_*.{png,pdf}``
"""

from __future__ import annotations

import json
import importlib
from pathlib import Path

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

# ── Re-use ITS building blocks ────────────────────────────────────────
_its = importlib.import_module("31_apocalypticism_its")
build_daily_series = _its.build_daily_series
build_event_window = _its.build_event_window
run_its_regression = _its.run_its_regression


# ══════════════════════════════════════════════════════════════════════
# Helper: merge per-event ITS betas with event metadata
# ══════════════════════════════════════════════════════════════════════

def build_event_beta_df(
    per_event_results: list[dict],
    events: pl.DataFrame,
) -> pd.DataFrame:
    """Join per-event ITS β₂ with full event metadata.

    Parameters
    ----------
    per_event_results : list[dict]
        ``results["per_event"]`` from apocalypticism_its_results.json.
    events : pl.DataFrame
        Full event catalogue with all characteristics.

    Returns
    -------
    pd.DataFrame with columns from both sources.
    """
    # Keep only valid (no-error) per-event results
    valid = [r for r in per_event_results if "error" not in r and "b_level" in r]
    if not valid:
        return pd.DataFrame()

    betas_df = pd.DataFrame(valid)
    events_pdf = events.to_pandas()

    # Merge on event_name / label
    merged = betas_df.merge(
        events_pdf,
        left_on="label",
        right_on="event_name",
        how="inner",
        suffixes=("_its", "_ev"),
    )

    # Resolve suffixed columns: prefer the richer event-catalogue version
    for base in ("ideology", "killed", "injured", "online_nexus",
                 "total_casualties", "location_country", "domestic"):
        ev_col = f"{base}_ev"
        its_col = f"{base}_its"
        if ev_col in merged.columns:
            merged[base] = merged[ev_col]
            merged.drop(columns=[ev_col], inplace=True)
            if its_col in merged.columns:
                merged.drop(columns=[its_col], inplace=True)

    return merged


# ══════════════════════════════════════════════════════════════════════
# 1. Severity correlations
# ══════════════════════════════════════════════════════════════════════

def severity_correlations(df: pd.DataFrame) -> dict:
    """Pearson and Spearman correlations of β₂ with casualty counts."""
    results = {}
    for col in ("killed", "injured", "total_casualties"):
        if col not in df.columns:
            continue
        valid = df[["b_level", col]].dropna()
        if len(valid) < 5:
            results[col] = {"error": "insufficient data"}
            continue
        if valid[col].nunique() < 2:
            results[col] = {"error": "no variation in predictor"}
            continue

        pr, pp = stats.pearsonr(valid["b_level"], valid[col])
        sr, sp = stats.spearmanr(valid["b_level"], valid[col])

        results[col] = {
            "n": int(len(valid)),
            "pearson_r": float(pr),
            "pearson_p": float(pp),
            "spearman_rho": float(sr),
            "spearman_p": float(sp),
        }

    return results


# ══════════════════════════════════════════════════════════════════════
# 2. Ideology group comparison
# ══════════════════════════════════════════════════════════════════════

def ideology_comparison(df: pd.DataFrame) -> dict:
    """Kruskal-Wallis test across ideology groups, plus descriptives."""
    # Filter to mass-violence only (non-violence has ideology=N/A)
    mdf = df[df["event_category"] == "mass_violence"] if "event_category" in df.columns else df

    groups = {}
    for ideology, subdf in mdf.groupby("ideology"):
        betas = subdf["b_level"].dropna().tolist()
        if len(betas) >= 2:
            groups[ideology] = betas

    if len(groups) < 2:
        return {"error": "fewer than 2 ideology groups with ≥2 events"}

    # Kruskal-Wallis (non-parametric)
    group_arrays = list(groups.values())
    h_stat, kw_p = stats.kruskal(*group_arrays)

    descriptives = {}
    for ideology, betas in groups.items():
        descriptives[ideology] = {
            "n": len(betas),
            "mean_beta": float(np.mean(betas)),
            "median_beta": float(np.median(betas)),
            "std_beta": float(np.std(betas, ddof=1)) if len(betas) > 1 else 0.0,
        }

    # Pairwise Mann-Whitney for groups with ≥3 events
    pairwise = []
    group_names = sorted(groups.keys())
    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            a, b = group_names[i], group_names[j]
            if len(groups[a]) < 3 or len(groups[b]) < 3:
                continue
            u_stat, mw_p = stats.mannwhitneyu(
                groups[a], groups[b], alternative="two-sided"
            )
            pairwise.append({
                "group_a": a,
                "group_b": b,
                "u_stat": float(u_stat),
                "p_value": float(mw_p),
                "n_a": len(groups[a]),
                "n_b": len(groups[b]),
            })

    return {
        "kruskal_wallis_H": float(h_stat),
        "kruskal_wallis_p": float(kw_p),
        "n_groups": len(groups),
        "descriptives": descriptives,
        "pairwise_mannwhitney": pairwise,
    }


# ══════════════════════════════════════════════════════════════════════
# 3. Domestic vs. international
# ══════════════════════════════════════════════════════════════════════

def domestic_comparison(df: pd.DataFrame) -> dict:
    """Mann-Whitney comparison of β₂ for domestic vs. international."""
    mdf = df[df["event_category"] == "mass_violence"] if "event_category" in df.columns else df

    domestic = mdf[mdf["domestic"] == True]["b_level"].dropna().tolist()  # noqa: E712
    international = mdf[mdf["domestic"] == False]["b_level"].dropna().tolist()  # noqa: E712

    if len(domestic) < 3 or len(international) < 3:
        return {"error": "too few events in one group"}

    u_stat, p = stats.mannwhitneyu(domestic, international, alternative="two-sided")

    return {
        "domestic_n": len(domestic),
        "domestic_mean": float(np.mean(domestic)),
        "domestic_median": float(np.median(domestic)),
        "international_n": len(international),
        "international_mean": float(np.mean(international)),
        "international_median": float(np.median(international)),
        "mannwhitney_U": float(u_stat),
        "mannwhitney_p": float(p),
    }


# ══════════════════════════════════════════════════════════════════════
# 4. Online nexus comparison
# ══════════════════════════════════════════════════════════════════════

def online_nexus_comparison(df: pd.DataFrame) -> dict:
    """Mann-Whitney comparison: online-nexus perpetrators vs. not."""
    mdf = df[df["event_category"] == "mass_violence"] if "event_category" in df.columns else df

    col = "online_nexus"
    online = mdf[mdf[col] == True]["b_level"].dropna().tolist()  # noqa: E712
    offline = mdf[mdf[col] == False]["b_level"].dropna().tolist()  # noqa: E712

    if len(online) < 3 or len(offline) < 3:
        return {"error": "too few events in one group"}

    u_stat, p = stats.mannwhitneyu(online, offline, alternative="two-sided")

    return {
        "online_n": len(online),
        "online_mean": float(np.mean(online)),
        "online_median": float(np.median(online)),
        "offline_n": len(offline),
        "offline_mean": float(np.mean(offline)),
        "offline_median": float(np.median(offline)),
        "mannwhitney_U": float(u_stat),
        "mannwhitney_p": float(p),
    }


# ══════════════════════════════════════════════════════════════════════
# 5. Geographic heterogeneity
# ══════════════════════════════════════════════════════════════════════

REGION_MAP = {
    "US": "North America", "Canada": "North America",
    "UK": "Europe", "France": "Europe", "Germany": "Europe",
    "Norway": "Europe", "Sweden": "Europe", "Finland": "Europe",
    "Denmark": "Europe", "Netherlands": "Europe", "Belgium": "Europe",
    "Spain": "Europe", "Italy": "Europe", "Czech Republic": "Europe",
    "Switzerland": "Europe", "Austria": "Europe",
    "Australia": "Oceania", "New Zealand": "Oceania",
    "Japan": "Asia", "South Korea": "Asia", "Israel": "Middle East",
    "Turkey": "Middle East", "Sri Lanka": "Asia",
    "Brazil": "Americas", "Mexico": "Americas",
}


def geographic_heterogeneity(df: pd.DataFrame) -> dict:
    """β₂ descriptives by country and region."""
    mdf = df[df["event_category"] == "mass_violence"] if "event_category" in df.columns else df

    country_col = "location_country"

    by_country = {}
    for country, subdf in mdf.groupby(country_col):
        betas = subdf["b_level"].dropna().tolist()
        if betas:
            by_country[country] = {
                "n": len(betas),
                "mean_beta": float(np.mean(betas)),
                "median_beta": float(np.median(betas)),
            }

    # Aggregate by region
    mdf = mdf.copy()
    mdf["region"] = mdf[country_col].map(REGION_MAP).fillna("Other")
    by_region = {}
    for region, subdf in mdf.groupby("region"):
        betas = subdf["b_level"].dropna().tolist()
        if len(betas) >= 2:
            by_region[region] = {
                "n": len(betas),
                "mean_beta": float(np.mean(betas)),
                "median_beta": float(np.median(betas)),
                "std_beta": float(np.std(betas, ddof=1)) if len(betas) > 1 else 0.0,
            }

    # Kruskal-Wallis across regions (if ≥3 groups)
    region_groups = {r: mdf[mdf["region"] == r]["b_level"].dropna().tolist()
                     for r in by_region if by_region[r]["n"] >= 2}
    kw_result = None
    if len(region_groups) >= 3:
        h, p = stats.kruskal(*region_groups.values())
        kw_result = {"H": float(h), "p": float(p), "n_groups": len(region_groups)}

    return {
        "by_country": by_country,
        "by_region": by_region,
        "region_kruskal_wallis": kw_result,
    }


# ══════════════════════════════════════════════════════════════════════
# 6. Multiple regression
# ══════════════════════════════════════════════════════════════════════

def multiple_regression(df: pd.DataFrame) -> dict:
    """OLS of β₂ on event characteristics.

    Predictors: log(1 + killed), domestic, online_nexus, ideology dummies.
    """
    mdf = df[df["event_category"] == "mass_violence"].copy() if "event_category" in df.columns else df.copy()

    killed_col = "killed"
    nexus_col = "online_nexus"

    mdf = mdf.dropna(subset=[killed_col, "domestic", nexus_col, "ideology", "b_level"])
    if mdf.empty:
        return {"error": "no valid rows after dropping NaN"}

    mdf["log_killed"] = np.log1p(mdf[killed_col].astype(float))
    mdf["domestic_int"] = mdf["domestic"].astype(int)
    mdf["nexus_int"] = mdf[nexus_col].astype(int)

    # Ideology dummies (drop first for identification)
    id_dummies = pd.get_dummies(mdf["ideology"], prefix="ideo", drop_first=True, dtype=float)

    predictors = mdf[["log_killed", "domestic_int", "nexus_int"]].copy()
    predictors = pd.concat([predictors, id_dummies], axis=1)
    predictors = sm.add_constant(predictors)

    y = mdf["b_level"].values
    valid_mask = ~np.isnan(y) & predictors.notna().all(axis=1)
    y = y[valid_mask]
    X = predictors[valid_mask]

    if len(y) < X.shape[1] + 5:
        return {"error": "too few observations for regression"}

    try:
        model = sm.OLS(y, X).fit(cov_type="HC1")
    except Exception as e:
        return {"error": str(e)}

    coefs = {}
    for name, b, se, p in zip(model.model.exog_names, model.params,
                               model.bse, model.pvalues):
        coefs[name] = {
            "coef": float(b),
            "se": float(se),
            "p": float(p),
        }

    return {
        "n_obs": int(len(y)),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "f_pvalue": float(model.f_pvalue) if model.f_pvalue is not None else None,
        "coefficients": coefs,
    }


# ══════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════

def plot_severity_scatter(df: pd.DataFrame, filename: str):
    """Scatter of β₂ vs. killed, injured, total casualties."""
    setup_plot_style()

    cols = []
    for base in ("killed", "injured", "total_casualties"):
        if base in df.columns:
            cols.append((base, base))

    if not cols:
        return

    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 5))
    if len(cols) == 1:
        axes = [axes]

    for ax, (label, c) in zip(axes, cols):
        valid = df[["b_level", c]].dropna()
        ax.scatter(valid[c], valid["b_level"],
                   color=CB_PALETTE[0], alpha=0.6, s=30, edgecolors="white")
        # Regression line
        if len(valid) >= 5:
            z = np.polyfit(valid[c].values, valid["b_level"].values, 1)
            xline = np.linspace(valid[c].min(), valid[c].max(), 100)
            ax.plot(xline, np.polyval(z, xline), "--",
                    color=CB_PALETTE[2], linewidth=2, alpha=0.8)
            sr, sp = stats.spearmanr(valid["b_level"], valid[c])
            ax.set_title(f"β₂ vs {label}\nρ = {sr:.3f} (p = {sp:.3f})")
        else:
            ax.set_title(f"β₂ vs {label}")

        ax.set_xlabel(label.replace("_", " ").title())
        ax.set_ylabel("β₂ (level change)")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)

    fig.suptitle("Severity–Apocalypticism Correlations", fontsize=14)
    fig.tight_layout()
    save_figure(fig, filename)


def plot_ideology_boxplot(df: pd.DataFrame, filename: str):
    """Box plot of β₂ by ideology group."""
    setup_plot_style()

    mdf = df[df["event_category"] == "mass_violence"] if "event_category" in df.columns else df
    groups = mdf.groupby("ideology")["b_level"].apply(list).to_dict()
    groups = {k: v for k, v in groups.items() if len(v) >= 2}

    if len(groups) < 2:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    labels = sorted(groups.keys(), key=lambda k: np.median(groups[k]))
    data = [groups[k] for k in labels]

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
    for patch, color in zip(bp["boxes"],
                            [CB_PALETTE[i % len(CB_PALETTE)] for i in range(len(labels))]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("β₂ (level change)")
    ax.set_xlabel("Ideology")
    ax.set_title("Post-Attack Apocalypticism by Perpetrator Ideology")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    save_figure(fig, filename)


def plot_domestic_comparison(df: pd.DataFrame, filename: str):
    """Side-by-side boxplot: domestic vs international."""
    setup_plot_style()
    mdf = df[df["event_category"] == "mass_violence"] if "event_category" in df.columns else df

    domestic = mdf[mdf["domestic"] == True]["b_level"].dropna().tolist()  # noqa: E712
    international = mdf[mdf["domestic"] == False]["b_level"].dropna().tolist()  # noqa: E712

    if len(domestic) < 2 or len(international) < 2:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot([domestic, international],
                    labels=["Domestic (US)", "International"],
                    patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor(CB_PALETTE[0])
    bp["boxes"][1].set_facecolor(CB_PALETTE[2])
    for box in bp["boxes"]:
        box.set_alpha(0.7)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("β₂ (level change)")
    ax.set_title("Domestic vs International Events")
    fig.tight_layout()
    save_figure(fig, filename)


def plot_regression_forest(reg_result: dict, filename: str):
    """Forest plot of multiple-regression coefficients."""
    setup_plot_style()

    coefs = reg_result.get("coefficients", {})
    if not coefs or "error" in reg_result:
        return

    # Skip the constant
    names = [k for k in coefs if k != "const"]
    vals = [coefs[k]["coef"] for k in names]
    ses = [coefs[k]["se"] for k in names]
    ci_lo = [v - 1.96 * s for v, s in zip(vals, ses)]
    ci_hi = [v + 1.96 * s for v, s in zip(vals, ses)]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.5)))
    y_pos = list(range(len(names)))

    colors = [CB_PALETTE[2] if coefs[n]["p"] < 0.05 else CB_PALETTE[0] for n in names]
    ax.errorbar(vals, y_pos,
                xerr=[np.array(vals) - np.array(ci_lo),
                      np.array(ci_hi) - np.array(vals)],
                fmt="o", capsize=4, markersize=7,
                color=CB_PALETTE[0], ecolor="gray")
    for i, c in enumerate(colors):
        ax.plot(vals[i], y_pos[i], "o", color=c, markersize=7, zorder=5)

    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([n.replace("ideo_", "").replace("_int", "")
                        for n in names], fontsize=10)
    ax.set_xlabel("Coefficient")
    ax.set_title(f"Multiple Regression: Predictors of β₂ "
                 f"(R² = {reg_result['r_squared']:.3f})")

    fig.tight_layout()
    save_figure(fig, filename)


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("STAGE 33: Attack-Characteristic Correlations")
    print("=" * 60)

    results: dict = {}

    # ── Load data ─────────────────────────────────────────────────────
    events_path = DATA_PROCESSED / "mass_casualty_events.parquet"
    its_path = RESULTS_DIR / "apocalypticism_its_results.json"

    if not events_path.exists():
        print(f"  ✗ {events_path.name} not found. Run stage 29 first.")
        return
    if not its_path.exists():
        print(f"  ✗ {its_path.name} not found. Run stage 31 first.")
        return

    events = pl.read_parquet(events_path)
    with open(its_path) as f:
        its_results = json.load(f)

    per_event = its_results.get("per_event", [])
    print(f"  Events in catalogue: {events.height}")
    print(f"  Per-event ITS results: {len(per_event)}")

    # ── Build merged DataFrame ────────────────────────────────────────
    df = build_event_beta_df(per_event, events)
    if df.empty:
        print("  ✗ No valid per-event results to analyze.")
        return

    # Filter to mass-violence only for most analyses
    n_violence = (df["event_category"] == "mass_violence").sum() if "event_category" in df.columns else len(df)
    print(f"  Merged events with valid β₂: {len(df)} "
          f"({n_violence} mass-violence)")

    # ── 1. Severity correlations ──────────────────────────────────────
    print("\n  1. Severity correlations…")
    sev = severity_correlations(df)
    results["severity_correlations"] = sev
    for col, r in sev.items():
        if "error" not in r:
            print(f"    {col}: Pearson r={r['pearson_r']:.4f} "
                  f"(p={r['pearson_p']:.4f}), "
                  f"Spearman ρ={r['spearman_rho']:.4f} "
                  f"(p={r['spearman_p']:.4f})")

    plot_severity_scatter(df, "apoc_corr_severity")

    # ── 2. Ideology comparison ────────────────────────────────────────
    print("\n  2. Ideology group comparison…")
    ideo = ideology_comparison(df)
    results["ideology_comparison"] = ideo
    if "error" not in ideo:
        print(f"    Kruskal-Wallis H={ideo['kruskal_wallis_H']:.4f} "
              f"(p={ideo['kruskal_wallis_p']:.4f})")
        for grp, desc in ideo["descriptives"].items():
            print(f"      {grp}: n={desc['n']}, "
                  f"mean β₂={desc['mean_beta']:.4f}")

    plot_ideology_boxplot(df, "apoc_corr_ideology")

    # ── 3. Domestic vs international ──────────────────────────────────
    print("\n  3. Domestic vs international…")
    dom = domestic_comparison(df)
    results["domestic_comparison"] = dom
    if "error" not in dom:
        print(f"    Domestic: n={dom['domestic_n']}, "
              f"mean β₂={dom['domestic_mean']:.4f}")
        print(f"    International: n={dom['international_n']}, "
              f"mean β₂={dom['international_mean']:.4f}")
        print(f"    Mann-Whitney p={dom['mannwhitney_p']:.4f}")

    plot_domestic_comparison(df, "apoc_corr_domestic")

    # ── 4. Online nexus ───────────────────────────────────────────────
    print("\n  4. Online nexus comparison…")
    nexus = online_nexus_comparison(df)
    results["online_nexus_comparison"] = nexus
    if "error" not in nexus:
        print(f"    Online: n={nexus['online_n']}, "
              f"mean β₂={nexus['online_mean']:.4f}")
        print(f"    Offline: n={nexus['offline_n']}, "
              f"mean β₂={nexus['offline_mean']:.4f}")
        print(f"    Mann-Whitney p={nexus['mannwhitney_p']:.4f}")

    # ── 5. Geographic heterogeneity ───────────────────────────────────
    print("\n  5. Geographic heterogeneity…")
    geo = geographic_heterogeneity(df)
    results["geographic"] = geo
    if geo.get("region_kruskal_wallis"):
        print(f"    Region Kruskal-Wallis H={geo['region_kruskal_wallis']['H']:.4f} "
              f"(p={geo['region_kruskal_wallis']['p']:.4f})")
    for region, desc in geo.get("by_region", {}).items():
        print(f"      {region}: n={desc['n']}, mean β₂={desc['mean_beta']:.4f}")

    # ── 6. Multiple regression ────────────────────────────────────────
    print("\n  6. Multiple regression…")
    reg = multiple_regression(df)
    results["multiple_regression"] = reg
    if "error" not in reg:
        print(f"    R² = {reg['r_squared']:.4f} "
              f"(adj. R² = {reg['adj_r_squared']:.4f})")
        for name, c in reg["coefficients"].items():
            sig = "***" if c["p"] < 0.001 else "**" if c["p"] < 0.01 else "*" if c["p"] < 0.05 else ""
            print(f"      {name}: β={c['coef']:.4f} "
                  f"(p={c['p']:.4f}) {sig}")

    plot_regression_forest(reg, "apoc_corr_regression")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n  ── Summary ──")
    sig_findings = []
    for col, r in sev.items():
        if "error" not in r and r["spearman_p"] < 0.05:
            sig_findings.append(f"severity({col})")
    if "error" not in ideo and ideo["kruskal_wallis_p"] < 0.05:
        sig_findings.append("ideology")
    if "error" not in dom and dom["mannwhitney_p"] < 0.05:
        sig_findings.append("domestic_vs_intl")
    if "error" not in nexus and nexus["mannwhitney_p"] < 0.05:
        sig_findings.append("online_nexus")
    if geo.get("region_kruskal_wallis") and geo["region_kruskal_wallis"]["p"] < 0.05:
        sig_findings.append("region")

    results["significant_predictors"] = sig_findings
    if sig_findings:
        print(f"    Significant (p < 0.05): {', '.join(sig_findings)}")
    else:
        print("    No characteristic significantly predicts β₂ at α = 0.05")

    # ── Save ──────────────────────────────────────────────────────────
    out_path = RESULTS_DIR / "attack_correlations_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✓ Attack-characteristic correlations complete. "
          f"Saved to {out_path.name}")


if __name__ == "__main__":
    main()
