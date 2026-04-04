"""27 – Sub-Theme Disaggregation (H20).

Breaks the monolithic "Siege rhetoric" signal into thematic sub-groups
and tracks each independently.  If one sub-theme (e.g. accelerationism)
transfers from IM to /pol/ while another (e.g. Atomwaffen-specific) does
not, the aggregated Granger/ITS tests may wash out the real signal.

Sub-themes
----------
1. accelerationism — accelerate, boogaloo, collapse, insurrection
2. mason_core      — james mason, siege, read siege, universal order, siegepill
3. atomwaffen_org  — atomwaffen, skull mask, american futurist, antipodean resistance
4. violence        — day of the rope, race war, rahowa, total attack, lone wolf,
                     leaderless resistance, armed struggle, propaganda of the deed
5. enemy_framing   — zog, zionist occupation, jew-capitalist, system pig, race traitor

For each sub-theme we run:
- Weekly prevalence time series on IM and /pol/.
- ITS at T0 (Siege publication) on /pol/.
- Granger causality IM→/pol/ and reverse.
"""

from __future__ import annotations

import json
import re

import numpy as np
import polars as pl
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests

from utils import (
    DATA_PROCESSED, RESULTS_DIR, ZEIGER_MEMBER_ID,
    setup_plot_style, save_figure, FIGSIZE_WIDE, CB_PALETTE,
)
import matplotlib.pyplot as plt

# ── Sub-theme definitions ─────────────────────────────────────────────
SUBTHEMES: dict[str, list[str]] = {
    "accelerationism": [
        r"\baccelerat(?:e|ion|ionism|ionist)\b",
        r"\bboogaloo\b",
        r"\bcollapse\b",
        r"\binsurrection\b",
    ],
    "mason_core": [
        r"\bjames\s+mason\b",
        r"\bread\s+siege\b",
        r"\bsiege\s*culture\b",
        r"\bsiegepill(?:ed)?\b",
        r"\buniversal\s+order\b",
        r"\bjoseph\s+tommasi\b",
        r"\btommasi\b",
    ],
    "atomwaffen_org": [
        r"\batomwaffen\b",
        r"\bskull\s*mask\b",
        r"\bamerican\s+futurist\b",
        r"\bantipodean\s+resistance\b",
        r"\bnslf\b",
    ],
    "violence": [
        r"\bday\s+of\s+the\s+rope\b",
        r"\brace\s+war\b",
        r"\brahowa\b",
        r"\btotal\s+attack\b",
        r"\blone\s+wolf\b",
        r"\bleaderless\s+resistance\b",
        r"\barmed\s+struggle\b",
        r"\bpropaganda\s+of\s+the\s+deed\b",
        r"\bpolitical\s+terror\b",
    ],
    "enemy_framing": [
        r"\bzog\b",
        r"\bzionist\s+occupation\s+government\b",
        r"\bjew[- ]?capitalist\b",
        r"\bsystem\s+pig\b",
        r"\brace\s+trait(?:or|ors?)\b",
        r"\banti[- ]?system\b",
    ],
}


def _compile_subtheme(patterns: list[str]) -> re.Pattern:
    """Compile a sub-theme's patterns into a single regex."""
    combined = "|".join(f"(?:{p})" for p in patterns)
    return re.compile(combined, re.IGNORECASE)


def score_subthemes(df: pl.DataFrame) -> pl.DataFrame:
    """Add boolean columns for each sub-theme to the dataframe."""
    compiled = {name: _compile_subtheme(pats) for name, pats in SUBTHEMES.items()}

    # Vectorised-ish: extract texts, score in Python, add columns
    texts = df["text"].to_list()
    cols = {name: [] for name in SUBTHEMES}

    for text in texts:
        t = text if text else ""
        for name, pat in compiled.items():
            cols[name].append(bool(pat.search(t)))

    for name, values in cols.items():
        df = df.with_columns(pl.Series(f"st_{name}", values))
    return df


def subtheme_weekly(
    df: pl.DataFrame, platform: str,
) -> dict[str, pl.DataFrame]:
    """Build weekly prevalence for each sub-theme."""
    result = {}
    for name in SUBTHEMES:
        col = f"st_{name}"
        weekly = (
            df.filter(pl.col("date").is_not_null())
            .with_columns(pl.col("date").dt.truncate("1w").alias("week"))
            .group_by("week")
            .agg([
                pl.col(col).cast(pl.Int32).sum().alias("hits"),
                pl.len().alias("total"),
            ])
            .with_columns(
                (pl.col("hits") / pl.col("total").cast(pl.Float64))
                .alias("prevalence")
            )
            .sort("week")
        )
        result[name] = weekly
    return result


def run_subtheme_its(weekly: pl.DataFrame, t0, label: str) -> dict:
    """ITS for a sub-theme on /pol/ at T0."""
    # Align timezone with the week column
    week_dtype = weekly["week"].dtype
    if hasattr(week_dtype, "time_zone") and week_dtype.time_zone:  # type: ignore[union-attr]
        import datetime as _dt
        t0_cmp = t0 if t0.tzinfo else t0.replace(tzinfo=_dt.timezone.utc)
    else:
        t0_cmp = t0.replace(tzinfo=None) if hasattr(t0, 'tzinfo') and t0.tzinfo else t0
    weekly = weekly.with_columns([
        (pl.col("week") >= t0_cmp).cast(pl.Int8).alias("post_treatment"),
    ])

    pdf = weekly.to_pandas()
    pdf["time_idx"] = range(len(pdf))

    t0_row = pdf.loc[pdf["post_treatment"] == 1]
    if len(t0_row) == 0:
        return {"label": label, "error": "no post-treatment weeks"}
    shift = t0_row["time_idx"].iloc[0]
    pdf["time_centered"] = pdf["time_idx"] - shift
    pdf["time_x_post"] = pdf["time_centered"] * pdf["post_treatment"]

    y = pdf["prevalence"].values
    X = sm.add_constant(
        pdf[["time_centered", "post_treatment", "time_x_post"]].values
    )

    if len(y) < 10 or np.std(y) < 1e-10:
        return {"label": label, "error": "insufficient variation"}

    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    return {
        "label": label,
        "n_weeks": int(len(pdf)),
        "b_level": float(model.params[2]),
        "b_slope": float(model.params[3]),
        "p_level": float(model.pvalues[2]),
        "p_slope": float(model.pvalues[3]),
        "r_squared": float(model.rsquared),
    }


def run_subtheme_granger(
    im_weekly: pl.DataFrame,
    pol_weekly: pl.DataFrame,
    max_lag: int = 4,
) -> dict:
    """Granger causality test for a sub-theme between platforms."""
    paired = (
        im_weekly.select([
            pl.col("week"),
            pl.col("prevalence").alias("im"),
        ]).join(
            pol_weekly.select([
                pl.col("week"),
                pl.col("prevalence").alias("pol"),
            ]),
            on="week",
            how="inner",
        ).sort("week")
    )

    if paired.height < 20:
        return {"error": "insufficient overlapping weeks",
                "n_weeks": paired.height}

    data = paired.select(["pol", "im"]).to_pandas().dropna().values
    if len(data) < 20 or np.std(data[:, 0]) < 1e-10:
        return {"error": "insufficient variation"}

    try:
        gc = grangercausalitytests(data, maxlag=max_lag, verbose=False)
        best_lag = min(
            gc.keys(),
            key=lambda k: gc[k][0]["ssr_ftest"][1],
        )
        best_p = gc[best_lag][0]["ssr_ftest"][1]
        best_f = gc[best_lag][0]["ssr_ftest"][0]
        return {
            "direction": "im_to_pol",
            "best_lag": int(best_lag),
            "f_statistic": float(best_f),
            "p_value": float(best_p),
            "significant_05": bool(best_p < 0.05),
            "n_weeks": int(len(data)),
        }
    except Exception as e:
        return {"error": str(e)}


def plot_subtheme_timeseries(
    im_weeklies: dict[str, pl.DataFrame],
    pol_weeklies: dict[str, pl.DataFrame],
    filename: str,
):
    """Multi-panel time series of sub-theme prevalence on both platforms."""
    setup_plot_style()
    n_themes = len(SUBTHEMES)
    fig, axes = plt.subplots(n_themes, 1, figsize=(14, 3 * n_themes),
                             sharex=True)
    if n_themes == 1:
        axes = [axes]

    for ax, name in zip(axes, SUBTHEMES):
        if name in im_weeklies:
            im_pdf = im_weeklies[name].to_pandas()
            ax.plot(im_pdf["week"], im_pdf["prevalence"],
                    label="IM", color=CB_PALETTE[0], alpha=0.7)
        if name in pol_weeklies:
            pol_pdf = pol_weeklies[name].to_pandas()
            ax.plot(pol_pdf["week"], pol_pdf["prevalence"],
                    label="/pol/", color=CB_PALETTE[2], alpha=0.7)
        ax.set_ylabel("Prevalence")
        ax.set_title(name.replace("_", " ").title())
        ax.legend(loc="upper left", fontsize=8)

    axes[-1].set_xlabel("Date")
    fig.suptitle("Sub-Theme Prevalence: IM vs /pol/", fontsize=14, y=1.01)
    fig.tight_layout()
    save_figure(fig, filename)


def main():
    print("=" * 60)
    print("H20: Sub-Theme Disaggregation")
    print("=" * 60)

    im_path = DATA_PROCESSED / "siege_scores.parquet"
    pol_path = DATA_PROCESSED / "pol_siege_scores.parquet"
    if not im_path.exists() or not pol_path.exists():
        print("  ✗ Missing scored data.")
        return

    im = pl.read_parquet(im_path).filter(
        (pl.col("channel") == "forum")
        & (pl.col("author_id") != ZEIGER_MEMBER_ID)
    )
    pol = pl.read_parquet(pol_path)
    print(f"  IM posts: {im.height:,}  |  /pol/ posts: {pol.height:,}")

    # Score sub-themes
    print("\n  Scoring sub-themes on IM…")
    im = score_subthemes(im)
    print("  Scoring sub-themes on /pol/…")
    pol = score_subthemes(pol)

    # Weekly prevalence
    im_weeklies = subtheme_weekly(im, "im")
    pol_weeklies = subtheme_weekly(pol, "pol")

    # Load T0
    import datetime
    td_path = DATA_PROCESSED / "treatment_dates.json"
    if td_path.exists():
        with open(td_path) as f:
            t0 = datetime.datetime.fromisoformat(json.load(f)["T0"])
    else:
        t0 = datetime.datetime(2015, 6, 3, tzinfo=datetime.timezone.utc)

    results: dict = {"subthemes": list(SUBTHEMES.keys())}

    for name in SUBTHEMES:
        print(f"\n  ── {name} ──")
        im_hits = im[f"st_{name}"].sum()
        pol_hits = pol[f"st_{name}"].sum()
        print(f"    IM hits: {im_hits:,}  |  /pol/ hits: {pol_hits:,}")

        st_result: dict = {
            "im_hits": int(im_hits),
            "pol_hits": int(pol_hits),
        }

        # ITS on /pol/
        its = run_subtheme_its(pol_weeklies[name], t0, f"{name}_its")
        st_result["its"] = its
        if "error" not in its:
            sig = "★" if its["p_level"] < 0.05 else " "
            print(f"    ITS: β₂={its['b_level']:.6f} "
                  f"p={its['p_level']:.4f} {sig}")

        # Granger
        gc = run_subtheme_granger(im_weeklies[name], pol_weeklies[name])
        st_result["granger_im_to_pol"] = gc
        if "error" not in gc:
            sig = "★" if gc["significant_05"] else " "
            print(f"    Granger IM→pol: F={gc['f_statistic']:.2f} "
                  f"p={gc['p_value']:.4f} lag={gc['best_lag']} {sig}")

        results[name] = st_result

    # Plot
    plot_subtheme_timeseries(im_weeklies, pol_weeklies, "subtheme_diffusion")

    with open(RESULTS_DIR / "subtheme_diffusion_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\n✓ Sub-theme diffusion results saved.")


if __name__ == "__main__":
    main()
