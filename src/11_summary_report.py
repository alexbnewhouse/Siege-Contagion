"""Generate a narrative summary report from all JSON result files."""

from __future__ import annotations

import json
from pathlib import Path

from utils import RESULTS_DIR


def load_json(name: str) -> dict:
    path = RESULTS_DIR / name
    if not path.exists():
        return {"error": f"File not found: {name}"}
    with open(path) as f:
        return json.load(f)


def format_p(p: float) -> str:
    if p < 0.001:
        return "p < 0.001"
    elif p < 0.01:
        return f"p = {p:.3f}"
    elif p < 0.05:
        return f"p = {p:.3f}"
    else:
        return f"p = {p:.3f}"


def sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    elif p < 0.1:
        return "†"
    return ""


def generate_report():
    lines = []
    lines.append("# Siege Culture Diffusion in Iron March: Summary Report\n")
    lines.append("*Auto-generated from analysis results.*\n")

    # ── H1: Interrupted Time Series ───────────────────────────────────
    lines.append("\n## H1: Interrupted Time Series (Siege as Structural Break)\n")
    its = load_json("its_results.json")
    if "error" not in its:
        for key, val in its.items():
            if isinstance(val, dict) and "b_post_treatment" in val:
                lines.append(f"\n### {val.get('label', key)}\n")
                lines.append(f"- N weeks: {val['n_weeks']}")
                lines.append(f"- Level change (β₂): {val['b_post_treatment']:.4f} "
                             f"({format_p(val['p_post_treatment'])}) {sig_stars(val['p_post_treatment'])}")
                lines.append(f"- Slope change (β₃): {val['b_time_x_post']:.4f} "
                             f"({format_p(val['p_time_x_post'])}) {sig_stars(val['p_time_x_post'])}")
                lines.append(f"- R²: {val['r_squared']:.4f}")
            elif isinstance(val, dict) and "changepoints" in val:
                lines.append(f"\n**Change points ({val.get('label', key)}):** {val['changepoints']}")
    else:
        lines.append(f"*Error: {its.get('error', 'unknown')}*\n")

    # ── H2: Social Contagion ──────────────────────────────────────────
    lines.append("\n\n## H2: Social Contagion in the Interaction Network\n")
    cont = load_json("contagion_results.json")
    if "error" not in cont:
        lines.append(f"- N observations: {cont.get('n_obs', 'N/A')}")
        lines.append(f"- R²: {cont.get('r_squared', 'N/A')}")
        for key in sorted(cont.keys()):
            if key.startswith("b_") and "exposure" in key:
                p_key = key.replace("b_", "p_")
                p_val = cont.get(p_key, 1.0)
                lines.append(f"- {key}: {cont[key]:.4f} ({format_p(p_val)}) {sig_stars(p_val)}")
            if key.startswith("perm_p_"):
                lines.append(f"- Permutation {key}: {cont[key]:.4f}")
    else:
        lines.append(f"*Error: {cont.get('error', 'unknown')}*\n")

    lines.append("\n**Endogeneity warning:** Network contagion models are susceptible to "
                 "homophily confounds. User fixed effects partially address this. "
                 "Permutation p-values provide a robustness check.\n")

    # ── H3: Granger Causality ─────────────────────────────────────────
    lines.append("\n## H3: Zeiger as Ideological Entrepreneur (Granger Causality)\n")
    granger = load_json("granger_results.json")
    if "error" not in granger:
        lines.append("\n### Zeiger → Community\n")
        lines.append("| Lag | F-stat | p-value | Sig. |")
        lines.append("|-----|--------|---------|------|")
        for lag in range(1, 9):
            key = f"zeiger_to_community_lag{lag}"
            if key in granger:
                r = granger[key]
                lines.append(f"| {lag} | {r['f_stat']:.3f} | {r['p_value']:.4f} | {sig_stars(r['p_value'])} |")

        lines.append("\n### Community → Zeiger\n")
        lines.append("| Lag | F-stat | p-value | Sig. |")
        lines.append("|-----|--------|---------|------|")
        for lag in range(1, 9):
            key = f"community_to_zeiger_lag{lag}"
            if key in granger:
                r = granger[key]
                lines.append(f"| {lag} | {r['f_stat']:.3f} | {r['p_value']:.4f} | {sig_stars(r['p_value'])} |")
    else:
        lines.append(f"*Error: {granger.get('error', 'unknown')}*\n")

    # ── H4: Cohort Analysis ──────────────────────────────────────────
    lines.append("\n\n## H4: Cohort-Stratified Adoption\n")
    cohort = load_json("cohort_results.json")
    if "pre_siege_did" in cohort:
        did = cohort["pre_siege_did"]
        if "error" not in did:
            lines.append(f"- Pre-Siege joiners with pre/post data: {did['n_users']}")
            lines.append(f"- Mean score change: {did['mean_diff']:.4f}")
            lines.append(f"- t-test: t={did['t_stat']:.3f}, {format_p(did['t_pvalue'])} {sig_stars(did['t_pvalue'])}")
            lines.append(f"- Wilcoxon: {format_p(did['wilcoxon_pvalue'])} {sig_stars(did['wilcoxon_pvalue'])}")
    if "entry_comparison" in cohort:
        ec = cohort["entry_comparison"]
        lines.append(f"\n**Entry-level comparison:**")
        lines.append(f"- Pre-joiners first posts mean: {ec['pre_mean']:.4f}")
        lines.append(f"- Post-joiners first posts mean: {ec['post_mean']:.4f}")
        lines.append(f"- Mann-Whitney: {format_p(ec['mann_whitney_p'])} {sig_stars(ec['mann_whitney_p'])}")

    # ── H5: DM Pipeline ──────────────────────────────────────────────
    lines.append("\n\n## H5: Private-to-Public Pipeline\n")
    pipe = load_json("pipeline_results.json")
    if "error" not in pipe and "n_users" in pipe:
        lines.append(f"- Users with siege in both DM and forum: {pipe['n_users']}")
        lines.append(f"- Mean DM lead: {pipe['mean_dm_lead_days']:.1f} days")
        lines.append(f"- DM first: {pipe['dm_first_pct']:.1f}%")
        lines.append(f"- Forum first: {pipe['forum_first_pct']:.1f}%")
        lines.append(f"- t-test: {format_p(pipe['t_pvalue'])} {sig_stars(pipe['t_pvalue'])}")
    else:
        lines.append(f"*{pipe.get('error', 'Insufficient data')}*\n")

    lines.append("\n**Small-N caution:** The DM corpus is much smaller than the forum corpus. "
                 "Results may lack statistical power.\n")

    # ── H6: Reputation Diffusion ──────────────────────────────────────
    lines.append("\n## H6: Reputation-Mediated Diffusion\n")
    rep = load_json("reputation_results.json")
    if "error" not in rep:
        lines.append("\n**Cox Proportional Hazards (standardized covariates):**\n")
        lines.append("| Variable | Coef | HR | SE | p-value | Sig. |")
        lines.append("|----------|------|----|----|---------|------|")
        for var in ["degree_centrality", "betweenness_centrality", "reputation"]:
            if var in rep:
                r = rep[var]
                lines.append(f"| {var} | {r['coef']:.4f} | {r['exp_coef']:.4f} | "
                             f"{r['se']:.4f} | {format_p(r['p'])} | {sig_stars(r['p'])} |")
    else:
        lines.append(f"*Error: {rep.get('error', 'unknown')}*\n")

    # ── Methodological notes ──────────────────────────────────────────
    lines.append("\n\n## Methodological Notes\n")
    lines.append("1. **Multiple comparisons:** All primary hypothesis tests should be "
                 "evaluated with Benjamini-Hochberg correction applied across the full "
                 "set of tests.")
    lines.append("2. **Endogeneity:** Network contagion results are subject to homophily "
                 "confounds. Permutation tests provide model-free robustness checks.")
    lines.append("3. **DM corpus size:** The DM corpus (≈21.7K messages) is much smaller "
                 "than the forum corpus (≈195K posts). DM-based analyses may be underpowered.")
    lines.append("4. **'Siege' overloading:** The word 'siege' appears in non-Mason contexts. "
                 "The embedding approach provides natural robustness against this issue.")

    report = "\n".join(lines)
    out_path = RESULTS_DIR / "summary_report.md"
    with open(out_path, "w") as f:
        f.write(report)
    print(f"  Summary report written to {out_path}")
    return report


def main():
    print("=" * 60)
    print("Generating Summary Report")
    print("=" * 60)
    generate_report()
    print("\n✓ Done.")


if __name__ == "__main__":
    main()
