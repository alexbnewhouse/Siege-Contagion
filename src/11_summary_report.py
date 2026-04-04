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

    # ── H7: Reputation Reinforcement ─────────────────────────────────
    lines.append("\n\n## H7: Reputation Reinforcement of Siege Rhetoric\n")
    rep_reinf = load_json("reputation_reinforcement_results.json")
    if "error" not in rep_reinf:
        lines.append(f"- Posts analysed: {rep_reinf.get('n_posts', 'N/A'):,}")
        lines.append(f"- Siege posts: {rep_reinf.get('n_siege_posts', 'N/A'):,}")
        lines.append(f"- Mean rep (Siege): {rep_reinf.get('mean_rep_siege', 0):.3f}")
        lines.append(f"- Mean rep (non-Siege): {rep_reinf.get('mean_rep_non_siege', 0):.3f}")
        lines.append(f"- Mann-Whitney U: {format_p(rep_reinf.get('mann_whitney_p', 1))} "
                     f"{sig_stars(rep_reinf.get('mann_whitney_p', 1))}")
        if "month_fe_model" in rep_reinf:
            m = rep_reinf["month_fe_model"]
            lines.append(f"\n**Negative binomial (month FE):**")
            lines.append(f"- Siege coefficient: {m['siege_coef']:.4f} "
                         f"({format_p(m['siege_p'])}) {sig_stars(m['siege_p'])}")
            lines.append(f"- Incidence rate ratio: {m['siege_irr']:.4f}")
        elif "simple_model" in rep_reinf:
            m = rep_reinf["simple_model"]
            lines.append(f"\n**Negative binomial (no FE):**")
            lines.append(f"- Siege coefficient: {m['siege_coef']:.4f} "
                         f"({format_p(m['siege_p'])}) {sig_stars(m['siege_p'])}")
            lines.append(f"- Incidence rate ratio: {m['siege_irr']:.4f}")
    else:
        lines.append(f"*{rep_reinf.get('error', 'No data')}*\n")

    lines.append("\n**Caution:** Reputation data reliability has not been independently "
                 "validated. These results should be interpreted with care.\n")

    # ── H8: Within-Thread Escalation ──────────────────────────────────
    lines.append("\n## H8: Within-Thread Escalation\n")
    thread_esc = load_json("thread_escalation_results.json")
    if "error" not in thread_esc:
        lines.append(f"- Siege threads (≥3 posts): {thread_esc.get('n_siege_threads', 'N/A'):,}")
        lines.append(f"- Posts in Siege threads: {thread_esc.get('n_posts_in_siege_threads', 'N/A'):,}")
        if "regression" in thread_esc:
            reg = thread_esc["regression"]
            lines.append(f"\n**Position regression (cluster-robust SE):**")
            lines.append(f"- Position coefficient: {reg['position_coef']:.4f} "
                         f"({format_p(reg['position_p'])}) {sig_stars(reg['position_p'])}")
            lines.append(f"- R²: {reg['r_squared']:.4f}")
        if "first_vs_last" in thread_esc:
            fvl = thread_esc["first_vs_last"]
            lines.append(f"\n**First vs. last post in thread:**")
            lines.append(f"- Mean first: {fvl['mean_first']:.4f}")
            lines.append(f"- Mean last: {fvl['mean_last']:.4f}")
            lines.append(f"- Mean diff: {fvl['mean_diff']:.4f}")
            lines.append(f"- t-test: t={fvl['t_stat']:.3f}, {format_p(fvl['t_p'])} "
                         f"{sig_stars(fvl['t_p'])}")
            lines.append(f"- Wilcoxon: {format_p(fvl['wilcoxon_p'])} "
                         f"{sig_stars(fvl['wilcoxon_p'])}")
    else:
        lines.append(f"*{thread_esc.get('error', 'No data')}*\n")

    # ── H9: Thread Exposure → Adoption ────────────────────────────────
    lines.append("\n\n## H9: Thread Exposure → Subsequent Adoption\n")
    thread_exp = load_json("thread_exposure_results.json")
    if "error" not in thread_exp:
        lines.append(f"- Panel observations: {thread_exp.get('n_user_months', 'N/A'):,} user-months")
        lines.append(f"- Unique users: {thread_exp.get('n_users', 'N/A'):,}")
        lines.append(f"- Thread exposure coefficient: {thread_exp.get('thread_exposure_coef', 0):.4f} "
                     f"({format_p(thread_exp.get('thread_exposure_p', 1))}) "
                     f"{sig_stars(thread_exp.get('thread_exposure_p', 1))}")
        lines.append(f"- Lagged own score coefficient: {thread_exp.get('lagged_own_score_coef', 0):.4f} "
                     f"({format_p(thread_exp.get('lagged_own_score_p', 1))}) "
                     f"{sig_stars(thread_exp.get('lagged_own_score_p', 1))}")
        lines.append(f"- R²: {thread_exp.get('r_squared', 0):.4f}")
        if "tercile_means" in thread_exp:
            lines.append(f"\n**Mean next-month siege score by thread exposure tercile:**")
            for k, v in thread_exp["tercile_means"].items():
                lines.append(f"- {k}: {v:.4f}")
    else:
        lines.append(f"*{thread_exp.get('error', 'No data')}*\n")

    # ── H10: Semantic Convergence ─────────────────────────────────────
    lines.append("\n\n## H10: Semantic Convergence\n")
    conv = load_json("semantic_convergence_results.json")
    if "error" not in conv:
        if "cv_keyword" in conv:
            ck = conv["cv_keyword"]
            lines.append(f"**CV of keyword score (ITS):**")
            lines.append(f"- Level change (β₂): {ck['level_change']:.4f} "
                         f"({format_p(ck['level_change_p'])}) {sig_stars(ck['level_change_p'])}")
            lines.append(f"- Slope change (β₃): {ck['slope_change']:.6f} "
                         f"({format_p(ck['slope_change_p'])}) {sig_stars(ck['slope_change_p'])}")
            lines.append(f"- R²: {ck['r_squared']:.4f}")
        if "pre_post_comparison" in conv:
            pp = conv["pre_post_comparison"]
            lines.append(f"\n**Pre/post comparison:**")
            lines.append(f"- Pre-Siege mean CV: {pp['pre_mean_cv']:.4f}")
            lines.append(f"- Post-Siege mean CV: {pp['post_mean_cv']:.4f}")
            lines.append(f"- Direction: {pp['direction']}")
            lines.append(f"- t-test: {format_p(pp['t_p'])} {sig_stars(pp['t_p'])}")
    else:
        lines.append(f"*{conv.get('error', 'No data')}*\n")

    # ── H11: Subforum Diffusion ───────────────────────────────────────
    lines.append("\n\n## H11: Subforum Diffusion Geography\n")
    subforum = load_json("subforum_diffusion_results.json")
    if "error" not in subforum:
        lines.append(f"- Subforums analysed: {subforum.get('n_subforums', 'N/A')}")
        lines.append(f"- Herfindahl index (overall): {subforum.get('herfindahl_overall', 0):.4f}")
        lines.append(f"- Pre-Siege HHI: {subforum.get('herfindahl_pre', 0):.4f}")
        lines.append(f"- Post-Siege HHI: {subforum.get('herfindahl_post', 0):.4f}")
        lines.append(f"- Direction: {subforum.get('diffusion_direction', 'N/A')}")
        if "top_subforums" in subforum:
            lines.append(f"\n**Top subforums by Siege prevalence:**\n")
            lines.append("| Subforum | Prevalence | Posts |")
            lines.append("|----------|-----------|-------|")
            for sf in subforum["top_subforums"][:10]:
                lines.append(f"| {sf['name']} | {sf['prevalence']:.3f} | {sf['posts']:,} |")
        if "biggest_increases" in subforum:
            lines.append(f"\n**Biggest pre→post increases:**\n")
            lines.append("| Subforum | Pre | Post | Δ |")
            lines.append("|----------|-----|------|---|")
            for sf in subforum["biggest_increases"][:10]:
                lines.append(f"| {sf['name']} | {sf['pre']:.3f} | {sf['post']:.3f} | "
                             f"{sf['change']:+.3f} |")
    else:
        lines.append(f"*{subforum.get('error', 'No data')}*\n")

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
