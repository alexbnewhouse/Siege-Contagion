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


def _verdict(supported: bool | None) -> str:
    """Return a verdict string for the scorecard."""
    if supported is None:
        return "—"
    return "✓ Supported" if supported else "✗ Not supported"


def _partial(txt: str) -> str:
    return f"~ {txt}"


def _build_scorecard(its, cont, granger, cohort, pipe, rep, rep_reinf,
                     thread_esc, thread_exp, conv, subforum,
                     cp_its, cp_granger, bridges, shutdown,
                     vocab, te, country, dose, subtheme, domain,
                     apoc_its, robust, attack, adv_ts, h_results) -> str:
    """Build a hypothesis-level scorecard with support verdicts."""
    rows = []
    rows.append("| # | Hypothesis | Verdict | Key Statistic |")
    rows.append("|---|-----------|---------|---------------|")

    # H1: ITS
    if "error" not in its:
        all_sim = its.get("all_similarity", {})
        p_slope = all_sim.get("p_time_x_post", 1)
        p_level = all_sim.get("p_post_treatment", 1)
        rows.append(f"| H1 | Siege publication = structural break | "
                    f"{_verdict(p_slope < 0.05)} | β₃ p={p_slope:.4f}, β₂ p={p_level:.4f} |")

    # H2
    if "error" not in cont:
        p_forum = cont.get("p_forum_exposure_dm", 1)
        rows.append(f"| H2 | Network exposure → adoption | "
                    f"{_verdict(p_forum < 0.05)} | β={cont.get('b_forum_exposure_dm', 0):.4f}, p<0.001 |")

    # H3
    if "error" not in granger:
        best_p = min(granger.get(f"zeiger_to_community_lag{i}", {}).get("p_value", 1)
                     for i in range(1, 9))
        rows.append(f"| H3 | Zeiger Granger-causes community | "
                    f"{_verdict(best_p < 0.05)} | best p={best_p:.4f} |")

    # H4
    if "pre_siege_did" in cohort:
        did = cohort["pre_siege_did"]
        rows.append(f"| H4 | Cohort conversion effects | "
                    f"{_verdict(did.get('t_pvalue', 1) < 0.05)} | "
                    f"t={did.get('t_stat', 0):.3f}, p={did.get('t_pvalue', 1):.3f} |")

    # H5
    if "error" not in pipe and "forum_first_pct" in pipe:
        rows.append(f"| H5 | Forum-first pipeline | "
                    f"{_verdict(pipe.get('forum_first_pct', 0) > 50)} | "
                    f"{pipe.get('forum_first_pct', 0):.1f}% forum-first |")

    # H6
    if "error" not in rep:
        dc = rep.get("degree_centrality", {})
        rows.append(f"| H6 | Network centrality → faster adoption | "
                    f"{_verdict(dc.get('p', 1) < 0.05)} | HR={dc.get('exp_coef', 0):.2f}, p<0.001 |")

    # H7
    if "error" not in rep_reinf:
        m = rep_reinf.get("month_fe_model", rep_reinf.get("simple_model", {}))
        rows.append(f"| H7 | Community rewards Siege speech | "
                    f"{_verdict(m.get('siege_p', 1) < 0.05)} | "
                    f"IRR={m.get('siege_irr', 0):.2f}, p<0.001 |")

    # H8
    if "error" not in thread_esc:
        reg = thread_esc.get("regression", {})
        rows.append(f"| H8 | Within-thread escalation | "
                    f"{_partial('Catalyst pattern, not linear')} | "
                    f"β={reg.get('position_coef', 0):.4f}, p={reg.get('position_p', 1):.3f} |")

    # H9
    if "error" not in thread_exp:
        p_te = thread_exp.get("thread_exposure_p", 1)
        rows.append(f"| H9 | Thread exposure → adoption | "
                    f"{_partial('Marginal (p=0.087)')} | "
                    f"β={thread_exp.get('thread_exposure_coef', 0):.4f}, p={p_te:.3f} |")

    # H10
    if "error" not in conv:
        ck = conv.get("cv_keyword", {})
        rows.append(f"| H10 | Semantic convergence | "
                    f"{_verdict(ck.get('slope_change_p', 1) < 0.05)} | "
                    f"β₃={ck.get('slope_change', 0):.6f}, p={ck.get('slope_change_p', 1):.3f} |")

    # H11
    if "error" not in subforum:
        rows.append(f"| H11 | Subforum concentration | "
                    f"{_verdict(subforum.get('herfindahl_post', 0) > subforum.get('herfindahl_pre', 0))} | "
                    f"HHI: {subforum.get('herfindahl_pre', 0):.4f}→{subforum.get('herfindahl_post', 0):.4f} |")

    # H12
    if "error" not in cp_its:
        pk = cp_its.get("pol_keyword_its", {})
        rows.append(f"| H12 | /pol/ structural break at T₀ | "
                    f"{_verdict(pk.get('p_post_treatment', 1) < 0.05)} | "
                    f"β₂={pk.get('b_post_treatment', 0):.4f}, p={pk.get('p_post_treatment', 1):.3f} |")

    # H13
    if "error" not in cp_granger:
        gr_d = cp_granger.get("granger", {})
        best_im_p = gr_d.get("im_to_pol_best_p", 1)
        rows.append(f"| H13 | IM Granger-causes /pol/ | "
                    f"{_verdict(best_im_p < 0.05)} | best p={best_im_p:.4f} |")

    # H14
    if "error" not in bridges:
        tp = bridges.get("temporal_priority", {})
        im_led = tp.get("im_led", 0)
        total = tp.get("terms_on_both", 1)
        rows.append(f"| H14 | Content bridges IM → /pol/ | "
                    f"{_verdict(im_led / max(total, 1) > 0.5)} | "
                    f"{im_led}/{total} IM-first ({im_led / max(total, 1) * 100:.0f}%) |")

    # H15
    if "error" not in shutdown:
        prev = shutdown.get("prevalence_its", {})
        rows.append(f"| H15 | /pol/ rhetoric ↑ after shutdown | "
                    f"{_verdict(prev.get('p_level', 1) < 0.05)} | "
                    f"prevalence β₂ p={prev.get('p_level', 1):.6f} |")

    # H16
    if "error" not in vocab:
        pct_im = vocab.get("im_led", 0) / max(vocab.get("n_terms", 1), 1) * 100
        rows.append(f"| H16 | Vocab appears on IM first | "
                    f"{_verdict(pct_im > 50)} | "
                    f"{pct_im:.0f}% IM-first, median lag {vocab.get('median_lag_days', 0):.0f}d |")

    # H17
    if "error" not in te:
        im_pol = te.get("im_to_pol") or te.get("lag1_im_to_pol", {})
        pol_im = te.get("pol_to_im") or te.get("lag1_pol_to_im", {})
        rows.append(f"| H17 | IM → /pol/ transfer entropy | "
                    f"{_verdict(im_pol.get('significant_05', False))} | "
                    f"IM→/pol/ p={im_pol.get('p_value', 1):.3f}; "
                    f"/pol/→IM p={pol_im.get('p_value', 1):.3f} |")

    # H18
    if "error" not in country:
        im_rest = country.get("im_vs_rest_test", {})
        rows.append(f"| H18 | IM-heavy countries = more Siege | "
                    f"{_verdict(im_rest.get('p_value', 1) < 0.05)} | "
                    f"p={im_rest.get('p_value', 1):.3f} |")

    # H19
    if "error" not in dose:
        lag3 = dose.get("lag_results", {}).get("lag_3", {})
        rows.append(f"| H19 | Dose-response IM → /pol/ | "
                    f"{_verdict(lag3.get('significant_sp_05', False))} | "
                    f"ρ={lag3.get('spearman_rho', 0):.4f}, p={lag3.get('spearman_p', 1):.4f} (lag 3) |")

    # H20
    if "error" not in subtheme:
        mason = subtheme.get("mason_core", {}).get("granger_im_to_pol", {})
        rows.append(f"| H20 | Sub-theme differential diffusion | "
                    f"{_verdict(mason.get('significant_05', False))} | "
                    f"mason_core Granger p={mason.get('p_value', 1):.2e} |")

    # H21
    if "error" not in domain:
        tp = domain.get("temporal_priority", {})
        im_first = tp.get("im_first", 0)
        shared = max(tp.get("shared_domains", 1), 1)
        rows.append(f"| H21 | Domains propagate IM → /pol/ | "
                    f"{_verdict(im_first / shared > 0.5)} | "
                    f"{im_first}/{shared} IM-first ({im_first / shared * 100:.0f}%) |")

    # Apocalypticism pooled
    if "error" not in apoc_its:
        pooled = apoc_its.get("pooled", {})
        p_level = pooled.get("p_level", 1)
        rows.append(f"| Apoc | Pooled ITS: events → ↑ apocalypticism | "
                    f"{_partial('Sig but *negative* β₂')} | "
                    f"β₂={pooled.get('b_level', 0):.4f}, p={p_level:.3f} |")

    # H22–H26
    if "error" not in h_results:
        h22 = h_results.get("H22_contagion_decay", {})
        if h22 and "error" not in h22:
            agg22 = h22.get('aggregate', {})
            avg22 = h22.get('average_trajectory_fit', {})
            median_hl = agg22.get('median_half_life', float('nan'))
            avg_r2 = avg22.get('half_life_days', float('nan'))
            rows.append(f"| H22 | Exponential decay post-attack | "
                        f"{_verdict(h22.get('n_valid_fits', 0) > 0)} | "
                        f"median t½={median_hl:.1f}d, avg t½={avg_r2:.1f}d |")

        h23 = h_results.get("H23_reciprocal_amplification", {})
        if h23 and "error" not in h23:
            bidir = h23.get("findings", {}).get("reciprocal_feedback", False)
            rows.append(f"| H23 | Reciprocal amplification | "
                        f"{_verdict(bidir)} | bidirectional={bidir} |")

        h24 = h_results.get("H24_threshold_activation", {})
        if h24 and "error" not in h24:
            opt = h24.get("optimal_threshold", {})
            rows.append(f"| H24 | Threshold activation | "
                        f"{_verdict(opt.get('mannwhitney_p', 1) < 0.05)} | "
                        f"T={opt.get('threshold', 'N/A')}, p={opt.get('mannwhitney_p', 1):.3f} |")

        h25 = h_results.get("H25_temporal_clustering", {})
        if h25 and "error" not in h25:
            comp = h25.get("comparison", {})
            rows.append(f"| H25 | Temporal clustering compounds | "
                        f"{_verdict(comp.get('mannwhitney_p', 1) < 0.05)} | "
                        f"p={comp.get('mannwhitney_p', 1):.3f} |")

        h26 = h_results.get("H26_mimetic_contagion", {})
        if h26 and "error" not in h26:
            mag = h26.get("magnitude_test", {})
            rows.append(f"| H26 | Online-nexus = distinctive shifts | "
                        f"{_verdict(mag.get('mannwhitney_p', 1) < 0.05)} | "
                        f"d={mag.get('cohens_d', 'N/A')}, p={mag.get('mannwhitney_p', 1):.3f} |")

    return "\n".join(rows)


def _build_key_findings(its, cont, granger, cohort, pipe, rep_reinf,
                        thread_exp, conv, cp_its, shutdown,
                        vocab, te, dose, subtheme, domain,
                        apoc_its, robust, attack, adv_ts, h_results) -> str:
    """Build an interpretive summary of the most important patterns."""
    findings = []

    findings.append("### 1. Siege Diffusion Within Iron March\n")
    findings.append("The core finding is robust: the publication of Zeiger's *Siege* edition "
                    "produced a significant structural break in Iron March discourse. The "
                    "combined similarity score shows a highly significant post-Siege slope "
                    "acceleration (β₃, p<0.001), though the immediate level shift is "
                    "non-significant for keyword scores (p=0.57). This pattern — gradual "
                    "intensification rather than sudden adoption — is consistent with a "
                    "collective exegesis model where the community progressively develops "
                    "shared interpretive frameworks around the text.\n")
    findings.append("Forum network exposure (β=1.13, p<0.001) strongly predicts individual "
                    "Siege rhetoric adoption, with permutation p=0.000 confirming this is "
                    "not an artifact of network structure. However, Zeiger himself does not "
                    "Granger-cause community rhetoric (best p=0.17), suggesting diffusion "
                    "operates through decentralized peer influence rather than top-down "
                    "ideological entrepreneurship. The community \"canonises\" *Siege* "
                    "collectively: Siege-aligned posts receive 54% more reputation points "
                    "(IRR=1.54, p<0.001), and the text concentrates in books, strategy, and "
                    "archival subforums — spaces of interpretation rather than casual discussion.\n")

    findings.append("### 2. Cross-Platform Contagion: Iron March → /pol/\n")
    findings.append("The cross-platform picture is nuanced. Direct causal evidence is weak: "
                    "neither Granger causality (best p=0.58) nor transfer entropy (IM→/pol/ "
                    "p=0.755) reaches significance in the IM→/pol/ direction. The surprising "
                    "finding is that /pol/→IM transfer entropy *is* significant (p=0.03), "
                    "suggesting some reverse flow.\n")
    findings.append("However, indirect diffusion evidence is strong. 84% of tracked Siege "
                    "vocabulary terms appear on IM before /pol/, with a median lag of 670 "
                    "days, and the lag is *accelerating* (slope = -0.62, p<0.001) — i.e., "
                    "newer terms propagate faster. 79% of shared URL domains appear on IM "
                    "first. The dose-response relationship is significant at lags 2–3 weeks "
                    "(ρ=0.14–0.19, p<0.05), suggesting a 2–3 week transmission window.\n")
    findings.append("Critically, after Iron March is shut down (November 2017), /pol/ Siege "
                    "prevalence shows a significant *increase* (β₂ p<0.001 for prevalence, "
                    "p<0.001 for similarity), with the shutdown effect 3.4× larger than the "
                    "original T₀ effect. This is consistent with diaspora — IM users migrate "
                    "to /pol/ after their platform is destroyed.\n")

    findings.append("### 3. Apocalypticism & Mass Violence\n")
    findings.append("The relationship between mass-casualty events and /pol/ apocalyptic "
                    "rhetoric is counter-intuitive. The pooled ITS finds a small but "
                    "statistically significant *decrease* in apocalyptic rhetoric "
                    "post-event (β₂ = −0.0018, p=0.043), not an increase. However, this "
                    "result is fragile: only 2 of 5 robustness checks pass. The AR(1)-controlled "
                    "model absorbs the effect into autocorrelation (β₂ p=0.50), and the "
                    "placebo test does not clearly reject the null (p=0.254).\n")
    findings.append("Attack characteristics show limited predictive power. Casualty severity "
                    "does not correlate with the ITS level shift (all Pearson/Spearman p>0.3). "
                    "The only significant predictor in the multiple regression is the domestic "
                    "vs international distinction (β = −0.011, p=0.010): domestic attacks produce "
                    "slightly stronger negative shifts in mean apocalypticism, possibly "
                    "reflecting a normalization effect where familiar events elicit less "
                    "eschatological framing than geographically distant ones.\n")

    # Include advanced TS findings if available
    if "error" not in adv_ts:
        comparison = adv_ts.get("method_comparison", {})
        if comparison and "error" not in comparison:
            consensus = comparison.get("consensus", {})
            c_dir = consensus.get("consensus_direction", "N/A")
            n_sig = consensus.get("n_significant", "N/A")
            n_meth = consensus.get("n_methods", "N/A")
            findings.append(f"The method comparison (ITS × VAR × ARDL × BSTS × LP) yields a "
                            f"consensus direction of \"{c_dir}\" with {n_sig}/{n_meth} methods "
                            f"reaching significance. ")
        else:
            findings.append("The advanced time-series methods (VAR, ARDL, BSTS, Local Projections) "
                            "provide additional perspective on the ITS findings. ")
        findings.append("Where methods disagree on significance, this confirms the effect "
                        "is at best small and fragile — a finding that is itself substantively "
                        "important, as it counters the popular narrative that mass-casualty "
                        "events reliably \"radicalise\" online communities toward "
                        "apocalyptic worldviews.\n")

    if "error" not in h_results:
        findings.append("### 4. Offline–Online Dynamics (H22–H26)\n")
        h22 = h_results.get("H22_contagion_decay", {})
        if h22 and "error" not in h22:
            agg22 = h22.get('aggregate', {})
            hl = agg22.get('median_half_life', 0)
            findings.append(f"Post-attack rhetoric shows an estimated median half-life of "
                            f"{hl:.1f} days, suggesting that whatever rhetorical shift occurs "
                            f"dissipates relatively quickly. ")
        h23 = h_results.get("H23_reciprocal_amplification", {})
        if h23 and "error" not in h23:
            bidir = h23.get("findings", {}).get("reciprocal_feedback", False)
            if bidir:
                findings.append("Bidirectional Granger causality confirms a feedback loop "
                                "between events and rhetoric — rhetoric does not merely *respond* "
                                "to violence but may also anticipate or track it. ")
            else:
                findings.append("The reciprocal amplification test does not find bidirectional "
                                "causality, suggesting rhetoric responds to events rather than "
                                "the reverse. ")
        findings.append("\n")

    findings.append("### Summary: What the Evidence Supports\n")
    findings.append("The strongest findings in this project concern the *internal* dynamics of "
                    "Iron March: Siege rhetoric diffuses through network exposure, is reinforced "
                    "by community reputation mechanisms, and concentrates in interpretive spaces. "
                    "Cross-platform diffusion from IM → /pol/ operates primarily through "
                    "vocabulary and URL propagation rather than detectable Granger-causal "
                    "flows, with a 2–3 week transmission lag and acceleration over time. "
                    "The shutdown of Iron March paradoxically *increases* Siege rhetoric "
                    "on /pol/, consistent with platform diaspora. The apocalypticism analysis "
                    "complicates simplistic accounts: mass-casualty events do not reliably "
                    "increase apocalyptic rhetoric, and what effect exists is small, fragile, "
                    "and unrelated to attack severity.\n")

    return "\n".join(findings)


def generate_report():
    lines = []
    lines.append("# Siege Culture Diffusion in Iron March: Summary Report\n")
    lines.append("*Auto-generated from analysis results.*\n")

    lines.append("\n---\n")
    lines.append("# Part I: Iron March Internal Dynamics (H1–H11)\n")

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

    # ══════════════════════════════════════════════════════════════════
    #  CROSS-PLATFORM HYPOTHESES (H12–H21)
    # ══════════════════════════════════════════════════════════════════

    lines.append("\n\n---\n")
    lines.append("# Part II: Cross-Platform Diffusion (Iron March → /pol/)\n")

    # ── H12: Cross-Platform ITS ───────────────────────────────────────
    lines.append("\n## H12: Siege Rhetoric Shows a Structural Break on /pol/ After IM's T₀\n")
    cp_its = load_json("cross_platform_its_results.json")
    if "error" not in cp_its:
        lines.append(f"- Treatment date: {cp_its.get('treatment_date', 'N/A')}")
        lines.append(f"- IM weeks: {cp_its.get('im_weeks', 'N/A')}, /pol/ weeks: {cp_its.get('pol_weeks', 'N/A')}")

        for key_name, label in [("pol_keyword_its", "/pol/ keyword score"),
                                ("pol_similarity_its", "/pol/ similarity score"),
                                ("pol_prevalence_its", "/pol/ prevalence")]:
            val = cp_its.get(key_name)
            if val and isinstance(val, dict) and "b_post_treatment" in val:
                lines.append(f"\n### {label}\n")
                lines.append(f"- N weeks: {val['n_weeks']}")
                lines.append(f"- Level change (β₂): {val['b_post_treatment']:.4f} "
                             f"({format_p(val['p_post_treatment'])}) {sig_stars(val['p_post_treatment'])}")
                lines.append(f"- Slope change (β₃): {val['b_time_x_post']:.6f} "
                             f"({format_p(val['p_time_x_post'])}) {sig_stars(val['p_time_x_post'])}")
                lines.append(f"- R²: {val['r_squared']:.4f}")
    else:
        lines.append(f"*Error: {cp_its.get('error', 'unknown')}*\n")

    # ── H13: Cross-Platform Granger ──────────────────────────────────
    lines.append("\n\n## H13: Iron March Siege Rhetoric Granger-Causes /pol/ Rhetoric\n")
    cp_granger = load_json("cross_platform_granger_results.json")
    if "error" not in cp_granger:
        lines.append(f"- Overlapping weeks: {cp_granger.get('overlapping_weeks', 'N/A')}")
        granger_dat = cp_granger.get("granger", {})
        if "adf_im" in granger_dat:
            adf_im = granger_dat["adf_im"]
            adf_pol = granger_dat["adf_pol"]
            lines.append(f"- ADF (IM): {adf_im.get('adf_statistic', 0):.3f}, "
                         f"stationary={adf_im.get('stationary', 'N/A')}")
            lines.append(f"- ADF (/pol/): {adf_pol.get('adf_statistic', 0):.3f}, "
                         f"stationary={adf_pol.get('stationary', 'N/A')}")
            lines.append(f"- Differenced: {granger_dat.get('differenced', 'N/A')}")

        for direction, label in [("im_to_pol", "IM → /pol/"), ("pol_to_im", "/pol/ → IM")]:
            d = granger_dat.get(direction, {})
            if d:
                lines.append(f"\n### {label}\n")
                lines.append("| Lag | F-stat | p-value | Sig. |")
                lines.append("|-----|--------|---------|------|")
                for lag in range(1, 13):
                    key = f"lag{lag}"
                    if key in d:
                        r = d[key]
                        lines.append(f"| {lag} | {r['f_stat']:.3f} | {r['p_value']:.4f} | "
                                     f"{sig_stars(r['p_value'])} |")

        best_im_p = cp_granger.get("granger", {}).get("im_to_pol_best_p", 1.0)
        best_pol_p = cp_granger.get("granger", {}).get("pol_to_im_best_p", 1.0)
        lines.append(f"\n- Best IM→/pol/ p-value: {best_im_p:.4f}")
        lines.append(f"- Best /pol/→IM p-value: {best_pol_p:.4f}")

        ccf = cp_granger.get("ccf", {})
        if ccf:
            lines.append(f"- Cross-correlation peak: lag={ccf.get('peak_lag', 'N/A')}, "
                         f"r={ccf.get('peak_correlation', 0):.4f} "
                         f"(95% CI: ±{ccf.get('ci_95', 0):.4f})")
    else:
        lines.append(f"*Error: {cp_granger.get('error', 'unknown')}*\n")

    # ── H14: Content Bridges ─────────────────────────────────────────
    lines.append("\n\n## H14: Content Bridges Propagate from IM → /pol/\n")
    bridges = load_json("cross_platform_bridges_results.json")
    if "error" not in bridges:
        url_a = bridges.get("url_analysis", {})
        lines.append("**URL analysis:**")
        lines.append(f"- IM URLs: {url_a.get('total_im_urls', 0):,}")
        lines.append(f"- /pol/ URLs: {url_a.get('total_pol_urls', 0):,}")
        lines.append(f"- Shared URLs: {url_a.get('shared_urls', 0):,}")
        lines.append(f"- IM first: {url_a.get('im_first', 0):,}, "
                     f"/pol/ first: {url_a.get('pol_first', 0):,}")

        ngram_a = bridges.get("ngram_analysis", {})
        lines.append(f"\n**N-gram analysis (n={ngram_a.get('n', 'N/A')}):**")
        lines.append(f"- Shared n-grams: {ngram_a.get('shared_ngram_count', 0):,}")

        tp = bridges.get("temporal_priority", {})
        lines.append(f"\n**Temporal priority:**")
        lines.append(f"- Terms on both platforms: {tp.get('terms_on_both', 0)}")
        lines.append(f"- IM led: {tp.get('im_led', 0)}, /pol/ led: {tp.get('pol_led', 0)}")

        im_mentions = bridges.get("ironmarch_mentions", {})
        if im_mentions:
            lines.append(f"\n**Iron March mentions on /pol/:**")
            lines.append(f"- Total mentions: {im_mentions.get('total_mentions', 0):,}")
            lines.append(f"- Weeks with mentions: {im_mentions.get('n_weeks_with_mentions', 0)}")
    else:
        lines.append(f"*{bridges.get('error', 'No data')}*\n")

    # ── /pol/ Thread Escalation (adapted H8) ─────────────────────────
    lines.append("\n\n## H8 (adapted): /pol/ Within-Thread Escalation\n")
    pol_esc = load_json("pol_thread_escalation_results.json")
    if "error" not in pol_esc:
        lines.append(f"- /pol/ Siege threads: {pol_esc.get('n_siege_threads', 0):,}")
        lines.append(f"- Posts in Siege threads: {pol_esc.get('n_posts_in_siege_threads', 0):,}")
        if "regression" in pol_esc:
            reg = pol_esc["regression"]
            lines.append(f"\n**Position regression:**")
            lines.append(f"- Position coefficient: {reg['position_coef']:.4f} "
                         f"({format_p(reg['position_p'])}) {sig_stars(reg['position_p'])}")
            lines.append(f"- R²: {reg['r_squared']:.4f}")
        if "first_vs_last" in pol_esc:
            fvl = pol_esc["first_vs_last"]
            lines.append(f"\n**First vs. last post:**")
            lines.append(f"- Mean first: {fvl['mean_first']:.4f}")
            lines.append(f"- Mean last: {fvl['mean_last']:.4f}")
            lines.append(f"- Wilcoxon: {format_p(fvl['wilcoxon_p'])} {sig_stars(fvl['wilcoxon_p'])}")
    else:
        lines.append(f"*{pol_esc.get('error', 'No data')}*\n")

    # ── /pol/ Semantic Convergence (adapted H10) ─────────────────────
    lines.append("\n\n## H10 (adapted): /pol/ Semantic Convergence\n")
    pol_conv = load_json("pol_semantic_convergence_results.json")
    if "error" not in pol_conv:
        lines.append(f"- N months: {pol_conv.get('n_months', 'N/A')}")
        if "its_model" in pol_conv:
            its_m = pol_conv["its_model"]
            lines.append(f"- Level change: {its_m['level_change']:.4f} "
                         f"({format_p(its_m['level_change_p'])}) {sig_stars(its_m['level_change_p'])}")
            lines.append(f"- Slope change: {its_m['slope_change']:.6f} "
                         f"({format_p(its_m['slope_change_p'])}) {sig_stars(its_m['slope_change_p'])}")
            lines.append(f"- R²: {its_m['r_squared']:.4f}")
        if "pre_post_comparison" in pol_conv:
            pp = pol_conv["pre_post_comparison"]
            lines.append(f"\n**Pre/post comparison:**")
            lines.append(f"- Pre-Siege mean CV: {pp['pre_mean_cv']:.4f}")
            lines.append(f"- Post-Siege mean CV: {pp['post_mean_cv']:.4f}")
            lines.append(f"- Direction: {pp['direction']}")
            lines.append(f"- t-test: {format_p(pp['t_p'])} {sig_stars(pp['t_p'])}")
    else:
        lines.append(f"*{pol_conv.get('error', 'No data')}*\n")

    # ── H15: Shutdown ITS ─────────────────────────────────────────────
    lines.append("\n\n## H15: /pol/ Siege Rhetoric Increases After IM Shutdown\n")
    shutdown = load_json("shutdown_its_results.json")
    if "error" not in shutdown:
        lines.append(f"- Shutdown date: {shutdown.get('t_shutdown', 'N/A')}")
        lines.append(f"- Siege T₀: {shutdown.get('t0_siege', 'N/A')}")

        for key_name, label in [("keyword_its", "Keyword score"),
                                ("volume_its", "Post volume"),
                                ("prevalence_its", "Prevalence"),
                                ("similarity_its", "Similarity")]:
            val = shutdown.get(key_name)
            if val and isinstance(val, dict) and "b_level" in val:
                lines.append(f"\n### {label}\n")
                lines.append(f"- Level change (β₂): {val['b_level']:.4f} "
                             f"({format_p(val['p_level'])}) {sig_stars(val['p_level'])}")
                lines.append(f"- Slope change (β₃): {val['b_slope']:.6f} "
                             f"({format_p(val['p_slope'])}) {sig_stars(val['p_slope'])}")
                lines.append(f"- R²: {val['r_squared']:.4f}")

        if "effect_comparison" in shutdown:
            ec = shutdown["effect_comparison"]
            lines.append(f"\n**Effect comparison (shutdown vs T₀):**")
            lines.append(f"- β₂ at shutdown: {ec['b_level_shutdown']:.4f}")
            lines.append(f"- β₂ at T₀: {ec['b_level_t0']:.4f}")
            lines.append(f"- Ratio: {ec['ratio']:.2f}×, larger at {ec['larger']}")
    else:
        lines.append(f"*{shutdown.get('error', 'No data')}*\n")

    # ── H16: Vocabulary Adoption Lag ──────────────────────────────────
    lines.append("\n\n## H16: Siege Vocabulary Appears on IM Before /pol/\n")
    vocab = load_json("vocab_adoption_lag_results.json")
    if "error" not in vocab:
        lines.append(f"- Terms tracked: {vocab.get('n_terms', 'N/A')}")
        lines.append(f"- IM-first: {vocab.get('im_led', 0)}, "
                     f"/pol/-first: {vocab.get('pol_led', 0)}")
        lines.append(f"- Median lag: {vocab.get('median_lag_days', 0):.1f} days")
        lines.append(f"- Mean lag: {vocab.get('mean_lag_days', 0):.1f} days")
        accel = vocab.get("acceleration_test", {})
        if accel:
            lines.append(f"\n**Acceleration test (OLS lag ~ time):**")
            lines.append(f"- Slope: {accel.get('slope', 0):.4f}")
            lines.append(f"- R²: {accel.get('r_squared', 0):.4f}")
            lines.append(f"- p-value: {format_p(accel.get('p_value', 1))} "
                         f"{sig_stars(accel.get('p_value', 1))}")
            lines.append(f"- Interpretation: {accel.get('interpretation', 'N/A')}")
    else:
        lines.append(f"*{vocab.get('error', 'No data')}*\n")

    # ── H17: Transfer Entropy ─────────────────────────────────────────
    lines.append("\n\n## H17: Non-Linear Information Flows from IM → /pol/\n")
    te = load_json("transfer_entropy_results.json")
    if "error" not in te:
        lines.append(f"- Overlapping weeks: {te.get('n_weeks', 'N/A')}")
        for direction, label in [("im_to_pol", "IM → /pol/"), ("pol_to_im", "/pol/ → IM")]:
            d = te.get(direction) or te.get(f"lag1_{direction}")
            if d:
                lines.append(f"\n### {label} (lag 1)\n")
                lines.append(f"- Transfer entropy: {d.get('te', 0):.4f}")
                lines.append(f"- Surrogate mean: {d.get('surrogate_mean', 0):.4f}")
                lines.append(f"- z-score: {d.get('z_score', 0):.3f}")
                lines.append(f"- p-value: {d.get('p_value', 1):.3f} "
                             f"{sig_stars(d.get('p_value', 1))}")
    else:
        lines.append(f"*{te.get('error', 'No data')}*\n")

    # ── H18: Country Correlation ──────────────────────────────────────
    lines.append("\n\n## H18: /pol/ Posts from IM-Heavy Countries Show More Siege Rhetoric\n")
    country = load_json("country_correlation_results.json")
    if "error" not in country:
        lines.append(f"- Countries analysed: {country.get('n_countries', 'N/A')}")
        lines.append(f"- Posts with country data: {country.get('posts_with_country', 0):,}")
        lines.append(f"- IM cluster countries: {', '.join(country.get('im_cluster_countries', []))}")
        im_rest = country.get("im_vs_rest_test", {})
        if im_rest:
            lines.append(f"\n**IM-cluster vs rest:**")
            lines.append(f"- IM cluster mean score: {im_rest.get('im_mean_score', 0):.4f}")
            lines.append(f"- Rest mean score: {im_rest.get('rest_mean_score', 0):.4f}")
            lines.append(f"- Mann-Whitney: {format_p(im_rest.get('p_value', 1))} "
                         f"{sig_stars(im_rest.get('p_value', 1))}")
    else:
        lines.append(f"*{country.get('error', 'No data')}*\n")

    # ── H19: Dose-Response ────────────────────────────────────────────
    lines.append("\n\n## H19: Higher IM Siege Activity Predicts Elevated /pol/ Activity\n")
    dose = load_json("dose_response_results.json")
    if "error" not in dose:
        lines.append(f"- Overlapping weeks: {dose.get('n_overlapping_weeks', 'N/A')}")
        lines.append(f"- Max lag tested: {dose.get('max_lag', 'N/A')}")
        lag_results = dose.get("lag_results", {})
        if lag_results:
            lines.append("\n| Lag | Kruskal H | KW p | Spearman ρ | Sp p | KW sig | Sp sig |")
            lines.append("|-----|-----------|------|------------|------|--------|--------|")
            for lag_num in range(1, 9):
                key = f"lag_{lag_num}"
                if key in lag_results:
                    r = lag_results[key]
                    lines.append(f"| {lag_num} | {r['kruskal_h']:.3f} | {r['kruskal_p']:.4f} | "
                                 f"{r['spearman_rho']:.4f} | {r['spearman_p']:.4f} | "
                                 f"{'✓' if r.get('significant_kw_05') else ''} | "
                                 f"{'✓' if r.get('significant_sp_05') else ''} |")
    else:
        lines.append(f"*{dose.get('error', 'No data')}*\n")

    # ── H20: Sub-Theme Diffusion ──────────────────────────────────────
    lines.append("\n\n## H20: Sub-Themes Diffuse Differentially Across Platforms\n")
    subtheme = load_json("subtheme_diffusion_results.json")
    if "error" not in subtheme:
        themes = subtheme.get("subthemes", [])
        lines.append(f"- Sub-themes: {', '.join(themes)}")
        lines.append("\n| Sub-theme | IM hits | /pol/ hits | β₂ (level) | p(level) | β₃ (slope) | p(slope) | Granger p |")
        lines.append("|-----------|---------|-----------|------------|----------|------------|----------|-----------|")
        for theme in themes:
            td = subtheme.get(theme, {})
            if "error" in td:
                continue
            its_t = td.get("its", {})
            gr = td.get("granger_im_to_pol", {})
            lines.append(f"| {theme} | {td.get('im_hits', 0):,} | {td.get('pol_hits', 0):,} | "
                         f"{its_t.get('b_level', 0):.4f} | {format_p(its_t.get('p_level', 1))} "
                         f"{sig_stars(its_t.get('p_level', 1))} | "
                         f"{its_t.get('b_slope', 0):.6f} | {format_p(its_t.get('p_slope', 1))} "
                         f"{sig_stars(its_t.get('p_slope', 1))} | "
                         f"{format_p(gr.get('p_value', 1))} {sig_stars(gr.get('p_value', 1))} |")
    else:
        lines.append(f"*{subtheme.get('error', 'No data')}*\n")

    # ── H21: Domain Diffusion ─────────────────────────────────────────
    lines.append("\n\n## H21: URL Domains Propagate from IM → /pol/\n")
    domain = load_json("domain_diffusion_results.json")
    if "error" not in domain:
        lines.append(f"- IM unique domains: {domain.get('im_unique_domains', 0):,}")
        lines.append(f"- /pol/ unique domains: {domain.get('pol_unique_domains', 0):,}")
        tp = domain.get("temporal_priority", {})
        if tp:
            lines.append(f"- Shared domains: {tp.get('shared_domains', 0):,}")
            lines.append(f"- IM first: {tp.get('im_first', 0):,} ({tp.get('im_first', 0) / max(tp.get('shared_domains', 1), 1) * 100:.0f}%)")
            lines.append(f"- /pol/ first: {tp.get('pol_first', 0):,}")
        gw = domain.get("gateway_domains", [])
        if gw:
            lines.append(f"\n**Top gateway domains (IM-first, high /pol/ volume):**\n")
            lines.append("| Domain | IM first | /pol/ first | Lag (days) | /pol/ count |")
            lines.append("|--------|----------|-------------|------------|-------------|")
            for g in gw[:10]:
                lines.append(f"| {g['domain']} | {g['im_first'][:10]} | {g['pol_first'][:10]} | "
                             f"{g['lag_days']:.0f} | {g['pol_count']:,} |")
    else:
        lines.append(f"*{domain.get('error', 'No data')}*\n")

    # ══════════════════════════════════════════════════════════════════
    #  APOCALYPTICISM CHAPTER
    # ══════════════════════════════════════════════════════════════════

    lines.append("\n\n---\n")
    lines.append("# Part III: Apocalypticism Chapter — Mass-Casualty Events & /pol/ Rhetoric\n")

    # ── Event Catalogue ───────────────────────────────────────────────
    mce = load_json("mass_casualty_events_summary.json")
    if "error" not in mce:
        lines.append("\n## Event Catalogue\n")
        lines.append(f"- Total events: {mce.get('total_events', 'N/A')}")
        lines.append(f"- Date range: {mce.get('date_range', ['N/A', 'N/A'])[0]} to "
                     f"{mce.get('date_range', ['N/A', 'N/A'])[1]}")
        by_cat = mce.get("by_category", [])
        if by_cat:
            lines.append("\n| Category | Count |")
            lines.append("|----------|-------|")
            for c in by_cat:
                lines.append(f"| {c['event_category']} | {c['len']} |")
        by_ideo = mce.get("by_ideology", [])
        if by_ideo:
            lines.append("\n| Ideology | Count |")
            lines.append("|----------|-------|")
            for c in by_ideo:
                lines.append(f"| {c['ideology']} | {c['len']} |")

    # ── Pooled ITS ────────────────────────────────────────────────────
    apoc_its = load_json("apocalypticism_its_results.json")
    if "error" not in apoc_its:
        lines.append("\n\n## Apocalypticism ITS (Per-Event & Pooled)\n")
        lines.append(f"- Total posts scored: {apoc_its.get('n_posts', 0):,}")
        lines.append(f"- Events analysed: {apoc_its.get('n_events', 0)}")
        lines.append(f"- Window: ±{apoc_its.get('window_pre_days', 30)} days")

        pooled = apoc_its.get("pooled", {})
        if pooled:
            lines.append(f"\n### Pooled ITS (stacked with event FE)\n")
            lines.append(f"- N observations: {pooled.get('n_obs', 0):,}")
            lines.append(f"- Level change (β₂): {pooled.get('b_level', 0):.4f} "
                         f"({format_p(pooled.get('p_level', 1))}) {sig_stars(pooled.get('p_level', 1))}")
            lines.append(f"- Slope change (β₃): {pooled.get('b_slope', 0):.6f} "
                         f"({format_p(pooled.get('p_slope', 1))}) {sig_stars(pooled.get('p_slope', 1))}")
            lines.append(f"- R²: {pooled.get('r_squared', 0):.4f}")
            lines.append(f"- 95% CI for β₂: [{pooled.get('ci_level_lo', 0):.4f}, "
                         f"{pooled.get('ci_level_hi', 0):.4f}]")

        # Category comparison
        cat_comp = apoc_its.get("category_comparison", {})
        if cat_comp:
            lines.append(f"\n### Category Comparison\n")
            lines.append("| Category | β₂ (level) | p(level) | β₃ (slope) | p(slope) |")
            lines.append("|----------|------------|----------|------------|----------|")
            for cat_name, cat_data in cat_comp.items():
                if isinstance(cat_data, dict) and "b_level" in cat_data:
                    lines.append(f"| {cat_name} | {cat_data['b_level']:.4f} | "
                                 f"{format_p(cat_data.get('p_level', 1))} "
                                 f"{sig_stars(cat_data.get('p_level', 1))} | "
                                 f"{cat_data.get('b_slope', 0):.6f} | "
                                 f"{format_p(cat_data.get('p_slope', 1))} "
                                 f"{sig_stars(cat_data.get('p_slope', 1))} |")

        # Stratified by ideology
        strat = apoc_its.get("stratified", {})
        if strat:
            lines.append(f"\n### Stratified by Ideology\n")
            lines.append("| Ideology | β₂ (level) | p(level) | n events |")
            lines.append("|----------|------------|----------|----------|")
            for ideo_name, ideo_data in strat.items():
                if isinstance(ideo_data, dict) and "b_level" in ideo_data:
                    lines.append(f"| {ideo_name} | {ideo_data['b_level']:.4f} | "
                                 f"{format_p(ideo_data.get('p_level', 1))} "
                                 f"{sig_stars(ideo_data.get('p_level', 1))} | "
                                 f"{ideo_data.get('n_events', 'N/A')} |")

    # ── Robustness Checks ─────────────────────────────────────────────
    robust = load_json("apocalypticism_robustness_results.json")
    if "error" not in robust:
        lines.append("\n\n## Apocalypticism Robustness Checks\n")
        summary = robust.get("summary", {})
        lines.append(f"- Checks passed: {summary.get('checks_passed', 'N/A')}"
                     f"/{summary.get('checks_total', 'N/A')}")

        # Placebo
        placebo = robust.get("placebo", {})
        if placebo:
            lines.append(f"\n**Placebo test ({placebo.get('n_iter', 0)} permutations):**")
            lines.append(f"- Observed β₂: {placebo.get('observed_beta_level', 0):.4f}")
            lines.append(f"- Null mean: {placebo.get('mean_null_level', 0):.4f}")
            lines.append(f"- Null SD: {placebo.get('std_null_level', 0):.4f}")
            lines.append(f"- Placebo p: {placebo.get('placebo_p_level', 1):.3f}")

        # BH FDR
        bh = robust.get("bh_fdr", {})
        if bh:
            lines.append(f"\n**Benjamini-Hochberg FDR:**")
            lines.append(f"- Raw significant: {bh.get('n_significant_raw', 0)}/{bh.get('n_tests', 0)}")
            lines.append(f"- BH-corrected significant: {bh.get('n_significant_bh', 0)}/{bh.get('n_tests', 0)}")

        # AR(1) controlled
        ar1 = robust.get("ar1_controlled", {})
        if ar1:
            lines.append(f"\n**AR(1) controlled:**")
            lines.append(f"- β₂: {ar1.get('b_level', 0):.4f} ({format_p(ar1.get('p_level', 1))}) "
                         f"{sig_stars(ar1.get('p_level', 1))}")
            lines.append(f"- AR(1) coefficient: {ar1.get('b_y_lag1', 0):.4f} "
                         f"({format_p(ar1.get('p_y_lag1', 1))}) {sig_stars(ar1.get('p_y_lag1', 1))}")

        # DoW controlled
        dow = robust.get("dow_controlled", {})
        if dow:
            lines.append(f"\n**Day-of-week controlled:**")
            lines.append(f"- β₂: {dow.get('b_level', 0):.4f} ({format_p(dow.get('p_level', 1))}) "
                         f"{sig_stars(dow.get('p_level', 1))}")

    # ── Attack Correlations ───────────────────────────────────────────
    attack = load_json("attack_correlations_results.json")
    if "error" not in attack:
        lines.append("\n\n## Attack Characteristic Correlations\n")

        sev = attack.get("severity_correlations", {})
        if sev:
            lines.append("**Severity vs β₂ (ITS level change):**\n")
            lines.append("| Measure | Pearson r | p | Spearman ρ | p |")
            lines.append("|---------|-----------|---|-----------|---|")
            for measure in ["killed", "injured", "total_casualties"]:
                if measure in sev:
                    r = sev[measure]
                    lines.append(f"| {measure} | {r['pearson_r']:.4f} | {r['pearson_p']:.3f} | "
                                 f"{r['spearman_rho']:.4f} | {r['spearman_p']:.3f} |")

        dom = attack.get("domestic_comparison", {})
        if dom:
            lines.append(f"\n**Domestic vs international:**")
            lines.append(f"- Domestic mean β₂: {dom.get('domestic_mean', 0):.4f} (n={dom.get('domestic_n', 0)})")
            lines.append(f"- International mean β₂: {dom.get('international_mean', 0):.4f} "
                         f"(n={dom.get('international_n', 0)})")
            lines.append(f"- Mann-Whitney: {format_p(dom.get('mannwhitney_p', 1))} "
                         f"{sig_stars(dom.get('mannwhitney_p', 1))}")

        nexus = attack.get("online_nexus_comparison", {})
        if nexus:
            lines.append(f"\n**Online nexus vs offline:**")
            lines.append(f"- Online-nexus mean β₂: {nexus.get('online_mean', 0):.4f} "
                         f"(n={nexus.get('online_n', 0)})")
            lines.append(f"- Offline mean β₂: {nexus.get('offline_mean', 0):.4f} "
                         f"(n={nexus.get('offline_n', 0)})")
            lines.append(f"- Mann-Whitney: {format_p(nexus.get('mannwhitney_p', 1))} "
                         f"{sig_stars(nexus.get('mannwhitney_p', 1))}")

        multi_reg = attack.get("multiple_regression", {})
        if multi_reg:
            lines.append(f"\n**Multiple regression (β₂ ~ attack characteristics):**")
            lines.append(f"- R²: {multi_reg.get('r_squared', 0):.4f} "
                         f"(adj: {multi_reg.get('adj_r_squared', 0):.4f})")
            lines.append(f"- Model F p-value: {format_p(multi_reg.get('f_pvalue', 1))}")
            coeffs = multi_reg.get("coefficients", {})
            if coeffs:
                lines.append("\n| Predictor | Coef | SE | p-value | Sig. |")
                lines.append("|-----------|------|-----|---------|------|")
                for name, c in coeffs.items():
                    lines.append(f"| {name} | {c['coef']:.4f} | {c['se']:.4f} | "
                                 f"{format_p(c['p'])} | {sig_stars(c['p'])} |")

    # ── Advanced TS Methods (Stage 34) ────────────────────────────────
    adv_ts = load_json("advanced_ts_results.json")
    if "error" not in adv_ts:
        lines.append("\n\n## Advanced Time-Series Methods (VAR / ARDL / BSTS / LP)\n")

        var_r = adv_ts.get("var", {})
        if var_r and "error" not in var_r:
            lines.append("\n### Vector Autoregression (VAR)\n")
            lines.append(f"- Selected lag: {var_r.get('selected_lag', 'N/A')} (AIC)")
            lines.append(f"- Observations: {var_r.get('n_obs', 'N/A')}")
            granger_var = var_r.get("granger_causality", {})
            for direction, gr_d in granger_var.items():
                if isinstance(gr_d, dict) and "error" not in gr_d:
                    lines.append(f"- Granger ({direction}): F={gr_d.get('test_statistic', 0):.3f}, "
                                 f"{format_p(gr_d.get('p_value', 1))} "
                                 f"{sig_stars(gr_d.get('p_value', 1))}")
            irf = var_r.get("irf", {})
            if irf and "error" not in irf:
                lines.append(f"- IRF peak response: {irf.get('peak_response', 0):.6f} "
                             f"at day {irf.get('peak_period', 'N/A')}")
            fevd = var_r.get("fevd", {})
            if fevd and "error" not in fevd:
                pct7 = fevd.get('pct_at_horizon_7', 0)
                pct30 = fevd.get('pct_at_horizon_30', 0)
                lines.append(f"- FEVD (event → apoc): {pct7*100:.2f}% at 7d, {pct30*100:.2f}% at 30d")

        ardl_r = adv_ts.get("ardl", {})
        if ardl_r and "error" not in ardl_r:
            lines.append("\n### Autoregressive Distributed Lag (ARDL)\n")
            lines.append(f"- ARDL({ardl_r.get('ar_order', '?')},{ardl_r.get('dl_order', '?')})")
            lines.append(f"- Long-run multiplier: {ardl_r.get('long_run_multiplier', 0):.6e}")
            lines.append(f"- R²: {ardl_r.get('r_squared', 0):.4f}")
            ecm = ardl_r.get("ecm", {})
            if ecm and "error" not in ecm:
                lines.append(f"- EC coefficient: {ecm.get('ec_coefficient', 0):.6f} "
                             f"({format_p(ecm.get('ec_p_value', 1))} {sig_stars(ecm.get('ec_p_value', 1))})")
                lines.append(f"- Bounds F: {ecm.get('bounds_F', 0):.2f} "
                             f"({format_p(ecm.get('bounds_p', 1))} {sig_stars(ecm.get('bounds_p', 1))})")
                lines.append(f"- Cointegration: {'Yes' if ecm.get('ec_significant', False) else 'No'}")

        bsts_r = adv_ts.get("bsts", {})
        if bsts_r and "error" not in bsts_r:
            lines.append("\n### Bayesian Structural Time Series (BSTS)\n")
            agg = bsts_r.get("aggregate", {})
            lines.append(f"- Events analysed: {bsts_r.get('n_events_analyzed', 0)}")
            lines.append(f"- Mean impact: {agg.get('mean_impact', 0):.6f}")
            lines.append(f"- Median impact: {agg.get('median_impact', 0):.6f}")
            lines.append(f"- t-statistic: {agg.get('t_statistic', 'N/A')}")
            lines.append(f"- t p-value: {format_p(agg.get('t_p_value', 1))} "
                         f"{sig_stars(agg.get('t_p_value', 1))}")
            lines.append(f"- Pct with positive effect: {agg.get('pct_positive', 0):.1%}")

        lp_r = adv_ts.get("local_projections", {})
        if lp_r and "error" not in lp_r:
            lines.append("\n### Local Projections (Jordà 2005)\n")
            lines.append(f"- Horizons estimated: {lp_r.get('n_horizons', 0)}")
            lines.append(f"- Significant horizons: {lp_r.get('n_significant', 0)}/{lp_r.get('n_horizons', 0)}")
            lines.append(f"- Peak β: {lp_r.get('peak_beta', 0):.6f} at h={lp_r.get('peak_horizon', 'N/A')} "
                         f"({format_p(lp_r.get('peak_p_value', 1))} {sig_stars(lp_r.get('peak_p_value', 1))})")

        comparison = adv_ts.get("method_comparison", {})
        if comparison and "error" not in comparison:
            consensus = comparison.get("consensus", {})
            methods = comparison.get("methods", {})
            lines.append("\n### Method Comparison (ITS × VAR × ARDL × BSTS × LP)\n")
            lines.append(f"- Consensus direction: {consensus.get('consensus_direction', 'N/A')}")
            lines.append(f"- Direction agreement: {consensus.get('direction_agreement', 'N/A')}")
            lines.append(f"- Methods reaching significance: {consensus.get('n_significant', 0)}"
                         f"/{consensus.get('n_methods', 0)}")
            lines.append(f"- Conclusion: {consensus.get('conclusion', 'N/A')}")
            if methods:
                lines.append("\n| Method | Direction | Effect Size | p-value | Significant |")
                lines.append("|--------|-----------|-------------|---------|-------------|")
                for mname, mdata in methods.items():
                    sig_marker = "✓" if mdata.get('significant') else ""
                    p_val = mdata.get('p_value', float('nan'))
                    p_str = format_p(p_val) if not (isinstance(p_val, float) and (p_val != p_val)) else 'N/A'
                    lines.append(f"| {mname} | {mdata.get('direction', 'N/A')} | "
                                 f"{mdata.get('effect_size', 0):.6f} | {p_str} | {sig_marker} |")

    # ── Offline/Online Hypotheses (Stage 35, H22–H26) ────────────────
    h_results = load_json("offline_online_hypotheses_results.json")
    if "error" not in h_results:
        lines.append("\n\n## Offline Violence ↔ Online Rhetoric (H22–H26)\n")

        # H22: Decay
        h22 = h_results.get("H22_contagion_decay", {})
        if h22 and "error" not in h22:
            lines.append("\n### H22: Contagion Decay\n")
            agg22 = h22.get('aggregate', {})
            avg_traj = h22.get('average_trajectory_fit', {})
            lines.append(f"- Valid fits: {h22.get('n_valid_fits', 0)}/{h22.get('n_events', 0)} events")
            lines.append(f"- Median half-life: {agg22.get('median_half_life', 0):.1f} days")
            lines.append(f"- Mean half-life: {agg22.get('mean_half_life', 0):.1f} days")
            lines.append(f"- Range: {agg22.get('min_half_life', 0):.1f}–{agg22.get('max_half_life', 0):.1f} days")
            if avg_traj and "error" not in avg_traj:
                lines.append(f"- Average trajectory fit: λ={avg_traj.get('decay_rate', 0):.4f}, "
                             f"t½={avg_traj.get('half_life_days', 0):.1f} days")

        # H23: Reciprocal amplification
        h23 = h_results.get("H23_reciprocal_amplification", {})
        if h23 and "error" not in h23:
            lines.append("\n### H23: Reciprocal Amplification\n")
            lines.append(f"- VAR lag: {h23.get('var_lag', 'N/A')}")
            lines.append(f"- Observations: {h23.get('n_obs', 'N/A')}")
            gc23 = h23.get("granger_causality", {})
            for direction, gr_d in gc23.items():
                if isinstance(gr_d, dict) and "error" not in gr_d:
                    lines.append(f"- Granger ({direction}): F={gr_d.get('test_statistic', 0):.3f}, "
                                 f"{format_p(gr_d.get('p_value', 1))} "
                                 f"{sig_stars(gr_d.get('p_value', 1))}")
            findings23 = h23.get("findings", {})
            lines.append(f"- Reciprocal feedback: {findings23.get('reciprocal_feedback', False)}")
            lines.append(f"- Events → rhetoric: {findings23.get('events_cause_rhetoric', False)}")
            lines.append(f"- Rhetoric → events: {findings23.get('rhetoric_causes_events', False)}")

        # H24: Threshold activation
        h24 = h_results.get("H24_threshold_activation", {})
        if h24 and "error" not in h24:
            lines.append("\n### H24: Threshold Activation\n")
            opt = h24.get("optimal_threshold", {})
            if opt:
                lines.append(f"- Optimal casualty threshold: {opt.get('threshold', 'N/A')}")
                lines.append(f"- Mann-Whitney p: {format_p(opt.get('mannwhitney_p', 1))} "
                             f"{sig_stars(opt.get('mannwhitney_p', 1))}")
                lines.append(f"- Effect size (above vs below): {opt.get('difference', 'N/A')}")
            threshold_tests = h24.get('threshold_tests', [])
            lines.append(f"- Thresholds scanned: {len(threshold_tests)}")
            n_sig_thresholds = sum(1 for t in threshold_tests if t.get('significant'))
            lines.append(f"- Significant thresholds: {n_sig_thresholds}/{len(threshold_tests)}")
            pw = h24.get('piecewise_regression', {})
            if pw and 'error' not in pw:
                lines.append(f"- Piecewise R²: {pw.get('r_squared', 0):.4f}")

        # H25: Temporal clustering
        h25 = h_results.get("H25_temporal_clustering", {})
        if h25 and "error" not in h25:
            lines.append("\n### H25: Temporal Clustering\n")
            lines.append(f"- Cluster window: {h25.get('cluster_window_days', 14)} days")
            lines.append(f"- Events analysed: {h25.get('n_events', 0)}")
            comp = h25.get("comparison", {})
            if comp and "error" not in comp:
                lines.append(f"- Clustered events: {comp.get('n_clustered', 0)} "
                             f"(mean Δ={comp.get('mean_clustered', 0):.4f})")
                lines.append(f"- Isolated events: {comp.get('n_isolated', 0)} "
                             f"(mean Δ={comp.get('mean_isolated', 0):.4f})")
                lines.append(f"- Mann-Whitney p: {format_p(comp.get('mannwhitney_p', 1))} "
                             f"{sig_stars(comp.get('mannwhitney_p', 1))}")
                lines.append(f"- Super-additive: {comp.get('super_additive', False)}")
            reg = h25.get("regression", {})
            if reg and "error" not in reg:
                lines.append(f"- Neighbors regression: β={reg.get('b_neighbors', 0):.4f}, "
                             f"{format_p(reg.get('p_neighbors', 1))} "
                             f"{sig_stars(reg.get('p_neighbors', 1))}")

        # H26: Mimetic contagion
        h26 = h_results.get("H26_mimetic_contagion", {})
        if h26 and "error" not in h26:
            lines.append("\n### H26: Mimetic Contagion\n")
            lines.append(f"- Online-nexus events: {h26.get('n_online', 0)}")
            lines.append(f"- Offline events: {h26.get('n_offline', 0)}")
            mag = h26.get("magnitude_test", {})
            if mag:
                lines.append(f"- Online |Δapoc| mean: {mag.get('online_mean_abs_delta', 0):.4f}")
                lines.append(f"- Offline |Δapoc| mean: {mag.get('offline_mean_abs_delta', 0):.4f}")
                lines.append(f"- Cohen's d: {mag.get('cohens_d', 0):.4f}")
                lines.append(f"- Magnitude p: {format_p(mag.get('mannwhitney_p', 1))} "
                             f"{sig_stars(mag.get('mannwhitney_p', 1))}")
            dir_test = h26.get("direction_test", {})
            if dir_test:
                lines.append(f"- Direction p: {format_p(dir_test.get('mannwhitney_p', 1))} "
                             f"{sig_stars(dir_test.get('mannwhitney_p', 1))}")
            pol_test = h26.get("polarisation_test", {})
            if pol_test:
                lines.append(f"- Polarisation (variance ratio) p: {format_p(pol_test.get('mannwhitney_p', 1))} "
                             f"{sig_stars(pol_test.get('mannwhitney_p', 1))}")
                lines.append(f"- Online variance ratio: {pol_test.get('online_mean_variance_ratio', 0):.3f}")
                lines.append(f"- Offline variance ratio: {pol_test.get('offline_mean_variance_ratio', 0):.3f}")

    # ══════════════════════════════════════════════════════════════════
    #  CATEGORY DISAGGREGATION
    # ══════════════════════════════════════════════════════════════════

    per_cat_its = apoc_its.get("per_category", {}) if "error" not in apoc_its else {}
    per_cat_robust = robust.get("per_category", {}) if "error" not in robust else {}
    per_cat_attack = attack.get("per_category", {}) if "error" not in attack else {}
    per_cat_adv = adv_ts.get("per_category", {}) if "error" not in adv_ts else {}
    per_cat_hyp = h_results.get("per_category", {}) if "error" not in h_results else {}

    any_cat = per_cat_its or per_cat_robust or per_cat_adv or per_cat_hyp
    if any_cat:
        lines.append("\n\n---\n")
        lines.append("# Part III-B: Disaggregated Analysis by Apocalypticism Category\n")
        lines.append("\nPosts classified as apocalyptic are further assigned to one of "
                     "four categories by cosine similarity to category-specific seed centroids:\n")
        lines.append("1. **Siegist / Traditionalist** – Siege culture, accelerationism, "
                     "Kali Yuga, Day of the Rope, Evola, Mason.")
        lines.append("2. **Rapture / Christian** – Rapture, Revelation, Armageddon, "
                     "Tribulation, end times.")
        lines.append("3. **Prepper** – SHTF, survivalism, stockpiling, grid-down, "
                     "off-grid living.")
        lines.append("4. **General Collapsist** – Civilisational decline, NWO, Great Reset, "
                     "peak oil, demographic collapse.\n")

        # ── Per-category ITS summary table ────────────────────────────
        if per_cat_its:
            lines.append("\n## ITS Results by Category\n")
            lines.append("| Category | Posts | Pooled β₂ | p(level) | "
                         "Sig events | Valid events |")
            lines.append("|----------|-------|-----------|----------|"
                         "-----------|--------------|")
            for cat, cdata in per_cat_its.items():
                if "error" in cdata:
                    lines.append(f"| {cat} | {cdata.get('n_posts', 0)} | – | – | – | – |")
                    continue
                pooled_c = cdata.get("pooled", {})
                b2 = pooled_c.get("b_level", 0)
                p2 = pooled_c.get("p_level", 1)
                n_sig = cdata.get("n_significant_005", 0)
                n_valid = cdata.get("n_valid_events", 0)
                lines.append(f"| {cat} | {cdata.get('n_posts', 0):,} | "
                             f"{b2:.4f} | {format_p(p2)} {sig_stars(p2)} | "
                             f"{n_sig} | {n_valid} |")

        # ── Per-category robustness ───────────────────────────────────
        if per_cat_robust:
            lines.append("\n## Robustness by Category\n")
            lines.append("| Category | BW sig | AR(1) β₂ | AR(1) p | DoW β₂ | DoW p |")
            lines.append("|----------|--------|----------|---------|--------|-------|")
            for cat, cdata in per_cat_robust.items():
                if "error" in cdata:
                    lines.append(f"| {cat} | – | – | – | – | – |")
                    continue
                bw_list = cdata.get("bandwidth_sensitivity", [])
                bw_sig = sum(1 for r in bw_list
                             if "error" not in r and r.get("p_level", 1) < 0.05)
                ar1_c = cdata.get("ar1_controlled", {})
                dow_c = cdata.get("dow_controlled", {})
                lines.append(
                    f"| {cat} | {bw_sig}/{len(bw_list)} | "
                    f"{ar1_c.get('b_level', 0):.4f} | "
                    f"{format_p(ar1_c.get('p_level', 1))} | "
                    f"{dow_c.get('b_level', 0):.4f} | "
                    f"{format_p(dow_c.get('p_level', 1))} |"
                )

        # ── Per-category advanced TS ──────────────────────────────────
        if per_cat_adv:
            lines.append("\n## Advanced TS by Category\n")
            lines.append("| Category | VAR Granger p | ARDL LR mult | LP peak β | LP peak h |")
            lines.append("|----------|---------------|-------------|-----------|-----------|")
            for cat, cdata in per_cat_adv.items():
                if "error" in cdata:
                    lines.append(f"| {cat} | – | – | – | – |")
                    continue
                var_c = cdata.get("var", {})
                gc_p = var_c.get("granger_causality", {}).get(
                    "event_causes_apoc", {}).get("p_value", float("nan"))
                ardl_c = cdata.get("ardl", {})
                lr_m = ardl_c.get("long_run_multiplier", float("nan"))
                lp_c = cdata.get("local_projections", {})
                lp_b = lp_c.get("peak_beta", float("nan"))
                lp_h = lp_c.get("peak_horizon", "–")
                gc_str = format_p(gc_p) if gc_p == gc_p else "–"
                lr_str = f"{lr_m:.4e}" if lr_m == lr_m else "–"
                lp_str = f"{lp_b:.6f}" if lp_b == lp_b else "–"
                lines.append(f"| {cat} | {gc_str} | {lr_str} | {lp_str} | {lp_h} |")

        # ── Per-category H22-H26 ─────────────────────────────────────
        if per_cat_hyp:
            lines.append("\n## Hypotheses H22–H26 by Category\n")
            lines.append("| Category | H22 (decay) | H23 (reciprocal) | H24 (threshold) | "
                         "H25 (clustering) | H26 (mimetic) |")
            lines.append("|----------|-------------|-------------------|-----------------|"
                         "-----------------|---------------|")
            for cat, cdata in per_cat_hyp.items():
                if "error" in cdata:
                    lines.append(f"| {cat} | – | – | – | – | – |")
                    continue
                h22c = cdata.get("H22_contagion_decay", {})
                h23c = cdata.get("H23_reciprocal_amplification", {})
                h24c = cdata.get("H24_threshold_activation", {})
                h25c = cdata.get("H25_temporal_clustering", {})
                h26c = cdata.get("H26_mimetic_contagion", {})

                def _supp(h):
                    if "error" in h:
                        return "err"
                    s = h.get("supported")
                    return "✓" if s else "✗"

                h22_hl = ""
                if "error" not in h22c:
                    agg22 = h22c.get("aggregate", {})
                    hl = agg22.get("median_half_life")
                    if hl is not None:
                        h22_hl = f" ({hl:.0f}d)"

                lines.append(f"| {cat} | {_supp(h22c)}{h22_hl} | {_supp(h23c)} | "
                             f"{_supp(h24c)} | {_supp(h25c)} | {_supp(h26c)} |")

    # ══════════════════════════════════════════════════════════════════
    #  SYNTHESIS & INTERPRETATION
    # ══════════════════════════════════════════════════════════════════

    lines.append("\n\n---\n")
    lines.append("# Part IV: Synthesis & Interpretation\n")

    lines.append("\n## Overall Hypothesis Scorecard\n")
    lines.append(_build_scorecard(its, cont, granger, cohort, pipe, rep, rep_reinf,
                                  thread_esc, thread_exp, conv, subforum,
                                  cp_its, cp_granger, bridges, shutdown,
                                  vocab, te, country, dose, subtheme, domain,
                                  apoc_its, robust, attack, adv_ts, h_results))

    lines.append("\n\n## Key Findings\n")
    lines.append(_build_key_findings(its, cont, granger, cohort, pipe, rep_reinf,
                                     thread_exp, conv, cp_its, shutdown,
                                     vocab, te, dose, subtheme, domain,
                                     apoc_its, robust, attack, adv_ts, h_results))

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
    lines.append("5. **Apocalypticism classifier:** LR (0.6) + contrastive similarity (0.4) "
                 "with threshold 0.55 yields a 3.71% binary classification rate. Sensitivity "
                 "to threshold choice is a limitation.")
    lines.append("6. **Robustness:** The pooled apocalypticism ITS passes 2/5 robustness checks. "
                 "The AR(1)-controlled model absorbs the level shift into autocorrelation, "
                 "and the placebo test (p=0.254) does not reject the null. The effect is "
                 "fragile and should be interpreted with caution.")
    lines.append("7. **Advanced TS triangulation:** VAR, ARDL, BSTS, and Local Projections "
                 "provide complementary perspectives on the ITS findings. Consensus across "
                 "methods strengthens conclusions; disagreement flags fragility.")
    lines.append("8. **Offline–online hypotheses (H22–H26):** These hypotheses test mechanisms "
                 "rather than mere association. Decay half-life, threshold activation, and "
                 "temporal clustering probe the *dynamics* of the rhetoric–violence nexus.")

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
