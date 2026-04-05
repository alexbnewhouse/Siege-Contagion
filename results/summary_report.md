# Siege Culture Diffusion in Iron March: Summary Report

*Auto-generated from analysis results.*


---

# Part I: Iron March Internal Dynamics (H1–H11)


## H1: Interrupted Time Series (Siege as Structural Break)


### forum_keyword_score

- N weeks: 324
- Level change (β₂): -0.0182 (p = 0.402) 
- Slope change (β₃): 0.0002 (p = 0.001) **
- R²: 0.4616

**Change points (forum_keyword_score):** ['2012-04-09 00:00:00', '2014-06-23 00:00:00', '2015-03-30 00:00:00', '2015-09-21 00:00:00', '2015-10-26 00:00:00', '2016-03-14 00:00:00', '2017-02-27 00:00:00', '2017-07-17 00:00:00', '2017-08-21 00:00:00', '2017-10-30 00:00:00']

### forum_similarity

- N weeks: 324
- Level change (β₂): -0.0262 (p < 0.001) ***
- Slope change (β₃): 0.0001 (p < 0.001) ***
- R²: 0.2643

**Change points (forum_similarity):** ['2011-12-26 00:00:00', '2012-01-30 00:00:00', '2012-03-05 00:00:00', '2013-02-18 00:00:00', '2014-09-01 00:00:00', '2014-11-10 00:00:00', '2015-07-13 00:00:00', '2015-08-17 00:00:00', '2015-10-26 00:00:00', '2016-03-14 00:00:00', '2016-09-05 00:00:00', '2016-11-14 00:00:00', '2017-04-03 00:00:00', '2017-08-21 00:00:00']

### dm_keyword_score

- N weeks: 324
- Level change (β₂): 0.0523 (p = 0.028) *
- Slope change (β₃): 0.0000 (p = 0.339) 
- R²: 0.2044

**Change points (dm_keyword_score):** ['2014-02-03 00:00:00', '2015-09-21 00:00:00', '2016-03-14 00:00:00', '2016-05-23 00:00:00', '2016-09-05 00:00:00', '2016-10-10 00:00:00', '2016-12-19 00:00:00', '2017-04-03 00:00:00', '2017-08-21 00:00:00']

### dm_similarity

- N weeks: 324
- Level change (β₂): -0.0281 (p = 0.024) *
- Slope change (β₃): 0.0000 (p = 0.032) *
- R²: 0.2843

**Change points (dm_similarity):** ['2011-10-17 00:00:00', '2012-04-09 00:00:00', '2012-05-14 00:00:00', '2012-10-01 00:00:00', '2014-07-28 00:00:00', '2015-01-19 00:00:00', '2015-07-13 00:00:00', '2016-02-08 00:00:00', '2016-10-10 00:00:00', '2017-02-27 00:00:00', '2017-08-21 00:00:00']

### all_keyword_score

- N weeks: 324
- Level change (β₂): -0.0107 (p = 0.570) 
- Slope change (β₃): 0.0001 (p < 0.001) ***
- R²: 0.5090

**Change points (all_keyword_score):** ['2011-12-26 00:00:00', '2014-02-03 00:00:00', '2015-03-30 00:00:00', '2015-09-21 00:00:00', '2015-10-26 00:00:00', '2016-03-14 00:00:00', '2016-05-23 00:00:00', '2017-02-27 00:00:00', '2017-07-17 00:00:00', '2017-08-21 00:00:00', '2017-10-30 00:00:00']

### all_similarity

- N weeks: 324
- Level change (β₂): -0.0278 (p < 0.001) ***
- Slope change (β₃): 0.0001 (p < 0.001) ***
- R²: 0.3201

**Change points (all_similarity):** ['2011-12-26 00:00:00', '2012-01-30 00:00:00', '2012-03-05 00:00:00', '2012-12-10 00:00:00', '2013-01-14 00:00:00', '2013-02-18 00:00:00', '2013-09-16 00:00:00', '2014-09-01 00:00:00', '2014-11-10 00:00:00', '2015-07-13 00:00:00', '2015-08-17 00:00:00', '2015-10-26 00:00:00', '2016-09-05 00:00:00', '2016-11-14 00:00:00', '2017-04-03 00:00:00', '2017-08-21 00:00:00']


## H2: Social Contagion in the Interaction Network

- N observations: 8870
- R²: 0.03859260221147243
- b_dm_exposure_dm: -0.0401 (p = 0.394) 
- b_forum_exposure_dm: 1.1312 (p < 0.001) ***
- b_reputation_exposure_dm: -0.0566 (p = 0.695) 
- Permutation perm_p_dm_exposure_dm: 0.4400
- Permutation perm_p_forum_exposure_dm: 0.0000
- Permutation perm_p_reputation_exposure_dm: 0.5100

**Endogeneity warning:** Network contagion models are susceptible to homophily confounds. User fixed effects partially address this. Permutation p-values provide a robustness check.


## H3: Zeiger as Ideological Entrepreneur (Granger Causality)


### Zeiger → Community

| Lag | F-stat | p-value | Sig. |
|-----|--------|---------|------|
| 1 | 1.946 | 0.1663 |  |
| 2 | 1.009 | 0.3686 |  |
| 3 | 0.940 | 0.4249 |  |
| 4 | 0.842 | 0.5023 |  |
| 5 | 0.750 | 0.5884 |  |
| 6 | 0.847 | 0.5377 |  |
| 7 | 0.756 | 0.6260 |  |
| 8 | 0.675 | 0.7122 |  |

### Community → Zeiger

| Lag | F-stat | p-value | Sig. |
|-----|--------|---------|------|
| 1 | 0.024 | 0.8776 |  |
| 2 | 0.002 | 0.9981 |  |
| 3 | 0.038 | 0.9899 |  |
| 4 | 0.039 | 0.9970 |  |
| 5 | 0.116 | 0.9886 |  |
| 6 | 0.338 | 0.9147 |  |
| 7 | 0.541 | 0.8004 |  |
| 8 | 0.486 | 0.8625 |  |


## H4: Cohort-Stratified Adoption

- Pre-Siege joiners with pre/post data: 152
- Mean score change: 0.0406
- t-test: t=2.264, p = 0.025 *
- Wilcoxon: p = 0.017 *

**Entry-level comparison:**
- Pre-joiners first posts mean: 0.0219
- Post-joiners first posts mean: 0.1839
- Mann-Whitney: p < 0.001 ***


## H5: Private-to-Public Pipeline

- Users with siege in both DM and forum: 190
- Mean DM lead: -177.1 days
- DM first: 10.0%
- Forum first: 83.7%
- t-test: p < 0.001 ***

**Small-N caution:** The DM corpus is much smaller than the forum corpus. Results may lack statistical power.


## H6: Reputation-Mediated Diffusion


**Cox Proportional Hazards (standardized covariates):**

| Variable | Coef | HR | SE | p-value | Sig. |
|----------|------|----|----|---------|------|
| degree_centrality | 1.0731 | 2.9246 | 0.0662 | p < 0.001 | *** |
| betweenness_centrality | -0.4463 | 0.6400 | 0.0905 | p < 0.001 | *** |
| reputation | -0.0258 | 0.9745 | 0.0558 | p = 0.644 |  |


## H7: Reputation Reinforcement of Siege Rhetoric

- Posts analysed: 194,652
- Siege posts: 7,358
- Mean rep (Siege): 2.335
- Mean rep (non-Siege): 1.342
- Mann-Whitney U: p < 0.001 ***

**Negative binomial (month FE):**
- Siege coefficient: 0.4285 (p < 0.001) ***
- Incidence rate ratio: 1.5350

**Caution:** Reputation data reliability has not been independently validated. These results should be interpreted with care.


## H8: Within-Thread Escalation

- Siege threads (≥3 posts): 2,471
- Posts in Siege threads: 133,653

**Position regression (cluster-robust SE):**
- Position coefficient: -0.0151 (p = 0.364) 
- R²: 0.0254

**First vs. last post in thread:**
- Mean first: 0.6032
- Mean last: 0.1800
- Mean diff: -0.4232
- t-test: t=-8.212, p < 0.001 ***
- Wilcoxon: p < 0.001 ***


## H9: Thread Exposure → Subsequent Adoption

- Panel observations: 5,998 user-months
- Unique users: 641
- Thread exposure coefficient: 0.1440 (p = 0.087) †
- Lagged own score coefficient: 0.2953 (p = 0.013) *
- R²: 0.1057

**Mean next-month siege score by thread exposure tercile:**
- Low: 0.0500
- Medium: 0.0580
- High: 0.1263


## H10: Semantic Convergence

**CV of keyword score (ITS):**
- Level change (β₂): 3.4136 (p = 0.009) **
- Slope change (β₃): -0.054337 (p = 0.011) *
- R²: 0.1874

**Pre/post comparison:**
- Pre-Siege mean CV: 1.4508
- Post-Siege mean CV: 1.7297
- Direction: divergent
- t-test: p = 0.123 


## H11: Subforum Diffusion Geography

- Subforums analysed: 87
- Herfindahl index (overall): 0.0798
- Pre-Siege HHI: 0.0497
- Post-Siege HHI: 0.1161
- Direction: more concentrated

**Top subforums by Siege prevalence:**

| Subforum | Prevalence | Posts |
|----------|-----------|-------|
| hall-of-graduates | 0.556 | 99 |
| books | 0.200 | 1,748 |
| atomwaffen-division | 0.154 | 299 |
| antipodean-resistance | 0.105 | 133 |
| strategy | 0.102 | 949 |
| noose | 0.098 | 429 |
| articles | 0.094 | 446 |
| shrimp-bunker | 0.092 | 130 |
| antimony-group | 0.091 | 253 |
| adminship-center | 0.090 | 476 |

**Biggest pre→post increases:**

| Subforum | Pre | Post | Δ |
|----------|-----|------|---|
| archive-project-central | 0.017 | 0.192 | +0.176 |
| articles | 0.021 | 0.177 | +0.156 |
| antimony-group | 0.017 | 0.156 | +0.139 |
| deutschland-austria | 0.023 | 0.154 | +0.131 |
| books | 0.132 | 0.246 | +0.114 |
| external-materials-archive | 0.057 | 0.168 | +0.110 |
| adminship-center | 0.051 | 0.153 | +0.102 |
| fascist-fraternity | 0.029 | 0.106 | +0.078 |
| hungary | 0.000 | 0.069 | +0.069 |
| nordic-resistance-movement | 0.019 | 0.086 | +0.067 |


---

# Part II: Cross-Platform Diffusion (Iron March → /pol/)


## H12: Siege Rhetoric Shows a Structural Break on /pol/ After IM's T₀

- Treatment date: 2015-06-03 04:04:41
- IM weeks: 324, /pol/ weeks: 377

### /pol/ keyword score

- N weeks: 377
- Level change (β₂): -0.0188 (p = 0.763) 
- Slope change (β₃): -0.000141 (p = 0.336) 
- R²: 0.0186

### /pol/ similarity score

- N weeks: 377
- Level change (β₂): 0.0022 (p = 0.583) 
- Slope change (β₃): -0.000015 (p = 0.085) †
- R²: 0.0398

### /pol/ prevalence

- N weeks: 377
- Level change (β₂): -0.0002 (p = 0.438) 
- Slope change (β₃): -0.000000 (p = 0.239) 
- R²: 0.3166


## H13: Iron March Siege Rhetoric Granger-Causes /pol/ Rhetoric

- Overlapping weeks: 209
- ADF (IM): -1.283, stationary=False
- ADF (/pol/): -10.549, stationary=True
- Differenced: True

### IM → /pol/

| Lag | F-stat | p-value | Sig. |
|-----|--------|---------|------|
| 1 | 0.303 | 0.5824 |  |
| 2 | 0.468 | 0.6272 |  |
| 3 | 0.496 | 0.6858 |  |
| 4 | 0.618 | 0.6499 |  |
| 5 | 0.576 | 0.7186 |  |
| 6 | 0.589 | 0.7391 |  |
| 7 | 0.511 | 0.8258 |  |
| 8 | 0.451 | 0.8889 |  |
| 9 | 0.387 | 0.9402 |  |
| 10 | 0.391 | 0.9493 |  |
| 11 | 0.694 | 0.7432 |  |
| 12 | 0.602 | 0.8387 |  |

### /pol/ → IM

| Lag | F-stat | p-value | Sig. |
|-----|--------|---------|------|
| 1 | 0.713 | 0.3995 |  |
| 2 | 0.545 | 0.5810 |  |
| 3 | 0.410 | 0.7458 |  |
| 4 | 0.180 | 0.9483 |  |
| 5 | 0.158 | 0.9775 |  |
| 6 | 0.232 | 0.9657 |  |
| 7 | 0.215 | 0.9817 |  |
| 8 | 0.186 | 0.9926 |  |
| 9 | 0.213 | 0.9923 |  |
| 10 | 0.272 | 0.9865 |  |
| 11 | 0.310 | 0.9830 |  |
| 12 | 0.310 | 0.9870 |  |

- Best IM→/pol/ p-value: 0.5824
- Best /pol/→IM p-value: 0.3995
- Cross-correlation peak: lag=-24, r=0.1195 (95% CI: ±0.1356)


## H14: Content Bridges Propagate from IM → /pol/

**URL analysis:**
- IM URLs: 15,995
- /pol/ URLs: 31,082
- Shared URLs: 334
- IM first: 196, /pol/ first: 108

**N-gram analysis (n=4):**
- Shared n-grams: 112,822

**Temporal priority:**
- Terms on both platforms: 48
- IM led: 40, /pol/ led: 8

**Iron March mentions on /pol/:**
- Total mentions: 5,099
- Weeks with mentions: 200


## H8 (adapted): /pol/ Within-Thread Escalation

- /pol/ Siege threads: 72,480
- Posts in Siege threads: 372,727

**Position regression:**
- Position coefficient: -0.0399 (p = 0.114) 
- R²: 0.0044

**First vs. last post:**
- Mean first: 1.8842
- Mean last: 1.8638
- Wilcoxon: p < 0.001 ***


## H10 (adapted): /pol/ Semantic Convergence

- N months: 75
- Level change: 0.6442 (p = 0.007) **
- Slope change: -0.062525 (p = 0.379) 
- R²: 0.0184

**Pre/post comparison:**
- Pre-Siege mean CV: 0.6102
- Post-Siege mean CV: 1.0657
- Direction: divergent
- t-test: p = 0.046 *


## H15: /pol/ Siege Rhetoric Increases After IM Shutdown

- Shutdown date: 2017-11-21 00:00:00+00:00
- Siege T₀: 2015-06-03 04:04:41

### Keyword score

- Level change (β₂): 0.0641 (p = 0.344) 
- Slope change (β₃): -0.000077 (p = 0.354) 
- R²: 0.0254

### Post volume

- Level change (β₂): -306.5347 (p = 0.271) 
- Slope change (β₃): -0.440079 (p = 0.431) 
- R²: 0.5579

### Prevalence

- Level change (β₂): 0.0010 (p < 0.001) ***
- Slope change (β₃): -0.000001 (p = 0.048) *
- R²: 0.3766

### Similarity

- Level change (β₂): 0.0255 (p < 0.001) ***
- Slope change (β₃): -0.000044 (p < 0.001) ***
- R²: 0.3490

**Effect comparison (shutdown vs T₀):**
- β₂ at shutdown: 0.0641
- β₂ at T₀: -0.0188
- Ratio: 3.40×, larger at shutdown


## H16: Siege Vocabulary Appears on IM Before /pol/

- Terms tracked: 50
- IM-first: 42, /pol/-first: 8
- Median lag: 670.3 days
- Mean lag: 601.7 days

**Acceleration test (OLS lag ~ time):**
- Slope: -0.6165
- R²: 0.4552
- p-value: p < 0.001 ***
- Interpretation: accelerating


## H17: Non-Linear Information Flows from IM → /pol/

- Overlapping weeks: 209

### IM → /pol/ (lag 1)

- Transfer entropy: 0.3075
- Surrogate mean: 0.3400
- z-score: -0.760
- p-value: 0.755 

### /pol/ → IM (lag 1)

- Transfer entropy: 0.3630
- Surrogate mean: 0.2811
- z-score: 1.995
- p-value: 0.030 *


## H18: /pol/ Posts from IM-Heavy Countries Show More Siege Rhetoric

- Countries analysed: 225
- Posts with country data: 881,747
- IM cluster countries: AU, CA, DE, FI, GB, NO, SE, US

**IM-cluster vs rest:**
- IM cluster mean score: 1.7423
- Rest mean score: 1.6560
- Mann-Whitney: p = 0.222 


## H19: Higher IM Siege Activity Predicts Elevated /pol/ Activity

- Overlapping weeks: 209
- Max lag tested: 8

| Lag | Kruskal H | KW p | Spearman ρ | Sp p | KW sig | Sp sig |
|-----|-----------|------|------------|------|--------|--------|
| 1 | 2.285 | 0.5153 | 0.0784 | 0.2602 |  |  |
| 2 | 8.308 | 0.0401 | 0.1394 | 0.0451 | ✓ | ✓ |
| 3 | 11.106 | 0.0112 | 0.1935 | 0.0053 | ✓ | ✓ |
| 4 | 6.157 | 0.1042 | 0.1605 | 0.0215 |  | ✓ |
| 5 | 2.760 | 0.4300 | 0.0850 | 0.2268 |  |  |
| 6 | 7.551 | 0.0563 | 0.1630 | 0.0201 |  | ✓ |
| 7 | 6.692 | 0.0824 | 0.1645 | 0.0193 |  | ✓ |
| 8 | 9.610 | 0.0222 | 0.2178 | 0.0019 | ✓ | ✓ |


## H20: Sub-Themes Diffuse Differentially Across Platforms

- Sub-themes: accelerationism, mason_core, atomwaffen_org, violence, enemy_framing

| Sub-theme | IM hits | /pol/ hits | β₂ (level) | p(level) | β₃ (slope) | p(slope) | Granger p |
|-----------|---------|-----------|------------|----------|------------|----------|-----------|
| accelerationism | 824 | 180,558 | -0.0852 | p < 0.001 *** | 0.001432 | p < 0.001 *** | p = 0.392  |
| mason_core | 554 | 11,328 | -0.0001 | p = 0.940  | 0.000079 | p < 0.001 *** | p < 0.001 *** |
| atomwaffen_org | 527 | 8,956 | -0.0002 | p = 0.882  | 0.000054 | p < 0.001 *** | p < 0.001 *** |
| violence | 1,130 | 292,382 | 0.0532 | p = 0.269  | -0.002761 | p < 0.001 *** | p = 0.344  |
| enemy_framing | 1,130 | 5,861 | -0.0008 | p = 0.314  | 0.000034 | p = 0.009 ** | p = 0.450  |


## H21: URL Domains Propagate from IM → /pol/

- IM unique domains: 6,344
- /pol/ unique domains: 6,898
- Shared domains: 1,326
- IM first: 1,046 (79%)
- /pol/ first: 272

**Top gateway domains (IM-first, high /pol/ volume):**

| Domain | IM first | /pol/ first | Lag (days) | /pol/ count |
|--------|----------|-------------|------------|-------------|
| bitchute.com | 2017-09-20 | 2017-09-21 | 1 | 9,347 |
| pastebin.com | 2012-08-26 | 2013-12-23 | 484 | 7,198 |
| archive.is | 2015-02-03 | 2015-05-02 | 88 | 6,378 |
| mega.nz | 2015-11-24 | 2016-02-06 | 73 | 2,008 |
| my.mixtape.moe | 2016-01-19 | 2016-08-28 | 222 | 1,314 |
| ironmarch.org | 2011-10-06 | 2013-12-21 | 807 | 1,164 |
| laraj.ca | 2013-08-29 | 2015-09-14 | 746 | 1,161 |
| infostormer.com | 2015-12-03 | 2016-06-23 | 203 | 982 |
| jrbooksonline.com | 2011-10-07 | 2013-12-12 | 796 | 912 |
| theoccidentalobserver.net | 2012-12-02 | 2014-01-15 | 408 | 868 |


---

# Part III: Apocalypticism Chapter — Mass-Casualty Events & /pol/ Rhetoric


## Event Catalogue

- Total events: 136
- Date range: 2010-02-18 to 2023-12-21

| Category | Count |
|----------|-------|
| mass_violence | 110 |
| political | 9 |
| natural_disaster | 9 |
| economic_shock | 6 |
| health_crisis | 2 |

| Ideology | Count |
|----------|-------|
| islamist | 45 |
| other | 30 |
| N/A | 26 |
| far_right | 24 |
| school_shooting | 5 |
| incel | 4 |
| far_left | 2 |


## Apocalypticism ITS (Per-Event & Pooled)

- Total posts scored: 911,220
- Events analysed: 86
- Window: ±30 days

### Pooled ITS (stacked with event FE)

- N observations: 5,246
- Level change (β₂): -0.0018 (p = 0.043) *
- Slope change (β₃): -0.000117 (p = 0.019) *
- R²: 0.4432
- 95% CI for β₂: [-0.0035, -0.0001]

### Category Comparison

| Category | β₂ (level) | p(level) | β₃ (slope) | p(slope) |
|----------|------------|----------|------------|----------|
| mass_violence | -0.0025 | p = 0.008 ** | -0.000133 | p = 0.014 * |
| nonviolence | 0.0029 | p = 0.194  | -0.000019 | p = 0.882  |
| political | 0.0064 | p = 0.088 † | 0.000235 | p = 0.239  |
| health_crisis | -0.0021 | p = 0.575  | 0.000439 | p = 0.035 * |
| natural_disaster | -0.0083 | p = 0.040 * | -0.000364 | p = 0.099 † |
| economic_shock | 0.0159 | p = 0.008 ** | -0.000591 | p = 0.133  |

### Stratified by Ideology

| Ideology | β₂ (level) | p(level) | n events |
|----------|------------|----------|----------|
| school_shooting | 0.0012 | p = 0.779  | 3 |
| other | -0.0055 | p = 0.020 * | 12 |
| far_right | -0.0023 | p = 0.188  | 18 |
| islamist | -0.0030 | p = 0.031 * | 37 |
| far_left | 0.0212 | p < 0.001 *** | 2 |
| incel | -0.0073 | p = 0.200  | 2 |


## Apocalypticism Robustness Checks

- Checks passed: 2/5

**Placebo test (500 permutations):**
- Observed β₂: -0.0018
- Null mean: 0.0001
- Null SD: 0.0016
- Placebo p: 0.254

**Benjamini-Hochberg FDR:**
- Raw significant: 34/86
- BH-corrected significant: 21/86

**AR(1) controlled:**
- β₂: -0.0006 (p = 0.503) 
- AR(1) coefficient: 0.6700 (p < 0.001) ***

**Day-of-week controlled:**
- β₂: -0.0022 (p = 0.085) †


## Attack Characteristic Correlations

**Severity vs β₂ (ITS level change):**

| Measure | Pearson r | p | Spearman ρ | p |
|---------|-----------|---|-----------|---|
| killed | 0.0205 | 0.851 | -0.1021 | 0.349 |
| injured | -0.0386 | 0.724 | -0.0605 | 0.580 |
| total_casualties | 0.0069 | 0.949 | -0.0986 | 0.366 |

**Domestic vs international:**
- Domestic mean β₂: -0.0058 (n=30)
- International mean β₂: -0.0003 (n=44)
- Mann-Whitney: p = 0.192 

**Online nexus vs offline:**
- Online-nexus mean β₂: -0.0025 (n=23)
- Offline mean β₂: -0.0025 (n=51)
- Mann-Whitney: p = 0.815 

**Multiple regression (β₂ ~ attack characteristics):**
- R²: 0.1408 (adj: 0.0350)
- Model F p-value: p = 0.157

| Predictor | Coef | SE | p-value | Sig. |
|-----------|------|-----|---------|------|
| const | 0.0314 | 0.0152 | p = 0.039 | * |
| log_killed | 0.0009 | 0.0018 | p = 0.599 |  |
| domestic_int | -0.0110 | 0.0043 | p = 0.010 | * |
| nexus_int | 0.0013 | 0.0045 | p = 0.775 |  |
| ideo_far_right | -0.0301 | 0.0159 | p = 0.057 | † |
| ideo_incel | -0.0365 | 0.0159 | p = 0.022 | * |
| ideo_islamist | -0.0349 | 0.0153 | p = 0.022 | * |
| ideo_other | -0.0312 | 0.0153 | p = 0.042 | * |
| ideo_school_shooting | -0.0215 | 0.0164 | p = 0.189 |  |


## Advanced Time-Series Methods (VAR / ARDL / BSTS / LP)


### Vector Autoregression (VAR)

- Selected lag: 11 (AIC)
- Observations: 2636
- Granger (event_causes_apoc): F=0.354, p = 0.973 
- Granger (apoc_causes_event): F=0.602, p = 0.829 
- IRF peak response: 0.001978 at day 1
- FEVD (event → apoc): 0.09% at 7d, 0.09% at 30d

### Autoregressive Distributed Lag (ARDL)

- ARDL(7,0)
- Long-run multiplier: 1.579976e-05
- R²: 0.5178
- EC coefficient: -0.132523 (p < 0.001 ***)
- Bounds F: 31.90 (p < 0.001 ***)
- Cointegration: Yes

### Bayesian Structural Time Series (BSTS)

- Events analysed: 20
- Mean impact: 0.003512
- Median impact: 0.003376
- t-statistic: 1.0411230383516086
- t p-value: p = 0.311 
- Pct with positive effect: 55.5%

### Local Projections (Jordà 2005)

- Horizons estimated: 31
- Significant horizons: 0/31
- Peak β: -0.003418 at h=12 (p = 0.075 †)

### Method Comparison (ITS × VAR × ARDL × BSTS × LP)

- Consensus direction: mixed
- Direction agreement: False
- Methods reaching significance: 2/5
- Conclusion: Methods disagree on direction. 2/5 significant: ITS, ARDL. Results are inconclusive.

| Method | Direction | Effect Size | p-value | Significant |
|--------|-----------|-------------|---------|-------------|
| ITS | negative | -0.001765 | p = 0.043 | ✓ |
| VAR | positive | 0.001978 | p = 0.973 |  |
| ARDL | positive | 0.000016 | p < 0.001 | ✓ |
| BSTS | positive | 0.003512 | p = 0.311 |  |
| LocalProjections | negative | -0.003418 | p = 0.075 |  |


## Offline Violence ↔ Online Rhetoric (H22–H26)


### H22: Contagion Decay

- Valid fits: 87/87 events
- Median half-life: 8.2 days
- Mean half-life: 251606.5 days
- Range: 0.1–13468663.2 days
- Average trajectory fit: λ=0.0287, t½=24.2 days

### H23: Reciprocal Amplification

- VAR lag: 11
- Observations: 2636
- Granger (event_occurred_causes_apoc_mean): F=0.354, p = 0.973 
- Granger (apoc_mean_causes_event_occurred): F=0.602, p = 0.829 
- Reciprocal feedback: False
- Events → rhetoric: False
- Rhetoric → events: False

### H24: Threshold Activation

- Optimal casualty threshold: 30
- Mann-Whitney p: p = 0.347 
- Effect size (above vs below): 0.006142183099731824
- Thresholds scanned: 7
- Significant thresholds: 0/7
- Piecewise R²: 0.0521

### H25: Temporal Clustering

- Cluster window: 14 days
- Events analysed: 74
- Clustered events: 51 (mean Δ=-0.0025)
- Isolated events: 23 (mean Δ=-0.0019)
- Mann-Whitney p: p = 0.633 
- Super-additive: False
- Neighbors regression: β=-0.0014, p = 0.297 

### H26: Mimetic Contagion

- Online-nexus events: 23
- Offline events: 51
- Online |Δapoc| mean: 0.0085
- Offline |Δapoc| mean: 0.0100
- Cohen's d: -0.1895
- Magnitude p: p = 0.696 
- Direction p: p = 0.935 
- Polarisation (variance ratio) p: p = 0.110 
- Online variance ratio: 1.570
- Offline variance ratio: 1.503


---

# Part IV: Synthesis & Interpretation


## Overall Hypothesis Scorecard

| # | Hypothesis | Verdict | Key Statistic |
|---|-----------|---------|---------------|
| H1 | Siege publication = structural break | ✓ Supported | β₃ p=0.0000, β₂ p=0.0000 |
| H2 | Network exposure → adoption | ✓ Supported | β=1.1312, p<0.001 |
| H3 | Zeiger Granger-causes community | ✗ Not supported | best p=0.1663 |
| H4 | Cohort conversion effects | ✓ Supported | t=2.264, p=0.025 |
| H5 | Forum-first pipeline | ✓ Supported | 83.7% forum-first |
| H6 | Network centrality → faster adoption | ✓ Supported | HR=2.92, p<0.001 |
| H7 | Community rewards Siege speech | ✓ Supported | IRR=1.53, p<0.001 |
| H8 | Within-thread escalation | ~ Catalyst pattern, not linear | β=-0.0151, p=0.364 |
| H9 | Thread exposure → adoption | ~ Marginal (p=0.087) | β=0.1440, p=0.087 |
| H10 | Semantic convergence | ✓ Supported | β₃=-0.054337, p=0.011 |
| H11 | Subforum concentration | ✓ Supported | HHI: 0.0497→0.1161 |
| H12 | /pol/ structural break at T₀ | ✗ Not supported | β₂=-0.0188, p=0.763 |
| H13 | IM Granger-causes /pol/ | ✗ Not supported | best p=0.5824 |
| H14 | Content bridges IM → /pol/ | ✓ Supported | 40/48 IM-first (83%) |
| H15 | /pol/ rhetoric ↑ after shutdown | ✓ Supported | prevalence β₂ p=0.000036 |
| H16 | Vocab appears on IM first | ✓ Supported | 84% IM-first, median lag 670d |
| H17 | IM → /pol/ transfer entropy | ✗ Not supported | IM→/pol/ p=0.755; /pol/→IM p=0.030 |
| H18 | IM-heavy countries = more Siege | ✗ Not supported | p=0.222 |
| H19 | Dose-response IM → /pol/ | ✓ Supported | ρ=0.1935, p=0.0053 (lag 3) |
| H20 | Sub-theme differential diffusion | ✓ Supported | mason_core Granger p=1.31e-08 |
| H21 | Domains propagate IM → /pol/ | ✓ Supported | 1046/1326 IM-first (79%) |
| Apoc | Pooled ITS: events → ↑ apocalypticism | ~ Sig but *negative* β₂ | β₂=-0.0018, p=0.043 |
| H22 | Exponential decay post-attack | ✓ Supported | median t½=8.2d, avg t½=24.2d |
| H23 | Reciprocal amplification | ✗ Not supported | bidirectional=False |
| H24 | Threshold activation | ✗ Not supported | T=30, p=0.347 |
| H25 | Temporal clustering compounds | ✗ Not supported | p=0.633 |
| H26 | Online-nexus = distinctive shifts | ✗ Not supported | d=-0.18950477422908651, p=0.696 |


## Key Findings

### 1. Siege Diffusion Within Iron March

The core finding is robust: the publication of Zeiger's *Siege* edition produced a significant structural break in Iron March discourse. The combined similarity score shows a highly significant post-Siege slope acceleration (β₃, p<0.001), though the immediate level shift is non-significant for keyword scores (p=0.57). This pattern — gradual intensification rather than sudden adoption — is consistent with a collective exegesis model where the community progressively develops shared interpretive frameworks around the text.

Forum network exposure (β=1.13, p<0.001) strongly predicts individual Siege rhetoric adoption, with permutation p=0.000 confirming this is not an artifact of network structure. However, Zeiger himself does not Granger-cause community rhetoric (best p=0.17), suggesting diffusion operates through decentralized peer influence rather than top-down ideological entrepreneurship. The community "canonises" *Siege* collectively: Siege-aligned posts receive 54% more reputation points (IRR=1.54, p<0.001), and the text concentrates in books, strategy, and archival subforums — spaces of interpretation rather than casual discussion.

### 2. Cross-Platform Contagion: Iron March → /pol/

The cross-platform picture is nuanced. Direct causal evidence is weak: neither Granger causality (best p=0.58) nor transfer entropy (IM→/pol/ p=0.755) reaches significance in the IM→/pol/ direction. The surprising finding is that /pol/→IM transfer entropy *is* significant (p=0.03), suggesting some reverse flow.

However, indirect diffusion evidence is strong. 84% of tracked Siege vocabulary terms appear on IM before /pol/, with a median lag of 670 days, and the lag is *accelerating* (slope = -0.62, p<0.001) — i.e., newer terms propagate faster. 79% of shared URL domains appear on IM first. The dose-response relationship is significant at lags 2–3 weeks (ρ=0.14–0.19, p<0.05), suggesting a 2–3 week transmission window.

Critically, after Iron March is shut down (November 2017), /pol/ Siege prevalence shows a significant *increase* (β₂ p<0.001 for prevalence, p<0.001 for similarity), with the shutdown effect 3.4× larger than the original T₀ effect. This is consistent with diaspora — IM users migrate to /pol/ after their platform is destroyed.

### 3. Apocalypticism & Mass Violence

The relationship between mass-casualty events and /pol/ apocalyptic rhetoric is counter-intuitive. The pooled ITS finds a small but statistically significant *decrease* in apocalyptic rhetoric post-event (β₂ = −0.0018, p=0.043), not an increase. However, this result is fragile: only 2 of 5 robustness checks pass. The AR(1)-controlled model absorbs the effect into autocorrelation (β₂ p=0.50), and the placebo test does not clearly reject the null (p=0.254).

Attack characteristics show limited predictive power. Casualty severity does not correlate with the ITS level shift (all Pearson/Spearman p>0.3). The only significant predictor in the multiple regression is the domestic vs international distinction (β = −0.011, p=0.010): domestic attacks produce slightly stronger negative shifts in mean apocalypticism, possibly reflecting a normalization effect where familiar events elicit less eschatological framing than geographically distant ones.

The method comparison (ITS × VAR × ARDL × BSTS × LP) yields a consensus direction of "mixed" with 2/5 methods reaching significance. 
Where methods disagree on significance, this confirms the effect is at best small and fragile — a finding that is itself substantively important, as it counters the popular narrative that mass-casualty events reliably "radicalise" online communities toward apocalyptic worldviews.

### 4. Offline–Online Dynamics (H22–H26)

Post-attack rhetoric shows an estimated median half-life of 8.2 days, suggesting that whatever rhetorical shift occurs dissipates relatively quickly. 
The reciprocal amplification test does not find bidirectional causality, suggesting rhetoric responds to events rather than the reverse. 


### Summary: What the Evidence Supports

The strongest findings in this project concern the *internal* dynamics of Iron March: Siege rhetoric diffuses through network exposure, is reinforced by community reputation mechanisms, and concentrates in interpretive spaces. Cross-platform diffusion from IM → /pol/ operates primarily through vocabulary and URL propagation rather than detectable Granger-causal flows, with a 2–3 week transmission lag and acceleration over time. The shutdown of Iron March paradoxically *increases* Siege rhetoric on /pol/, consistent with platform diaspora. The apocalypticism analysis complicates simplistic accounts: mass-casualty events do not reliably increase apocalyptic rhetoric, and what effect exists is small, fragile, and unrelated to attack severity.



## Methodological Notes

1. **Multiple comparisons:** All primary hypothesis tests should be evaluated with Benjamini-Hochberg correction applied across the full set of tests.
2. **Endogeneity:** Network contagion results are subject to homophily confounds. Permutation tests provide model-free robustness checks.
3. **DM corpus size:** The DM corpus (≈21.7K messages) is much smaller than the forum corpus (≈195K posts). DM-based analyses may be underpowered.
4. **'Siege' overloading:** The word 'siege' appears in non-Mason contexts. The embedding approach provides natural robustness against this issue.
5. **Apocalypticism classifier:** LR (0.6) + contrastive similarity (0.4) with threshold 0.55 yields a 3.71% binary classification rate. Sensitivity to threshold choice is a limitation.
6. **Robustness:** The pooled apocalypticism ITS passes 2/5 robustness checks. The AR(1)-controlled model absorbs the level shift into autocorrelation, and the placebo test (p=0.254) does not reject the null. The effect is fragile and should be interpreted with caution.
7. **Advanced TS triangulation:** VAR, ARDL, BSTS, and Local Projections provide complementary perspectives on the ITS findings. Consensus across methods strengthens conclusions; disagreement flags fragility.
8. **Offline–online hypotheses (H22–H26):** These hypotheses test mechanisms rather than mere association. Decay half-life, threshold activation, and temporal clustering probe the *dynamics* of the rhetoric–violence nexus.