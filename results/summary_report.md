# Siege Culture Diffusion in Iron March: Summary Report

*Auto-generated from analysis results.*


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


## Methodological Notes

1. **Multiple comparisons:** All primary hypothesis tests should be evaluated with Benjamini-Hochberg correction applied across the full set of tests.
2. **Endogeneity:** Network contagion results are subject to homophily confounds. Permutation tests provide model-free robustness checks.
3. **DM corpus size:** The DM corpus (≈21.7K messages) is much smaller than the forum corpus (≈195K posts). DM-based analyses may be underpowered.
4. **'Siege' overloading:** The word 'siege' appears in non-Mason contexts. The embedding approach provides natural robustness against this issue.