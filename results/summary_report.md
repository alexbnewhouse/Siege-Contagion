# Siege Culture Diffusion in Iron March: Summary Report

*Auto-generated from analysis results.*


## H1: Interrupted Time Series (Siege as Structural Break)


### forum_keyword_score

- N weeks: 324
- Level change (β₂): -0.0078 (p = 0.679) 
- Slope change (β₃): 0.0001 (p < 0.001) ***
- R²: 0.4466

**Change points (forum_keyword_score):** ['2014-06-23 00:00:00', '2015-03-30 00:00:00', '2015-07-13 00:00:00', '2015-09-21 00:00:00', '2015-10-26 00:00:00', '2016-03-14 00:00:00', '2017-02-27 00:00:00', '2017-05-08 00:00:00', '2017-06-12 00:00:00', '2017-07-17 00:00:00', '2017-08-21 00:00:00', '2017-10-30 00:00:00']

### forum_similarity

- N weeks: 324
- Level change (β₂): -0.0252 (p < 0.001) ***
- Slope change (β₃): 0.0001 (p < 0.001) ***
- R²: 0.2596

**Change points (forum_similarity):** ['2011-12-26 00:00:00', '2012-01-30 00:00:00', '2012-03-05 00:00:00', '2012-06-18 00:00:00', '2013-02-18 00:00:00', '2014-03-10 00:00:00', '2014-09-01 00:00:00', '2014-11-10 00:00:00', '2015-07-13 00:00:00', '2015-08-17 00:00:00', '2015-10-26 00:00:00', '2016-03-14 00:00:00', '2016-09-05 00:00:00', '2017-08-21 00:00:00']

### dm_keyword_score

- N weeks: 324
- Level change (β₂): 0.0595 (p = 0.006) **
- Slope change (β₃): 0.0000 (p = 0.304) 
- R²: 0.2438

**Change points (dm_keyword_score):** ['2015-09-21 00:00:00', '2016-03-14 00:00:00', '2016-05-23 00:00:00', '2016-10-10 00:00:00', '2016-12-19 00:00:00', '2017-04-03 00:00:00', '2017-09-25 00:00:00', '2017-10-30 00:00:00']

### dm_similarity

- N weeks: 324
- Level change (β₂): -0.0267 (p = 0.026) *
- Slope change (β₃): 0.0000 (p = 0.030) *
- R²: 0.2769

**Change points (dm_similarity):** ['2011-10-17 00:00:00', '2012-04-09 00:00:00', '2012-05-14 00:00:00', '2012-10-01 00:00:00', '2014-07-28 00:00:00', '2015-01-19 00:00:00', '2015-07-13 00:00:00', '2016-02-08 00:00:00', '2016-10-10 00:00:00', '2017-02-27 00:00:00', '2017-08-21 00:00:00']

### all_keyword_score

- N weeks: 324
- Level change (β₂): -0.0007 (p = 0.969) 
- Slope change (β₃): 0.0001 (p < 0.001) ***
- R²: 0.4920

**Change points (all_keyword_score):** ['2014-06-23 00:00:00', '2015-03-30 00:00:00', '2015-09-21 00:00:00', '2015-10-26 00:00:00', '2016-04-18 00:00:00', '2016-05-23 00:00:00', '2016-11-14 00:00:00', '2017-02-27 00:00:00', '2017-05-08 00:00:00', '2017-06-12 00:00:00', '2017-07-17 00:00:00', '2017-08-21 00:00:00', '2017-10-30 00:00:00']

### all_similarity

- N weeks: 324
- Level change (β₂): -0.0267 (p < 0.001) ***
- Slope change (β₃): 0.0001 (p < 0.001) ***
- R²: 0.3131

**Change points (all_similarity):** ['2011-12-26 00:00:00', '2012-01-30 00:00:00', '2012-03-05 00:00:00', '2012-06-18 00:00:00', '2012-12-10 00:00:00', '2013-01-14 00:00:00', '2013-02-18 00:00:00', '2013-09-16 00:00:00', '2014-09-01 00:00:00', '2014-11-10 00:00:00', '2015-07-13 00:00:00', '2015-08-17 00:00:00', '2015-11-30 00:00:00', '2016-02-08 00:00:00', '2016-10-10 00:00:00', '2017-04-03 00:00:00', '2017-08-21 00:00:00']


## H2: Social Contagion in the Interaction Network

- N observations: 8870
- R²: 0.03385167003833123
- b_dm_exposure_dm: -0.0407 (p = 0.415) 
- b_forum_exposure_dm: 1.0926 (p < 0.001) ***
- b_reputation_exposure_dm: -0.1337 (p = 0.400) 
- Permutation perm_p_dm_exposure_dm: 0.4300
- Permutation perm_p_forum_exposure_dm: 0.0050
- Permutation perm_p_reputation_exposure_dm: 0.1600

**Endogeneity warning:** Network contagion models are susceptible to homophily confounds. User fixed effects partially address this. Permutation p-values provide a robustness check.


## H3: Zeiger as Ideological Entrepreneur (Granger Causality)


### Zeiger → Community

| Lag | F-stat | p-value | Sig. |
|-----|--------|---------|------|
| 1 | 0.007 | 0.9331 |  |
| 2 | 0.022 | 0.9778 |  |
| 3 | 0.184 | 0.9070 |  |
| 4 | 0.169 | 0.9536 |  |
| 5 | 0.458 | 0.8062 |  |
| 6 | 0.360 | 0.9022 |  |
| 7 | 0.440 | 0.8740 |  |
| 8 | 0.383 | 0.9266 |  |

### Community → Zeiger

| Lag | F-stat | p-value | Sig. |
|-----|--------|---------|------|
| 1 | 0.445 | 0.5065 |  |
| 2 | 0.587 | 0.5579 |  |
| 3 | 0.678 | 0.5681 |  |
| 4 | 1.016 | 0.4041 |  |
| 5 | 0.784 | 0.5644 |  |
| 6 | 0.654 | 0.6864 |  |
| 7 | 0.726 | 0.6505 |  |
| 8 | 0.627 | 0.7527 |  |


## H4: Cohort-Stratified Adoption

- Pre-Siege joiners with pre/post data: 152
- Mean score change: 0.0322
- t-test: t=1.806, p = 0.073 †
- Wilcoxon: p = 0.001 **

**Entry-level comparison:**
- Pre-joiners first posts mean: 0.0150
- Post-joiners first posts mean: 0.1462
- Mann-Whitney: p < 0.001 ***


## H5: Private-to-Public Pipeline

- Users with siege in both DM and forum: 155
- Mean DM lead: -176.3 days
- DM first: 16.8%
- Forum first: 75.5%
- t-test: p < 0.001 ***

**Small-N caution:** The DM corpus is much smaller than the forum corpus. Results may lack statistical power.


## H6: Reputation-Mediated Diffusion


**Cox Proportional Hazards (standardized covariates):**

| Variable | Coef | HR | SE | p-value | Sig. |
|----------|------|----|----|---------|------|
| degree_centrality | 1.1103 | 3.0353 | 0.0664 | p < 0.001 | *** |
| betweenness_centrality | -0.3516 | 0.7036 | 0.0798 | p < 0.001 | *** |
| reputation | -0.0744 | 0.9283 | 0.0504 | p = 0.140 |  |


## Methodological Notes

1. **Multiple comparisons:** All primary hypothesis tests should be evaluated with Benjamini-Hochberg correction applied across the full set of tests.
2. **Endogeneity:** Network contagion results are subject to homophily confounds. Permutation tests provide model-free robustness checks.
3. **DM corpus size:** The DM corpus (≈21.7K messages) is much smaller than the forum corpus (≈195K posts). DM-based analyses may be underpowered.
4. **'Siege' overloading:** The word 'siege' appears in non-Mason contexts. The embedding approach provides natural robustness against this issue.