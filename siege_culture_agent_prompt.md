# Instructional Prompt: Testing Siege Culture Diffusion in the Iron March Dataset

## Context & Research Goal

You are a research coding agent supporting a political science dissertation on online extremism. The researcher studies the Iron March (IM) forum — a now-defunct neo-fascist message board (2011–2017) whose leaked MySQL database is publicly available. The central question is whether the user **Zeiger** (member_id `2170`) publishing the "Iron March edition" of James Mason's *Siege* (first edition circa mid-2015) catalyzed the development of **Siege Culture** — an accelerationist ideological tendency — within the IM community.

You will produce a fully reproducible Python analysis pipeline. The researcher works in Python (Polars preferred, pandas acceptable) and uses PyTorch for NLP. All code should be modular, well-documented, and output both statistical results and publication-quality visualizations.

---

## Data Source

**Repository:** https://github.com/knapply/ironmarch

This is an R package containing the full IM database as `.rda` files in the `/data` directory. You have two options for data ingestion (implement whichever works):

1. **Preferred:** Use `pyreadr` to read `.rda` files directly from the cloned repo.
2. **Fallback:** Install R, load the package, and export all relevant tables as CSV/Parquet using a helper R script.

### Key Tables

| Table | Location | Description | Key Columns |
|---|---|---|---|
| `forums_posts` | `im_forums_dfs$forums_posts` | 195,128 public forum posts | `pid`, `author_id`, `author_name`, `post_date`, `post` (HTML text), `topic_id`, `new_topic` |
| `core_message_posts` | `im_core_dfs$core_message_posts` | 21,715 private DMs | `msg_id`, `msg_topic_id`, `msg_date`, `msg_post` (HTML text), `msg_author_id` |
| `core_message_topic_user_map` | `im_core_dfs$core_message_topic_user_map` | Maps users to DM conversations | `map_user_id`, `map_topic_id`, `map_is_starter` |
| `core_message_topics` | `im_core_dfs$core_message_topics` | DM thread metadata | `mt_id`, `mt_title`, `mt_starter_id`, `mt_date` |
| `core_members` | `im_core_dfs$core_members` | 1,207 members (core DB) | `member_id`, `name`, `joined`, `member_posts`, `pp_reputation_points` |
| `orig_members` | `im_orig_dfs$orig_members` | 763 members (original DB) | `member_id`, `name`, `joined`, `posts` |
| `core_reputation_index` | `im_core_dfs$core_reputation_index` | 272,129 likes/reputation events | `member_id` (giver), `member_received` (receiver), `rep_date`, `type_id` (post id) |
| `core_follow` | `im_core_dfs$core_follow` | 8,066 follow relationships | `follow_member_id`, `follow_rel_id`, `follow_area`, `follow_added` |
| `core_search_index_tags` | `im_core_dfs$core_search_index_tags` | 5,057 content tags | `index_id`, `index_tag` (includes "siege", "james mason", "siegeculture") |
| `forums_topics` | `im_forums_dfs$forums_topics` | Forum thread metadata | `tid`, `title`, `starter_id`, `start_date`, `forum_id` |
| `core_pfields_content` | `im_core_dfs$core_pfields_content` | User profile fields | `member_id`, `field_11` (political ideology self-label) |

**Important note on `.rda` parsing:** The R package stores data as named lists of tibbles (e.g., `im_core_dfs` is a list containing `core_members`, `core_message_posts`, etc.). When using `pyreadr`, you'll get the top-level list objects. You may need to iterate over them to extract individual DataFrames. Some numeric IDs may appear as float due to R↔Python conversion artifacts — cast to int early.

Also: the `build_members()` and `build_messages()` functions in R reconcile data across `core_*` and `orig_*` tables. In Python, you should replicate this by merging on `member_id` / `msg_id`, preferring `core_*` values where both exist and filling from `orig_*`.

---

## Treatment Event

**Zeiger** (member_id `2170`) published the first "Iron March edition" of *Siege* in **mid-2015**. You must empirically verify this date from the data itself:

1. Search `forums_posts` for posts by `author_id == 2170` containing references to "Siege" (case-insensitive) with `new_topic == True` or in threads tagged "siege" / "james mason" / "siegeculture" (cross-reference with `core_search_index_tags`).
2. Search `forums_topics` for topics started by Zeiger (starter_id == 2170) with "siege" in the title.
3. The earliest such post constitutes the **treatment date** (`T0`). Based on external sources this should be approximately mid-2015, but let the data confirm.
4. If you can identify subsequent editions (there are believed to be ~3 total), record those dates as `T1`, `T2` for supplementary analysis.

---

## Hypotheses & Analyses

### Phase 0: Data Ingestion & Preprocessing

1. Clone the repo and ingest all tables listed above into Polars (or pandas) DataFrames.
2. Strip HTML from all post text fields (`post`, `msg_post`). Use `BeautifulSoup` with `html.parser` to extract plain text. Preserve the original HTML column for reference.
3. Verify Zeiger's identity: confirm member_id 2170 corresponds to username containing "Zeiger" in the members table.
4. Produce a **data inventory report**: row counts per table, date ranges for posts and DMs, number of unique active users by year, and Zeiger's posting activity over time.

### Phase 1: Construct the Siegist Rhetoric Measure

This is the most critical methodological step. Implement **two complementary approaches**:

#### Approach A: Dictionary-Based Siegist Score

Build a weighted keyword dictionary capturing Siege Culture rhetoric. The dictionary should include (but is not limited to):

**Core Siege terms:** siege, james mason, universal order, atomwaffen, rape, NSLF (National Socialist Liberation Front), total attack, system, the system, leaderless resistance, lone wolf

**Accelerationist rhetoric:** acceleration, accelerate, collapse, boogaloo, race war, day of the rope, RAHOWA, DOTR, revolution, armed struggle, insurrection

**Mason-specific concepts:** universal order, Charles Manson, Manson, ATWA, helter skelter

**Siege Culture markers:** read siege, siege culture, siegepill, IronPill, skulls, death cult

**Counter-indicators (to distinguish from casual usage):** assign negative weight to posts that use "siege" in clearly non-Mason contexts (e.g., medieval siege, siege of a city).

For each post, compute:
- `siege_keyword_count`: raw count of dictionary hits
- `siege_keyword_density`: hits normalized by post word count
- `siege_binary`: 1 if any dictionary term appears, 0 otherwise

#### Approach B: Embedding-Based Semantic Similarity

1. Construct a **Siege reference corpus** from: (a) all posts by Zeiger (member_id 2170) that contain dictionary terms from Approach A, and (b) all posts in threads tagged "siege", "james mason", or "siegeculture".
2. Compute sentence embeddings for all forum posts and DMs using `sentence-transformers` (model: `all-MiniLM-L6-v2` for speed, or `all-mpnet-base-v2` for quality).
3. Compute a **Siege similarity score** for each post as the cosine similarity between that post's embedding and the centroid of the Siege reference corpus.
4. This provides a continuous measure of how semantically close any given post is to the Siege ideological cluster.

**Output:** A unified post-level DataFrame with columns: `post_id`, `author_id`, `date`, `text`, `channel` ("forum" or "dm"), `siege_keyword_count`, `siege_keyword_density`, `siege_binary`, `siege_similarity`, `word_count`.

### Phase 2: H1 — Interrupted Time Series (Siege Publication as Structural Break)

**Question:** Did Siege Culture rhetoric increase after Zeiger published Siege?

1. Aggregate the siege measures (both dictionary and embedding) to **weekly** time series, separately for:
   - All forum posts (excluding Zeiger's own)
   - All DMs (excluding Zeiger's own)
   - All posts combined
2. Run an **Interrupted Time Series (ITS)** regression for each:

   ```
   Y_t = β0 + β1*time + β2*post_treatment + β3*(time × post_treatment) + ε_t
   ```

   Where:
   - `Y_t` = weekly average siege score (run separately for dictionary and embedding measures)
   - `time` = week index (centered on T0)
   - `post_treatment` = binary indicator (1 after T0)
   - `β2` captures the **level change** at T0
   - `β3` captures the **slope change** (acceleration of adoption)

3. Use Newey-West standard errors to account for autocorrelation. Optionally fit a Bayesian structural time series (CausalImpact-style) as robustness.
4. **Control for confounders:**
   - Include a control for total posting volume (posts per week) to distinguish genuine rhetorical shift from activity changes.
   - If there are other known IM events (e.g., major news, organizational changes), include those as covariates.
5. Run a **Bayesian change-point detection** (e.g., using `ruptures` or a Bayesian approach) on the siege score time series as a model-free complement. Does the detected change point coincide with Zeiger's publication?

**Visualizations:**
- Time series plot with T0 marked, pre/post regression lines, 95% CI bands.
- Cumulative adoption curve (share of unique users who have used ≥1 siege term) over time.
- Zeiger's own posting intensity overlaid on community-wide siege score.

### Phase 3: H2 — Social Contagion in the Interaction Network

**Question:** Does exposure to siegist rhetoric through network ties predict subsequent adoption?

#### Step 1: Build Interaction Networks

Construct **three** user-to-user networks:

1. **DM Network:** Undirected edge between users who share a DM conversation (from `core_message_topic_user_map`). Weight = number of shared conversations.
2. **Forum Co-participation Network:** Undirected edge between users who posted in the same topic (`forums_posts` grouped by `topic_id`). Weight = number of shared topics.
3. **Reputation Network:** Directed edge from liker to liked-user (from `core_reputation_index`, `member_id` → `member_received`). Weight = total likes given.

Store as `networkx` graphs. Also export edge lists for potential use with `graph-tool` or `igraph`.

#### Step 2: Compute Network Exposure

For each user `i` at each time period `t` (monthly), compute:

```
exposure_i_t = (1/|N_i|) * Σ_{j ∈ N_i} siege_score_j_{t-1}
```

Where `N_i` is user `i`'s network neighbors and `siege_score_j_{t-1}` is neighbor `j`'s average siege score in the *preceding* period. This is the **lagged network exposure** variable.

Compute this separately for each of the three networks.

#### Step 3: Contagion Regression

Run panel regressions (user-month level) predicting siege rhetoric adoption:

```
siege_score_i_t = α_i + γ_t + β1*dm_exposure_i_{t-1} + β2*forum_exposure_i_{t-1} + β3*rep_exposure_i_{t-1} + β4*post_treatment_t + Controls + ε_i_t
```

Where:
- `α_i` = user fixed effects (absorbs time-invariant user traits)
- `γ_t` = month fixed effects (absorbs community-wide time trends)
- `dm_exposure`, `forum_exposure`, `rep_exposure` = lagged network exposure from each channel
- Controls: `log(post_count_i_t)`, `user_tenure_i_t`, `reputation_i_t`

Use `linearmodels` (Python) for panel regression with fixed effects, or `statsmodels` with entity dummies for smaller specifications.

**Key tests:**
- Are `β1`, `β2`, `β3` individually significant? Which channel has the strongest contagion effect?
- Is DM exposure (`β1`) larger than forum exposure (`β2`)? (Private channels may be more potent for radicalization.)
- Does contagion strengthen after T0? (Interact exposure terms with `post_treatment`.)

#### Step 4: Robustness — Permutation Test

To rule out spurious correlation from shared community trends:
1. Randomly permute the network edges 1000 times.
2. Re-compute exposure and re-run the contagion regression each time.
3. Compare the observed β coefficients to the permutation distribution. Report permutation p-values.

### Phase 4: H3 — Zeiger as Ideological Entrepreneur (Granger Causality)

**Question:** Does Zeiger's rhetoric *lead* the community's, or does he merely reflect a broader trend?

1. Construct two weekly time series:
   - `zeiger_siege_t`: Zeiger's average siege score in week `t`
   - `community_siege_t`: Community average siege score (excluding Zeiger) in week `t`
2. Run pairwise **Granger causality tests** (using `statsmodels.tsa.stattools.grangercausalitytests`) at lags 1–8 weeks in both directions.
3. If Zeiger Granger-causes community but not vice versa, this supports the ideological entrepreneur interpretation.
4. Visualize with a **cross-correlation function** (CCF) plot.

### Phase 5: H4 — Cohort-Stratified Adoption

**Question:** Do pre-Siege members show a conversion effect, or is Siege adoption driven by self-selected newcomers?

1. Split users into cohorts by join date: (a) pre-Siege joiners (joined before T0), (b) post-Siege joiners (joined after T0).
2. For pre-Siege joiners, compute a **within-user difference-in-differences**: compare each user's siege score before vs. after T0.
3. For post-Siege joiners, compare their siege scores at entry to pre-Siege joiners' scores at entry.
4. Plot cohort-level adoption curves on the same axes.

### Phase 6: H5 — Private-to-Public Pipeline

**Question:** Does siegist rhetoric appear in DMs before public forum posts for the same users?

1. For each user who uses siege rhetoric in *both* DMs and forum posts, identify:
   - `first_dm_siege_date`: First DM containing siege rhetoric
   - `first_forum_siege_date`: First forum post containing siege rhetoric
2. Compute the lag: `dm_lead = first_forum_siege_date - first_dm_siege_date`
3. Test whether the mean lag is significantly positive (DMs precede forum posts) using a one-sample t-test or Wilcoxon signed-rank test.
4. Report the distribution of lags (histogram).

### Phase 7: H6 — Reputation-Mediated Diffusion

**Question:** Do high-status users drive faster adoption?

1. Compute each user's **betweenness centrality** and **degree centrality** in the forum co-participation network.
2. Compute the average siege score of likes received (`core_reputation_index`) — i.e., when a user's siegist post gets liked, does it predict broader adoption?
3. Regress community adoption rate on whether high-centrality vs. low-centrality users have adopted siege rhetoric, using a hazard model (time-to-first-siege-post) with centrality as a covariate.
4. Use `lifelines` (Python) for Cox proportional hazards.

---

## Output Specifications

### File Structure

```
siege_culture_analysis/
├── README.md                     # Project overview and reproduction instructions
├── requirements.txt              # Python dependencies
├── data/
│   ├── raw/                      # Symlink or copy of ironmarch repo data/
│   └── processed/                # Parquet files after preprocessing
│       ├── forum_posts.parquet
│       ├── dm_posts.parquet
│       ├── members.parquet
│       ├── networks/
│       │   ├── dm_edgelist.parquet
│       │   ├── forum_edgelist.parquet
│       │   └── reputation_edgelist.parquet
│       └── siege_scores.parquet  # Unified post-level siege measures
├── src/
│   ├── 00_ingest.py              # Data ingestion from .rda files
│   ├── 01_preprocess.py          # HTML stripping, text cleaning, member reconciliation
│   ├── 02_siege_lexicon.py       # Dictionary-based siege scoring
│   ├── 03_siege_embeddings.py    # Embedding-based siege scoring
│   ├── 04_its_analysis.py        # H1: Interrupted time series
│   ├── 05_network_construction.py # Build interaction networks
│   ├── 06_contagion_model.py     # H2: Network exposure and panel regression
│   ├── 07_granger_causality.py   # H3: Zeiger as ideological entrepreneur
│   ├── 08_cohort_analysis.py     # H4: Cohort-stratified adoption
│   ├── 09_dm_pipeline.py         # H5: Private-to-public pipeline
│   ├── 10_reputation_diffusion.py # H6: Reputation-mediated diffusion
│   └── utils.py                  # Shared utilities (text cleaning, plotting style)
├── figures/                      # All saved figures (PNG + PDF)
└── results/
    ├── its_results.json          # ITS coefficients and p-values
    ├── contagion_results.json    # Panel regression results
    ├── granger_results.json      # Granger causality test results
    └── summary_report.md         # Narrative summary of all findings
```

### Visualization Style

- Use `matplotlib` with `seaborn` style. Set a consistent style at the top of each script.
- All figures should include: descriptive title, axis labels, treatment date(s) marked with vertical dashed lines, and legend where applicable.
- Save each figure as both PNG (300 dpi) and PDF.
- Use a colorblind-friendly palette (e.g., `seaborn.color_palette("colorblind")`).

### Key Dependencies

```
polars >= 0.20
pandas >= 2.0
pyreadr
beautifulsoup4
sentence-transformers
torch
scikit-learn
statsmodels
linearmodels
networkx
ruptures
lifelines
matplotlib
seaborn
```

---

## Execution Order

Run scripts in numeric order (`00` through `10`). Each script should:
1. Load only the data it needs from `data/processed/`.
2. Print progress messages to stdout.
3. Save results to `results/` and figures to `figures/`.
4. Be idempotent (safe to re-run).

The `summary_report.md` in `results/` should be generated by a final aggregation step that reads all JSON result files and produces a narrative interpretation of findings across all hypotheses.

---

## Critical Methodological Notes

1. **Endogeneity warning:** Network contagion models are notoriously susceptible to homophily confounds (users who interact may share unobserved traits that independently predict siege adoption). The user fixed effects partially address this, but the permutation test (Phase 3, Step 4) is essential. Flag this limitation prominently in the summary report.

2. **Multiple comparisons:** You are running many tests. Apply Benjamini-Hochberg correction across all primary hypothesis tests and report both raw and adjusted p-values.

3. **Small-N caution for DMs:** The DM corpus (21.7K messages) is much smaller than the forum corpus (195K posts). The DM contagion channel may lack statistical power. Report power analyses where feasible.

4. **HTML artifacts:** Forum posts contain HTML markup including blockquotes (replies), embedded images, and emoticons. The text cleaning step must handle nested quotes carefully — you may want to distinguish original text from quoted text in posts, as quoted text reflects the *quoted* user's rhetoric, not the posting user's.

5. **Zeiger's multiple identities:** Check the `core_member_history` table for display name changes. Zeiger may have used different names over time. Also check if he appears in both `core_members` and `orig_members`.

6. **The "Siege" term is overloaded.** The word "siege" can appear in non-Mason contexts (historical sieges, the video game Rainbow Six Siege, etc.). Your dictionary approach must account for this. The embedding approach provides a natural robustness check since semantically dissimilar uses will not cluster near the Siege reference corpus.
