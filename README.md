# Siege Culture Diffusion in the Iron March Dataset & 4chan /pol/

Analysis pipeline testing whether the publication of the "Iron March edition" of James Mason's *Siege* by user Zeiger (member_id 2170) catalyzed the development of Siege Culture within the Iron March forum community, and whether that rhetoric subsequently diffused to 4chan's /pol/ board.

## Research Overview

This project implements a multi-method quantitative analysis of ideological diffusion on the Iron March (IM) neo-fascist forum (2011–2017) and 4chan's /pol/ board, using the leaked Iron March MySQL database and the 4plebs /pol/ archive (Asagi schema). The pipeline tests fourteen hypotheses spanning macro-temporal trends, network contagion, collective exegesis, spatial diffusion, and cross-platform contagion, plus an apocalypticism chapter examining post-attack rhetoric:

| # | Hypothesis | Method | Key Finding |
|---|-----------|--------|-------------|
| H1 | Siege publication caused a structural break in rhetoric | Interrupted Time Series | β₃ significant (p<0.001\*\*\*), R²=0.51 |
| H2 | Network exposure predicts siege rhetoric adoption | Panel regression with FE | Forum exposure β=1.13 (p<0.001\*\*\*) |
| H3 | Zeiger's rhetoric Granger-causes the community's | Granger causality | Not significant (F=1.95, p=0.17) |
| H4 | Pre-Siege members show conversion effects | Cohort difference-in-differences | t=2.26 (p=0.025\*) |
| H5 | Siege rhetoric appears in forums before DMs | Forum-to-DM temporal ordering | 83.7% forum-first (p<0.001\*\*\*) |
| H6 | Network-central users drive faster adoption | Cox proportional hazards | Degree HR=2.87 (p<0.001\*\*\*) |
| H7 | Community rewards Siege-aligned speech | Negative binomial on reputation | IRR=1.54 (p<0.001\*\*\*) |
| H8 | Siege rhetoric escalates within threads | Within-thread position regression | Not supported linearly; catalyst pattern |
| H9 | Thread exposure predicts subsequent adoption | Lagged panel regression | β=0.14 (p=0.087†), clear dose-response |
| H10 | Community develops shared Siege interpretation | Inter-user CV convergence (ITS) | Converging slope β₃=-0.054 (p=0.011\*) |
| H11 | Siege rhetoric concentrates in ideological subforums | Herfindahl index, subforum mapping | Post-Siege HHI increases; books, strategy lead |

### Cross-Platform Hypotheses (H12–H14)

| # | Hypothesis | Method |
|---|-----------|--------|
| H12 | Siege rhetoric shows a structural break on /pol/ after IM's T0 | Cross-platform ITS with prevalence correction |
| H13 | Iron March Siege rhetoric Granger-causes /pol/ Siege rhetoric | Bidirectional Granger tests with stationarity checks |
| H14 | Content bridges (URLs, phrases) propagate from IM → /pol/ | URL overlap, n-gram fingerprinting, temporal priority |

Additionally, H8 (thread escalation) and H10 (semantic convergence) are adapted for /pol/'s anonymous structure.

### Apocalypticism Chapter

A separate analysis chapter tests whether mass-casualty violent events trigger apocalyptic rhetoric on /pol/. Uses a transformer-based classifier (logistic regression + multi-facet contrastive similarity on `all-MiniLM-L6-v2` embeddings) to identify five sub-themes of apocalypticism: racial apocalypse, eschatological, civilizational collapse, accelerationist, and conspiratorial.

| Analysis | Method | Key Finding |
|----------|--------|-------------|
| Event catalogue | 136 events (110 mass-violence + 26 non-violence controls) across 2010–2023 | 23 countries, 6 ideology categories |
| Classifier | LR (0.6) + contrastive similarity (0.4), threshold 0.55 | 3.71% binary classification rate |
| Per-event ITS | Newey-West HAC regression per event | 34/86 significant at α=0.05 |
| Pooled ITS | Stacked ITS with event FE | β₂ = −0.0018 (p=0.043) — slight *decrease* |
| Category comparison | Violence vs non-violence falsification | Mass-violence β₂=−0.0025; economic-shock β₂=+0.016 |
| Severity correlations | Pearson/Spearman of β₂ vs casualties | No significant correlation |
| Ideology comparison | Kruskal-Wallis across ideology groups | H=3.22 (p=0.67), no significant differences |
| Domestic vs intl | Mann-Whitney comparison | p=0.19, no significant difference |
| Multiple regression | OLS of β₂ on log(killed), domestic, nexus, ideology | R²=0.14; domestic (p=0.01\*) is only significant predictor |
| Robustness | Placebo, bandwidth, dose–response, lag, DoW, BH FDR, AR(1) | 2/5 checks passed |

### Theoretical Framework

Hypotheses H1–H6 test baseline diffusion and contagion dynamics. Hypotheses H7–H11 operationalise a **collective exegesis** theory: Iron March collectively "anoints" *Siege* as a sacred text, performing group interpretation that produces a shared worldview. Evidence includes:
- **Canonisation** (H7): Siege posts receive 54% more reputation points
- **Exegetical spaces** (H11): Rhetoric concentrates in books, strategy, and archival subforums
- **Convergence** (H10): Inter-user variability in Siege scores *decreases* over time after an initial burst
- **Thread-level contagion** (H9): Participating in high-Siege threads predicts future adoption

## Measurement

The pipeline uses a **hybrid dictionary + embedding** approach to score each post for Siege-related content:

1. **Dictionary scoring** (`02_siege_lexicon.py`): A 52-term, three-tier weighted lexicon covering unambiguous Siege markers (e.g., "james mason", "read siege", "atomwaffen"), Siege-adjacent ideology (e.g., "accelerationism", "leaderless resistance"), and context-dependent terms (e.g., "the system", "collapse") with reduced weights. Includes counter-indicators (e.g., "Total War" as a video game) to reduce false positives.

2. **Embedding scoring** (`03_siege_embeddings.py`): Cosine similarity of each post's `all-MiniLM-L6-v2` embedding to a Siege reference corpus. An **embedding boost** adjusts context-dependent keyword scores based on semantic similarity — amplifying matches in ideologically relevant contexts and attenuating false positives.

## Setup

### Requirements

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) for environment management
- R (for data export from `.rda` files)
- CUDA-capable GPU recommended (embedding computation)

### Installation

```bash
# Clone and install
git clone https://github.com/alexbnewhouse/Siege-Contagion.git
cd Siege-Contagion
uv sync --all-extras

# Clone the Iron March data
git clone --depth 1 https://github.com/knapply/ironmarch.git data/raw/ironmarch

# Export R data to CSV
Rscript export_rda.R
```

## Running the Pipeline

The pipeline has 17 Iron March stages plus 9 /pol/ and cross-platform stages plus 5 apocalypticism stages, orchestrated by `main.py`. Each stage is idempotent.

```bash
# Run the Iron March pipeline (default)
uv run python main.py

# Run the /pol/ pipeline
uv run python main.py --platform pol

# Run everything: IM + /pol/ + cross-platform analyses
uv run python main.py --platform both

# Run the apocalypticism chapter
uv run python main.py --platform apoc

# Resume from a specific stage (e.g. after fixing an error)
uv run python main.py --from 4

# Run only specific stages
uv run python main.py --only 2 3
```

### Iron March Pipeline Stages

| Stage | Script | Description |
|-------|--------|-------------|
| 00 | `00_ingest.py` | Data ingestion (CSV → Parquet) |
| 01 | `01_preprocess.py` | HTML stripping, member reconciliation, treatment dates |
| 02 | `02_siege_lexicon.py` | Dictionary-based siege scoring (52-term lexicon) |
| 03 | `03_siege_embeddings.py` | Embedding-based scoring + embedding boost |
| 04 | `04_its_analysis.py` | H1: Interrupted time series |
| 05 | `05_network_construction.py` | Build interaction networks (forum, DM, reputation) |
| 06 | `06_contagion_model.py` | H2: Social contagion panel model |
| 07 | `07_granger_causality.py` | H3: Granger causality (Zeiger ↔ community) |
| 08 | `08_cohort_analysis.py` | H4: Cohort-stratified adoption |
| 09 | `09_dm_pipeline.py` | H5: Forum-to-DM temporal ordering |
| 10 | `10_reputation_diffusion.py` | H6: Cox PH survival analysis |
| 12 | `12_reputation_reinforcement.py` | H7: Reputation reinforcement of Siege posts |
| 13 | `13_thread_escalation.py` | H8: Within-thread escalation |
| 14 | `14_thread_exposure.py` | H9: Thread exposure → subsequent adoption |
| 15 | `15_semantic_convergence.py` | H10: Semantic convergence (ITS on CV) |
| 16 | `16_subforum_diffusion.py` | H11: Subforum diffusion geography |
| 11 | `11_summary_report.py` | Summary report generation |

### /pol/ Platform Stages (`--platform pol`)

| Stage | Script | Description |
|-------|--------|-------------|
| 00b | `00b_ingest_pol.py` | Streaming ingest from 4plebs tar.gz with Siege pre-filter |
| 01b | `01b_preprocess_pol.py` | 4chan HTML cleaning, schema normalisation |
| 02b | `02b_siege_lexicon_pol.py` | Dictionary-based siege scoring for /pol/ |
| 03b | `03b_siege_embeddings_pol.py` | Embedding scoring using IM's centroid |
| 20  | `20_pol_thread_escalation.py` | H8-pol: Within-thread escalation on /pol/ |
| 21  | `21_pol_semantic_convergence.py` | H10-pol: Semantic convergence on /pol/ |

### Cross-Platform Stages (included in `--platform both`)

| Stage | Script | Description |
|-------|--------|-------------|
| 17 | `17_cross_platform_its.py` | H12: Cross-platform ITS |
| 18 | `18_cross_platform_granger.py` | H13: Cross-platform Granger causality |
| 19 | `19_cross_platform_bridges.py` | H14: Content bridge detection |

### Apocalypticism Chapter Stages (`--platform apoc`)

| Stage | Script | Description |
|-------|--------|-------------|
| 29 | `29_mass_casualty_events.py` | Mass-casualty & discontinuity event dataset (136 events) |
| 30 | `30_pol_apocalypticism.py` | Transformer-based apocalypticism classifier (LR + contrastive) |
| 31 | `31_apocalypticism_its.py` | Per-event & pooled ITS, stratified by ideology & event category |
| 32 | `32_apocalypticism_robustness.py` | 8-test robustness battery (placebo, bandwidth, dose–response, etc.) |
| 33 | `33_attack_characteristic_correlations.py` | Severity, ideology, geography & multiple regression on β₂ |

## Running Tests

```bash
uv run pytest tests/ -v
```

262 tests covering lexicon scoring, embedding boost, preprocessing, network construction, contagion model, report generation, all five exegesis-theory modules, /pol/ ingest/preprocessing, cross-platform analysis modules, and the full apocalypticism pipeline (event validation, transformer classifier, ITS regression, robustness checks, and attack-characteristic correlations).

## Project Structure

```
Siege-Contagion/
├── README.md
├── pyproject.toml
├── main.py                   # Pipeline orchestrator (--platform im|pol|both)
├── export_rda.R              # R helper to export .rda → CSV
├── siege_culture_agent_prompt.md  # Research design document
├── data/
│   ├── raw/                  # ironmarch repo + CSV exports
│   ├── pol/                  # 4plebs /pol/ archive (pol.csv.tar.gz)
│   └── processed/            # Parquet files after preprocessing
│       └── networks/         # Edge lists (forum, DM, reputation)
├── src/
│   ├── utils.py              # Shared utilities and constants
│   ├── 00_ingest.py          # IM data ingestion
│   ├── 00b_ingest_pol.py     # /pol/ streaming ingest with Siege pre-filter
│   ├── 01_preprocess.py      # IM HTML stripping, member reconciliation
│   ├── 01b_preprocess_pol.py # 4chan HTML cleaning, schema normalisation
│   ├── 02_siege_lexicon.py   # Dictionary-based siege scoring (52-term lexicon)
│   ├── 02b_siege_lexicon_pol.py # /pol/ lexicon scoring
│   ├── 03_siege_embeddings.py # Embedding-based scoring + boost
│   ├── 03b_siege_embeddings_pol.py # /pol/ embedding scoring (shared centroid)
│   ├── 04_its_analysis.py    # H1: Interrupted time series
│   ├── 05_network_construction.py # Build networks
│   ├── 06_contagion_model.py # H2: Network contagion
│   ├── 07_granger_causality.py # H3: Granger causality
│   ├── 08_cohort_analysis.py # H4: Cohort analysis
│   ├── 09_dm_pipeline.py     # H5: Forum-to-DM pipeline
│   ├── 10_reputation_diffusion.py # H6: Reputation diffusion
│   ├── 11_summary_report.py  # Report generation
│   ├── 12_reputation_reinforcement.py # H7: Reputation reinforcement
│   ├── 13_thread_escalation.py # H8: Thread escalation
│   ├── 14_thread_exposure.py # H9: Thread exposure → adoption
│   ├── 15_semantic_convergence.py # H10: Semantic convergence
│   ├── 16_subforum_diffusion.py # H11: Subforum diffusion
│   ├── 17_cross_platform_its.py # H12: Cross-platform ITS
│   ├── 18_cross_platform_granger.py # H13: Cross-platform Granger
│   ├── 19_cross_platform_bridges.py # H14: Content bridges
│   ├── 20_pol_thread_escalation.py # H8-pol: /pol/ thread escalation
│   ├── 21_pol_semantic_convergence.py # H10-pol: /pol/ convergence
│   ├── 29_mass_casualty_events.py # Apocalypticism: event dataset
│   ├── 30_pol_apocalypticism.py   # Apocalypticism: LR+contrastive classifier
│   ├── 31_apocalypticism_its.py   # Apocalypticism: ITS analysis
│   ├── 32_apocalypticism_robustness.py # Apocalypticism: robustness battery
│   └── 33_attack_characteristic_correlations.py # Apocalypticism: correlations
├── tests/                    # 262 unit tests
│   ├── conftest.py
│   ├── test_apocalypticism.py # 88 apocalypticism pipeline tests
│   ├── test_contagion.py
│   ├── test_exegesis.py      # H7–H11 tests
│   ├── test_ingest.py
│   ├── test_lexicon.py       # 34 lexicon tests
│   ├── test_networks.py
│   ├── test_pol.py           # 53 /pol/ + cross-platform tests
│   ├── test_preprocess.py
│   ├── test_report.py
│   └── test_utils.py
├── figures/                  # PNG + PDF plots
└── results/                  # JSON results + summary report
```

## Data Sources

### Iron March
The data comes from the [ironmarch R package](https://github.com/knapply/ironmarch) containing the leaked Iron March database. Key tables:

- **forums_posts** (195K posts): Public forum discussion
- **forums_topics** (7.2K threads): Thread metadata with titles, subforum IDs
- **forums_forums** (144 subforums): Board structure and hierarchy
- **core_message_posts** (21.7K messages): Private DMs
- **core_members** (1.2K members): User profiles with reputation points
- **core_reputation_index** (272K events): Likes/reputation interactions

### 4chan /pol/
The /pol/ data comes from the [4plebs](https://archive.4plebs.org/) archive, stored in the standard Asagi MySQL dump format (Hine et al. 2017; Papasavva et al. 2020). The archive is a ~28GB `pol.csv.tar.gz` containing a single headerless CSV with 28 columns following the Asagi schema.

Place the archive at `data/pol/pol.csv.tar.gz`. The streaming ingest pipeline (`00b_ingest_pol.py`) applies a lightweight Siege pre-filter during extraction, retaining only the ~0.1% of posts matching Siege-related terms while collecting weekly aggregate counts for prevalence normalization.

## Methodological Notes

1. **Endogeneity:** Network contagion models are susceptible to homophily confounds. Permutation tests provide robustness checks.
2. **Multiple comparisons:** Benjamini-Hochberg correction should be applied across primary tests.
3. **DM corpus size:** The smaller DM corpus (~21.7K vs ~195K forum posts) may lack statistical power.
4. **"Siege" overloading:** The embedding boost approach provides robustness against non-Mason uses of the word.
5. **Reputation data:** Platform reputation/like data has not been independently validated. H7 results should be interpreted cautiously.

## License

Research use only. The Iron March data is a public interest leak of a now-defunct website.
