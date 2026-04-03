# Siege Culture Diffusion in the Iron March Dataset

Analysis pipeline testing whether the publication of the "Iron March edition" of James Mason's *Siege* by user Zeiger (member_id 2170) catalyzed the development of Siege Culture within the Iron March forum community.

## Research Overview

This project implements a multi-method quantitative analysis of ideological diffusion on the Iron March (IM) neo-fascist forum (2011–2017), using the leaked MySQL database. The pipeline tests six hypotheses:

| # | Hypothesis | Method |
|---|-----------|--------|
| H1 | Siege publication caused a structural break in rhetoric | Interrupted Time Series |
| H2 | Network exposure predicts siege rhetoric adoption | Panel regression with FE |
| H3 | Zeiger's rhetoric leads the community's | Granger causality |
| H4 | Pre-Siege members show conversion effects | Cohort difference-in-differences |
| H5 | Siege rhetoric appears in DMs before forums | Private-to-public pipeline |
| H6 | High-status users drive faster adoption | Cox proportional hazards |

## Setup

### Requirements

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) for environment management
- R (for data export from `.rda` files)

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

Scripts are numbered and should be run in order. Each script is idempotent.

```bash
# Run the full pipeline end-to-end
uv run python main.py

# Resume from a specific stage (e.g. after fixing an error)
uv run python main.py --from 4

# Run only specific stages
uv run python main.py --only 2 3
```

Individual scripts can also be run directly:

```bash
cd src/

# Phase 0: Data ingestion and preprocessing
uv run python 00_ingest.py
uv run python 01_preprocess.py

# Phase 1: Siege scoring
uv run python 02_siege_lexicon.py
uv run python 03_siege_embeddings.py     # Requires GPU/MPS or patience

# Phase 2+: Analysis
uv run python 04_its_analysis.py         # H1: Interrupted time series
uv run python 05_network_construction.py # Build interaction networks
uv run python 06_contagion_model.py      # H2: Social contagion
uv run python 07_granger_causality.py    # H3: Granger causality
uv run python 08_cohort_analysis.py      # H4: Cohort analysis
uv run python 09_dm_pipeline.py          # H5: DM pipeline
uv run python 10_reputation_diffusion.py # H6: Reputation diffusion

# Generate summary report
uv run python 11_summary_report.py
```

## Running Tests

```bash
uv run pytest tests/ -v
```

## Project Structure

```
siege_culture_analysis/
├── README.md
├── pyproject.toml
├── export_rda.R              # R helper to export .rda → CSV
├── data/
│   ├── raw/                  # ironmarch repo + CSV exports
│   └── processed/            # Parquet files after preprocessing
│       └── networks/         # Edge lists
├── src/
│   ├── utils.py              # Shared utilities and constants
│   ├── 00_ingest.py          # Data ingestion
│   ├── 01_preprocess.py      # HTML stripping, member reconciliation
│   ├── 02_siege_lexicon.py   # Dictionary-based siege scoring
│   ├── 03_siege_embeddings.py # Embedding-based scoring
│   ├── 04_its_analysis.py    # H1: Interrupted time series
│   ├── 05_network_construction.py # Build networks
│   ├── 06_contagion_model.py # H2: Network contagion
│   ├── 07_granger_causality.py # H3: Granger causality
│   ├── 08_cohort_analysis.py # H4: Cohort analysis
│   ├── 09_dm_pipeline.py     # H5: Private-to-public
│   ├── 10_reputation_diffusion.py # H6: Reputation diffusion
│   └── 11_summary_report.py  # Report generation
├── tests/                    # Unit tests
├── figures/                  # PNG + PDF plots
└── results/                  # JSON results + summary report
```

## Data Source

The data comes from the [ironmarch R package](https://github.com/knapply/ironmarch) containing the leaked Iron March database. Key tables:

- **forums_posts** (195K posts): Public forum discussion
- **core_message_posts** (21.7K messages): Private DMs
- **core_members** (1.2K members): User profiles
- **core_reputation_index** (272K events): Likes/reputation

## Methodological Notes

1. **Endogeneity:** Network contagion models are susceptible to homophily confounds. Permutation tests provide robustness checks.
2. **Multiple comparisons:** Benjamini-Hochberg correction should be applied to primary tests.
3. **DM corpus size:** The smaller DM corpus may lack statistical power for some analyses.
4. **"Siege" overloading:** The embedding approach provides robustness against non-Mason uses of the word.

## License

Research use only. The Iron March data is a public interest leak of a now-defunct website.
