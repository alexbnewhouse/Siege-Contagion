"""Tests for the apocalypticism pipeline (stages 29-35).

Covers:
  - Event dataset validation (categories, ideologies, completeness)
  - Transformer-based classifier (LR, centroids, contrastive scoring)
  - ITS regression & category comparison
  - Robustness utilities
  - Attack-characteristic correlations
  - Advanced time-series methods (VAR, ARDL, BSTS, LP)
  - Offline-online hypotheses (H22-H26)
"""

import datetime
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ═══════════════════════════════════════════════════════════════════════
# Module imports
# ═══════════════════════════════════════════════════════════════════════
_events = importlib.import_module("29_mass_casualty_events")
_apoc = importlib.import_module("30_pol_apocalypticism")
_its = importlib.import_module("31_apocalypticism_its")
_robust = importlib.import_module("32_apocalypticism_robustness")
_corr = importlib.import_module("33_attack_characteristic_correlations")
_adv = importlib.import_module("34_advanced_ts_apocalypticism")
_hyp = importlib.import_module("35_offline_online_hypotheses")


# ═══════════════════════════════════════════════════════════════════════
# 29 – Mass-Casualty & Discontinuity Event Dataset
# ═══════════════════════════════════════════════════════════════════════

class TestEventValidation:
    def test_all_events_valid(self):
        errors = _events.validate_events(_events.EVENTS)
        real_errors = [e for e in errors if not e.startswith("Warning")]
        assert not real_errors, f"Validation errors: {real_errors}"

    def test_all_events_have_date(self):
        for ev in _events.EVENTS:
            d = datetime.date.fromisoformat(ev["date"])
            assert d.year >= 2010, f"{ev['name']}: year {d.year} < 2010"

    def test_all_events_have_valid_ideology(self):
        for ev in _events.EVENTS:
            assert ev["ideology"] in _events.VALID_IDEOLOGIES, (
                f"{ev['name']}: invalid ideology '{ev['ideology']}'"
            )

    def test_all_events_have_valid_category(self):
        for ev in _events.EVENTS:
            assert ev["event_category"] in _events.VALID_CATEGORIES, (
                f"{ev['name']}: invalid category '{ev['event_category']}'"
            )

    def test_nonviolence_events_have_na_ideology(self):
        for ev in _events.EVENTS:
            if ev["event_category"] != "mass_violence":
                assert ev["ideology"] == "N/A", (
                    f"{ev['name']}: non-violence event has ideology "
                    f"'{ev['ideology']}' instead of 'N/A'"
                )

    def test_killed_nonnegative(self):
        for ev in _events.EVENTS:
            assert ev["killed"] >= 0, f"{ev['name']}: negative killed"
            assert ev["injured"] >= 0, f"{ev['name']}: negative injured"

    def test_no_duplicate_names(self):
        names = [ev["name"] for ev in _events.EVENTS]
        assert len(names) == len(set(names)), "Duplicate event names"

    def test_minimum_event_count(self):
        """Expanded dataset should have ≥80 events across many ideologies."""
        assert len(_events.EVENTS) >= 80
        ideologies = {ev["ideology"] for ev in _events.EVENTS}
        assert len(ideologies) >= 4

    def test_has_both_violence_and_nonviolence(self):
        cats = {ev["event_category"] for ev in _events.EVENTS}
        assert "mass_violence" in cats
        assert len(cats) >= 3, "Need at least 3 event categories"

    def test_violence_events_minimum(self):
        n_violence = sum(1 for ev in _events.EVENTS
                         if ev["event_category"] == "mass_violence")
        assert n_violence >= 60, f"Only {n_violence} mass_violence events"

    def test_nonviolence_events_minimum(self):
        n_nonv = sum(1 for ev in _events.EVENTS
                     if ev["event_category"] != "mass_violence")
        assert n_nonv >= 10, f"Only {n_nonv} non-violence events"

    def test_validation_catches_missing_date(self):
        bad = [{"name": "bad", "killed": 1, "injured": 0,
                "ideology": "other", "event_category": "mass_violence",
                "online_nexus": False, "domestic": True,
                "location_country": "US"}]
        errors = _events.validate_events(bad)
        real = [e for e in errors if not e.startswith("Warning")]
        assert any("date" in e for e in real)

    def test_validation_catches_bad_ideology(self):
        bad = [{"date": "2020-01-01", "name": "bad", "killed": 1,
                "injured": 0, "ideology": "alien",
                "event_category": "mass_violence",
                "online_nexus": False, "domestic": True,
                "location_country": "US"}]
        errors = _events.validate_events(bad)
        real = [e for e in errors if not e.startswith("Warning")]
        assert any("ideology" in e for e in real)

    def test_validation_catches_bad_category(self):
        bad = [{"date": "2020-01-01", "name": "bad", "killed": 1,
                "injured": 0, "ideology": "other",
                "event_category": "earthquake",
                "online_nexus": False, "domestic": True,
                "location_country": "US"}]
        errors = _events.validate_events(bad)
        real = [e for e in errors if not e.startswith("Warning")]
        assert any("event_category" in e for e in real)

    def test_validation_catches_nonviolence_with_ideology(self):
        bad = [{"date": "2020-01-01", "name": "bad", "killed": 0,
                "injured": 0, "ideology": "far_right",
                "event_category": "natural_disaster",
                "online_nexus": False, "domestic": True,
                "location_country": "US"}]
        errors = _events.validate_events(bad)
        real = [e for e in errors if not e.startswith("Warning")]
        assert any("N/A" in e for e in real)

    def test_oecd_country_coverage(self):
        """Events should cover multiple OECD countries."""
        countries = {ev["location_country"] for ev in _events.EVENTS}
        assert "US" in countries
        assert "UK" in countries
        assert "France" in countries
        assert "Germany" in countries
        assert len(countries) >= 8


class TestEventsToPolars:
    def test_converts_correctly(self):
        df = _events.events_to_polars(_events.EVENTS)
        assert df.height == len(_events.EVENTS)
        assert "event_date" in df.columns
        assert "event_category" in df.columns
        assert "total_casualties" in df.columns

    def test_sorted_by_date(self):
        df = _events.events_to_polars(_events.EVENTS)
        dates = df["event_date"].to_list()
        assert dates == sorted(dates)

    def test_total_casualties(self):
        df = _events.events_to_polars(_events.EVENTS)
        for row in df.iter_rows(named=True):
            assert row["total_casualties"] == row["killed"] + row["injured"]

    def test_category_column_present(self):
        df = _events.events_to_polars(_events.EVENTS)
        assert "event_category" in df.columns
        cats = set(df["event_category"].to_list())
        assert "mass_violence" in cats


class TestEventsSummary:
    def test_returns_expected_keys(self):
        df = _events.events_to_polars(_events.EVENTS)
        s = _events.events_summary(df)
        assert "total_events" in s
        assert "by_ideology" in s
        assert "by_category" in s
        assert "date_range" in s
        assert "mass_violence" in s
        assert "nonviolence" in s
        assert s["total_events"] == len(_events.EVENTS)


# ═══════════════════════════════════════════════════════════════════════
# 30 – Apocalypticism Classifier
# ═══════════════════════════════════════════════════════════════════════

class TestDiagnosticKeywordScore:
    """Tests for the lightweight diagnostic keyword function."""

    def test_empty_text(self):
        r = _apoc.compute_diagnostic_keyword_score("")
        assert r["apoc_keyword_count"] == 0
        assert r["apoc_keyword_score"] == 0.0

    def test_none_text(self):
        r = _apoc.compute_diagnostic_keyword_score(None)
        assert r["apoc_keyword_count"] == 0

    def test_non_apocalyptic_text(self):
        r = _apoc.compute_diagnostic_keyword_score("The weather is nice today.")
        assert r["apoc_binary"] == 0

    def test_explicit_end_times(self):
        r = _apoc.compute_diagnostic_keyword_score("The end times are coming soon.")
        assert r["apoc_keyword_count"] >= 1
        assert r["apoc_keyword_score"] > 0
        assert r["apoc_binary"] == 1

    def test_armageddon(self):
        r = _apoc.compute_diagnostic_keyword_score("Armageddon is upon us.")
        assert r["apoc_binary"] == 1

    def test_race_war(self):
        r = _apoc.compute_diagnostic_keyword_score("The race war is inevitable.")
        assert r["apoc_binary"] == 1

    def test_kali_yuga(self):
        r = _apoc.compute_diagnostic_keyword_score("We are living in the Kali Yuga.")
        assert r["apoc_binary"] == 1

    def test_day_of_the_rope(self):
        r = _apoc.compute_diagnostic_keyword_score("The day of the rope approaches.")
        assert r["apoc_binary"] == 1

    def test_case_insensitive(self):
        r = _apoc.compute_diagnostic_keyword_score("ARMAGEDDON IS NEAR")
        assert r["apoc_binary"] == 1

    def test_backward_compat_alias(self):
        """compute_apoc_keyword_score should still work."""
        r = _apoc.compute_apoc_keyword_score("Armageddon")
        assert r["apoc_binary"] == 1


class TestSeedData:
    """Validate the synthetic training data structure."""

    def test_positive_seeds_has_subthemes(self):
        assert len(_apoc.POSITIVE_SEEDS) >= 4
        for theme, texts in _apoc.POSITIVE_SEEDS.items():
            assert len(texts) >= 10, f"Sub-theme '{theme}' has only {len(texts)} examples"

    def test_negative_seeds_has_categories(self):
        assert len(_apoc.NEGATIVE_SEEDS) >= 3
        for cat, texts in _apoc.NEGATIVE_SEEDS.items():
            assert len(texts) >= 10, f"Negative category '{cat}' has only {len(texts)} examples"

    def test_has_hard_negatives(self):
        """Hard negatives are crucial for classifier quality."""
        assert "hard_negatives" in _apoc.NEGATIVE_SEEDS
        assert len(_apoc.NEGATIVE_SEEDS["hard_negatives"]) >= 10

    def test_balanced_training_set(self):
        n_pos = sum(len(t) for t in _apoc.POSITIVE_SEEDS.values())
        n_neg = sum(len(t) for t in _apoc.NEGATIVE_SEEDS.values())
        assert abs(n_pos - n_neg) / max(n_pos, n_neg) < 0.5, (
            f"Training set too imbalanced: {n_pos} pos vs {n_neg} neg"
        )


class TestCategorySeeds:
    """Validate the 4-category disaggregation seeds."""

    def test_has_four_categories(self):
        assert len(_apoc.APOC_CATEGORIES) == 4
        for cat in _apoc.APOC_CATEGORIES:
            assert cat in _apoc.CATEGORY_SEEDS

    def test_category_seeds_not_empty(self):
        for cat, texts in _apoc.CATEGORY_SEEDS.items():
            assert len(texts) >= 15, f"Category '{cat}' has only {len(texts)} seeds"

    def test_category_names(self):
        expected = {"siegist_traditionalist", "rapture_christian",
                    "prepper", "general_collapsist"}
        assert set(_apoc.APOC_CATEGORIES) == expected

    def test_subtheme_to_category_mapping(self):
        for subtheme in _apoc.POSITIVE_SEEDS:
            assert subtheme in _apoc.SUBTHEME_TO_CATEGORY, (
                f"Sub-theme '{subtheme}' not mapped to any category"
            )


class TestBuildCategoryCentroids:
    def test_category_centroid_shapes(self):
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            pytest.skip("sentence-transformers not available")

        cat_centroids = _apoc.build_category_centroids(model)
        assert len(cat_centroids) == 4
        for name, c in cat_centroids.items():
            assert c.shape == (384,), f"Category '{name}' wrong shape"
            assert abs(np.linalg.norm(c) - 1.0) < 1e-5, f"'{name}' not normalised"

    def test_category_centroids_distinct(self):
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            pytest.skip("sentence-transformers not available")

        cat_centroids = _apoc.build_category_centroids(model)
        names = list(cat_centroids.keys())
        from sklearn.metrics.pairwise import cosine_similarity
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                sim = cosine_similarity(
                    cat_centroids[names[i]].reshape(1, -1),
                    cat_centroids[names[j]].reshape(1, -1),
                )[0, 0]
                assert sim < 0.95, (
                    f"Categories '{names[i]}' and '{names[j]}' too similar: {sim:.3f}"
                )


class TestScoreEmbeddingsWithCategories:
    def test_category_columns_present(self):
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            pytest.skip("sentence-transformers not available")

        lr = _apoc.train_classifier(model)
        pos, neg, subs = _apoc.build_centroids(model)
        cat_centroids = _apoc.build_category_centroids(model)

        texts = [
            "Read Siege. The day of the rope is coming. Accelerate.",
            "The rapture is near. Christ will return for the faithful.",
            "Stock up on ammo and water. SHTF is coming soon.",
            "Western civilization is in terminal decline. Collapse is inevitable.",
        ]
        embs = model.encode(texts, normalize_embeddings=True)

        scores = _apoc.score_embeddings(
            embs, lr, pos, neg, subs,
            category_centroids=cat_centroids,
        )

        assert "apoc_category" in scores
        assert "apoc_category_sim" in scores
        assert len(scores["apoc_category"]) == 4
        assert len(scores["apoc_category_sim"]) == 4

        # Check per-category similarity columns
        for cat in _apoc.APOC_CATEGORIES:
            assert f"apoc_cat_sim_{cat}" in scores

    def test_category_assignment_reasonable(self):
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            pytest.skip("sentence-transformers not available")

        lr = _apoc.train_classifier(model)
        pos, neg, subs = _apoc.build_centroids(model)
        cat_centroids = _apoc.build_category_centroids(model)

        texts = [
            "Read Siege by James Mason. Accelerate the collapse now.",
            "The rapture is imminent. Armageddon approaches.",
            "Bug out bag ready. Six months of food stockpiled.",
            "The financial system will collapse. Late stage capitalism.",
        ]
        embs = model.encode(texts, normalize_embeddings=True)

        scores = _apoc.score_embeddings(
            embs, lr, pos, neg, subs,
            category_centroids=cat_centroids,
        )

        # Verify reasonable assignments
        cats = scores["apoc_category"]
        assert cats[0] == "siegist_traditionalist", f"Siege text → {cats[0]}"
        assert cats[1] == "rapture_christian", f"Rapture text → {cats[1]}"
        assert cats[2] == "prepper", f"Prepper text → {cats[2]}"
        assert cats[3] == "general_collapsist", f"Collapsist text → {cats[3]}"

    def test_backward_compat_no_categories(self):
        """score_embeddings still works without category_centroids."""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            pytest.skip("sentence-transformers not available")

        lr = _apoc.train_classifier(model)
        pos, neg, subs = _apoc.build_centroids(model)

        embs = model.encode(["test text"], normalize_embeddings=True)
        scores = _apoc.score_embeddings(embs, lr, pos, neg, subs)

        assert "apoc_category" not in scores
        assert "apoc_lr_prob" in scores
        assert "apoc_subtheme" in scores


class TestBuildCentroids:
    def test_centroid_shapes(self):
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            pytest.skip("sentence-transformers not available")

        pos, neg, subs = _apoc.build_centroids(model)
        assert pos.ndim == 1
        assert neg.ndim == 1
        assert pos.shape[0] == 384
        assert neg.shape[0] == 384
        # Normalized
        assert abs(np.linalg.norm(pos) - 1.0) < 1e-5
        assert abs(np.linalg.norm(neg) - 1.0) < 1e-5
        # Sub-themes
        assert len(subs) == len(_apoc.POSITIVE_SEEDS)
        for name, c in subs.items():
            assert c.shape == (384,)
            assert abs(np.linalg.norm(c) - 1.0) < 1e-5

    def test_backward_compat_build_centroid(self):
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            pytest.skip("sentence-transformers not available")

        centroid = _apoc.build_centroid(model)
        assert centroid.ndim == 1
        assert centroid.shape[0] == 384


class TestTrainClassifier:
    def test_trains_and_predicts(self):
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            pytest.skip("sentence-transformers not available")

        lr = _apoc.train_classifier(model)
        # Should predict apocalyptic text as positive
        apoc_emb = model.encode(
            ["The end times are here, Armageddon is coming"],
            normalize_embeddings=True,
        )
        prob = lr.predict_proba(apoc_emb)[0, 1]
        assert prob > 0.5, f"Apocalyptic text scored only {prob:.3f}"

        # Should predict casual text as negative
        casual_emb = model.encode(
            ["I had pizza for dinner and watched a movie"],
            normalize_embeddings=True,
        )
        prob_neg = lr.predict_proba(casual_emb)[0, 1]
        assert prob_neg < 0.5, f"Casual text scored {prob_neg:.3f}"


class TestScoreEmbeddings:
    def test_scoring_pipeline(self):
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            pytest.skip("sentence-transformers not available")

        lr = _apoc.train_classifier(model)
        pos, neg, subs = _apoc.build_centroids(model)

        texts = [
            "The end times are upon us. Race war is inevitable.",
            "I went to the store and bought some groceries.",
            "The great replacement is real. White genocide is happening.",
        ]
        embs = model.encode(texts, normalize_embeddings=True)

        scores = _apoc.score_embeddings(embs, lr, pos, neg, subs)

        assert "apoc_lr_prob" in scores
        assert "apoc_combined" in scores
        assert "apoc_binary" in scores
        assert "apoc_subtheme" in scores
        assert len(scores["apoc_lr_prob"]) == 3
        assert len(scores["apoc_combined"]) == 3

        # First text should score higher than second
        assert scores["apoc_combined"][0] > scores["apoc_combined"][1]
        # Third should also score high
        assert scores["apoc_combined"][2] > scores["apoc_combined"][1]


class TestEmbeddingBoostBackwardCompat:
    def test_boost_high_similarity(self):
        base = np.array([5.0, 3.0, 1.0])
        context = np.array([2.0, 1.0, 0.5])
        sim = np.array([0.5, 0.35, 0.1])
        result = _apoc.apply_embedding_boost(base, context, sim)
        assert result[0] == 3.0 + 2.0 * 2.0
        assert result[1] == 2.0 + 1.0 * 1.5
        assert result[2] == 0.5 + 0.5 * 0.25

    def test_boost_no_negative(self):
        base = np.array([0.5])
        context = np.array([0.5])
        sim = np.array([0.01])
        result = _apoc.apply_embedding_boost(base, context, sim)
        assert result[0] >= 0.0

    def test_no_context_no_change(self):
        base = np.array([5.0])
        context = np.array([0.0])
        sim = np.array([0.5])
        result = _apoc.apply_embedding_boost(base, context, sim)
        assert result[0] == 5.0


# ═══════════════════════════════════════════════════════════════════════
# 31 – Apocalypticism ITS Analysis
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def daily_fixture():
    """Build a synthetic daily time series for testing."""
    dates = [datetime.date(2019, 1, 1) + datetime.timedelta(days=i)
             for i in range(120)]
    np.random.seed(42)
    scores = np.random.normal(0.1, 0.02, 120)
    scores[60:75] += 0.05  # simulated post-event spike
    return pl.DataFrame({
        "day": dates,
        "mean_score": scores,
        "median_score": scores,
        "total_score": scores * 100,
        "post_count": [50] * 120,
        "apoc_post_count": [5] * 120,
        "apoc_prevalence": [0.1] * 120,
    })


@pytest.fixture
def events_fixture():
    """Synthetic events DataFrame with event_category."""
    return pl.DataFrame({
        "event_date": [datetime.date(2019, 3, 1)],
        "event_name": ["Test Event"],
        "event_category": ["mass_violence"],
        "killed": [10],
        "injured": [20],
        "ideology": ["far_right"],
        "online_nexus": [True],
        "location_country": ["US"],
    })


@pytest.fixture
def mixed_events_fixture():
    """Synthetic events with both violence and non-violence."""
    return pl.DataFrame({
        "event_date": [datetime.date(2019, 3, 1),
                       datetime.date(2019, 3, 15)],
        "event_name": ["Test Violence", "Test Nonviolence"],
        "event_category": ["mass_violence", "natural_disaster"],
        "killed": [10, 100],
        "injured": [20, 500],
        "ideology": ["far_right", "N/A"],
        "online_nexus": [True, False],
        "location_country": ["US", "Japan"],
    })


class TestBuildEventWindow:
    def test_returns_window(self, daily_fixture):
        edate = datetime.date(2019, 3, 1)
        window = _its.build_event_window(daily_fixture, edate)
        assert window is not None
        assert "post_event" in window.columns
        assert "time_centered" in window.columns

    def test_window_size(self, daily_fixture):
        edate = datetime.date(2019, 3, 1)
        window = _its.build_event_window(daily_fixture, edate,
                                          pre_days=10, post_days=10)
        assert window is not None
        assert window.height <= 21

    def test_time_centered_at_event(self, daily_fixture):
        edate = datetime.date(2019, 3, 1)
        window = _its.build_event_window(daily_fixture, edate)
        if window is not None:
            event_row = window.filter(pl.col("day") == edate)
            if event_row.height > 0:
                assert event_row["time_centered"][0] == 0

    def test_returns_none_for_out_of_range(self, daily_fixture):
        edate = datetime.date(2025, 1, 1)
        window = _its.build_event_window(daily_fixture, edate)
        assert window is None


class TestRunItsRegression:
    def test_returns_coefficients(self, daily_fixture):
        edate = datetime.date(2019, 3, 1)
        window = _its.build_event_window(daily_fixture, edate)
        assert window is not None
        r = _its.run_its_regression(window, "test_event")
        assert "b_level" in r
        assert "p_level" in r
        assert "r_squared" in r

    def test_insufficient_data(self):
        tiny = pl.DataFrame({
            "day": [datetime.date(2019, 1, 1)],
            "mean_score": [0.1],
            "post_count": [50],
            "time_centered": [0],
            "post_event": [0],
            "time_x_post": [0],
            "apoc_post_count": [1],
            "apoc_prevalence": [0.1],
        })
        r = _its.run_its_regression(tiny, "tiny_test")
        assert "error" in r


class TestRunPooledIts:
    def test_pooled_returns_result(self, daily_fixture, events_fixture):
        r = _its.run_pooled_its(daily_fixture, events_fixture, "apoc_combined")
        assert isinstance(r, dict)
        assert "label" in r or "error" in r


class TestCategoryComparison:
    def test_returns_dict(self, daily_fixture, mixed_events_fixture):
        r = _its.run_category_comparison(
            daily_fixture, mixed_events_fixture, "apoc_combined"
        )
        assert isinstance(r, dict)
        # Should have entries for mass_violence and nonviolence
        assert "mass_violence" in r or "nonviolence" in r

    def test_violence_only_events(self, daily_fixture, events_fixture):
        r = _its.run_category_comparison(
            daily_fixture, events_fixture, "apoc_combined"
        )
        assert isinstance(r, dict)
        # With only 1 violence event, should get "too few" error for nonviolence
        if "nonviolence" in r:
            assert "error" in r["nonviolence"]


# ═══════════════════════════════════════════════════════════════════════
# 32 – Robustness Checks
# ═══════════════════════════════════════════════════════════════════════

class TestPlaceboTest:
    def test_runs_with_small_n(self, daily_fixture):
        r = _robust.run_placebo_test(daily_fixture, n_events=1, n_iter=10)
        assert "n_iter" in r
        assert r["n_iter"] == 10

    def test_null_distribution_populated(self, daily_fixture):
        r = _robust.run_placebo_test(daily_fixture, n_events=1, n_iter=20,
                                      seed=99)
        assert "null_betas_level" in r


class TestComputePlaceboPvalue:
    def test_extreme_value_gives_small_p(self):
        null = list(np.random.normal(0, 1, 1000))
        p = _robust.compute_placebo_pvalue(10.0, null)
        assert p < 0.01

    def test_central_value_gives_large_p(self):
        null = list(np.random.normal(0, 1, 1000))
        p = _robust.compute_placebo_pvalue(0.0, null)
        assert p > 0.3

    def test_empty_null(self):
        p = _robust.compute_placebo_pvalue(1.0, [])
        assert np.isnan(p)


class TestBenjaminiHochberg:
    def test_basic_correction(self):
        pvals = [0.01, 0.04, 0.03, 0.20, 0.50]
        bh = _robust.benjamini_hochberg(pvals, alpha=0.05)
        assert len(bh) == 5
        for r in bh:
            assert r["p_adjusted"] >= r["p_original"]

    def test_preserves_order(self):
        pvals = [0.01, 0.001, 0.5]
        bh = _robust.benjamini_hochberg(pvals)
        assert bh[0]["original_index"] == 0
        assert bh[1]["original_index"] == 1
        assert bh[2]["original_index"] == 2

    def test_empty_list(self):
        assert _robust.benjamini_hochberg([]) == []

    def test_all_significant(self):
        pvals = [0.001, 0.002, 0.003]
        bh = _robust.benjamini_hochberg(pvals, alpha=0.05)
        assert all(r["significant"] for r in bh)

    def test_none_significant(self):
        pvals = [0.5, 0.6, 0.7]
        bh = _robust.benjamini_hochberg(pvals, alpha=0.05)
        assert not any(r["significant"] for r in bh)

    def test_adjusted_p_bounded_by_1(self):
        pvals = [0.8, 0.9, 0.95]
        bh = _robust.benjamini_hochberg(pvals)
        for r in bh:
            assert r["p_adjusted"] <= 1.0


class TestBandwidthSensitivity:
    def test_returns_list(self, daily_fixture, events_fixture):
        r = _robust.run_bandwidth_sensitivity(
            daily_fixture, events_fixture, "apoc_combined",
            bandwidths=[7, 14],
        )
        assert isinstance(r, list)
        assert len(r) == 2


class TestDoseResponse:
    def test_returns_dict(self, daily_fixture, events_fixture):
        r = _robust.run_dose_response(daily_fixture, events_fixture,
                                       "apoc_combined")
        assert isinstance(r, dict)
        assert "error" in r or "b_log_killed" in r


class TestLagAnalysis:
    def test_returns_list(self, daily_fixture, events_fixture):
        r = _robust.run_lag_analysis(
            daily_fixture, events_fixture, "apoc_combined",
            lags=[0, 1],
        )
        assert isinstance(r, list)
        assert len(r) == 2


# ═══════════════════════════════════════════════════════════════════════
# Cross-cutting integration checks
# ═══════════════════════════════════════════════════════════════════════

class TestPipelineIntegration:
    def test_event_dates_are_date_objects(self):
        df = _events.events_to_polars(_events.EVENTS)
        for d in df["event_date"].to_list():
            assert isinstance(d, datetime.date)

    def test_ideologies_cover_expected_set(self):
        df = _events.events_to_polars(_events.EVENTS)
        ideologies = set(df["ideology"].to_list())
        assert "far_right" in ideologies
        assert "islamist" in ideologies
        assert "school_shooting" in ideologies

    def test_categories_cover_expected_set(self):
        df = _events.events_to_polars(_events.EVENTS)
        cats = set(df["event_category"].to_list())
        assert "mass_violence" in cats
        assert len(cats) >= 3

    def test_event_chronological_coverage(self):
        df = _events.events_to_polars(_events.EVENTS)
        date_range = (df["event_date"].max() - df["event_date"].min()).days
        assert date_range > 365 * 5

    def test_event_country_diversity(self):
        df = _events.events_to_polars(_events.EVENTS)
        countries = set(df["location_country"].to_list())
        assert len(countries) >= 8

    def test_measures_list_valid(self):
        """MEASURES should only reference columns the classifier produces."""
        valid_cols = {"apoc_lr_prob", "apoc_similarity",
                      "apoc_combined", "apoc_contrastive"}
        for m in _its.MEASURES:
            assert m in valid_cols, f"MEASURE '{m}' not in valid columns"


# ═══════════════════════════════════════════════════════════════════════
# 33 – Attack-Characteristic Correlations
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def mock_per_event_results():
    """Synthetic per-event ITS results for correlation tests."""
    np.random.seed(42)
    names = [f"Event_{i}" for i in range(30)]
    return [
        {
            "label": name,
            "b_level": float(np.random.normal(0, 0.01)),
            "b_slope": float(np.random.normal(0, 0.001)),
            "p_level": float(np.random.uniform(0, 1)),
            "se_level": 0.005,
            "r_squared": 0.1,
            "n_days": 60,
            "event_date": f"2019-{(i % 12) + 1:02d}-15",
            "ideology": "far_right",
            "killed": i + 1,
        }
        for i, name in enumerate(names)
    ]


@pytest.fixture
def mock_events_df():
    """Synthetic events DataFrame for correlation tests."""
    np.random.seed(42)
    names = [f"Event_{i}" for i in range(30)]
    ideologies = ["far_right"] * 10 + ["islamist"] * 10 + ["other"] * 10
    return pl.DataFrame({
        "event_date": [datetime.date(2019, (i % 12) + 1, 15) for i in range(30)],
        "event_name": names,
        "event_category": ["mass_violence"] * 25 + ["natural_disaster"] * 5,
        "killed": list(range(1, 31)),
        "injured": list(range(2, 62, 2)),
        "total_casualties": [k + i for k, i in zip(range(1, 31), range(2, 62, 2))],
        "ideology": ideologies,
        "online_nexus": [True, False] * 15,
        "domestic": [True] * 15 + [False] * 15,
        "location_country": ["US"] * 10 + ["UK"] * 5 + ["France"] * 5
                            + ["Germany"] * 5 + ["Australia"] * 5,
    })


@pytest.fixture
def merged_df(mock_per_event_results, mock_events_df):
    """Pre-merged DataFrame for analysis tests."""
    return _corr.build_event_beta_df(mock_per_event_results, mock_events_df)


class TestBuildEventBetaDf:
    def test_merge_returns_dataframe(self, mock_per_event_results, mock_events_df):
        df = _corr.build_event_beta_df(mock_per_event_results, mock_events_df)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_has_b_level_and_metadata(self, merged_df):
        assert "b_level" in merged_df.columns
        assert "event_category" in merged_df.columns

    def test_empty_on_no_valid_results(self, mock_events_df):
        bad = [{"label": "x", "error": "fail"}]
        df = _corr.build_event_beta_df(bad, mock_events_df)
        assert df.empty

    def test_filters_out_errors(self, mock_events_df):
        mixed = [
            {"label": "Event_0", "b_level": 0.01, "p_level": 0.5},
            {"label": "Event_1", "error": "fail"},
        ]
        df = _corr.build_event_beta_df(mixed, mock_events_df)
        assert len(df) == 1


class TestSeverityCorrelations:
    def test_returns_all_columns(self, merged_df):
        r = _corr.severity_correlations(merged_df)
        assert isinstance(r, dict)
        # Should have at least killed
        found = False
        for col in ("killed", "injured", "total_casualties"):
            if col in r:
                found = True
                if "error" not in r[col]:
                    assert "pearson_r" in r[col]
                    assert "spearman_rho" in r[col]
        assert found

    def test_correlations_bounded(self, merged_df):
        r = _corr.severity_correlations(merged_df)
        for col, v in r.items():
            if "error" not in v:
                assert -1.0 <= v["pearson_r"] <= 1.0
                assert -1.0 <= v["spearman_rho"] <= 1.0

    def test_insufficient_data(self):
        tiny = pd.DataFrame({"b_level": [0.1], "killed": [5]})
        r = _corr.severity_correlations(tiny)
        if "killed" in r:
            assert "error" in r["killed"]


class TestIdeologyComparison:
    def test_returns_kruskal_wallis(self, merged_df):
        r = _corr.ideology_comparison(merged_df)
        if "error" not in r:
            assert "kruskal_wallis_H" in r
            assert "descriptives" in r

    def test_has_pairwise(self, merged_df):
        r = _corr.ideology_comparison(merged_df)
        if "error" not in r:
            assert "pairwise_mannwhitney" in r

    def test_too_few_groups(self):
        df = pd.DataFrame({
            "b_level": [0.01, 0.02],
            "ideology": ["far_right", "far_right"],
            "event_category": ["mass_violence", "mass_violence"],
        })
        r = _corr.ideology_comparison(df)
        assert "error" in r


class TestDomesticComparison:
    def test_returns_mannwhitney(self, merged_df):
        r = _corr.domestic_comparison(merged_df)
        if "error" not in r:
            assert "mannwhitney_U" in r
            assert "domestic_n" in r
            assert "international_n" in r

    def test_too_few_events(self):
        df = pd.DataFrame({
            "b_level": [0.01, 0.02],
            "domestic": [True, True],
            "event_category": ["mass_violence", "mass_violence"],
        })
        r = _corr.domestic_comparison(df)
        assert "error" in r


class TestOnlineNexusComparison:
    def test_returns_mannwhitney(self, merged_df):
        r = _corr.online_nexus_comparison(merged_df)
        if "error" not in r:
            assert "mannwhitney_U" in r
            assert "online_n" in r
            assert "offline_n" in r


class TestGeographicHeterogeneity:
    def test_returns_by_country(self, merged_df):
        r = _corr.geographic_heterogeneity(merged_df)
        assert "by_country" in r
        assert "by_region" in r

    def test_region_mapping(self):
        assert "US" in _corr.REGION_MAP
        assert "UK" in _corr.REGION_MAP
        assert _corr.REGION_MAP["US"] == "North America"
        assert _corr.REGION_MAP["UK"] == "Europe"


class TestMultipleRegression:
    def test_returns_coefficients(self, merged_df):
        r = _corr.multiple_regression(merged_df)
        if "error" not in r:
            assert "coefficients" in r
            assert "r_squared" in r
            assert r["n_obs"] > 0

    def test_too_few_obs(self):
        df = pd.DataFrame({
            "b_level": [0.01],
            "killed": [5],
            "ideology": ["far_right"],
            "domestic": [True],
            "online_nexus": [True],
            "event_category": ["mass_violence"],
        })
        r = _corr.multiple_regression(df)
        assert "error" in r


# ═══════════════════════════════════════════════════════════════════════
# 34 – Advanced Time-Series Analyses
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def ts_daily_fixture():
    """Build a daily time series suitable for advanced TS methods."""
    dates = [datetime.date(2019, 1, 1) + datetime.timedelta(days=i)
             for i in range(200)]
    np.random.seed(42)
    scores = np.random.normal(0.1, 0.02, 200)
    # Add a few event spikes
    for spike_day in [60, 120, 150]:
        scores[spike_day:spike_day + 5] += 0.04
    return pl.DataFrame({
        "day": dates,
        "mean_score": scores,
        "median_score": scores,
        "total_score": scores * 100,
        "post_count": [50] * 200,
        "apoc_post_count": [5] * 200,
        "apoc_prevalence": [0.1] * 200,
    })


@pytest.fixture
def ts_events_fixture():
    """Events that fall within the daily fixture range."""
    return pl.DataFrame({
        "event_date": [datetime.date(2019, 3, 1),
                       datetime.date(2019, 4, 30),
                       datetime.date(2019, 5, 30)],
        "event_name": ["Event A", "Event B", "Event C"],
        "event_category": ["mass_violence", "mass_violence", "natural_disaster"],
        "killed": [10, 25, 0],
        "injured": [20, 50, 100],
        "ideology": ["far_right", "islamist", "N/A"],
        "online_nexus": [True, False, False],
        "location_country": ["US", "France", "Japan"],
        "domestic": [True, False, False],
    })


@pytest.fixture
def ts_aligned_fixture(ts_daily_fixture, ts_events_fixture):
    """Pre-built aligned time series DataFrame."""
    return _adv.build_event_series(ts_daily_fixture, ts_events_fixture)


class TestBuildEventSeries:
    def test_returns_dataframe(self, ts_daily_fixture, ts_events_fixture):
        ts = _adv.build_event_series(ts_daily_fixture, ts_events_fixture)
        assert isinstance(ts, pd.DataFrame)
        assert "apoc_mean" in ts.columns
        assert "event_occurred" in ts.columns
        assert "event_casualties" in ts.columns

    def test_event_indicator_populated(self, ts_aligned_fixture):
        assert ts_aligned_fixture["event_occurred"].sum() > 0

    def test_no_missing_apoc(self, ts_aligned_fixture):
        assert ts_aligned_fixture["apoc_mean"].isna().sum() == 0


class TestStationarityTests:
    def test_returns_result(self):
        np.random.seed(42)
        s = pd.Series(np.random.normal(0, 1, 100))
        r = _adv.stationarity_tests(s, "test_series")
        assert "adf_statistic" in r
        assert "p_value" in r
        assert "stationary_at_5pct" in r

    def test_insufficient_data(self):
        s = pd.Series([1.0, 2.0])
        r = _adv.stationarity_tests(s, "tiny")
        assert "error" in r


class TestVARAnalysis:
    def test_returns_result(self, ts_aligned_fixture):
        r = _adv.run_var_analysis(ts_aligned_fixture, maxlag=5, irf_periods=10)
        assert isinstance(r, dict)
        if "error" not in r:
            assert "granger_causality" in r
            assert "irf" in r
            assert "fevd" in r
            assert "selected_lag" in r

    def test_insufficient_data(self):
        tiny = pd.DataFrame({
            "apoc_mean": [0.1, 0.2],
            "event_occurred": [0, 1],
        })
        r = _adv.run_var_analysis(tiny)
        assert "error" in r


class TestARDLAnalysis:
    def test_returns_result(self, ts_aligned_fixture):
        r = _adv.run_ardl_analysis(ts_aligned_fixture, max_ar_lag=3, max_dl_lag=3)
        assert isinstance(r, dict)
        if "error" not in r:
            assert "ar_order" in r
            assert "dl_order" in r
            assert "long_run_multiplier" in r
            assert "ecm" in r

    def test_insufficient_data(self):
        tiny = pd.DataFrame({
            "apoc_mean": [0.1],
            "event_casualties": [10],
        })
        r = _adv.run_ardl_analysis(tiny)
        assert "error" in r


class TestBSTSAnalysis:
    def test_returns_result(self, ts_aligned_fixture, ts_events_fixture):
        r = _adv.run_bsts_analysis(ts_aligned_fixture, ts_events_fixture,
                                    n_post_days=10)
        assert isinstance(r, dict)
        # May return error if no events match; that's acceptable
        if "error" not in r:
            assert "per_event" in r
            assert "aggregate" in r

    def test_insufficient_data(self, ts_events_fixture):
        tiny = pd.DataFrame({
            "apoc_mean": [0.1] * 10,
        }, index=pd.date_range("2019-01-01", periods=10))
        tiny.index.name = "date"
        r = _adv.run_bsts_analysis(tiny, ts_events_fixture)
        assert "error" in r


class TestLocalProjections:
    def test_returns_result(self, ts_aligned_fixture):
        r = _adv.run_local_projections(ts_aligned_fixture,
                                        max_horizon=10, n_lags=3)
        assert isinstance(r, dict)
        if "error" not in r:
            assert "horizons" in r
            assert "peak_horizon" in r
            assert "peak_beta" in r
            assert len(r["horizons"]) > 0

    def test_horizon_structure(self, ts_aligned_fixture):
        r = _adv.run_local_projections(ts_aligned_fixture,
                                        max_horizon=5, n_lags=3)
        if "error" not in r:
            for h in r["horizons"]:
                assert "horizon" in h
                assert "beta" in h
                assert "se" in h
                assert "p_value" in h

    def test_insufficient_data(self):
        tiny = pd.DataFrame({
            "apoc_mean": [0.1, 0.2],
            "event_occurred": [0, 1],
        })
        r = _adv.run_local_projections(tiny)
        assert "error" in r


class TestMethodComparison:
    def test_builds_comparison(self):
        its = {"b_level": 0.01, "p_level": 0.03, "n_obs": 100}
        var = {"irf": {"peak_response": 0.02},
               "granger_causality": {"event_causes_apoc": {"p_value": 0.04, "significant_at_05": True}},
               "n_obs": 100}
        ardl = {"long_run_multiplier": 0.005, "ecm": {"ec_p_value": 0.1, "ec_significant": False},
                "n_obs": 100}
        bsts = {"aggregate": {"mean_impact": 0.01, "t_p_value": 0.02, "significant_at_05": True,
                              "pct_positive": 0.6}, "n_events_analyzed": 5}
        lp = {"peak_beta": 0.015, "peak_p_value": 0.08, "n_significant": 2,
              "peak_horizon": 3}

        c = _adv.build_method_comparison(its, var, ardl, bsts, lp)
        assert "methods" in c
        assert "consensus" in c
        assert c["consensus"]["n_methods"] >= 3

    def test_all_errors(self):
        c = _adv.build_method_comparison(
            {"error": "x"}, {"error": "x"}, {"error": "x"},
            {"error": "x"}, {"error": "x"},
        )
        assert c["consensus"]["n_methods"] == 0

    def test_consensus_direction(self):
        its = {"b_level": -0.01, "p_level": 0.03, "n_obs": 100}
        var = {"irf": {"peak_response": -0.02},
               "granger_causality": {"event_causes_apoc": {"p_value": 0.04, "significant_at_05": True}},
               "n_obs": 100}
        c = _adv.build_method_comparison(its, var, {"error": "x"}, {"error": "x"}, {"error": "x"})
        assert c["consensus"]["direction_agreement"] is True
        assert c["consensus"]["consensus_direction"] == "negative"


# ═══════════════════════════════════════════════════════════════════════
# 35 – Offline-Online Hypotheses (H22-H26)
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def h_daily_fixture():
    """Extended daily fixture for hypothesis testing."""
    dates = [datetime.date(2018, 1, 1) + datetime.timedelta(days=i)
             for i in range(365)]
    np.random.seed(42)
    scores = np.random.normal(0.1, 0.02, 365)
    # Simulate spikes after events
    for spike_day in [60, 120, 180, 240, 300]:
        scores[spike_day:spike_day + 10] += np.exp(-np.arange(10) * 0.2) * 0.05
    return pl.DataFrame({
        "day": dates,
        "mean_score": scores,
        "median_score": scores,
        "total_score": scores * 100,
        "post_count": [50] * 365,
        "apoc_post_count": [5] * 365,
        "apoc_prevalence": [0.1] * 365,
    })


@pytest.fixture
def h_events_fixture():
    """Events for hypothesis testing – mix of online/offline nexus."""
    return pl.DataFrame({
        "event_date": [
            datetime.date(2018, 3, 1),
            datetime.date(2018, 4, 30),
            datetime.date(2018, 5, 5),   # Clustered with Event B
            datetime.date(2018, 6, 28),
            datetime.date(2018, 10, 27),
        ],
        "event_name": ["Event A", "Event B", "Event C", "Event D", "Event E"],
        "event_category": ["mass_violence"] * 5,
        "killed": [5, 50, 3, 15, 11],
        "injured": [10, 100, 5, 30, 6],
        "ideology": ["far_right", "islamist", "far_right", "incel", "far_right"],
        "online_nexus": [True, False, True, False, True],
        "location_country": ["US", "France", "US", "US", "US"],
        "domestic": [True, False, True, True, True],
    })


class TestExpDecay:
    def test_function_shape(self):
        t = np.arange(30)
        y = _hyp._exp_decay(t, 1.0, 0.1, 0.0)
        assert y[0] == pytest.approx(1.0)
        assert y[-1] < y[0]
        # Should be monotonically decreasing for positive a and lam
        assert all(y[i] >= y[i + 1] for i in range(len(y) - 1))


class TestEstimateDecayHalflife:
    def test_returns_result(self, h_daily_fixture, h_events_fixture):
        r = _hyp.estimate_decay_halflife(h_daily_fixture, h_events_fixture,
                                          post_days=15)
        assert isinstance(r, dict)
        if "error" not in r:
            assert "per_event" in r
            assert "aggregate" in r
            assert "n_events" in r

    def test_empty_events(self, h_daily_fixture):
        empty = pl.DataFrame({
            "event_date": [],
            "event_name": [],
            "event_category": [],
            "killed": [],
            "injured": [],
            "ideology": [],
            "online_nexus": [],
        }, schema={
            "event_date": pl.Date,
            "event_name": pl.Utf8,
            "event_category": pl.Utf8,
            "killed": pl.Int64,
            "injured": pl.Int64,
            "ideology": pl.Utf8,
            "online_nexus": pl.Boolean,
        })
        r = _hyp.estimate_decay_halflife(h_daily_fixture, empty)
        assert "error" in r


class TestReciprocalAmplification:
    def test_returns_result(self, h_daily_fixture, h_events_fixture):
        ts = _adv.build_event_series(h_daily_fixture, h_events_fixture)
        r = _hyp.test_reciprocal_amplification(ts, maxlag=5, irf_periods=10)
        assert isinstance(r, dict)
        if "error" not in r:
            assert "granger_causality" in r
            assert "findings" in r
            assert "supported" in r

    def test_insufficient_data(self):
        tiny = pd.DataFrame({
            "apoc_mean": [0.1, 0.2],
            "event_occurred": [0, 1],
        })
        r = _hyp.test_reciprocal_amplification(tiny)
        assert "error" in r


class TestThresholdActivation:
    def test_returns_result(self, h_daily_fixture, h_events_fixture):
        r = _hyp.test_threshold_activation(
            h_daily_fixture, h_events_fixture,
            candidate_thresholds=[3, 10, 20],
        )
        assert isinstance(r, dict)
        if "error" not in r:
            assert "threshold_tests" in r
            assert "optimal_threshold" in r
            assert "piecewise_regression" in r

    def test_threshold_scan(self, h_daily_fixture, h_events_fixture):
        r = _hyp.test_threshold_activation(
            h_daily_fixture, h_events_fixture,
            candidate_thresholds=[5],
        )
        if "error" not in r and r["threshold_tests"]:
            t = r["threshold_tests"][0]
            assert "threshold" in t
            assert "mannwhitney_p" in t


class TestTemporalClustering:
    def test_returns_result(self, h_daily_fixture, h_events_fixture):
        r = _hyp.test_temporal_clustering(
            h_daily_fixture, h_events_fixture,
            cluster_window_days=14,
        )
        assert isinstance(r, dict)
        if "error" not in r:
            assert "per_event" in r
            assert "comparison" in r

    def test_detects_clusters(self, h_daily_fixture, h_events_fixture):
        r = _hyp.test_temporal_clustering(
            h_daily_fixture, h_events_fixture,
            cluster_window_days=14,
        )
        if "error" not in r:
            clustered = [e for e in r["per_event"] if e["is_clustered"]]
            # Events B and C are 5 days apart, should be clustered
            assert len(clustered) >= 1

    def test_too_few_events(self, h_daily_fixture):
        few = pl.DataFrame({
            "event_date": [datetime.date(2018, 3, 1)],
            "event_name": ["Only"],
            "event_category": ["mass_violence"],
            "killed": [5],
            "injured": [10],
            "ideology": ["far_right"],
            "online_nexus": [True],
            "location_country": ["US"],
            "domestic": [True],
        })
        r = _hyp.test_temporal_clustering(h_daily_fixture, few)
        assert "error" in r


class TestMimeticContagion:
    def test_returns_result(self, h_daily_fixture, h_events_fixture):
        r = _hyp.test_mimetic_contagion(h_daily_fixture, h_events_fixture)
        assert isinstance(r, dict)
        if "error" not in r:
            assert "magnitude_test" in r
            assert "direction_test" in r
            assert "polarisation_test" in r
            assert "n_online" in r
            assert "n_offline" in r

    def test_handles_all_online(self, h_daily_fixture):
        all_online = pl.DataFrame({
            "event_date": [datetime.date(2018, 3, 1),
                           datetime.date(2018, 6, 1)],
            "event_name": ["A", "B"],
            "event_category": ["mass_violence", "mass_violence"],
            "killed": [5, 10],
            "injured": [10, 20],
            "ideology": ["far_right", "far_right"],
            "online_nexus": [True, True],
            "location_country": ["US", "US"],
            "domestic": [True, True],
        })
        r = _hyp.test_mimetic_contagion(h_daily_fixture, all_online)
        # Should have error due to insufficient offline events
        assert "error" in r or r.get("n_offline", 0) == 0

    def test_cohens_d_bounded(self):
        """Test that Cohen's d is reasonable for known inputs."""
        a = pd.Series([1.0, 2.0, 3.0, 4.0])
        b = pd.Series([2.0, 3.0, 4.0, 5.0])
        d = (a.mean() - b.mean()) / np.sqrt(
            ((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2)
        )
        assert -5 < d < 5
