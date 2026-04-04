"""Tests for 02_siege_lexicon.py – dictionary-based scoring."""

import importlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import polars as pl

_mod = importlib.import_module("02_siege_lexicon")
compute_siege_keyword_score = _mod.compute_siege_keyword_score
score_dataframe = _mod.score_dataframe
apply_embedding_boost = _mod.apply_embedding_boost
embedding_boost_factor = _mod.embedding_boost_factor


class TestComputeSiegeKeywordScore:
    def test_no_keywords(self):
        result = compute_siege_keyword_score("Hello this is a normal post")
        assert result["keyword_count"] == 0
        assert result["keyword_score"] == 0.0
        assert result["keyword_density"] == 0.0

    def test_single_keyword_siege(self):
        result = compute_siege_keyword_score("Read Siege now")
        # "siege" + "read siege"
        assert result["keyword_count"] >= 1
        assert result["keyword_score"] > 0

    def test_james_mason(self):
        result = compute_siege_keyword_score("James Mason wrote this book")
        assert result["keyword_count"] >= 1
        assert result["keyword_score"] > 0

    def test_counter_indicator(self):
        result = compute_siege_keyword_score("The medieval siege was brutal")
        # "siege" (+1.0) + "medieval siege" (-2.0) = net negative
        assert result["keyword_score"] < 1.0

    def test_rainbow_six(self):
        result = compute_siege_keyword_score("I play Rainbow Six Siege")
        # "siege" (+1.0) + "rainbow six" (-3.0) = net negative
        assert result["keyword_score"] < 0

    def test_multiple_terms(self):
        result = compute_siege_keyword_score(
            "Read Siege by James Mason. Acceleration of the collapse is coming."
        )
        assert result["keyword_count"] >= 3
        assert result["keyword_score"] > 5.0

    def test_case_insensitive(self):
        result = compute_siege_keyword_score("SIEGE")
        assert result["keyword_count"] >= 1

    def test_none_input(self):
        result = compute_siege_keyword_score(None)
        assert result["keyword_count"] == 0

    def test_empty_string(self):
        result = compute_siege_keyword_score("")
        assert result["keyword_count"] == 0

    def test_density_normalisation(self):
        result = compute_siege_keyword_score("siege " * 10)
        assert result["keyword_density"] > 0
        # 10 occurrences of "siege" in 10 words → density = 10*1.0/10 = 1.0
        assert abs(result["keyword_density"] - 1.0) < 0.1

    def test_atomwaffen(self):
        result = compute_siege_keyword_score("Atomwaffen Division is dangerous")
        assert result["keyword_count"] >= 1
        assert result["keyword_score"] >= 3.0

    def test_accelerate(self):
        result = compute_siege_keyword_score("We must accelerate the process")
        assert result["keyword_count"] >= 1

    def test_rahowa(self):
        result = compute_siege_keyword_score("RAHOWA now")
        assert result["keyword_count"] >= 1

    # ── New terms from expanded dictionary ────────────────────────────

    def test_tommasi(self):
        result = compute_siege_keyword_score("Tommasi was a true revolutionary")
        assert result["keyword_count"] >= 1
        assert result["keyword_score"] >= 2.5

    def test_joseph_tommasi(self):
        result = compute_siege_keyword_score("Joseph Tommasi founded the NSLF")
        assert result["keyword_score"] >= 3.0

    def test_william_pierce(self):
        result = compute_siege_keyword_score("William Luther Pierce wrote the Turner Diaries")
        assert result["keyword_count"] >= 1
        assert result["keyword_score"] >= 2.0

    def test_turner_diaries(self):
        result = compute_siege_keyword_score("The Turner Diaries inspired violence")
        assert result["keyword_count"] >= 1
        assert result["keyword_score"] >= 2.0

    def test_o9a(self):
        result = compute_siege_keyword_score("O9A influences within the group")
        assert result["keyword_count"] >= 1
        assert result["keyword_score"] >= 2.0

    def test_savitri_devi(self):
        result = compute_siege_keyword_score("Savitri Devi esoteric hitlerism")
        assert result["keyword_count"] >= 1
        assert result["keyword_score"] >= 2.0

    def test_skull_mask(self):
        result = compute_siege_keyword_score("The skull mask network spreads")
        assert result["keyword_count"] >= 1
        assert result["keyword_score"] >= 2.5

    def test_ethnostate(self):
        result = compute_siege_keyword_score("Build the white ethnostate")
        assert result["keyword_count"] >= 1

    def test_breivik(self):
        result = compute_siege_keyword_score("Breivik was a saint to them")
        assert result["keyword_count"] >= 1
        assert result["keyword_score"] >= 1.5

    def test_political_terror(self):
        result = compute_siege_keyword_score("Political terror is the only thing they understand")
        assert result["keyword_count"] >= 1
        assert result["keyword_score"] >= 3.0

    # ── Context-dependent scoring ─────────────────────────────────────

    def test_context_dependent_the_system(self):
        result = compute_siege_keyword_score("We must destroy the system")
        assert result["keyword_context_score"] > 0
        # "the system" is context-dependent with weight 0.3
        assert abs(result["keyword_context_score"] - 0.3) < 0.01

    def test_context_dependent_collapse(self):
        result = compute_siege_keyword_score("The collapse of civilisation")
        assert result["keyword_context_score"] > 0

    def test_no_context_score_for_unconditional(self):
        result = compute_siege_keyword_score("James Mason wrote Siege")
        # james mason and siege are unconditional — no context_score
        assert result["keyword_context_score"] == 0.0

    def test_total_war_game_counter(self):
        result = compute_siege_keyword_score("I love Total War Warhammer")
        # total war (+0.3) + total war warhammer (-2.0) = net negative
        assert result["keyword_score"] < 0

    # ── Embedding boost ───────────────────────────────────────────────

    def test_embedding_boost_factor_high_sim(self):
        assert embedding_boost_factor(0.5) == 2.0
        assert embedding_boost_factor(0.4) == 2.0

    def test_embedding_boost_factor_medium_sim(self):
        assert embedding_boost_factor(0.35) == 1.5
        assert embedding_boost_factor(0.3) == 1.5

    def test_embedding_boost_factor_low_sim(self):
        assert embedding_boost_factor(0.1) == 0.25
        assert embedding_boost_factor(0.0) == 0.25

    def test_embedding_boost_factor_neutral(self):
        assert embedding_boost_factor(0.2) == 1.0
        assert embedding_boost_factor(0.25) == 1.0

    def test_apply_embedding_boost(self):
        kw = [5.0, 5.0, 5.0]
        ctx = [1.0, 1.0, 1.0]
        sims = [0.5, 0.2, 0.1]
        adj = apply_embedding_boost(kw, ctx, sims)
        # sim=0.5 → boost=2.0 → 4.0 + 1.0*2.0 = 6.0
        assert abs(adj[0] - 6.0) < 0.01
        # sim=0.2 → boost=1.0 → 4.0 + 1.0*1.0 = 5.0
        assert abs(adj[1] - 5.0) < 0.01
        # sim=0.1 → boost=0.25 → 4.0 + 1.0*0.25 = 4.25
        assert abs(adj[2] - 4.25) < 0.01


class TestScoreDataframe:
    def test_adds_columns(self):
        df = pl.DataFrame({
            "text": ["Hello world", "Read Siege", "Normal post"],
        })
        result = score_dataframe(df)
        assert "siege_keyword_count" in result.columns
        assert "siege_keyword_score" in result.columns
        assert "siege_keyword_density" in result.columns
        assert "siege_binary" in result.columns
        assert "siege_keyword_context_score" in result.columns

    def test_binary_correctness(self):
        df = pl.DataFrame({
            "text": ["Hello world", "Read Siege now", "The castle siege was long"],
        })
        result = score_dataframe(df)
        binaries = result["siege_binary"].to_list()
        assert binaries[0] == 0   # no siege terms
        assert binaries[1] == 1   # positive siege score
        assert binaries[2] == 0   # counter-indicated: castle siege has net negative score
