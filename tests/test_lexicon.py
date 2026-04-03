"""Tests for 02_siege_lexicon.py – dictionary-based scoring."""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import polars as pl

_mod = importlib.import_module("02_siege_lexicon")
compute_siege_keyword_score = _mod.compute_siege_keyword_score
score_dataframe = _mod.score_dataframe


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

    def test_binary_correctness(self):
        df = pl.DataFrame({
            "text": ["Hello world", "Read Siege now", "The castle siege was long"],
        })
        result = score_dataframe(df)
        binaries = result["siege_binary"].to_list()
        assert binaries[0] == 0   # no siege terms
        assert binaries[1] == 1   # positive siege score
        assert binaries[2] == 0   # counter-indicated: castle siege has net negative score
