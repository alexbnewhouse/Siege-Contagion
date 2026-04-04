"""Tests for /pol/ ingest, preprocessing, and cross-platform modules."""

import importlib
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ═══════════════════════════════════════════════════════════════════════
# 00b_ingest_pol tests
# ═══════════════════════════════════════════════════════════════════════

_ingest = importlib.import_module("00b_ingest_pol")


class TestCleanValue:
    def test_mysql_null(self):
        assert _ingest._clean_value("\\N") is None

    def test_normal_value(self):
        assert _ingest._clean_value("hello") == "hello"

    def test_empty_string(self):
        assert _ingest._clean_value("") == ""


class TestSafeInt:
    def test_normal_int(self):
        assert _ingest._safe_int("42") == 42

    def test_mysql_null(self):
        assert _ingest._safe_int("\\N") is None

    def test_none(self):
        assert _ingest._safe_int(None) is None

    def test_non_numeric(self):
        assert _ingest._safe_int("abc") is None


class TestUnixToIso:
    def test_valid_timestamp(self):
        result = _ingest._unix_to_iso("1420070400")
        assert result is not None
        assert "2015" in result

    def test_null_timestamp(self):
        assert _ingest._unix_to_iso("\\N") is None

    def test_zero_timestamp(self):
        assert _ingest._unix_to_iso("0") is None


class TestPrefilterRegex:
    def test_matches_siege(self):
        assert _ingest.PREFILTER_RE.search("Read Siege by James Mason")

    def test_matches_atomwaffen(self):
        assert _ingest.PREFILTER_RE.search("Atomwaffen Division")

    def test_matches_accelerate(self):
        assert _ingest.PREFILTER_RE.search("we must accelerate")

    def test_no_match_normal_text(self):
        assert not _ingest.PREFILTER_RE.search("The weather is nice today")

    def test_case_insensitive(self):
        assert _ingest.PREFILTER_RE.search("SIEGE")

    def test_matches_ironmarch(self):
        assert _ingest.PREFILTER_RE.search("check out iron march")

    def test_matches_boogaloo(self):
        assert _ingest.PREFILTER_RE.search("boogaloo boys")


class TestAsagiColumns:
    def test_column_count(self):
        assert len(_ingest.ASAGI_COLUMNS) == 28

    def test_first_column_is_num(self):
        assert _ingest.ASAGI_COLUMNS[0] == "num"

    def test_comment_at_index_22(self):
        assert _ingest.ASAGI_COLUMNS[22] == "comment"

    def test_last_column_is_exif(self):
        assert _ingest.ASAGI_COLUMNS[-1] == "exif"

    def test_poster_hash_at_index_25(self):
        assert _ingest.ASAGI_COLUMNS[25] == "poster_hash"


class TestRowsToDataframe:
    def test_empty_rows(self):
        df = _ingest.rows_to_dataframe([])
        assert df.height == 0
        assert "num" in df.columns

    def test_single_row(self):
        row = {
            "num": 12345,
            "subnum": 0,
            "thread_num": 12345,
            "op": 1,
            "timestamp": 1420070400,
            "name": "Anonymous",
            "trip": None,
            "title": "Test thread",
            "comment": "Hello world",
            "sticky": 0,
            "locked": 0,
            "poster_hash": "abc123",
            "poster_country": "US",
        }
        df = _ingest.rows_to_dataframe([row])
        assert df.height == 1
        assert df["num"][0] == 12345
        assert "date" in df.columns


# ═══════════════════════════════════════════════════════════════════════
# 01b_preprocess_pol tests
# ═══════════════════════════════════════════════════════════════════════

_preprocess = importlib.import_module("01b_preprocess_pol")


class TestStrip4chanHtml:
    def test_quotelink_removed(self):
        html = '<a href="#p12345" class="quotelink">&gt;&gt;12345</a>'
        # Default: quotelinks are removed entirely
        assert _preprocess.strip_4chan_html(html) == ""

    def test_greentext_removed(self):
        html = '<span class="quote">&gt;implying</span>'
        # Default: greentext is removed
        result = _preprocess.strip_4chan_html(html)
        assert ">implying" not in result

    def test_greentext_kept(self):
        html = '<span class="quote">&gt;implying</span>'
        result = _preprocess.strip_4chan_html(html, keep_greentext=True)
        assert ">implying" in result

    def test_spoiler(self):
        html = '<s>spoiler text</s>'
        result = _preprocess.strip_4chan_html(html)
        assert "spoiler text" in result

    def test_br_to_newline(self):
        html = 'line one<br>line two'
        result = _preprocess.strip_4chan_html(html)
        assert "\n" in result

    def test_wbr_removal(self):
        html = 'long<wbr>word'
        result = _preprocess.strip_4chan_html(html)
        assert result == "longword"

    def test_deadlink_removed(self):
        html = '<span class="deadlink">&gt;&gt;99999</span>'
        # Deadlinks are removed
        result = _preprocess.strip_4chan_html(html)
        assert result == ""

    def test_none_input(self):
        result = _preprocess.strip_4chan_html(None)
        assert result == ""

    def test_empty_string(self):
        result = _preprocess.strip_4chan_html("")
        assert result == ""

    def test_plain_text(self):
        assert _preprocess.strip_4chan_html("just text") == "just text"

    def test_html_entities(self):
        result = _preprocess.strip_4chan_html("&amp; &lt; &gt;")
        assert "&" in result


class TestNormalisePolSchema:
    def test_renames_columns(self):
        df = pl.DataFrame({
            "num": [1],
            "thread_num": [1],
            "poster_hash": ["abc"],
            "comment": ["text"],
            "timestamp": [1420070400],
            "date": [None],
        })
        result = _preprocess.normalise_pol_schema(df)
        assert "post_id" in result.columns
        assert "thread_id" in result.columns
        assert "author_id" in result.columns

    def test_null_poster_hash_becomes_anon(self):
        df = pl.DataFrame({
            "num": [1],
            "thread_num": [1],
            "poster_hash": [None],
            "comment": ["text"],
            "timestamp": [1420070400],
            "date": [None],
        })
        result = _preprocess.normalise_pol_schema(df)
        assert result["author_id"][0] == "anon"


# ═══════════════════════════════════════════════════════════════════════
# 19_cross_platform_bridges tests
# ═══════════════════════════════════════════════════════════════════════

_bridges = importlib.import_module("19_cross_platform_bridges")


class TestExtractUrls:
    def test_simple_url(self):
        urls = _bridges.extract_urls("Check out https://siege.example.com/post")
        assert len(urls) == 1
        assert "siege.example.com" in urls[0]

    def test_noise_domain_filtered(self):
        urls = _bridges.extract_urls("See https://youtube.com/watch?v=123")
        assert len(urls) == 0

    def test_multiple_urls(self):
        text = "Visit https://a.example.com and https://b.example.com"
        urls = _bridges.extract_urls(text)
        assert len(urls) == 2

    def test_none_input(self):
        assert _bridges.extract_urls(None) == []

    def test_no_urls(self):
        assert _bridges.extract_urls("No links here") == []


class TestExtractNgrams:
    def test_four_grams(self):
        text = "one two three four five"
        ngrams = _bridges.extract_ngrams(text, n=4)
        assert len(ngrams) == 2
        assert ngrams[0] == "one two three four"

    def test_short_text(self):
        ngrams = _bridges.extract_ngrams("too short", n=4)
        assert len(ngrams) == 0

    def test_none_input(self):
        assert _bridges.extract_ngrams(None) == []


class TestExtractDomain:
    def test_simple(self):
        assert _bridges._extract_domain("https://example.com/page") == "example.com"

    def test_with_www(self):
        assert _bridges._extract_domain("https://www.example.com/page") == "example.com"

    def test_invalid(self):
        assert _bridges._extract_domain("not a url") is None


# ═══════════════════════════════════════════════════════════════════════
# 18_cross_platform_granger tests
# ═══════════════════════════════════════════════════════════════════════

_granger = importlib.import_module("18_cross_platform_granger")


class TestBuildCrossPlatformWeekly:
    def test_builds_paired_series(self):
        from datetime import datetime
        im = pl.DataFrame({
            "date": [datetime(2015, 6, 1), datetime(2015, 6, 8),
                     datetime(2015, 6, 15)],
            "siege_keyword_score": [1.0, 2.0, 3.0],
            "author_id": [100, 101, 102],
        })
        pol = pl.DataFrame({
            "date": [datetime(2015, 6, 1), datetime(2015, 6, 8),
                     datetime(2015, 6, 15)],
            "siege_keyword_score": [0.5, 1.5, 2.5],
        })
        result = _granger.build_cross_platform_weekly(im, pol)
        assert "im_siege" in result.columns
        assert "pol_siege" in result.columns
        assert result.height > 0


class TestStationarity:
    def test_stationary_series(self):
        np.random.seed(42)
        series = np.random.normal(0, 1, 100)
        result = _granger.test_stationarity(series, "test")
        assert "stationary" in result
        assert bool(result["stationary"]) in (True, False)

    def test_returns_dict_with_keys(self):
        np.random.seed(42)
        series = np.random.normal(0, 1, 50)
        result = _granger.test_stationarity(series, "test")
        assert "adf_statistic" in result
        assert "p_value" in result
        assert "series" in result


class TestComputeCCF:
    def test_ccf_result_structure(self):
        from datetime import datetime, timedelta
        base = datetime(2015, 1, 5)
        dates = [base + timedelta(weeks=i) for i in range(30)]
        paired = pl.DataFrame({
            "week": dates,
            "im_siege": np.random.normal(1.0, 0.3, 30),
            "pol_siege": np.random.normal(0.5, 0.2, 30),
        })
        result = _granger.compute_ccf(paired, max_lag=5)
        assert "peak_lag" in result
        assert "peak_correlation" in result
        assert "ci_95" in result
        assert "lags" in result
        assert "correlations" in result


# ═══════════════════════════════════════════════════════════════════════
# 17_cross_platform_its tests
# ═══════════════════════════════════════════════════════════════════════

_its = importlib.import_module("17_cross_platform_its")


class TestBuildWeeklySeriesPol:
    def test_builds_weekly_with_its_vars(self):
        from datetime import datetime
        dates = [
            datetime(2015, 5, 1), datetime(2015, 5, 8),
            datetime(2015, 6, 5), datetime(2015, 6, 12),
        ]
        df = pl.DataFrame({
            "date": dates,
            "siege_keyword_score": [1.0, 2.0, 3.0, 4.0],
            "siege_binary": [0, 1, 1, 1],
        })
        t0 = datetime(2015, 6, 3)
        result = _its.build_weekly_series_pol(df, t0, "siege_keyword_score")
        assert "week" in result.columns
        assert "post_treatment" in result.columns
        assert "time_centered" in result.columns
        assert result.height > 0
