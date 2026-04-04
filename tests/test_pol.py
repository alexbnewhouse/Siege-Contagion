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


# ═══════════════════════════════════════════════════════════════════════
# 22_shutdown_its tests
# ═══════════════════════════════════════════════════════════════════════

_shutdown = importlib.import_module("22_shutdown_its")


class TestShutdownBuildWeeklySeries:
    def test_builds_weekly_with_treatment(self):
        from datetime import datetime, timezone
        dates = [
            datetime(2017, 11, 1, tzinfo=timezone.utc),
            datetime(2017, 11, 8, tzinfo=timezone.utc),
            datetime(2017, 11, 22, tzinfo=timezone.utc),
            datetime(2017, 11, 29, tzinfo=timezone.utc),
        ]
        df = pl.DataFrame({
            "date": dates,
            "siege_keyword_score": [1.0, 2.0, 3.0, 4.0],
            "siege_binary": [0, 1, 1, 1],
        })
        t0 = datetime(2017, 11, 21, tzinfo=timezone.utc)
        result = _shutdown.build_weekly_series(df, t0, "siege_keyword_score")
        assert "post_treatment" in result.columns
        assert "time_centered" in result.columns
        assert "time_x_post" in result.columns
        assert result.height > 0

    def test_pre_post_split(self):
        from datetime import datetime, timezone
        dates = [datetime(2017, 10, i, tzinfo=timezone.utc) for i in range(1, 29)]
        dates += [datetime(2017, 12, i, tzinfo=timezone.utc) for i in range(1, 29)]
        df = pl.DataFrame({
            "date": dates,
            "siege_keyword_score": np.random.rand(56),
            "siege_binary": np.random.randint(0, 2, 56).tolist(),
        })
        t0 = datetime(2017, 11, 21, tzinfo=timezone.utc)
        result = _shutdown.build_weekly_series(df, t0, "siege_keyword_score")
        pre = result.filter(pl.col("post_treatment") == 0).height
        post = result.filter(pl.col("post_treatment") == 1).height
        assert pre > 0
        assert post > 0


class TestShutdownRunIts:
    def test_returns_coefficients(self):
        from datetime import datetime, timezone
        np.random.seed(42)
        n = 50
        dates = [datetime(2017, 1, 1, tzinfo=timezone.utc)]
        for i in range(1, n):
            dates.append(dates[-1] + __import__("datetime").timedelta(weeks=1))
        df = pl.DataFrame({
            "date": dates,
            "siege_keyword_score": np.random.rand(n),
            "siege_binary": np.random.randint(0, 2, n).tolist(),
        })
        t0 = datetime(2017, 6, 1, tzinfo=timezone.utc)
        weekly = _shutdown.build_weekly_series(df, t0, "siege_keyword_score")
        result = _shutdown.run_its(weekly, "test_label")
        assert "b_level" in result
        assert "b_slope" in result
        assert "p_level" in result
        assert "r_squared" in result


class TestShutdownTreatmentDate:
    def test_t_shutdown_correct(self):
        from datetime import datetime, timezone
        expected = datetime(2017, 11, 21, tzinfo=timezone.utc)
        assert _shutdown.T_SHUTDOWN == expected


# ═══════════════════════════════════════════════════════════════════════
# 23_vocab_adoption_lags tests
# ═══════════════════════════════════════════════════════════════════════

_lags = importlib.import_module("23_vocab_adoption_lags")


class TestLagAcceleration:
    def test_returns_result_keys(self):
        from datetime import datetime
        term_lags = [
            {"im_first": datetime(2015, 1, 1), "lag_days": 100.0},
            {"im_first": datetime(2015, 3, 1), "lag_days": 90.0},
            {"im_first": datetime(2015, 5, 1), "lag_days": 80.0},
            {"im_first": datetime(2015, 7, 1), "lag_days": 70.0},
            {"im_first": datetime(2015, 9, 1), "lag_days": 60.0},
        ]
        result = _lags.test_lag_acceleration(term_lags)
        assert "slope" in result
        assert "p_value" in result
        assert "interpretation" in result

    def test_insufficient_terms(self):
        result = _lags.test_lag_acceleration([{"im_first": None, "lag_days": 1}])
        assert "error" in result


# ═══════════════════════════════════════════════════════════════════════
# 24_transfer_entropy tests
# ═══════════════════════════════════════════════════════════════════════

_te = importlib.import_module("24_transfer_entropy")


class TestDiscretise:
    def test_output_shape(self):
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        result = _te._discretise(x, n_bins=3)
        assert len(result) == 10

    def test_all_same(self):
        x = np.ones(20)
        result = _te._discretise(x, n_bins=5)
        assert len(result) == 20


class TestTransferEntropy:
    def test_zero_for_independent(self):
        np.random.seed(42)
        x = np.random.normal(0, 1, 200)
        y = np.random.normal(0, 1, 200)
        te = _te._transfer_entropy(x, y, lag=1, n_bins=3)
        assert isinstance(te, float)
        # Should be near zero for independent series
        assert te < 0.5

    def test_positive_for_dependent(self):
        np.random.seed(42)
        x = np.random.normal(0, 1, 200)
        # y follows x with lag
        y = np.zeros(200)
        y[1:] = 0.8 * x[:-1] + 0.2 * np.random.normal(0, 1, 199)
        te = _te._transfer_entropy(x, y, lag=1, n_bins=3)
        assert te >= 0.0

    def test_short_series(self):
        te = _te._transfer_entropy(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        assert te == 0.0


class TestTransferEntropySignificance:
    def test_result_structure(self):
        np.random.seed(42)
        x = np.random.normal(0, 1, 100)
        y = np.random.normal(0, 1, 100)
        result = _te.transfer_entropy_with_significance(
            x, y, lag=1, n_bins=3, n_surrogates=20, seed=42
        )
        assert "te" in result
        assert "p_value" in result
        assert "z_score" in result
        assert "significant_05" in result


class TestBuildWeeklyPair:
    def test_builds_aligned_arrays(self):
        from datetime import datetime
        im = pl.DataFrame({
            "date": [datetime(2016, 1, i) for i in range(1, 15)],
            "siege_keyword_score": np.random.rand(14),
            "author_id": [100] * 14,
        })
        pol = pl.DataFrame({
            "date": [datetime(2016, 1, i) for i in range(1, 15)],
            "siege_keyword_score": np.random.rand(14),
        })
        im_arr, pol_arr = _te.build_weekly_pair(im, pol)
        assert len(im_arr) == len(pol_arr)
        assert len(im_arr) > 0


# ═══════════════════════════════════════════════════════════════════════
# 25_country_correlation tests
# ═══════════════════════════════════════════════════════════════════════

_country = importlib.import_module("25_country_correlation")


class TestCountrySummary:
    def test_groups_by_country(self):
        df = pl.DataFrame({
            "poster_country": ["US", "US", "GB", "DE", None],
            "siege_keyword_score": [1.0, 2.0, 3.0, 4.0, 5.0],
            "siege_binary": [1, 1, 0, 1, 0],
        })
        result = _country.country_summary(df)
        assert result.height == 3  # US, GB, DE (null excluded)
        assert "mean_score" in result.columns
        assert "prevalence" in result.columns

    def test_empty_country_excluded(self):
        df = pl.DataFrame({
            "poster_country": ["", "US"],
            "siege_keyword_score": [1.0, 2.0],
            "siege_binary": [0, 1],
        })
        result = _country.country_summary(df)
        assert result.height == 1


class TestImHeavyCountries:
    def test_us_is_im_heavy(self):
        assert "US" in _country.IM_HEAVY_COUNTRIES

    def test_gb_is_im_heavy(self):
        assert "GB" in _country.IM_HEAVY_COUNTRIES

    def test_jp_not_im_heavy(self):
        assert "JP" not in _country.IM_HEAVY_COUNTRIES


class TestImVsRestTest:
    def test_with_sufficient_data(self):
        np.random.seed(42)
        countries = list("US GB CA AU SE FI NO DE FR IT ES PL RU JP BR".split())
        n = len(countries)
        summary = pl.DataFrame({
            "poster_country": countries,
            "n_posts": np.random.randint(100, 10000, n).tolist(),
            "mean_score": np.random.rand(n),
            "prevalence": np.random.rand(n),
        })
        result = _country.im_vs_rest_test(summary)
        assert "u_statistic" in result
        assert "p_value" in result
        assert "im_cluster_n" in result


# ═══════════════════════════════════════════════════════════════════════
# 26_dose_response tests
# ═══════════════════════════════════════════════════════════════════════

_dose = importlib.import_module("26_dose_response")


class TestBuildWeeklyScores:
    def test_joins_platforms(self):
        from datetime import datetime
        im = pl.DataFrame({
            "date": [datetime(2016, 1, i) for i in range(1, 22)],
            "siege_keyword_score": np.random.rand(21),
            "author_id": [100] * 21,
        })
        pol = pl.DataFrame({
            "date": [datetime(2016, 1, i) for i in range(1, 22)],
            "siege_keyword_score": np.random.rand(21),
        })
        result = _dose.build_weekly_scores(im, pol)
        assert "im_score" in result.columns
        assert "pol_score" in result.columns


class TestDoseResponseAnalysis:
    def test_produces_lag_results(self):
        from datetime import datetime, timedelta
        np.random.seed(42)
        n = 60
        base = datetime(2016, 1, 4)
        weeks = [base + timedelta(weeks=i) for i in range(n)]
        weekly = pl.DataFrame({
            "week": weeks,
            "im_score": np.random.rand(n),
            "pol_score": np.random.rand(n),
        })
        result = _dose.dose_response_analysis(weekly, max_lag=4)
        assert "lag_1" in result
        assert "kruskal_h" in result["lag_1"]
        assert "spearman_rho" in result["lag_1"]


# ═══════════════════════════════════════════════════════════════════════
# 27_subtheme_diffusion tests
# ═══════════════════════════════════════════════════════════════════════

_subtheme = importlib.import_module("27_subtheme_diffusion")


class TestSubthemeDefinitions:
    def test_five_subthemes(self):
        assert len(_subtheme.SUBTHEMES) == 5

    def test_accelerationism_patterns(self):
        assert "accelerationism" in _subtheme.SUBTHEMES
        assert len(_subtheme.SUBTHEMES["accelerationism"]) > 0


class TestScoreSubthemes:
    def test_adds_columns(self):
        df = pl.DataFrame({
            "text": [
                "accelerate the collapse",
                "read siege by james mason",
                "atomwaffen skull mask",
                "nothing related here",
            ],
        })
        result = _subtheme.score_subthemes(df)
        assert "st_accelerationism" in result.columns
        assert "st_mason_core" in result.columns
        assert result["st_accelerationism"][0] is True
        assert result["st_mason_core"][1] is True
        assert result["st_atomwaffen_org"][2] is True

    def test_none_text_handled(self):
        df = pl.DataFrame({"text": [None, ""]})
        result = _subtheme.score_subthemes(df)
        assert result.height == 2


class TestCompileSubtheme:
    def test_compiles_and_matches(self):
        pat = _subtheme._compile_subtheme([r"\bsiege\b", r"\bmason\b"])
        assert pat.search("read siege")
        assert pat.search("james mason")
        assert not pat.search("hello world")


# ═══════════════════════════════════════════════════════════════════════
# 28_domain_diffusion tests
# ═══════════════════════════════════════════════════════════════════════

_domain = importlib.import_module("28_domain_diffusion")


class TestDomainExtraction:
    def test_extract_domain(self):
        assert _domain._extract_domain("https://example.com/page") == "example.com"

    def test_extract_domain_www(self):
        assert _domain._extract_domain("https://www.example.com") == "example.com"

    def test_extract_domain_none(self):
        assert _domain._extract_domain("not a url") is None


class TestExtractDomainsWithDates:
    def test_extracts_domains(self):
        from datetime import datetime
        df = pl.DataFrame({
            "text": [
                "Check https://siege-site.example.com/post1",
                "Another https://siege-site.example.com/post2",
            ],
            "date": [datetime(2016, 1, 1), datetime(2016, 1, 2)],
        })
        result = _domain.extract_domains_with_dates(df)
        assert "siege-site.example.com" in result
        # Should keep earliest date
        assert result["siege-site.example.com"] == datetime(2016, 1, 1)

    def test_noise_filtered(self):
        from datetime import datetime
        df = pl.DataFrame({
            "text": ["Visit https://youtube.com/watch?v=123"],
            "date": [datetime(2016, 1, 1)],
        })
        result = _domain.extract_domains_with_dates(df)
        assert len(result) == 0


class TestDomainTemporalPriority:
    def test_counts_correctly(self):
        from datetime import datetime
        im = {
            "a.com": datetime(2015, 1, 1),
            "b.com": datetime(2016, 6, 1),
            "c.com": datetime(2016, 1, 1),
        }
        pol = {
            "a.com": datetime(2016, 1, 1),  # IM first
            "b.com": datetime(2015, 1, 1),  # pol first
            "d.com": datetime(2016, 1, 1),  # no IM match
        }
        result = _domain.domain_temporal_priority(im, pol)
        assert result["shared_domains"] == 2
        assert result["im_first"] == 1
        assert result["pol_first"] == 1
