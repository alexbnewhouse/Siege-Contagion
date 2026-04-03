"""Tests for 05_network_construction.py – network building."""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import polars as pl
import networkx as nx

_mod = importlib.import_module("05_network_construction")
build_dm_network = _mod.build_dm_network
build_forum_coparticipation_network = _mod.build_forum_coparticipation_network
build_reputation_network = _mod.build_reputation_network


class TestBuildDmNetwork:
    def test_creates_edges_for_shared_conversations(self):
        msg_map = pl.DataFrame({
            "map_user_id": [1, 2, 1, 3],
            "map_topic_id": [100, 100, 200, 200],
        })
        G = build_dm_network(msg_map)
        assert G.has_edge(1, 2)
        assert G.has_edge(1, 3)
        assert not G.has_edge(2, 3)

    def test_edge_weight_counts_conversations(self):
        msg_map = pl.DataFrame({
            "map_user_id": [1, 2, 1, 2],
            "map_topic_id": [100, 100, 200, 200],
        })
        G = build_dm_network(msg_map)
        assert G[1][2]["weight"] == 2

    def test_empty_input(self):
        msg_map = pl.DataFrame({
            "map_user_id": pl.Series([], dtype=pl.Int64),
            "map_topic_id": pl.Series([], dtype=pl.Int64),
        })
        G = build_dm_network(msg_map)
        assert G.number_of_nodes() == 0

    def test_single_user_no_edges(self):
        msg_map = pl.DataFrame({
            "map_user_id": [1],
            "map_topic_id": [100],
        })
        G = build_dm_network(msg_map)
        assert G.number_of_edges() == 0


class TestBuildForumCoparticipation:
    def test_creates_edges_for_shared_topics(self):
        fp = pl.DataFrame({
            "author_id": [1, 2, 3, 1],
            "topic_id": [10, 10, 20, 20],
        })
        G = build_forum_coparticipation_network(fp)
        assert G.has_edge(1, 2)
        assert G.has_edge(1, 3)
        assert not G.has_edge(2, 3)


class TestBuildReputationNetwork:
    def test_creates_directed_edges(self):
        rep = pl.DataFrame({
            "member_id": [1, 1, 2],
            "member_received": [2, 3, 1],
        })
        G = build_reputation_network(rep)
        assert isinstance(G, nx.DiGraph)
        assert G.has_edge(1, 2)
        assert G.has_edge(2, 1)

    def test_excludes_self_likes(self):
        rep = pl.DataFrame({
            "member_id": [1, 1],
            "member_received": [1, 2],
        })
        G = build_reputation_network(rep)
        assert not G.has_edge(1, 1)
        assert G.has_edge(1, 2)
