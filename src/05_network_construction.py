"""05 – Network Construction.

Builds three user-to-user interaction networks:
1. DM network (shared conversations)
2. Forum co-participation network (shared topics)
3. Reputation network (likes given)
"""

from __future__ import annotations

import multiprocessing
import os
from collections import defaultdict
from itertools import combinations

import networkx as nx
import polars as pl

from utils import DATA_PROCESSED, NETWORKS_DIR

_N_WORKERS = min(multiprocessing.cpu_count(), int(os.environ.get("SIEGE_WORKERS", "16")))


def build_dm_network(msg_map: pl.DataFrame) -> nx.Graph:
    """Build undirected DM network: edge between users sharing a conversation."""
    G = nx.Graph()

    # Group users by topic
    topics = (
        msg_map.select(["map_user_id", "map_topic_id"])
        .group_by("map_topic_id")
        .agg(pl.col("map_user_id").alias("users"))
    )

    edge_weights: dict[tuple, int] = defaultdict(int)
    for row in topics.iter_rows(named=True):
        users = row["users"]
        if len(users) < 2:
            continue
        for u1, u2 in combinations(sorted(set(users)), 2):
            if u1 != u2:
                edge_weights[(u1, u2)] += 1

    for (u1, u2), w in edge_weights.items():
        G.add_edge(u1, u2, weight=w)

    print(f"  DM network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def _process_topic_authors(authors: list) -> list[tuple[tuple[int, int], int]]:
    """Compute pairwise edges for a single topic's author list."""
    unique_authors = sorted(set(authors))
    if len(unique_authors) < 2 or len(unique_authors) > 50:
        return []
    edges = []
    for u1, u2 in combinations(unique_authors, 2):
        if u1 != u2:
            edges.append(((u1, u2), 1))
    return edges


def build_forum_coparticipation_network(forum_posts: pl.DataFrame) -> nx.Graph:
    """Build undirected forum co-participation network."""
    G = nx.Graph()

    # Group authors by topic
    topics = (
        forum_posts.select(["author_id", "topic_id"])
        .filter(pl.col("author_id").is_not_null() & pl.col("topic_id").is_not_null())
        .unique()
        .group_by("topic_id")
        .agg(pl.col("author_id").alias("authors"))
    )

    author_lists = [row["authors"] for row in topics.iter_rows(named=True)]
    skipped = sum(1 for a in author_lists if len(set(a)) > 50)
    if skipped:
        print(f"  Skipped {skipped} topics with >50 unique authors")

    print(f"  Processing {len(author_lists)} topics with {_N_WORKERS} workers…")
    with multiprocessing.Pool(_N_WORKERS) as pool:
        all_edges = pool.map(_process_topic_authors, author_lists, chunksize=64)

    edge_weights: dict[tuple, int] = defaultdict(int)
    for topic_edges in all_edges:
        for (u1, u2), w in topic_edges:
            edge_weights[(u1, u2)] += w

    for (u1, u2), w in edge_weights.items():
        G.add_edge(u1, u2, weight=w)

    print(f"  Forum co-participation network: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")
    return G


def build_reputation_network(rep_index: pl.DataFrame) -> nx.DiGraph:
    """Build directed reputation network: liker → liked user."""
    G = nx.DiGraph()

    edges = (
        rep_index.filter(
            pl.col("member_id").is_not_null()
            & pl.col("member_received").is_not_null()
        )
        .group_by(["member_id", "member_received"])
        .agg(pl.len().alias("weight"))
    )

    for row in edges.iter_rows(named=True):
        giver = row["member_id"]
        receiver = row["member_received"]
        if giver != receiver:
            G.add_edge(giver, receiver, weight=row["weight"])

    print(f"  Reputation network: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")
    return G


def save_edgelist(G: nx.Graph | nx.DiGraph, name: str):
    """Save edge list as parquet."""
    edges = []
    directed = isinstance(G, nx.DiGraph)
    for u, v, data in G.edges(data=True):
        edges.append({"source": u, "target": v, "weight": data.get("weight", 1)})

    if not edges:
        print(f"  ⚠ {name}: no edges to save")
        return

    df = pl.DataFrame(edges)
    df.write_parquet(NETWORKS_DIR / f"{name}_edgelist.parquet")
    print(f"  Saved {name}_edgelist.parquet ({df.height:,} edges)")


def main():
    print("=" * 60)
    print("PHASE 3a: Network Construction")
    print("=" * 60)

    # ── DM network ────────────────────────────────────────────────────
    print("\nBuilding DM network…")
    msg_map = pl.read_parquet(DATA_PROCESSED / "core_message_topic_user_map.parquet")
    dm_net = build_dm_network(msg_map)
    save_edgelist(dm_net, "dm")
    nx.write_graphml(dm_net, str(NETWORKS_DIR / "dm_network.graphml"))

    # ── Forum co-participation ────────────────────────────────────────
    print("\nBuilding forum co-participation network…")
    fp = pl.read_parquet(DATA_PROCESSED / "forum_posts.parquet")
    forum_net = build_forum_coparticipation_network(fp)
    save_edgelist(forum_net, "forum")
    nx.write_graphml(forum_net, str(NETWORKS_DIR / "forum_network.graphml"))

    # ── Reputation network ────────────────────────────────────────────
    print("\nBuilding reputation network…")
    rep = pl.read_parquet(DATA_PROCESSED / "core_reputation_index.parquet")
    rep_net = build_reputation_network(rep)
    save_edgelist(rep_net, "reputation")
    nx.write_graphml(rep_net, str(NETWORKS_DIR / "reputation_network.graphml"))

    print("\n✓ Network construction complete.")


if __name__ == "__main__":
    main()
