"""The linkage graph behind a resolved profile.

A `master_citizen_id` is the output of a graph computation: records are
nodes, Splink's scored pairs are edges, and a cluster is a connected
component at the chosen threshold. The rest of the app only ever sees the
final answer - "these eleven records are one citizen" - which is exactly
the part an analyst reviewing a questionable profile cannot check.

This module exposes the working. For one cluster it returns the actual
nodes and edges, the field-level evidence behind each edge, and - most
usefully - which edges are *load-bearing*.

--------------------------------------------------------------------------
Why bridges are the interesting part
--------------------------------------------------------------------------
A cluster held together by many mutually-reinforcing high-probability
edges is safe: no single mistaken comparison could have created it. A
cluster held together by one weak edge is the classic over-merge
signature - two genuinely separate groups of records joined by a single
coincidence, such as two different people who share a surname and a
postcode.

A *bridge* (an edge whose removal disconnects the graph) is precisely that
single point of failure, and networkx computes them directly. Reporting
the weakest bridge turns "this profile looks odd" into "this profile
exists because of this one link, scored 0.78, and here are the two records
it joins" - which is something an analyst can actually adjudicate.

No graph database is involved or needed. The edges are already in DuckDB,
one cluster is at most a few dozen nodes, and networkx is already a
transitive dependency of the existing stack.
"""

from __future__ import annotations

import networkx as nx

from app.data_generator import get_connection
from app.splink_service import CLUSTER_MATCH_THRESHOLD, EDGES_TABLE

# Identity fields Splink compares, in the order the UI should show them.
# Each has a `gamma_<field>` column on the persisted edge list.
_COMPARISON_FIELDS = (
    "first_name",
    "last_name",
    "date_of_birth",
    "email",
    "phone",
    "address",
    "postcode",
)

# Splink's gamma convention: -1 means the comparison could not be made
# (at least one side was NULL), 0 means the values disagreed, and any
# positive value is an agreement level, with higher meaning closer.
_GAMMA_NOT_COMPARABLE = -1
_GAMMA_NO_AGREEMENT = 0


def _agreement_label(gamma: int) -> str:
    if gamma == _GAMMA_NOT_COMPARABLE:
        return "not comparable"
    if gamma == _GAMMA_NO_AGREEMENT:
        return "disagreed"
    return "agreed"


def get_cluster_graph(master_citizen_id: str) -> dict | None:
    """Nodes, edges and structural weak points for one resolved profile.

    Returns None if the profile does not exist. Raises ValueError if the
    pairwise edge list has not been persisted - a database linked before
    edges were stored still has valid clusters, so the caller can surface a
    "re-run linkage" prompt instead of an error.
    """
    conn = get_connection()
    try:
        tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
        if EDGES_TABLE not in tables:
            raise ValueError(
                "No pairwise edges found. Re-run POST /run-linkage to persist them."
            )

        node_rows = conn.execute(
            """
            SELECT r.source_record_id, r.agency, r.record_type, r.record_date,
                   r.first_name, r.last_name, r.date_of_birth,
                   r.email, r.phone, r.address, r.postcode
            FROM clusters c JOIN records r USING (source_record_id)
            WHERE c.master_citizen_id = ?
            ORDER BY r.record_date
            """,
            [master_citizen_id],
        ).fetchall()
        if not node_rows:
            return None
        node_columns = [c[0] for c in conn.description]
        nodes = [dict(zip(node_columns, row)) for row in node_rows]
        by_id = {node["source_record_id"]: node for node in nodes}

        gamma_columns = ", ".join(f"e.gamma_{field}" for field in _COMPARISON_FIELDS)
        edge_rows = conn.execute(
            f"""
            SELECT e.source_record_id_l, e.source_record_id_r, e.match_probability, {gamma_columns}
            FROM {EDGES_TABLE} e
            JOIN clusters cl ON cl.source_record_id = e.source_record_id_l
            JOIN clusters cr ON cr.source_record_id = e.source_record_id_r
            WHERE cl.master_citizen_id = ? AND cr.master_citizen_id = ?
            ORDER BY e.match_probability DESC
            """,
            [master_citizen_id, master_citizen_id],
        ).fetchall()
    finally:
        conn.close()

    edges = []
    for row in edge_rows:
        left_id, right_id, probability = row[0], row[1], float(row[2])
        gammas = dict(zip(_COMPARISON_FIELDS, row[3:]))
        left, right = by_id[left_id], by_id[right_id]
        edges.append(
            {
                "source": left_id,
                "target": right_id,
                "match_probability": round(probability, 6),
                # The actual values on both sides, not just the level, so the
                # evidence is inspectable rather than merely asserted.
                "evidence": [
                    {
                        "field": field,
                        "agreement": _agreement_label(int(gammas[field])),
                        "gamma": int(gammas[field]),
                        "left_value": left.get(field),
                        "right_value": right.get(field),
                    }
                    for field in _COMPARISON_FIELDS
                ],
                "agreeing_fields": [
                    field for field in _COMPARISON_FIELDS if int(gammas[field]) > _GAMMA_NO_AGREEMENT
                ],
            }
        )

    # Structural analysis runs on the graph that actually *created* this
    # cluster - edges at or above the clustering threshold - not on every
    # scored pair. The stored edge list reaches down to
    # `splink_service.PREDICT_MIN_THRESHOLD`, and those sub-threshold pairs
    # were explicitly rejected as evidence. Leaving them in would make every
    # cluster look densely cross-confirmed and would hide the single
    # load-bearing link that a bridge analysis exists to find.
    load_bearing = [e for e in edges if e["match_probability"] >= CLUSTER_MATCH_THRESHOLD]

    graph = nx.Graph()
    graph.add_nodes_from(by_id)
    for edge in load_bearing:
        graph.add_edge(edge["source"], edge["target"], weight=edge["match_probability"])

    bridge_pairs = {frozenset(pair) for pair in nx.bridges(graph)} if graph.number_of_edges() else set()
    for edge in edges:
        edge["is_load_bearing"] = edge["match_probability"] >= CLUSTER_MATCH_THRESHOLD
        edge["is_bridge"] = frozenset((edge["source"], edge["target"])) in bridge_pairs and edge["is_load_bearing"]

    bridges = [edge for edge in edges if edge["is_bridge"]]
    weakest_bridge = min(bridges, key=lambda e: e["match_probability"]) if bridges else None

    return {
        "master_citizen_id": master_citizen_id,
        "nodes": [
            {
                "source_record_id": node["source_record_id"],
                "agency": node["agency"],
                "record_type": node["record_type"],
                "record_date": node["record_date"],
                "name": f"{node['first_name']} {node['last_name']}",
                "date_of_birth": node["date_of_birth"],
                "degree": graph.degree(node["source_record_id"]),
            }
            for node in nodes
        ],
        "edges": edges,
        "node_count": len(nodes),
        "edge_count": len(edges),
        # A fully-connected cluster (every record scored against every other)
        # is the strongest possible evidence; density says how close to that
        # this one is.
        "density": round(nx.density(graph), 4) if len(nodes) > 1 else 0.0,
        "load_bearing_edge_count": len(load_bearing),
        "cluster_threshold": CLUSTER_MATCH_THRESHOLD,
        "bridge_count": len(bridges),
        "weakest_bridge": weakest_bridge,
        "min_edge_probability": round(
            min((e["match_probability"] for e in load_bearing), default=1.0), 6
        ),
    }
