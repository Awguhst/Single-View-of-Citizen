"""Linkage-graph structure: bridges, load-bearing edges, and evidence.

The behaviour worth protecting here is the threshold rule. Bridges must be
computed on the graph that actually formed the cluster - edges at or above
`CLUSTER_MATCH_THRESHOLD` - not on every stored pair. Getting that wrong
makes every cluster look densely cross-confirmed and hides exactly the
single weak link the feature exists to surface.
"""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from app import graph_service
from app.splink_service import CLUSTER_MATCH_THRESHOLD

_BELOW_THRESHOLD = CLUSTER_MATCH_THRESHOLD - 0.2
_ABOVE_THRESHOLD = CLUSTER_MATCH_THRESHOLD + 0.1


def _build_graph_db(db_path, edges: list[tuple[str, str, float]], record_ids: list[str]):
    """A single-cluster database with the given records and scored edges."""
    records = pd.DataFrame(
        [
            {
                "source_record_id": record_id,
                "person_index": 0,
                "agency": "Healthcare",
                "record_type": "HOSPITAL_VISIT",
                "record_date": "2024-01-01",
                "first_name": "Ada",
                "last_name": "Lovelace",
                "date_of_birth": "1990-01-01",
                "email": "ada@example.test",
                "phone": "07000000000",
                "address": "1 Test Street",
                "city": "Testville",
                "postcode": "TE1 1ST",
            }
            for record_id in record_ids
        ]
    )
    clusters = pd.DataFrame(
        [{"source_record_id": r, "master_citizen_id": "C1", "match_probability": 0.99} for r in record_ids]
    )
    edge_rows = pd.DataFrame(
        [
            {
                "source_record_id_l": left,
                "source_record_id_r": right,
                "match_probability": probability,
                "gamma_first_name": 3,
                "gamma_last_name": 3,
                "gamma_date_of_birth": 3,
                "gamma_email": -1,
                "gamma_phone": 1,
                "gamma_address": 2,
                "gamma_postcode": 0,
            }
            for left, right, probability in edges
        ]
    )

    conn = duckdb.connect(str(db_path))
    try:
        for name, frame in (("records", records), ("clusters", clusters), ("linkage_edges", edge_rows)):
            conn.register(f"_{name}", frame)
            conn.execute(f"CREATE TABLE {name} AS SELECT * FROM _{name}")
    finally:
        conn.close()


def test_chain_cluster_reports_every_edge_as_a_bridge(db_path):
    """A -- B -- C: removing either link splits the cluster."""
    _build_graph_db(
        db_path,
        [("R1", "R2", _ABOVE_THRESHOLD), ("R2", "R3", _ABOVE_THRESHOLD)],
        ["R1", "R2", "R3"],
    )
    graph = graph_service.get_cluster_graph("C1")

    assert graph["node_count"] == 3
    assert graph["bridge_count"] == 2
    assert graph["weakest_bridge"] is not None


def test_fully_connected_cluster_has_no_bridges(db_path):
    """A triangle is mutually reinforcing - no single edge holds it together."""
    _build_graph_db(
        db_path,
        [
            ("R1", "R2", _ABOVE_THRESHOLD),
            ("R2", "R3", _ABOVE_THRESHOLD),
            ("R1", "R3", _ABOVE_THRESHOLD),
        ],
        ["R1", "R2", "R3"],
    )
    graph = graph_service.get_cluster_graph("C1")

    assert graph["bridge_count"] == 0
    assert graph["weakest_bridge"] is None
    assert graph["density"] == 1.0


def test_sub_threshold_edges_are_not_treated_as_support(db_path):
    """The regression this file exists for.

    R1--R2--R3 is a chain of load-bearing links. A weak R1--R3 edge was
    scored but rejected by the clustering threshold, so it must not make the
    chain look like a triangle.
    """
    _build_graph_db(
        db_path,
        [
            ("R1", "R2", _ABOVE_THRESHOLD),
            ("R2", "R3", _ABOVE_THRESHOLD),
            ("R1", "R3", _BELOW_THRESHOLD),
        ],
        ["R1", "R2", "R3"],
    )
    graph = graph_service.get_cluster_graph("C1")

    assert graph["edge_count"] == 3
    assert graph["load_bearing_edge_count"] == 2
    assert graph["bridge_count"] == 2, "a rejected edge was counted as structural support"

    weak = next(e for e in graph["edges"] if e["match_probability"] == pytest.approx(_BELOW_THRESHOLD))
    assert weak["is_load_bearing"] is False
    assert weak["is_bridge"] is False


def test_weakest_bridge_is_the_lowest_scoring_one(db_path):
    _build_graph_db(
        db_path,
        [("R1", "R2", 0.99), ("R2", "R3", 0.80)],
        ["R1", "R2", "R3"],
    )
    graph = graph_service.get_cluster_graph("C1")
    assert graph["weakest_bridge"]["match_probability"] == pytest.approx(0.80)


def test_edge_evidence_reports_both_sides_and_agreement(db_path):
    _build_graph_db(db_path, [("R1", "R2", _ABOVE_THRESHOLD)], ["R1", "R2"])
    graph = graph_service.get_cluster_graph("C1")

    evidence = {item["field"]: item for item in graph["edges"][0]["evidence"]}
    # gamma -1 means a value was missing, so no comparison was possible.
    assert evidence["email"]["agreement"] == "not comparable"
    # gamma 0 means the values were compared and disagreed.
    assert evidence["postcode"]["agreement"] == "disagreed"
    assert evidence["first_name"]["agreement"] == "agreed"
    assert evidence["first_name"]["left_value"] == "Ada"
    assert evidence["first_name"]["right_value"] == "Ada"
    assert "postcode" not in graph["edges"][0]["agreeing_fields"]


def test_unknown_citizen_returns_none(db_path):
    _build_graph_db(db_path, [("R1", "R2", _ABOVE_THRESHOLD)], ["R1", "R2"])
    assert graph_service.get_cluster_graph("NOPE") is None


def test_missing_edge_table_raises_a_recoverable_error(db_path):
    """A database linked before edges were persisted still has valid
    clusters, so this must be a clear prompt rather than a crash."""
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute("CREATE TABLE clusters (source_record_id VARCHAR, master_citizen_id VARCHAR)")
        conn.execute("CREATE TABLE records (source_record_id VARCHAR)")
    finally:
        conn.close()

    with pytest.raises(ValueError, match="Re-run"):
        graph_service.get_cluster_graph("C1")
