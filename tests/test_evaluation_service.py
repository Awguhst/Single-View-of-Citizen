"""Ground-truth evaluation metrics.

Every expected value here is derived by hand in `conftest._FIXTURE_ROWS`
rather than captured from a previous run, so a regression in the SQL shows
up as a failure instead of a quietly updated snapshot.
"""

from __future__ import annotations

import duckdb
import pytest

from app import evaluation_service
from tests.conftest import EXPECTED_F1, EXPECTED_PRECISION, EXPECTED_RECALL

_PARTITION_SQL = """
SELECT c.master_citizen_id AS predicted_id, r.person_index
FROM clusters c JOIN records r USING (source_record_id)
"""


def test_pairwise_metrics_match_hand_computed_values(tiny_linked_db):
    conn = duckdb.connect(str(tiny_linked_db))
    try:
        metrics = evaluation_service.metrics_for_partition(conn, _PARTITION_SQL)
    finally:
        conn.close()

    assert metrics.pairwise_precision == pytest.approx(EXPECTED_PRECISION, abs=1e-6)
    assert metrics.pairwise_recall == pytest.approx(EXPECTED_RECALL, abs=1e-6)
    assert metrics.pairwise_f1 == pytest.approx(EXPECTED_F1, abs=1e-6)


def test_cluster_shape_counts_separate_the_two_failure_modes(tiny_linked_db):
    conn = duckdb.connect(str(tiny_linked_db))
    try:
        metrics = evaluation_service.metrics_for_partition(conn, _PARTITION_SQL)
    finally:
        conn.close()

    assert metrics.predicted_clusters == 4
    assert metrics.true_citizens == 4
    # Only person 0 lands in exactly one cluster that contains nobody else.
    assert metrics.exactly_resolved == 1
    assert metrics.over_split_citizens == 1  # person 1, across C2 and C3
    assert metrics.over_merged_clusters == 1  # C4, holding persons 2 and 3


def test_perfect_partition_scores_one(tiny_linked_db):
    """A partition equal to the ground truth must score 1.0 on everything -
    the sanity check that the metric is not systematically off."""
    conn = duckdb.connect(str(tiny_linked_db))
    try:
        metrics = evaluation_service.metrics_for_partition(
            conn, "SELECT CAST(person_index AS VARCHAR) AS predicted_id, person_index FROM records"
        )
    finally:
        conn.close()

    assert metrics.pairwise_precision == 1.0
    assert metrics.pairwise_recall == 1.0
    assert metrics.pairwise_f1 == 1.0
    assert metrics.adjusted_rand_index == 1.0
    assert metrics.exactly_resolved == metrics.true_citizens
    assert metrics.over_split_citizens == 0
    assert metrics.over_merged_clusters == 0


def test_degenerate_partitions_bracket_the_scale(tiny_linked_db):
    """Linking nothing and linking everything are the two extremes the
    baseline table uses to make F1 interpretable."""
    conn = duckdb.connect(str(tiny_linked_db))
    try:
        nothing = evaluation_service.metrics_for_partition(
            conn, "SELECT source_record_id AS predicted_id, person_index FROM records"
        )
        everything = evaluation_service.metrics_for_partition(
            conn, "SELECT 'ALL' AS predicted_id, person_index FROM records"
        )
    finally:
        conn.close()

    # Nothing linked: no pair was ever asserted, so precision is vacuously
    # perfect and recall is zero.
    assert nothing.pairwise_precision == 1.0
    assert nothing.pairwise_recall == 0.0
    assert nothing.over_merged_clusters == 0

    # Everything linked: every true pair is found, but almost nothing else is right.
    assert everything.pairwise_recall == 1.0
    assert everything.pairwise_precision < 0.5
    assert everything.predicted_clusters == 1

    # Neither degenerate extreme may score well overall.
    assert nothing.adjusted_rand_index == pytest.approx(0.0, abs=1e-9)
    assert everything.adjusted_rand_index == pytest.approx(0.0, abs=1e-9)


def test_baseline_partition_sql_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown baseline"):
        evaluation_service.baseline_partition_sql("not_a_real_baseline")


def test_null_blocking_key_never_merges_records(tiny_linked_db):
    """A baseline that declines to link a record must leave it a singleton,
    not lump every NULL-keyed record into one giant cluster."""
    conn = duckdb.connect(str(tiny_linked_db))
    try:
        conn.execute("UPDATE records SET email = NULL")
        metrics = evaluation_service.metrics_for_partition(
            conn, evaluation_service.baseline_partition_sql("exact_email")
        )
    finally:
        conn.close()

    assert metrics.predicted_clusters == 8  # one per record
    assert metrics.over_merged_clusters == 0
    assert metrics.pairwise_recall == 0.0


def test_has_ground_truth_requires_person_index(db_path):
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute("CREATE TABLE records (source_record_id VARCHAR, first_name VARCHAR)")
    finally:
        conn.close()

    assert evaluation_service.has_ground_truth() is False
