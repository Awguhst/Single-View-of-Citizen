"""Entity-resolution evaluation against the synthetic ground truth.

The rest of this application deliberately never looks at `person_index` -
the ground-truth citizen id carried on every row of `records`. Splink
resolves identities from the noisy comparison columns alone, exactly as a
production system would, and `citizen_service.py` aggregates whatever
clusters come out. That is the right boundary for the *product*.

This module is the one deliberate exception, and it exists because the
data is synthetic: we know who each record really belongs to, so we can
measure how well the linkage actually did instead of asserting it. It is
developer/analyst-facing evaluation only - nothing here feeds a citizen
profile, a recommendation, or any user-facing decision about a person.

--------------------------------------------------------------------------
Why these metrics?
--------------------------------------------------------------------------
Entity resolution has no single natural accuracy number, because the
output is a *partition* of records, not a label per record. Three
complementary views are reported, each answering a different question an
analyst actually asks:

* **Pairwise precision / recall / F1** - over the set of record *pairs*
  placed in the same cluster. This is the standard ER metric and the one
  that degrades gracefully: a cluster that is one record short is
  penalised proportionally rather than counted as a total miss. Precision
  falls when unrelated records are merged; recall falls when one person's
  records are split apart.

* **Cluster-level counts** - how many real citizens were resolved
  *exactly* right, how many were split across several clusters
  (over-splitting), and how many clusters merged more than one real
  citizen (over-merging). Pairwise F1 hides the *shape* of the errors, and
  the two failure modes have very different operational costs: an
  over-split means duplicate outreach and a fragmented profile; an
  over-merge means one citizen's profile showing another citizen's
  records, which is far more serious.

* **Adjusted Rand Index** - chance-corrected agreement between the
  predicted and true partitions. Included because it is the standard
  clustering-quality measure and, unlike raw accuracy, is not flattered by
  the fact that most clusters here are small.

--------------------------------------------------------------------------
Why baselines?
--------------------------------------------------------------------------
A single F1 in isolation says nothing - 0.99 is either excellent or
embarrassing depending on how hard the problem is. Every baseline below is
a deterministic rule someone would genuinely reach for before installing a
probabilistic linkage engine, so the comparison answers the only question
that matters: *is the model earning its complexity?* The two degenerate
baselines (link nothing / link everything) bracket the range and make the
scale interpretable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import duckdb
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from app import cache
from app.data_generator import get_connection
from app.splink_service import CLUSTER_MATCH_THRESHOLD, EDGES_TABLE

# Thresholds swept over the persisted pairwise edges. The lower end sits
# just above `splink_service.PREDICT_MIN_THRESHOLD` (below which edges were
# never stored, so results would silently flatten out) and the upper end
# stops short of 1.0, where every cluster degenerates to a singleton.
_SWEEP_THRESHOLDS = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999)

# Deterministic baseline rules, each expressed as a SQL blocking key over
# the `records` table. Records sharing a non-NULL key are treated as the
# same person; a NULL key means "this rule declines to link this record",
# leaving it a singleton. Ordered weakest-to-strongest so the results table
# reads as a progression from the floor up to the ceiling.
_BASELINE_RULES: dict[str, dict[str, str]] = {
    "no_linkage": {
        "key": "NULL",
        "label": "No linkage (every record is its own citizen)",
        "note": "The 'do nothing' floor: vacuously perfect precision, zero recall.",
    },
    "exact_email": {
        "key": "NULLIF(LOWER(email), '')",
        "label": "Exact email match",
        "note": "A strong signal on its own, but ~12% of records carry no email at all.",
    },
    "exact_name_dob": {
        "key": "LOWER(first_name) || '|' || LOWER(last_name) || '|' || date_of_birth",
        "label": "Exact first name + surname + date of birth",
        "note": "The obvious first attempt; defeated by nicknames, initials and case variants.",
    },
    "surname_dob_postcode": {
        "key": "LOWER(last_name) || '|' || date_of_birth || '|' || LOWER(REPLACE(postcode, ' ', ''))",
        "label": "Surname + date of birth + normalised postcode",
        "note": "The strongest hand-written rule here: sidesteps first-name noise entirely.",
    },
    "all_one_cluster": {
        "key": "'ALL'",
        "label": "Link everything (one single citizen)",
        "note": "The opposite degenerate ceiling: perfect recall, no precision.",
    },
}


@dataclass(frozen=True)
class PartitionMetrics:
    """Quality of one predicted partition of `records` against ground truth."""

    pairwise_precision: float
    pairwise_recall: float
    pairwise_f1: float
    predicted_clusters: int
    true_citizens: int
    exactly_resolved: int
    over_split_citizens: int
    over_merged_clusters: int
    adjusted_rand_index: float

    def as_dict(self) -> dict:
        return asdict(self)


def _contingency_sums(conn: duckdb.DuckDBPyConnection, predicted_sql: str) -> tuple[float, float, float, int]:
    """Return the three "n choose 2" pair sums plus the record count.

    Every ER metric below is a function of these four numbers, which is why
    this stays a pure aggregation and never materialises the pairs
    themselves - the `all_one_cluster` baseline alone would otherwise
    enumerate ~2.8 billion of them.

    Returned as (agreeing_pairs, predicted_pairs, true_pairs, n_records).
    """
    row = conn.execute(
        f"""
        WITH labelled AS ({predicted_sql}),
        agreeing_groups AS (SELECT COUNT(*) AS n FROM labelled GROUP BY predicted_id, person_index),
        pred AS (SELECT COUNT(*) AS n FROM labelled GROUP BY predicted_id),
        truth AS (SELECT COUNT(*) AS n FROM labelled GROUP BY person_index)
        SELECT
            (SELECT COALESCE(SUM(n * (n - 1) / 2.0), 0) FROM agreeing_groups),
            (SELECT COALESCE(SUM(n * (n - 1) / 2.0), 0) FROM pred),
            (SELECT COALESCE(SUM(n * (n - 1) / 2.0), 0) FROM truth),
            (SELECT COUNT(*) FROM labelled)
        """
    ).fetchone()
    return float(row[0]), float(row[1]), float(row[2]), int(row[3])


def _cluster_shape_counts(conn: duckdb.DuckDBPyConnection, predicted_sql: str) -> tuple[int, int, int, int, int]:
    """Return (predicted_clusters, true_citizens, exactly_resolved,
    over_split_citizens, over_merged_clusters).

    `exactly_resolved` counts real citizens whose records all landed in a
    single cluster *and* whose cluster contains nobody else - the only
    outcome that is unambiguously correct. A citizen can be neither
    over-split nor sitting in an over-merged cluster only if both
    conditions hold, so the two error counts do not simply sum to the
    complement.
    """
    row = conn.execute(
        f"""
        WITH labelled AS ({predicted_sql}),
        per_citizen AS (
            SELECT person_index,
                   COUNT(DISTINCT predicted_id) AS n_clusters,
                   ANY_VALUE(predicted_id) AS only_cluster
            FROM labelled GROUP BY person_index
        ),
        per_cluster AS (
            SELECT predicted_id, COUNT(DISTINCT person_index) AS n_citizens
            FROM labelled GROUP BY predicted_id
        )
        SELECT
            (SELECT COUNT(*) FROM per_cluster),
            (SELECT COUNT(*) FROM per_citizen),
            (SELECT COUNT(*) FROM per_citizen pc
                JOIN per_cluster cl ON cl.predicted_id = pc.only_cluster
                WHERE pc.n_clusters = 1 AND cl.n_citizens = 1),
            (SELECT COUNT(*) FROM per_citizen WHERE n_clusters > 1),
            (SELECT COUNT(*) FROM per_cluster WHERE n_citizens > 1)
        """
    ).fetchone()
    return int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4])


def _adjusted_rand_index(agreeing: float, predicted: float, true: float, n_records: int) -> float:
    """Chance-corrected agreement between the predicted and true partitions.

    Computed straight from the contingency pair sums rather than via
    sklearn, so it needs only four scalars instead of the full 75k-row
    label vectors in memory.
    """
    if n_records < 2:
        return 0.0
    total_pairs = n_records * (n_records - 1) / 2.0
    expected = predicted * true / total_pairs
    maximum = (predicted + true) / 2.0
    if maximum == expected:
        return 0.0
    return (agreeing - expected) / (maximum - expected)


def metrics_for_partition(conn: duckdb.DuckDBPyConnection, predicted_sql: str) -> PartitionMetrics:
    """Evaluate one predicted partition against ground truth.

    `predicted_sql` must SELECT exactly two columns - `predicted_id` and
    `person_index` - one row per record.
    """
    agreeing, predicted_pairs, true_pairs, n_records = _contingency_sums(conn, predicted_sql)

    # With zero predicted pairs nothing was ever asserted to match, so
    # precision is vacuously perfect; likewise recall when the ground truth
    # itself contains no duplicate pairs to find.
    precision = agreeing / predicted_pairs if predicted_pairs else 1.0
    recall = agreeing / true_pairs if true_pairs else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    clusters, citizens, exact, over_split, over_merged = _cluster_shape_counts(conn, predicted_sql)

    return PartitionMetrics(
        pairwise_precision=round(precision, 6),
        pairwise_recall=round(recall, 6),
        pairwise_f1=round(f1, 6),
        predicted_clusters=clusters,
        true_citizens=citizens,
        exactly_resolved=exact,
        over_split_citizens=over_split,
        over_merged_clusters=over_merged,
        adjusted_rand_index=round(
            _adjusted_rand_index(agreeing, predicted_pairs, true_pairs, n_records), 6
        ),
    )


_SPLINK_PARTITION_SQL = """
SELECT c.master_citizen_id AS predicted_id, r.person_index
FROM clusters c JOIN records r USING (source_record_id)
"""


def _baseline_partition_sql(key_expression: str) -> str:
    """A baseline's partition: records sharing a non-NULL blocking key form
    one cluster; a NULL key leaves the record a singleton keyed by its own
    id, rather than lumping every NULL together - which would be a bug
    dressed up as a match."""
    return f"""
    SELECT COALESCE(CAST({key_expression} AS VARCHAR), 'SINGLETON:' || source_record_id) AS predicted_id,
           person_index
    FROM records
    """


def baseline_partition_sql(method: str) -> str:
    """The partition SQL for one named baseline in `_BASELINE_RULES`.

    Public so `benchmark_service` can re-score the same baseline against its
    own in-memory `records` frame at each noise level, instead of
    duplicating the rule definition.
    """
    if method not in _BASELINE_RULES:
        raise ValueError(
            f"Unknown baseline '{method}'. Known: {', '.join(_BASELINE_RULES)}"
        )
    return _baseline_partition_sql(_BASELINE_RULES[method]["key"])


def has_ground_truth() -> bool:
    """Ground-truth evaluation needs `records.person_index`, which only the
    synthetic generator can supply. Guarded rather than assumed so the
    evaluation endpoints fail with a clear message instead of a SQL error
    if this is ever pointed at real (unlabelled) data."""
    conn = get_connection()
    try:
        tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
        if "records" not in tables:
            return False
        columns = {row[0] for row in conn.execute("DESCRIBE records").fetchall()}
        return "person_index" in columns
    finally:
        conn.close()


def evaluate_linkage() -> dict:
    """Score the current `clusters` table against ground truth, alongside
    every deterministic baseline - the Evaluation page's headline table.

    Memoised on the cluster/record counts (see `app/cache.py`): six full
    partition scorings over 75,000 records is ~0.5s of SQL that only changes
    when the pipeline re-runs."""
    return cache.memoize("linkage_evaluation", ("clusters", "records"), _evaluate_linkage)


def _evaluate_linkage() -> dict:
    conn = get_connection()
    try:
        tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
        if "clusters" not in tables:
            raise ValueError("No linkage results found. Call POST /run-linkage first.")

        splink = metrics_for_partition(conn, _SPLINK_PARTITION_SQL)

        baselines = [
            {
                "method": name,
                "label": rule["label"],
                "note": rule["note"],
                **metrics_for_partition(conn, _baseline_partition_sql(rule["key"])).as_dict(),
            }
            for name, rule in _BASELINE_RULES.items()
        ]

        total_records = conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        best_baseline = max(baselines, key=lambda b: b["pairwise_f1"])

        return {
            "total_records": total_records,
            "splink": {
                "method": "splink",
                "label": "Splink probabilistic linkage (current pipeline)",
                "note": "Trained per-comparison m/u probabilities, clustered at the configured threshold.",
                **splink.as_dict(),
            },
            "baselines": baselines,
            "best_baseline_method": best_baseline["method"],
            "best_baseline_f1": best_baseline["pairwise_f1"],
            "f1_improvement_over_best_baseline": round(
                splink.pairwise_f1 - best_baseline["pairwise_f1"], 6
            ),
        }
    finally:
        conn.close()


def _connected_components_at(
    record_ids: pd.Series, edges: pd.DataFrame, threshold: float
) -> pd.DataFrame:
    """Re-cluster the persisted edges at `threshold`, returning a
    (source_record_id, predicted_id) frame.

    This is exactly what Splink's own clustering step does - keep every
    edge at or above the threshold and take connected components - so
    sweeping the threshold costs one graph traversal per point instead of
    retraining the model. Records with no surviving edge simply come back
    as their own singleton component, which is the correct behaviour: an
    unlinked record *is* a resolved citizen of one.
    """
    ordinal = pd.Series(np.arange(len(record_ids)), index=record_ids.values)
    kept = edges[edges["match_probability"] >= threshold]

    rows = ordinal.reindex(kept["source_record_id_l"].values).to_numpy()
    cols = ordinal.reindex(kept["source_record_id_r"].values).to_numpy()
    data = np.ones(len(rows), dtype=np.int8)
    n = len(record_ids)

    graph = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    _, labels = connected_components(graph, directed=False)

    return pd.DataFrame({"source_record_id": record_ids.values, "predicted_id": labels})


def sweep_cluster_threshold() -> dict:
    """Evaluate the linkage at a range of clustering thresholds.

    Memoised on the edge/record/cluster counts - see `app/cache.py`. The
    result is a pure function of those tables, and recomputing ~3.3s of
    graph traversal on every page load is not something a small container
    should be asked to do.

    `splink_service.CLUSTER_MATCH_THRESHOLD` is a single hand-picked
    number, and its docstring says it was chosen "empirically" - this is
    the sweep that turns that judgement call into a measurement. Because
    the pairwise edges are persisted, every point on the curve is a graph
    re-clustering rather than a model retrain, so the whole sweep runs in
    seconds.

    The shape of the curve is the useful part: precision rises and recall
    falls as the threshold climbs, and the operating point that maximises
    F1 is rarely exactly where someone guessed.
    """
    return cache.memoize(
        "threshold_sweep", (EDGES_TABLE, "records", "clusters"), _sweep_cluster_threshold
    )


def _sweep_cluster_threshold() -> dict:
    conn = get_connection()
    try:
        tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
        if EDGES_TABLE not in tables:
            raise ValueError(
                "No pairwise edges found. Re-run POST /run-linkage to persist them."
            )

        record_ids = conn.execute("SELECT source_record_id FROM records ORDER BY source_record_id").df()[
            "source_record_id"
        ]
        edges = conn.execute(
            f"SELECT source_record_id_l, source_record_id_r, match_probability FROM {EDGES_TABLE}"
        ).df()

        points = []
        for threshold in _SWEEP_THRESHOLDS:
            partition = _connected_components_at(record_ids, edges, threshold)
            conn.register("_sweep_partition", partition)
            try:
                metrics = metrics_for_partition(
                    conn,
                    """
                    SELECT p.predicted_id, r.person_index
                    FROM _sweep_partition p JOIN records r USING (source_record_id)
                    """,
                )
            finally:
                conn.unregister("_sweep_partition")
            points.append(
                {
                    "threshold": threshold,
                    "is_current": threshold == CLUSTER_MATCH_THRESHOLD,
                    **metrics.as_dict(),
                }
            )

        best = max(points, key=lambda p: p["pairwise_f1"])
        return {
            "current_threshold": CLUSTER_MATCH_THRESHOLD,
            "best_threshold": best["threshold"],
            "best_f1": best["pairwise_f1"],
            "edge_count": len(edges),
            "points": points,
        }
    finally:
        conn.close()
