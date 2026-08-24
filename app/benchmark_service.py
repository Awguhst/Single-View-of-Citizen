"""Linkage benchmark: how well does entity resolution hold up as the data gets worse?

`evaluation_service.py` answers "how good is the linkage on *this* dataset".
That is one point on a curve, and a single point cannot distinguish a
strong model from an easy problem. This module sweeps the generator's
`NoiseProfile` from pristine to severe, re-running the whole pipeline at
each level, and scores every run against ground truth - which turns the
headline F1 into a characterisation of where the approach actually starts
to break down.

Why it is built the way it is:

* **Small populations, not the full 10,000.** Each level trains a complete
  Splink model, so the sweep is inherently several pipeline runs. A few
  thousand people per level keeps the whole benchmark to roughly a minute
  while still giving each level tens of thousands of record pairs to score
  on - far more than enough for the metrics to be stable.

* **Nothing touches the live database.** `generate_all(persist=False)`
  returns frames, the linker runs over those frames, and the metrics are
  computed in an in-memory DuckDB connection. Persisting would drop the
  real `clusters`/`citizen_profiles` tables that the running app is serving
  from, so the benchmark deliberately never can.

* **The deterministic baseline is re-scored at every level.** Splink
  beating a hand-written rule on clean data proves very little; the
  interesting question is whether the gap *widens* as the data degrades,
  which is exactly what a probabilistic model should do and what a
  brittle rule cannot.

* **Results are persisted, not recomputed per request.** A sweep is far too
  slow for a page load, so it follows the same POST-to-compute /
  GET-to-read shape the anomaly page already uses.
"""

from __future__ import annotations

import logging
import time

import duckdb
import pandas as pd

from app import evaluation_service
from app.data_generator import NOISE_LEVELS, SEED, generate_all, get_connection
from app.splink_service import CLUSTER_MATCH_THRESHOLD, UNIQUE_ID_COL, _build_and_train_linker

logger = logging.getLogger("citizenlink")

RESULTS_TABLE = "benchmark_results"

# Population size per noise level. Small enough that five full Splink
# trainings finish in about a minute, large enough that each level still
# yields tens of thousands of ground-truth pairs.
DEFAULT_BENCHMARK_POPULATION = 1_500

# The baseline re-scored at every noise level. This is the strongest
# hand-written rule in `evaluation_service._BASELINE_RULES` - comparing
# against the weakest one would flatter the model.
BASELINE_METHOD = "surname_dob_postcode"


def _cluster_records(records_df: pd.DataFrame) -> pd.DataFrame:
    """Run the real linkage pipeline over one benchmark dataset.

    Uses `splink_service._build_and_train_linker` rather than a
    reimplementation, so the benchmark measures the configuration the
    application actually ships. Ground-truth `person_index` is dropped
    before the linker sees the data - the whole point is to resolve
    identity from noisy fields alone.
    """
    linkage_pool = records_df[
        [UNIQUE_ID_COL, "first_name", "last_name", "date_of_birth", "email", "phone", "address", "city", "postcode"]
    ].copy()

    linker = _build_and_train_linker(linkage_pool)
    predictions = linker.inference.predict(threshold_match_probability=0.05)
    clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
        predictions, threshold_match_probability=CLUSTER_MATCH_THRESHOLD
    )
    clustered = clusters.as_pandas_dataframe()[["cluster_id", UNIQUE_ID_COL]]
    return clustered.rename(columns={"cluster_id": "predicted_id"})


def _score_level(level: str, n_people: int, seed: int) -> dict:
    """Generate, link and score one noise level."""
    started = time.perf_counter()
    generated = generate_all(
        seed=seed, noise=NOISE_LEVELS[level], n_people=n_people, persist=False
    )
    partition = _cluster_records(generated.records_df)

    # An in-memory connection so `evaluation_service`'s SQL metrics can run
    # against these frames without any contact with the on-disk database.
    conn = duckdb.connect()
    try:
        conn.register("records", generated.records_df)
        conn.register("_partition", partition)

        splink_metrics = evaluation_service.metrics_for_partition(
            conn,
            f"""
            SELECT p.predicted_id, r.person_index
            FROM _partition p JOIN records r USING ({UNIQUE_ID_COL})
            """,
        )
        baseline_metrics = evaluation_service.metrics_for_partition(
            conn,
            evaluation_service.baseline_partition_sql(BASELINE_METHOD),
        )
    finally:
        conn.close()

    return {
        "noise_level": level,
        "people": generated.people,
        "records": generated.records,
        "splink_precision": splink_metrics.pairwise_precision,
        "splink_recall": splink_metrics.pairwise_recall,
        "splink_f1": splink_metrics.pairwise_f1,
        "splink_exactly_resolved": splink_metrics.exactly_resolved,
        "splink_over_split": splink_metrics.over_split_citizens,
        "splink_over_merged": splink_metrics.over_merged_clusters,
        "baseline_precision": baseline_metrics.pairwise_precision,
        "baseline_recall": baseline_metrics.pairwise_recall,
        "baseline_f1": baseline_metrics.pairwise_f1,
        "f1_advantage": round(splink_metrics.pairwise_f1 - baseline_metrics.pairwise_f1, 6),
        "seconds": round(time.perf_counter() - started, 2),
    }


def run_benchmark(
    n_people: int = DEFAULT_BENCHMARK_POPULATION,
    seed: int = SEED,
    levels: tuple[str, ...] | None = None,
) -> dict:
    """Sweep every noise level, score each, and persist the results."""
    selected = tuple(levels) if levels else tuple(NOISE_LEVELS)
    unknown = [level for level in selected if level not in NOISE_LEVELS]
    if unknown:
        raise ValueError(f"Unknown noise level(s): {', '.join(unknown)}")

    rows = []
    for level in selected:
        logger.info("Benchmarking noise level '%s' (%s people)...", level, n_people)
        rows.append(_score_level(level, n_people, seed))

    results_df = pd.DataFrame(rows)
    results_df["population"] = n_people
    results_df["seed"] = seed

    conn = get_connection()
    try:
        # Same register -> CREATE OR REPLACE TABLE AS SELECT -> unregister
        # pattern used everywhere else in this project.
        conn.register("_benchmark_incoming", results_df)
        conn.execute(f"CREATE OR REPLACE TABLE {RESULTS_TABLE} AS SELECT * FROM _benchmark_incoming")
        conn.unregister("_benchmark_incoming")
    finally:
        conn.close()

    return get_benchmark_results()


def has_benchmark_results() -> bool:
    conn = get_connection()
    try:
        tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
        return RESULTS_TABLE in tables
    finally:
        conn.close()


def get_benchmark_results() -> dict:
    """Read the most recently persisted benchmark sweep."""
    conn = get_connection()
    try:
        tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
        if RESULTS_TABLE not in tables:
            return {
                "population": 0,
                "seed": SEED,
                "baseline_method": BASELINE_METHOD,
                "levels": [],
            }

        df = conn.execute(f"SELECT * FROM {RESULTS_TABLE}").df()
    finally:
        conn.close()

    # Preserve the pristine -> severe ordering regardless of row order on disk.
    order = {level: i for i, level in enumerate(NOISE_LEVELS)}
    df = df.sort_values("noise_level", key=lambda s: s.map(order))

    return {
        "population": int(df["population"].iloc[0]),
        "seed": int(df["seed"].iloc[0]),
        "baseline_method": BASELINE_METHOD,
        "levels": df.drop(columns=["population", "seed"]).to_dict("records"),
    }
