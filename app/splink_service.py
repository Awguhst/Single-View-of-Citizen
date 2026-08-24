"""Splink-powered entity-resolution service.

This module configures and runs the probabilistic record-linkage model
that turns tens of thousands of noisy records - tax filings, benefit
claims, healthcare registrations, licences, and every other government
record type alike - into a set of `master_citizen_id` clusters, one per
real individual. No agency's record is trusted via a clean ground-truth
index; every one carries the same noisy identity fields and must be
resolved the same way.

--------------------------------------------------------------------------
Why these Splink configuration choices?
--------------------------------------------------------------------------
* **link_type="dedupe_only"**: every noisy record - across all twelve
  record types - lives in a single pool (there is no second "dataset" to
  link against) - we are deduplicating one table, not linking two separate
  tables. `_load_linkage_pool()` is what reads every noisy record from the
  unified `records` table before Splink ever sees it; everything below
  this point operates purely on column names it doesn't know or care
  which agency or record_type a row belongs to.

* **DuckDBAPI backend**: Splink's in-process DuckDB engine is fast enough
  for tens of thousands of rows, requires no external services, and
  keeps the whole POC self-contained (per the brief).

* **Comparisons** map 1:1 onto the seven attributes specified in the
  brief, using Splink's purpose-built comparison templates instead of
  hand-rolled SQL:
    - `NameComparison` for first/last name: handles exact match, a
      Jaro-Winkler tier for typos/case, and a phonetic (double-metaphone)
      fallback - exactly what's needed for "Jon" vs "Jonathan" or a
      transposed-letter surname.
    - `DateOfBirthComparison` parses the ISO date string and scores by
      calendar distance, so a day/month transcription error still scores
      far higher than an unrelated DOB.
    - `EmailComparison` understands the username@domain structure and
      rewards a matching username even when the domain differs (or
      vice-versa) - exactly the "john.smith@gmail.com" vs
      "johnsmith@gmail.com" pattern called out in the brief.
    - `LevenshteinAtThresholds` for phone and address: both are
      formatting-sensitive free-text fields (punctuation, abbreviations)
      where edit distance is a robust, cheap similarity signal.
    - `PostcodeComparison` with term-frequency adjustment: postcodes are
      strong identity signals, but adjustment stops common
      postcode districts from over-crediting unrelated matches.

* **Blocking rules** exist purely for performance: comparing all
  ~25,000^2/2 pairs would be wasteful. Each rule captures a different
  realistic "this is probably the same person" signal (shared email,
  shared phone, name+surname, DOB+postcode, surname+postcode, or
  first-initial+surname+DOB to catch "J Smith" style records), and
  Splink takes the union of all rules' candidate pairs before scoring.

* **Two-stage probability estimation**: a handful of high-precision
  deterministic rules give a sensible starting estimate of how rare a
  random match is, `estimate_u_using_random_sampling` learns how often
  attributes agree *by chance*, and `estimate_parameters_using_expectation_maximisation`
  is run against several different blocking rules so every comparison
  gets trained on data it wasn't blocked on (a standard Splink
  identifiability requirement).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import duckdb
import pandas as pd
from splink import DuckDBAPI, Linker, SettingsCreator, block_on
import splink.comparison_library as cl

from app.data_generator import get_connection

UNIQUE_ID_COL = "source_record_id"

# Predictions below this are dropped before clustering purely to keep the
# pairwise predictions table small; the real decision threshold is
# CLUSTER_MATCH_THRESHOLD below.
PREDICT_MIN_THRESHOLD = 0.05

# A pairwise match probability at/above this becomes a graph edge during
# clustering; two records end up in the same master_citizen_id cluster iff
# they are connected (directly or transitively) by edges at-or-above this
# threshold. 0.75 was chosen empirically on this synthetic dataset: it is
# comfortably above the "two random people happen to share a surname"
# noise floor while still bridging the heavier name/address noise we
# inject (e.g. initial-only first name + abbreviated address).
CLUSTER_MATCH_THRESHOLD = 0.75

EM_SEED = 42


def _settings() -> SettingsCreator:
    return SettingsCreator(
        link_type="dedupe_only",
        unique_id_column_name=UNIQUE_ID_COL,
        comparisons=[
            cl.NameComparison("first_name"),
            cl.NameComparison("last_name").configure(term_frequency_adjustments=True),
            cl.DateOfBirthComparison(
                "date_of_birth",
                input_is_string=True,
                datetime_format="%Y-%m-%d",
            ),
            cl.EmailComparison("email"),
            cl.LevenshteinAtThresholds("phone", [1, 2]),
            cl.LevenshteinAtThresholds("address", [2, 5, 10]),
            cl.PostcodeComparison("postcode").configure(term_frequency_adjustments=True),
        ],
        blocking_rules_to_generate_predictions=[
            block_on("first_name", "last_name"),
            block_on("email"),
            block_on("phone"),
            block_on("date_of_birth", "postcode"),
            block_on("last_name", "postcode"),
            block_on("substr(first_name,1,1)", "last_name", "date_of_birth"),
        ],
        retain_intermediate_calculation_columns=True,
    )


def _deterministic_rules() -> list[str]:
    """High-precision rules used only to seed the prior probability that two
    random records match. None of these alone decides a final cluster -
    that is what the trained model + clustering threshold do."""
    return [
        "l.email = r.email and l.email is not null",
        "l.phone = r.phone and l.phone is not null and l.last_name = r.last_name",
        "l.date_of_birth = r.date_of_birth and l.postcode = r.postcode and l.last_name = r.last_name",
    ]


def _load_linkage_pool(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load every record Splink should resolve together: every row of the
    unified `records` table (all twelve government record types alike),
    projected to just the columns Splink actually uses (the unique id plus
    the 7 noisy comparison columns). `record_type` and the type-specific
    payload columns (`agency_reference_id`/`status`/`amount`/
    `provider_name`/`detail`) are deliberately left out of this projection
    - Splink never reads them, and their meaning varies by record_type
    (an NHS number is not a passport number) enough that exposing them to
    the comparison model would be misleading rather than useful."""
    return conn.execute(
        """
        SELECT source_record_id, person_index, agency,
               first_name, last_name, date_of_birth, email, phone, address, city, postcode
        FROM records
        """
    ).df()


def _build_and_train_linker(df: pd.DataFrame) -> Linker:
    linker = Linker(df, _settings(), db_api=DuckDBAPI())

    linker.training.estimate_probability_two_random_records_match(
        _deterministic_rules(), recall=0.6
    )
    linker.training.estimate_u_using_random_sampling(max_pairs=2e6, seed=EM_SEED)

    # Train m-probabilities against several different blocking rules so
    # every comparison vector level gets observed outside the rule it was
    # blocked on (Splink's standard EM identifiability pattern).
    linker.training.estimate_parameters_using_expectation_maximisation(
        block_on("first_name", "last_name")
    )
    linker.training.estimate_parameters_using_expectation_maximisation(
        block_on("date_of_birth")
    )
    linker.training.estimate_parameters_using_expectation_maximisation(
        block_on("postcode")
    )
    return linker


def _assign_master_citizen_ids(clusters_df: pd.DataFrame) -> pd.DataFrame:
    """Map Splink's internal `cluster_id` to a stable, human-friendly
    `master_citizen_id` (e.g. "MC00001"), ordered by cluster_id for full
    reproducibility given a fixed seed."""
    distinct_clusters = sorted(clusters_df["cluster_id"].unique())
    id_map = {cid: f"MC{i + 1:05d}" for i, cid in enumerate(distinct_clusters)}
    clusters_df = clusters_df.copy()
    clusters_df["master_citizen_id"] = clusters_df["cluster_id"].map(id_map)
    return clusters_df


def _per_record_match_probability(
    clusters_df: pd.DataFrame, predictions_df: pd.DataFrame
) -> pd.Series:
    """Splink scores pairwise edges, not individual records, so we derive a
    per-record confidence as the strongest edge connecting that record to
    another member of its own cluster. Records in a singleton cluster (no
    duplicates found) have nothing to compare against, so they default to
    1.0 - there is no ambiguity about which person they are."""
    cluster_of = clusters_df.set_index(UNIQUE_ID_COL)["cluster_id"]

    within_cluster = predictions_df[
        cluster_of.reindex(predictions_df[f"{UNIQUE_ID_COL}_l"]).values
        == cluster_of.reindex(predictions_df[f"{UNIQUE_ID_COL}_r"]).values
    ]

    best_l = within_cluster.groupby(f"{UNIQUE_ID_COL}_l")["match_probability"].max()
    best_r = within_cluster.groupby(f"{UNIQUE_ID_COL}_r")["match_probability"].max()
    best = pd.concat([best_l, best_r]).groupby(level=0).max()

    return clusters_df[UNIQUE_ID_COL].map(best).fillna(1.0)


# Splink's pairwise scores are the evidence *behind* every cluster: which
# specific records it decided were the same person, how strongly, and which
# comparisons drove that. Clustering collapses them into a single id, so
# without persisting them the reasoning is lost the moment the pipeline
# finishes. Keeping them is what lets the app answer three questions it
# otherwise cannot:
#   1. "What would happen at a different clustering threshold?" - the
#      threshold sweep in `evaluation_service` re-clusters these edges,
#      with no need to retrain the model.
#   2. "Why are these records on one profile?" - `graph_service` walks the
#      edges within a cluster to show the actual chain of evidence.
#   3. "Which link is holding this questionable cluster together?" - a
#      single weak bridge edge is the classic over-merge signature.
EDGES_TABLE = "linkage_edges"


def _edges_frame(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Project Splink's prediction output down to a persistable edge list.

    Keeps the two record ids, the score, and the per-comparison `gamma_*`
    agreement levels - the small integer "how well did this field agree"
    codes that are the human-readable reason a pair scored as it did. The
    much wider `bf_*`/`tf_*` intermediate columns are dropped: they are
    derivable from the gammas plus the trained model and would multiply the
    stored width for no additional explanatory value.
    """
    id_columns = [f"{UNIQUE_ID_COL}_l", f"{UNIQUE_ID_COL}_r"]
    gamma_columns = [c for c in predictions_df.columns if c.startswith("gamma_")]
    return predictions_df[id_columns + ["match_probability"] + gamma_columns].copy()


def has_linkage_edges() -> bool:
    """Whether the persisted pairwise edge list is available.

    Separate from `has_run_linkage()` because a database produced before
    edges were persisted still has valid `clusters` - the edge-backed
    features degrade with a "re-run linkage" prompt rather than erroring.
    """
    conn = get_connection()
    try:
        tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
        return EDGES_TABLE in tables
    finally:
        conn.close()


@dataclass
class LinkageResult:
    clusters: int
    duplicates_found: int
    avg_match_probability: float
    training_seconds: float


def run_full_pipeline() -> LinkageResult:
    """Run the end-to-end Splink pipeline and persist results to DuckDB.

    Writes two tables. `clusters` (source_record_id, master_citizen_id,
    match_probability) is the resolved identity the rest of the app builds
    on; `linkage_edges` is the pairwise evidence behind it (see
    `EDGES_TABLE` above). Every noisy record in the linkage pool - every row
    of the unified `records` table, across all twelve government record
    types - is resolved into this same cluster space; every record type
    attaches to a profile the same way, via `clusters`, in
    `citizen_service.py`. There is no separate clean-index bridge any more.
    """
    start = time.perf_counter()
    conn = get_connection()
    try:
        source_df = _load_linkage_pool(conn)
        if source_df.empty:
            raise RuntimeError("No source records found - call /generate-data first.")

        linker = _build_and_train_linker(source_df)

        predictions = linker.inference.predict(threshold_match_probability=PREDICT_MIN_THRESHOLD)
        predictions_df = predictions.as_pandas_dataframe()

        clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
            predictions, threshold_match_probability=CLUSTER_MATCH_THRESHOLD
        )
        clusters_df = clusters.as_pandas_dataframe()[["cluster_id", UNIQUE_ID_COL]]

        clusters_df = _assign_master_citizen_ids(clusters_df)
        clusters_df["match_probability"] = _per_record_match_probability(clusters_df, predictions_df)

        result_df = clusters_df[[UNIQUE_ID_COL, "master_citizen_id", "match_probability"]]

        _persist(conn, "clusters", result_df)
        _persist(conn, EDGES_TABLE, _edges_frame(predictions_df))

        n_records = len(source_df)
        n_clusters = result_df["master_citizen_id"].nunique()
        avg_prob = float(result_df["match_probability"].mean())
    finally:
        conn.close()

    return LinkageResult(
        clusters=n_clusters,
        duplicates_found=n_records - n_clusters,
        avg_match_probability=round(avg_prob, 6),
        training_seconds=round(time.perf_counter() - start, 2),
    )


def _persist(conn: duckdb.DuckDBPyConnection, table_name: str, df: pd.DataFrame) -> None:
    view_name = f"_{table_name}_incoming"
    conn.register(view_name, df)
    conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM {view_name}")
    conn.unregister(view_name)


def has_run_linkage() -> bool:
    conn = get_connection()
    try:
        tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
        return "clusters" in tables
    finally:
        conn.close()
