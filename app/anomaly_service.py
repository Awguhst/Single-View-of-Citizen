"""Anomaly detection over resolved citizen profiles.

Feeds the Review Queue page: ranks profiles whose
*linkage/data pattern* is statistically unusual - an unusually high or low
number of linked records/agencies relative to the resolved population, an
unusual amount of internal disagreement between the records that were
merged, or an odd combination of government service-usage counts. This is
the same job the old "lowest match confidence" review queue tried to do,
but with a signal that actually varies: confidence sits at effectively 1.0
for the whole population here, so ranking on it ranked on noise. This page
replaced it.

--------------------------------------------------------------------------
What this page is actually detecting
--------------------------------------------------------------------------
Worth stating plainly, because it reframes the whole feature: on this
dataset an "anomalous profile" is overwhelmingly a *linkage artefact*, not
an unusual citizen. The generator caps every citizen at six agencies, so a
profile showing seven or more is structurally impossible and can only mean
two different people were merged into one cluster. `evaluate_detectors()`
below confirms this against ground truth rather than assuming it.

That is why the feature set includes the three `distinct_*` identity-strain
counts. A cluster holding four different dates of birth is not describing a
citizen with four birthdays; it is describing a merge that should not have
happened. These are the signals that make the page useful to an analyst
reviewing the linkage, and they are what the old feature set was missing.

--------------------------------------------------------------------------
Why `confidence_score` was removed from the feature set
--------------------------------------------------------------------------
It was measured at standard deviation 8e-6 across the whole population -
effectively the constant 1.0. A constant column contributes nothing to an
Isolation Forest (no split can ever separate on it) while appearing in the
UI as though it were evidence. It is still shown on the results table as
context, but it is no longer an input to any model.

--------------------------------------------------------------------------
Design boundary (unchanged, and worth restating)
--------------------------------------------------------------------------
Every feature below is structural: counts, linkage shape, and internal
record agreement. The model deliberately never sees `age`,
`marital_status`, or anything derived from them. Scoring or flagging a
citizen using demographic/lifestyle proxies is a well-documented source of
discriminatory harm, and that holds even for synthetic data, since the
scoring pattern itself - not the underlying data - is the reusable
artifact. An "anomaly score" is not exempt from that policy just because it
isn't named a risk score. If you extend this feature, keep any new input
structural (counts, confidence, linkage shape), never demographic or
lifestyle-derived.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from app import cache
from app.data_generator import SEED, get_connection

DEFAULT_CONTAMINATION = 0.05

# Structural/linkage signals only - see the design-boundary note above.
# The `distinct_*` columns count how much the records merged into one
# profile actually disagree with each other about the citizen's identity;
# they are computed from the linked records rather than the profile, which
# is why this reads from `clusters`/`records` rather than `citizen_profiles`
# alone.
_FEATURE_QUERY = """
WITH linked AS (
    SELECT c.master_citizen_id, r.first_name, r.last_name, r.date_of_birth, r.postcode
    FROM clusters c JOIN records r USING (source_record_id)
),
strain AS (
    SELECT
        master_citizen_id,
        COUNT(DISTINCT LOWER(first_name)) AS distinct_first_names,
        COUNT(DISTINCT date_of_birth) AS distinct_dobs,
        COUNT(DISTINCT LOWER(REPLACE(postcode, ' ', ''))) AS distinct_postcodes
    FROM linked
    GROUP BY 1
)
SELECT
    cp.master_citizen_id,
    cp.record_count,
    cp.agency_count,
    cp.record_count / NULLIF(cp.agency_count, 0) AS records_per_agency,
    s.distinct_first_names,
    s.distinct_dobs,
    s.distinct_postcodes,
    cp.hospital_visits,
    len(cp.benefits_received) AS benefits_count,
    len(cp.current_prescriptions) AS prescriptions_count,
    len(cp.education_history) AS education_count
FROM citizen_profiles cp
JOIN strain s USING (master_citizen_id)
-- Not cosmetic: IsolationForest subsamples its training rows, so the *order*
-- of this result set changes which rows each tree sees and therefore which
-- profiles get flagged. Without an explicit sort DuckDB returns hash-join
-- order, which differs between database files - so rebuilding the database
-- from the same seed produced a different review queue (measured: 501 vs 486
-- flagged on byte-identical clusters). Sorting makes the model's input fully
-- determined by the data, per this project's "every random choice is seeded"
-- policy.
ORDER BY master_citizen_id
"""

_FEATURE_COLUMNS = [
    "record_count",
    "agency_count",
    "records_per_agency",
    "distinct_first_names",
    "distinct_dobs",
    "distinct_postcodes",
    "hospital_visits",
    "benefits_count",
    "prescriptions_count",
    "education_count",
]

# Plain-language names for the results UI, so an explanation reads as a
# sentence rather than a column name.
FEATURE_LABELS = {
    "record_count": "linked records",
    "agency_count": "distinct agencies",
    "records_per_agency": "records per agency",
    "distinct_first_names": "different first-name spellings",
    "distinct_dobs": "different dates of birth",
    "distinct_postcodes": "different postcodes",
    "hospital_visits": "hospital visits",
    "benefits_count": "benefit types",
    "prescriptions_count": "prescriptions",
    "education_count": "qualifications",
}

# How many contributing features to keep per flagged profile. Three is
# enough to explain a flag in a sentence without turning the table into a
# feature dump.
TOP_FACTORS_PER_PROFILE = 3

# Neighbourhood size for the Local Outlier Factor comparison in
# `evaluate_detectors()`. Well above sklearn's default of 20 because these
# features are low-cardinality integers with many exact duplicates - see the
# comment at the call site.
LOF_NEIGHBORS = 50

_SCORE_BUCKET_LABELS = ["0-20", "20-40", "40-60", "60-80", "80-100"]


def has_anomaly_results() -> bool:
    conn = get_connection()
    try:
        tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
        return "anomaly_results" in tables
    finally:
        conn.close()


def _load_features() -> pd.DataFrame:
    conn = get_connection()
    try:
        return conn.execute(_FEATURE_QUERY).fetchdf()
    finally:
        conn.close()


def _normalize(raw_scores: np.ndarray) -> np.ndarray:
    """Min-max a "higher is more anomalous" raw score onto 0-100."""
    lo, hi = raw_scores.min(), raw_scores.max()
    if hi <= lo:
        return np.zeros_like(raw_scores)
    return (raw_scores - lo) / (hi - lo) * 100


def _attribute_scores(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Per-feature attribution for the fitted Isolation Forest.

    An Isolation Forest score is not additively decomposable, so there is no
    exact per-feature split of it. What *is* exact is a counterfactual: for
    each feature, re-score every profile with that one feature replaced by
    the population median and measure how much the anomaly score falls. A
    large drop means "this profile looks much more normal once we ignore
    this feature" - which is precisely the claim the UI wants to make.

    This is a genuine ablation against the actual fitted model rather than a
    proxy statistic, and it costs only one extra `decision_function` call
    per feature (ten passes over ~10k rows - milliseconds).

    Returns an (n_samples, n_features) array of score drops, in the units of
    the raw inverted decision function.
    """
    baseline = -model.decision_function(X)
    medians = np.median(X, axis=0)

    contributions = np.zeros_like(X, dtype=float)
    for j in range(X.shape[1]):
        neutralised = X.copy()
        neutralised[:, j] = medians[j]
        contributions[:, j] = baseline - (-model.decision_function(neutralised))
    return contributions


def _top_factors(
    contributions: np.ndarray, features: pd.DataFrame, population_medians: pd.Series
) -> list[list[dict]]:
    """Turn the attribution matrix into the top few human-readable reasons
    per profile, each citing the profile's own value and how it compares to
    the rest of the population."""
    out: list[list[dict]] = []
    for row_index in range(contributions.shape[0]):
        row = contributions[row_index]
        ranked = np.argsort(row)[::-1][:TOP_FACTORS_PER_PROFILE]
        factors = []
        for j in ranked:
            if row[j] <= 0:
                continue  # This feature made the profile look more normal, not less.
            column = _FEATURE_COLUMNS[j]
            value = float(features.iloc[row_index][column])
            median = float(population_medians[column])
            factors.append(
                {
                    "feature": column,
                    "label": FEATURE_LABELS[column],
                    "value": round(value, 2),
                    "population_median": round(median, 2),
                    "direction": "above" if value > median else ("below" if value < median else "at"),
                    "contribution": round(float(row[j]), 6),
                }
            )
        out.append(factors)
    return out


def run_anomaly_detection(contamination: float = DEFAULT_CONTAMINATION) -> dict:
    """(Re)fit an Isolation Forest over every resolved citizen profile's
    structural/linkage features and persist the result, together with a
    per-profile explanation of which features drove each score.
    """
    df = _load_features()
    if df.empty:
        raise ValueError("No citizen profiles found. Call POST /run-linkage first.")

    X = df[_FEATURE_COLUMNS].to_numpy(dtype=float)

    # random_state=SEED matches this project's "every random choice is
    # seeded" reproducibility policy (see data_generator.py's SEED).
    model = IsolationForest(contamination=contamination, random_state=SEED)
    model.fit(X)

    # decision_function() is higher for inliers, lower (often negative) for
    # outliers - inverted and min-max normalized to a 0-100 scale so higher
    # always reads as "more anomalous" in the UI.
    df["anomaly_score"] = _normalize(-model.decision_function(X))
    df["is_anomaly"] = model.predict(X) == -1
    df["contamination"] = contamination

    contributions = _attribute_scores(model, X)
    medians = df[_FEATURE_COLUMNS].median()
    df["top_factors"] = _top_factors(contributions, df, medians)

    result_df = df[["master_citizen_id", "anomaly_score", "is_anomaly", "contamination", "top_factors"]]

    conn = get_connection()
    try:
        # Same register -> CREATE OR REPLACE TABLE AS SELECT -> unregister
        # pattern used by data_generator._persist().
        conn.register("anomaly_results_view", result_df)
        conn.execute("CREATE OR REPLACE TABLE anomaly_results AS SELECT * FROM anomaly_results_view")
        conn.unregister("anomaly_results_view")
    finally:
        conn.close()

    return get_anomaly_summary()


def _score_distribution(scores: list[float]) -> list[dict]:
    counts = [0] * len(_SCORE_BUCKET_LABELS)
    for score in scores:
        bucket = min(int(score // 20), len(_SCORE_BUCKET_LABELS) - 1)
        counts[bucket] += 1
    return [{"label": label, "count": count} for label, count in zip(_SCORE_BUCKET_LABELS, counts)]


def _empty_summary() -> dict:
    return {
        "total_profiles_analyzed": 0,
        "anomalies_detected": 0,
        "normal_count": 0,
        "pct_anomalous": 0.0,
        "contamination": DEFAULT_CONTAMINATION,
        "score_distribution": _score_distribution([]),
        "results": [],
        "total": 0,
    }


# Identity fields whose disagreement inside a single cluster is concrete,
# reviewable evidence. A reviewer cannot act on "anomaly score 87", but they
# can act on "these records claim two different dates of birth" - so the
# review queue carries the actual conflicting values alongside the score.
_CONFLICT_FIELDS = {
    "date_of_birth": "date of birth",
    "postcode": "postcode",
}

_CONFLICTS_SQL = """
WITH linked AS (
    SELECT cl.master_citizen_id,
           r.date_of_birth,
           UPPER(REPLACE(r.postcode, ' ', '')) AS postcode
    FROM clusters cl JOIN records r USING (source_record_id)
)
SELECT master_citizen_id,
       LIST(DISTINCT date_of_birth) AS date_of_birth,
       LIST(DISTINCT postcode) AS postcode
FROM linked
GROUP BY 1
"""


def _conflicts_for(row_values: dict) -> list[dict]:
    """Turn a cluster's distinct-value lists into reviewable conflicts.

    Only fields where the linked records genuinely disagree are returned - a
    single agreed value is not a conflict, it is the normal case.
    """
    conflicts = []
    for field, label in _CONFLICT_FIELDS.items():
        values = [v for v in (row_values.get(field) or []) if v]
        if len(values) > 1:
            conflicts.append({"field": field, "label": label, "values": sorted(values)})
    return conflicts


def get_anomaly_summary(limit: int = 50) -> dict:
    """Read the most recently persisted `anomaly_results`, joined to
    `citizen_profiles` for display fields - powers the Review Queue page's
    KPIs, chart, and worklist.

    Each row carries both halves of an explanation: `top_factors` (what the
    model reacted to) and `conflicts` (the actual disagreeing values in the
    cluster, which is what a reviewer can adjudicate).
    """
    conn = get_connection()
    try:
        tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
        if "anomaly_results" not in tables:
            return _empty_summary()

        all_scores = [row[0] for row in conn.execute("SELECT anomaly_score FROM anomaly_results").fetchall()]

        totals_row = conn.execute(
            "SELECT COUNT(*), SUM(CASE WHEN is_anomaly THEN 1 ELSE 0 END), MAX(contamination) FROM anomaly_results"
        ).fetchone()
        total_profiles_analyzed = int(totals_row[0]) if totals_row[0] is not None else 0
        anomalies_detected = int(totals_row[1]) if totals_row[1] is not None else 0
        contamination = float(totals_row[2]) if totals_row[2] is not None else DEFAULT_CONTAMINATION
        normal_count = total_profiles_analyzed - anomalies_detected
        pct_anomalous = round(anomalies_detected / total_profiles_analyzed * 100, 2) if total_profiles_analyzed else 0.0

        rows = conn.execute(
            f"""
            WITH conflicts AS ({_CONFLICTS_SQL})
            SELECT
                cp.master_citizen_id, cp.preferred_name, cp.agency_count, cp.record_count,
                cp.confidence_score, ar.anomaly_score, ar.is_anomaly, ar.top_factors,
                c.date_of_birth, c.postcode
            FROM anomaly_results ar
            JOIN citizen_profiles cp USING (master_citizen_id)
            LEFT JOIN conflicts c USING (master_citizen_id)
            ORDER BY ar.anomaly_score DESC
            LIMIT ?
            """,
            [limit],
        ).fetchall()
        results = [
            {
                "master_citizen_id": r[0],
                "preferred_name": r[1],
                "agency_count": r[2],
                "record_count": r[3],
                "confidence_score": round(float(r[4]), 6),
                "anomaly_score": round(float(r[5]), 2),
                "status": "Anomalous" if r[6] else "Normal",
                "top_factors": [dict(f) for f in (r[7] or [])],
                "conflicts": _conflicts_for({"date_of_birth": r[8], "postcode": r[9]}),
            }
            for r in rows
        ]

        return {
            "total_profiles_analyzed": total_profiles_analyzed,
            "anomalies_detected": anomalies_detected,
            "normal_count": normal_count,
            "pct_anomalous": pct_anomalous,
            "contamination": contamination,
            "score_distribution": _score_distribution(all_scores),
            "results": results,
            "total": total_profiles_analyzed,
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Detector evaluation
# ---------------------------------------------------------------------------
# Unsupervised anomaly detection normally cannot be scored - there are no
# labels. Here there are, because the data is synthetic: a cluster either
# does or does not correspond one-to-one with a real citizen. That gives a
# concrete, checkable question to rank against - "does this detector surface
# the profiles whose linkage actually went wrong?" - and it is the only
# reason more than one algorithm is implemented below. Each one fails
# differently on this task, and the comparison is the point:
#
#   * Isolation Forest - the incumbent. Tree-based, so it isolates profiles
#     with unusual feature *combinations* and needs no distance metric or
#     scaling.
#   * Local Outlier Factor - density-based. Where the forest asks "is this
#     globally rare?", LOF asks "is this rare compared to its neighbours?",
#     which suits a population made of distinct engagement segments.
#   * Max robust z-score - the interpretable baseline that must be beaten
#     before either model is worth its complexity: flag whatever is furthest
#     from the median on any single feature, using median/MAD so a handful
#     of extreme profiles cannot inflate the spread.


def _linkage_error_labels(master_ids: pd.Series) -> np.ndarray:
    """Ground-truth label per profile: did this cluster resolve exactly one
    real citizen, completely?

    A cluster is marked an error if it merged records from more than one
    real person, or if it holds only part of a person whose remaining
    records ended up in another cluster. Both are linkage failures an
    analyst would want surfaced.
    """
    conn = get_connection()
    try:
        labels = conn.execute(
            """
            WITH linked AS (
                SELECT c.master_citizen_id, r.person_index
                FROM clusters c JOIN records r USING (source_record_id)
            ),
            split_people AS (
                SELECT person_index FROM linked GROUP BY 1 HAVING COUNT(DISTINCT master_citizen_id) > 1
            )
            SELECT
                master_citizen_id,
                (COUNT(DISTINCT person_index) > 1
                 OR BOOL_OR(person_index IN (SELECT person_index FROM split_people))) AS is_error
            FROM linked
            GROUP BY 1
            """
        ).df()
    finally:
        conn.close()
    lookup = labels.set_index("master_citizen_id")["is_error"]
    return lookup.reindex(master_ids.values).fillna(False).to_numpy(dtype=bool)


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision of a ranking, computed directly.

    Used instead of sklearn's `average_precision_score` only so the whole
    evaluation path stays free of surprises about tie handling: ties are
    broken by the stable sort, exactly as the results table would display
    them.
    """
    order = np.argsort(-scores, kind="stable")
    ordered = labels[order]
    if not ordered.any():
        return 0.0
    hits = np.cumsum(ordered)
    precision_at_hit = hits[ordered] / (np.flatnonzero(ordered) + 1)
    return float(precision_at_hit.mean())


def _precision_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    order = np.argsort(-scores, kind="stable")[:k]
    return float(labels[order].mean()) if k else 0.0


def evaluate_detectors(contamination: float = DEFAULT_CONTAMINATION) -> dict:
    """Rank three detectors by how well they surface real linkage errors.

    Memoised on the profile/cluster counts and the contamination asked for
    (see `app/cache.py`) - fitting three models over the whole population
    costs ~2.4s, and the answer only moves when the pipeline is re-run.

    Every detector scores the same feature matrix; only the scoring rule
    differs. Reported alongside the rate you would get by picking profiles
    at random, which is the only honest reference point for a task where
    roughly 1% of the population is a positive.
    """
    return cache.memoize(
        f"detectors:{contamination}",
        ("citizen_profiles", "clusters", "records"),
        lambda: _evaluate_detectors(contamination),
    )


def _evaluate_detectors(contamination: float) -> dict:
    df = _load_features()
    if df.empty:
        raise ValueError("No citizen profiles found. Call POST /run-linkage first.")

    X = df[_FEATURE_COLUMNS].to_numpy(dtype=float)
    labels = _linkage_error_labels(df["master_citizen_id"])

    # LOF and the z-score baseline are distance/spread based, so they need
    # the features on a common scale; the forest is tree-based and does not.
    X_scaled = StandardScaler().fit_transform(X)

    forest = IsolationForest(contamination=contamination, random_state=SEED).fit(X)
    # Most of these features are small integers, so thousands of profiles
    # share an identical feature vector. LOF's default 20 neighbours can
    # then be entirely duplicates, making the local density estimate
    # degenerate (sklearn warns about exactly this). A wider neighbourhood
    # reaches past the duplicates and gives LOF a fair shot before it is
    # compared against the others.
    lof = LocalOutlierFactor(n_neighbors=LOF_NEIGHBORS, contamination=contamination)
    with warnings.catch_warnings():
        # sklearn warns that duplicate points make the local density estimate
        # unreliable. That is a true statement about this data and it is
        # precisely the finding this comparison reports - LOF is a poor fit
        # for low-cardinality integer features. Widening the neighbourhood
        # above is the fair-shot mitigation; silencing the warning here keeps
        # it from being emitted on every request for a result we already
        # account for.
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.neighbors")
        lof.fit_predict(X_scaled)

    median = np.median(X, axis=0)
    mad = np.median(np.abs(X - median), axis=0)
    # 1.4826 rescales the MAD to be a consistent estimator of the standard
    # deviation for normally distributed data; the floor keeps a constant
    # feature from dividing by zero.
    robust_sigma = np.maximum(mad * 1.4826, 1e-9)

    detectors = {
        "isolation_forest": {
            "label": "Isolation Forest",
            "scores": -forest.decision_function(X),
            "note": "Incumbent. Isolates unusual feature combinations; no scaling needed.",
        },
        "local_outlier_factor": {
            "label": "Local Outlier Factor",
            "scores": -lof.negative_outlier_factor_,
            "note": "Density-based: unusual relative to nearby profiles rather than globally.",
        },
        "max_robust_zscore": {
            "label": "Max robust z-score (baseline)",
            "scores": np.abs((X - median) / robust_sigma).max(axis=1),
            "note": "Interpretable baseline: furthest from the median on any single feature.",
        },
    }

    n_positives = int(labels.sum())
    n_profiles = len(labels)
    base_rate = n_positives / n_profiles if n_profiles else 0.0
    k = max(int(round(contamination * n_profiles)), 1)

    results = []
    for name, detector in detectors.items():
        scores = np.asarray(detector["scores"], dtype=float)
        average_precision = _average_precision(scores, labels)
        results.append(
            {
                "detector": name,
                "label": detector["label"],
                "note": detector["note"],
                "average_precision": round(average_precision, 6),
                "precision_at_k": round(_precision_at_k(scores, labels, k), 6),
                "recall_at_k": round(
                    float(labels[np.argsort(-scores, kind="stable")[:k]].sum() / n_positives)
                    if n_positives
                    else 0.0,
                    6,
                ),
                "lift_over_random": round(average_precision / base_rate, 2) if base_rate else 0.0,
            }
        )

    results.sort(key=lambda r: r["average_precision"], reverse=True)
    return {
        "profiles": n_profiles,
        "linkage_errors": n_positives,
        "base_rate": round(base_rate, 6),
        "k": k,
        "contamination": contamination,
        "detectors": results,
        "best_detector": results[0]["detector"] if results else None,
    }
