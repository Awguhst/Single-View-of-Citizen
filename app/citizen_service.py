"""Citizen aggregation: turns Splink clusters + the unified `records` table
into the Unified Citizen Profile per resolved individual.

Aggregation choices worth calling out:

* **preferred_name** and **date_of_birth** are picked via a "most agreed,
  most complete" resolution over every linked record - the same logic
  regardless of which agency a record came from, since every record type
  carries the same noisy identity fields.
* **current_address** is resolved the same way, with the most recent
  `record_date` used as the final tie-break, since an address is the one
  identity field genuinely expected to *change over time* (unlike a name
  or DOB) rather than just being noisily re-captured.
* **tax_status / driving_licence_status / passport_status /
  employment_status / housing_assistance / veteran_status** each take the
  status of that citizen's most recent record of the matching type
  (`arg_max(status, record_date)`), since these are all point-in-time
  states that can genuinely change - a citizen's most recent tax return
  status matters, not some average of every one they've ever filed.
* **benefits_received / current_prescriptions / education_history** are
  the distinct set of `detail` values seen across that citizen's records
  of the matching type, since a citizen can genuinely hold several
  distinct benefits, prescriptions, or qualifications at once - these
  aren't a single evolving status like the fields above.
* **healthcare_registrations / hospital_visits** are simple counts: how
  many of that record type this citizen has, however many that turns out
  to be.
* **marital_status** resolves the same way as the point-in-time status
  fields (`arg_max(marital_status, record_date)`), but is only ever
  sourced from `TAX_RECORD`/`BENEFITS_RECORD` rows - the two record types
  where a real agency would realistically capture it (see
  `data_generator.py`'s `captures_marital_status` flag).
* **age** is computed at read time from the resolved `date_of_birth`
  against real `date.today()` - it is not stored, and is the one profile
  field that isn't frozen relative to a given dataset snapshot (it
  increments on real birthdays), which is the correct behavior for a
  "current age" field.
* There is no monetary aggregation anywhere in this table - `amount` (tax
  paid, monthly benefit, housing benefit) is real per-record detail shown
  on the profile dossier, never summed into a single "wealth" figure.
* Government records are no longer a clean, ground-truth attachment. Each
  one carries the same kind of noisy, independently-captured identity
  fields every other record does, and flows through the *same* Splink
  dedupe pool (see `splink_service.py`). A record attaches to a
  `master_citizen_id` exactly the way every other record does - by being a
  member of that cluster in the `clusters` table - not via any
  ground-truth shortcut.

Design boundary, worth stating explicitly given the sensitivity: this
module computes no risk, likelihood, or predictive score of any kind.
`engagement_tier` is a plain threshold on `agency_count` and
`lifestyle_summary` (see `_lifestyle_summary` below) is a set of
plain-language restatements of counts/values already on the profile, each
citing its exact evidence - neither ranks, scores, or predicts future
behavior. This is a deliberate scope boundary, not an oversight: scoring
people's likelihood of any behavior from demographic/lifestyle proxies is
a well-documented source of discriminatory harm, and this demo does not
do it.

The Overview page's service-coverage section (`get_service_coverage_summary` below)
extends this same boundary to service enrollment: it reports plain counts
of "how many citizens have no on-file record with agency X", plus the same
first-missing-agency rule in a fixed order aggregated across the whole
population. There is no confidence percentage, no ranking of citizens by
likelihood of enrolling, and no model of any kind; the underlying
`linked_agencies` field it reads is fully governed by the same design
boundary as the rest of this module.

Ranking one citizen's own gaps lives in `recommendation_service.py`, which
stays inside the same boundary by a different route: it reports a
descriptive population statistic ("what share of citizens with a comparable
service footprint hold this record") with the peer group and sample size
attached, never a prediction about the individual, and never a ranking of
citizens against one another.
"""

from __future__ import annotations

from datetime import date

from app.data_generator import get_connection
from app.models import RecordType

# agency_count thresholds used to label a profile's engagement tier on the
# detail page - how many distinct government agencies hold a record for
# this citizen, from barely-engaged (one agency) to fully engaged (the
# 6-agency cap applied during data generation).
_ENGAGEMENT_TIERS = (
    (1, "Minimal Engagement"),
    (2, "Limited Engagement"),
    (4, "Moderate Engagement"),
    (5, "High Engagement"),
)
_ENGAGEMENT_TIER_TOP = "Full Engagement"


def _engagement_tier(agency_count: int) -> str:
    for ceiling, label in _ENGAGEMENT_TIERS:
        if agency_count <= ceiling:
            return label
    return _ENGAGEMENT_TIER_TOP


def _engagement_tier_case_sql(column: str = "agency_count") -> str:
    """Build a SQL CASE expression equivalent to `_engagement_tier()`, so
    tier assignment can be pushed into a GROUP BY/WHERE instead of computed
    row-by-row in Python. Derived from `_ENGAGEMENT_TIERS`/
    `_ENGAGEMENT_TIER_TOP` so the two can never drift apart - there is
    exactly one place tier thresholds are defined."""
    clauses = " ".join(f"WHEN {column} <= {ceiling} THEN '{label}'" for ceiling, label in _ENGAGEMENT_TIERS)
    return f"CASE {clauses} ELSE '{_ENGAGEMENT_TIER_TOP}' END"


# Ordered tier labels, lowest to highest engagement - used to sort tier
# summaries and to validate a requested tier in the members lookup.
_ENGAGEMENT_TIER_ORDER = [label for _, label in _ENGAGEMENT_TIERS] + [_ENGAGEMENT_TIER_TOP]

# Government agencies a citizen might plausibly be missing a record from.
# This is now the single definition of that list - the frontend reads the
# agency names back off this module's `get_service_coverage_summary()`
# response rather than restating them, so the two can no longer drift.
#
# The order matters only for `_next_service_gap_case_sql()` below, which
# still powers the population-wide "which gap comes up first" summary. It is
# no longer used to rank an individual citizen's gaps: that is done by
# measured peer prevalence in `recommendation_service.py`, which replaced the
# fixed-order rule this list used to encode.
_SERVICE_GAP_AGENCY_PRIORITY = (
    "Healthcare",
    "Revenue & Tax",
    "Employment",
    "Driver Licensing",
    "Passport Office",
    "Social Security",
    "Education",
    "Housing",
)


def _next_service_gap_case_sql(column: str = "linked_agencies") -> str:
    """Build a SQL CASE expression picking each citizen's first missing
    agency in _SERVICE_GAP_AGENCY_PRIORITY order, pushed into SQL (same
    technique as _engagement_tier_case_sql above) so it can be aggregated
    across every profile in one query. NULL means no gap - every priority
    agency is already linked.

    This fixed order is only ever used for the population-wide summary
    ("which gap comes up first most often"). An individual citizen's gaps
    are ranked by measured peer prevalence in `recommendation_service.py`."""
    clauses = " ".join(
        f"WHEN NOT list_contains({column}, '{agency}') THEN '{agency}'"
        for agency in _SERVICE_GAP_AGENCY_PRIORITY
    )
    return f"CASE {clauses} ELSE NULL END"


_CITIZEN_PROFILES_SQL = """
CREATE OR REPLACE TABLE citizen_profiles AS
WITH joined AS (
    -- Drawn from every record in the cluster regardless of record_type -
    -- every government record type is an equally first-class, noisy
    -- identity capture now, all living in one `records` table.
    SELECT c.master_citizen_id, c.match_probability, r.*
    FROM clusters c
    JOIN records r USING (source_record_id)
),
name_counts AS (
    SELECT master_citizen_id, first_name, last_name, COUNT(*) AS occurrences
    FROM joined
    GROUP BY 1, 2, 3
),
ranked_names AS (
    -- Pick the "best" of the noisy name variants linked into each cluster for
    -- display: prefer a fuller name over an initial (e.g. "Billy" over "B"),
    -- then proper-case spelling over ALL CAPS / all lowercase, then whichever
    -- variant was recorded most often.
    SELECT
        master_citizen_id,
        first_name,
        last_name,
        ROW_NUMBER() OVER (
            PARTITION BY master_citizen_id
            ORDER BY
                LENGTH(first_name) DESC,
                (first_name != UPPER(first_name) AND first_name != LOWER(first_name)) DESC,
                (last_name != UPPER(last_name) AND last_name != LOWER(last_name)) DESC,
                occurrences DESC
        ) AS rn
    FROM name_counts
),
best_name AS (
    SELECT master_citizen_id, first_name || ' ' || last_name AS name
    FROM ranked_names
    WHERE rn = 1
),
dob_counts AS (
    SELECT master_citizen_id, date_of_birth, COUNT(*) AS occurrences
    FROM joined
    GROUP BY 1, 2
),
ranked_dob AS (
    SELECT master_citizen_id, date_of_birth,
        ROW_NUMBER() OVER (PARTITION BY master_citizen_id ORDER BY occurrences DESC) AS rn
    FROM dob_counts
),
best_dob AS (
    SELECT master_citizen_id, date_of_birth
    FROM ranked_dob
    WHERE rn = 1
),
address_candidates AS (
    -- Unlike name/DOB, an address is expected to genuinely change over
    -- time, not just be noisily re-captured - so after ranking by
    -- agreement/completeness, the final tie-break is recency
    -- (record_date DESC) rather than raw frequency alone.
    SELECT
        master_citizen_id, address, city, postcode, record_date,
        COUNT(*) OVER (PARTITION BY master_citizen_id, address, city, postcode) AS freq,
        (CASE WHEN address IS NOT NULL THEN 1 ELSE 0 END
            + CASE WHEN city IS NOT NULL THEN 1 ELSE 0 END
            + CASE WHEN postcode IS NOT NULL THEN 1 ELSE 0 END) AS completeness
    FROM joined
    WHERE address IS NOT NULL OR city IS NOT NULL OR postcode IS NOT NULL
),
ranked_address AS (
    SELECT *, ROW_NUMBER() OVER (
        PARTITION BY master_citizen_id
        ORDER BY freq DESC, completeness DESC, record_date DESC
    ) AS rn
    FROM address_candidates
),
best_address AS (
    SELECT master_citizen_id, address AS current_address, city AS current_city, postcode AS current_postcode
    FROM ranked_address
    WHERE rn = 1
),
cluster_agg AS (
    SELECT
        master_citizen_id,
        AVG(match_probability) AS confidence_score,
        COUNT(*) AS record_count,
        COUNT(DISTINCT agency) AS agency_count,
        LIST(DISTINCT agency) AS linked_agencies,
        arg_max(marital_status, record_date) FILTER (WHERE record_type IN ('TAX_RECORD', 'BENEFITS_RECORD')) AS marital_status,
        arg_max(status, record_date) FILTER (WHERE record_type = 'TAX_RECORD') AS tax_status,
        LIST(DISTINCT detail) FILTER (WHERE record_type = 'BENEFITS_RECORD') AS benefits_received,
        COUNT(*) FILTER (WHERE record_type = 'HEALTHCARE_REGISTRATION') AS healthcare_registrations,
        COUNT(*) FILTER (WHERE record_type = 'HOSPITAL_VISIT') AS hospital_visits,
        LIST(DISTINCT detail) FILTER (WHERE record_type = 'PRESCRIPTION') AS current_prescriptions,
        LIST(DISTINCT detail) FILTER (WHERE record_type = 'EDUCATION_RECORD') AS education_history,
        arg_max(status, record_date) FILTER (WHERE record_type = 'DRIVING_LICENCE') AS driving_licence_status,
        arg_max(status, record_date) FILTER (WHERE record_type = 'PASSPORT') AS passport_status,
        arg_max(status, record_date) FILTER (WHERE record_type = 'EMPLOYMENT_REGISTRATION') AS employment_status,
        arg_max(status, record_date) FILTER (WHERE record_type = 'HOUSING_BENEFIT') AS housing_assistance,
        arg_max(status, record_date) FILTER (WHERE record_type = 'VETERAN_RECORD') AS veteran_status
    FROM joined
    GROUP BY 1
)
SELECT
    bn.master_citizen_id,
    bn.name AS preferred_name,
    bd.date_of_birth,
    ca.marital_status,
    ba.current_address,
    ba.current_city,
    ba.current_postcode,
    ca.confidence_score,
    ca.record_count,
    ca.agency_count,
    ca.linked_agencies,
    ca.tax_status,
    COALESCE(ca.benefits_received, []) AS benefits_received,
    ca.healthcare_registrations,
    ca.hospital_visits,
    COALESCE(ca.current_prescriptions, []) AS current_prescriptions,
    COALESCE(ca.education_history, []) AS education_history,
    ca.driving_licence_status,
    ca.passport_status,
    ca.employment_status,
    ca.housing_assistance,
    ca.veteran_status
FROM best_name bn
JOIN best_dob bd USING (master_citizen_id)
LEFT JOIN best_address ba USING (master_citizen_id)
JOIN cluster_agg ca USING (master_citizen_id)
"""


def _years_between(start: date, end: date) -> int:
    years = end.year - start.year
    if (end.month, end.day) < (start.month, start.day):
        years -= 1
    return years


def _compute_age(date_of_birth: str) -> int:
    return _years_between(date.fromisoformat(date_of_birth), date.today())


def _with_age(rows: list[dict]) -> list[dict]:
    for row in rows:
        row["age"] = _compute_age(row["date_of_birth"])
    return rows


def build_citizen_profiles() -> int:
    """(Re)build the `citizen_profiles` table from clusters + the unified
    `records` table.

    Returns the number of Unified Citizen Profiles created.
    """
    conn = get_connection()
    try:
        conn.execute(_CITIZEN_PROFILES_SQL)
        return conn.execute("SELECT COUNT(*) FROM citizen_profiles").fetchone()[0]
    finally:
        conn.close()


def get_citizen_profile(master_citizen_id: str) -> dict | None:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT *, {tier} AS engagement_tier FROM citizen_profiles WHERE master_citizen_id = ?".format(
                tier=_engagement_tier_case_sql()
            ),
            [master_citizen_id],
        ).fetchone()
        if row is None:
            return None
        columns = [c[0] for c in conn.description]
        profile = dict(zip(columns, row))
        profile["age"] = _compute_age(profile["date_of_birth"])
        return profile
    finally:
        conn.close()


_FIELD_AGREEMENT_COLUMNS = ("email", "phone", "address", "postcode", "date_of_birth", "last_name")


def get_citizen_profile_detail(master_citizen_id: str) -> dict | None:
    """The full profile dossier behind the "Profile" page: the Unified
    Citizen Profile plus every linked record (across all agencies, via the
    unified `records` table), sorted chronologically to power the profile
    timeline, and a field-by-field agreement breakdown explaining *why*
    Splink linked these records (and flagging anywhere the underlying
    records still disagree)."""
    conn = get_connection()
    try:
        profile_row = conn.execute(
            "SELECT *, {tier} AS engagement_tier FROM citizen_profiles WHERE master_citizen_id = ?".format(
                tier=_engagement_tier_case_sql()
            ),
            [master_citizen_id],
        ).fetchone()
        if profile_row is None:
            return None
        profile = dict(zip([c[0] for c in conn.description], profile_row))
        profile["age"] = _compute_age(profile["date_of_birth"])

        record_rows = conn.execute(
            """
            SELECT
                r.source_record_id, r.agency, r.record_type, r.agency_reference_id,
                r.first_name, r.last_name, r.date_of_birth, r.email, r.phone, r.address, r.city, r.postcode,
                r.record_date, r.expiry_date, r.status, r.amount, r.provider_name, r.detail, r.attributes,
                c.match_probability
            FROM clusters c
            JOIN records r USING (source_record_id)
            WHERE c.master_citizen_id = ?
            ORDER BY r.record_date ASC
            """,
            [master_citizen_id],
        ).fetchall()
        record_cols = [c[0] for c in conn.description]
        all_records = [dict(zip(record_cols, row)) for row in record_rows]
        profile["records"] = all_records

        # Field agreement spans every linked record, across every agency,
        # since all of them carry the same noisy, independently-captured
        # identity fields.
        profile["field_agreement"] = [
            {
                "field": field,
                "is_consistent": len({r[field] for r in all_records if r[field]}) <= 1,
                "distinct_values": sorted({r[field] for r in all_records if r[field]}),
            }
            for field in _FIELD_AGREEMENT_COLUMNS
        ]

        profile["lifestyle_summary"] = _lifestyle_summary(profile, all_records)

        return profile
    finally:
        conn.close()


# Thresholds for _lifestyle_summary - simple, tunable counts, not load-bearing
# in any statistical sense. Each tag below is a plain restatement of a count
# or value already present on the profile/its records, with the exact
# evidence cited in `basis` - never a score, ranking, or predictive judgment.
_FREQUENT_HOSPITAL_VISITS_THRESHOLD = 3
_MULTIPLE_PRESCRIPTIONS_THRESHOLD = 2
_MULTIPLE_BENEFITS_THRESHOLD = 2
_MULTIPLE_QUALIFICATIONS_THRESHOLD = 2
_LONG_EMPLOYMENT_YEARS_THRESHOLD = 10


def _lifestyle_summary(profile: dict, records: list[dict]) -> list[dict]:
    """Plain-language, evidence-cited restatements of literal data already
    on this profile or its linked records - simple threshold checks over
    counts/values that already exist, never a score, ranking, or
    prediction. See this module's docstring for the explicit policy
    against risk/likelihood scoring of any kind."""
    tags: list[dict] = []

    if profile["hospital_visits"] >= _FREQUENT_HOSPITAL_VISITS_THRESHOLD:
        tags.append(
            {
                "tag": "Frequent healthcare service user",
                "basis": f"{profile['hospital_visits']} hospital visits on record.",
            }
        )
    if len(profile["current_prescriptions"]) >= _MULTIPLE_PRESCRIPTIONS_THRESHOLD:
        tags.append(
            {
                "tag": "Multiple active prescriptions",
                "basis": f"{len(profile['current_prescriptions'])} prescriptions on record: "
                f"{', '.join(sorted(profile['current_prescriptions']))}.",
            }
        )
    if len(profile["benefits_received"]) >= _MULTIPLE_BENEFITS_THRESHOLD:
        tags.append(
            {
                "tag": "Multiple benefit types on record",
                "basis": f"{len(profile['benefits_received'])} distinct benefit types on record: "
                f"{', '.join(sorted(profile['benefits_received']))}.",
            }
        )
    if len(profile["education_history"]) >= _MULTIPLE_QUALIFICATIONS_THRESHOLD:
        tags.append(
            {
                "tag": "Multiple education qualifications on record",
                "basis": f"{len(profile['education_history'])} qualifications on record: "
                f"{', '.join(sorted(profile['education_history']))}.",
            }
        )

    employment_records = [r for r in records if r["record_type"] == RecordType.EMPLOYMENT_REGISTRATION.value]
    if employment_records:
        earliest = min(r["record_date"] for r in employment_records)
        years = _years_between(date.fromisoformat(earliest), date.today())
        if years >= _LONG_EMPLOYMENT_YEARS_THRESHOLD:
            tags.append(
                {
                    "tag": "Long-standing employment record",
                    "basis": f"Employment record on file since {earliest} ({years} years).",
                }
            )

    if profile["housing_assistance"] is not None:
        tags.append(
            {
                "tag": "Registered for housing assistance",
                "basis": f"Housing assistance record on file, status '{profile['housing_assistance']}'.",
            }
        )
    if profile["veteran_status"] is not None:
        tags.append(
            {
                "tag": "Veteran of the armed forces",
                "basis": f"Veterans Affairs record on file, status '{profile['veteran_status']}'.",
            }
        )

    return tags


def get_showcase_example() -> dict | None:
    """Pick one resolved cluster that makes a good "before/after Splink"
    showcase for the dashboard: 3-4 total linked records with at least 2
    different first-name spellings/variants among them, so the "before"
    side visibly looks like unrelated people. Among qualifying candidates,
    one spanning more distinct agencies is preferred (so the showcase can
    tell a richer cross-agency story), but this is only a soft preference -
    it never overrides the hard requirements above. Picked from the live
    `clusters`/`records` tables only - no ground truth involved, so this
    works exactly the same way it would in production.

    Returns the same shape as `get_citizen_profile_detail`, or None if no
    cluster in the current dataset meets that bar (falls back to the
    single largest cluster, or None if there are no clusters at all).
    """
    conn = get_connection()
    try:
        # Both queries below explicitly exclude any cluster containing a
        # CRIMINAL_RECORD row - this is the most sensitive record type in
        # the dataset, and the showcase is auto-picked (not curated), so it
        # must never be the one spotlighted by chance on the default
        # dashboard landing view every user sees on first login.
        candidate = conn.execute(
            """
            SELECT c.master_citizen_id
            FROM clusters c
            JOIN records s USING (source_record_id)
            GROUP BY c.master_citizen_id
            HAVING COUNT(*) BETWEEN 3 AND 4 AND COUNT(DISTINCT LOWER(s.first_name)) >= 2
                AND COUNT(*) FILTER (WHERE s.record_type = 'CRIMINAL_RECORD') = 0
            ORDER BY COUNT(DISTINCT s.agency) DESC, COUNT(*) DESC, c.master_citizen_id
            LIMIT 1
            """
        ).fetchone()
        if candidate is None:
            candidate = conn.execute(
                """
                SELECT c.master_citizen_id
                FROM clusters c
                JOIN records s USING (source_record_id)
                GROUP BY c.master_citizen_id
                HAVING COUNT(*) FILTER (WHERE s.record_type = 'CRIMINAL_RECORD') = 0
                ORDER BY COUNT(*) DESC, c.master_citizen_id
                LIMIT 1
                """
            ).fetchone()
        if candidate is None:
            return None
        master_citizen_id = candidate[0]
    finally:
        conn.close()

    return get_citizen_profile_detail(master_citizen_id)


def search_person(query: str, limit: int = 50) -> tuple[list[dict], int]:
    """Search resolved profiles by name, matching either the chosen display
    name or any of the underlying linked records - across every agency,
    since all carry the same noisy identity fields - so a search for
    "Jonathan Smith" still finds a cluster whose display name resolved to
    the more common "John Smith" variant, even if that spelling only
    appears on one of the citizen's agency records.

    An empty/blank query skips filtering entirely and instead browses *all*
    profiles alphabetically by name - this is what powers the Directory
    view's default listing before the user has typed anything.

    Returns (results, total) - total is the full match count, which may
    exceed len(results) once `limit` kicks in.
    """
    query = query.strip()
    tier_select = f", {_engagement_tier_case_sql()} AS engagement_tier"
    conn = get_connection()
    try:
        if query:
            like_query = f"%{query}%"
            total = conn.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT DISTINCT c.master_citizen_id
                    FROM clusters c
                    JOIN records s USING (source_record_id)
                    WHERE (s.first_name || ' ' || s.last_name) ILIKE ?
                    UNION
                    SELECT master_citizen_id FROM citizen_profiles WHERE preferred_name ILIKE ?
                )
                """,
                [like_query, like_query],
            ).fetchone()[0]
            rows = conn.execute(
                f"""
                WITH matched_clusters AS (
                    SELECT DISTINCT c.master_citizen_id
                    FROM clusters c
                    JOIN records s USING (source_record_id)
                    WHERE (s.first_name || ' ' || s.last_name) ILIKE ?
                    UNION
                    SELECT master_citizen_id FROM citizen_profiles WHERE preferred_name ILIKE ?
                )
                SELECT cp.*{tier_select}
                FROM citizen_profiles cp
                JOIN matched_clusters mc USING (master_citizen_id)
                ORDER BY cp.preferred_name ASC
                LIMIT ?
                """,
                [like_query, like_query, limit],
            ).fetchall()
        else:
            total = conn.execute("SELECT COUNT(*) FROM citizen_profiles").fetchone()[0]
            rows = conn.execute(
                f"""
                SELECT cp.*{tier_select}
                FROM citizen_profiles cp
                ORDER BY cp.preferred_name ASC
                LIMIT ?
                """,
                [limit],
            ).fetchall()

        columns = [c[0] for c in conn.description]
        results = _with_age([dict(zip(columns, row)) for row in rows])
        return results, total
    finally:
        conn.close()


def get_dashboard_summary() -> dict:
    conn = get_connection()
    try:
        tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}

        total_citizens = (
            conn.execute("SELECT COUNT(*) FROM citizens").fetchone()[0] if "citizens" in tables else 0
        )
        government_records = (
            conn.execute("SELECT COUNT(*) FROM records").fetchone()[0] if "records" in tables else 0
        )

        resolved_citizens = 0
        total_linked_records = 0
        avg_match_probability = 0.0
        if "clusters" in tables:
            resolved_citizens = conn.execute("SELECT COUNT(DISTINCT master_citizen_id) FROM clusters").fetchone()[0]
            total_linked_records = conn.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]
            avg_row = conn.execute("SELECT AVG(match_probability) FROM clusters").fetchone()[0]
            avg_match_probability = round(float(avg_row), 6) if avg_row is not None else 0.0

        participating_agencies = 0
        agency_record_counts: dict[str, int] = {}
        if "records" in tables:
            agency_record_counts = dict(
                conn.execute("SELECT agency, COUNT(*) FROM records GROUP BY 1 ORDER BY 1").fetchall()
            )
            participating_agencies = len(agency_record_counts)

        avg_agencies_per_citizen = 0.0
        if "citizen_profiles" in tables:
            avg_row = conn.execute("SELECT AVG(agency_count) FROM citizen_profiles").fetchone()[0]
            avg_agencies_per_citizen = round(float(avg_row), 2) if avg_row is not None else 0.0

        return {
            "total_citizens": total_citizens,
            "government_records": government_records,
            "resolved_citizens": resolved_citizens,
            "avg_match_probability": avg_match_probability,
            "duplicates_eliminated": max(total_linked_records - resolved_citizens, 0),
            "participating_agencies": participating_agencies,
            "avg_agencies_per_citizen": avg_agencies_per_citizen,
            "agency_record_counts": agency_record_counts,
        }
    finally:
        conn.close()


def get_quality_metrics(review_queue_size: int = 10) -> dict:
    """Linkage-quality metrics for the Data Quality dashboard page: how
    confident the model was across all clusters, how big clusters typically
    are, and which clusters are least confident (candidates for manual
    analyst review)."""
    conn = get_connection()
    try:
        tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
        if "clusters" not in tables:
            return {
                "match_probability_histogram": [],
                "cluster_size_distribution": [],
                "review_queue": [],
                "total_clusters": 0,
                "avg_match_probability": 0.0,
                "multi_record_cluster_count": 0,
                "high_confidence_pct": 0.0,
            }

        histogram_rows = conn.execute(
            """
            SELECT
                CASE
                    WHEN match_probability >= 0.99 THEN '>= 0.99'
                    WHEN match_probability >= 0.95 THEN '0.95 - 0.99'
                    WHEN match_probability >= 0.90 THEN '0.90 - 0.95'
                    WHEN match_probability >= 0.75 THEN '0.75 - 0.90'
                    ELSE '< 0.75'
                END AS bucket,
                COUNT(*) AS n
            FROM clusters
            GROUP BY 1
            """
        ).fetchall()
        bucket_order = ['< 0.75', '0.75 - 0.90', '0.90 - 0.95', '0.95 - 0.99', '>= 0.99']
        histogram_counts = dict(histogram_rows)
        match_probability_histogram = [
            {"label": label, "count": histogram_counts.get(label, 0)} for label in bucket_order
        ]

        size_rows = conn.execute(
            """
            WITH sizes AS (
                SELECT master_citizen_id, COUNT(*) AS n FROM clusters GROUP BY 1
            )
            SELECT CASE WHEN n >= 4 THEN '4+' ELSE CAST(n AS VARCHAR) END AS bucket, COUNT(*) AS clusters
            FROM sizes
            GROUP BY 1
            """
        ).fetchall()
        size_order = ['1', '2', '3', '4+']
        size_counts = dict(size_rows)
        cluster_size_distribution = [
            {"label": label, "count": size_counts.get(label, 0)} for label in size_order
        ]

        review_queue_rows = conn.execute(
            """
            SELECT cp.master_citizen_id, cp.preferred_name, cp.confidence_score, cp.record_count, cp.linked_agencies
            FROM citizen_profiles cp
            WHERE cp.record_count > 1
            ORDER BY cp.confidence_score ASC
            LIMIT ?
            """,
            [review_queue_size],
        ).fetchall()
        review_queue = [
            {
                "master_citizen_id": r[0],
                "name": r[1],
                "match_probability": round(float(r[2]), 6),
                "record_count": r[3],
                "linked_agencies": list(r[4]),
            }
            for r in review_queue_rows
        ]

        totals_row = conn.execute(
            "SELECT COUNT(DISTINCT master_citizen_id), AVG(match_probability) FROM clusters"
        ).fetchone()
        total_clusters = int(totals_row[0]) if totals_row and totals_row[0] is not None else 0
        avg_match_probability = round(float(totals_row[1]), 6) if totals_row and totals_row[1] is not None else 0.0
        # Clusters with 2+ linked records of any type - i.e. genuinely
        # deduplicated identities. Derived from size_counts (already computed
        # above for cluster_size_distribution) rather than a new query.
        multi_record_cluster_count = total_clusters - size_counts.get('1', 0)
        # histogram_counts buckets *records* (one clusters row per linked
        # record), not clusters, so the denominator here must be the total
        # record count from that same histogram - not total_clusters, which
        # would produce a >100% figure.
        total_linked_records = sum(histogram_counts.values())
        high_confidence_pct = (
            round((histogram_counts.get('>= 0.99', 0) / total_linked_records * 100), 2) if total_linked_records else 0.0
        )

        return {
            "match_probability_histogram": match_probability_histogram,
            "cluster_size_distribution": cluster_size_distribution,
            "review_queue": review_queue,
            "total_clusters": total_clusters,
            "avg_match_probability": avg_match_probability,
            "multi_record_cluster_count": multi_record_cluster_count,
            "high_confidence_pct": high_confidence_pct,
        }
    finally:
        conn.close()


def get_engagement_summary() -> dict:
    """Groups every resolved citizen profile into the 5 engagement tiers
    (see `_ENGAGEMENT_TIERS`) for the Engagement Segments page: one summary
    row per tier with population share and average linkage stats, so an
    analyst can see the shape of the citizen population at a glance before
    drilling into any one tier's members."""
    conn = get_connection()
    try:
        tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
        if "citizen_profiles" not in tables:
            return {"total_profiles": 0, "tiers": []}

        tier_case = _engagement_tier_case_sql()
        rows = conn.execute(
            f"""
            SELECT
                {tier_case} AS engagement_tier,
                COUNT(*) AS citizen_count,
                AVG(agency_count) AS avg_agency_count,
                AVG(confidence_score) AS avg_confidence_score
            FROM citizen_profiles
            GROUP BY 1
            """
        ).fetchall()
        columns = [c[0] for c in conn.description]
        by_tier = {row[0]: dict(zip(columns, row)) for row in rows}

        total_profiles = conn.execute("SELECT COUNT(*) FROM citizen_profiles").fetchone()[0]

        # Tier floor/ceiling for the "agency count range" shown on each card -
        # derived from _ENGAGEMENT_TIERS/_ENGAGEMENT_TIER_TOP directly (not
        # from data) so an empty tier still shows its correct defined range
        # instead of nulls, and so all 5 tiers always render even if one has
        # zero members in this dataset.
        tier_bounds = []
        for i, (ceiling, label) in enumerate(_ENGAGEMENT_TIERS):
            floor = _ENGAGEMENT_TIERS[i - 1][0] + 1 if i > 0 else 1
            tier_bounds.append((label, floor, ceiling))
        tier_bounds.append((_ENGAGEMENT_TIER_TOP, _ENGAGEMENT_TIERS[-1][0] + 1, None))

        tiers = []
        for label, floor, ceiling in tier_bounds:
            stats = by_tier.get(label)
            citizen_count = stats["citizen_count"] if stats else 0
            tiers.append(
                {
                    "engagement_tier": label,
                    "min_agency_count": floor,
                    "max_agency_count": ceiling,
                    "citizen_count": citizen_count,
                    "pct_of_population": round(citizen_count / total_profiles * 100, 2) if total_profiles else 0.0,
                    "avg_agency_count": round(float(stats["avg_agency_count"]), 2) if stats else 0.0,
                    "avg_confidence_score": round(float(stats["avg_confidence_score"]), 6) if stats else 0.0,
                }
            )

        return {"total_profiles": total_profiles, "tiers": tiers}
    finally:
        conn.close()


def get_engagement_tier_members(tier: str, limit: int = 50) -> tuple[list[dict], int]:
    """Members of one engagement tier (see `_ENGAGEMENT_TIERS`), same shape
    as `search_person()` - powers the Engagement Segments drill-down table.
    Returns (results, total), same contract as `search_person()`.

    Raises ValueError if `tier` isn't one of the 5 known tier labels.
    """
    if tier not in _ENGAGEMENT_TIER_ORDER:
        raise ValueError(f"Unknown engagement tier: '{tier}'")

    conn = get_connection()
    try:
        tier_case = _engagement_tier_case_sql("agency_count")
        total = conn.execute(
            f"SELECT COUNT(*) FROM citizen_profiles WHERE {tier_case} = ?", [tier]
        ).fetchone()[0]
        rows = conn.execute(
            f"""
            SELECT cp.*, '{tier}' AS engagement_tier
            FROM citizen_profiles cp
            WHERE {tier_case} = ?
            ORDER BY cp.agency_count DESC, cp.preferred_name ASC
            LIMIT ?
            """,
            [tier, limit],
        ).fetchall()
        columns = [c[0] for c in conn.description]
        return _with_age([dict(zip(columns, row)) for row in rows]), total
    finally:
        conn.close()


def get_service_coverage_summary() -> dict:
    """Population-wide Service Coverage Gaps stats: for each of the 8
    illustrative agencies in _SERVICE_GAP_AGENCY_PRIORITY, how many
    resolved citizens have no linked record there, plus the first-gap-in-
    fixed-order rule aggregated across every resolved profile in one query
    (not sampled from one page of /search results).

    The frontend also reads the candidate agency names off this response
    rather than restating them, so the list has exactly one definition.

    A plain aggregation over linked_agencies already on citizen_profiles -
    see this module's docstring for the boundary this stays within."""
    conn = get_connection()
    try:
        tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
        if "citizen_profiles" not in tables:
            return {
                "total_profiles": 0,
                "missing_record_counts": {},
                "next_gap_counts": {},
                "fully_covered_count": 0,
            }

        total_profiles = conn.execute("SELECT COUNT(*) FROM citizen_profiles").fetchone()[0]

        missing_filters = ", ".join(
            f'COUNT(*) FILTER (WHERE NOT list_contains(linked_agencies, \'{agency}\')) AS "{agency}"'
            for agency in _SERVICE_GAP_AGENCY_PRIORITY
        )
        missing_row = conn.execute(f"SELECT {missing_filters} FROM citizen_profiles").fetchone()
        missing_record_counts = dict(zip(_SERVICE_GAP_AGENCY_PRIORITY, missing_row))

        gap_case = _next_service_gap_case_sql()
        gap_rows = conn.execute(
            f"SELECT {gap_case} AS next_gap, COUNT(*) AS citizen_count FROM citizen_profiles GROUP BY 1"
        ).fetchall()
        next_gap_counts = {agency: count for agency, count in gap_rows if agency is not None}
        fully_covered_count = next((count for agency, count in gap_rows if agency is None), 0)

        return {
            "total_profiles": total_profiles,
            "missing_record_counts": missing_record_counts,
            "next_gap_counts": next_gap_counts,
            "fully_covered_count": fully_covered_count,
        }
    finally:
        conn.close()


def has_citizen_profiles() -> bool:
    conn = get_connection()
    try:
        tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
        return "citizen_profiles" in tables
    finally:
        conn.close()
