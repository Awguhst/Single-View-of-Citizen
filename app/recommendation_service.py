"""Service-coverage gaps, ranked by how common each service is among peers.

--------------------------------------------------------------------------
What this replaces, and why
--------------------------------------------------------------------------
The original rule was a single line: walk a hardcoded priority list and
return the first agency the citizen has no record with. It was honest and
fast, but the ordering carried no information - "Healthcare before Revenue
& Tax" was a developer's guess baked into a constant, and it was duplicated
in both Python and JavaScript with a comment asking future editors to keep
the two in sync by hand.

This ranks the same gaps by a measured quantity instead: among citizens
whose service footprint already looks like this one's, what share also hold
a record with the missing agency? A gap that 94% of comparable citizens
have filled is a more useful thing to surface than one that 11% have, and
that ordering comes from the data rather than from an opinion.

--------------------------------------------------------------------------
The design boundary this stays inside
--------------------------------------------------------------------------
This project deliberately computes no risk, likelihood, or predictive score
about a person (see `citizen_service.py`'s module docstring and the README
section it points at). That boundary is not relaxed here, and the
distinction is precise:

* The peer group is built **only** from which agencies hold a record for a
  citizen. It never uses age, marital status, address, service *amounts*,
  criminal-justice records, or anything else demographic or lifestyle-
  derived. The features are administrative coverage facts, nothing more.

* The number reported is a **descriptive population statistic** - "82% of
  the 1,204 citizens with this service footprint have a Healthcare record"
  - not a prediction about this individual. It says nothing about what this
  citizen will do, should do, or is likely to do, and every figure ships
  with the exact peer-group size and definition it was computed from, so it
  can be checked rather than trusted.

* Citizens are never ranked against each other. The ranking is over one
  citizen's own coverage gaps.

The same three agencies the original rule excluded are still excluded, for
the same reasons (see `_CANDIDATE_AGENCIES`).

--------------------------------------------------------------------------
Backoff, and why it matters
--------------------------------------------------------------------------
The natural peer group - citizens holding records with *every* agency this
one does - shrinks fast as a footprint grows, and a prevalence computed
from nine peers is noise dressed up as evidence. So the peer definition
backs off: if the group is smaller than `MIN_PEER_GROUP`, the least common
agency is dropped from the conditioning set and the group is rebuilt, until
it is either large enough or has widened to the whole population. Every
response states which conditioning set was actually used, so a weakly
conditioned figure is visibly weakly conditioned instead of silently so.
"""

from __future__ import annotations

from app.data_generator import get_connection

# Agencies a citizen might plausibly be missing a record from. Immigration
# and Veterans Affairs are excluded because they apply only to citizens in
# specific circumstances, and Criminal Justice because "acquire a criminal
# record" is not a service gap. This is the same list, for the same
# reasons, that the original hardcoded rule used - only the *ordering*
# within it is now measured rather than assumed.
_CANDIDATE_AGENCIES = (
    "Healthcare",
    "Revenue & Tax",
    "Employment",
    "Driver Licensing",
    "Passport Office",
    "Social Security",
    "Education",
    "Housing",
)

# Below this many peers, a prevalence figure is too noisy to show without
# widening the group first. 100 keeps the margin of error on a proportion
# to roughly +/-5 points, which is well inside the precision this is
# presented at.
MIN_PEER_GROUP = 100


def _peer_filter_sql(agencies: tuple[str, ...]) -> str:
    """A WHERE clause matching profiles linked to every agency in `agencies`.

    An empty conditioning set matches the whole population, which is the
    final backoff step.
    """
    if not agencies:
        return "TRUE"
    return " AND ".join(f"list_contains(linked_agencies, '{agency}')" for agency in agencies)


def get_coverage_recommendations(master_citizen_id: str) -> dict | None:
    """Rank one citizen's service-coverage gaps by peer prevalence.

    Returns None if the profile does not exist.
    """
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT linked_agencies FROM citizen_profiles WHERE master_citizen_id = ?",
            [master_citizen_id],
        ).fetchone()
        if row is None:
            return None
        linked = list(row[0])

        gaps = [agency for agency in _CANDIDATE_AGENCIES if agency not in linked]
        if not gaps:
            return {
                "master_citizen_id": master_citizen_id,
                "linked_agencies": linked,
                "recommendations": [],
                "fully_covered": True,
                "candidate_agencies": list(_CANDIDATE_AGENCIES),
            }

        # Condition on the citizen's own footprint, restricted to the
        # candidate list - conditioning on Immigration or Veterans Affairs
        # would fragment the peer group around circumstances this feature
        # deliberately does not reason about.
        conditioning = tuple(a for a in _CANDIDATE_AGENCIES if a in linked)

        # Order for backoff: drop the *rarest* agency first, since it is the
        # one doing most of the narrowing.
        population = conn.execute("SELECT COUNT(*) FROM citizen_profiles").fetchone()[0]

        agency_frequency = {
            agency: conn.execute(
                "SELECT COUNT(*) FROM citizen_profiles WHERE list_contains(linked_agencies, ?)",
                [agency],
            ).fetchone()[0]
            for agency in conditioning
        }
        backoff_order = sorted(conditioning, key=lambda a: agency_frequency[a])

        used = conditioning
        peer_count = 0
        while True:
            peer_count = conn.execute(
                f"SELECT COUNT(*) FROM citizen_profiles WHERE {_peer_filter_sql(used)}"
            ).fetchone()[0]
            if peer_count >= MIN_PEER_GROUP or not used:
                break
            drop = backoff_order[0]
            backoff_order = backoff_order[1:]
            used = tuple(a for a in used if a != drop)

        peer_filter = _peer_filter_sql(used)
        gap_counts = conn.execute(
            f"""
            SELECT {', '.join(
                f'''COUNT(*) FILTER (WHERE list_contains(linked_agencies, '{agency}')) AS "{agency}"'''
                for agency in gaps
            )}
            FROM citizen_profiles WHERE {peer_filter}
            """
        ).fetchone()

        recommendations = [
            {
                "agency": agency,
                "peers_with_record": int(count),
                "peer_group_size": peer_count,
                "peer_prevalence": round(count / peer_count, 4) if peer_count else 0.0,
            }
            for agency, count in zip(gaps, gap_counts)
        ]
        recommendations.sort(key=lambda r: r["peer_prevalence"], reverse=True)

        return {
            "master_citizen_id": master_citizen_id,
            "linked_agencies": linked,
            "recommendations": recommendations,
            "fully_covered": False,
            "candidate_agencies": list(_CANDIDATE_AGENCIES),
            "peer_group_size": peer_count,
            "peer_group_definition": list(used),
            "peer_group_is_population": not used,
            "backed_off": used != conditioning,
            "population": population,
        }
    finally:
        conn.close()
