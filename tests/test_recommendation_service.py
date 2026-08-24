"""Peer-prevalence coverage recommendations.

Two things need protecting: the prevalence arithmetic itself, and the
backoff rule that stops a figure being quoted off a peer group of nine.
"""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from app import recommendation_service
from app.recommendation_service import MIN_PEER_GROUP


def _build_profiles_db(db_path, footprints: list[list[str]]):
    """A `citizen_profiles` table holding just the columns this service reads."""
    profiles = pd.DataFrame(
        [
            {"master_citizen_id": f"MC{i:05d}", "linked_agencies": agencies, "agency_count": len(agencies)}
            for i, agencies in enumerate(footprints)
        ]
    )
    conn = duckdb.connect(str(db_path))
    try:
        conn.register("_profiles", profiles)
        conn.execute("CREATE TABLE citizen_profiles AS SELECT * FROM _profiles")
    finally:
        conn.close()
    return profiles


def test_prevalence_is_the_share_of_peers_holding_the_record(db_path):
    """Target citizen has Healthcare only. Of the other Healthcare holders,
    exactly 3 of 200 also have Employment, and 150 of 200 have Revenue & Tax."""
    footprints = [["Healthcare"]]  # the citizen being looked at
    footprints += [["Healthcare", "Employment"]] * 3
    footprints += [["Healthcare", "Revenue & Tax"]] * 150
    footprints += [["Healthcare"]] * 47
    _build_profiles_db(db_path, footprints)

    result = recommendation_service.get_coverage_recommendations("MC00000")
    by_agency = {r["agency"]: r for r in result["recommendations"]}

    # The peer group is everyone with Healthcare, including the citizen itself.
    assert by_agency["Revenue & Tax"]["peer_group_size"] == 201
    assert by_agency["Revenue & Tax"]["peers_with_record"] == 150
    assert by_agency["Revenue & Tax"]["peer_prevalence"] == pytest.approx(150 / 201, abs=1e-4)
    assert by_agency["Employment"]["peers_with_record"] == 3


def test_gaps_are_ranked_by_prevalence_not_a_fixed_list(db_path):
    """The whole point of the change: ordering comes from the data.

    Housing sits last in the old hardcoded priority list, but here it is by
    far the most common thing comparable citizens hold, so it must rank first.
    """
    footprints = [["Healthcare"]]
    footprints += [["Healthcare", "Housing"]] * 180
    footprints += [["Healthcare", "Revenue & Tax"]] * 20
    _build_profiles_db(db_path, footprints)

    result = recommendation_service.get_coverage_recommendations("MC00000")
    assert result["recommendations"][0]["agency"] == "Housing"
    assert result["recommendations"][0]["peer_prevalence"] > result["recommendations"][1]["peer_prevalence"]


def test_narrow_footprint_backs_off_to_a_usable_peer_group(db_path):
    """One citizen has a rare four-agency footprint. Conditioning on all of
    it would leave a peer group of one, so the group must widen and say so."""
    rare = ["Healthcare", "Revenue & Tax", "Education", "Driver Licensing"]
    footprints = [rare]
    footprints += [["Healthcare", "Revenue & Tax"]] * 300
    _build_profiles_db(db_path, footprints)

    result = recommendation_service.get_coverage_recommendations("MC00000")

    assert result["backed_off"] is True
    assert result["peer_group_size"] >= MIN_PEER_GROUP
    # The conditioning set actually used must be a subset of the original.
    assert set(result["peer_group_definition"]) < set(rare)


def test_backoff_stops_at_the_whole_population(db_path):
    """When even the population is smaller than MIN_PEER_GROUP, the service
    must return the population figure rather than looping forever."""
    _build_profiles_db(db_path, [["Healthcare"], ["Healthcare", "Housing"], ["Revenue & Tax"]])

    result = recommendation_service.get_coverage_recommendations("MC00000")

    assert result["peer_group_is_population"] is True
    assert result["peer_group_definition"] == []
    assert result["peer_group_size"] == 3


def test_fully_covered_citizen_gets_no_recommendations(db_path):
    everything = list(recommendation_service._CANDIDATE_AGENCIES)
    _build_profiles_db(db_path, [everything])

    result = recommendation_service.get_coverage_recommendations("MC00000")
    assert result["fully_covered"] is True
    assert result["recommendations"] == []


def test_excluded_agencies_are_never_recommended(db_path):
    """Immigration, Veterans Affairs and Criminal Justice are deliberately
    outside the candidate list and must not appear as a 'gap'."""
    _build_profiles_db(db_path, [["Healthcare"]] * 150)

    result = recommendation_service.get_coverage_recommendations("MC00000")
    recommended = {r["agency"] for r in result["recommendations"]}

    assert not recommended & {"Immigration", "Veterans Affairs", "Criminal Justice"}


def test_unknown_citizen_returns_none(db_path):
    _build_profiles_db(db_path, [["Healthcare"]])
    assert recommendation_service.get_coverage_recommendations("NOPE") is None
