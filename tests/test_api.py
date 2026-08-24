"""API contract tests for the endpoints added alongside the ML work.

`TestClient` is used *without* its context manager on purpose: entering it
runs the app's lifespan, which bootstraps the full 10,000-citizen dataset
and a real Splink training run. These tests exercise routing and response
shapes against the tiny fixture database instead.
"""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app

NEW_ENDPOINTS = [
    "/evaluation/linkage",
    "/evaluation/threshold-sweep",
    "/benchmark",
    "/anomaly-detection/evaluation",
    "/citizen/C1/graph",
    "/citizen/C1/recommendations",
]


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.mark.parametrize("path", NEW_ENDPOINTS)
def test_endpoints_fail_cleanly_with_no_data(client, db_path, path):
    """An empty database must produce a 400 telling the caller which
    pipeline step to run, never a 500 from a missing table."""
    response = client.get(path)
    assert response.status_code == 400, response.text
    assert response.json()["detail"]


def test_endpoints_are_open(client, db_path):
    """The app is a local single-user demo with no authentication, so no
    endpoint should ever answer 401/403."""
    for path in NEW_ENDPOINTS + ["/dashboard", "/quality", "/service-coverage"]:
        assert client.get(path).status_code not in (401, 403), path


def test_no_auth_routes_are_registered(client):
    """Authentication was removed deliberately; this catches it being
    reintroduced by accident (for example by restoring an old main.py)."""
    paths = {route.path for route in app.routes if hasattr(route, "methods")}
    assert not {p for p in paths if p.startswith("/auth")}


def test_linkage_evaluation_returns_metrics_and_baselines(client, tiny_linked_db):
    response = client.get("/evaluation/linkage")
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["total_records"] == 8
    assert body["splink"]["pairwise_precision"] == pytest.approx(4 / 6, abs=1e-6)
    assert body["splink"]["over_merged_clusters"] == 1
    assert body["splink"]["over_split_citizens"] == 1

    methods = {b["method"] for b in body["baselines"]}
    assert {"no_linkage", "exact_name_dob", "all_one_cluster"} <= methods
    # Every baseline must carry the note explaining what it is.
    assert all(b["note"] for b in body["baselines"])


def test_citizen_recommendations_expose_the_peer_group(client, db_path):
    profiles = pd.DataFrame(
        [{"master_citizen_id": "MC00000", "linked_agencies": ["Healthcare"], "agency_count": 1}]
        + [
            {"master_citizen_id": f"MC{i:05d}", "linked_agencies": ["Healthcare", "Employment"], "agency_count": 2}
            for i in range(1, 150)
        ]
    )
    conn = duckdb.connect(str(db_path))
    try:
        conn.register("_p", profiles)
        conn.execute("CREATE TABLE citizen_profiles AS SELECT * FROM _p")
    finally:
        conn.close()

    response = client.get("/citizen/MC00000/recommendations")
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["fully_covered"] is False
    top = body["recommendations"][0]
    assert top["agency"] == "Employment"
    assert top["peer_prevalence"] == pytest.approx(149 / 150, abs=1e-3)
    # The evidence behind the number must always travel with it.
    assert top["peer_group_size"] == 150
    assert body["peer_group_definition"] == ["Healthcare"]


def test_unknown_citizen_returns_404(client, db_path):
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute(
            "CREATE TABLE citizen_profiles (master_citizen_id VARCHAR, linked_agencies VARCHAR[], agency_count INTEGER)"
        )
    finally:
        conn.close()

    assert client.get("/citizen/NOPE/recommendations").status_code == 404


def test_openapi_schema_builds(client):
    """Catches a malformed response_model on any of the new endpoints."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    for endpoint in ("/evaluation/linkage", "/benchmark", "/citizen/{master_citizen_id}/graph"):
        assert endpoint in paths


def test_pending_pipeline_errors_name_the_outstanding_step(client, tiny_linked_db):
    """Each 400 must point at the step the caller actually owes next.

    `/anomaly-detection` used to check for anomaly results without first
    checking whether any profiles existed, so before linkage it told the user
    to "Call POST /anomaly-detection/run" - a step that cannot succeed with
    nothing to score. The frontend renders these messages, so the wrong one
    sends the user to a button that fails.
    """
    response = client.get("/anomaly-detection")
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "run-linkage" in detail, detail
    assert "anomaly-detection/run" not in detail, detail


def test_every_pending_error_matches_a_known_pipeline_step(client, db_path):
    """The frontend maps these messages onto a named step (see
    `PIPELINE_STEPS` in app.js). An endpoint inventing different wording would
    fall through to a raw API string in the UI."""
    known = ("POST /generate-data", "POST /run-linkage", "POST /anomaly-detection/run", "POST /benchmark/run")
    for path in NEW_ENDPOINTS + ["/anomaly-detection", "/quality", "/engagement", "/search"]:
        response = client.get(path)
        if response.status_code != 400:
            continue
        detail = response.json()["detail"]
        assert any(step in detail for step in known), f"{path}: {detail}"
