"""Shared fixtures.

The tests deliberately do *not* run Splink. Training a real model takes
tens of seconds and would be testing Splink rather than this project. What
this project owns is the SQL, the metrics, the noise model and the API
contracts, so the DB-backed fixtures below build a tiny hand-written
dataset whose correct answers can be worked out on paper - see
`tiny_linked_db` for the arithmetic.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from app import data_generator


@pytest.fixture
def db_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point every `get_connection()` call at a throwaway database.

    `data_generator.DB_PATH` is read inside `get_connection()` rather than
    captured at import, so patching the module attribute is enough to
    redirect every service in the app.
    """
    path = tmp_path / "test.duckdb"
    monkeypatch.setattr(data_generator, "DB_PATH", path)
    return path


# One row per record: (record id, ground-truth person, predicted cluster).
#
# Chosen so every metric has a hand-checkable answer:
#   * person 0 -> 3 records, all in cluster C1        (resolved exactly)
#   * person 1 -> 2 records, split across C2 and C3   (over-split)
#   * person 2 -> 2 records, both in C4               (in an over-merged cluster)
#   * person 3 -> 1 record, also in C4                (the over-merge)
#
# Pairwise arithmetic that falls out of this:
#   predicted same-cluster pairs = C(3,2) + 0 + 0 + C(3,2) = 3 + 3 = 6
#   truly same-person pairs      = C(3,2) + C(2,2) + C(2,2) = 3 + 1 + 1 = 5
#   correct (agreeing) pairs     = 3 (inside C1) + 1 (R6,R7 inside C4)  = 4
#   precision = 4/6, recall = 4/5
_FIXTURE_ROWS = [
    ("R1", 0, "C1"),
    ("R2", 0, "C1"),
    ("R3", 0, "C1"),
    ("R4", 1, "C2"),
    ("R5", 1, "C3"),
    ("R6", 2, "C4"),
    ("R7", 2, "C4"),
    ("R8", 3, "C4"),
]

EXPECTED_PRECISION = 4 / 6
EXPECTED_RECALL = 4 / 5
EXPECTED_F1 = 2 * EXPECTED_PRECISION * EXPECTED_RECALL / (EXPECTED_PRECISION + EXPECTED_RECALL)


@pytest.fixture
def tiny_linked_db(db_path: Path) -> Path:
    """A minimal `records` + `clusters` database with known linkage errors."""
    records = pd.DataFrame(
        [
            {
                "source_record_id": record_id,
                "person_index": person,
                "agency": "Healthcare" if index % 2 else "Revenue & Tax",
                "record_type": "HOSPITAL_VISIT" if index % 2 else "TAX_RECORD",
                "first_name": f"Person{person}",
                "last_name": "Test",
                "date_of_birth": f"199{person}-01-01",
                "email": f"person{person}@example.test",
                "phone": f"0700000000{person}",
                "address": f"{person} Test Street",
                "city": "Testville",
                "postcode": f"TE{person} 1ST",
                "record_date": "2024-01-01",
            }
            for index, (record_id, person, _) in enumerate(_FIXTURE_ROWS)
        ]
    )
    clusters = pd.DataFrame(
        [
            {"source_record_id": record_id, "master_citizen_id": cluster, "match_probability": 0.99}
            for record_id, _, cluster in _FIXTURE_ROWS
        ]
    )

    conn = duckdb.connect(str(db_path))
    try:
        conn.register("_records", records)
        conn.execute("CREATE TABLE records AS SELECT * FROM _records")
        conn.register("_clusters", clusters)
        conn.execute("CREATE TABLE clusters AS SELECT * FROM _clusters")
    finally:
        conn.close()
    return db_path
