"""The memo cache behind the two expensive evaluation endpoints.

Correctness must never depend on the cache - only latency. These tests pin
the two properties that matter: a hit is returned without recomputing, and a
change to the underlying tables invalidates the entry.
"""

from __future__ import annotations

import duckdb
import pytest

from app import cache


@pytest.fixture(autouse=True)
def clean_cache():
    cache.clear()
    yield
    cache.clear()


def _make_table(db_path, rows: int) -> None:
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute("CREATE OR REPLACE TABLE widgets AS SELECT * FROM range(?) t(i)", [rows])
    finally:
        conn.close()


def test_second_call_does_not_recompute(db_path):
    _make_table(db_path, 3)
    calls = []

    def compute():
        calls.append(1)
        return "value"

    assert cache.memoize("k", ("widgets",), compute) == "value"
    assert cache.memoize("k", ("widgets",), compute) == "value"
    assert len(calls) == 1


def test_changing_the_source_table_invalidates(db_path):
    _make_table(db_path, 3)
    calls = []

    def compute():
        calls.append(1)
        return len(calls)

    assert cache.memoize("k", ("widgets",), compute) == 1
    _make_table(db_path, 4)  # a pipeline re-run changes the row count
    assert cache.memoize("k", ("widgets",), compute) == 2
    assert len(calls) == 2


def test_missing_table_is_a_miss_not_an_error(db_path):
    """A cache probe made before the pipeline has ever run must behave like
    any other miss rather than raising a catalog error."""
    assert cache.memoize("k", ("never_created",), lambda: "computed") == "computed"


def test_distinct_keys_do_not_collide(db_path):
    _make_table(db_path, 1)
    assert cache.memoize("a", ("widgets",), lambda: "first") == "first"
    assert cache.memoize("b", ("widgets",), lambda: "second") == "second"
    assert cache.memoize("a", ("widgets",), lambda: "ignored") == "first"


def test_clear_forces_recompute(db_path):
    _make_table(db_path, 1)
    calls = []
    compute = lambda: (calls.append(1), "v")[1]  # noqa: E731

    cache.memoize("k", ("widgets",), compute)
    cache.clear()
    cache.memoize("k", ("widgets",), compute)
    assert len(calls) == 2
