"""A tiny memo cache for the two expensive, deterministic evaluation results.

The threshold sweep re-clusters 244,000 pairwise edges at nine thresholds,
and the detector comparison fits three models over every resolved profile.
Measured on the shipped dataset they cost roughly 3.3s and 2.4s of CPU
respectively - and both are pure functions of tables that only change when
the pipeline is re-run.

Recomputing them per request was tolerable on a developer machine. It is not
on a small container: they are CPU-bound Python, so the GIL serialises them
even when the browser requests them in parallel, making the Evaluation page
cost about six seconds of a single vCPU every time anyone opens it.

Rather than persist another table, results are held in process and keyed on a
cheap fingerprint of the inputs. `COUNT(*)` on DuckDB is a metadata lookup,
so validating the key costs microseconds, and any pipeline re-run changes a
count and invalidates the entry on its own. A restart simply recomputes -
correctness never depends on the cache, only latency.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.data_generator import get_connection

_cache: dict[str, tuple[tuple, Any]] = {}


def _fingerprint(tables: tuple[str, ...]) -> tuple:
    """Row counts of the tables a result depends on.

    A missing table contributes None rather than raising, so a cache probe
    made before the pipeline has run behaves like any other miss.
    """
    conn = get_connection()
    try:
        present = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
        return tuple(
            conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] if t in present else None
            for t in tables
        )
    finally:
        conn.close()


def memoize(key: str, tables: tuple[str, ...], compute: Callable[[], Any]) -> Any:
    """Return a cached result, recomputing if its input tables have changed."""
    fingerprint = _fingerprint(tables)
    cached = _cache.get(key)
    if cached is not None and cached[0] == fingerprint:
        return cached[1]

    value = compute()
    _cache[key] = (fingerprint, value)
    return value


def clear() -> None:
    """Drop every entry. Used by the tests, and harmless at any other time."""
    _cache.clear()
