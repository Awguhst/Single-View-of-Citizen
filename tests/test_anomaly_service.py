"""Anomaly scoring: ground-truth labelling, ranking metrics, and attribution."""

from __future__ import annotations

import numpy as np
import pytest

from app import anomaly_service


def test_average_precision_of_a_perfect_ranking_is_one():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([True, True, False, False])
    assert anomaly_service._average_precision(scores, labels) == pytest.approx(1.0)


def test_average_precision_of_an_inverted_ranking_is_poor():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([False, False, True, True])
    # Hits land at ranks 3 and 4: (1/3 + 2/4) / 2.
    assert anomaly_service._average_precision(scores, labels) == pytest.approx((1 / 3 + 0.5) / 2)


def test_average_precision_with_no_positives_is_zero():
    scores = np.array([0.9, 0.1])
    labels = np.array([False, False])
    assert anomaly_service._average_precision(scores, labels) == 0.0


def test_precision_at_k_counts_only_the_top_k():
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    labels = np.array([True, False, True, True])
    assert anomaly_service._precision_at_k(scores, labels, 2) == pytest.approx(0.5)
    assert anomaly_service._precision_at_k(scores, labels, 4) == pytest.approx(0.75)


def test_linkage_error_labels_flag_both_failure_modes(tiny_linked_db):
    """Against the hand-built fixture: C1 is clean, C2/C3 hold an over-split
    person, and C4 is an over-merge."""
    import pandas as pd

    labels = anomaly_service._linkage_error_labels(pd.Series(["C1", "C2", "C3", "C4"]))
    assert list(labels) == [False, True, True, True]


def test_attribution_blames_the_feature_that_is_actually_extreme():
    """A profile that is ordinary except for one wildly outlying feature must
    have that feature as its top reason."""
    from sklearn.ensemble import IsolationForest

    rng = np.random.default_rng(0)
    n_features = len(anomaly_service._FEATURE_COLUMNS)
    X = rng.integers(1, 4, size=(400, n_features)).astype(float)

    # Make the last row extreme on exactly one feature.
    target_feature = anomaly_service._FEATURE_COLUMNS.index("record_count")
    X[-1, :] = 2.0
    X[-1, target_feature] = 200.0

    model = IsolationForest(contamination=0.05, random_state=0).fit(X)
    contributions = anomaly_service._attribute_scores(model, X)

    assert int(np.argmax(contributions[-1])) == target_feature


def test_attribution_matrix_matches_the_feature_matrix_shape():
    from sklearn.ensemble import IsolationForest

    rng = np.random.default_rng(1)
    X = rng.integers(0, 5, size=(120, len(anomaly_service._FEATURE_COLUMNS))).astype(float)
    model = IsolationForest(contamination=0.05, random_state=0).fit(X)

    assert anomaly_service._attribute_scores(model, X).shape == X.shape


def test_confidence_score_is_not_a_model_input():
    """It was measured at std 8e-6 - a constant column that contributed
    nothing while appearing in the UI as evidence. It must stay out."""
    assert "confidence_score" not in anomaly_service._FEATURE_COLUMNS


def test_no_demographic_or_lifestyle_features_are_used():
    """The project-wide design boundary, enforced rather than documented."""
    forbidden = {"age", "date_of_birth", "marital_status", "sex", "gender", "ethnicity", "postcode", "amount"}
    assert not forbidden & set(anomaly_service._FEATURE_COLUMNS)


def test_every_feature_has_a_display_label():
    """A missing label would raise a KeyError only when a profile happened to
    be attributed to that feature."""
    assert set(anomaly_service._FEATURE_COLUMNS) <= set(anomaly_service.FEATURE_LABELS)


def test_normalize_maps_scores_onto_zero_to_one_hundred():
    scores = np.array([-0.2, 0.0, 0.3])
    normalized = anomaly_service._normalize(scores)
    assert normalized.min() == 0.0
    assert normalized.max() == 100.0


def test_normalize_handles_a_constant_score_vector():
    """Min-max scaling divides by the range, which is zero here."""
    assert list(anomaly_service._normalize(np.array([0.5, 0.5, 0.5]))) == [0.0, 0.0, 0.0]


def test_feature_query_is_deterministically_ordered():
    """Regression: IsolationForest subsamples its training rows, so the order
    of the feature query decides which profiles get flagged.

    Without an explicit sort DuckDB returns hash-join order, which differs
    between database files - rebuilding from the same seed produced a
    different review queue (501 vs 486 flagged on byte-identical clusters).
    Containerising the app is what exposed it, since the image builds its own
    database.
    """
    assert "ORDER BY master_citizen_id" in anomaly_service._FEATURE_QUERY
