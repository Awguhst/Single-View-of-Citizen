"""Synthetic generator: reproducibility, and that the noise knobs do what they say.

The most important test in this file is
`test_default_profile_reproduces_the_shipped_dataset`. Making the noise
constants adjustable was a refactor of working code, and the guarantee that
made it safe is that the default profile still produces exactly the dataset
it always did. That guarantee is worth nothing unless something checks it.
"""

from __future__ import annotations

import random

import pytest

from app import data_generator
from app.data_generator import NOISE_LEVELS, NoiseProfile, generate_all

# Digest of the default 10,000-citizen dataset, captured from the generator
# as it behaved *before* `NoiseProfile` existed and verified row-for-row
# against the previously generated database at the time of the refactor.
# If this changes, the default synthetic dataset changed - which is either a
# bug or a deliberate decision that needs this constant updated alongside it.
_DEFAULT_DATASET_DIGEST = "b888500321e6f12bdc15"
_DEFAULT_RECORD_COUNT = 74_955
_DEFAULT_PEOPLE = 10_000


def _digest(records_df) -> str:
    import hashlib

    return hashlib.sha256(records_df.astype(str).to_csv(index=False).encode()).hexdigest()[:20]


@pytest.mark.slow
def test_default_profile_reproduces_the_shipped_dataset():
    result = generate_all(persist=False)

    assert result.people == _DEFAULT_PEOPLE
    assert result.records == _DEFAULT_RECORD_COUNT
    assert _digest(result.records_df) == _DEFAULT_DATASET_DIGEST


def test_generation_is_deterministic_for_a_given_seed():
    first = generate_all(n_people=120, persist=False)
    second = generate_all(n_people=120, persist=False)
    assert _digest(first.records_df) == _digest(second.records_df)


def test_different_seeds_produce_different_data():
    first = generate_all(n_people=120, seed=1, persist=False)
    second = generate_all(n_people=120, seed=2, persist=False)
    assert _digest(first.records_df) != _digest(second.records_df)


def test_persist_false_writes_nothing(db_path):
    generate_all(n_people=50, persist=False)
    assert not db_path.exists()


def test_noise_profile_rejects_non_monotonic_first_name_thresholds():
    """The first-name thresholds are cumulative, so an inverted profile
    would silently make a branch unreachable rather than erroring."""
    with pytest.raises(ValueError, match="first-name thresholds"):
        NoiseProfile(first_name_unchanged_prob=0.9, first_name_nickname_prob=0.5)


def test_noise_profile_rejects_non_monotonic_last_name_thresholds():
    with pytest.raises(ValueError, match="last-name thresholds"):
        NoiseProfile(last_name_unchanged_prob=0.99, last_name_case_variant_prob=0.5)


def test_pristine_profile_captures_identity_perfectly():
    """With every noise probability turned off, each person's records must
    agree exactly - this is the control condition the benchmark's sanity
    check depends on."""
    result = generate_all(n_people=150, noise=NOISE_LEVELS["pristine"], persist=False)
    df = result.records_df

    for column in ("first_name", "last_name", "date_of_birth", "email", "phone", "address", "postcode"):
        variants = df.groupby("person_index")[column].nunique()
        assert variants.max() == 1, f"{column} varied within a person under the pristine profile"

    assert df[["email", "phone", "address"]].isna().sum().sum() == 0


def test_noise_increases_monotonically_across_levels():
    """Each named level must actually be harder than the one before it,
    otherwise the benchmark's curve means nothing."""
    null_rates = []
    name_variants = []
    for level in ("pristine", "light", "default", "heavy", "severe"):
        df = generate_all(n_people=400, noise=NOISE_LEVELS[level], persist=False).records_df
        null_rates.append(df[["email", "phone", "address"]].isna().to_numpy().mean())
        name_variants.append(df.groupby("person_index")["first_name"].nunique().mean())

    assert null_rates == sorted(null_rates), f"missing-value rate not monotonic: {null_rates}"
    assert name_variants == sorted(name_variants), f"name variation not monotonic: {name_variants}"


def test_vary_functions_are_identity_under_pristine_noise():
    pristine = NOISE_LEVELS["pristine"]
    rng = random.Random(0)
    for _ in range(50):
        assert data_generator._vary_first_name("Jonathan", rng, pristine) == "Jonathan"
        assert data_generator._vary_last_name("Smith", rng, pristine) == "Smith"
        assert data_generator._vary_dob("1990-02-03", rng, pristine) == "1990-02-03"
        assert data_generator._vary_postcode("SW1A 1AA", rng, pristine) == "SW1A 1AA"
        assert data_generator._vary_address("10 Main Street", rng, pristine) == "10 Main Street"


def test_ground_truth_person_index_is_present_on_every_record():
    """Every evaluation in this project depends on this column existing."""
    df = generate_all(n_people=80, persist=False).records_df
    assert "person_index" in df.columns
    assert df["person_index"].notna().all()


def test_every_citizen_gets_at_least_one_record():
    result = generate_all(n_people=300, persist=False)
    assert result.records_df["person_index"].nunique() == result.people


def test_agency_count_respects_the_generator_cap():
    """The anomaly page treats an agency count above this cap as evidence of
    an over-merge, so the cap has to actually hold in the source data."""
    df = generate_all(n_people=500, persist=False).records_df
    per_person = df.groupby("person_index")["agency"].nunique()
    assert per_person.max() <= data_generator.MAX_AGENCIES_PER_CITIZEN
