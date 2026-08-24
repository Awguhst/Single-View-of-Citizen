"""Synthetic data generation for the CitizenLink (Single View of Citizen) demo.

Generates a reproducible (seeded) population of citizens, simulates how
eleven government agencies would *independently* and *imperfectly* record
those same people (name variants, address abbreviations, email-format
differences, missing fields) across their own record types - tax filings,
benefit claims, healthcare registrations, licences, and the rest - which
carry the same kind of noisy identity capture rather than a clean,
ground-truth attachment. Every kind of record is persisted into one
unified `records` table, distinguished by its `record_type` column (and,
derived from that, its `agency` column).

Nothing here is real data - it is all Faker-generated - but the shapes
and the data-quality issues mirror what a government sees when trying to
consolidate feeds from agencies that have never agreed on a common
citizen identifier.
"""

from __future__ import annotations

import math
import os
import random
import string
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date as date_cls
from datetime import timedelta
from pathlib import Path

import duckdb
import pandas as pd
from faker import Faker

from app.models import AGENCY_RECORD_TYPES, Agency, RecordType

# ---------------------------------------------------------------------------
# Reproducibility & scale constants
# ---------------------------------------------------------------------------
SEED = 6999
N_PEOPLE = 10_000

# Fixed "as of" date the synthetic dataset is generated relative to -
# deliberately not `date.today()`, so record dates (and therefore the whole
# dataset) stay identical however many days from now this is regenerated.
AS_OF_DATE = date_cls(2025, 6, 1)

# Probability a given citizen has at least one record with a given agency,
# sampled independently per agency (not a fixed enumerated bundle) - some
# agencies are near-universal (Revenue & Tax, Healthcare), some are rare
# (Immigration, Veterans Affairs), matching how real citizens' footprints
# across government agencies vary enormously. Selections are clamped to
# 1-6 agencies per citizen (see `_sample_agencies`).
AGENCY_INCLUSION_PROBABILITY: dict[Agency, float] = {
    Agency.REVENUE_TAX: 0.90,
    Agency.SOCIAL_SECURITY: 0.35,
    Agency.HEALTHCARE: 0.93,
    Agency.EDUCATION: 0.45,
    Agency.IMMIGRATION: 0.07,
    Agency.DRIVER_LICENSING: 0.55,
    Agency.PASSPORT_OFFICE: 0.60,
    Agency.EMPLOYMENT: 0.70,
    Agency.HOUSING: 0.18,
    Agency.VETERANS_AFFAIRS: 0.05,
    Agency.CRIMINAL_JUSTICE: 0.08,
}
MIN_AGENCIES_PER_CITIZEN = 1
MAX_AGENCIES_PER_CITIZEN = 6

# Healthcare is the one agency with repeatable sub-records: every
# registered citizen gets 1-3 hospital visits and 0-2 prescriptions,
# mirroring the old "1-4 accounts of the same type" multiplicity pattern.
HOSPITAL_VISIT_COUNT_RANGE = (1, 3)
PRESCRIPTION_COUNT_RANGE = (0, 2)

# ---------------------------------------------------------------------------
# Demographic constants
# ---------------------------------------------------------------------------
MINIMUM_AGE = 21
MAXIMUM_AGE = 66

# Assigned once per citizen at generation time (ground truth); flows onto
# TAX_RECORD/BENEFITS_RECORD rows only - see `_RECORD_TYPE_CONFIG`'s
# `captures_marital_status` flag - since those are the two record types a
# real agency would realistically capture this on.
MARITAL_STATUSES = ["Single", "Married", "Divorced", "Widowed", "Civil Partnership", "Separated"]
MARITAL_STATUS_WEIGHTS = [0.32, 0.42, 0.12, 0.06, 0.04, 0.04]

# ---------------------------------------------------------------------------
# Identity-noise thresholds shared by every record-emitting call site via
# `_noisy_identity_capture`. Each constant is named for, and gates, the
# branch it literally selects (e.g. `*_UNCHANGED_PROB`/`*_REUSE_PROB` gate
# the "leave it as-is" branch) so a future edit can't accidentally invert
# its sense. The first-name/last-name thresholds are *cumulative* - each
# `_vary_*` function draws one random number and compares it against these
# in sequence - not independent per-branch probabilities.
# ---------------------------------------------------------------------------
FIRST_NAME_UNCHANGED_PROB = 0.55
FIRST_NAME_NICKNAME_PROB = 0.80
FIRST_NAME_INITIAL_PROB = 0.92
LAST_NAME_UNCHANGED_PROB = 0.85
LAST_NAME_CASE_VARIANT_PROB = 0.93
ADDRESS_ABBREVIATION_PROB = 0.5  # per street-type token
POSTCODE_STRIP_SPACE_PROB = 0.3
EMAIL_REUSE_PROB = 0.5  # probability of reusing the canonical email verbatim
EMAIL_NUMERIC_SUFFIX_PROB = 0.2
PHONE_REUSE_PROB = 0.5  # probability of reusing the canonical phone verbatim
DOB_UNCHANGED_PROB = 0.95
EMAIL_NULL_PROB = 0.12
PHONE_NULL_PROB = 0.12
ADDRESS_NULL_PROB = 0.10


@dataclass(frozen=True)
class NoiseProfile:
    """How badly the agencies capture identity, as one adjustable object.

    Every field defaults to the module constant directly above it, so
    `NoiseProfile()` reproduces the original hard-coded behaviour exactly -
    the same random draws in the same order, and therefore the same dataset
    byte-for-byte given the same seed. Nothing about the default dataset
    changed when this was introduced; a test pins that.

    The point of making these adjustable is evaluation. With one fixed
    noise level, "the linkage scores F1 0.999" is a single unfalsifiable
    data point - you cannot tell whether the model is strong or the problem
    is easy. Sweeping the noise (see `benchmark_service.py`) turns it into a
    curve, which is the thing that actually characterises the model.

    Threshold semantics are inherited unchanged from the constants: the
    first-name and last-name probabilities are *cumulative* (one random
    draw compared against each in sequence), so within a profile they must
    stay in non-decreasing order. `__post_init__` enforces that rather than
    letting an inverted profile silently produce nonsense.
    """

    first_name_unchanged_prob: float = FIRST_NAME_UNCHANGED_PROB
    first_name_nickname_prob: float = FIRST_NAME_NICKNAME_PROB
    first_name_initial_prob: float = FIRST_NAME_INITIAL_PROB
    last_name_unchanged_prob: float = LAST_NAME_UNCHANGED_PROB
    last_name_case_variant_prob: float = LAST_NAME_CASE_VARIANT_PROB
    address_abbreviation_prob: float = ADDRESS_ABBREVIATION_PROB
    postcode_strip_space_prob: float = POSTCODE_STRIP_SPACE_PROB
    email_reuse_prob: float = EMAIL_REUSE_PROB
    email_numeric_suffix_prob: float = EMAIL_NUMERIC_SUFFIX_PROB
    phone_reuse_prob: float = PHONE_REUSE_PROB
    dob_unchanged_prob: float = DOB_UNCHANGED_PROB
    email_null_prob: float = EMAIL_NULL_PROB
    phone_null_prob: float = PHONE_NULL_PROB
    address_null_prob: float = ADDRESS_NULL_PROB

    def __post_init__(self) -> None:
        if not (
            self.first_name_unchanged_prob
            <= self.first_name_nickname_prob
            <= self.first_name_initial_prob
        ):
            raise ValueError(
                "first-name thresholds are cumulative and must be non-decreasing: "
                f"{self.first_name_unchanged_prob} <= {self.first_name_nickname_prob} "
                f"<= {self.first_name_initial_prob}"
            )
        if not self.last_name_unchanged_prob <= self.last_name_case_variant_prob:
            raise ValueError(
                "last-name thresholds are cumulative and must be non-decreasing: "
                f"{self.last_name_unchanged_prob} <= {self.last_name_case_variant_prob}"
            )


DEFAULT_NOISE = NoiseProfile()

# Named noise levels for the benchmark sweep, ordered easiest to hardest.
# `default` is the profile the shipped dataset uses; the others exist to
# bracket it. `pristine` is the sanity check - with no noise at all, any
# competent linker should score ~1.0, so anything less indicates a bug in
# the pipeline rather than a hard problem. `severe` is the stress case:
# names rarely survive intact, a third of contact fields are missing, and
# DOB transcription errors are ten times more common than by default.
NOISE_LEVELS: dict[str, NoiseProfile] = {
    "pristine": NoiseProfile(
        first_name_unchanged_prob=1.0,
        first_name_nickname_prob=1.0,
        first_name_initial_prob=1.0,
        last_name_unchanged_prob=1.0,
        last_name_case_variant_prob=1.0,
        address_abbreviation_prob=0.0,
        postcode_strip_space_prob=0.0,
        email_reuse_prob=1.0,
        email_numeric_suffix_prob=0.0,
        phone_reuse_prob=1.0,
        dob_unchanged_prob=1.0,
        email_null_prob=0.0,
        phone_null_prob=0.0,
        address_null_prob=0.0,
    ),
    "light": NoiseProfile(
        first_name_unchanged_prob=0.80,
        first_name_nickname_prob=0.92,
        first_name_initial_prob=0.97,
        last_name_unchanged_prob=0.94,
        last_name_case_variant_prob=0.98,
        address_abbreviation_prob=0.25,
        postcode_strip_space_prob=0.15,
        email_reuse_prob=0.80,
        email_numeric_suffix_prob=0.10,
        phone_reuse_prob=0.80,
        dob_unchanged_prob=0.99,
        email_null_prob=0.05,
        phone_null_prob=0.05,
        address_null_prob=0.04,
    ),
    "default": DEFAULT_NOISE,
    "heavy": NoiseProfile(
        first_name_unchanged_prob=0.35,
        first_name_nickname_prob=0.65,
        first_name_initial_prob=0.85,
        last_name_unchanged_prob=0.70,
        last_name_case_variant_prob=0.85,
        address_abbreviation_prob=0.70,
        postcode_strip_space_prob=0.50,
        email_reuse_prob=0.30,
        email_numeric_suffix_prob=0.35,
        phone_reuse_prob=0.30,
        dob_unchanged_prob=0.90,
        email_null_prob=0.25,
        phone_null_prob=0.25,
        address_null_prob=0.20,
    ),
    "severe": NoiseProfile(
        first_name_unchanged_prob=0.20,
        first_name_nickname_prob=0.50,
        first_name_initial_prob=0.75,
        last_name_unchanged_prob=0.55,
        last_name_case_variant_prob=0.75,
        address_abbreviation_prob=0.85,
        postcode_strip_space_prob=0.70,
        email_reuse_prob=0.15,
        email_numeric_suffix_prob=0.50,
        phone_reuse_prob=0.15,
        dob_unchanged_prob=0.80,
        email_null_prob=0.35,
        phone_null_prob=0.35,
        address_null_prob=0.30,
    ),
}

DB_PATH = Path(
    os.environ.get("CITIZENLINK_DB_PATH", Path(__file__).resolve().parent.parent / "data" / "svoc.duckdb")
)

EMAIL_DOMAINS = ["gmail.com", "outlook.com", "yahoo.co.uk", "hotmail.com", "icloud.com"]

# A representative (not exhaustive) set of common nickname variants used to
# simulate the "John Smith / Jonathan Smith / Jon Smith" style of duplicate.
NICKNAMES: dict[str, list[str]] = {
    "james": ["Jim", "Jimmy"],
    "john": ["Jon", "Johnny"],
    "jonathan": ["Jon", "Jonny"],
    "robert": ["Rob", "Bob", "Bobby"],
    "william": ["Will", "Bill", "Billy"],
    "richard": ["Rick", "Dick", "Rich"],
    "michael": ["Mike", "Mick"],
    "elizabeth": ["Liz", "Beth", "Eliza"],
    "katherine": ["Kate", "Kathy", "Kat"],
    "margaret": ["Maggie", "Meg", "Peggy"],
    "thomas": ["Tom", "Tommy"],
    "charles": ["Charlie", "Chuck"],
    "christopher": ["Chris"],
    "daniel": ["Dan", "Danny"],
    "matthew": ["Matt"],
    "anthony": ["Tony"],
    "patricia": ["Pat", "Patty", "Trish"],
    "jennifer": ["Jen", "Jenny"],
    "samuel": ["Sam", "Sammy"],
    "alexander": ["Alex"],
    "benjamin": ["Ben", "Benny"],
    "nicholas": ["Nick"],
    "andrew": ["Andy", "Drew"],
    "joseph": ["Joe", "Joey"],
    "edward": ["Ed", "Eddie", "Ted"],
    "stephanie": ["Steph"],
    "rebecca": ["Becky", "Becca"],
    "victoria": ["Vicky", "Tori"],
    "deborah": ["Debbie", "Deb"],
    "susan": ["Sue", "Susie"],
    "timothy": ["Tim", "Timmy"],
    "gregory": ["Greg"],
    "kenneth": ["Ken", "Kenny"],
    "donald": ["Don", "Donnie"],
    "frederick": ["Fred", "Freddie"],
    "barbara": ["Barb", "Babs"],
    "alfred": ["Alf", "Alfie"],
}

# UK address-component abbreviations used to simulate the
# "10 Main Street" vs "10 Main St" class of duplicate.
ADDRESS_ABBREVIATIONS = {
    "Street": "St",
    "Road": "Rd",
    "Avenue": "Ave",
    "Lane": "Ln",
    "Drive": "Dr",
    "Court": "Ct",
    "Place": "Pl",
    "Square": "Sq",
    "Crescent": "Cres",
    "Gardens": "Gdns",
    "Close": "Cl",
    "Terrace": "Ter",
    "Grove": "Gr",
    "Park": "Pk",
}

# ---------------------------------------------------------------------------
# Per-record-type generation config: statuses, providers, "detail" values,
# an optional amount distribution, and how far back/how old a citizen must
# be for that record's date to plausibly fall. This is the single place
# that defines what a record of a given type looks like; `_emit_record`
# below is generic over it.
# ---------------------------------------------------------------------------
TAX_OFFICES = [
    "Ashford Tax Office",
    "Bellmoor Revenue Office",
    "Cranswick Tax Office",
    "Dunholme Revenue Office",
    "Elmsgate Tax Office",
]
BENEFITS_OFFICES = [
    "Northgate Jobcentre Plus",
    "Aldermoor Benefits Office",
    "Kingsmere Social Security Office",
    "Rosedale Benefits Centre",
    "Fenwick Jobcentre Plus",
]
GP_PRACTICES = [
    "Thornfield Group Practice",
    "Rosewell Health Centre",
    "Ferngate Medical Practice",
    "Millbrook Surgery",
    "Oakhaven Family Practice",
]
HOSPITALS = [
    "St. Aldric's Hospital",
    "Cranmoor General Hospital",
    "Whitfield Royal Infirmary",
    "Sandringham District Hospital",
    "Longmere Community Hospital",
]
PHARMACIES = [
    "Meadowbank Pharmacy",
    "Oldgate Pharmacy",
    "Silverwood Chemist",
    "Riverside Pharmacy",
    "Hollyfield Pharmacy",
]
SCHOOLS_AND_COLLEGES = [
    "Kingswood Secondary School",
    "Elmbridge College",
    "Northfield University",
    "Aldergate Sixth Form College",
    "Bramley Community College",
]
IMMIGRATION_OFFICES = [
    "Riverside Immigration Office",
    "Home Office Visa Section",
    "Portholme Border & Immigration Office",
    "Castlegate Immigration Centre",
]
LICENSING_OFFICES = [
    "Swansea Driver Licensing Centre",
    "Northshire Licensing Office",
    "Eastgate Licensing Centre",
]
PASSPORT_OFFICES = [
    "Rosewell Passport Office",
    "Bellhaven Passport Office",
    "Castlemoor Passport Office",
    "Kingsford Passport Office",
]
HOUSING_AUTHORITIES = [
    "Kingsmere Housing Authority",
    "Ashfield Borough Council Housing",
    "Thornbury District Housing",
    "Millbrook Housing Trust",
]
VETERAN_OFFICES = [
    "Veterans Support Office",
    "Regiment Support Centre",
    "Ashcombe Veterans Affairs Office",
]

BENEFIT_TYPES = [
    "Universal Credit",
    "State Pension",
    "Disability Living Allowance",
    "Jobseeker's Allowance",
    "Personal Independence Payment",
    "Child Benefit",
]
MEDICATIONS = [
    "Amoxicillin 500mg",
    "Atorvastatin 20mg",
    "Metformin 500mg",
    "Salbutamol Inhaler",
    "Omeprazole 20mg",
    "Sertraline 50mg",
    "Amlodipine 5mg",
]
HOSPITAL_DEPARTMENTS = [
    "Cardiology",
    "Orthopaedics",
    "Emergency",
    "Paediatrics",
    "Oncology",
    "General Surgery",
]
QUALIFICATIONS = [
    "GCSEs",
    "A-Levels",
    "BTEC Diploma",
    "Bachelor's Degree",
    "Master's Degree",
    "Apprenticeship Certificate",
]
VISA_TYPES = [
    "Work Visa",
    "Student Visa",
    "Indefinite Leave to Remain",
    "Family Visa",
    "Asylum Application",
]
LICENCE_CATEGORIES = [
    "Category B",
    "Category B, C1",
    "Category A, B",
    "Provisional - Category B",
    "Category B, BE",
]
OCCUPATIONS = [
    "Retail Assistant",
    "Software Developer",
    "Teacher",
    "Nurse",
    "Electrician",
    "Accountant",
    "Delivery Driver",
    "Chef",
    "Administrator",
    "Civil Engineer",
]
TENANCY_TYPES = ["Council Tenancy", "Housing Association", "Private Rented (LHA)"]
SERVICE_BRANCHES = ["Army", "Royal Navy", "Royal Air Force", "Royal Marines"]
COURTS_AND_POLICE_STATIONS = [
    "Kingsmere Crown Court",
    "Ashford Magistrates' Court",
    "Northgate Police Station",
    "Bellmoor Crown Court",
    "Riverside Magistrates' Court",
]
OFFENSE_CATEGORIES = [
    "Motoring Offence",
    "Theft",
    "Public Order Offence",
    "Drug Possession",
    "Criminal Damage",
    "Assault",
    "Fraud",
]

# ---------------------------------------------------------------------------
# Per-record-type "attributes" value pools - the richer, type-specific detail
# each record carries beyond the generic status/amount/provider/detail
# columns above. See `_ATTRIBUTE_GENERATORS` (defined after the date/amount
# helper functions it depends on) for how these are assembled per record.
# ---------------------------------------------------------------------------
BLOOD_TYPES = ["O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-"]
BLOOD_TYPE_WEIGHTS = [0.35, 0.07, 0.30, 0.08, 0.08, 0.02, 0.03, 0.07]
ALLERGIES = ["None known", "Penicillin", "Nuts", "Latex", "Pollen", "Shellfish"]
ALLERGY_WEIGHTS = [0.60, 0.10, 0.08, 0.05, 0.10, 0.07]
HOSPITAL_ADMISSION_TYPES = ["Planned", "Emergency", "Day case"]
HOSPITAL_ADMISSION_WEIGHTS = [0.35, 0.35, 0.30]
HOSPITAL_VISIT_REASONS = [
    "Routine review",
    "Follow-up appointment",
    "Investigation",
    "Elective procedure",
    "Diagnostic imaging",
    "Outpatient consultation",
]
DISCHARGE_DESTINATIONS = ["Home", "Care facility", "Transferred"]
DISCHARGE_DESTINATION_WEIGHTS = [0.85, 0.08, 0.07]
PRESCRIPTION_DOSAGES = ["Once daily", "Twice daily", "Three times daily", "As required", "Once weekly"]
PRESCRIPTION_QUANTITIES = ["28", "30", "56", "60", "90"]
GRADE_POOLS: dict[str, list[str]] = {
    "GCSEs": ["9-4 (A*-C equivalent)", "5-1 (D-G equivalent)"],
    "A-Levels": ["AAB", "ABB", "BBC", "BCC", "CCD"],
    "BTEC Diploma": ["Distinction*", "Distinction", "Merit", "Pass"],
    "Bachelor's Degree": ["First Class", "2:1", "2:2", "Third Class", "Pass"],
    "Master's Degree": ["Distinction", "Merit", "Pass"],
    "Apprenticeship Certificate": ["Pass", "Distinction"],
}
STUDY_MODES = ["Full-time", "Part-time", "Distance learning"]
FUNDING_SOURCES = ["Student Loan", "Self-funded", "Employer-sponsored", "Scholarship"]
IMMIGRATION_SPONSOR_TYPES = ["Employer", "Family", "Self", "None"]
BIOMETRIC_PERMIT_STATUSES = ["Issued", "Not issued", "Pending"]
LICENCE_POINTS = [0, 3, 6, 9, 12]
LICENCE_POINTS_WEIGHTS = [0.70, 0.15, 0.08, 0.04, 0.03]
LICENCE_RESTRICTIONS = ["None", "Glasses must be worn", "Automatic only"]
LICENCE_ENDORSEMENTS = [
    "SP30 - Exceeding statutory speed limit",
    "CU80 - Using a hand-held mobile phone",
    "DR10 - Driving with excess alcohol",
    "TT99 - Miscellaneous offence",
]
PASSPORT_TYPES = ["Standard 10-year", "Frequent traveller 34-page", "Child 5-year"]
EMPLOYMENT_TYPES = ["Full-time", "Part-time", "Zero-hours", "Contract"]
EMPLOYMENT_TYPE_WEIGHTS = [0.6, 0.2, 0.1, 0.1]
HOUSING_PROPERTY_TYPES = ["Flat", "Terraced house", "Semi-detached", "Detached"]
HOUSING_LANDLORD_TYPES = ["Private landlord", "Housing association", "Local authority"]
RANKS_BY_BRANCH: dict[str, list[str]] = {
    "Army": [
        "Private", "Lance Corporal", "Corporal", "Sergeant", "Staff Sergeant",
        "Warrant Officer", "Second Lieutenant", "Lieutenant", "Captain", "Major",
    ],
    "Royal Navy": [
        "Able Seaman", "Leading Seaman", "Petty Officer", "Chief Petty Officer",
        "Sub-Lieutenant", "Lieutenant", "Lieutenant Commander",
    ],
    "Royal Air Force": [
        "Aircraftman", "Leading Aircraftman", "Corporal", "Sergeant",
        "Flight Sergeant", "Pilot Officer", "Flying Officer", "Flight Lieutenant",
    ],
    "Royal Marines": [
        "Marine", "Lance Corporal", "Corporal", "Sergeant",
        "Colour Sergeant", "Second Lieutenant", "Lieutenant",
    ],
}
DEPLOYMENT_REGIONS = ["None", "Northern Ireland", "Afghanistan", "Iraq", "Balkans"]
DEPLOYMENT_REGION_WEIGHTS = [0.4, 0.15, 0.2, 0.15, 0.1]
# A real, factual administrative classification a Veterans Affairs-equivalent
# agency tracks for benefit-entitlement purposes (like `driving_licence.points`
# or `tax_band`) - a recorded fact about an already-assessed condition, never
# a computed judgment. Must never feed `_lifestyle_summary` or any other
# derived field (see citizen_service.py's design-boundary docstring).
DISABILITY_RATINGS = ["None", "10%", "30%", "50%", "100%"]
DISABILITY_RATING_WEIGHTS = [0.55, 0.15, 0.15, 0.1, 0.05]

CRIMINAL_STATUSES = ["Convicted", "Cautioned", "Acquitted", "Case Dismissed", "Under Investigation"]
# "N/A" is included alongside real sentence types since only Convicted/
# Cautioned records get a real one (see `_criminal_record_attributes`) -
# a factual outcome classification, not a judgment about the person.
SENTENCE_TYPES = [
    "Fine", "Community Order", "Suspended Sentence", "Custodial Sentence",
    "Conditional Discharge", "Absolute Discharge",
]
DISCLOSURE_LEVELS = ["Basic Disclosure", "Standard DBS", "Enhanced DBS", "Not Applicable"]
DISCLOSURE_LEVEL_WEIGHTS = [0.4, 0.3, 0.2, 0.1]

# Per record_type: (statuses, providers-or-None, details-or-None,
# amount_range-or-None, min_age_years, lookback_years, expires,
# captures_marital_status). providers=None means "use faker.company()"
# (employment only). amount_range is (floor, median_above_floor, sigma,
# high) for `_lognormal`, or None if this record type carries no monetary
# figure. captures_marital_status is True only for the two record types a
# real agency would realistically capture this on (tax, benefits) - see
# `_emit_record`.
_RECORD_TYPE_CONFIG: dict[RecordType, dict] = {
    RecordType.TAX_RECORD: dict(
        statuses=["Up to date", "Overdue", "In dispute", "Under review"],
        providers=TAX_OFFICES,
        details=None,
        amount_range=(500.0, 3_000.0, 0.9, 80_000.0),
        min_age_years=18,
        lookback_years=4,
        expires=False,
        captures_marital_status=True,
    ),
    RecordType.BENEFITS_RECORD: dict(
        statuses=["Active", "Under Review", "Suspended", "Closed"],
        providers=BENEFITS_OFFICES,
        details=BENEFIT_TYPES,
        amount_range=(50.0, 400.0, 0.7, 3_000.0),
        min_age_years=18,
        lookback_years=6,
        expires=False,
        captures_marital_status=True,
    ),
    RecordType.HEALTHCARE_REGISTRATION: dict(
        statuses=["Active", "Inactive", "Transferred"],
        providers=GP_PRACTICES,
        details=None,
        amount_range=None,
        min_age_years=0,
        lookback_years=100,
        expires=False,
        captures_marital_status=False,
    ),
    RecordType.HOSPITAL_VISIT: dict(
        statuses=["Discharged", "Admitted", "Referred", "Outpatient"],
        providers=HOSPITALS,
        details=HOSPITAL_DEPARTMENTS,
        amount_range=None,
        min_age_years=0,
        lookback_years=5,
        expires=False,
        captures_marital_status=False,
    ),
    RecordType.PRESCRIPTION: dict(
        statuses=["Dispensed", "Pending", "Cancelled"],
        providers=PHARMACIES,
        details=MEDICATIONS,
        amount_range=None,
        min_age_years=0,
        lookback_years=2,
        expires=False,
        captures_marital_status=False,
    ),
    RecordType.EDUCATION_RECORD: dict(
        statuses=["Completed", "In Progress", "Withdrawn"],
        providers=SCHOOLS_AND_COLLEGES,
        details=QUALIFICATIONS,
        amount_range=None,
        min_age_years=16,
        lookback_years=100,
        expires=False,
        captures_marital_status=False,
    ),
    RecordType.IMMIGRATION_RECORD: dict(
        statuses=["Granted", "Pending", "Refused", "Expired"],
        providers=IMMIGRATION_OFFICES,
        details=VISA_TYPES,
        amount_range=None,
        min_age_years=0,
        lookback_years=15,
        expires=False,
        captures_marital_status=False,
    ),
    RecordType.DRIVING_LICENCE: dict(
        statuses=["Valid", "Expired", "Revoked", "Provisional"],
        providers=LICENSING_OFFICES,
        details=LICENCE_CATEGORIES,
        amount_range=None,
        min_age_years=17,
        lookback_years=100,
        expires=True,
        captures_marital_status=False,
    ),
    RecordType.PASSPORT: dict(
        statuses=["Valid", "Expired", "Lost/Stolen", "Revoked"],
        providers=PASSPORT_OFFICES,
        details=None,
        amount_range=None,
        min_age_years=0,
        lookback_years=10,
        expires=True,
        captures_marital_status=False,
    ),
    RecordType.EMPLOYMENT_REGISTRATION: dict(
        statuses=["Employed", "Self-Employed", "Unemployed", "Seeking Work"],
        providers=None,
        details=OCCUPATIONS,
        amount_range=(16_000.0, 14_000.0, 0.6, 150_000.0),
        min_age_years=16,
        lookback_years=100,
        expires=False,
        captures_marital_status=False,
    ),
    RecordType.HOUSING_BENEFIT: dict(
        statuses=["Active", "Under Review", "Ended"],
        providers=HOUSING_AUTHORITIES,
        details=TENANCY_TYPES,
        amount_range=(100.0, 500.0, 0.6, 2_000.0),
        min_age_years=18,
        lookback_years=5,
        expires=False,
        captures_marital_status=False,
    ),
    RecordType.VETERAN_RECORD: dict(
        statuses=["Discharged - Honourable", "Reserve", "Medically Discharged"],
        providers=VETERAN_OFFICES,
        details=SERVICE_BRANCHES,
        amount_range=None,
        min_age_years=18,
        lookback_years=100,
        expires=False,
        captures_marital_status=False,
    ),
    RecordType.CRIMINAL_RECORD: dict(
        statuses=CRIMINAL_STATUSES,
        providers=COURTS_AND_POLICE_STATIONS,
        details=OFFENSE_CATEGORIES,
        # No generic `amount` here - a fine is only meaningful for one
        # specific sentence_type, decided inside the attributes generator,
        # so it lives in `attributes["fine_amount"]` instead (see
        # `_criminal_record_attributes`) rather than being sampled
        # unconditionally like every other type's `amount`.
        amount_range=None,
        min_age_years=18,
        lookback_years=100,
        expires=False,
        captures_marital_status=False,
    ),
}


def get_connection() -> duckdb.DuckDBPyConnection:
    """Open a fresh connection to the on-disk DuckDB store.

    DuckDB file connections are cheap to open/close and only support a
    single writer at a time; opening a short-lived connection per
    request (rather than holding one open for the app's lifetime) keeps
    the demo simple and avoids cross-request lock contention.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(DB_PATH))


def _vary_first_name(first_name: str, rng: random.Random, noise: NoiseProfile = DEFAULT_NOISE) -> str:
    """Return a noisy variant of a first name: full / nickname / initial / case."""
    key = first_name.lower()
    roll = rng.random()
    if roll < noise.first_name_unchanged_prob:
        return first_name
    if key in NICKNAMES and roll < noise.first_name_nickname_prob:
        return rng.choice(NICKNAMES[key])
    if roll < noise.first_name_initial_prob:
        return first_name[0]
    return first_name.upper() if rng.random() < 0.5 else first_name.lower()


def _vary_last_name(last_name: str, rng: random.Random, noise: NoiseProfile = DEFAULT_NOISE) -> str:
    """Return a noisy variant of a surname: mostly unchanged, occasional case/typo."""
    roll = rng.random()
    if roll < noise.last_name_unchanged_prob:
        return last_name
    if roll < noise.last_name_case_variant_prob:
        return last_name.upper() if rng.random() < 0.5 else last_name.lower()
    # Single-character transcription typo (swap two adjacent letters).
    if len(last_name) > 3:
        i = rng.randrange(1, len(last_name) - 1)
        chars = list(last_name)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        return "".join(chars)
    return last_name


def _vary_address(address: str, rng: random.Random, noise: NoiseProfile = DEFAULT_NOISE) -> str:
    """Abbreviate street-type tokens with ~ADDRESS_ABBREVIATION_PROB probability per occurrence."""
    out = address
    for full, abbr in ADDRESS_ABBREVIATIONS.items():
        if full in out and rng.random() < noise.address_abbreviation_prob:
            out = out.replace(full, abbr)
    return out


def _vary_postcode(postcode: str, rng: random.Random, noise: NoiseProfile = DEFAULT_NOISE) -> str:
    """UK postcodes are sometimes typed without the internal space."""
    if rng.random() < noise.postcode_strip_space_prob:
        return postcode.replace(" ", "")
    return postcode


def _make_email(first_name: str, last_name: str, rng: random.Random, noise: NoiseProfile = DEFAULT_NOISE) -> str:
    domain = rng.choice(EMAIL_DOMAINS)
    style = rng.choice(["dot", "nodot", "initial", "underscore"])
    f, l = first_name.lower(), last_name.lower()
    if style == "dot":
        local = f"{f}.{l}"
    elif style == "nodot":
        local = f"{f}{l}"
    elif style == "initial":
        local = f"{f[0]}{l}"
    else:
        local = f"{f}_{l}"
    if rng.random() < noise.email_numeric_suffix_prob:
        local += str(rng.randint(1, 99))
    return f"{local}@{domain}"


def _vary_email(canonical_email: str, first_name: str, last_name: str, rng: random.Random, noise: NoiseProfile = DEFAULT_NOISE) -> str:
    """Either reuse the canonical address verbatim, or regenerate a same-person
    variant under a different formatting convention (simulating a different
    agency's email-issuing system)."""
    if rng.random() < noise.email_reuse_prob:
        return canonical_email
    return _make_email(first_name, last_name, rng, noise)


def _vary_phone(canonical_phone: str, rng: random.Random, noise: NoiseProfile = DEFAULT_NOISE) -> str:
    """Either reuse the canonical number, or strip formatting characters
    (spaces/brackets/dashes) to simulate a different system's storage format."""
    if rng.random() < noise.phone_reuse_prob:
        return canonical_phone
    return "".join(ch for ch in canonical_phone if ch not in " ()-")


def _vary_dob(iso_date: str, rng: random.Random, noise: NoiseProfile = DEFAULT_NOISE) -> str:
    """Introduce an occasional realistic data-entry error: day/month transposed."""
    if rng.random() < noise.dob_unchanged_prob:
        return iso_date
    year, month, day = iso_date.split("-")
    if day <= "12":
        return f"{year}-{day}-{month}"
    return iso_date


def _noisy_identity_capture(person: dict, rng: random.Random, noise: NoiseProfile = DEFAULT_NOISE) -> dict:
    """Independently capture one agency system's noisy snapshot of a
    citizen's identity - the same noise model originally inlined in the
    first record-emitting loop, factored out so every record-emitting call
    site (all thirteen record types) shares one seven-field noise/nulling
    path."""
    first_name_noisy = _vary_first_name(person["first_name"], rng, noise)
    last_name_noisy = _vary_last_name(person["last_name"], rng, noise)
    dob_noisy = _vary_dob(person["date_of_birth"], rng, noise)
    address_noisy = _vary_address(person["address"], rng, noise)
    postcode_noisy = _vary_postcode(person["postcode"], rng, noise)
    email_noisy = _vary_email(person["email"], person["first_name"], person["last_name"], rng, noise)
    phone_noisy = _vary_phone(person["phone"], rng, noise)

    # Missing-value simulation: independently null out phone/email/address.
    return {
        "first_name": first_name_noisy,
        "last_name": last_name_noisy,
        "date_of_birth": dob_noisy,
        "email": None if rng.random() < noise.email_null_prob else email_noisy,
        "phone": None if rng.random() < noise.phone_null_prob else phone_noisy,
        "address": None if rng.random() < noise.address_null_prob else address_noisy,
        "city": person["city"],
        "postcode": postcode_noisy,
    }


def _lognormal(rng: random.Random, floor: float, median_above_floor: float, sigma: float, high: float) -> float:
    """Sample a right-skewed value: `floor` plus a lognormal-distributed
    excess over it (median `median_above_floor`, spread `sigma`), capped at
    `high`. Amounts like tax paid or monthly benefits are not uniformly
    distributed in reality - most sit well below the maximum, tapering
    smoothly down toward a floor with a shrinking population stretching
    out toward the top.

    A *shifted* lognormal (rather than a plain lognormal clamped at the
    floor) matters here: clamping piles up a spike of values sitting exactly
    on the floor, which looks obviously synthetic in a histogram; shifting
    means the floor is simply unreachable, with density tapering toward it.
    """
    return min(high, floor + rng.lognormvariate(math.log(median_above_floor), sigma))


def _sample_agencies(rng: random.Random) -> list[Agency]:
    """Pick which agencies hold a record for this citizen: an independent
    Bernoulli draw per agency (not a fixed enumerated bundle), clamped to
    1-6 agencies. Every draw uses the shared seeded `rng` - never Python's
    unseeded global `random` module - so the whole dataset stays
    reproducible given a fixed seed."""
    selected = [agency for agency, p in AGENCY_INCLUSION_PROBABILITY.items() if rng.random() < p]
    if not selected:
        # Force-include at least one agency, weighted toward the
        # near-universal ones rather than picking uniformly at random.
        agencies = list(AGENCY_INCLUSION_PROBABILITY.keys())
        weights = list(AGENCY_INCLUSION_PROBABILITY.values())
        selected = [rng.choices(agencies, weights=weights, k=1)[0]]
    if len(selected) > MAX_AGENCIES_PER_CITIZEN:
        selected = rng.sample(selected, MAX_AGENCIES_PER_CITIZEN)
    return selected


def _random_date_between(rng: random.Random, start: date_cls, end: date_cls) -> date_cls:
    if end <= start:
        return start
    return start + timedelta(days=rng.randint(0, (end - start).days))


def _add_years(d: date_cls, years: int) -> date_cls:
    """`d` shifted by `years`, clamping Feb 29 -> Feb 28 in non-leap target
    years rather than raising."""
    try:
        return d.replace(year=d.year + years)
    except ValueError:
        return d.replace(year=d.year + years, day=28)


def _sample_record_date(rng: random.Random, dob: date_cls, min_age_years: int, lookback_years: int) -> date_cls:
    """A plausible date for a record's event: no earlier than the citizen
    turning `min_age_years` old, and no earlier than `lookback_years`
    before `AS_OF_DATE` - whichever constraint is tighter - and never
    later than `AS_OF_DATE`."""
    earliest_by_age = _add_years(dob, min_age_years)
    earliest_by_lookback = _add_years(AS_OF_DATE, -lookback_years)
    start = min(max(earliest_by_age, earliest_by_lookback), AS_OF_DATE)
    return _random_date_between(rng, start, AS_OF_DATE)


def _make_agency_reference(record_type: RecordType, rng: random.Random) -> str:
    """A realistic-looking, agency-specific reference number format -
    different agencies use entirely different identifier schemes for the
    same citizen, exactly the "different identifiers" data-quality issue
    real government data consolidation runs into."""
    if record_type == RecordType.TAX_RECORD:
        return str(rng.randint(1_000_000_000, 9_999_999_999))
    if record_type == RecordType.BENEFITS_RECORD:
        letters = "".join(rng.choices(string.ascii_uppercase, k=2))
        return f"{letters}{rng.randint(100000, 999999)}{rng.choice(string.ascii_uppercase)}"
    if record_type == RecordType.HEALTHCARE_REGISTRATION:
        return str(rng.randint(1_000_000_000, 9_999_999_999))
    if record_type == RecordType.HOSPITAL_VISIT:
        return f"VIS{rng.randint(100000, 999999)}"
    if record_type == RecordType.PRESCRIPTION:
        return f"RX{rng.randint(100000, 999999)}"
    if record_type == RecordType.EDUCATION_RECORD:
        return f"EDU{rng.randint(100000, 999999)}"
    if record_type == RecordType.IMMIGRATION_RECORD:
        return f"IMM{rng.randint(100000, 999999)}"
    if record_type == RecordType.DRIVING_LICENCE:
        letters = "".join(rng.choices(string.ascii_uppercase, k=5))
        return f"{letters}{rng.randint(100000000, 999999999)}"
    if record_type == RecordType.PASSPORT:
        return str(rng.randint(100_000_000, 999_999_999))
    if record_type == RecordType.EMPLOYMENT_REGISTRATION:
        return f"EMP{rng.randint(100000, 999999)}"
    if record_type == RecordType.HOUSING_BENEFIT:
        return f"HSG{rng.randint(100000, 999999)}"
    if record_type == RecordType.VETERAN_RECORD:
        return f"SN{rng.randint(100000, 999999)}"
    if record_type == RecordType.CRIMINAL_RECORD:
        return f"CJ{rng.randint(100000, 999999)}"
    raise ValueError(f"Unknown record_type: {record_type}")


# ---------------------------------------------------------------------------
# Per-record-type "attributes" generators - the richer, type-specific detail
# each record carries (tax band, blood type, service rank, ...) beyond the
# generic status/amount/provider/detail columns. Every generator takes only
# the shared seeded `rng`/`faker` instances plus a `ctx` dict of values
# `_emit_record` has already sampled for this record (dob, record_date,
# status, detail, marital_status), so a generator can stay internally
# consistent with its own record (e.g. driving-licence endorsements match
# points; veteran rank matches the service branch already in `detail`).
# Every value returned is a plain, already-stringified administrative fact -
# never a score, ranking, or prediction (see citizen_service.py's
# module docstring for the project-wide policy this extends).
# ---------------------------------------------------------------------------
def _tax_attributes(rng: random.Random, faker: Faker, ctx: dict) -> dict[str, str]:
    record_date = ctx["record_date"]
    tax_year_start = record_date.year if record_date.month >= 4 else record_date.year - 1
    tax_year = f"{tax_year_start}/{str(tax_year_start + 1)[-2:]}"
    income_declared = _lognormal(rng, 8_000.0, 22_000.0, 0.5, 200_000.0)
    if income_declared < 12_570:
        tax_band = "Non-Taxpayer"
    elif income_declared < 50_270:
        tax_band = "Basic Rate"
    elif income_declared < 125_140:
        tax_band = "Higher Rate"
    else:
        tax_band = "Additional Rate"
    allowance_pool = ["Personal Allowance", "Marriage Allowance", "None"]
    allowance_weights = [0.6, 0.25, 0.15] if ctx["marital_status"] in ("Married", "Civil Partnership") else [0.75, 0.05, 0.20]
    return {
        "tax_year": tax_year,
        "income_declared": f"£{income_declared:,.2f}",
        "tax_band": tax_band,
        "filing_method": rng.choices(["Online", "Paper", "Agent-filed"], weights=[0.75, 0.15, 0.10])[0],
        "allowances_claimed": rng.choices(allowance_pool, weights=allowance_weights)[0],
    }


def _benefits_attributes(rng: random.Random, faker: Faker, ctx: dict) -> dict[str, str]:
    household_size = rng.randint(1, 6)
    dependents_count = rng.randint(0, max(0, household_size - 1))
    review_date = ctx["record_date"] + timedelta(days=rng.randint(180, 540))
    return {
        "household_size": str(household_size),
        "dependents_count": str(dependents_count),
        "payment_frequency": rng.choices(["Weekly", "Monthly", "Four-weekly"], weights=[0.3, 0.5, 0.2])[0],
        "review_date": review_date.isoformat(),
        "claim_channel": rng.choices(["Online", "Phone", "In-person"], weights=[0.6, 0.25, 0.15])[0],
    }


def _healthcare_attributes(rng: random.Random, faker: Faker, ctx: dict) -> dict[str, str]:
    return {
        "blood_type": rng.choices(BLOOD_TYPES, weights=BLOOD_TYPE_WEIGHTS)[0],
        "allergies": rng.choices(ALLERGIES, weights=ALLERGY_WEIGHTS)[0],
        "registration_type": rng.choices(["NHS", "Private", "Both"], weights=[0.85, 0.05, 0.10])[0],
        "interpreter_required": rng.choices(["Yes", "No"], weights=[0.05, 0.95])[0],
    }


def _hospital_visit_attributes(rng: random.Random, faker: Faker, ctx: dict) -> dict[str, str]:
    admission_type = rng.choices(HOSPITAL_ADMISSION_TYPES, weights=HOSPITAL_ADMISSION_WEIGHTS)[0]
    length_of_stay = "0" if admission_type == "Day case" else str(rng.randint(1, 10))
    return {
        "admission_type": admission_type,
        "reason_for_visit": rng.choice(HOSPITAL_VISIT_REASONS),
        "length_of_stay_days": length_of_stay,
        "discharge_destination": rng.choices(DISCHARGE_DESTINATIONS, weights=DISCHARGE_DESTINATION_WEIGHTS)[0],
        "attending_clinician": faker.name(),
    }


def _prescription_attributes(rng: random.Random, faker: Faker, ctx: dict) -> dict[str, str]:
    return {
        "dosage": rng.choice(PRESCRIPTION_DOSAGES),
        "prescriber": faker.name(),
        "repeat_prescription": rng.choices(["Yes", "No"], weights=[0.4, 0.6])[0],
        "quantity_supplied": rng.choice(PRESCRIPTION_QUANTITIES),
    }


def _education_attributes(rng: random.Random, faker: Faker, ctx: dict) -> dict[str, str]:
    qualification = ctx["detail"]
    dob = ctx["dob"]
    record_date = ctx["record_date"]
    year_started = max(dob.year + 16, record_date.year - rng.randint(1, 4))
    return {
        "grade_or_classification": rng.choice(GRADE_POOLS.get(qualification, ["Pass"])),
        "study_mode": rng.choices(STUDY_MODES, weights=[0.7, 0.2, 0.1])[0],
        "year_started": str(year_started),
        "funding_source": rng.choices(FUNDING_SOURCES, weights=[0.4, 0.35, 0.15, 0.1])[0],
    }


def _immigration_attributes(rng: random.Random, faker: Faker, ctx: dict) -> dict[str, str]:
    record_date = ctx["record_date"]
    dob = ctx["dob"]
    entry_start = max(_add_years(record_date, -10), dob)
    entry_date = _random_date_between(rng, entry_start, record_date)
    return {
        "nationality": faker.country(),
        "sponsor_type": rng.choices(IMMIGRATION_SPONSOR_TYPES, weights=[0.45, 0.25, 0.25, 0.05])[0],
        "entry_date": entry_date.isoformat(),
        "biometric_residence_permit": rng.choices(BIOMETRIC_PERMIT_STATUSES, weights=[0.7, 0.1, 0.2])[0],
    }


def _driving_licence_attributes(rng: random.Random, faker: Faker, ctx: dict) -> dict[str, str]:
    points = rng.choices(LICENCE_POINTS, weights=LICENCE_POINTS_WEIGHTS)[0]
    endorsements = "None" if points == 0 else rng.choice(LICENCE_ENDORSEMENTS)
    first_issued = _random_date_between(rng, _add_years(ctx["dob"], 17), ctx["record_date"])
    return {
        "points": str(points),
        "endorsements": endorsements,
        "restrictions": rng.choices(LICENCE_RESTRICTIONS, weights=[0.85, 0.10, 0.05])[0],
        "first_issued_date": first_issued.isoformat(),
    }


def _passport_attributes(rng: random.Random, faker: Faker, ctx: dict) -> dict[str, str]:
    age_at_record = _years_between_dates(ctx["dob"], ctx["record_date"])
    if age_at_record < 16:
        passport_type = "Child 5-year"
    else:
        passport_type = rng.choices(["Standard 10-year", "Frequent traveller 34-page"], weights=[0.85, 0.15])[0]
    return {
        "nationality": "British" if rng.random() < 0.9 else faker.country(),
        "issue_country": "United Kingdom" if rng.random() < 0.95 else faker.country(),
        "passport_type": passport_type,
    }


def _employment_attributes(rng: random.Random, faker: Faker, ctx: dict) -> dict[str, str]:
    employment_type = rng.choices(EMPLOYMENT_TYPES, weights=EMPLOYMENT_TYPE_WEIGHTS)[0]
    if ctx["status"] in ("Unemployed", "Seeking Work"):
        probation_status = "N/A"
    else:
        probation_status = rng.choices(["Completed", "In progress"], weights=[0.85, 0.15])[0]
    if employment_type == "Contract":
        contract_end_date = (ctx["record_date"] + timedelta(days=rng.randint(90, 730))).isoformat()
    else:
        contract_end_date = "N/A"
    return {
        "employment_type": employment_type,
        "probation_status": probation_status,
        "contract_end_date": contract_end_date,
    }


def _housing_benefit_attributes(rng: random.Random, faker: Faker, ctx: dict) -> dict[str, str]:
    return {
        "household_members": str(rng.randint(1, 6)),
        "property_type": rng.choices(HOUSING_PROPERTY_TYPES, weights=[0.35, 0.30, 0.25, 0.10])[0],
        "bedrooms": str(rng.choices([1, 2, 3, 4, 5], weights=[0.25, 0.35, 0.25, 0.1, 0.05])[0]),
        "landlord_type": rng.choices(HOUSING_LANDLORD_TYPES, weights=[0.35, 0.35, 0.30])[0],
        "weekly_rent": f"£{_lognormal(rng, 60.0, 90.0, 0.4, 400.0):,.2f}",
    }


def _veteran_attributes(rng: random.Random, faker: Faker, ctx: dict) -> dict[str, str]:
    branch = ctx["detail"]
    dob = ctx["dob"]
    record_date = ctx["record_date"]
    service_start_year = rng.randint(dob.year + 18, max(dob.year + 18, record_date.year))
    service_end_year = min(service_start_year + rng.randint(2, 25), record_date.year)
    return {
        "rank": rng.choice(RANKS_BY_BRANCH.get(branch, ["Private"])),
        "service_start_year": str(service_start_year),
        "service_end_year": str(service_end_year),
        "deployment_regions": rng.choices(DEPLOYMENT_REGIONS, weights=DEPLOYMENT_REGION_WEIGHTS)[0],
        "disability_rating": rng.choices(DISABILITY_RATINGS, weights=DISABILITY_RATING_WEIGHTS)[0],
    }


def _criminal_record_attributes(rng: random.Random, faker: Faker, ctx: dict) -> dict[str, str]:
    """Plain factual outcome/sentencing detail for one criminal-justice
    case record. `rehabilitation_status` reflects the real Rehabilitation
    of Offenders Act concept (whether a conviction is legally considered
    "spent") - a recorded administrative fact, not a judgment. Only
    Convicted/Cautioned records get a real sentence - everything else is
    "N/A"/"Not Applicable", the same "N/A unless applicable" discipline
    used for e.g. `contract_end_date`/`length_of_stay_days` elsewhere in
    this file."""
    status = ctx["status"]
    if status in ("Convicted", "Cautioned"):
        sentence_type = rng.choice(SENTENCE_TYPES)
        years_since = _years_between_dates(ctx["record_date"], AS_OF_DATE)
        rehabilitation_status = "Spent" if years_since >= rng.randint(2, 7) else "Unspent"
    else:
        sentence_type = "N/A"
        rehabilitation_status = "Not Applicable"

    if sentence_type in ("Custodial Sentence", "Suspended Sentence"):
        sentence_length = f"{rng.choice([3, 6, 12, 18, 24, 36])} months"
    else:
        sentence_length = "N/A"
    fine_amount = f"£{_lognormal(rng, 50.0, 300.0, 0.7, 5_000.0):,.2f}" if sentence_type == "Fine" else "N/A"

    return {
        "sentence_type": sentence_type,
        "sentence_length": sentence_length,
        "fine_amount": fine_amount,
        "rehabilitation_status": rehabilitation_status,
        "disclosure_level": rng.choices(DISCLOSURE_LEVELS, weights=DISCLOSURE_LEVEL_WEIGHTS)[0],
    }


def _years_between_dates(start: date_cls, end: date_cls) -> int:
    years = end.year - start.year
    if (end.month, end.day) < (start.month, start.day):
        years -= 1
    return years


_ATTRIBUTE_GENERATORS: dict[RecordType, Callable[[random.Random, Faker, dict], dict[str, str]]] = {
    RecordType.TAX_RECORD: _tax_attributes,
    RecordType.BENEFITS_RECORD: _benefits_attributes,
    RecordType.HEALTHCARE_REGISTRATION: _healthcare_attributes,
    RecordType.HOSPITAL_VISIT: _hospital_visit_attributes,
    RecordType.PRESCRIPTION: _prescription_attributes,
    RecordType.EDUCATION_RECORD: _education_attributes,
    RecordType.IMMIGRATION_RECORD: _immigration_attributes,
    RecordType.DRIVING_LICENCE: _driving_licence_attributes,
    RecordType.PASSPORT: _passport_attributes,
    RecordType.EMPLOYMENT_REGISTRATION: _employment_attributes,
    RecordType.HOUSING_BENEFIT: _housing_benefit_attributes,
    RecordType.VETERAN_RECORD: _veteran_attributes,
    RecordType.CRIMINAL_RECORD: _criminal_record_attributes,
}


# Column order for the unified `records` table.
RECORD_COLUMNS = [
    "source_record_id",
    "person_index",
    "agency",
    "record_type",
    "agency_reference_id",
    "first_name",
    "last_name",
    "date_of_birth",
    "email",
    "phone",
    "address",
    "city",
    "postcode",
    "record_date",
    "expiry_date",
    "status",
    "amount",
    "provider_name",
    "detail",
    "marital_status",
    "attributes",
]


@dataclass
class GenerationResult:
    people: int
    records: int
    persons_df: pd.DataFrame = field(repr=False)
    records_df: pd.DataFrame = field(repr=False)


def generate_all(
    seed: int = SEED,
    noise: NoiseProfile = DEFAULT_NOISE,
    n_people: int = N_PEOPLE,
    persist: bool = True,
) -> GenerationResult:
    """Generate the full synthetic dataset and (by default) persist it to DuckDB.

    Returns the row counts plus the generated frames (handy for the demo
    script, which wants to show "before Splink" duplicates without a
    second DB round-trip).

    All four arguments default to the shipped configuration, so
    `generate_all()` produces exactly the dataset it always has.

    `noise` selects how badly agencies capture identity (see `NoiseProfile`);
    `n_people` sizes the population; `persist=False` returns the frames
    without writing anything. Together those three let
    `benchmark_service.py` generate and score small datasets at varying
    noise levels without ever touching the live database - which matters,
    because persisting would drop the real `clusters`/`citizen_profiles`
    tables the running app is serving from.
    """
    rng = random.Random(seed)
    faker = Faker("en_GB")
    Faker.seed(seed)

    # --- 1. Ground-truth citizens ------------------------------------------
    persons: list[dict] = []
    for idx in range(n_people):
        first_name = faker.first_name()
        last_name = faker.last_name()
        dob = faker.date_of_birth(minimum_age=MINIMUM_AGE, maximum_age=MAXIMUM_AGE)
        address = faker.street_address().replace("\n", ", ")
        city = faker.city()
        postcode = faker.postcode()
        email = _make_email(first_name, last_name, rng, noise)
        phone = faker.phone_number()
        marital_status = rng.choices(MARITAL_STATUSES, weights=MARITAL_STATUS_WEIGHTS, k=1)[0]
        persons.append(
            {
                "person_index": idx,
                "first_name": first_name,
                "last_name": last_name,
                "date_of_birth": dob.isoformat(),
                "email": email,
                "phone": phone,
                "address": address,
                "city": city,
                "postcode": postcode,
                "marital_status": marital_status,
            }
        )
    persons_df = pd.DataFrame(persons)

    # --- 2. Noisy multi-agency government records --------------------------
    records: list[dict] = []
    record_seq = 0

    def _emit_record(idx: int, person: dict, record_type: RecordType, agency: Agency) -> None:
        nonlocal record_seq
        record_seq += 1
        config = _RECORD_TYPE_CONFIG[record_type]
        dob = date_cls.fromisoformat(person["date_of_birth"])

        record_date = _sample_record_date(rng, dob, config["min_age_years"], config["lookback_years"])
        expiry_date = _add_years(record_date, 10) if config["expires"] else None

        providers = config["providers"]
        provider_name = faker.company() if providers is None else rng.choice(providers)
        details = config["details"]
        detail = rng.choice(details) if details else None
        amount_range = config["amount_range"]
        amount = round(_lognormal(rng, *amount_range), 2) if amount_range else None
        marital_status = person["marital_status"] if config["captures_marital_status"] else None
        status = rng.choice(config["statuses"])

        attributes = _ATTRIBUTE_GENERATORS[record_type](
            rng,
            faker,
            {
                "person": person,
                "dob": dob,
                "record_date": record_date,
                "expiry_date": expiry_date,
                "status": status,
                "detail": detail,
                "marital_status": marital_status,
            },
        )

        records.append(
            {
                "source_record_id": f"REC{record_seq:06d}",
                "person_index": idx,
                "agency": agency.value,
                "record_type": record_type.value,
                "agency_reference_id": _make_agency_reference(record_type, rng),
                **_noisy_identity_capture(person, rng, noise),
                "record_date": record_date.isoformat(),
                "expiry_date": expiry_date.isoformat() if expiry_date else None,
                "status": status,
                "amount": amount,
                "provider_name": provider_name,
                "detail": detail,
                "marital_status": marital_status,
                "attributes": attributes,
            }
        )

    for idx, person in enumerate(persons):
        for agency in _sample_agencies(rng):
            if agency == Agency.HEALTHCARE:
                _emit_record(idx, person, RecordType.HEALTHCARE_REGISTRATION, agency)
                n_visits = rng.randint(*HOSPITAL_VISIT_COUNT_RANGE)
                for _ in range(n_visits):
                    _emit_record(idx, person, RecordType.HOSPITAL_VISIT, agency)
                n_prescriptions = rng.randint(*PRESCRIPTION_COUNT_RANGE)
                for _ in range(n_prescriptions):
                    _emit_record(idx, person, RecordType.PRESCRIPTION, agency)
            else:
                record_type = AGENCY_RECORD_TYPES[agency][0]
                _emit_record(idx, person, record_type, agency)

    records_df = pd.DataFrame(records)[RECORD_COLUMNS]

    # --- 3. Persist everything to DuckDB ------------------------------------
    if not persist:
        return GenerationResult(
            people=len(persons_df),
            records=len(records_df),
            persons_df=persons_df,
            records_df=records_df,
        )

    conn = get_connection()
    try:
        _persist(conn, "citizens", persons_df)
        _persist(conn, "records", records_df)
        # Downstream linkage/profile/anomaly tables are now stale - drop them
        # so the API can detect "linkage hasn't been (re)run since the last
        # generation" rather than serving results against old data.
        # anomaly_results in particular is derived from citizen_profiles (via
        # the Isolation Forest fit), so leaving it behind would let
        # /anomaly-detection try to join against a citizen_profiles table
        # that no longer exists.
        for stale_table in ("clusters", "linkage_edges", "citizen_profiles", "anomaly_results"):
            conn.execute(f"DROP TABLE IF EXISTS {stale_table}")
    finally:
        conn.close()

    return GenerationResult(
        people=len(persons_df),
        records=len(records_df),
        persons_df=persons_df,
        records_df=records_df,
    )


def _persist(conn: duckdb.DuckDBPyConnection, table_name: str, df: pd.DataFrame) -> None:
    view_name = f"_{table_name}_incoming"
    conn.register(view_name, df)
    conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM {view_name}")
    conn.unregister(view_name)


def has_generated_data() -> bool:
    conn = get_connection()
    try:
        tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
        return "records" in tables
    finally:
        conn.close()
