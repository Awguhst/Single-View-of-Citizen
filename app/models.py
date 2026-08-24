"""Domain models for the CitizenLink (Single View of Citizen) platform.

These Pydantic models describe the core entities that flow through the
system: the citizens a government's agencies separately hold records
about, the noisy per-agency records about them (tax, benefits, healthcare,
licensing, and the rest, unified into one `Record` shape), and the results
produced by the Splink entity-resolution pipeline. They are used internally
by the services (data generation, linkage, citizen aggregation) and double
up as a single source of truth for the shape of each DuckDB table.

`schemas.py` builds the public API request/response contracts on top of
these, so the two files stay in sync without duplicating field semantics.
"""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class Agency(str, Enum):
    """The eleven government agencies that independently hold citizen records.

    Member names double as short, stable codes used for
    `agency_reference_id` generation in `data_generator.py`; member values
    are the display names shown throughout the API and frontend.
    """

    REVENUE_TAX = "Revenue & Tax"
    SOCIAL_SECURITY = "Social Security"
    HEALTHCARE = "Healthcare"
    EDUCATION = "Education"
    IMMIGRATION = "Immigration"
    DRIVER_LICENSING = "Driver Licensing"
    PASSPORT_OFFICE = "Passport Office"
    EMPLOYMENT = "Employment"
    HOUSING = "Housing"
    VETERANS_AFFAIRS = "Veterans Affairs"
    CRIMINAL_JUSTICE = "Criminal Justice"


class RecordType(str, Enum):
    """The thirteen kinds of noisy record the unified `records` table can
    hold. Member values are uppercase and equal to the member name,
    matching the `record_type` column's literal values in DuckDB. Each
    type belongs to exactly one agency - see `AGENCY_RECORD_TYPES`."""

    TAX_RECORD = "TAX_RECORD"
    BENEFITS_RECORD = "BENEFITS_RECORD"
    HEALTHCARE_REGISTRATION = "HEALTHCARE_REGISTRATION"
    HOSPITAL_VISIT = "HOSPITAL_VISIT"
    PRESCRIPTION = "PRESCRIPTION"
    EDUCATION_RECORD = "EDUCATION_RECORD"
    IMMIGRATION_RECORD = "IMMIGRATION_RECORD"
    DRIVING_LICENCE = "DRIVING_LICENCE"
    PASSPORT = "PASSPORT"
    EMPLOYMENT_REGISTRATION = "EMPLOYMENT_REGISTRATION"
    HOUSING_BENEFIT = "HOUSING_BENEFIT"
    VETERAN_RECORD = "VETERAN_RECORD"
    CRIMINAL_RECORD = "CRIMINAL_RECORD"


# Single source of truth for which agency issues which record type(s).
# Healthcare is the only agency that issues more than one kind of record
# (a registration, plus repeatable hospital-visit and prescription
# records); every other agency issues exactly one. Both
# `data_generator.py` (what to generate per included agency) and
# `citizen_service.py` (which record_type feeds which profile field) key
# off this mapping so the two can never drift apart.
AGENCY_RECORD_TYPES: dict[Agency, list[RecordType]] = {
    Agency.REVENUE_TAX: [RecordType.TAX_RECORD],
    Agency.SOCIAL_SECURITY: [RecordType.BENEFITS_RECORD],
    Agency.HEALTHCARE: [
        RecordType.HEALTHCARE_REGISTRATION,
        RecordType.HOSPITAL_VISIT,
        RecordType.PRESCRIPTION,
    ],
    Agency.EDUCATION: [RecordType.EDUCATION_RECORD],
    Agency.IMMIGRATION: [RecordType.IMMIGRATION_RECORD],
    Agency.DRIVER_LICENSING: [RecordType.DRIVING_LICENCE],
    Agency.PASSPORT_OFFICE: [RecordType.PASSPORT],
    Agency.EMPLOYMENT: [RecordType.EMPLOYMENT_REGISTRATION],
    Agency.HOUSING: [RecordType.HOUSING_BENEFIT],
    Agency.VETERANS_AFFAIRS: [RecordType.VETERAN_RECORD],
    Agency.CRIMINAL_JUSTICE: [RecordType.CRIMINAL_RECORD],
}

# Reverse lookup: given a record_type, which agency issued it. Derived
# from AGENCY_RECORD_TYPES rather than hand-maintained separately.
RECORD_TYPE_AGENCY: dict[RecordType, Agency] = {
    record_type: agency for agency, record_types in AGENCY_RECORD_TYPES.items() for record_type in record_types
}


class Person(BaseModel):
    """A ground-truth individual, as known only to the data generator.

    In reality no single government system ever observes this clean
    record directly - it is reconstructed (imperfectly) by Splink from the
    noisy `Record` rows. We persist it purely so the demo can show
    "before/after" and compute linkage-quality metrics; production systems
    would not have this table.
    """

    person_index: int
    first_name: str
    last_name: str
    date_of_birth: date
    email: str
    phone: str
    address: str
    city: str
    postcode: str
    marital_status: str = Field(..., description="Ground-truth marital status, assigned once per citizen.")


class Record(BaseModel):
    """A single agency system's noisy record of one citizen, distinguished
    by `record_type`. This is the unit of record Splink deduplicates: one
    row per (person, agency, record_type) capture, carrying the 7 noisy
    identity comparison columns plus a small set of generic payload
    columns reused across every record type (`agency_reference_id`,
    `record_date`, `expiry_date`, `status`, `amount`, `provider_name`,
    `detail`, `marital_status`) - what each one means varies by
    `record_type` (see `data_generator.py`), but no per-type columns are
    needed.
    """

    model_config = ConfigDict(use_enum_values=True)

    source_record_id: str
    person_index: int = Field(..., description="Ground-truth person id; hidden from Splink.")
    agency: Agency
    record_type: RecordType
    agency_reference_id: str | None = None
    first_name: str
    last_name: str
    date_of_birth: str = Field(..., description="ISO date string; may carry data-entry noise.")
    email: str | None = None
    phone: str | None = None
    address: str | None = None
    city: str | None = None
    postcode: str | None = None
    record_date: str = Field(..., description="ISO date string; when this record's event occurred.")
    expiry_date: str | None = Field(None, description="ISO date string; licences/passports only.")
    status: str | None = None
    amount: float | None = None
    provider_name: str | None = None
    detail: str | None = None
    marital_status: str | None = Field(
        None, description="Populated only for record types where an agency would realistically capture this (tax, benefits)."
    )
    attributes: dict[str, str] | None = Field(
        None,
        description=(
            "Per-record-type administrative detail captured by the issuing agency "
            "(e.g. tax_year/blood_type/rank) - see data_generator.py's "
            "_ATTRIBUTE_GENERATORS for the full field list per record_type. Every "
            "key is a plain administrative fact; never a score, ranking, or "
            "prediction - see citizen_service.py's module docstring for the "
            "project-wide policy this extends."
        ),
    )


class ClusterAssignment(BaseModel):
    """One row of Splink's clustering output, as persisted to DuckDB.

    `match_probability` is not a native per-record Splink output (Splink
    scores pairwise edges, not records) - see `splink_service.py` for how
    it is derived as a per-record confidence score.
    """

    source_record_id: str
    master_citizen_id: str
    match_probability: float


class CitizenProfile(BaseModel):
    """The aggregated Unified Citizen Profile for one resolved person."""

    master_citizen_id: str
    preferred_name: str
    date_of_birth: str
    age: int = Field(..., description="Computed at read time from date_of_birth; not stored.")
    marital_status: str | None
    current_address: str | None
    current_city: str | None
    current_postcode: str | None
    linked_agencies: list[str]
    agency_count: int
    tax_status: str | None
    benefits_received: list[str]
    healthcare_registrations: int
    hospital_visits: int
    current_prescriptions: list[str]
    education_history: list[str]
    driving_licence_status: str | None
    passport_status: str | None
    employment_status: str | None
    housing_assistance: str | None
    veteran_status: str | None
