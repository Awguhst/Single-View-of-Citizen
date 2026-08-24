"""API request/response schemas (the FastAPI/OpenAPI contract layer).

Kept separate from `models.py` so the internal domain shape (DuckDB
tables) can evolve independently of what the API promises callers.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateDataResponse(BaseModel):
    """Response for POST /generate-data."""

    people: int = Field(..., description="Number of unique ground-truth citizens generated.")
    records: int = Field(..., description="Total noisy government records generated across all agencies.")


class RunLinkageResponse(BaseModel):
    """Response for POST /run-linkage."""

    clusters: int = Field(..., description="Number of resolved master_citizen_id clusters.")
    duplicates_found: int = Field(
        ..., description="Records identified as duplicates of another record (records - clusters)."
    )


class CitizenProfileResponse(BaseModel):
    """Response for GET /citizen/{master_citizen_id} - the Unified Citizen
    Profile: identity, linkage provenance, and a status/list summary of
    every government service this citizen is known to use."""

    master_citizen_id: str
    preferred_name: str
    date_of_birth: str
    age: int = Field(..., description="Computed at read time from date_of_birth against today's date; not stored.")
    marital_status: str | None = Field(
        None, description="From the citizen's most recent tax or benefits record, where on file."
    )
    current_address: str | None
    current_city: str | None
    current_postcode: str | None
    linked_agencies: list[str] = Field(..., description="Agencies whose records were clustered into this citizen.")
    agency_count: int = Field(..., description="Number of distinct agencies linked to this citizen.")
    engagement_tier: str = Field(..., description="e.g. 'Moderate Engagement' - see EngagementTierSummary.")
    confidence_score: float = Field(..., description="Average per-record linkage confidence for this cluster.")
    record_count: int = Field(..., description="Number of source records linked into this cluster.")
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


class SearchResponse(BaseModel):
    query: str
    results: list[CitizenProfileResponse]
    total: int = Field(
        ..., description="Total profiles matching the query (or all profiles, if query is empty) - may exceed len(results)."
    )


class DashboardSummaryResponse(BaseModel):
    """Response for GET /dashboard."""

    total_citizens: int = Field(..., description="Ground-truth citizens generated (ALWAYS the data-gen figure).")
    government_records: int = Field(..., description="Total noisy records ingested across all agencies.")
    resolved_citizens: int = Field(..., description="Resolved master_citizen_id clusters after Splink linkage.")
    avg_match_probability: float = Field(..., description="Mean per-record linkage confidence across all clusters.")
    duplicates_eliminated: int = Field(..., description="Total linked records minus resolved clusters.")
    participating_agencies: int = Field(..., description="Number of distinct agencies represented in the dataset.")
    avg_agencies_per_citizen: float = Field(..., description="Mean number of distinct agencies linked per resolved citizen.")
    agency_record_counts: dict[str, int] = Field(
        ..., description="Number of source records contributed by each agency."
    )


class LinkedRecord(BaseModel):
    """One agency's raw noisy record, as linked into a resolved profile,
    distinguished by `record_type`. This is a genuinely Splink-resolved
    cluster member; `match_probability` is this record's own per-record
    linkage confidence, not a trusted ground-truth attachment. `status`,
    `amount`, `provider_name`, and `detail` are generic columns whose
    meaning varies by `record_type` (see `app/data_generator.py`)."""

    source_record_id: str
    agency: str
    record_type: str = Field(
        ...,
        description=(
            "One of 'TAX_RECORD', 'BENEFITS_RECORD', 'HEALTHCARE_REGISTRATION', 'HOSPITAL_VISIT', "
            "'PRESCRIPTION', 'EDUCATION_RECORD', 'IMMIGRATION_RECORD', 'DRIVING_LICENCE', 'PASSPORT', "
            "'EMPLOYMENT_REGISTRATION', 'HOUSING_BENEFIT', 'VETERAN_RECORD', 'CRIMINAL_RECORD'."
        ),
    )
    agency_reference_id: str | None
    first_name: str
    last_name: str
    date_of_birth: str
    email: str | None
    phone: str | None
    address: str | None
    city: str | None
    postcode: str | None
    record_date: str
    expiry_date: str | None
    status: str | None
    amount: float | None
    provider_name: str | None
    detail: str | None
    attributes: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Per-record-type administrative detail captured by the issuing agency "
            "(e.g. tax_year, blood_type, rank) - see app/data_generator.py's "
            "_ATTRIBUTE_GENERATORS. Plain administrative facts only - never a "
            "score, ranking, or prediction."
        ),
    )
    match_probability: float = Field(..., description="This record's own per-record linkage confidence.")


class FieldAgreement(BaseModel):
    """Whether a given identity field agreed across all of a cluster's linked
    records, or varied (and if so, the distinct values seen)."""

    field: str
    is_consistent: bool
    distinct_values: list[str]


class LifestyleSummaryItem(BaseModel):
    """One plain-language, evidence-cited observation about a citizen's
    recorded government service usage - a threshold check over a count or
    value already present elsewhere on the profile, never a score,
    ranking, or predictive judgment. See citizen_service.py's module
    docstring for the explicit policy against risk/likelihood scoring."""

    tag: str = Field(..., description="Short, neutral, factual label, e.g. 'Frequent healthcare service user'.")
    basis: str = Field(
        ..., description="The exact underlying evidence this tag restates, e.g. '3 hospital visits on record.'"
    )


class CitizenProfileDetailResponse(CitizenProfileResponse):
    """Response for GET /citizen/{master_citizen_id}/detail - the full
    profile dossier page: everything in CitizenProfileResponse plus every
    linked agency record (sorted chronologically - powers the profile
    timeline), a field-level match explanation, and a transparent,
    evidence-cited lifestyle/engagement summary."""

    records: list[LinkedRecord] = Field(
        ..., description="Every record linked into this profile, across all agencies, sorted by record_date ascending."
    )
    lifestyle_summary: list[LifestyleSummaryItem] = Field(
        ...,
        description="Plain-language, evidence-cited observations derived from this profile's own fields - never a score or prediction.",
    )
    field_agreement: list[FieldAgreement] = Field(
        ..., description="Field-by-field agreement across every linked record."
    )


class QualityHistogramBucket(BaseModel):
    label: str
    count: int


class ReviewQueueItem(BaseModel):
    """A low-confidence cluster surfaced for manual analyst review."""

    master_citizen_id: str
    name: str
    match_probability: float
    record_count: int
    linked_agencies: list[str]


class QualityResponse(BaseModel):
    """Response for GET /quality - feeds the Data Quality dashboard page."""

    match_probability_histogram: list[QualityHistogramBucket]
    cluster_size_distribution: list[QualityHistogramBucket]
    review_queue: list[ReviewQueueItem] = Field(
        ..., description="Lowest-confidence clusters, worst first - candidates for manual analyst review."
    )
    total_clusters: int = Field(..., description="Total resolved master_citizen_id clusters in the current dataset.")
    avg_match_probability: float = Field(..., description="Mean per-record linkage confidence across all clusters.")
    multi_record_cluster_count: int = Field(
        ..., description="Clusters with 2 or more linked records of any type - i.e. genuinely deduplicated identities."
    )
    high_confidence_pct: float = Field(
        ..., description="Percentage of linked records with match probability >= 0.99."
    )


class EngagementTierSummary(BaseModel):
    """One engagement tier's aggregate stats - one card on the Engagement
    Segments page."""

    engagement_tier: str = Field(..., description="e.g. 'Moderate Engagement'.")
    min_agency_count: int | None = Field(
        None, description="Inclusive lower bound of this tier's agency_count range."
    )
    max_agency_count: int | None = Field(
        None, description="Inclusive upper bound of this tier's agency_count range, or null for the top tier."
    )
    citizen_count: int
    pct_of_population: float = Field(..., description="This tier's share of all resolved profiles, as a percentage.")
    avg_agency_count: float
    avg_confidence_score: float


class EngagementResponse(BaseModel):
    """Response for GET /engagement."""

    total_profiles: int
    tiers: list[EngagementTierSummary]


class EngagementTierMembersResponse(BaseModel):
    """Response for GET /engagement/{tier}/members."""

    engagement_tier: str
    results: list[CitizenProfileResponse]
    total: int = Field(..., description="Total members of this tier - may exceed len(results) once limit kicks in.")


class ServiceCoverageSummaryResponse(BaseModel):
    """Response for GET /service-coverage - feeds the Service Coverage
    Gaps page's summary cards and chart. Every figure here is an exact
    count over every resolved profile, not a sample of what's shown in
    the searchable table below it (that table is powered by GET /search
    and lists each citizen's gaps as plain facts).

    The frontend also reads its candidate agency names off
    `missing_record_counts` rather than restating the list, so the set of
    agencies has exactly one definition."""

    total_profiles: int
    missing_record_counts: dict[str, int] = Field(
        ...,
        description=(
            "For each of the 8 illustrative agencies (see citizen_service."
            "_SERVICE_GAP_AGENCY_PRIORITY), how many resolved citizens have "
            "zero linked records with that agency - order-independent, "
            "a citizen can be counted under more than one agency here."
        ),
    )
    next_gap_counts: dict[str, int] = Field(
        ...,
        description=(
            "How many citizens' first missing agency, in the fixed priority "
            "order, is each agency. A population-level breakdown only - an "
            "individual citizen's gaps are ranked by measured peer prevalence "
            "at GET /citizen/{id}/recommendations, not by this fixed order. "
            "Each citizen is counted under exactly one agency here (or not "
            "at all, if fully_covered_count)."
        ),
    )
    fully_covered_count: int = Field(
        ..., description="Citizens with a linked record at every one of the 8 illustrative agencies."
    )


class IdentityConflict(BaseModel):
    """An identity field whose linked records disagree with each other.

    The reviewable half of an explanation: a caseworker cannot act on
    "anomaly score 87", but they can act on "these records claim two
    different dates of birth, 1985-03-04 and 1985-04-03"."""

    field: str
    label: str = Field(..., description="Plain-language field name, e.g. 'date of birth'.")
    values: list[str] = Field(..., description="The distinct conflicting values found in the cluster.")


class AnomalyFactor(BaseModel):
    """One reason a profile scored as anomalous.

    Produced by ablating a single feature against the fitted model (see
    `anomaly_service._attribute_scores`), so `contribution` is how much this
    profile's anomaly score falls when the feature is replaced by the
    population median - a counterfactual against the real model, not a
    proxy statistic."""

    feature: str
    label: str = Field(..., description="Plain-language feature name for display.")
    value: float = Field(..., description="This profile's value for the feature.")
    population_median: float
    direction: str = Field(..., description="'above', 'below' or 'at' the population median.")
    contribution: float = Field(
        ..., description="Drop in anomaly score when this feature is neutralised. Higher = more responsible."
    )


class AnomalyProfileResult(BaseModel):
    """One resolved citizen profile's Isolation Forest result - one row of
    the Review Queue page's worklist."""

    master_citizen_id: str
    preferred_name: str
    agency_count: int
    record_count: int
    confidence_score: float = Field(
        ...,
        description=(
            "Mean Splink match confidence across this profile's records. Shown as "
            "context only - it is near-constant across the population and is "
            "deliberately NOT a model input."
        ),
    )
    anomaly_score: float = Field(
        ..., description="0-100, higher = more anomalous. Normalized from the model's decision_function."
    )
    status: str = Field(..., description="'Anomalous' or 'Normal', per the model's prediction at the chosen contamination.")
    top_factors: list[AnomalyFactor] = Field(
        default_factory=list,
        description="The features most responsible for this profile's score, strongest first.",
    )
    conflicts: list[IdentityConflict] = Field(
        default_factory=list,
        description="Identity fields where this profile's linked records actually disagree.",
    )


class AnomalyDetectionResponse(BaseModel):
    """Response for GET /anomaly-detection and POST /anomaly-detection/run -
    feeds the Review Queue page. The underlying Isolation
    Forest is fit on structural/linkage signals only (record/agency counts,
    match confidence, government service-usage counts) - never demographic
    or lifestyle fields. See app/anomaly_service.py's module docstring."""

    total_profiles_analyzed: int
    anomalies_detected: int
    normal_count: int
    pct_anomalous: float
    contamination: float = Field(..., description="The contamination parameter used for the most recent run.")
    score_distribution: list[QualityHistogramBucket] = Field(
        ..., description="Anomaly score histogram, bucketed in steps of 20 (0-20 through 80-100)."
    )
    results: list[AnomalyProfileResult] = Field(..., description="Profiles ordered by anomaly_score, worst first.")
    total: int = Field(..., description="Total profiles with an anomaly result - may exceed len(results).")


class HealthResponse(BaseModel):
    status: str
    data_generated: bool
    linkage_run: bool




# ---------------------------------------------------------------------------
# Evaluation, benchmarking, graph and recommendation schemas
#
# These back the developer/analyst-facing pages. Everything scored here is
# scored against the synthetic ground truth (`records.person_index`), which
# no product-facing endpoint ever reads - see app/evaluation_service.py.
# ---------------------------------------------------------------------------


class PartitionMetricsResponse(BaseModel):
    """Quality of one predicted grouping of records, versus ground truth."""

    method: str
    label: str
    note: str
    pairwise_precision: float = Field(
        ..., description="Of record pairs put in the same cluster, the share genuinely the same person."
    )
    pairwise_recall: float = Field(
        ..., description="Of record pairs genuinely the same person, the share put in the same cluster."
    )
    pairwise_f1: float
    predicted_clusters: int
    true_citizens: int
    exactly_resolved: int = Field(
        ..., description="Real citizens whose records landed in exactly one cluster containing nobody else."
    )
    over_split_citizens: int = Field(
        ..., description="Real citizens whose records were spread across more than one cluster."
    )
    over_merged_clusters: int = Field(
        ..., description="Clusters that merged records from more than one real citizen."
    )
    adjusted_rand_index: float


class LinkageEvaluationResponse(BaseModel):
    """Response for GET /evaluation/linkage - the Splink pipeline scored
    against ground truth alongside deterministic baselines."""

    total_records: int
    splink: PartitionMetricsResponse
    baselines: list[PartitionMetricsResponse]
    best_baseline_method: str
    best_baseline_f1: float
    f1_improvement_over_best_baseline: float


class ThresholdSweepPoint(BaseModel):
    threshold: float
    is_current: bool = Field(..., description="Whether this is the threshold the shipped pipeline uses.")
    pairwise_precision: float
    pairwise_recall: float
    pairwise_f1: float
    predicted_clusters: int
    true_citizens: int
    exactly_resolved: int
    over_split_citizens: int
    over_merged_clusters: int
    adjusted_rand_index: float


class ThresholdSweepResponse(BaseModel):
    """Response for GET /evaluation/threshold-sweep - the precision/recall
    trade-off across clustering thresholds, re-clustered from the persisted
    pairwise edges without retraining the model."""

    current_threshold: float
    best_threshold: float
    best_f1: float
    edge_count: int
    points: list[ThresholdSweepPoint]


class BenchmarkLevelResult(BaseModel):
    """One noise level's result in the linkage benchmark sweep."""

    noise_level: str
    people: int
    records: int
    splink_precision: float
    splink_recall: float
    splink_f1: float
    splink_exactly_resolved: int
    splink_over_split: int
    splink_over_merged: int
    baseline_precision: float
    baseline_recall: float
    baseline_f1: float
    f1_advantage: float = Field(
        ..., description="Splink F1 minus the deterministic baseline's F1 at this noise level."
    )
    seconds: float


class BenchmarkResponse(BaseModel):
    """Response for GET /benchmark and POST /benchmark/run."""

    population: int = Field(..., description="Citizens generated per noise level.")
    seed: int
    baseline_method: str
    levels: list[BenchmarkLevelResult] = Field(..., description="Ordered from cleanest to noisiest.")


class DetectorResult(BaseModel):
    detector: str
    label: str
    note: str
    average_precision: float
    precision_at_k: float
    recall_at_k: float
    lift_over_random: float = Field(
        ..., description="Average precision divided by the base rate of linkage errors. 1.0 = no better than chance."
    )


class DetectorComparisonResponse(BaseModel):
    """Response for GET /anomaly-detection/evaluation - how well each
    detector surfaces clusters whose linkage actually went wrong, scored
    against the synthetic ground truth."""

    profiles: int
    linkage_errors: int = Field(..., description="Clusters that did not resolve exactly one real citizen.")
    base_rate: float
    k: int = Field(
        ..., description="Cut-off for precision@k / recall@k - the contamination share of the population."
    )
    contamination: float
    detectors: list[DetectorResult] = Field(..., description="Ordered by average precision, best first.")
    best_detector: str | None


class GraphEdgeEvidence(BaseModel):
    field: str
    agreement: str = Field(..., description="'agreed', 'disagreed', or 'not comparable' (a value was missing).")
    gamma: int = Field(..., description="Splink agreement level: -1 not comparable, 0 disagreed, higher is closer.")
    left_value: str | None
    right_value: str | None


class GraphEdge(BaseModel):
    source: str
    target: str
    match_probability: float
    evidence: list[GraphEdgeEvidence]
    agreeing_fields: list[str]
    is_load_bearing: bool = Field(
        ..., description="Whether this edge is at/above the clustering threshold, i.e. actually helped form the cluster."
    )
    is_bridge: bool = Field(
        ..., description="Whether removing this edge would split the cluster - a single point of failure."
    )


class GraphNode(BaseModel):
    source_record_id: str
    agency: str
    record_type: str
    record_date: str
    name: str
    date_of_birth: str
    degree: int


class ClusterGraphResponse(BaseModel):
    """Response for GET /citizen/{master_citizen_id}/graph - the pairwise
    linkage evidence that produced this profile."""

    master_citizen_id: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    node_count: int
    edge_count: int
    load_bearing_edge_count: int
    cluster_threshold: float
    density: float = Field(
        ...,
        description=(
            "Share of possible record pairs that are load-bearing links. "
            "1.0 means every record was confirmed against every other."
        ),
    )
    bridge_count: int
    weakest_bridge: GraphEdge | None
    min_edge_probability: float


class CoverageRecommendation(BaseModel):
    """One service-coverage gap, with the peer statistic behind its rank."""

    agency: str
    peers_with_record: int
    peer_group_size: int
    peer_prevalence: float = Field(
        ...,
        description=(
            "Share of the peer group holding a record with this agency. A descriptive "
            "population statistic, not a prediction about this citizen."
        ),
    )


class CoverageRecommendationResponse(BaseModel):
    """Response for GET /citizen/{master_citizen_id}/recommendations.

    Ranks this citizen's service-coverage gaps by how common each service is
    among citizens with a comparable service footprint. The peer group is
    built only from which agencies hold a record - never from demographic or
    lifestyle data, and never a prediction about the individual. See
    app/recommendation_service.py for the full design boundary."""

    master_citizen_id: str
    linked_agencies: list[str]
    recommendations: list[CoverageRecommendation]
    fully_covered: bool
    candidate_agencies: list[str]
    peer_group_size: int = 0
    peer_group_definition: list[str] = Field(
        default_factory=list,
        description="Agencies the peer group was conditioned on. Empty means the whole population.",
    )
    peer_group_is_population: bool = False
    backed_off: bool = Field(
        default=False, description="True if the peer group had to be widened to reach a usable size."
    )
    population: int = 0
