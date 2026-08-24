"""FastAPI application for the CitizenLink (Single View of Citizen) platform.

Government agencies each hold their own citizen records with no shared
identifier and the usual data-quality issues (name variants, address
abbreviations, missing fields). This service:

1. Generates that synthetic multi-agency government record dataset.
2. Runs a Splink probabilistic record-linkage pipeline to resolve
   duplicate identities into `master_citizen_id` clusters.
3. Aggregates each cluster's records across every agency into a single
   Unified Citizen Profile.
4. Exposes all of the above over a documented REST API (see /docs).

The dataset is generated automatically on first startup (seeded, so it
is reproducible) - no manual setup is required before exploring the API.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from app import (
    anomaly_service,
    benchmark_service,
    bootstrap,
    citizen_service,
    data_generator,
    evaluation_service,
    exports,
    graph_service,
    recommendation_service,
    splink_service,
)
from app.schemas import (
    AnomalyDetectionResponse,
    BenchmarkResponse,
    CitizenProfileDetailResponse,
    CitizenProfileResponse,
    ClusterGraphResponse,
    CoverageRecommendationResponse,
    DashboardSummaryResponse,
    DetectorComparisonResponse,
    EngagementResponse,
    EngagementTierMembersResponse,
    GenerateDataResponse,
    HealthResponse,
    LinkageEvaluationResponse,
    QualityResponse,
    RunLinkageResponse,
    SearchResponse,
    ServiceCoverageSummaryResponse,
    ThresholdSweepResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("citizenlink")

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    bootstrap.bootstrap()
    yield


app = FastAPI(
    title="CitizenLink - Single View of Citizen",
    description=(
        "Entity-resolution proof-of-concept for government agencies: links noisy "
        "multi-agency citizen records with Splink and aggregates the result into "
        "a single Unified Citizen Profile per individual."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    """Serve the dashboard frontend. The interactive API docs remain at /docs."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        data_generated=data_generator.has_generated_data(),
        linkage_run=splink_service.has_run_linkage(),
    )


@app.post(
    "/generate-data",
    response_model=GenerateDataResponse,
    tags=["Data Generation"],
)
def generate_data() -> GenerateDataResponse:
    """Generate (or regenerate) the synthetic dataset: 10,000 ground-truth
    citizens and their noisy multi-agency government records. Uses a fixed
    seed, so repeated calls are reproducible.

    Regenerating drops any existing clusters/citizen profiles, since they
    were computed against the previous dataset - call /run-linkage again
    afterwards.
    """
    result = data_generator.generate_all()
    return GenerateDataResponse(people=result.people, records=result.records)


@app.post(
    "/run-linkage",
    response_model=RunLinkageResponse,
    tags=["Entity Resolution"],
)
def run_linkage() -> RunLinkageResponse:
    """Run the Splink entity-resolution pipeline over the generated source
    records: trains the probabilistic model, predicts pairwise match
    probabilities, clusters records into `master_citizen_id` groups, and
    rebuilds citizen profiles on top of the new clusters.
    """
    if not data_generator.has_generated_data():
        raise HTTPException(status_code=400, detail="No data found. Call POST /generate-data first.")

    result = splink_service.run_full_pipeline()
    citizen_service.build_citizen_profiles()
    logger.info(
        "Linkage complete: %s clusters, %s duplicates found, avg confidence %.3f (%.1fs).",
        result.clusters,
        result.duplicates_found,
        result.avg_match_probability,
        result.training_seconds,
    )
    return RunLinkageResponse(clusters=result.clusters, duplicates_found=result.duplicates_found)


@app.get(
    "/citizen/{master_citizen_id}",
    response_model=CitizenProfileResponse,
    tags=["Citizen"],
)
def get_citizen(master_citizen_id: str) -> CitizenProfileResponse:
    """Return the Unified Citizen Profile for a resolved person,
    e.g. `/citizen/MC00001`."""
    if not citizen_service.has_citizen_profiles():
        raise HTTPException(status_code=400, detail="No citizen profiles found. Call POST /run-linkage first.")
    profile = citizen_service.get_citizen_profile(master_citizen_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"No profile found for '{master_citizen_id}'")
    return CitizenProfileResponse(**profile)


@app.get(
    "/citizen/{master_citizen_id}/detail",
    response_model=CitizenProfileDetailResponse,
    tags=["Citizen"],
)
def get_citizen_detail(master_citizen_id: str) -> CitizenProfileDetailResponse:
    """Return the full profile dossier for a resolved person: the Unified
    Citizen Profile, every linked agency record (sorted chronologically -
    powers the profile timeline), and a field-by-field explanation of how
    confidently Splink agreed those records describe the same individual."""
    if not citizen_service.has_citizen_profiles():
        raise HTTPException(status_code=400, detail="No citizen profiles found. Call POST /run-linkage first.")
    profile = citizen_service.get_citizen_profile_detail(master_citizen_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"No profile found for '{master_citizen_id}'")
    return CitizenProfileDetailResponse(**profile)


@app.get(
    "/citizen/{master_citizen_id}/export/pdf",
    tags=["Citizen"],
)
def export_citizen_pdf(master_citizen_id: str) -> Response:
    """Download the full profile dossier (same data as `/detail`) as a
    Citizen Summary Report PDF."""
    if not citizen_service.has_citizen_profiles():
        raise HTTPException(status_code=400, detail="No citizen profiles found. Call POST /run-linkage first.")
    profile = citizen_service.get_citizen_profile_detail(master_citizen_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"No profile found for '{master_citizen_id}'")
    pdf_bytes = exports.build_profile_pdf(profile)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{master_citizen_id}.pdf"'},
    )


@app.get(
    "/search",
    response_model=SearchResponse,
    tags=["Citizen"],
)
def search(
    q: str = Query(
        "", description="Name to search for, e.g. 'john smith'. Leave empty to browse all profiles alphabetically."
    ),
    limit: int = Query(50, description="Max profiles to return."),
) -> SearchResponse:
    """Search resolved profiles by name, matching both the chosen display
    name and any underlying linked source record (so a search still finds
    a cluster even if a different name variant was chosen for display).
    An empty query returns all profiles sorted alphabetically by name."""
    if not citizen_service.has_citizen_profiles():
        raise HTTPException(status_code=400, detail="No citizen profiles found. Call POST /run-linkage first.")

    rows, total = citizen_service.search_person(q, limit=limit)
    results = [CitizenProfileResponse(**row) for row in rows]
    return SearchResponse(query=q, results=results, total=total)


@app.get(
    "/dashboard",
    response_model=DashboardSummaryResponse,
    tags=["Dashboard"],
)
def dashboard() -> DashboardSummaryResponse:
    """High-level summary metrics for the CitizenLink platform: population
    size, linkage quality, and agency participation."""
    summary = citizen_service.get_dashboard_summary()
    return DashboardSummaryResponse(**summary)


@app.get(
    "/dashboard/showcase",
    response_model=CitizenProfileDetailResponse,
    tags=["Dashboard"],
)
def dashboard_showcase() -> CitizenProfileDetailResponse:
    """One representative resolved profile, picked because its linked
    records show clearly different name spellings - feeds the dashboard's
    "Before / After Splink" panel. Same shape as /citizen/{id}/detail."""
    if not splink_service.has_run_linkage():
        raise HTTPException(status_code=400, detail="No linkage results found. Call POST /run-linkage first.")
    example = citizen_service.get_showcase_example()
    if example is None:
        raise HTTPException(status_code=404, detail="No clusters available to showcase.")
    return CitizenProfileDetailResponse(**example)


@app.get(
    "/quality",
    response_model=QualityResponse,
    tags=["Dashboard"],
)
def quality(
    limit: int = Query(10, description="Max review-queue clusters to return, worst first."),
) -> QualityResponse:
    """Linkage-quality diagnostics shown on the Evaluation page: a match-confidence
    histogram, cluster-size distribution, and a manual-review queue of the
    lowest-confidence multi-record clusters."""
    if not splink_service.has_run_linkage():
        raise HTTPException(status_code=400, detail="No linkage results found. Call POST /run-linkage first.")
    return QualityResponse(**citizen_service.get_quality_metrics(review_queue_size=limit))


@app.post(
    "/anomaly-detection/run",
    response_model=AnomalyDetectionResponse,
    tags=["Dashboard"],
)
def run_anomaly_detection(
    contamination: float = Query(
        anomaly_service.DEFAULT_CONTAMINATION,
        ge=0.01,
        le=0.5,
        description="Expected proportion of anomalous profiles in the population.",
    ),
) -> AnomalyDetectionResponse:
    """(Re)fit an Isolation Forest over every resolved citizen profile's
    structural/linkage signals (record/agency counts, match confidence,
    government service-usage counts - never demographic or lifestyle
    fields, see app/anomaly_service.py) and persist the result. Feeds the
    Review Queue page.
    """
    if not citizen_service.has_citizen_profiles():
        raise HTTPException(status_code=400, detail="No citizen profiles found. Call POST /run-linkage first.")
    result = anomaly_service.run_anomaly_detection(contamination)
    return AnomalyDetectionResponse(**result)


@app.get(
    "/anomaly-detection",
    response_model=AnomalyDetectionResponse,
    tags=["Dashboard"],
)
def anomaly_detection(
    limit: int = Query(50, description="Max profiles to return, most anomalous first."),
) -> AnomalyDetectionResponse:
    """The most recently computed Isolation Forest anomaly-detection
    results. Call POST /anomaly-detection/run first to (re)compute."""
    if not anomaly_service.has_anomaly_results():
        raise HTTPException(status_code=400, detail="No anomaly results found. Call POST /anomaly-detection/run first.")
    return AnomalyDetectionResponse(**anomaly_service.get_anomaly_summary(limit))


@app.get(
    "/engagement",
    response_model=EngagementResponse,
    tags=["Citizen"],
)
def engagement() -> EngagementResponse:
    """Groups every resolved citizen profile into 5 engagement tiers, by
    how many distinct agencies hold a record for them (Minimal / Limited /
    Moderate / High / Full Engagement) and returns per-tier summary stats -
    feeds the Engagement Segments page's summary cards and charts."""
    if not citizen_service.has_citizen_profiles():
        raise HTTPException(status_code=400, detail="No citizen profiles found. Call POST /run-linkage first.")
    return EngagementResponse(**citizen_service.get_engagement_summary())


@app.get(
    "/engagement/{tier}/members",
    response_model=EngagementTierMembersResponse,
    tags=["Citizen"],
)
def engagement_members(
    tier: str,
    limit: int = Query(50, description="Max members to return, most agencies linked first."),
) -> EngagementTierMembersResponse:
    """Resolved profiles assigned to one engagement tier, e.g.
    `/engagement/Moderate Engagement/members` - powers the Engagement
    Segments drill-down table. `tier` must be one of the 5 known tier
    labels."""
    if not citizen_service.has_citizen_profiles():
        raise HTTPException(status_code=400, detail="No citizen profiles found. Call POST /run-linkage first.")
    try:
        rows, total = citizen_service.get_engagement_tier_members(tier, limit=limit)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    results = [CitizenProfileResponse(**row) for row in rows]
    return EngagementTierMembersResponse(engagement_tier=tier, results=results, total=total)


@app.get(
    "/service-coverage",
    response_model=ServiceCoverageSummaryResponse,
    tags=["Citizen"],
)
def service_coverage() -> ServiceCoverageSummaryResponse:
    """Population-wide counts of resolved citizens missing an on-file
    record with each of 8 illustrative agencies, plus a first-gap-in-fixed-
    order breakdown aggregated across the whole population.

    A plain aggregation over already-stored data - see the design boundary
    note in citizen_service.py; not a prediction of individual behavior.
    For one citizen's gaps ranked by measured peer prevalence, see
    GET /citizen/{master_citizen_id}/recommendations."""
    if not citizen_service.has_citizen_profiles():
        raise HTTPException(status_code=400, detail="No citizen profiles found. Call POST /run-linkage first.")
    return ServiceCoverageSummaryResponse(**citizen_service.get_service_coverage_summary())


@app.get(
    "/export/directory.csv",
    tags=["Exports"],
)
def export_directory_csv(
    q: str = Query("", description="Same filter as /search?q= - leave empty to export the full directory."),
) -> Response:
    """Download every resolved profile matching `q` (or all profiles) as CSV."""
    if not citizen_service.has_citizen_profiles():
        raise HTTPException(status_code=400, detail="No citizen profiles found. Call POST /run-linkage first.")
    csv_text = exports.build_directory_csv(q)
    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="directory.csv"'},
    )


@app.get(
    "/export/review-queue.csv",
    tags=["Exports"],
)
def export_review_queue_csv(
    limit: int = Query(
        exports.DEFAULT_REVIEW_QUEUE_EXPORT_LIMIT,
        description="Max number of lowest-confidence clusters to include, worst first.",
    ),
) -> Response:
    """Download the manual-review backlog (low-confidence multi-record
    clusters) as CSV - the full backlog, not just the dashboard's top 10."""
    if not splink_service.has_run_linkage():
        raise HTTPException(status_code=400, detail="No linkage results found. Call POST /run-linkage first.")
    csv_text = exports.build_review_queue_csv(limit)
    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="review-queue.csv"'},
    )


@app.get(
    "/export/engagement-members.csv",
    tags=["Exports"],
)
def export_engagement_members_csv(
    tier: str = Query(..., description="Engagement tier to export, e.g. 'Moderate Engagement' - same labels shown on the Engagement Segments page."),
) -> Response:
    """Download every resolved profile assigned to one engagement tier as CSV."""
    if not citizen_service.has_citizen_profiles():
        raise HTTPException(status_code=400, detail="No citizen profiles found. Call POST /run-linkage first.")
    try:
        csv_text = exports.build_engagement_members_csv(tier)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="engagement-{tier.lower().replace(" ", "-")}.csv"'},
    )


# ---------------------------------------------------------------------------
# Evaluation & benchmarking
#
# These endpoints are the only ones in the application that read the
# synthetic ground truth (`records.person_index`). They exist so the ML
# claims the rest of the app makes can be checked rather than asserted; no
# product-facing endpoint depends on them. See app/evaluation_service.py.
# ---------------------------------------------------------------------------


@app.get(
    "/evaluation/linkage",
    response_model=LinkageEvaluationResponse,
    tags=["Evaluation"],
)
def evaluation_linkage() -> LinkageEvaluationResponse:
    """Score the Splink pipeline against the synthetic ground truth, next to
    deterministic baselines.

    Answers the question the dashboard alone cannot: is the probabilistic
    model actually better than a hand-written matching rule, and by how
    much? Reports pairwise precision/recall/F1, how many citizens were
    resolved exactly, and the two failure modes (over-splitting and
    over-merging) separately, since they have very different costs.
    """
    if not evaluation_service.has_ground_truth():
        raise HTTPException(
            status_code=400,
            detail="No ground truth available. Call POST /generate-data first.",
        )
    try:
        return LinkageEvaluationResponse(**evaluation_service.evaluate_linkage())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get(
    "/evaluation/threshold-sweep",
    response_model=ThresholdSweepResponse,
    tags=["Evaluation"],
)
def evaluation_threshold_sweep() -> ThresholdSweepResponse:
    """Re-cluster the persisted pairwise edges at a range of thresholds and
    score each against ground truth.

    `splink_service.CLUSTER_MATCH_THRESHOLD` is one hand-picked number; this
    turns that judgement call into a measured precision/recall curve. Each
    point is a graph re-clustering rather than a model retrain, so the whole
    sweep runs in seconds.
    """
    if not evaluation_service.has_ground_truth():
        raise HTTPException(
            status_code=400,
            detail="No ground truth available. Call POST /generate-data first.",
        )
    try:
        return ThresholdSweepResponse(**evaluation_service.sweep_cluster_threshold())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post(
    "/benchmark/run",
    response_model=BenchmarkResponse,
    tags=["Evaluation"],
)
def run_benchmark(
    population: int = Query(
        benchmark_service.DEFAULT_BENCHMARK_POPULATION,
        ge=200,
        le=5000,
        description="Citizens generated per noise level. Larger is more stable but slower.",
    ),
) -> BenchmarkResponse:
    """Re-run the full linkage pipeline at every noise level and score each
    against ground truth.

    Takes roughly a minute: each noise level trains a complete Splink model.
    Nothing here touches the live dataset - every level is generated,
    linked and scored in memory.
    """
    return BenchmarkResponse(**benchmark_service.run_benchmark(n_people=population))


@app.get(
    "/benchmark",
    response_model=BenchmarkResponse,
    tags=["Evaluation"],
)
def benchmark() -> BenchmarkResponse:
    """The most recently computed noise-robustness benchmark. Call
    POST /benchmark/run first to (re)compute."""
    if not benchmark_service.has_benchmark_results():
        raise HTTPException(
            status_code=400, detail="No benchmark results found. Call POST /benchmark/run first."
        )
    return BenchmarkResponse(**benchmark_service.get_benchmark_results())


@app.get(
    "/anomaly-detection/evaluation",
    response_model=DetectorComparisonResponse,
    tags=["Evaluation"],
)
def anomaly_detection_evaluation(
    contamination: float = Query(
        anomaly_service.DEFAULT_CONTAMINATION,
        ge=0.01,
        le=0.5,
        description="Share of the population treated as flagged, for precision@k / recall@k.",
    ),
) -> DetectorComparisonResponse:
    """Compare anomaly detectors by how well they surface real linkage errors.

    Unsupervised detection normally cannot be scored. Here it can, because
    the data is synthetic: a cluster either does or does not correspond
    one-to-one with a real citizen. That gives a checkable ranking task, and
    it is the reason more than one algorithm is implemented - each fails
    differently, and the comparison is the point.
    """
    if not citizen_service.has_citizen_profiles():
        raise HTTPException(status_code=400, detail="No citizen profiles found. Call POST /run-linkage first.")
    if not evaluation_service.has_ground_truth():
        raise HTTPException(
            status_code=400, detail="No ground truth available. Call POST /generate-data first."
        )
    return DetectorComparisonResponse(**anomaly_service.evaluate_detectors(contamination))


# ---------------------------------------------------------------------------
# Per-citizen explanation
# ---------------------------------------------------------------------------


@app.get(
    "/citizen/{master_citizen_id}/graph",
    response_model=ClusterGraphResponse,
    tags=["Citizen"],
)
def citizen_graph(master_citizen_id: str) -> ClusterGraphResponse:
    """The pairwise linkage evidence that produced this profile.

    Returns the records as nodes, Splink's scored pairs as edges with their
    field-level agreement, and which edges are load-bearing - in particular
    any *bridge*, an edge whose removal would split the cluster. A profile
    resting on one weak bridge is the classic over-merge signature and is
    exactly what an analyst reviewing it needs to see.
    """
    if not citizen_service.has_citizen_profiles():
        raise HTTPException(status_code=400, detail="No citizen profiles found. Call POST /run-linkage first.")
    try:
        graph = graph_service.get_cluster_graph(master_citizen_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if graph is None:
        raise HTTPException(status_code=404, detail=f"No profile found for '{master_citizen_id}'")
    return ClusterGraphResponse(**graph)


@app.get(
    "/citizen/{master_citizen_id}/recommendations",
    response_model=CoverageRecommendationResponse,
    tags=["Citizen"],
)
def citizen_recommendations(master_citizen_id: str) -> CoverageRecommendationResponse:
    """This citizen's service-coverage gaps, ranked by peer prevalence.

    For each agency with no record on file, reports what share of citizens
    with a comparable service footprint do hold one. The peer group is built
    only from which agencies hold a record - never from demographic or
    lifestyle data - and every figure ships with the peer-group size and
    definition behind it.

    This is a descriptive population statistic, not a prediction about the
    individual, and citizens are never ranked against each other. See
    app/recommendation_service.py for the full design boundary.
    """
    if not citizen_service.has_citizen_profiles():
        raise HTTPException(status_code=400, detail="No citizen profiles found. Call POST /run-linkage first.")
    result = recommendation_service.get_coverage_recommendations(master_citizen_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No profile found for '{master_citizen_id}'")
    return CoverageRecommendationResponse(**result)
