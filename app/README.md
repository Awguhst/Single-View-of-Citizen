# `app/` package reference

This package implements the CitizenLink (Single View of Citizen) proof-of-concept.
For setup/run instructions, see the [top-level README](../README.md).
This file documents what each module is responsible for and the key
design decisions inside it.

## Module map

| Module | Responsibility |
|---|---|
| `models.py` | Internal domain models (Pydantic) describing each DuckDB table's row shape: `Person` (ground truth), `Record` (a noisy per-agency record, distinguished by `record_type`), `ClusterAssignment`, `CitizenProfile`. Also defines `Agency` (11 government agencies), `RecordType` (13 record kinds), and `AGENCY_RECORD_TYPES`/`RECORD_TYPE_AGENCY` - the single source of truth mapping each agency to the record type(s) it issues. |
| `schemas.py` | Public API request/response contracts (Pydantic), kept separate from `models.py` so the OpenAPI schema can evolve independently of the storage layer. |
| `data_generator.py` | Seeded synthetic data generation: 10,000 ground-truth citizens, each independently sampled into 1-6 of the 10 government agencies (`AGENCY_INCLUSION_PROBABILITY`), producing ~75,000 noisy government records (tax, benefits, healthcare, education, immigration, driving licence, passport, employment, housing, veteran - Healthcare additionally yields repeatable hospital-visit/prescription records), all persisted into one unified `records` table. Every distribution parameter (population scale, agency-inclusion probabilities, per-record-type status/provider/detail pools, identity-noise thresholds) is a named module-level constant with a sensible default. The identity-noise thresholds are additionally bundled into a `NoiseProfile` dataclass whose defaults are those same constants, so `NoiseProfile()` reproduces the shipped dataset byte-for-byte; `NOISE_LEVELS` provides five named levels from `pristine` to `severe` for the benchmark sweep. `generate_all()` also accepts `n_people` and `persist=False`, which is what lets the benchmark generate and score datasets without touching the live database. `_ATTRIBUTE_GENERATORS` holds one function per `RecordType` producing that type's richer `attributes` map (e.g. tax band, blood type, service rank) - see "Per-record `attributes`" below. |
| `splink_service.py` | Splink configuration and the train -> predict -> cluster pipeline, run once over every row of the unified `records` table (all 13 record types alike). Persists `clusters` (source_record_id, master_citizen_id, match_probability) and `linkage_edges` (the pairwise evidence behind those clusters, with per-comparison agreement levels). |
| `citizen_service.py` | Aggregates clusters + every agency's records into `citizen_profiles`; exposes lookup, search, dashboard summary, engagement-tier, service-coverage, and data-quality queries. |
| `anomaly_service.py` | Fits a scikit-learn Isolation Forest over structural/linkage signals (record/agency counts, service-usage counts, and how much a cluster's records disagree about identity - never demographic/lifestyle fields) to flag statistically unusual profiles; persists results plus a per-profile explanation to `anomaly_results`. Also holds `evaluate_detectors()`, which ranks three detectors by how well each surfaces real linkage errors. |
| `evaluation_service.py` | The one module that reads the synthetic ground truth (`records.person_index`). Scores any predicted partition of `records` - pairwise precision/recall/F1, cluster-level failure modes, adjusted Rand index - against five deterministic baselines, and sweeps the clustering threshold by re-clustering the persisted pairwise edges. |
| `benchmark_service.py` | Regenerates the population at five noise levels, re-runs the full linkage pipeline on each, and scores every one against ground truth. Runs entirely in memory - it never touches the live database. Persists to `benchmark_results`. |
| `graph_service.py` | The pairwise linkage evidence behind one resolved profile: records as nodes, Splink's scored pairs as edges with field-level agreement, and any *bridge* whose removal would split the cluster. Uses `networkx`; no graph database involved. |
| `recommendation_service.py` | Ranks a citizen's service-coverage gaps by how common each service is among citizens with a comparable service footprint, with backoff when the peer group would be too small to be meaningful. Peer groups are built only from which agencies hold a record. |
| `exports.py` | Server-rendered exports: CSV for the directory listing, the engagement-tier drill-down, and the manual-review queue; PDF for a single citizen's Summary Report (via `reportlab`). Pure rendering on top of `citizen_service`'s existing queries - no new DB access. |
| `main.py` | FastAPI app wiring, lifespan auto-bootstrap, static-frontend mount, and the HTTP endpoints. |
| `static/` | The dashboard frontend (`index.html` + `app.js`) - plain HTML/JS + Tailwind/Chart.js CDN, served at `/`. Talks to the JSON API only; no server-side templating. |

## No authentication

There is none. Every endpoint is open and the dashboard opens straight onto
the Overview page with no sign-in.

This is a deliberate scope decision for a local, single-user demo over
entirely synthetic data - and it is the single biggest gap between this and
anything that could hold real records, which would need authentication,
per-user authorisation, and an audit trail of who viewed whose profile.
`tests/test_api.py::test_no_auth_routes_are_registered` pins the current
state so auth cannot reappear by accident.

## Data model (DuckDB tables)

```
citizens             -- ground truth only: person_index, name, dob, email, phone, address, city,
                         postcode, marital_status
records              -- ~75,000 noisy rows, one unified table for every kind of agency record:
                         source_record_id, person_index (hidden FK), agency, record_type (13 kinds -
                         see RecordType in models.py), name/contact fields (with noise/nulls, same
                         model for every record_type), plus generic payload columns reused across all
                         record types - agency_reference_id, record_date, expiry_date, status, amount,
                         provider_name, detail, marital_status (all nullable; meaning of status/amount/
                         detail varies by record_type, see data_generator._RECORD_TYPE_CONFIG;
                         marital_status is only ever populated on TAX_RECORD/BENEFITS_RECORD rows),
                         attributes MAP(VARCHAR, VARCHAR) - richer per-record-type detail (tax band,
                         blood type, service rank, ...), see data_generator._ATTRIBUTE_GENERATORS and
                         "Per-record attributes" below; deliberately excluded from Splink's
                         _load_linkage_pool projection, same as the other payload columns
clusters             -- source_record_id, master_citizen_id, match_probability   (Splink output, covering
                         every row of `records`, regardless of record_type)
linkage_edges        -- source_record_id_l, source_record_id_r, match_probability, gamma_<field> x7
                         (Splink's pairwise scores, kept rather than discarded after clustering -
                         they are the evidence *behind* each cluster. Enables the threshold sweep
                         (re-cluster without retraining), the per-profile Linkage Evidence panel,
                         and bridge detection. gamma is Splink's agreement level per comparison:
                         -1 not comparable, 0 disagreed, higher is closer)
benchmark_results    -- noise_level, people, records, splink_*/baseline_* metrics, f1_advantage,
                         population, seed   (latest noise-robustness sweep)
citizen_profiles     -- master_citizen_id, preferred_name, date_of_birth, marital_status,
                         current_address/city/postcode, linked_agencies, agency_count,
                         confidence_score, record_count, plus per-agency status/list fields
                         (tax_status, benefits_received, healthcare_registrations, hospital_visits,
                         current_prescriptions, education_history, driving_licence_status,
                         passport_status, employment_status, housing_assistance, veteran_status).
                         `age` and `lifestyle_summary` are NOT columns here - both are computed at
                         read time in citizen_service.py (see Citizen aggregation choices below).
                         CRIMINAL_RECORD is deliberately NOT one of these per-agency status fields -
                         it has no dedicated `citizen_profiles` column at all, and surfaces only via
                         the generic `records` list in the detail endpoint (see "Per-record
                         attributes" below), the same mechanism HOSPITAL_VISIT/PRESCRIPTION use.
anomaly_results       -- master_citizen_id, anomaly_score (0-100, higher = more anomalous),
                         is_anomaly, contamination  (Isolation Forest output, one row per
                         analyzed profile - see anomaly_service.py; rebuilt wholesale on every
                         POST /anomaly-detection/run, same CREATE OR REPLACE pattern as the
                         other derived tables)
```

A citizen appears in 1-6 of the 11 agencies (independently sampled per agency - see
`AGENCY_INCLUSION_PROBABILITY`), each contributing its own noisily-captured record. Healthcare
is the one agency with repeatable sub-records: a registered citizen additionally gets 1-3
hospital-visit records and 0-2 prescription records, each its own independent noisy identity
capture - reflecting how a citizen's healthcare history accumulates multiple encounters over
time, the way a real government's health service would.

`citizens` and the `person_index` column on `records` exist only because this is a
*synthetic* demo - they represent ground truth the data generator knows but a real government
never would. They are what let the Overview page show a true "before/after" comparison. A production
system would not have a `citizens` table; it would only ever see `records`.

## Why Splink, and why these specific settings

See the module docstring and inline comments in `splink_service.py` for the full rationale.
In short:

* **`dedupe_only`** link type because every noisy record - across all 13 record types, from
  all 11 agencies - lives in one pool to be deduplicated, not separate datasets to be linked.
  `splink_service._load_linkage_pool` is what reads every row of the unified `records` table
  before Splink ever sees it.
* **`DuckDBAPI` backend** keeps the whole pipeline in-process and dependency-free.
* **Comparisons** use Splink's purpose-built templates (`NameComparison`, `DateOfBirthComparison`,
  `EmailComparison`, `PostcodeComparison`) for the fields where naive exact-match would fail on the
  injected noise, and Levenshtein-distance comparisons for free-text phone/address fields.
* **Blocking rules** are a union of six different "this pair is plausibly the same person" signals
  (shared email, shared phone, name+surname, DOB+postcode, surname+postcode, initial+surname+DOB) -
  this keeps the candidate-pair count tractable while still catching every noise pattern the
  generator injects.
* **Training** uses a small set of high-precision deterministic rules to seed the prior, then
  `estimate_u_using_random_sampling` plus three EM passes against different blocking rules
  (Splink's standard identifiability pattern - each comparison must be trained on data it wasn't
  blocked on).
* **Cluster threshold** (`CLUSTER_MATCH_THRESHOLD = 0.75`) was chosen by sweeping thresholds from
  0.3 to 0.95 and measuring pairwise precision/recall against the generator's ground truth (only
  possible in this synthetic demo). 0.5-0.9 all produced near-identical, near-optimal results
  (precision and recall both close to 100%); 0.75 sits in the middle of that stable plateau.

## Per-record `match_probability`

Splink scores pairwise edges, not individual records, so a single confidence value per
`source_record_id` is derived in `splink_service._per_record_match_probability`: the strongest
edge connecting that record to another member of its own cluster. A record in a singleton
cluster (no duplicate found) defaults to `1.0` - there's no second record to be uncertain against.

## Citizen aggregation choices

* `preferred_name` and `date_of_birth` are resolved via a "most agreed, most complete" pick
  across every linked record - the same logic regardless of which agency a record came from,
  since every record type carries the same noisy identity fields.
* `current_address` is resolved the same way, with the most recent `record_date` used as the
  final tie-break - unlike a name or DOB, an address is genuinely expected to change over time,
  not just be noisily re-captured.
* `tax_status` / `driving_licence_status` / `passport_status` / `employment_status` /
  `housing_assistance` / `veteran_status` each take the status of that citizen's most recent
  record of the matching type (`arg_max(status, record_date) FILTER (WHERE record_type = ...)`),
  since these are point-in-time states that can genuinely change.
* `benefits_received` / `current_prescriptions` / `education_history` are the distinct set of
  `detail` values seen across that citizen's records of the matching type - a citizen can
  genuinely hold several distinct benefits, prescriptions, or qualifications at once, unlike
  the single evolving statuses above.
* `healthcare_registrations` / `hospital_visits` are simple counts of that record type.
* `marital_status` resolves the same way as the point-in-time status fields above, but is only
  ever sourced from `TAX_RECORD`/`BENEFITS_RECORD` rows (`arg_max(marital_status, record_date)
  FILTER (WHERE record_type IN ('TAX_RECORD', 'BENEFITS_RECORD'))`) - the two record types a real
  agency would realistically capture it on (see `data_generator.py`'s `captures_marital_status`
  flag). `age` is not stored at all - it's computed at read time from the resolved
  `date_of_birth` against the real current date, in `citizen_service._compute_age`.
* There is no monetary aggregation anywhere - `amount` (tax paid, monthly benefit, housing
  benefit) is real per-record detail shown on the profile dossier, never summed into a
  single figure. Government records are no longer a clean, ground-truth attachment: each one
  carries the same kind of noisy, independently-captured identity fields every other record
  does, and flows through the *same* Splink dedupe pool (see `splink_service.py`). A record
  attaches to a `master_citizen_id` exactly the way every other record does - by being a
  member of that cluster in the `clusters` table - not via any ground-truth shortcut (see
  `citizen_service.py`'s `name_counts` CTE, which pulls names from every row of `records`
  regardless of `record_type`, so a cluster never ends up nameless).

## Per-record `attributes`

Each of the 13 record types carries 4-6 additional administrative fields beyond the shared
`status`/`amount`/`provider_name`/`detail` columns - e.g. a Tax Record's `tax_year`/
`income_declared`/`tax_band`/`filing_method`/`allowances_claimed`, a Veteran Record's
`rank`/`service_start_year`/`service_end_year`/`deployment_regions`/`disability_rating`, or a
Criminal Record's `sentence_type`/`sentence_length`/`fine_amount`/`rehabilitation_status`/
`disclosure_level`. These live in one `MAP(VARCHAR, VARCHAR)` column, `attributes`, populated
per record by `data_generator._ATTRIBUTE_GENERATORS` (one function per `RecordType`). This is
deliberately **not** aggregated into `citizen_profiles` at all - `get_citizen_profile_detail`'s
per-record `SELECT` returns each record's own `attributes` map unaggregated, and the frontend
groups `p.records` by `record_type` client-side to build the case-file card grid (one card per
record type, showing that type's most recent record's full `attributes`) and to show every
individual record's full detail inline in the Records Timeline. No new `citizen_profiles`
column, no new list-view field - `attributes` stays exactly as detail-only as
`records`/`field_agreement`/`lifestyle_summary` already are.

Every `attributes` value is a plain administrative fact, same policy as everywhere else in this
file - see "Design boundary" below, which this extends explicitly.

**`CRIMINAL_RECORD` (agency: Criminal Justice)** is worth its own callout, since it's the most
sensitive record type in the dataset (criminal-offence data is a GDPR special category, similar
in sensitivity to health data). It's generated with the same "plain administrative fact, never
scored" discipline as everything else - `sentence_type`/`sentence_length`/`fine_amount` are only
populated when applicable ("N/A" otherwise, e.g. an Acquitted record has no sentence),
`rehabilitation_status` reflects the real Rehabilitation of Offenders Act "spent conviction"
concept, and `disclosure_level` mirrors the real UK DBS check tiers - none of this is a score or
a judgment. Two extra safeguards specific to this type, given its sensitivity: (1)
`get_showcase_example()` explicitly excludes any cluster containing a `CRIMINAL_RECORD` row, so
it's never the one auto-selected for the dashboard's "before/after" showcase every user sees by
default; (2) Criminal Justice is excluded from the `_SERVICE_GAP_AGENCY_PRIORITY` "Service Coverage
Gaps" gap-check list, since suggesting someone acquire a criminal record makes no sense. Note what is
**not** restricted: `linked_agencies`/`agency_count`/`engagement_tier` stay fully generic, so
"Criminal Justice" appears in a citizen's agency list in the Directory/search/CSV exactly like
any other agency (the same treatment Immigration/Veterans Affairs already get) - only the
record's *content* (status, offense category, sentence detail) is scoped to the full detail
dossier, not a bespoke two-tier access model.

## Profile dossier endpoint (`GET /citizen/{id}/detail`)

Backs the "Profile" page (reached by clicking through from the Citizens list,
the Review Queue, or the Overview showcase).
Everything on it is derived, not hardcoded:

* `engagement_tier` - a simple threshold-based label on `agency_count` (Minimal / Limited /
  Moderate / High / Full Engagement) - how many distinct agencies hold a record for this citizen.
* `records` - every row of the unified `records` table clustered into this profile, across
  every agency, each with its own per-record `match_probability`, sorted by `record_date`
  ascending. This is the real "data lineage" and what powers the profile's **records timeline**:
  which agency, which `record_type`, what that agency's system recorded
  (`agency_reference_id`/`status`/`amount`/`provider_name`/`detail` - all nullable, meaning
  varies by `record_type`).
* `field_agreement` - for each identity field (email, phone, address, postcode,
  date_of_birth, last_name), whether *every* linked record - across every agency - agreed on a
  single non-null value. Where they don't, the distinct values are surfaced (e.g.
  "TF2R 2LQ vs. TF2R2LQ") so an analyst can see exactly what was noisy versus what was a
  stronger signal - this is what the page's "Match Explanation" panel renders, instead of a
  fabricated narrative.
* `lifestyle_summary` - a list of `{tag, basis}` items computed by `citizen_service._lifestyle_summary`,
  each a plain-language restatement of a count or value already elsewhere on the profile (e.g.
  hospital visit count, prescription list length, years since the earliest employment record),
  with the exact evidence cited in `basis`. **This is not a score, ranking, or predictive
  judgment of any kind** - see the design-boundary note below.

### Design boundary: no risk or likelihood scoring

`engagement_tier` and `lifestyle_summary` are the only two derived/summary fields on the
Unified Citizen Profile, and both are simple threshold checks over data already present -
never a computed risk, likelihood, or predictive score. This is intentional: scoring a
citizen's likelihood of any behavior (including criminal activity) from demographic/lifestyle
proxies is a well-documented source of discriminatory harm, and it holds even for synthetic
data, since the scoring pattern itself - not the underlying data - is the reusable artifact.
If you're extending this profile further, keep new derived fields in the same shape: a
plain-language restatement of literal data, with its exact evidence cited, never a score.

The same policy covers every record's `attributes` map (see "Per-record `attributes`" above):
every field (e.g. `tax_band`, `blood_type`, `rank`) is a plain administrative fact the issuing
agency would realistically hold on file, never fed into `lifestyle_summary` or any other
derived field. `disability_rating` on `VETERAN_RECORD` is worth naming explicitly, since it
could otherwise read as evaluative: it's a real classification a Veterans Affairs-equivalent
agency genuinely tracks for benefit-entitlement purposes - a recorded fact about an
already-assessed condition, the same kind of thing as `driving_licence.points` or `tax_band`,
not a computed judgment about the person.

`CRIMINAL_RECORD` (agency: Criminal Justice) is the clearest case this boundary was written
for: it's a factual record of a documented, already-adjudicated legal outcome (conviction,
caution, sentence) - never a prediction of future behavior, never a "likelihood of offending,"
and never derived from any other field on the profile (marital status, lifestyle, etc.). It
gets two extra safeguards beyond the general policy, given its sensitivity:
`citizen_service.get_showcase_example()` excludes any cluster containing one from the
dashboard's auto-picked showcase, and the `_SERVICE_GAP_AGENCY_PRIORITY` "Service Coverage Gaps"
gap-check list excludes Criminal Justice entirely (same treatment as Immigration/Veterans Affairs,
for the more basic reason that "recommending" someone acquire any of these doesn't make sense).

**`anomaly_service.py`'s Isolation Forest** is bound by the same policy, and it's worth stating
explicitly since a statistical "anomaly score" can look like an exemption from it: the model is
fit on structural/linkage signals only (record/agency counts, match confidence, service-usage
counts) and deliberately never sees `age`, `marital_status`, or any other demographic/lifestyle
field. Feeding those fields in would reproduce the same demographic-proxy scoring pattern this
section exists to prevent, just relabeled "anomaly" instead of "risk." If you extend this
feature, keep any new input structural (a count, a confidence value, something about linkage
shape) - never demographic or lifestyle-derived.

`get_service_coverage_summary` (see "Service coverage endpoint" above)
is bound by the same policy: it reports counts of a fixed, hardcoded
rule ("first agency in a fixed priority order this citizen isn't
linked to yet"), never a score, confidence percentage, or ranking of
citizens by likelihood of any future action.

## Data-quality endpoint (`GET /quality`)

Backs the dashboard's "Data Quality" page. All three pieces are computed directly from the
`clusters` table (no extra ground truth needed, so this works the same way in production):

* `match_probability_histogram` - bucketed counts of the per-record confidence score described
  above, so an analyst can see at a glance how much of the population is high- vs low-confidence.
* `cluster_size_distribution` - how many resolved citizens had 1, 2, 3, or 4+ linked records,
  counting every record in the cluster across every agency.
* `review_queue` - the lowest-confidence *multi-record* clusters (singletons are excluded - there's
  nothing to review), worst first. This is the closest thing this POC has to a human-in-the-loop
  reconciliation queue: in production, a government wouldn't blindly trust every Splink merge, it
  would route the borderline ones to a caseworker.

## Anomaly detection endpoints (`POST /anomaly-detection/run`, `GET /anomaly-detection`)

Backs the dashboard's "Anomaly Detection" page. `anomaly_service.run_anomaly_detection`
fits an `sklearn.ensemble.IsolationForest(contamination=..., random_state=SEED)` over every
resolved profile's `record_count`, `agency_count`, `records_per_agency`, `hospital_visits`, the
sizes of `benefits_received`/`current_prescriptions`/`education_history`, and three identity-strain
counts computed from the cluster's own records (`distinct_first_names`, `distinct_dobs`,
`distinct_postcodes`) - structural/linkage signals only.

`confidence_score` was **removed** from this feature set: it was measured at standard deviation
8e-6 across the population, i.e. effectively the constant 1.0. A constant column cannot be split on
by any tree, so it contributed nothing to the model while appearing in the UI as though it were
evidence. It is still shown on the results table as context.

The identity-strain counts are the ones that make this page useful, and they encode what it is
really detecting: on this dataset an "anomalous profile" is overwhelmingly a *linkage artefact*
rather than an unusual citizen. A cluster holding four different dates of birth is not describing a
citizen with four birthdays. `evaluate_detectors()` confirms this against ground truth rather than
asserting it.

Each persisted result carries a `top_factors` explanation - the features most responsible for that
profile's score. Because an Isolation Forest score is not additively decomposable, there is no exact
per-feature split of it; what *is* exact is a counterfactual, so `_attribute_scores()` re-scores
every profile with one feature replaced by the population median and measures how far the score
falls. That is an ablation against the actual fitted model rather than a proxy statistic, and it
costs one extra `decision_function()` call per feature.

`decision_function()` is
inverted and min-max normalized to a 0-100 `anomaly_score` (higher = more anomalous); `predict()`
becomes the `is_anomaly` flag at the requested `contamination`. Results are persisted to
`anomaly_results` (`GET /anomaly-detection` then reads the latest run
without recomputing) via the same `conn.register` -> `CREATE OR REPLACE TABLE ... AS SELECT` ->
`conn.unregister` pattern `data_generator._persist()` uses. `random_state=SEED` matches this
project's "every random choice is seeded" reproducibility policy - see Reproducibility in the
top-level README.

See the "Design boundary" section below - the feature set here is deliberately narrower than
`citizen_profiles` itself, and that's intentional, not an oversight.

## Service coverage endpoint (`GET /service-coverage`)

Backs the "Service Coverage Gaps" page's summary cards and chart.
`citizen_service.get_service_coverage_summary` computes a true
population-wide aggregate in one query over `citizen_profiles`, instead
of every profile going to the client to be totalled up:
`missing_record_counts` is a plain per-agency count of citizens with no
linked record there, and `next_gap_counts`/`fully_covered_count` apply a
fixed-priority "first gap" rule aggregated via
`_next_service_gap_case_sql` (same CASE-expression-from-a-Python-list
technique as `_engagement_tier_case_sql`).

`_SERVICE_GAP_AGENCY_PRIORITY` is now the **single** definition of the
candidate agency list. The frontend reads the agency names back off this
endpoint's response rather than restating them - the list used to be
duplicated in Python and JavaScript with a comment asking editors to keep
the two in sync by hand.

The page's table is powered by the existing `GET /search` and lists each
citizen's actual gaps as plain facts, with no ordering claim. Ranking an
individual citizen's gaps moved to `recommendation_service.py` (see
above), where the peer group and sample size can be reported alongside
every figure. The population summary and the row-level table stay
deliberately decoupled (see `app.js`'s `loadServiceCoverageSummary` vs
`loadServiceCoverageTable`) so the cards always reflect the true
population, not whatever page of results happens to be on screen.

See the "Design boundary" section below - the population counts stay a
plain aggregation over already-stored data, never a score.

## Evaluation endpoints (`GET /evaluation/*`, `/benchmark`, `/anomaly-detection/evaluation`)

`evaluation_service.py` is the only module in the application that reads
`records.person_index`. Everything else - linkage, aggregation, profiles,
recommendations - resolves identity from the noisy comparison fields alone,
exactly as a production system would. That boundary is deliberate and load-bearing:
the product must never depend on knowing the answer.

But the data *is* synthetic, so the answer is available afterwards, and refusing to
use it would mean shipping ML with no idea whether it works.

### Why three families of metric

Entity resolution has no single natural accuracy number, because the output is a
*partition* of records rather than a label per record:

* **Pairwise precision/recall/F1** - the standard ER metric, and the one that
  degrades gracefully: a cluster one record short is penalised proportionally
  rather than counted as a total miss.
* **Cluster-level counts** - `exactly_resolved`, `over_split_citizens`,
  `over_merged_clusters`. Pairwise F1 hides the *shape* of the errors, and the two
  failure modes cost very different things. An over-split fragments one citizen
  across several profiles (duplicate outreach). An over-merge puts one citizen's
  records on another citizen's profile - far more serious, and the reason it gets
  its own column everywhere it appears.
* **Adjusted Rand index** - chance-corrected agreement between the two partitions,
  the standard clustering-quality measure.

All of these are computed from four scalars (the three "n choose 2" contingency
sums plus the record count), which is why `_contingency_sums` stays a pure SQL
aggregation and never materialises the pairs. The `all_one_cluster` baseline alone
would otherwise enumerate roughly 2.8 billion of them.

### Why baselines at all

An F1 in isolation says nothing - 0.99 is either excellent or embarrassing depending
on how hard the problem is. Each baseline in `_BASELINE_RULES` is a deterministic
rule someone would genuinely reach for before installing a probabilistic linkage
engine, and the two degenerate rows (link nothing / link everything) bracket the
scale so the number becomes interpretable.

A NULL blocking key leaves a record a singleton rather than lumping every NULL
together - otherwise a baseline would score its own missing data as a match, which
is a bug dressed up as a result.

### Threshold sweep

`CLUSTER_MATCH_THRESHOLD` is a single hand-picked number whose docstring says it was
chosen "empirically". `sweep_cluster_threshold()` turns that judgement call into a
measurement. Because `linkage_edges` persists Splink's pairwise scores, each point on
the curve is a `scipy.sparse.csgraph.connected_components` traversal rather than a
model retrain - the same operation Splink's own clustering step performs - so the
whole sweep runs in seconds.

### Noise benchmark (`benchmark_service.py`)

The threshold sweep varies the *model*; this varies the *problem*. It regenerates the
population at each `NOISE_LEVELS` profile, re-runs the real pipeline (via
`splink_service._build_and_train_linker`, so it measures the shipped configuration
rather than a reimplementation), and scores each level against ground truth alongside
the strongest deterministic baseline.

Two design constraints worth stating:

* **Nothing touches the live database.** `generate_all(persist=False)` returns frames,
  the linker runs over those frames, and metrics are computed in an in-memory DuckDB
  connection. Persisting would drop the `clusters`/`citizen_profiles` tables the
  running app is serving from, so the benchmark is built so that it cannot.
* **Small populations.** Each level trains a complete Splink model, so the sweep is
  inherently several pipeline runs. A few thousand people per level keeps it to about
  a minute while still giving tens of thousands of ground-truth pairs per point.

The `pristine` level doubles as a sanity check: with no noise at all, any competent
linker should score 1.0, so anything less indicates a bug in the pipeline rather than
a hard problem.

### Detector comparison

Unsupervised anomaly detection normally cannot be scored. Here it can, because a
cluster either does or does not correspond one-to-one with a real citizen - a concrete,
checkable ranking task. That is the *only* reason `anomaly_service` implements more
than one algorithm; each fails differently on this task, and the comparison is the
point:

* **Isolation Forest** - the incumbent. Tree-based, isolates unusual feature
  combinations, needs no distance metric or scaling.
* **Local Outlier Factor** - density-based: unusual relative to its neighbours rather
  than globally, which suits a population made of distinct engagement segments. It is
  given a widened neighbourhood (`LOF_NEIGHBORS = 50`) before being judged, because
  these features are low-cardinality integers with many exact duplicates and sklearn's
  default of 20 neighbours can be entirely duplicates.
* **Max robust z-score** - the interpretable baseline that must be beaten before either
  model is worth its complexity. Median/MAD based, so a handful of extreme profiles
  cannot inflate the spread.

Results are reported as average precision, precision@k / recall@k, and lift over the
base rate - the only honest reference point when roughly 1% of the population is a
positive.

## Linkage graph (`graph_service.py`, `GET /citizen/{id}/graph`)

A `master_citizen_id` is the output of a graph computation: records are nodes, scored
pairs are edges, and a cluster is a connected component at the chosen threshold. The
rest of the app only ever shows the final answer, which is exactly the part an analyst
reviewing a questionable profile cannot check.

The load-bearing detail is the threshold rule. Structural analysis runs on edges at or
above `CLUSTER_MATCH_THRESHOLD` - the graph that actually *created* the cluster - not
on every stored pair. `linkage_edges` reaches down to `PREDICT_MIN_THRESHOLD`, and
those sub-threshold pairs were explicitly rejected as evidence; including them makes
every cluster look densely cross-confirmed and hides the single weak link a bridge
analysis exists to find. There is a regression test for exactly this.

A *bridge* - an edge whose removal disconnects the graph - is a single point of
failure, and the classic over-merge signature is two separate groups of records joined
by one coincidence. Reporting the weakest bridge turns "this profile looks odd" into
"this profile exists because of this one link, scored 0.78, joining these two records
on these fields" - something an analyst can adjudicate.

No graph database is involved or needed: the edges are already in DuckDB, one cluster
is at most a few dozen nodes, and `networkx` was already available in the stack.

## Coverage recommendations (`recommendation_service.py`, `GET /citizen/{id}/recommendations`)

The original rule was one line - walk a hardcoded priority list, return the first
agency with no record. Honest and fast, but the ordering carried no information
("Healthcare before Revenue & Tax" was a guess baked into a constant) and the list was
duplicated in Python and JavaScript with a comment asking editors to keep the two in
sync by hand.

Gaps are now ranked by a measured quantity: among citizens whose service footprint
already looks like this one's, what share also hold a record with the missing agency?
That ordering varies per citizen in a way a fixed list cannot.

**Backoff.** The natural peer group - citizens holding records with every agency this
one does - shrinks fast as a footprint grows, and a prevalence computed from nine
peers is noise dressed up as evidence. Below `MIN_PEER_GROUP`, the least common agency
is dropped from the conditioning set and the group rebuilt, until it is either large
enough or has widened to the whole population. Every response states which conditioning
set was actually used, so a weakly conditioned figure is visibly weakly conditioned.

**The design boundary is unchanged.** See the section below - the peer group is built
only from which agencies hold a record, the output is a descriptive population
statistic rather than a prediction about the individual, and citizens are never ranked
against each other.

## Exports (`exports.py`)

Server-rendered downloads:

* `GET /export/directory.csv` - every resolved profile matching `?q=` (or all profiles, if
  blank) as CSV. Calls `citizen_service.search_person` with a much larger limit than its
  on-screen default of 50, so the export never silently truncates the dataset it claims to cover.
* `GET /export/review-queue.csv` - the manual-review backlog as CSV, defaulting to the worst
  500 clusters rather than the dashboard widget's top 10 (`?limit=` to override).
* `GET /export/engagement-members.csv?tier=` - every resolved profile in one engagement tier as CSV.
* `GET /citizen/{id}/export/pdf` - one citizen's full dossier (same data as
  `/citizen/{id}/detail`) rendered as a Citizen Summary Report PDF via `reportlab`, alongside
  the dashboard's existing client-side-only JSON export.

## Dashboard showcase endpoint (`GET /dashboard/showcase`)

Backs the dashboard's "Entity Resolution In Action" before/after panel. `citizen_service.get_showcase_example`
picks one cluster straight out of the live `clusters`/`records` tables - no ground truth involved,
so the same query would work against real production data:

* Selection criteria: 3-4 linked records with at least 2 different first-name spellings/variants,
  so the "before" side visibly looks like unrelated people. Among qualifying clusters, one
  spanning more distinct agencies is preferred (a richer cross-agency story), then the largest.
* Returns the exact same shape as `/citizen/{id}/detail`, so the frontend renders the "before"
  raw records and the "after" Unified Citizen Profile from one response.
