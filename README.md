# CitizenLink - Single View of Citizen

A proof-of-concept for a government that wants to consolidate citizen records
from multiple independent agencies, resolve duplicate citizen identities (no
agency shares a common citizen ID, and every agency's data has the usual
real-world quality issues), and produce a single **Unified Citizen Profile**
per individual to support joined-up public service delivery.

Governments typically run dozens of disconnected systems - a tax authority,
social security, healthcare, education, immigration, driver licensing, a
passport office, employment services, housing, veterans affairs, criminal
justice - each capturing the same citizen differently: different identifiers, missing
fields, nicknames, spelling variants, outdated addresses, inconsistent phone
formatting, multiple emails. Entity resolution is what lets a government
automatically determine "these records all belong to the same real citizen"
without every agency first agreeing on a shared ID scheme - exactly the
problem this proof-of-concept demonstrates.

Entity resolution is performed with **[Splink](https://moj-analytical-services.github.io/splink/)**
(probabilistic record linkage, DuckDB backend). Everything else - the API,
the dashboard frontend, the synthetic data, the storage - is a self-contained
FastAPI + DuckDB app with no external services.

## Architecture

```
                 ┌────────────────────┐
  Faker-seeded   │  data_generator.py │   10,000 citizens -> ~75,000 noisy
  synthetic data │                    │   multi-agency government records
                 └─────────┬──────────┘   across 11 agencies
                           │  DuckDB (data/svoc.duckdb)
                           v
                 ┌────────────────────┐
                 │  splink_service.py │   train -> predict -> cluster
                 │  (Splink, DuckDB)  │   -> master_citizen_id clusters
                 └─────────┬──────────┘
                           v
                 ┌─────────────────────┐
                 │ citizen_service.py  │   aggregate every agency's records
                 │                     │   per cluster -> Unified Citizen
                 └─────────┬───────────┘   Profile
                           v
                 ┌────────────────────┐
                 │     main.py        │   FastAPI: /generate-data,
                 │   (FastAPI app)    │   /run-linkage, /citizen/{id},
                 └─────────┬──────────┘   /search, /dashboard, /quality
                           v
                 ┌────────────────────┐
                 │  app/static/*      │   Dashboard frontend (HTML/JS,
                 │  (served at /)     │   Chart.js) consuming the JSON API
                 └────────────────────┘

  Evaluation lane (developer/analyst-facing, reads ground truth):

                 ┌──────────────────────┐
                 │ evaluation_service.py│  Splink vs deterministic baselines,
                 │                      │  threshold sweep, cluster metrics
                 └──────────────────────┘
                 ┌──────────────────────┐
                 │ benchmark_service.py │  re-run the pipeline at 5 noise
                 │                      │  levels, score each vs ground truth
                 └──────────────────────┘
                 ┌──────────────────────┐
                 │   graph_service.py   │  the pairwise linkage evidence
                 │                      │  behind one resolved profile
                 └──────────────────────┘
```

See [`app/README.md`](app/README.md) for the per-module breakdown and the
detailed rationale behind every Splink configuration choice.

## Evaluation: is any of this actually working?

The data is synthetic, so the true identity behind every record is known
(`records.person_index`). The linkage pipeline never sees it - it resolves
identity from the noisy comparison fields alone, exactly as a production
system would - but the **Model Evaluation** page uses it afterwards to check
the result. These are the only endpoints in the application that read ground
truth, and no citizen-facing page depends on them.

### Is the model beating a hand-written rule?

Pairwise metrics over the shipped 74,955-record dataset:

| Method | Precision | Recall | F1 | Resolved exactly | Over-merged |
|---|---|---|---|---|---|
| No linkage (every record its own citizen) | 1.0000 | 0.0000 | 0.0000 | 18 | 0 |
| Exact email match | 0.8972 | 0.2132 | 0.3445 | 97 | 1,277 |
| Exact first name + surname + DOB | 1.0000 | 0.4464 | 0.6173 | 352 | 0 |
| Surname + DOB + normalised postcode | 1.0000 | 0.8495 | 0.9187 | 5,549 | 0 |
| **Splink (current pipeline)** | **0.9992** | **0.9991** | **0.9991** | **9,947** | **3** |
| Link everything (one single citizen) | 0.0001 | 1.0000 | 0.0002 | 0 | 1 |

The best hand-written rule gets to F1 0.919 and resolves 5,549 of 10,000
citizens exactly. Splink resolves 9,947. The two degenerate rows bracket the
scale, so the F1 column is interpretable rather than just large.

Two failure modes are counted separately because they cost very different
things: an **over-split** (47 citizens) fragments one person across several
profiles, causing duplicate outreach; an **over-merge** (3 clusters) puts one
citizen's records on another citizen's profile, which is far more serious.

### Does the threshold matter?

`splink_service.CLUSTER_MATCH_THRESHOLD` is a single hand-picked number whose
docstring says it was chosen "empirically". Sweeping it says two things:

* **0.90 scores better than the configured 0.75** - F1 0.9994 vs 0.9991, with
  over-merged clusters falling from 3 to 1 at no cost to recall. The shipped
  value is left as it is rather than quietly tuned to the evaluation set, but
  the page now shows the trade-off instead of hiding it.
* **Recall is completely flat across the sweep.** Every missed link is a pair
  the *blocking rules* never proposed for scoring, so no threshold can recover
  it. Improving recall here means changing the blocking rules - a different
  fix from the one someone would reach for by default.

Each point re-clusters the persisted pairwise edges, so the whole sweep is
graph traversal rather than model retraining.

### Does it hold up when the data gets worse?

`POST /benchmark/run` regenerates the population at five noise levels and
re-runs the entire pipeline against each. Representative run (1,500 citizens
per level, ~4,500 records each):

| Noise level | Splink F1 | Best deterministic rule | Advantage |
|---|---|---|---|
| Pristine (no noise at all) | 1.0000 | 1.0000 | +0.0000 |
| Light | 1.0000 | 0.9765 | +0.0235 |
| Default (the shipped dataset) | 0.9994 | 0.9180 | +0.0814 |
| Heavy | 0.9968 | 0.8431 | +0.1537 |
| Severe | 0.9895 | 0.7208 | +0.2686 |

This is the result that justifies the complexity. On clean data a probabilistic
model and a fixed rule are indistinguishable; the gap widens monotonically as
the data degrades, which is exactly what a trained model should do and what a
brittle rule cannot. The pristine row scoring exactly 1.0000 also serves as a
sanity check - anything less would indicate a bug in the pipeline rather than
a hard problem.

### Which anomaly detector is worth having?

Unsupervised anomaly detection normally cannot be scored. Here it can, because
a cluster either does or does not correspond one-to-one with a real citizen -
which turns it into a checkable ranking task. Ranking the 97 clusters (0.97%
of the population) whose linkage actually went wrong:

| Detector | Avg precision | Recall@5% | Lift vs random |
|---|---|---|---|
| **Isolation Forest** | **0.2387** | **0.6082** | **24.7x** |
| Local Outlier Factor | 0.0122 | 0.0515 | 1.26x |
| Max robust z-score (baseline) | 0.0110 | 0.0515 | 1.14x |

This is the only reason more than one algorithm exists in the codebase. The
Isolation Forest surfaces 61% of all real linkage errors within the top 5% of
profiles. LOF is a poor fit here and the reason is legible: most features are
low-cardinality integers, so thousands of profiles share an identical feature
vector and its local density estimate degenerates (it was given a widened
neighbourhood before being judged). The z-score baseline is barely better than
chance, which is what makes the forest's 24.7x worth the complexity.

## No authentication

There is none, by design. This is a local, single-user demo over entirely
synthetic data, so every endpoint is open and the app opens straight onto the
dashboard - no login, no accounts, no roles.

That is a deliberate scope decision, not an oversight, and it is the main
thing that would have to change before this shape of app went anywhere real:
a genuine deployment holding citizen records would need authentication,
per-user authorisation, and an audit trail of who looked at whose profile.
None of that is simulated here.

## Quickstart

This project's `env/` folder is a conda environment (Python 3.14) with
`fastapi`, `uvicorn`, `duckdb`, `splink`, `faker`, `pandas`, `scikit-learn`,
`scipy`, `networkx` and `reportlab` already installed. From the project root:

```bash
# Windows
.\env\python.exe -m uvicorn app.main:app --reload --port 8000

# macOS/Linux, or any other Python 3.11+ environment:
pip install -r app/requirements.txt
uvicorn app.main:app --reload --port 8000
```

On Windows, `run.ps1` does the same thing in one step: it activates the
`env/` conda environment and starts the server from it, so you get an
activated conda shell and a running app together.

```powershell
.\run.ps1                # activates env/ and starts uvicorn on port 8000
.\run.ps1 -Port 8001      # override the port
```

Then open **http://localhost:8000/** for the dashboard, or
**http://localhost:8000/docs** for the interactive Swagger UI. No sign-in is
required for either.

On first startup the app automatically:
1. Generates the synthetic dataset (seeded, reproducible).
2. Runs the Splink linkage pipeline.
3. Builds the Unified Citizen Profiles.

This takes roughly 20-40 seconds; watch the server logs for progress. The
dataset persists to `data/svoc.duckdb`, so subsequent restarts skip
regeneration - delete that file (or use the "Generate Data" / "Run Linkage"
buttons in the dashboard) to start fresh.

## Deploying to Railway

The repo ships a `Dockerfile` and `railway.json`, so a Railway service
pointed at this repo builds and deploys with no further configuration. No
environment variables are required.

```bash
railway init      # or point a new service at the GitHub repo in the dashboard
railway up
```

### The one decision worth knowing about

**The synthetic dataset is built during `docker build`, not on first boot.**

On startup the app would otherwise generate 10,000 citizens, train a Splink
model, build the profiles and fit the review-queue model - about 40 seconds,
measured. A platform health check gives up long before that, and because
container filesystems are ephemeral the cost would be paid again on every
restart and every redeploy.

`python -m app.bootstrap` therefore runs as a build step (see the
`Dockerfile`), and the image ships with a ready `svoc.duckdb`. The container
then boots in about a second with all four pages already populated. Because
generation is seeded, every replica and every rebuild serves identical data.

The trade-off is a slower build and roughly 20MB of image. Both are the
right way round for something deployed far less often than it is started.

### Other deployment notes

* **`.dockerignore` excludes `env/`.** That is the local conda environment,
  hundreds of megabytes, and the image installs its own dependencies from
  `app/requirements.txt`. It is the single most important line in that file.
  It also excludes `data/`, so a stale local database cannot override the one
  built into the image.
* **`$PORT` is honoured, and the start command lives only in the
  `Dockerfile`.** Railway injects `$PORT`; the `CMD` expands it via `sh -c`
  and falls back to 8000, so a plain `docker run -p 8000:8000 citizenlink`
  works too.

  Do **not** add a `startCommand` to `railway.json`. It overrides the image's
  `CMD` and is delivered to the container *without shell expansion*, so
  `--port $PORT` reaches uvicorn as the four literal characters `$PORT` and
  the container crash-loops with
  `Error: Invalid value for '--port': '$PORT' is not a valid integer`. This
  repo hit exactly that; the fix was deleting `startCommand` so there is one
  shell-expanded definition in one place. Verified by running the image with
  `-e PORT=9123` on a non-default port.
* **`CITIZENLINK_DB_PATH`** controls where the DuckDB file lives (the image
  sets `/data/svoc.duckdb`). Point it at a mounted volume if you ever want
  generated data to survive a redeploy - though for this demo, rebuilding
  from seed is both faster and more reproducible.
* **Test dependencies are not installed in the image.** `pytest` and `httpx`
  live in `app/requirements-dev.txt`:

  ```bash
  pip install -r app/requirements.txt -r app/requirements-dev.txt
  ```

* **The Evaluation page's two heavy endpoints are memoised in process**
  (`app/cache.py`). The threshold sweep and detector comparison cost ~3.3s
  and ~2.4s of CPU, are pure functions of tables that only change when the
  pipeline re-runs, and are keyed on those tables' row counts. First view
  after a restart pays full price (the page shows loading states); every
  later view is ~40ms. Correctness never depends on the cache - a restart
  just recomputes.
* **Memory.** The image loads DuckDB, pandas, scikit-learn and Splink;
  expect roughly 400-600MB resident once the Evaluation page has been
  opened. Railway's smallest instances can be tight - if the container is
  OOM-killed, that is the first thing to raise.
* **There is no authentication.** Every endpoint is public, including the
  compute-heavy `POST /generate-data`, `POST /run-linkage` and
  `POST /benchmark/run`. On a private demo URL that is fine; on a public one,
  anyone can trigger a minute of CPU per request. If this is going somewhere
  publicly reachable, put auth or at least rate limiting in front of those
  three routes first - see [No authentication](#no-authentication).

## Dashboard frontend

`app/static/index.html` + `app/static/app.js` is a small dependency-free
dashboard (Tailwind CDN + Chart.js CDN, no build step) served directly by
FastAPI at `/`. It opens straight onto the Overview - there is no login.
Every view is backed by live calls to the JSON API below; nothing is
hardcoded.

There are four pages, and each one answers a single question:

| Page | Answers | Backed by |
|---|---|---|
| **Overview** | What does the population look like? | `/dashboard`, `/dashboard/showcase`, `/service-coverage` |
| **Citizens** | Find and inspect a person | `/search`, `/engagement`, then `/citizen/{id}/*` |
| **Review Queue** | What needs a human? | `/anomaly-detection` |
| **Evaluation** | Is the system working? | `/evaluation/*`, `/benchmark`, `/quality` |

* **Overview** - KPI cards (total citizens, government records, resolved
  citizens, duplicates eliminated, match confidence, participating
  agencies); an **"Entity Resolution In Action"** before/after panel
  (`GET /dashboard/showcase`) showing one real resolved cluster from the
  current run exactly as it looked pre-Splink (several agency records that
  look like different people) and post-Splink (one Unified Citizen Profile,
  with a link straight into its full Profile page); a records-by-agency bar
  chart; and a **Service Coverage Gaps** section counting how many resolved
  citizens have no on-file record with each of 8 illustrative services.
  Those counts are a plain population aggregate - the *ranked*, per-citizen
  version lives on each profile, where the peer group behind every figure
  can be shown alongside it.

* **Citizens** - an **Engagement Segments** summary (`GET /engagement`)
  grouping resolved citizens into 5 tiers by how many distinct agencies are
  linked to them, one card per tier plus an all-citizens aggregate, each
  clickable through to that tier's members. Below it, the directory table
  browses all resolved profiles alphabetically by default (`GET /search`
  with an empty `q`) or by name search, showing each citizen's engagement
  tier and their service-coverage gaps. Click any row for the full
  **Profile**, or "Export CSV" for the current listing.

* **Profile** (opened from any list) - a single resolved citizen's full case
  file (`GET /citizen/{id}/detail`): identity summary, agencies represented,
  linkage confidence, a **case-file card grid** with one full-detail card
  per linked record type (up to 12 - tax, benefits, healthcare, education,
  immigration, driving licence, passport, employment, housing, veteran, plus
  healthcare's hospital-visit and prescription sub-types), each showing that
  type's status and every administrative field the issuing agency captured;
  lists of benefits/prescriptions/education history; a **Lifestyle &
  Engagement Summary** of plain-language, evidence-cited observations - see
  [Design boundary](#design-boundary-no-risk-or-likelihood-scoring); a
  chronological **records timeline** where every individual linked record
  shows its own captured detail inline; a field-by-field **Match
  Explanation** calling out which identity fields agreed across the linked
  records and which still vary; a **Service Coverage** panel ranking this
  citizen's gaps by how common each service is among citizens with a
  comparable footprint, always with the peer-group size and definition
  attached; and a **Linkage Evidence** panel (`GET /citizen/{id}/graph`)
  reporting how strongly the records confirm each other - how many are
  cross-confirmed, and whether any single link is holding the profile
  together (a *bridge*, the classic over-merge signature). Includes working
  "Export Linked Data" (JSON), "Export Citizen Summary Report" (PDF), and
  "Copy Citizen ID" actions.

* **Review Queue** - resolved profiles worth a second look, worst first
  (`GET /anomaly-detection`). An Isolation Forest ranks profiles by how
  unusual their *linkage pattern* is - record and agency counts, records per
  agency, service-usage counts, and how much the merged records disagree
  about identity - never demographic or lifestyle fields. Each row carries
  both halves of an explanation: **what the model reacted to** (the features
  most responsible, measured by neutralising one at a time against the
  fitted model) and **the conflicting values actually on file** (e.g. two
  different dates of birth), which is the part a reviewer can adjudicate. A
  "Run Analysis" button refits with a configurable `contamination`.

  This page replaced a separate "Data Quality" review queue that ranked
  clusters by lowest match confidence. That was dropped because confidence
  is effectively constant on this data (see the distribution on Evaluation)
  - it was ranking on noise, while this ranking demonstrably surfaces 61% of
  all real linkage errors inside the top 5% of the population.

* **Evaluation** - every model in the application scored against the
  synthetic ground truth: the Splink-vs-baselines table, the clustering
  threshold trade-off curve with plain-language readings of what it says,
  the noise-robustness benchmark (with a "Run Benchmark" button), the
  anomaly-detector comparison, and the two linkage-quality distributions
  (match confidence and cluster size) that describe the linkage *without*
  reference to ground truth. Having both halves on one page is the point:
  the left-hand view is what an operator sees in production, the right-hand
  view is what the ground truth says is actually true. See
  [Evaluation](#evaluation-is-any-of-this-actually-working) above for what
  these numbers currently say.

## API endpoints

Every endpoint is open - there is no authentication (see
[No authentication](#no-authentication) above).

| Method | Path | Description |
|---|---|---|
| `POST` | `/generate-data` | (Re)generate the synthetic dataset. Returns `{"people": 10000, "records": ~75000}`. |
| `POST` | `/run-linkage` | Run the Splink pipeline and rebuild citizen profiles. Returns `{"clusters": ..., "duplicates_found": ...}`. |
| `GET` | `/citizen/{master_citizen_id}` | Unified Citizen Profile for one resolved person, e.g. `/citizen/MC00001`. |
| `GET` | `/citizen/{master_citizen_id}/detail` | Full profile dossier: linked source records, agency status fields, field-agreement explanation. Backs the Profile page. |
| `GET` | `/citizen/{master_citizen_id}/export/pdf` | The same dossier, rendered as a downloadable Citizen Summary Report PDF. |
| `GET` | `/search?q=&limit=` | Search resolved profiles by name; omit/empty `q` to list all profiles alphabetically. `limit` defaults to 50. |
| `GET` | `/export/directory.csv?q=` | Every profile matching `q` (or all profiles) as CSV - no `/search`-style result cap. |
| `GET` | `/dashboard` | Platform-wide summary metrics (population, clusters, confidence, agency participation). |
| `GET` | `/dashboard/showcase` | One representative resolved profile (raw linked records + Unified Citizen Profile) for the dashboard's before/after panel. |
| `GET` | `/quality?limit=` | Match-confidence histogram, cluster-size distribution, and a manual-review queue. `limit` (default 10) caps the review queue. |
| `GET` | `/export/review-queue.csv?limit=` | The manual-review backlog as CSV (default 500, not just the dashboard's top 10). |
| `POST` | `/anomaly-detection/run?contamination=` | (Re)fit the Isolation Forest over current citizen profiles and persist the result. `contamination` defaults to 0.05 (range 0.01-0.5). |
| `GET` | `/anomaly-detection?limit=` | The most recently computed anomaly-detection results: summary stats, score distribution, and per-profile scores, most anomalous first. |
| `GET` | `/engagement` | Citizens grouped into 5 engagement tiers by agency count, with per-tier summary stats. |
| `GET` | `/engagement/{tier}/members` | Resolved profiles assigned to one engagement tier. |
| `GET` | `/export/engagement-members.csv?tier=` | Every profile in one engagement tier as CSV. |
| `GET` | `/service-coverage` | Population-wide counts of citizens missing each of 8 illustrative agencies. Feeds the Overview page's service-coverage section. |
| `GET` | `/citizen/{master_citizen_id}/recommendations` | This citizen's coverage gaps, ranked by how common each service is among citizens with a comparable service footprint. Every figure carries its peer-group size and definition. |
| `GET` | `/citizen/{master_citizen_id}/graph` | The pairwise linkage evidence behind this profile: records as nodes, scored pairs as edges with field-level agreement, and any *bridge* whose removal would split the cluster. |
| `GET` | `/evaluation/linkage` | The Splink pipeline scored against the synthetic ground truth, alongside five deterministic baselines. |
| `GET` | `/evaluation/threshold-sweep` | Precision/recall/F1 across clustering thresholds, re-clustered from the persisted pairwise edges without retraining. |
| `POST` | `/benchmark/run?population=` | Re-run the whole pipeline at five noise levels and score each against ground truth. Takes ~1 minute; never touches the live dataset. |
| `GET` | `/benchmark` | The most recently computed noise-robustness benchmark. |
| `GET` | `/anomaly-detection/evaluation?contamination=` | Compare Isolation Forest, Local Outlier Factor and a robust z-score baseline by how well each surfaces real linkage errors. |
| `GET` | `/health` | Liveness/readiness probe. |
| `GET` | `/` | The dashboard frontend. |

Full request/response schemas (with field descriptions) are in the
Swagger UI at `/docs` or the ReDoc view at `/redoc`.

## Reproducibility

Every random choice in `data_generator.py` and `splink_service.py` is seeded
(`SEED = 6999`), so `POST /generate-data` followed by `POST /run-linkage`
produces identical results on every run **on a given platform** - including
which agencies each citizen is sampled into (`_sample_agencies` draws from
the same shared seeded RNG, never Python's unseeded global `random`).

**Known limitation - the dataset is not identical across platforms.**
`faker.date_of_birth()` returns different values on Windows and Linux for
the same seed and the same Faker version (verified: 1991-05-03 vs
1981-01-24 on a fresh seed with no other calls). Because a different date of
birth changes which branch `_vary_first_name` takes, and therefore how many
draws it consumes, the entire downstream sequence shifts. The Linux
container generates 74,811 records / 10,037 clusters where Windows
generates 74,955 / 10,044, moving the headline F1 from 0.9991 to 0.9986.
The conclusions are unaffected - the model still beats every baseline by the
same margin - but the exact figures in this README are the Windows ones.

The fix, if this matters to you, is to stop using Faker's
`date_of_birth()` (which derives from the host's "today") and sample the
date of birth from `AS_OF_DATE` with the shared seeded RNG - exactly what
every *record* date already does, and why `AS_OF_DATE` exists. That would
change the dataset, and therefore every figure quoted here, so it is called
out rather than done silently.

Splink resolves records from all eleven agencies together in one pool - every
row of the unified `records` table, distinguished by `record_type` - so a
citizen's tax record, healthcare registration, and driving licence are all
deduplicated against the same candidate pool. On this seeded dataset the
pipeline currently resolves:

* **~75,000** noisy government records across 11 agencies -> **~10,045**
  clusters (**~64,500** duplicates found)
* **~100%** average per-record match confidence

`faker` is nonetheless pinned to an exact version in
`app/requirements.txt` rather than a floor, since a version bump would
redraw the whole population on top of the platform difference above.
`duckdb` and `splink` are still ranges; a patch bump there can move the
linkage figures in the last decimal place without changing any conclusion.

Two things are deliberately *not* reproducible:

* A citizen profile's `age`, computed at read time against the real current
  date rather than a frozen seed value - see
  [Design boundary](#design-boundary-no-risk-or-likelihood-scoring) below
  for why. Expect it to increment on a citizen's real birthday with no data
  regenerated.
* Nothing else. In particular, the Review Queue's model was a reproducibility
  bug until recently: its feature query had no `ORDER BY`, and because
  Isolation Forest subsamples its training rows, DuckDB's hash-join ordering
  (which differs between database files) changed which citizens were flagged
  - 501 vs 486 on byte-identical clusters. It is sorted now, with a
  regression test.

## Design boundary: no risk or likelihood scoring

This demo deliberately computes **no risk, likelihood, or predictive
score of any kind** from a citizen's demographic or lifestyle data. The
Unified Citizen Profile includes an `engagement_tier` (a plain threshold
on how many agencies are linked) and a `lifestyle_summary` (plain-language
observations like "Frequent healthcare service user" or "Multiple
benefit types on record") - both are simple threshold checks over counts
or values *already present* on the profile, and every `lifestyle_summary`
item cites the exact evidence it restates in a `basis` field. Neither
ranks, scores, or predicts future behavior.

This boundary is intentional, not an oversight: scoring people's
likelihood of any behavior (including criminal activity) from demographic
or lifestyle proxies - marital status, service usage patterns, and
similar - is a well-documented source of discriminatory harm (predictive
policing and similar risk-scoring systems have repeatedly been shown to
encode and amplify bias against protected groups, since those proxy
variables correlate with race, religion, disability, and socioeconomic
status). This holds even for a synthetic demo: the scoring pattern itself,
not the underlying data, is the reusable artifact, so this project does
not implement one.

This same policy covers each record's `attributes` (e.g. `tax_band`,
`blood_type`, `disability_rating`, `rank`) - every one is a plain
administrative fact the issuing agency would realistically hold on file
(see `app/data_generator.py`'s `_ATTRIBUTE_GENERATORS`), never fed into
`lifestyle_summary` or any other derived field. `disability_rating` in
particular is worth naming explicitly: it's a real classification a
Veterans Affairs-equivalent agency tracks for benefit entitlement, the
same kind of recorded fact as `driving_licence.points` or `tax_band` -
not a computed judgment.

This boundary also governs the **Anomaly Detection** page's Isolation Forest
(`app/anomaly_service.py`): it is fit on structural/linkage signals only -
record count, agency count, match confidence, and government
service-usage counts - and deliberately never sees `age`, `marital_status`,
or any other demographic/lifestyle field. An "anomaly score" is not exempt
from this policy just because it isn't called a risk score: feeding it
demographic data would reproduce the same discriminatory-proxy pattern
under a different label. The page flags unusual *data/linkage patterns*
(the same job the old lowest-confidence review queue tried to do, but on a signal that actually varies -
does), never a citizen's demographic profile.

The **Service Coverage Gaps** page extends this same boundary to
service enrollment, and is worth naming explicitly since it was built
in response to an original ask for individual "behavior prediction"
with confidence scores - deliberately rejected in favor of this
design. `citizen_service.get_service_coverage_summary()` reports plain
population counts of how many citizens have no record with each agency,
each citing the exact evidence (`linked_agencies`).

Ranking an individual citizen's gaps (`app/recommendation_service.py`)
stays inside the same boundary, and the distinction is precise:

* The peer group is built **only** from which agencies hold a record for
  a citizen. It never uses age, marital status, address, monetary
  amounts, criminal-justice records, or anything else demographic or
  lifestyle-derived.
* The number reported is a **descriptive population statistic** - "82% of
  the 1,204 citizens with this service footprint hold a Healthcare
  record" - not a prediction about the individual. It says nothing about
  what this citizen will do, should do, or is likely to do, and every
  figure ships with the peer-group size and definition it was computed
  from, so it can be checked rather than trusted. Where the peer group
  would be too small to be meaningful it is widened, and the response
  says so.
* Citizens are never ranked against each other. The ranking is over one
  citizen's own coverage gaps.

There is still no likelihood score, no ranking of citizens by propensity
to act, and no model that makes a decision about a person.

The **Criminal Justice** agency and its `CRIMINAL_RECORD` record type deserve the same explicit
treatment, since criminal-offence data is a GDPR special category (like health data): it is a
factual record of an already-adjudicated legal outcome (conviction, caution, sentence,
rehabilitation status) - never a prediction of future behavior, and never derived from any other
field (marital status, lifestyle, etc.) or used to derive one. Two extra safeguards apply given
its sensitivity: it's excluded from the dashboard's auto-picked "before/after" showcase example
(`citizen_service.get_showcase_example()`), and excluded from the "Service Coverage Gaps" page's
`_SERVICE_GAP_AGENCY_PRIORITY` gap-check list (and from the peer-prevalence ranking in
`recommendation_service.py`), the same treatment already given to Immigration/Veterans Affairs
there. Agency
*membership* is not restricted, though - "Criminal Justice" shows up in a citizen's agency list
in the Directory/search/CSV like any other agency; only the record's content is scoped to the
full profile dossier.

## Project structure

```
SCV/
├── app/
│   ├── main.py             FastAPI app + endpoints
│   ├── bootstrap.py        Idempotent pipeline bootstrap (startup + Docker build step)
│   ├── cache.py            In-process memo cache for the expensive evaluation endpoints
│   ├── data_generator.py   Synthetic data generation (Faker, seeded)
│   ├── splink_service.py   Splink configuration, training, linkage
│   ├── citizen_service.py  Citizen aggregation, search, dashboard/quality queries
│   ├── anomaly_service.py  Isolation Forest anomaly detection + detector comparison
│   ├── evaluation_service.py  Linkage metrics vs ground truth, baselines, threshold sweep
│   ├── benchmark_service.py   Noise-robustness sweep (regenerate -> relink -> rescore)
│   ├── graph_service.py    Per-cluster linkage graph, bridges, field-level evidence
│   ├── recommendation_service.py  Coverage gaps ranked by peer prevalence
│   ├── exports.py          CSV/PDF report rendering
│   ├── models.py           Internal domain models
│   ├── schemas.py          API request/response schemas
│   ├── static/             Dashboard frontend (index.html, app.js)
│   ├── requirements.txt
│   ├── requirements-dev.txt  Test-only deps, kept out of the image
│   └── README.md           Module-level documentation
├── Dockerfile              Container image (builds the dataset at build time)
├── railway.json            Railway build/deploy config
├── .dockerignore
├── tests/                  pytest suite (see "Tests" below)
├── pytest.ini
├── data/                   DuckDB file lives here (gitignored)
└── env/                    Local conda environment
```

## Tests

```bash
pip install -r app/requirements.txt -r app/requirements-dev.txt
pytest                 # everything
pytest -m "not slow"   # skips the full 10,000-citizen regeneration check
```

The suite deliberately does **not** run Splink - training a real model takes
tens of seconds and would be testing Splink rather than this project. What it
covers is what this project owns: the evaluation metrics (against a tiny
fixture whose correct answers are worked out by hand in `tests/conftest.py`),
the noise model, the graph analysis, the peer-prevalence arithmetic, and the
API contracts.

One test is worth calling out. `test_default_profile_reproduces_the_shipped_dataset`
pins a digest of the full default dataset. Making the generator's noise
adjustable was a refactor of working code, and the guarantee that made it safe
is that the default profile still produces exactly the dataset it always did -
verified row-for-row against the previously generated database at the time of
the change. If that digest ever changes, the default synthetic data changed.

## Notes & known simplifications (POC scope)

* All amounts (tax paid, monthly benefit/housing payments) are generated in
  GBP only, and only ever shown per-record - there is no aggregate financial
  figure anywhere in a citizen profile.
* Per-record amounts are sampled from right-skewed (lognormal) distributions
  rather than `uniform(low, high)` - most amounts sit well below the stated
  maximum with a shrinking population stretching toward it, which is what
  real income/benefit distributions look like (a flat uniform spread looks
  obviously synthetic by comparison).
* Every agency's records carry the same kind of noisy, independently-captured
  identity fields and are resolved by the *same* Splink pipeline - no agency's
  data is trusted via a clean index. A citizen appears in 1-6 of the 10
  agencies (sampled independently per agency, some near-universal like
  Revenue & Tax and Healthcare, some rare like Veterans Affairs), and
  Healthcare citizens additionally get 1-3 hospital visits and 0-2
  prescriptions - each its own independently-noised record, reflecting how
  a citizen's healthcare history really does accumulate multiple encounters
  over time.
* DuckDB is used as a single-writer embedded database, opened per request;
  this is appropriate for a demo/POC but not for high-concurrency production
  workloads.
* The frontend is intentionally framework-free (no bundler/build step) to
  keep the whole project runnable with a single `uvicorn` command.
* There is no authentication, authorisation, or access logging of any kind -
  every endpoint is open and the dashboard needs no sign-in. Fine for a local
  demo over synthetic data; the first thing that would have to change before
  anything resembling this held real records.
* "Service Coverage Gaps" covers 8 illustrative agencies, matched against
  what's missing from a profile's `linked_agencies`. It deliberately
  excludes Immigration/Veterans Affairs (apply only to citizens in
  specific circumstances) and Criminal Justice (recommending someone
  acquire a criminal record makes no sense). Gaps are ordered by measured
  peer prevalence rather than a hardcoded priority list, but this is a
  coverage check against synthetic data - not a real eligibility engine.
* The evaluation and benchmark pages exist because the data is synthetic.
  On real data there would be no `person_index` to score against, and
  those endpoints would have to be replaced by a labelled sample or
  clerical review. `evaluation_service.has_ground_truth()` guards for this
  rather than assuming the column is present.
* `marital_status` is assigned once per synthetic citizen and only ever
  appears on `TAX_RECORD`/`BENEFITS_RECORD` rows (the two record types a
  real agency would realistically capture it on); `age` is derived, not
  stored, from the citizen's resolved `date_of_birth`.
* Every record also carries an `attributes` map of 4-6 richer,
  record-type-specific administrative fields (e.g. `tax_band`,
  `blood_type`, `rank`) - see `app/data_generator.py`'s
  `_ATTRIBUTE_GENERATORS`. Like `amount`, these are per-record detail
  only, shown on the profile/timeline/PDF but never aggregated into a
  summary figure; CSV exports stay flat/summary-level since per-record
  attributes (which vary in shape by record_type) don't fit a
  one-row-per-citizen CSV.
