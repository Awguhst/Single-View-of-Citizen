// CitizenLink (Single View of Citizen) dashboard frontend.
// Plain JS + Chart.js, talking directly to the FastAPI JSON endpoints below.
// No build step / framework - this is a thin presentation layer over the API.

const GBP = new Intl.NumberFormat("en-GB", { style: "currency", currency: "GBP", maximumFractionDigits: 0 });
const INT = new Intl.NumberFormat("en-GB");

const CHART_COLORS = {
  primary: "#47a8f6",
  primaryDark: "#0055ff",
  accent: "#38bdf8",
  border: "#334155",
  muted: "#94a3b8",
  danger: "#f87171",
  // Material 3 semantic tokens, added for the tier <-> badge color alignment below.
  // primaryContainer/tertiary are numerically identical to primaryDark/accent
  // (same M3 seed color/theme) - this is a semantic rename, not a color change.
  error: "#f87171",
  secondary: "#94a3b8",
  primaryContainer: "#0055ff",
  tertiary: "#38bdf8",
};

const ENGAGEMENT_TIER_COLORS = {
  "Minimal Engagement": CHART_COLORS.error,
  "Limited Engagement": CHART_COLORS.secondary,
  "Moderate Engagement": CHART_COLORS.primary,
  "High Engagement": CHART_COLORS.primaryContainer,
  "Full Engagement": CHART_COLORS.tertiary,
};

const ENGAGEMENT_TIER_CLASSES = {
  "Minimal Engagement": "text-error bg-error/10 border-error/20",
  "Limited Engagement": "text-secondary bg-secondary/10 border-secondary/20",
  "Moderate Engagement": "text-primary bg-primary/10 border-primary/20",
  "High Engagement": "text-primary-container bg-primary-container/10 border-primary-container/20",
  "Full Engagement": "text-tertiary bg-tertiary/10 border-tertiary/20",
};

const RECORD_TYPE_META = {
  TAX_RECORD: { icon: "receipt_long", label: "Tax Record" },
  BENEFITS_RECORD: { icon: "payments", label: "Benefits Record" },
  HEALTHCARE_REGISTRATION: { icon: "health_and_safety", label: "Healthcare Registration" },
  HOSPITAL_VISIT: { icon: "local_hospital", label: "Hospital Visit" },
  PRESCRIPTION: { icon: "medication", label: "Prescription" },
  EDUCATION_RECORD: { icon: "school", label: "Education Record" },
  IMMIGRATION_RECORD: { icon: "public", label: "Immigration Record" },
  DRIVING_LICENCE: { icon: "directions_car", label: "Driving Licence" },
  PASSPORT: { icon: "flight", label: "Passport" },
  EMPLOYMENT_REGISTRATION: { icon: "work", label: "Employment Registration" },
  HOUSING_BENEFIT: { icon: "home", label: "Housing Benefit" },
  VETERAN_RECORD: { icon: "military_tech", label: "Veteran Record" },
  CRIMINAL_RECORD: { icon: "gavel", label: "Criminal Record" },
};

// Case-file section grouping - purely a display grouping over the 13
// record types in RECORD_TYPE_META, each type appears in exactly one
// section. Order within a section is the order cards render.
const CASE_FILE_SECTIONS = [
  { title: "Identity & Travel", types: ["PASSPORT", "DRIVING_LICENCE", "IMMIGRATION_RECORD"] },
  { title: "Health", types: ["HEALTHCARE_REGISTRATION", "HOSPITAL_VISIT", "PRESCRIPTION"] },
  { title: "Finance & Employment", types: ["TAX_RECORD", "EMPLOYMENT_REGISTRATION", "BENEFITS_RECORD", "HOUSING_BENEFIT"] },
  { title: "Education, Service & Legal", types: ["EDUCATION_RECORD", "VETERAN_RECORD", "CRIMINAL_RECORD"] },
];

// The only two record types with genuinely multiple instances per citizen -
// every other type is capped at 0-1 records per citizen by construction.
const MULTI_INSTANCE_RECORD_TYPES = new Set(["HOSPITAL_VISIT", "PRESCRIPTION"]);

const SERVICE_RECOMMENDATIONS = {
  "Healthcare": { icon: "local_hospital", title: "Register with a GP practice", blurb: "Ensures ongoing access to primary care and prescriptions.", office: "Ferngate Community Health Office" },
  "Revenue & Tax": { icon: "receipt_long", title: "Register for Self Assessment", blurb: "Keeps tax filings and payments up to date.", office: "Northshire Revenue Office" },
  "Employment": { icon: "work", title: "Register with Jobcentre Plus", blurb: "Unlocks employment support and job-matching services.", office: "Aldermoor Employment Office" },
  "Driver Licensing": { icon: "directions_car", title: "Apply for a driving licence", blurb: "Needed for driving eligibility and identity verification.", office: "Aldermoor Licensing Centre" },
  "Passport Office": { icon: "flight", title: "Apply for a passport", blurb: "Required for international travel and identity verification.", office: "Rosewell Passport Office" },
  "Social Security": { icon: "payments", title: "Check benefit entitlement", blurb: "May be eligible for support such as Universal Credit or a pension.", office: "Kingsmere Social Security Office" },
  "Education": { icon: "school", title: "Register for further education", blurb: "Explore adult education, apprenticeships, or training programmes.", office: "Aldergate Sixth Form College" },
  "Housing": { icon: "home", title: "Apply for housing assistance", blurb: "May qualify for housing benefit or social housing support.", office: "Kingsmere Housing Authority" },
};

let charts = {};

function destroyChart(key) {
  if (charts[key]) {
    charts[key].destroy();
    delete charts[key];
  }
}

function showToast(message, isError = false) {
  const toast = document.getElementById("toast");
  toast.textContent = message;
  toast.className =
    "toast fixed top-20 right-8 z-50 px-4 py-3 rounded-lg text-sm font-medium shadow-lg " +
    (isError ? "bg-red-500/90 text-white" : "bg-card border border-primary-dark text-on-surface");
  toast.style.opacity = "1";
  toast.style.pointerEvents = "auto";
  clearTimeout(window.__toastTimer);
  window.__toastTimer = setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.pointerEvents = "none";
  }, 4000);
}

async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `${path} failed (${res.status})`);
  }
  return res.json();
}

// Fetches the blob and triggers the save itself, rather than using a plain
// <a href>, so that a failed download surfaces as a toast with the server's
// error message instead of navigating away to an error page. Same
// client-side download pattern as exportProfileJson, just sourced from a
// server response instead of in-memory JSON.
async function downloadFile(path, fallbackName) {
  let res;
  try {
    res = await fetch(path);
  } catch (e) {
    showToast("Download failed - network error.", true);
    return;
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    showToast(body.detail || `Download failed (${res.status})`, true);
    return;
  }
  const blob = await res.blob();
  const disposition = res.headers.get("Content-Disposition") || "";
  const match = disposition.match(/filename="?([^"]+)"?/);
  const filename = match ? match[1] : fallbackName;
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// The backend returns a 400 ending in this exact phrase whenever citizen
// data hasn't been linked yet - "No citizen profiles found. Call POST
// /run-linkage first." from most endpoints, "No linkage results found. Call
// POST /run-linkage first." from a couple of others. Either way, this means
// data was (re)generated but /run-linkage hasn't (re)run yet - an expected
// step of the pipeline, not a failure, so callers should show a quiet inline
// empty state instead of a red error toast.
const NO_PROFILES_SUFFIX = "Call POST /run-linkage first.";
function isNoProfilesError(e) {
  return typeof e.message === "string" && e.message.endsWith(NO_PROFILES_SUFFIX);
}

// ---------------------------------------------------------------------------
// Startup
// ---------------------------------------------------------------------------
async function startApp() {
  await loadDashboard();
  // Also populates `coverageCandidateAgencies`, which the Citizens table's
  // coverage-gap column reads - so this has to run before the first listing.
  await loadServiceCoverageSummary();
  await runSearch("", { navigate: false }); // pre-populate Citizens with the A-Z listing
}

// ---------------------------------------------------------------------------
// View switching
// ---------------------------------------------------------------------------
// The profile page is a sub-page of another list view (reached only by
// clicking a result row in one), not a top-level nav destination - tracks
// whichever list view it was opened from so the nav highlight and the
// "Back to ..." button both return there instead of always Directory.
let lastListView = "directory";
const LIST_VIEW_LABELS = {
  dashboard: "Back to Overview",
  directory: "Back to Citizens",
  review: "Back to Review Queue",
  evaluation: "Back to Evaluation",
};

function switchView(view) {
  document.querySelectorAll(".view").forEach((el) => el.classList.remove("active"));
  document.getElementById(`view-${view}`).classList.add("active");

  const navTarget = view === "profile" ? lastListView : view;
  document.querySelectorAll(".nav-link").forEach((el) => el.classList.remove("active"));
  document.querySelector(`.nav-link[data-view="${navTarget}"]`)?.classList.add("active");

  if (view === "profile") {
    document.getElementById("btn-back-directory-label").textContent = LIST_VIEW_LABELS[lastListView];
  }

  if (view === "dashboard") { loadDashboard(); loadServiceCoverageSummary(); }
  if (view === "directory") loadEngagementSummary();
  if (view === "review") loadReviewQueue();
  if (view === "evaluation") loadEvaluation();
}

document.querySelectorAll(".nav-link").forEach((btn) => {
  // Every current .nav-link has a data-view (including Settings); this guard
  // is just defensive against a future nav button that doesn't.
  if (!btn.dataset.view) return;
  btn.addEventListener("click", () => switchView(btn.dataset.view));
});

// ---------------------------------------------------------------------------
// Dashboard view
// ---------------------------------------------------------------------------
function kpiCard(label, value, icon, sub) {
  return `
    <div class="entity-card rounded-lg p-5 flex flex-col justify-between min-h-28">
      <div class="flex items-center justify-between">
        <span class="text-[11px] uppercase tracking-wider text-on-surface-variant font-semibold">${label}</span>
        <span class="material-symbols-outlined text-primary">${icon}</span>
      </div>
      <div>
        <p class="text-2xl font-bold text-on-surface">${value}</p>
        ${sub ? `<p class="text-[11px] text-on-surface-variant mt-1">${sub}</p>` : ""}
      </div>
    </div>`;
}

async function loadDashboard() {
  lastListView = "dashboard";
  const grid = document.getElementById("kpi-grid");
  grid.innerHTML = Array(8)
    .fill('<div class="entity-card rounded-lg p-5 h-28"><div class="skeleton w-full h-full rounded"></div></div>')
    .join("");

  let d;
  try {
    d = await api("/dashboard");
  } catch (err) {
    showToast(err.message, true);
    return;
  }

  const duplicateReductionPct = d.government_records ? (d.duplicates_eliminated / d.government_records) * 100 : 0;

  grid.innerHTML = [
    kpiCard("Total Citizens", INT.format(d.total_citizens), "groups", "Ground-truth citizens generated"),
    kpiCard("Government Records", INT.format(d.government_records), "database", "Records ingested across all agencies"),
    kpiCard("Resolved Citizens", INT.format(d.resolved_citizens), "bubble_chart", "master_citizen_id groups"),
    kpiCard("Duplicate Records Eliminated", INT.format(d.duplicates_eliminated), "auto_fix_high", "Records merged into another cluster"),
    kpiCard("Avg Match Confidence", (d.avg_match_probability * 100).toFixed(2) + "%", "verified", "Mean per-record linkage confidence"),
    kpiCard("Participating Agencies", INT.format(d.participating_agencies), "account_balance", `Avg ${d.avg_agencies_per_citizen.toFixed(2)} agencies linked per citizen`),
    kpiCard("Avg Agencies per Citizen", d.avg_agencies_per_citizen.toFixed(2), "hub", "Distinct agencies linked, per resolved citizen"),
    kpiCard("Duplicate Reduction Rate", duplicateReductionPct.toFixed(1) + "%", "compress", "Share of ingested records consolidated via entity resolution"),
  ].join("");

  // Fetched separately from /dashboard above: citizen_profiles (and so
  // engagement tiers) only exist after /run-linkage, and data can be
  // (re)generated without linkage having (re)run yet. Isolating this call
  // means that expected in-between state only blanks this one chart instead
  // of failing the whole page (in particular, the agencies chart below
  // depends only on /dashboard and must still render).
  const engagementEmpty = document.getElementById("chart-engagement-empty");
  destroyChart("engagement");
  let e = null;
  try {
    e = await api("/engagement");
  } catch (err) {
    if (!isNoProfilesError(err)) showToast(err.message, true);
    engagementEmpty.textContent = isNoProfilesError(err)
      ? "Run linkage to see citizens grouped by engagement tier."
      : "Failed to load - see toast for details.";
    engagementEmpty.classList.remove("hidden");
  }

  if (e) {
    engagementEmpty.classList.add("hidden");
    charts.engagement = new Chart(document.getElementById("chart-engagement"), {
      type: "doughnut",
      data: {
        labels: e.tiers.map((t) => t.engagement_tier),
        datasets: [
          {
            data: e.tiers.map((t) => t.citizen_count),
            backgroundColor: e.tiers.map((t) => ENGAGEMENT_TIER_COLORS[t.engagement_tier]),
            borderColor: "#0f172a",
            borderWidth: 2,
          },
        ],
      },
      options: {
        maintainAspectRatio: false,
        plugins: { legend: { position: "bottom", labels: { color: "#94a3b8", font: { size: 11 } } } },
      },
    });
  }

  const agencies = Object.keys(d.agency_record_counts);
  const counts = Object.values(d.agency_record_counts);
  destroyChart("agencies");
  charts.agencies = new Chart(document.getElementById("chart-agencies"), {
    type: "bar",
    data: {
      labels: agencies,
      datasets: [{ data: counts, backgroundColor: CHART_COLORS.primaryDark, borderRadius: 4 }],
    },
    options: {
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#94a3b8" }, grid: { color: "#1e293b" } },
        y: { ticks: { color: "#94a3b8" }, grid: { color: "#1e293b" } },
      },
    },
  });

  loadShowcase();
}

function showcasePersonCard(r) {
  const meta = RECORD_TYPE_META[r.record_type] || { icon: "description", label: r.record_type };
  return `
    <div class="bg-surface-container-high/60 border border-border rounded-lg p-4">
      <div class="flex items-center gap-3 mb-2">
        <div class="w-8 h-8 rounded-full bg-red-500/10 border border-red-500/30 flex items-center justify-center text-xs font-bold text-red-400 shrink-0">
          ${initials(`${r.first_name} ${r.last_name}`)}
        </div>
        <div class="min-w-0">
          <p class="font-semibold text-sm truncate">${r.first_name} ${r.last_name}</p>
          <p class="text-[11px] text-on-surface-variant truncate">${r.agency} &middot; ${meta.label}</p>
        </div>
      </div>
      <p class="text-xs text-on-surface-variant truncate">${r.status ?? "no status on file"} &middot; ${r.record_date}</p>
      <p class="text-xs text-on-surface-variant truncate">${r.email ?? "no email on file"}</p>
      <p class="text-xs text-on-surface-variant truncate">${r.phone ?? "no phone on file"} &middot; ${r.postcode ?? "no postcode"}</p>
    </div>`;
}

async function loadShowcase() {
  const container = document.getElementById("showcase-content");
  container.innerHTML = `<div class="entity-card rounded-lg p-8"><div class="skeleton w-full h-40 rounded"></div></div>`;

  let s;
  try {
    s = await api("/dashboard/showcase");
  } catch (e) {
    container.innerHTML = `<div class="entity-card rounded-lg p-8 text-center text-on-surface-variant text-sm">Run linkage first to see a before/after example here.</div>`;
    return;
  }

  const showcaseRecords = s.records;

  container.innerHTML = `
    <div class="grid grid-cols-12 gap-4 items-stretch">
      <div class="col-span-12 lg:col-span-5 border-2 border-dashed border-red-500/30 rounded-lg p-5 bg-red-500/[0.03]">
        <span class="badge bg-red-500/15 text-red-400 mb-3 inline-block">Before Linking</span>
        <p class="text-sm text-on-surface-variant mb-4">${showcaseRecords.length} separate agency records look like ${showcaseRecords.length} different people</p>
        <div class="space-y-3">
          ${showcaseRecords.map(showcasePersonCard).join("")}
        </div>
      </div>

      <div class="col-span-12 lg:col-span-2 flex flex-row lg:flex-col items-center justify-center gap-2 py-4">
        <span class="material-symbols-outlined text-primary text-4xl">arrow_forward</span>
        <span class="text-[11px] uppercase tracking-wider text-on-surface-variant font-semibold">Probabilistic Linkage</span>
        ${confidenceBadge(s.confidence_score)}
      </div>

      <div class="col-span-12 lg:col-span-5 border-2 border-primary-dark/40 rounded-lg p-5 bg-primary/[0.04] flex flex-col">
        <span class="badge bg-emerald-500/15 text-emerald-400 mb-3 self-start">After Linking</span>
        <div class="flex items-center gap-3 mb-4">
          <div class="w-12 h-12 rounded-full bg-primary-dark/20 border-2 border-primary-dark flex items-center justify-center text-sm font-bold text-primary shrink-0">
            ${initials(s.preferred_name)}
          </div>
          <div class="min-w-0">
            <p class="font-bold truncate">${s.preferred_name}</p>
            <p class="text-xs font-mono text-on-surface-variant">${s.master_citizen_id}</p>
          </div>
        </div>
        <p class="text-[11px] text-on-surface-variant">Linked agencies</p>
        <p class="text-sm mb-4">${s.linked_agencies.join(", ")}</p>
        <div class="grid grid-cols-2 gap-3 text-sm mb-4">
          <div><p class="text-[11px] text-on-surface-variant">Records Linked</p><p class="font-semibold">${s.record_count}</p></div>
          <div><p class="text-[11px] text-on-surface-variant">Engagement Tier</p><p class="font-semibold">${s.engagement_tier}</p></div>
        </div>
        <p class="text-[11px] text-on-surface-variant">Government records on file</p>
        <p class="text-sm mb-4">${showcaseRecords.map((r) => `${(RECORD_TYPE_META[r.record_type] || { label: r.record_type }).label} (${r.agency})`).join(", ")}</p>
        <button id="btn-showcase-profile" class="mt-auto text-xs font-bold text-primary hover:underline flex items-center gap-1 self-start">
          View full profile <span class="material-symbols-outlined text-sm">arrow_forward</span>
        </button>
      </div>
    </div>`;

  document.getElementById("btn-showcase-profile").addEventListener("click", () => loadProfile(s.master_citizen_id));
}

// ---------------------------------------------------------------------------
// Citizen Directory / search view
// ---------------------------------------------------------------------------
// The coverage-gap column on the Citizens table. `coverageCandidateAgencies`
// is populated from the Overview's /service-coverage response, so the agency
// list still has exactly one definition - the backend's.
function coverageGapCell(linkedAgencies) {
  if (!coverageCandidateAgencies.length) {
    return `<span class="text-xs text-on-surface-variant">-</span>`;
  }
  const gaps = coverageCandidateAgencies.filter((a) => !linkedAgencies.includes(a));
  if (!gaps.length) {
    return `<span class="text-xs text-on-surface-variant">Fully covered</span>`;
  }
  return `<span class="text-xs" title="${gaps.join(", ")}">${gaps.length} gap${gaps.length === 1 ? "" : "s"}
    <span class="text-on-surface-variant">&middot; ${gaps.slice(0, 2).join(", ")}${gaps.length > 2 ? "&hellip;" : ""}</span></span>`;
}

function confidenceBadge(p) {
  const pct = (p * 100).toFixed(1) + "%";
  if (p >= 0.99) return `<span class="badge bg-emerald-500/15 text-emerald-400">${pct}</span>`;
  if (p >= 0.9) return `<span class="badge bg-amber-500/15 text-amber-400">${pct}</span>`;
  return `<span class="badge bg-red-500/15 text-red-400">${pct}</span>`;
}

const DIRECTORY_PAGE_SIZE = 50;
let directoryLimit = DIRECTORY_PAGE_SIZE;

async function runSearch(query, { navigate = true, resetLimit = true } = {}) {
  lastListView = "directory";
  if (navigate) switchView("directory");
  if (resetLimit) directoryLimit = DIRECTORY_PAGE_SIZE;

  const empty = document.getElementById("directory-empty");
  const table = document.getElementById("directory-table");
  const caption = document.getElementById("directory-caption");
  const rows = document.getElementById("directory-rows");
  const showMoreBtn = document.getElementById("btn-directory-show-more");
  const trimmed = query.trim();
  empty.textContent = trimmed ? "Searching..." : "Loading profiles...";
  empty.classList.remove("hidden");
  table.classList.add("hidden");
  caption.classList.add("hidden");
  showMoreBtn.classList.add("hidden");

  let data;
  try {
    data = await api(`/search?q=${encodeURIComponent(trimmed)}&limit=${directoryLimit}`);
  } catch (e) {
    if (isNoProfilesError(e)) {
      empty.textContent = "No resolved citizens yet - generate data and run linkage first.";
    } else {
      showToast(e.message, true);
      empty.textContent = "Search failed - see toast for details.";
    }
    return;
  }

  if (data.results.length === 0) {
    empty.textContent = trimmed
      ? `No resolved citizens matched "${trimmed}".`
      : "No resolved citizens yet - generate data and run linkage first.";
    empty.classList.remove("hidden");
    table.classList.add("hidden");
    return;
  }

  rows.innerHTML = data.results
    .map(
      (r) => `
        <tr class="hover:bg-surface-container-high transition-colors">
          <td class="px-5 py-3 font-mono text-xs text-primary">${r.master_citizen_id}</td>
          <td class="px-5 py-3 font-medium">${r.preferred_name}</td>
          <td class="px-5 py-3 text-xs text-on-surface-variant">${r.linked_agencies.join(", ")}</td>
          <td class="px-5 py-3"><span class="badge border ${ENGAGEMENT_TIER_CLASSES[r.engagement_tier]}">${r.engagement_tier}</span></td>
          <td class="px-5 py-3">${coverageGapCell(r.linked_agencies)}</td>
          <td class="px-5 py-3 text-center">
            <button class="view-profile text-on-surface-variant hover:text-primary" data-id="${r.master_citizen_id}">
              <span class="material-symbols-outlined text-base">visibility</span>
            </button>
          </td>
        </tr>`
    )
    .join("");

  caption.textContent = trimmed
    ? `Showing ${INT.format(data.results.length)} of ${INT.format(data.total)} matches for "${trimmed}"`
    : `Showing ${INT.format(data.results.length)} of ${INT.format(data.total)} citizens, A-Z`;
  caption.classList.remove("hidden");
  showMoreBtn.classList.toggle("hidden", data.results.length >= data.total);

  empty.classList.add("hidden");
  table.classList.remove("hidden");
}

document.getElementById("directory-rows")?.addEventListener("click", (e) => {
  const btn = e.target.closest(".view-profile");
  if (!btn) return;
  loadProfile(btn.dataset.id);
});

document.getElementById("btn-directory-show-more").addEventListener("click", async (e) => {
  const btn = e.currentTarget;
  setBusy(btn, true, "Loading...");
  directoryLimit += DIRECTORY_PAGE_SIZE;
  try {
    await runSearch(document.getElementById("search-input").value, { navigate: false, resetLimit: false });
  } finally {
    setBusy(btn, false);
  }
});

document.getElementById("btn-export-directory").addEventListener("click", () => {
  const q = document.getElementById("search-input").value.trim();
  downloadFile(`/export/directory.csv?q=${encodeURIComponent(q)}`, "directory.csv");
});

// Citizens is the only searchable list, so Enter always searches it - and
// jumps there from wherever you are.
document.getElementById("search-input").addEventListener("keydown", (e) => {
  if (e.key !== "Enter") return;
  runSearch(e.target.value);
});

// Clearing the box reverts to the alphabetical listing, but without yanking
// the user over to Citizens if they're looking at something else.
document.getElementById("search-input").addEventListener("input", (e) => {
  if (e.target.value.trim() !== "") return;
  runSearch("", { navigate: false });
});

// ---------------------------------------------------------------------------
// Profile detail page
// ---------------------------------------------------------------------------
function initials(name) {
  return name.split(" ").filter(Boolean).slice(0, 2).map((w) => w[0].toUpperCase()).join("");
}

// Generic, data-driven rendering for a record's `attributes` map - one
// renderer reused for every record type, rather than a bespoke template
// per type. Every value shown here is a plain administrative fact already
// on the record; this is display-only and never computes anything.
function humanizeKey(key) {
  return key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function attributesList(attrs) {
  const entries = Object.entries(attrs || {});
  if (!entries.length) return "";
  return `
    <dl class="grid grid-cols-2 gap-x-4 gap-y-1 text-xs mt-2 pt-2 border-t border-outline-variant/20">
      ${entries.map(([k, v]) => `<dt class="text-on-surface-variant">${humanizeKey(k)}</dt><dd class="font-medium text-right truncate">${v ?? "-"}</dd>`).join("")}
    </dl>`;
}

// Plain-language, evidence-cited observations only - never a score or
// prediction (see citizen_service.py's module docstring for the policy
// this UI is deliberately reinforcing).
function lifestyleTagRow(item) {
  return `
    <div class="flex items-start gap-3 text-sm py-2">
      <span class="material-symbols-outlined text-primary text-base mt-0.5">info</span>
      <div>
        <span class="font-medium">${item.tag}</span>
        <p class="text-xs text-on-surface-variant mt-0.5">${item.basis}</p>
      </div>
    </div>`;
}

function lifestyleSummaryPanel(items) {
  if (!items || items.length === 0) return "";
  return `
    <div class="entity-card rounded-lg p-6">
      <h3 class="text-base font-semibold mb-1">Lifestyle &amp; Engagement Summary</h3>
      <p class="text-on-surface-variant text-xs mb-3">Plain-language observations, each citing the exact record(s) it is based on.</p>
      <div class="divide-y divide-border/40">${items.map(lifestyleTagRow).join("")}</div>
    </div>`;
}

// ---------------------------------------------------------------------------
// Profile: service coverage (peer prevalence) and linkage evidence
// ---------------------------------------------------------------------------
// Both panels load after the main profile render rather than blocking it -
// they are supporting analysis, and a profile must still be readable if
// either is unavailable (for example on a database linked before pairwise
// edges were persisted).

function coverageBar(prevalence) {
  const pct = Math.round(prevalence * 100);
  return `
    <div class="h-1.5 w-full bg-surface-container-high rounded-full overflow-hidden">
      <div class="h-full bg-primary rounded-full" style="width:${pct}%"></div>
    </div>`;
}

function coverageRow(rec) {
  const meta = SERVICE_RECOMMENDATIONS[rec.agency] || { icon: "help", title: rec.agency, blurb: "" };
  const pct = (rec.peer_prevalence * 100).toFixed(1);
  return `
    <div class="py-3 border-t border-border/40">
      <div class="flex items-start justify-between gap-4 mb-2">
        <div class="flex items-start gap-3 min-w-0">
          <span class="material-symbols-outlined text-primary text-base mt-0.5">${meta.icon}</span>
          <div class="min-w-0">
            <p class="font-medium text-sm">${meta.title}</p>
            <p class="text-on-surface-variant text-xs mt-0.5">${meta.blurb}</p>
          </div>
        </div>
        <div class="text-right shrink-0">
          <p class="font-semibold text-sm">${pct}%</p>
          <p class="text-on-surface-variant text-[10px]">of peers</p>
        </div>
      </div>
      ${coverageBar(rec.peer_prevalence)}
      <p class="text-on-surface-variant text-[11px] mt-1.5">
        ${INT.format(rec.peers_with_record)} of ${INT.format(rec.peer_group_size)} comparable citizens hold a
        ${rec.agency} record. This citizen does not.
      </p>
    </div>`;
}

function serviceCoveragePanel(p) {
  // Rendered immediately with the agency chips the profile already carries;
  // the ranked gaps are filled in by loadCoverageRecommendations().
  return `
    <div class="entity-card rounded-lg p-6" id="coverage-panel">
      <h3 class="text-base font-semibold mb-1">Service Coverage</h3>
      <p class="text-on-surface-variant text-xs mb-3">
        Agencies this citizen is linked to, and their coverage gaps ranked by how common each service is among
        citizens with a comparable service footprint.
      </p>
      <div class="flex flex-wrap gap-1 mb-2">
        ${p.linked_agencies
          .map(
            (a) =>
              `<span class="text-[10px] px-2 py-0.5 rounded border border-outline-variant/40 text-on-surface-variant">${a}</span>`
          )
          .join("")}
      </div>
      <div id="coverage-recommendations" class="mt-3">
        <div class="skeleton w-full h-16 rounded"></div>
      </div>
    </div>`;
}

async function loadCoverageRecommendations(masterCitizenId) {
  const target = document.getElementById("coverage-recommendations");
  if (!target) return;

  let data;
  try {
    data = await api(`/citizen/${masterCitizenId}/recommendations`);
  } catch (e) {
    target.innerHTML = `<p class="text-on-surface-variant text-xs pt-3 border-t border-border/40">${e.message}</p>`;
    return;
  }

  if (data.fully_covered) {
    target.innerHTML = `
      <div class="flex items-start gap-3 text-sm pt-3 border-t border-border/40">
        <span class="material-symbols-outlined text-primary text-base mt-0.5">task_alt</span>
        <div>
          <span class="font-medium">On file with every agency checked</span>
          <p class="text-xs text-on-surface-variant mt-0.5">Linked to all of: ${data.candidate_agencies.join(", ")}.</p>
        </div>
      </div>`;
    return;
  }

  const peerDefinition = data.peer_group_is_population
    ? "the whole resolved population"
    : `citizens also linked to ${data.peer_group_definition.join(", ")}`;

  target.innerHTML = `
    ${data.recommendations.map(coverageRow).join("")}
    <div class="pt-3 mt-1 border-t border-border/40 text-[11px] text-on-surface-variant leading-relaxed">
      <span class="font-medium text-on-surface">Peer group:</span>
      ${INT.format(data.peer_group_size)} profiles - ${peerDefinition}.
      ${data.backed_off ? "Widened from this citizen's exact footprint to reach a usable sample size." : ""}
      These are descriptive population statistics, not predictions about this individual, and citizens are never
      ranked against one another. The peer group is built only from which agencies hold a record - never
      from demographic or lifestyle data.
    </div>`;
}

function evidenceChip(item) {
  const tone =
    item.agreement === "agreed"
      ? "text-primary border-primary/30"
      : item.agreement === "disagreed"
        ? "text-red-400 border-red-400/30"
        : "text-on-surface-variant border-outline-variant/40";
  return `<span class="text-[10px] px-2 py-0.5 rounded border ${tone}">${item.field.replace(/_/g, " ")}: ${item.agreement}</span>`;
}

function linkageEvidencePanel(graph) {
  if (graph.node_count < 2) {
    return `
      <p class="text-on-surface-variant text-xs">
        This profile rests on a single record, so there was no linkage decision to make.
      </p>`;
  }

  const densityPct = (graph.density * 100).toFixed(0);
  const bridge = graph.weakest_bridge;

  const verdict = bridge
    ? `
      <div class="flex items-start gap-3 p-3 rounded bg-amber-400/5 border border-amber-400/20">
        <span class="material-symbols-outlined text-amber-400 text-base mt-0.5">link_off</span>
        <div>
          <p class="font-medium text-sm">${graph.bridge_count} single point${graph.bridge_count === 1 ? "" : "s"} of failure</p>
          <p class="text-on-surface-variant text-xs mt-0.5">
            Removing one link would split this profile apart. The weakest scored
            <span class="font-mono">${bridge.match_probability.toFixed(4)}</span> and joins
            <span class="font-mono">${bridge.source}</span> to <span class="font-mono">${bridge.target}</span> on:
          </p>
          <div class="flex flex-wrap gap-1 mt-2">
            ${bridge.evidence.map(evidenceChip).join("")}
          </div>
        </div>
      </div>`
    : `
      <div class="flex items-start gap-3 p-3 rounded bg-primary/5 border border-primary/20">
        <span class="material-symbols-outlined text-primary text-base mt-0.5">verified</span>
        <div>
          <p class="font-medium text-sm">No single point of failure</p>
          <p class="text-on-surface-variant text-xs mt-0.5">
            Every record is confirmed by more than one link, so no individual comparison could have created this
            profile on its own.
          </p>
        </div>
      </div>`;

  return `
    <div class="grid grid-cols-3 gap-3 mb-4 text-center">
      <div>
        <p class="text-lg font-bold">${graph.node_count}</p>
        <p class="text-on-surface-variant text-[10px] uppercase tracking-wider">Records</p>
      </div>
      <div>
        <p class="text-lg font-bold">${graph.load_bearing_edge_count}</p>
        <p class="text-on-surface-variant text-[10px] uppercase tracking-wider">Confirming links</p>
      </div>
      <div>
        <p class="text-lg font-bold">${densityPct}%</p>
        <p class="text-on-surface-variant text-[10px] uppercase tracking-wider">Cross-confirmed</p>
      </div>
    </div>
    ${verdict}
    <p class="text-on-surface-variant text-[11px] mt-3 leading-relaxed">
      A link counts as confirming only at or above the clustering threshold of
      <span class="font-mono">${graph.cluster_threshold}</span>; weaker pairs were scored but rejected. The lowest
      confirming link here scored <span class="font-mono">${graph.min_edge_probability.toFixed(4)}</span>.
    </p>`;
}

async function loadLinkageEvidence(masterCitizenId) {
  const target = document.getElementById("linkage-evidence");
  if (!target) return;
  try {
    target.innerHTML = linkageEvidencePanel(await api(`/citizen/${masterCitizenId}/graph`));
  } catch (e) {
    target.innerHTML = `<p class="text-on-surface-variant text-xs">${e.message}</p>`;
  }
}

// Compact by default (icon, type+agency, date, status, amount, confidence) -
// the case-file sections above already show full detail for every record
// type, so the timeline's remaining unique value is one cross-category
// chronological view. Full per-record detail (name-on-file, provider,
// every attribute) is available on demand via the native <details> toggle
// (no JS event wiring needed).
function timelineRow(r) {
  const meta = RECORD_TYPE_META[r.record_type] || { icon: "description", label: r.record_type };
  return `
    <details class="group">
      <summary class="p-5 flex items-center justify-between gap-4 cursor-pointer hover:bg-surface-container-high/40">
        <div class="flex items-center gap-4 min-w-0">
          <div class="w-10 h-10 rounded bg-surface-container-high flex items-center justify-center shrink-0">
            <span class="material-symbols-outlined text-primary">${meta.icon}</span>
          </div>
          <div class="min-w-0">
            <p class="font-semibold text-sm">${meta.label} <span class="text-on-surface-variant font-normal">&middot; ${r.agency}</span></p>
            <p class="text-on-surface-variant text-xs truncate">${r.record_date}${r.status ? ` &middot; ${r.status}` : ""}</p>
          </div>
        </div>
        <div class="flex items-center gap-3 shrink-0">
          ${r.amount !== null && r.amount !== undefined ? `<span class="font-semibold text-sm">${GBP.format(r.amount)}</span>` : ""}
          ${confidenceBadge(r.match_probability)}
          <span class="material-symbols-outlined text-on-surface-variant text-base transition-transform group-open:rotate-180">expand_more</span>
        </div>
      </summary>
      <div class="px-5 pb-5">
        <p class="text-on-surface-variant text-xs mb-2">"${r.first_name} ${r.last_name}" &middot; ${r.provider_name ?? "provider unknown"}</p>
        ${attributesList(r.attributes)}
      </div>
    </details>`;
}

function fieldAgreementRow(f) {
  const label = f.field.replace(/_/g, " ");
  if (f.is_consistent) {
    return `
      <div class="flex items-center gap-3 text-sm py-2">
        <span class="material-symbols-outlined text-emerald-400 text-base">check_circle</span>
        <span class="capitalize text-on-surface-variant">${label}</span>
        <span class="ml-auto text-xs text-on-surface-variant">matched exactly</span>
      </div>`;
  }
  return `
    <div class="p-3 rounded-lg bg-amber-500/5 border border-amber-500/20 my-1.5">
      <div class="flex items-center gap-3 text-sm">
        <span class="material-symbols-outlined text-amber-400 text-base">warning</span>
        <span class="capitalize font-medium">${label} varies across linked records</span>
      </div>
      <p class="text-xs text-on-surface-variant mt-1 ml-8">${f.distinct_values.join("  vs.  ")}</p>
    </div>`;
}

// One "case file" card per record type the citizen actually has, grouped
// into labeled sections (CASE_FILE_SECTIONS) so the profile reads as an
// organized file rather than a flat, arbitrarily-ordered grid. Types capped
// at 0-1 records per citizen get a single-fact card; the two genuinely
// repeatable types (Hospital Visit, Prescription) get every instance shown
// together in one card, so e.g. all of a citizen's medications are visible
// at once instead of just the latest one.
function singleInstanceCard(recordType, record) {
  const meta = RECORD_TYPE_META[recordType] || { icon: "description", label: recordType };
  return `
    <div class="entity-card rounded-lg p-5">
      <div class="flex justify-between items-center mb-2">
        <span class="text-[11px] uppercase tracking-wider text-on-surface-variant font-semibold flex items-center gap-1.5">
          <span class="material-symbols-outlined text-primary text-base">${meta.icon}</span>${meta.label}
        </span>
        ${confidenceBadge(record.match_probability)}
      </div>
      ${record.status ? `<p class="text-lg font-bold text-on-surface mb-1">${record.status}</p>` : ""}
      <p class="text-[11px] text-on-surface-variant">${record.agency} &middot; ${record.provider_name ?? "provider unknown"} &middot; ${record.record_date}</p>
      ${attributesList(record.attributes)}
    </div>`;
}

function multiInstanceCard(recordType, records) {
  const meta = RECORD_TYPE_META[recordType] || { icon: "description", label: recordType };
  const sorted = [...records].sort((a, b) => b.record_date.localeCompare(a.record_date));
  return `
    <div class="entity-card rounded-lg p-5">
      <span class="text-[11px] uppercase tracking-wider text-on-surface-variant font-semibold flex items-center gap-1.5 mb-3">
        <span class="material-symbols-outlined text-primary text-base">${meta.icon}</span>${meta.label}${sorted.length > 1 ? ` (${sorted.length})` : ""}
      </span>
      <div class="divide-y divide-outline-variant/20">
        ${sorted
          .map(
            (r) => `
          <div class="py-3 first:pt-0 last:pb-0">
            <div class="flex justify-between items-center gap-3">
              <p class="font-bold text-sm text-on-surface truncate">${r.detail ?? meta.label}</p>
              ${confidenceBadge(r.match_probability)}
            </div>
            <p class="text-[11px] text-on-surface-variant mt-0.5">${r.status ? `${r.status} &middot; ` : ""}${r.agency} &middot; ${r.provider_name ?? "provider unknown"} &middot; ${r.record_date}</p>
            ${attributesList(r.attributes)}
          </div>`
          )
          .join("")}
      </div>
    </div>`;
}

function caseFileCard(recordType, records) {
  return MULTI_INSTANCE_RECORD_TYPES.has(recordType)
    ? multiInstanceCard(recordType, records)
    : singleInstanceCard(recordType, records[0]);
}

function caseFileSection(section, byType) {
  const presentTypes = section.types.filter((rt) => byType[rt] && byType[rt].length);
  if (!presentTypes.length) return "";
  return `
    <div>
      <h3 class="text-xs uppercase tracking-wider text-on-surface-variant font-semibold mb-3 px-1">${section.title}</h3>
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        ${presentTypes.map((rt) => caseFileCard(rt, byType[rt])).join("")}
      </div>
    </div>`;
}

function exportProfileJson(profile) {
  const blob = new Blob([JSON.stringify(profile, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${profile.master_citizen_id}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

async function loadProfile(masterCitizenId) {
  switchView("profile");
  const container = document.getElementById("profile-content");
  container.innerHTML = `<div class="entity-card rounded-lg p-12"><div class="skeleton w-full h-40 rounded"></div></div>`;

  let p;
  try {
    p = await api(`/citizen/${masterCitizenId}/detail`);
  } catch (e) {
    showToast(e.message, true);
    container.innerHTML = `<div class="entity-card rounded-lg p-12 text-center text-on-surface-variant">${e.message}</div>`;
    return;
  }

  const tierClass = ENGAGEMENT_TIER_CLASSES[p.engagement_tier] || ENGAGEMENT_TIER_CLASSES["Moderate Engagement"];
  // Most recent activity first - easier to scan as "what's happened lately".
  const timeline = [...p.records].sort((a, b) => b.record_date.localeCompare(a.record_date));
  const addressLine = p.current_address
    ? `${p.current_address}${p.current_city ? `, ${p.current_city}` : ""}${p.current_postcode ? ` &middot; ${p.current_postcode}` : ""}`
    : "Address unknown";

  // Group every linked record by record_type so the case-file sections below
  // can render one full-detail card per type this citizen actually has -
  // no fixed set of tiles, no blank cards for record types they don't hold.
  const byType = {};
  p.records.forEach((r) => {
    (byType[r.record_type] ??= []).push(r);
  });
  const caseFileSections = CASE_FILE_SECTIONS.map((s) => caseFileSection(s, byType)).join("");

  container.innerHTML = `
    <div class="grid grid-cols-12 gap-6">
      <div class="col-span-12 lg:col-span-8 entity-card rounded-lg p-6 flex items-center gap-6">
        <div class="w-20 h-20 rounded-full bg-primary-dark/20 border-2 border-primary-dark flex items-center justify-center text-2xl font-bold text-primary shrink-0">
          ${initials(p.preferred_name)}
        </div>
        <div class="min-w-0">
          <span class="badge border ${tierClass}">${p.engagement_tier}</span>
          <h2 class="text-2xl font-bold mt-2">${p.preferred_name}</h2>
          <p class="text-on-surface-variant text-sm mt-1">
            Age ${p.age}${p.marital_status ? ` &middot; ${p.marital_status}` : ""}
          </p>
          <p class="text-on-surface-variant text-sm flex items-center gap-1 mt-1">
            <span class="material-symbols-outlined text-base">location_on</span>
            ${addressLine}
          </p>
          <p class="text-on-surface-variant text-xs font-mono mt-2">${p.master_citizen_id}</p>
          <div class="flex flex-wrap gap-1 mt-2">
            ${p.linked_agencies.map((a) => `<span class="text-[10px] px-2 py-0.5 rounded border border-outline-variant/40 text-on-surface-variant">${a}</span>`).join("")}
          </div>
        </div>
      </div>
      <div class="col-span-12 lg:col-span-4 entity-card rounded-lg p-6 flex flex-col justify-between">
        <div class="flex justify-between items-center">
          <span class="text-[11px] uppercase tracking-wider text-on-surface-variant font-semibold">Linkage Confidence</span>
          <span class="material-symbols-outlined text-primary">verified</span>
        </div>
        <div class="flex items-baseline gap-2 mt-2">
          <span class="text-3xl font-bold text-primary">${(p.confidence_score * 100).toFixed(1)}%</span>
        </div>
        <div class="w-full bg-surface-container-high h-1.5 mt-3 rounded-full overflow-hidden">
          <div class="bg-primary h-full" style="width:${(p.confidence_score * 100).toFixed(1)}%"></div>
        </div>
      </div>
    </div>

    <div class="space-y-6">
      ${caseFileSections}
    </div>

    ${lifestyleSummaryPanel(p.lifestyle_summary)}

    ${serviceCoveragePanel(p)}

    <div class="grid grid-cols-12 gap-6">
      <div class="col-span-12 lg:col-span-7 space-y-6">
        <div class="entity-card rounded-lg overflow-hidden">
          <div class="p-6 border-b border-border flex justify-between items-center">
            <div>
              <h3 class="text-base font-semibold">Records Timeline</h3>
              <p class="text-on-surface-variant text-xs">Every agency record resolved into this profile, most recent first</p>
            </div>
            ${confidenceBadge(p.confidence_score)}
          </div>
          <div class="divide-y divide-border/60">
            ${
              timeline.length
                ? timeline.map(timelineRow).join("")
                : `<div class="p-5 text-center text-on-surface-variant text-sm">No linked records on file.</div>`
            }
          </div>
        </div>
      </div>

      <div class="col-span-12 lg:col-span-5 space-y-6">
        <div class="entity-card rounded-lg p-6">
          <h3 class="text-base font-semibold mb-1">Match Explanation</h3>
          <p class="text-on-surface-variant text-xs mb-4">Field-by-field agreement across this profile's ${p.records.length} linked record(s)</p>
          <div>${p.field_agreement.map(fieldAgreementRow).join("")}</div>
        </div>

        <div class="entity-card rounded-lg p-6">
          <h3 class="text-base font-semibold mb-1">Linkage Evidence</h3>
          <p class="text-on-surface-variant text-xs mb-4">How strongly the records behind this profile confirm each other</p>
          <div id="linkage-evidence"><div class="skeleton w-full h-24 rounded"></div></div>
        </div>

        <div class="entity-card rounded-lg p-6">
          <h3 class="text-base font-semibold mb-4">Actions</h3>
          <div class="space-y-1">
            <button id="btn-export-profile" class="w-full flex items-center justify-between p-3 rounded hover:bg-surface-container-high transition-colors text-sm font-medium">
              <span class="flex items-center gap-3"><span class="material-symbols-outlined text-on-surface-variant">download</span> Export Linked Data (JSON)</span>
              <span class="material-symbols-outlined text-xs text-on-surface-variant">chevron_right</span>
            </button>
            <button id="btn-export-profile-pdf" class="w-full flex items-center justify-between p-3 rounded hover:bg-surface-container-high transition-colors text-sm font-medium">
              <span class="flex items-center gap-3"><span class="material-symbols-outlined text-on-surface-variant">picture_as_pdf</span> Export Citizen Summary Report (PDF)</span>
              <span class="material-symbols-outlined text-xs text-on-surface-variant">chevron_right</span>
            </button>
            <button id="btn-copy-id" class="w-full flex items-center justify-between p-3 rounded hover:bg-surface-container-high transition-colors text-sm font-medium">
              <span class="flex items-center gap-3"><span class="material-symbols-outlined text-on-surface-variant">content_copy</span> Copy Citizen ID</span>
              <span class="material-symbols-outlined text-xs text-on-surface-variant">chevron_right</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  `;

  // Supporting analysis, loaded after the profile itself is on screen.
  loadCoverageRecommendations(p.master_citizen_id);
  loadLinkageEvidence(p.master_citizen_id);

  document.getElementById("btn-export-profile").addEventListener("click", () => exportProfileJson(p));
  document.getElementById("btn-export-profile-pdf").addEventListener("click", () =>
    downloadFile(`/citizen/${p.master_citizen_id}/export/pdf`, `${p.master_citizen_id}.pdf`)
  );
  document.getElementById("btn-copy-id").addEventListener("click", () => {
    navigator.clipboard?.writeText(p.master_citizen_id);
    showToast(`Copied ${p.master_citizen_id} to clipboard.`);
  });
}

document.getElementById("btn-back-directory").addEventListener("click", () => switchView(lastListView));

// ---------------------------------------------------------------------------
// Linkage-quality distributions (rendered on the Evaluation page)
// ---------------------------------------------------------------------------
// These two charts describe the linkage without reference to ground truth -
// what an operator would see in production. They sit next to the ground-truth
// scoring so the two halves of "how good is this?" read together.
async function loadQualityCharts() {
  let q;
  try {
    q = await api("/quality");
  } catch (e) {
    return;
  }

  destroyChart("confidence");
  charts.confidence = new Chart(document.getElementById("chart-confidence"), {
    type: "bar",
    data: {
      labels: q.match_probability_histogram.map((b) => b.label),
      datasets: [{ label: "Records", data: q.match_probability_histogram.map((b) => b.count), backgroundColor: CHART_COLORS.primary }],
    },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { grid: { display: false } } } },
  });

  destroyChart("clusterSize");
  charts.clusterSize = new Chart(document.getElementById("chart-clustersize"), {
    type: "bar",
    data: {
      labels: q.cluster_size_distribution.map((b) => b.label),
      datasets: [{ label: "Citizens", data: q.cluster_size_distribution.map((b) => b.count), backgroundColor: CHART_COLORS.accent }],
    },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { grid: { display: false } } } },
  });
}

// ---------------------------------------------------------------------------
// Review Queue view
// ---------------------------------------------------------------------------
// Renders the model's own reasons for a score. `direction` is relative to the
// population median, so "3 agencies, below typical 5" reads as the comparison
// that actually drove the flag.
function anomalyFactors(factors) {
  if (!factors || !factors.length) {
    return `<span class="text-xs text-on-surface-variant">-</span>`;
  }
  return `<div class="flex flex-wrap gap-1">${factors
    .map(
      (f) => `
      <span class="text-[10px] px-2 py-0.5 rounded border border-outline-variant/40 text-on-surface-variant"
            title="Neutralising this feature lowers the anomaly score by ${f.contribution}">
        ${f.label}: <span class="font-medium text-on-surface">${f.value}</span>
        ${f.direction === "at" ? "" : `<span class="text-[9px]">(${f.direction} ${f.population_median})</span>`}
      </span>`
    )
    .join("")}</div>`;
}

// The reviewable half of the explanation. Where the model's reasons say what
// it reacted to, these are the actual disagreeing values in the cluster - the
// thing a caseworker can adjudicate.
function conflictCells(conflicts) {
  if (!conflicts || !conflicts.length) {
    return `<span class="text-xs text-on-surface-variant">No conflicting values</span>`;
  }
  return conflicts
    .map(
      (c) => `
      <div class="text-xs mb-1 last:mb-0">
        <span class="text-on-surface-variant">${c.label}:</span>
        ${c.values.map((v) => `<span class="font-mono text-[11px] px-1.5 py-0.5 rounded bg-red-400/10 text-red-400 ml-1">${v}</span>`).join("")}
      </div>`
    )
    .join("");
}

function anomalyStatusBadge(status) {
  return status === "Anomalous"
    ? `<span class="badge bg-red-500/15 text-red-400">${status}</span>`
    : `<span class="badge bg-emerald-500/15 text-emerald-400">${status}</span>`;
}

const ANOMALY_PAGE_SIZE = 50;
let anomalyLimit = ANOMALY_PAGE_SIZE;

async function loadReviewQueue() {
  lastListView = "review";
  const empty = document.getElementById("anomaly-empty");
  const content = document.getElementById("anomaly-content");
  const grid = document.getElementById("anomaly-kpi-grid");
  grid.innerHTML = Array(4)
    .fill('<div class="entity-card rounded-lg p-5 h-28"><div class="skeleton w-full h-full rounded"></div></div>')
    .join("");

  let a;
  try {
    a = await api(`/anomaly-detection?limit=${anomalyLimit}`);
  } catch (e) {
    grid.innerHTML = "";
    empty.classList.remove("hidden");
    content.classList.add("hidden");
    return;
  }

  if (a.total_profiles_analyzed === 0) {
    grid.innerHTML = "";
    empty.classList.remove("hidden");
    content.classList.add("hidden");
    return;
  }

  empty.classList.add("hidden");
  content.classList.remove("hidden");
  document.getElementById("anomaly-contamination").value = a.contamination;

  grid.innerHTML = [
    kpiCard("Total Profiles Analyzed", INT.format(a.total_profiles_analyzed), "groups", "Resolved citizen profiles scored"),
    kpiCard("Anomalies Detected", INT.format(a.anomalies_detected), "warning", "Flagged by the Isolation Forest"),
    kpiCard("% Anomalous", a.pct_anomalous.toFixed(2) + "%", "percent", "Share of analyzed profiles flagged"),
    kpiCard("Contamination Used", a.contamination.toFixed(2), "tune", "Expected anomalous proportion for this run"),
  ].join("");

  destroyChart("anomalyDistribution");
  charts.anomalyDistribution = new Chart(document.getElementById("chart-anomaly-distribution"), {
    type: "bar",
    data: {
      labels: a.score_distribution.map((b) => b.label),
      datasets: [{ data: a.score_distribution.map((b) => b.count), backgroundColor: CHART_COLORS.primary, borderRadius: 4 }],
    },
    options: {
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#94a3b8" }, grid: { display: false } },
        y: { ticks: { color: "#94a3b8" }, grid: { color: "#1e293b" } },
      },
    },
  });

  destroyChart("anomalySplit");
  charts.anomalySplit = new Chart(document.getElementById("chart-anomaly-split"), {
    type: "doughnut",
    data: {
      labels: ["Normal", "Anomalous"],
      datasets: [
        {
          data: [a.normal_count, a.anomalies_detected],
          backgroundColor: [CHART_COLORS.primary, CHART_COLORS.danger],
          borderColor: "#0f172a",
          borderWidth: 2,
        },
      ],
    },
    options: {
      maintainAspectRatio: false,
      plugins: { legend: { position: "bottom", labels: { color: "#94a3b8", font: { size: 11 } } } },
    },
  });

  document.getElementById("anomaly-caption").textContent =
    `Showing ${INT.format(a.results.length)} of ${INT.format(a.total)} analyzed profiles`;
  const rows = document.getElementById("anomaly-rows");
  rows.innerHTML = a.results
    .map(
      (r) => `
    <tr class="hover:bg-surface-container-high transition-colors">
      <td class="px-5 py-3 font-mono text-xs text-primary">${r.master_citizen_id}</td>
      <td class="px-5 py-3 font-medium">${r.preferred_name}</td>
      <td class="px-5 py-3 text-center">${r.agency_count}</td>
      <td class="px-5 py-3 text-center">${r.record_count}</td>
      <td class="px-5 py-3">${anomalyFactors(r.top_factors)}</td>
      <td class="px-5 py-3">${conflictCells(r.conflicts)}</td>
      <td class="px-5 py-3 text-center font-semibold">${r.anomaly_score.toFixed(1)}</td>
      <td class="px-5 py-3 text-center">
        <button class="view-anomaly-profile text-on-surface-variant hover:text-primary" data-id="${r.master_citizen_id}">
          <span class="material-symbols-outlined text-base">visibility</span>
        </button>
      </td>
    </tr>`
    )
    .join("");

  const showMoreBtn = document.getElementById("btn-anomaly-show-more");
  showMoreBtn.classList.toggle("hidden", a.results.length >= a.total);
}

document.getElementById("anomaly-rows").addEventListener("click", (e) => {
  const btn = e.target.closest(".view-anomaly-profile");
  if (!btn) return;
  loadProfile(btn.dataset.id);
});

document.getElementById("btn-anomaly-show-more").addEventListener("click", async (e) => {
  const btn = e.currentTarget;
  setBusy(btn, true, "Loading...");
  anomalyLimit += ANOMALY_PAGE_SIZE;
  try {
    await loadReviewQueue();
  } finally {
    setBusy(btn, false);
  }
});

async function runAnomalyAnalysis(btn) {
  const contamination = parseFloat(document.getElementById("anomaly-contamination").value);
  setBusy(btn, true, "Running analysis...");
  try {
    const result = await api(`/anomaly-detection/run?contamination=${contamination}`, { method: "POST" });
    showToast(`Analysis complete: ${INT.format(result.anomalies_detected)} of ${INT.format(result.total_profiles_analyzed)} profiles flagged anomalous.`);
    anomalyLimit = ANOMALY_PAGE_SIZE;
    await loadReviewQueue();
  } catch (e) {
    showToast(e.message, true);
  } finally {
    setBusy(btn, false);
  }
}

document.getElementById("btn-run-anomaly").addEventListener("click", (e) => runAnomalyAnalysis(e.currentTarget));

// ---------------------------------------------------------------------------
// Engagement summary cards (embedded in the Citizen Directory view)
// ---------------------------------------------------------------------------
function engagementTierCard(s, { isAggregate = false } = {}) {
  const rangeLabel = isAggregate
    ? "All resolved citizens, all tiers"
    : s.max_agency_count === null
    ? `${s.min_agency_count}+ agencies`
    : s.min_agency_count === s.max_agency_count
    ? `${s.min_agency_count} agenc${s.min_agency_count === 1 ? "y" : "ies"}`
    : `${s.min_agency_count}-${s.max_agency_count} agencies`;
  const tierClass = isAggregate
    ? "text-on-surface bg-outline-variant/20 border-outline-variant/40"
    : ENGAGEMENT_TIER_CLASSES[s.engagement_tier] || ENGAGEMENT_TIER_CLASSES["Moderate Engagement"];
  const clickableAttrs = isAggregate ? "" : `data-tier="${s.engagement_tier}" role="button" tabindex="0"`;
  const clickableClasses = isAggregate ? "" : "cursor-pointer";
  return `
    <div class="entity-card rounded-xl p-6 text-left w-full ${clickableClasses}" ${clickableAttrs}>
      <div class="flex items-center justify-between mb-2">
        <span class="badge border ${tierClass}">${isAggregate ? "ALL CITIZENS" : s.engagement_tier}</span>
        <span class="text-xs text-on-surface-variant">${isAggregate ? "100%" : s.pct_of_population + "%"}</span>
      </div>
      <p class="text-xs text-on-surface-variant/60 mb-3">${rangeLabel}</p>
      <p class="text-2xl font-bold text-on-surface">${INT.format(s.citizen_count)}</p>
      <p class="text-[11px] text-on-surface-variant mb-3">citizens</p>
      <div class="grid grid-cols-2 gap-2 text-xs border-t border-outline-variant/30 pt-3">
        <div><p class="text-on-surface-variant">Avg agencies linked</p><p class="font-semibold">${s.avg_agency_count.toFixed(2)}</p></div>
        <div><p class="text-on-surface-variant">Avg confidence</p><p class="font-semibold">${(s.avg_confidence_score * 100).toFixed(1)}%</p></div>
      </div>
    </div>`;
}

// Whole-population aggregate for the 6th summary card slot - computed
// entirely from the existing /engagement response (no backend change).
// Both averages must be citizen-count-weighted across the 5 tiers, since
// an unweighted mean of the tiers' averages would be wrong given how
// unevenly populated they are.
function computeAggregateEngagement(d) {
  const totalProfiles = d.total_profiles || 0;
  const weightedAgencyCount = d.tiers.reduce((sum, s) => sum + s.avg_agency_count * s.citizen_count, 0);
  const weightedConfidence = d.tiers.reduce((sum, s) => sum + s.avg_confidence_score * s.citizen_count, 0);
  return {
    engagement_tier: "All Citizens",
    citizen_count: totalProfiles,
    avg_agency_count: totalProfiles ? weightedAgencyCount / totalProfiles : 0,
    avg_confidence_score: totalProfiles ? weightedConfidence / totalProfiles : 0,
  };
}

async function loadEngagementSummary() {
  const grid = document.getElementById("directory-engagement-cards");
  grid.innerHTML = Array(6)
    .fill('<div class="entity-card rounded-xl p-6 h-48"><div class="skeleton w-full h-full rounded"></div></div>')
    .join("");

  let d;
  try {
    d = await api("/engagement");
  } catch (e) {
    if (!isNoProfilesError(e)) showToast(e.message, true);
    grid.innerHTML = `<div class="entity-card rounded-xl p-8 text-center text-on-surface-variant text-sm col-span-full">${
      isNoProfilesError(e) ? "Run linkage to see engagement segments." : "Failed to load - see toast for details."
    }</div>`;
    return;
  }

  grid.innerHTML = d.tiers.map((s) => engagementTierCard(s)).join("") + engagementTierCard(computeAggregateEngagement(d), { isAggregate: true });
}

document.getElementById("directory-engagement-cards").addEventListener("click", (e) => {
  const card = e.target.closest("[data-tier]");
  if (!card) return;
  loadTierDrilldown(card.dataset.tier);
});

// role="button" implies keyboard activation - back it with real Enter/Space handling.
document.getElementById("directory-engagement-cards").addEventListener("keydown", (e) => {
  if (e.key !== "Enter" && e.key !== " ") return;
  const card = e.target.closest("[data-tier]");
  if (!card) return;
  e.preventDefault();
  loadTierDrilldown(card.dataset.tier);
});

// ---------------------------------------------------------------------------
// Engagement tier drill-down modal
// ---------------------------------------------------------------------------
const TIER_DRILLDOWN_PAGE_SIZE = 50;
let tierDrilldownLimit = TIER_DRILLDOWN_PAGE_SIZE;
let tierDrilldownTier = null;

function closeTierDrilldown() {
  document.getElementById("tier-drilldown-overlay").classList.add("hidden");
  tierDrilldownTier = null;
}

async function loadTierDrilldown(tier, { resetLimit = true } = {}) {
  tierDrilldownTier = tier;
  if (resetLimit) tierDrilldownLimit = TIER_DRILLDOWN_PAGE_SIZE;

  document.getElementById("tier-drilldown-overlay").classList.remove("hidden");
  document.getElementById("tier-drilldown-title").textContent = tier;

  const empty = document.getElementById("tier-drilldown-empty");
  const table = document.getElementById("tier-drilldown-table");
  const caption = document.getElementById("tier-drilldown-caption");
  const rows = document.getElementById("tier-drilldown-rows");
  const showMoreBtn = document.getElementById("btn-tier-drilldown-show-more");
  empty.textContent = "Loading members...";
  empty.classList.remove("hidden");
  table.classList.add("hidden");
  showMoreBtn.classList.add("hidden");

  let data;
  try {
    data = await api(`/engagement/${encodeURIComponent(tier)}/members?limit=${tierDrilldownLimit}`);
  } catch (e) {
    showToast(e.message, true);
    empty.textContent = "Failed to load - see toast for details.";
    return;
  }

  document.getElementById("tier-drilldown-subtitle").textContent =
    `${INT.format(data.total)} resolved citizen(s) in this tier`;

  if (data.results.length === 0) {
    empty.textContent = "No citizens in this tier.";
    return;
  }

  rows.innerHTML = data.results
    .map(
      (r) => `
        <tr class="hover:bg-surface-container-high transition-colors">
          <td class="px-5 py-3 font-mono text-xs text-primary">${r.master_citizen_id}</td>
          <td class="px-5 py-3 font-medium">${r.preferred_name}</td>
          <td class="px-5 py-3 text-xs text-on-surface-variant">${r.linked_agencies.join(", ")}</td>
          <td class="px-5 py-3 text-center">${confidenceBadge(r.confidence_score)}</td>
          <td class="px-5 py-3 text-center">
            <button class="view-tier-drilldown-profile text-on-surface-variant hover:text-primary" data-id="${r.master_citizen_id}">
              <span class="material-symbols-outlined text-base">visibility</span>
            </button>
          </td>
        </tr>`
    )
    .join("");

  caption.textContent = `Showing ${INT.format(data.results.length)} of ${INT.format(data.total)}`;
  showMoreBtn.classList.toggle("hidden", data.results.length >= data.total);
  empty.classList.add("hidden");
  table.classList.remove("hidden");
}

document.getElementById("tier-drilldown-rows").addEventListener("click", (e) => {
  const btn = e.target.closest(".view-tier-drilldown-profile");
  if (!btn) return;
  closeTierDrilldown();
  lastListView = "directory";
  loadProfile(btn.dataset.id);
});

document.getElementById("btn-tier-drilldown-show-more").addEventListener("click", async (e) => {
  const btn = e.currentTarget;
  setBusy(btn, true, "Loading...");
  tierDrilldownLimit += TIER_DRILLDOWN_PAGE_SIZE;
  try {
    await loadTierDrilldown(tierDrilldownTier, { resetLimit: false });
  } finally {
    setBusy(btn, false);
  }
});

document.getElementById("btn-export-tier-members").addEventListener("click", () => {
  if (!tierDrilldownTier) return;
  downloadFile(
    `/export/engagement-members.csv?tier=${encodeURIComponent(tierDrilldownTier)}`,
    `engagement-${tierDrilldownTier.toLowerCase().replace(/ /g, "-")}.csv`
  );
});

document.getElementById("btn-close-tier-drilldown").addEventListener("click", closeTierDrilldown);

document.getElementById("tier-drilldown-overlay").addEventListener("click", (e) => {
  if (e.target.id === "tier-drilldown-overlay") closeTierDrilldown();
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !document.getElementById("tier-drilldown-overlay").classList.contains("hidden")) {
    closeTierDrilldown();
  }
});

// ---------------------------------------------------------------------------
// Service Coverage Gaps view
// ---------------------------------------------------------------------------
// Summary cards/chart are population-wide (GET /service-coverage, an exact
// count over every resolved profile) and deliberately decoupled from the
// searchable table below, which is powered by paginated /search results -
// searching or paging the table must never change these population totals.
// The candidate agency list is taken from the population summary the backend
// already returns, rather than being restated here - this list used to be
// duplicated in both Python and JavaScript with a comment asking editors to
// keep the two in sync by hand.
//
// This table reports each citizen's gaps as a plain set of facts, with no
// ordering claim. The ranked, peer-prevalence analysis lives on the citizen's
// own profile page, where the peer group and sample size can be shown
// alongside every figure.
let coverageCandidateAgencies = [];

function renderServiceCoverageSummary(s) {
  coverageCandidateAgencies = Object.keys(s.missing_record_counts);
  const grid = document.getElementById("service-coverage-kpi-grid");
  const missingEntries = Object.entries(s.missing_record_counts);
  const [topAgency, topCount] = missingEntries.length
    ? missingEntries.reduce((best, e) => (e[1] > best[1] ? e : best), missingEntries[0])
    : [null, 0];
  const withGap = s.total_profiles - s.fully_covered_count;

  grid.innerHTML = [
    kpiCard("Citizens Profiled", INT.format(s.total_profiles), "groups", "All resolved citizens"),
    kpiCard("Citizens Fully Covered", INT.format(s.fully_covered_count), "task_alt", "Linked to all 8 illustrative agencies checked"),
    kpiCard("Citizens With A Service Gap", INT.format(withGap), "priority_high", "Missing at least one of the 8 illustrative agencies"),
    kpiCard(
      "Most Common Gap",
      topAgency || "None",
      "assignment_late",
      topAgency ? `${INT.format(topCount)} citizens have no on-file record with ${topAgency}` : "No gaps found"
    ),
  ].join("");

  destroyChart("serviceCoverage");
  charts.serviceCoverage = new Chart(document.getElementById("chart-service-coverage"), {
    type: "bar",
    data: {
      labels: missingEntries.map(([agency]) => agency),
      datasets: [{ data: missingEntries.map(([, count]) => count), backgroundColor: CHART_COLORS.primary, borderRadius: 4 }],
    },
    options: {
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#94a3b8" }, grid: { display: false } },
        y: { ticks: { color: "#94a3b8" }, grid: { color: "#1e293b" } },
      },
    },
  });
}

async function loadServiceCoverageSummary() {
  let s;
  try {
    s = await api("/service-coverage");
  } catch (e) {
    if (!isNoProfilesError(e)) showToast(e.message, true);
    document.getElementById("service-coverage-kpi-grid").innerHTML =
      `<div class="entity-card rounded-lg p-8 text-center text-on-surface-variant text-sm col-span-full">${
        isNoProfilesError(e) ? "Run linkage to see service coverage gaps." : "Failed to load - see toast for details."
      }</div>`;
    return;
  }
  renderServiceCoverageSummary(s);
}

// ---------------------------------------------------------------------------
// Pipeline action buttons
// ---------------------------------------------------------------------------
function setBusy(btn, busy, busyLabel) {
  btn.disabled = busy;
  btn.classList.toggle("opacity-50", busy);
  btn.classList.toggle("cursor-not-allowed", busy);
  if (busy) {
    btn.dataset.originalHtml = btn.innerHTML;
    btn.innerHTML = `<span class="material-symbols-outlined text-base animate-spin">progress_activity</span> ${busyLabel}`;
  } else if (btn.dataset.originalHtml) {
    btn.innerHTML = btn.dataset.originalHtml;
  }
}

function activeView() {
  return document.querySelector(".view.active")?.id.replace("view-", "");
}

async function generateData(btn) {
  setBusy(btn, true, "Generating...");
  try {
    const result = await api("/generate-data", { method: "POST" });
    showToast(`Generated ${INT.format(result.people)} citizens / ${INT.format(result.records)} records.`);
    const view = activeView();
    if (view === "dashboard") await loadDashboard();
    if (view === "directory") {
      await runSearch(document.getElementById("search-input").value, { navigate: false });
      await loadEngagementSummary();
    }
  } catch (e) {
    showToast(e.message, true);
  } finally {
    setBusy(btn, false);
  }
}

async function runLinkage(btn) {
  setBusy(btn, true, "Running linkage (~20s)...");
  try {
    const result = await api("/run-linkage", { method: "POST" });
    showToast(`Linkage complete: ${INT.format(result.clusters)} clusters, ${INT.format(result.duplicates_found)} duplicates found.`);
    const view = activeView();
    if (view === "dashboard") {
      await loadDashboard();
      await loadServiceCoverageSummary();
    }
    if (view === "review") await loadReviewQueue();
    if (view === "evaluation") await loadEvaluation();
    if (view === "directory") {
      await runSearch(document.getElementById("search-input").value, { navigate: false });
      await loadEngagementSummary();
    }
  } catch (e) {
    showToast(e.message, true);
  } finally {
    setBusy(btn, false);
  }
}

document.getElementById("btn-generate").addEventListener("click", (e) => generateData(e.currentTarget));
document.getElementById("btn-linkage").addEventListener("click", (e) => runLinkage(e.currentTarget));

// ---------------------------------------------------------------------------
// Model Evaluation view
// ---------------------------------------------------------------------------
// Everything on this page is scored against the synthetic ground truth. It is
// the only part of the frontend that reads endpoints backed by
// `records.person_index`; nothing a caseworker sees depends on it.

const PCT = (v) => `${(v * 100).toFixed(2)}%`;
const F4 = (v) => Number(v).toFixed(4);

// Noise levels in the order the benchmark reports them, easiest first.
const NOISE_LEVEL_LABELS = {
  pristine: "Pristine",
  light: "Light",
  default: "Default (shipped)",
  heavy: "Heavy",
  severe: "Severe",
};

function metricRow(m, { highlight = false } = {}) {
  return `
    <tr class="${highlight ? "bg-primary/5" : ""} hover:bg-surface-container-high/40 transition-colors">
      <td class="px-5 py-3">
        <p class="text-sm font-${highlight ? "semibold text-primary" : "medium"}">${m.label}</p>
        <p class="text-on-surface-variant text-[11px] mt-0.5">${m.note}</p>
      </td>
      <td class="px-5 py-3 text-center text-sm font-mono">${F4(m.pairwise_precision)}</td>
      <td class="px-5 py-3 text-center text-sm font-mono">${F4(m.pairwise_recall)}</td>
      <td class="px-5 py-3 text-center text-sm font-mono font-semibold ${highlight ? "text-primary" : ""}">${F4(m.pairwise_f1)}</td>
      <td class="px-5 py-3 text-center text-sm font-mono">${F4(m.adjusted_rand_index)}</td>
      <td class="px-5 py-3 text-center text-sm">${INT.format(m.exactly_resolved)}</td>
      <td class="px-5 py-3 text-center text-sm ${m.over_split_citizens ? "text-amber-400" : "text-on-surface-variant"}">${INT.format(m.over_split_citizens)}</td>
      <td class="px-5 py-3 text-center text-sm ${m.over_merged_clusters ? "text-red-400" : "text-on-surface-variant"}">${INT.format(m.over_merged_clusters)}</td>
    </tr>`;
}

function sweepNote(icon, title, body, tone = "text-primary") {
  return `
    <div class="flex items-start gap-3">
      <span class="material-symbols-outlined ${tone} text-base mt-0.5">${icon}</span>
      <div>
        <p class="font-medium text-sm">${title}</p>
        <p class="text-on-surface-variant text-xs mt-0.5 leading-relaxed">${body}</p>
      </div>
    </div>`;
}

function renderSweepNotes(sweep) {
  const current = sweep.points.find((p) => p.is_current);
  const best = sweep.points.find((p) => p.threshold === sweep.best_threshold);
  const notes = [];

  if (current && best && best.threshold !== current.threshold) {
    notes.push(
      sweepNote(
        "trending_up",
        `A threshold of ${best.threshold} scores better than the configured ${current.threshold}`,
        `F1 rises from ${F4(current.pairwise_f1)} to ${F4(best.pairwise_f1)}, and over-merged clusters fall from
         ${current.over_merged_clusters} to ${best.over_merged_clusters}.`,
        "text-amber-400"
      )
    );
  } else if (current) {
    notes.push(
      sweepNote(
        "check_circle",
        `The configured threshold of ${current.threshold} is the best on this sweep`,
        `No tested threshold produced a higher F1 than ${F4(current.pairwise_f1)}.`
      )
    );
  }

  // A recall that does not move across the whole sweep is diagnostic: those
  // misses are pairs the blocking rules never proposed, so no threshold can
  // recover them. Worth surfacing, because it points at a different fix.
  const recalls = sweep.points.map((p) => p.pairwise_recall);
  const recallSpread = Math.max(...recalls) - Math.min(...recalls);
  if (recallSpread < 1e-6) {
    notes.push(
      sweepNote(
        "info",
        "Recall is flat across every threshold",
        `Every missed link is a pair the blocking rules never proposed for scoring, so lowering the threshold
         cannot recover it. Improving recall means changing the blocking rules, not the threshold.`,
        "text-on-surface-variant"
      )
    );
  }

  notes.push(
    sweepNote(
      "hub",
      `${INT.format(sweep.edge_count)} scored pairs re-clustered per point`,
      "Each point is a graph re-clustering of the persisted pairwise edges, not a model retrain.",
      "text-on-surface-variant"
    )
  );

  document.getElementById("evaluation-sweep-notes").innerHTML = notes.join("");
}

function renderThresholdChart(sweep) {
  destroyChart("thresholdSweep");
  const ctx = document.getElementById("chart-threshold-sweep");
  charts.thresholdSweep = new Chart(ctx, {
    type: "line",
    data: {
      labels: sweep.points.map((p) => p.threshold),
      datasets: [
        {
          label: "Precision",
          data: sweep.points.map((p) => p.pairwise_precision),
          borderColor: CHART_COLORS.primary,
          backgroundColor: "transparent",
          tension: 0.3,
        },
        {
          label: "Recall",
          data: sweep.points.map((p) => p.pairwise_recall),
          borderColor: CHART_COLORS.danger,
          backgroundColor: "transparent",
          tension: 0.3,
        },
        {
          label: "F1",
          data: sweep.points.map((p) => p.pairwise_f1),
          borderColor: CHART_COLORS.accent,
          backgroundColor: "transparent",
          borderDash: [4, 3],
          tension: 0.3,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { title: { display: true, text: "Clustering threshold" }, grid: { display: false } },
        y: { suggestedMin: 0.99, suggestedMax: 1.0 },
      },
    },
  });
}

function renderBenchmark(benchmark) {
  const hasData = benchmark.levels && benchmark.levels.length > 0;
  document.getElementById("evaluation-benchmark-body").classList.toggle("hidden", !hasData);
  document.getElementById("evaluation-benchmark-empty").classList.toggle("hidden", hasData);
  if (!hasData) return;

  document.getElementById("evaluation-benchmark-rows").innerHTML = benchmark.levels
    .map(
      (l) => `
      <tr class="hover:bg-surface-container-high/40 transition-colors">
        <td class="px-4 py-3 text-sm font-medium">${NOISE_LEVEL_LABELS[l.noise_level] ?? l.noise_level}</td>
        <td class="px-4 py-3 text-center text-sm font-mono">${F4(l.splink_f1)}</td>
        <td class="px-4 py-3 text-center text-sm font-mono text-on-surface-variant">${F4(l.baseline_f1)}</td>
        <td class="px-4 py-3 text-center text-sm font-mono font-semibold text-primary">+${F4(l.f1_advantage)}</td>
      </tr>`
    )
    .join("");

  destroyChart("benchmark");
  charts.benchmark = new Chart(document.getElementById("chart-benchmark"), {
    type: "line",
    data: {
      labels: benchmark.levels.map((l) => NOISE_LEVEL_LABELS[l.noise_level] ?? l.noise_level),
      datasets: [
        {
          label: "Splink F1",
          data: benchmark.levels.map((l) => l.splink_f1),
          borderColor: CHART_COLORS.primary,
          backgroundColor: "transparent",
          tension: 0.3,
        },
        {
          label: "Deterministic rule F1",
          data: benchmark.levels.map((l) => l.baseline_f1),
          borderColor: CHART_COLORS.muted,
          backgroundColor: "transparent",
          borderDash: [4, 3],
          tension: 0.3,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { title: { display: true, text: "Data quality (worsening)" }, grid: { display: false } },
        y: { suggestedMin: 0.6, suggestedMax: 1.0 },
      },
    },
  });
}

function renderDetectors(comparison) {
  document.getElementById("evaluation-detector-caption").textContent =
    `${INT.format(comparison.linkage_errors)} of ${INT.format(comparison.profiles)} clusters (${PCT(comparison.base_rate)}) ` +
    `did not resolve exactly one real citizen. k = top ${INT.format(comparison.k)} profiles.`;

  document.getElementById("evaluation-detector-rows").innerHTML = comparison.detectors
    .map((d) => {
      const isBest = d.detector === comparison.best_detector;
      return `
      <tr class="${isBest ? "bg-primary/5" : ""} hover:bg-surface-container-high/40 transition-colors">
        <td class="px-5 py-3">
          <p class="text-sm font-${isBest ? "semibold text-primary" : "medium"}">${d.label}</p>
          <p class="text-on-surface-variant text-[11px] mt-0.5">${d.note}</p>
        </td>
        <td class="px-5 py-3 text-center text-sm font-mono">${F4(d.average_precision)}</td>
        <td class="px-5 py-3 text-center text-sm font-mono">${F4(d.precision_at_k)}</td>
        <td class="px-5 py-3 text-center text-sm font-mono">${F4(d.recall_at_k)}</td>
        <td class="px-5 py-3 text-center text-sm font-semibold ${d.lift_over_random >= 2 ? "text-primary" : "text-on-surface-variant"}">${d.lift_over_random}x</td>
      </tr>`;
    })
    .join("");
}

async function loadEvaluation() {
  // The threshold sweep and detector comparison each recompute over the whole
  // population and take a couple of seconds. They are independent of one
  // another and of everything else here, so they are fired together rather
  // than chained - sequential awaits made this page take ~5s to fill in.
  // Each panel renders the moment its own request lands.
  document.getElementById("evaluation-sweep-notes").innerHTML = sweepNote(
    "hourglass_top",
    "Re-clustering at each threshold…",
    "Scoring every operating point against ground truth.",
    "text-on-surface-variant"
  );
  document.getElementById("evaluation-detector-caption").textContent =
    "Ranking each detector against the known linkage errors…";

  const headline = api("/evaluation/linkage").then((evaluation) => {
    const splink = evaluation.splink;
    document.getElementById("evaluation-kpi-grid").innerHTML = [
      kpiCard("Pairwise F1", F4(splink.pairwise_f1), "target", `Precision ${F4(splink.pairwise_precision)} / recall ${F4(splink.pairwise_recall)}`),
      kpiCard("Gain over best rule", `+${F4(evaluation.f1_improvement_over_best_baseline)}`, "trending_up", `Best baseline scored ${F4(evaluation.best_baseline_f1)}`),
      kpiCard("Resolved exactly", INT.format(splink.exactly_resolved), "check_circle", `of ${INT.format(splink.true_citizens)} real citizens`),
      kpiCard("Linkage errors", INT.format(splink.over_split_citizens + splink.over_merged_clusters), "error", `${splink.over_split_citizens} over-split / ${splink.over_merged_clusters} over-merged`),
    ].join("");
    document.getElementById("evaluation-baseline-rows").innerHTML =
      metricRow(splink, { highlight: true }) + evaluation.baselines.map((b) => metricRow(b)).join("");
  });

  // One panel being unavailable - for example a database linked before
  // pairwise edges were persisted - must not blank out the rest of the page,
  // so every panel owns its own error state.
  const sweep = api("/evaluation/threshold-sweep")
    .then((data) => {
      renderThresholdChart(data);
      renderSweepNotes(data);
    })
    .catch((e) => {
      document.getElementById("evaluation-sweep-notes").innerHTML = sweepNote(
        "info",
        "Threshold sweep unavailable",
        e.message,
        "text-on-surface-variant"
      );
    });

  const detectors = api("/anomaly-detection/evaluation")
    .then(renderDetectors)
    .catch((e) => {
      document.getElementById("evaluation-detector-caption").textContent = e.message;
    });

  const benchmark = api("/benchmark")
    .then(renderBenchmark)
    .catch(() => renderBenchmark({ levels: [] }));

  await Promise.allSettled([headline, sweep, detectors, benchmark, loadQualityCharts()]);
}

async function runBenchmark(btn) {
  setBusy(btn, true, "Running...");
  try {
    renderBenchmark(await api("/benchmark/run", { method: "POST" }));
    showToast("Benchmark complete.");
  } catch (e) {
    showToast(e.message, true);
  } finally {
    setBusy(btn, false);
  }
}

document.getElementById("btn-run-benchmark").addEventListener("click", (e) => runBenchmark(e.currentTarget));

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
startApp();
