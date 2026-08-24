"""Server-rendered exports for the CitizenLink (Single View of Citizen) platform.

Pure rendering layer - every function here takes data already produced by
`citizen_service` and turns it into a downloadable file. No new DuckDB
queries beyond calling the existing `citizen_service`/`search_person` lookups
with a much larger limit than their on-screen-widget defaults, so an export
never silently truncates the dataset it claims to cover.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timezone

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app import citizen_service

# "Export everything" caps - large enough to cover this demo's ~10k profiles
# without a request needing to specify it explicitly.
DIRECTORY_EXPORT_LIMIT = 100_000
DEFAULT_REVIEW_QUEUE_EXPORT_LIMIT = 500

_DIRECTORY_HEADER = [
    "master_citizen_id",
    "name",
    "age",
    "marital_status",
    "engagement_tier",
    "linked_agencies",
    "agency_count",
    "record_count",
    "confidence_score",
    "tax_status",
    "employment_status",
    "driving_licence_status",
    "passport_status",
    "housing_assistance",
    "veteran_status",
    "healthcare_registrations",
    "hospital_visits",
]


def _directory_row(row: dict) -> list:
    return [
        row["master_citizen_id"],
        row["preferred_name"],
        row["age"],
        row["marital_status"],
        row["engagement_tier"],
        "|".join(row["linked_agencies"]),
        row["agency_count"],
        row["record_count"],
        round(float(row["confidence_score"]), 6),
        row["tax_status"],
        row["employment_status"],
        row["driving_licence_status"],
        row["passport_status"],
        row["housing_assistance"],
        row["veteran_status"],
        row["healthcare_registrations"],
        row["hospital_visits"],
    ]


def build_directory_csv(q: str = "") -> str:
    """CSV of every resolved profile matching `q` (or all profiles, if
    `q` is blank) - mirrors what `/search` shows, without its `limit=50`
    on-screen default."""
    rows, _total = citizen_service.search_person(q, limit=DIRECTORY_EXPORT_LIMIT)

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(_DIRECTORY_HEADER)
    for row in rows:
        writer.writerow(_directory_row(row))
    return buffer.getvalue()


def build_engagement_members_csv(tier: str) -> str:
    """CSV of every resolved profile assigned to one engagement tier -
    mirrors what the Engagement Segments drill-down table shows, without
    its on-screen limit."""
    rows, _total = citizen_service.get_engagement_tier_members(tier, limit=DIRECTORY_EXPORT_LIMIT)

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(_DIRECTORY_HEADER)
    for row in rows:
        writer.writerow(_directory_row(row))
    return buffer.getvalue()


def build_review_queue_csv(limit: int = DEFAULT_REVIEW_QUEUE_EXPORT_LIMIT) -> str:
    """CSV of the lowest-confidence multi-record clusters, worst first -
    the full manual-review backlog rather than the dashboard's top-10."""
    metrics = citizen_service.get_quality_metrics(review_queue_size=limit)

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["master_citizen_id", "name", "match_probability", "record_count", "linked_agencies"])
    for item in metrics["review_queue"]:
        writer.writerow(
            [
                item["master_citizen_id"],
                item["name"],
                item["match_probability"],
                item["record_count"],
                "|".join(item["linked_agencies"]),
            ]
        )
    return buffer.getvalue()


_PDF_STYLES = getSampleStyleSheet()
_SMALL_STYLE = ParagraphStyle("Small", parent=_PDF_STYLES["Normal"], fontSize=7, leading=9)
_TABLE_STYLE = TableStyle(
    [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c2e30")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
    ]
)
_RECORD_TABLE_STYLE = TableStyle(_TABLE_STYLE.getCommands() + [("VALIGN", (0, 0), (-1, -1), "TOP")])


def _format_attributes(attrs: dict) -> str:
    return "; ".join(f"{k.replace('_', ' ').title()}: {v}" for k, v in (attrs or {}).items())


def _status_row(profile: dict) -> tuple[list[str], list[str]]:
    labels = [
        "Tax Status",
        "Employment Status",
        "Healthcare Registrations",
        "Driving Licence",
        "Passport",
        "Housing Assistance",
        "Veteran Status",
    ]
    values = [
        profile["tax_status"] or "-",
        profile["employment_status"] or "-",
        str(profile["healthcare_registrations"]),
        profile["driving_licence_status"] or "-",
        profile["passport_status"] or "-",
        profile["housing_assistance"] or "-",
        profile["veteran_status"] or "-",
    ]
    return labels, values


def build_profile_pdf(profile: dict) -> bytes:
    """Render one resolved profile's full Citizen Summary Report as a PDF -
    same dict shape `citizen_service.get_citizen_profile_detail()` returns
    (and that backs the Profile page / `/citizen/{id}/detail` JSON
    response)."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=20 * mm, bottomMargin=20 * mm)
    story = []

    story.append(Paragraph("CitizenLink - Citizen Summary Report", _PDF_STYLES["Title"]))
    story.append(
        Paragraph(
            f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            _PDF_STYLES["Normal"],
        )
    )
    story.append(Spacer(1, 8 * mm))

    story.append(Paragraph(f"{profile['preferred_name']} ({profile['master_citizen_id']})", _PDF_STYLES["Heading2"]))
    story.append(
        Paragraph(
            f"Age: {profile['age']} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"Marital status: {profile['marital_status'] or 'Not on file'}",
            _PDF_STYLES["Normal"],
        )
    )
    story.append(
        Paragraph(
            f"Engagement tier: {profile['engagement_tier']} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"Linked agencies: {profile['agency_count']} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"Linkage confidence: {profile['confidence_score']:.1%}",
            _PDF_STYLES["Normal"],
        )
    )
    story.append(Spacer(1, 6 * mm))

    labels, values = _status_row(profile)
    status_table = Table([labels, values], hAlign="LEFT")
    status_table.setStyle(_TABLE_STYLE)
    story.append(status_table)
    story.append(Spacer(1, 8 * mm))

    story.append(Paragraph("Linked agency records", _PDF_STYLES["Heading3"]))
    record_rows = [["Agency", "Record Type", "Date", "Status", "Confidence", "Details"]]
    for r in profile["records"]:
        record_rows.append(
            [
                r["agency"],
                r["record_type"].replace("_", " ").title(),
                r["record_date"],
                r["status"] or "-",
                f"{r['match_probability']:.1%}",
                Paragraph(_format_attributes(r.get("attributes", {})) or "-", _SMALL_STYLE),
            ]
        )
    record_table = Table(record_rows, hAlign="LEFT", colWidths=[26 * mm, 30 * mm, 20 * mm, 22 * mm, 18 * mm, 54 * mm])
    record_table.setStyle(_RECORD_TABLE_STYLE)
    story.append(record_table)
    story.append(Spacer(1, 8 * mm))

    story.append(Paragraph("Lifestyle &amp; engagement summary", _PDF_STYLES["Heading3"]))
    if profile["lifestyle_summary"]:
        summary_rows = [["Tag", "Basis"]]
        for item in profile["lifestyle_summary"]:
            summary_rows.append([item["tag"], item["basis"]])
        summary_table = Table(summary_rows, hAlign="LEFT", colWidths=[50 * mm, 120 * mm])
        summary_table.setStyle(_TABLE_STYLE)
        story.append(summary_table)
    else:
        story.append(Paragraph("No summary tags met the reporting thresholds for this citizen.", _PDF_STYLES["Normal"]))
    story.append(Spacer(1, 8 * mm))

    story.append(Paragraph("Field-level match explanation", _PDF_STYLES["Heading3"]))
    agreement_rows = [["Field", "Consistent across records?", "Distinct values seen"]]
    for fa in profile["field_agreement"]:
        agreement_rows.append(
            [fa["field"], "Yes" if fa["is_consistent"] else "No", ", ".join(fa["distinct_values"])]
        )
    agreement_table = Table(agreement_rows, hAlign="LEFT", colWidths=[35 * mm, 45 * mm, 90 * mm])
    agreement_table.setStyle(_TABLE_STYLE)
    story.append(agreement_table)

    doc.build(story)
    return buffer.getvalue()
