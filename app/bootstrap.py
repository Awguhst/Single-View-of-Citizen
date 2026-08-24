"""Bring a fresh database up to a fully explorable state.

Every step is idempotent and skipped if its output already exists, so this
is safe to call on every startup - which is exactly what `main.py`'s
lifespan does, keeping the "no manual setup required" promise for local use.

It is also runnable on its own:

    python -m app.bootstrap

which is how the Docker image is built. Doing the work at build time rather
than first boot matters for a container host: generation plus Splink
training takes 40-60 seconds, comfortably longer than a platform health
check will wait, and an ephemeral filesystem would repeat that cost on every
restart. Baking it into the image means the container boots ready to serve.
"""

from __future__ import annotations

import logging

from app import anomaly_service, citizen_service, data_generator, splink_service

logger = logging.getLogger("citizenlink")


def bootstrap() -> None:
    """Generate data, resolve identities, build profiles, and fit the
    review-queue model - skipping any stage already present."""
    if not data_generator.has_generated_data():
        logger.info(
            "No data found - generating synthetic dataset "
            "(10,000 citizens / ~75,000 government records across 11 agencies)..."
        )
        result = data_generator.generate_all()
        logger.info("Generated %s citizens / %s records.", result.people, result.records)
    else:
        logger.info("Existing dataset found - skipping generation.")

    if not splink_service.has_run_linkage():
        logger.info("Running Splink entity-resolution pipeline (this can take ~1 minute)...")
        linkage = splink_service.run_full_pipeline()
        logger.info(
            "Linkage complete: %s clusters, %s duplicates found, avg confidence %.3f (%.1fs).",
            linkage.clusters,
            linkage.duplicates_found,
            linkage.avg_match_probability,
            linkage.training_seconds,
        )
    else:
        logger.info("Existing clusters found - skipping linkage.")

    if not citizen_service.has_citizen_profiles():
        logger.info("Building citizen profiles...")
        n_profiles = citizen_service.build_citizen_profiles()
        logger.info("Built %s citizen profiles.", n_profiles)

    # Fitting the Review Queue's model here too means a freshly deployed
    # instance opens on a populated queue rather than an empty state and a
    # "Run Analysis" button. It is seconds of work on top of a pipeline that
    # already took a minute.
    if not anomaly_service.has_anomaly_results():
        logger.info("Fitting the review-queue model...")
        summary = anomaly_service.run_anomaly_detection()
        logger.info(
            "Review queue ready: %s of %s profiles flagged.",
            summary["anomalies_detected"],
            summary["total_profiles_analyzed"],
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    bootstrap()
