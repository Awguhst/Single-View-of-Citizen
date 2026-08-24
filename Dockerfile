# CitizenLink - container image for Railway (or any container host).
#
# The one non-obvious decision here is that the synthetic dataset is built
# during `docker build` rather than on first boot.
#
# On startup the app normally generates 10,000 citizens, trains a Splink
# model and builds the profiles - around 40-60 seconds. A platform health
# check would time out long before the first request could be served, and
# because container filesystems are ephemeral that cost would be paid again
# on every redeploy and every restart. Baking it into the image instead means
# the container boots in about a second with data already present, and every
# replica serves byte-identical data (generation is seeded).
#
# The trade-off is a slower build and a slightly larger image (the DuckDB
# file is ~20MB). Both are the right way round for a demo that is deployed
# far less often than it is started.

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, so application edits do not invalidate the pip layer.
COPY app/requirements.txt ./app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt

COPY app ./app

# The database lives outside /app/data so a mounted volume (if one is ever
# attached) does not shadow the baked-in file.
ENV CITIZENLINK_DB_PATH=/data/svoc.duckdb

# Build the dataset, run linkage, build profiles, and fit the review-queue
# model - everything the app would otherwise do on first boot.
RUN python -m app.bootstrap

EXPOSE 8000

# Railway injects $PORT; default to 8000 for a plain `docker run`.
#
# `sh -c` is explicit rather than relying on Docker's shell form, and the
# start command deliberately lives ONLY here - not in railway.json. Railway's
# `startCommand` overrides this CMD and is passed to the container without
# shell expansion, so a `--port $PORT` written there arrives at uvicorn as the
# four literal characters "$PORT" and the container crash-loops with
# "Invalid value for '--port'". Keeping one shell-expanded definition in one
# place is what stops that recurring.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
