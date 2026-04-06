# Troubleshooting Log

## Issue 1: C drive exhaustion during builds

Symptoms:
- Build interruptions, daemon instability, odd install corruption.

Fix:
- Moved project to `D:\StreamSage`.
- Moved Docker data to `D:\DockerData\LocalDocker`.
- Created junction from old Docker path to D path.

## Issue 2: `binge-service` restart loop with `exec format error`

Symptoms:
- `exec /usr/local/bin/uvicorn: exec format error`
- Empty/corrupted package files found in image (`fastapi/__init__.py` was zero bytes).

Fix:
- Clean rebuild with `--no-cache`.
- Launch command stabilized via `python -m uvicorn`.

## Issue 3: `movie-assistant-service` startup crash with Chroma schema mismatch

Symptoms:
- `sqlite3.OperationalError: no such column: collections.topic`

Root cause:
- Runtime expected legacy column that was absent in existing DB schema.

Fix performed:
- Added `topic` column to `collections` table in `chroma.sqlite3` after backup.

## Issue 4: `movie-assistant-service` long/unhealthy startup due large model init/download

Symptoms:
- Health endpoint not reachable during startup.
- HF download timeout behavior observed.

Fix performed:
- Added HF timeout/env tuning in compose.
- Added persistent model cache volume (`movie_hf_cache`) to avoid repeated cold downloads.

## Issue 5: Frontend port conflict in earlier state

Fix:
- Frontend mapped to `3001:3000` in active runtime.

## Known remaining risk

- The movie assistant startup path is heavy (large model + CPU path) and can still be the slowest service to become healthy.
- This is expected for large embedding model workflows without GPU.
