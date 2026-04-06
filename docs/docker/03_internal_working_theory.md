# Internal Working Theory (Build, Run, Network, Storage)

## 1) Build phase internals

When `docker compose build` runs:

1. Docker reads each service `build.context`.
2. That context is tar-streamed to the Docker daemon.
3. Dockerfile instructions execute top-to-bottom.
4. Each instruction creates a cached layer.
5. Final image is a stack of immutable layers + metadata.

Key implication:
- If dependencies are installed before source code copy, rebuilds become much faster due cache reuse.

## 2) Run phase internals

When `docker compose up` runs:

1. Compose creates/uses project network (bridge).
2. Containers are created from images.
3. Env vars, mounts, and port mappings are attached.
4. Container main process (CMD) starts as PID 1.
5. Healthcheck command runs periodically inside container namespace.

## 3) Networking model

- Host-to-container uses `ports` mapping.
- Container-to-container uses service DNS name over bridge network.

Examples:
- Browser -> `localhost:8000` -> gateway container.
- Gateway -> `http://oracle-rag-service:8000` over internal Docker network.

## 4) Storage model

Two major patterns used:

1. Bind mounts:
- Map host project data into container (`./services/...:/app/...`).
- Good for local development visibility and persistence in source tree.

2. Named volumes:
- Managed by Docker (inside Docker storage root).
- Good for large runtime caches and databases.
- Added for movie assistant model cache persistence.

## 5) Why Docker migration to D worked

- Docker daemon still resolves original path (`C:\...\Local\Docker`) through a junction.
- Physical bytes now live on D (`D:\DockerData\LocalDocker`).
- This preserves compatibility while reclaiming C space.

## 6) Why one service can still be "unhealthy" while container is "up"

- Healthcheck status and process liveness are different.
- Container can be running but app startup can still be blocked in model init/download.
- Gateway health can be "degraded" even when its own process is healthy if dependency checks fail.
