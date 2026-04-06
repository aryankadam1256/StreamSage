# Commands Run And Why

This is the practical command history pattern used in this implementation.

## A) Docker sanity checks

```powershell
docker --version
docker compose version
docker info
```

Why:
- Validate CLI + daemon health before any build.

## B) Compose validation

```powershell
docker compose config -q
```

Why:
- Catch syntax/config issues early.

## C) Pull prebuilt image first

```powershell
docker compose pull ollama
```

Why:
- Ollama uses prebuilt image; this de-risks startup sequence.

## D) Build services one-by-one

```powershell
docker compose build oracle-rag-service
docker compose build binge-service
docker compose build sentiment-service
docker compose build movie-assistant-service
docker compose build gateway
docker compose build frontend
```

Why:
- Isolates failures and makes debugging deterministic.

## E) Start core and verify

```powershell
docker compose up -d ollama
docker compose up -d oracle-rag-service binge-service sentiment-service movie-assistant-service gateway frontend
docker compose ps -a
```

Why:
- Bring up stack in dependency order and inspect runtime state.

## F) Health checks

```powershell
Invoke-WebRequest http://localhost:8001/health
Invoke-WebRequest http://localhost:8002/health
Invoke-WebRequest http://localhost:8003/health
Invoke-WebRequest http://localhost:8004/health
Invoke-WebRequest http://localhost:8000/health
```

Why:
- Running container != healthy app. Endpoint checks verify app readiness.

## G) Logs for diagnosis

```powershell
docker compose logs --tail=120 binge-service
docker compose logs --tail=200 movie-assistant-service
docker compose logs --tail=80 gateway
```

Why:
- Find startup crashes, dependency mismatches, and external timeouts.

## H) Build cache/corruption recovery

```powershell
docker compose build --no-cache binge-service
docker compose up -d --force-recreate binge-service
```

Why:
- Earlier low-space event created corrupted package files in image layer cache.
- Clean rebuild fixed corrupted FastAPI/Uvicorn install artifacts.

## I) Project migration to D

```powershell
robocopy C:\Users\mailp\StreamSage D:\StreamSage /E /MOVE /COPY:DAT /R:1 /W:1 /MT:16
```

Why:
- Free C space and continue work from D.

## J) Docker data migration to D

```powershell
# stop Docker related processes and WSL
# move local Docker data
robocopy C:\Users\mailp\AppData\Local\Docker D:\DockerData\LocalDocker /E /MOVE /COPY:DAT /R:1 /W:1 /MT:16
# create compatibility junction
mklink /J C:\Users\mailp\AppData\Local\Docker D:\DockerData\LocalDocker
```

Why:
- Docker layer/image/cache storage was consuming major C drive space.

## K) Service-specific patching done

- `binge-service` command path stabilized via `python -m uvicorn` launcher.
- `movie-assistant-service` runtime env tuned for HF timeout resilience:
  - `HF_HUB_DISABLE_XET=1`
  - `HF_HUB_DOWNLOAD_TIMEOUT=600`
  - `HF_HUB_ETAG_TIMEOUT=600`
- Added named volume for model cache persistence:
  - `movie_hf_cache:/root/.cache/huggingface`
