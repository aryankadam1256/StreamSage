# Current Docker Status

## Environment migration

- Project moved to `D:\StreamSage`.
- Docker local data moved from `C:\Users\mailp\AppData\Local\Docker` to `D:\DockerData\LocalDocker`.
- Junction created:
  - `C:\Users\mailp\AppData\Local\Docker` -> `D:\DockerData\LocalDocker`

## Current service status summary

Healthy/running:
- `oracle-rag-service`
- `binge-service`
- `sentiment-service`
- `gateway`
- `frontend`
- `ollama`

Still pending stabilization:
- `movie-assistant-service`

Reason:
- Startup is blocked/delayed by heavy embedding model initialization (`BAAI/bge-large-en-v1.5`, 1024-dim model) and large model artifact fetch/load path.

## Why this one is special

- This service must match existing Chroma vector dimension (`1024`), so replacing with small 384-dim model would break retrieval quality/compatibility.
- It has the heaviest startup path among all services.

## Estimated remaining time

If network/model load succeeds cleanly:
- ~10 to 25 minutes.

If remote model host keeps timing out intermittently:
- could extend further.
