# Deployment Implementation Plan (Step-by-Step)

This is the execution plan for split deployment (Vercel + Render + Hugging Face) after Dockerization.

## Phase 1: Prepare Registry Images

1. Log in to Docker Hub.
2. Build and push all service images.
3. Verify images exist with expected tags.

Commands:

```powershell
cd D:\StreamSage
docker login
auth
# Build + push all local service images to your namespace
.\scripts\deploy\build_and_push_all.ps1 -RegistryUser <your_user> -Tag v1
```

Expected pushed images:
- `<your_user>/streamsage-oracle-rag-service:v1`
- `<your_user>/streamsage-binge-service:v1`
- `<your_user>/streamsage-sentiment-service:v1`
- `<your_user>/streamsage-movie-assistant-service:v1`
- `<your_user>/streamsage-gateway:v1`
- `<your_user>/streamsage-frontend:v1`

## Phase 2: Pull-Based Reproducible Run (Any Machine)

1. Copy `.env.deploy.example` to `.env.deploy`.
2. Fill `REGISTRY_USER`, `IMAGE_TAG`, and `HF_API_TOKEN`.
3. Pull and start from registry images.

Commands:

```powershell
cd D:\StreamSage
Copy-Item .env.deploy.example .env.deploy
# edit .env.deploy
.\scripts\deploy\pull_and_run_from_registry.ps1 -EnvFile .env.deploy
```

This proves that your stack can be reproduced without local builds.

## Phase 3: Platform Split Rollout

### 3.1 Frontend on Vercel
1. Import repo into Vercel.
2. Set root directory to `frontend`.
3. Set env var `VITE_API_URL=<render_gateway_url>/api/v1`.
4. Deploy and verify page load + API calls.

### 3.2 Backend APIs on Render
Deploy as separate services from Docker images:
- gateway
- binge-service
- sentiment-service
- oracle-rag-service (if resource budget allows)

Render setup per service:
1. Create Web Service.
2. Source: Docker image from Docker Hub.
3. Set port and health endpoint.
4. Set required env vars.
5. Deploy.

### 3.3 Model-heavy path on Hugging Face
Option A (recommended):
- Host assistant inference on HF endpoint and call from gateway.

Option B:
- Containerize movie-assistant on HF Space/other container host with persistent cache.

## Phase 4: URL and Secret Wiring

Update gateway env to hosted URLs:
- `ORACLE_SERVICE_URL`
- `BINGE_SERVICE_URL`
- `SENTIMENT_SERVICE_URL`
- `MOVIE_ASSISTANT_SERVICE_URL`

Never hardcode secrets in files; use platform secret managers.

## Phase 5: Verification Checklist

1. Health endpoints for each backend return healthy.
2. Gateway aggregate health shows all dependencies healthy.
3. Frontend can complete end-to-end recommendation flow.
4. Logs show no crash/restart loop.

Commands:

```powershell
cd D:\StreamSage
.\scripts\deploy\verify_stack_health.ps1 -AllowDegraded -ShowComposePs
.\scripts\deploy\run_smoke_tests.ps1
```

## Rollback Strategy

If release `v1` fails:
1. Change IMAGE_TAG to previous stable tag in env file.
2. Re-run pull and up script.
3. Confirm health.
