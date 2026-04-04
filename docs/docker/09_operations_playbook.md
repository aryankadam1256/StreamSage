# Docker Operations Playbook (Daily Use)

This playbook is the practical runbook for using StreamSage in Docker mode every day.

## 1) One command wrapper

Use this script for all routine actions:

```powershell
cd D:\StreamSage
.\scripts\deploy\docker_stack.ps1 -Command <up|down|restart|status|logs|health|smoke|rebuild|pull-run>
```

## 2) Most common daily flows

### Start local Docker stack

```powershell
.\scripts\deploy\docker_stack.ps1 -Command up
.\scripts\deploy\docker_stack.ps1 -Command health -AllowDegraded
```

### Check status and logs

```powershell
.\scripts\deploy\docker_stack.ps1 -Command status
.\scripts\deploy\docker_stack.ps1 -Command logs -Service gateway
```

### Run end-to-end smoke tests

```powershell
.\scripts\deploy\docker_stack.ps1 -Command smoke
```

### Stop stack

```powershell
.\scripts\deploy\docker_stack.ps1 -Command down
```

## 3) Rebuild one service after code changes

```powershell
.\scripts\deploy\docker_stack.ps1 -Command rebuild -Service gateway
```

This rebuilds only the specified service image and restarts only that container.

## 4) Registry pull mode (reproducible machine-to-machine)

1. Copy env template and fill values.
2. Pull images from registry.
3. Start stack from pulled images.

```powershell
Copy-Item .env.deploy.example .env.deploy
# edit .env.deploy (REGISTRY_USER, IMAGE_TAG, HF_API_TOKEN)
.\scripts\deploy\docker_stack.ps1 -Command pull-run -Registry -EnvFile .env.deploy
```

## 5) Health and smoke script details

### Health check script

```powershell
.\scripts\deploy\verify_stack_health.ps1 -AllowDegraded -ShowComposePs
```

Checks:
- Gateway health
- Oracle health
- Binge health
- Sentiment health
- Movie Assistant health
- Optional container table

### Smoke test script

```powershell
.\scripts\deploy\run_smoke_tests.ps1
```

Tests:
- Gateway /health
- Movie discovery POST flow
- Oracle sync ask flow
- Oracle SSE stream flow

## 6) Fast triage if something fails

1. Check stack status:

```powershell
.\scripts\deploy\docker_stack.ps1 -Command status
```

2. Check gateway and failing service logs:

```powershell
.\scripts\deploy\docker_stack.ps1 -Command logs -Service gateway
.\scripts\deploy\docker_stack.ps1 -Command logs -Service oracle-rag-service
```

3. Re-run health + smoke:

```powershell
.\scripts\deploy\docker_stack.ps1 -Command health -AllowDegraded
.\scripts\deploy\docker_stack.ps1 -Command smoke
```

4. If still failing, rebuild only the affected service:

```powershell
.\scripts\deploy\docker_stack.ps1 -Command rebuild -Service <service-name>
```

## 7) Data persistence expectations

- Containers are replaceable runtime instances.
- Images are immutable build artifacts.
- Persistent data survives container recreation through mounts/volumes:
  - Oracle subtitle/chromadb bind mounts
  - Ollama named volume
  - HuggingFace cache named volume

So the stack should behave consistently across restarts unless volumes are deleted, image tags change, or code/config is modified.
