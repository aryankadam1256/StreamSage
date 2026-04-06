# How Image Pull Works For This Project

This project is microservices, so you pull multiple images, not one giant image.

## Push flow

1. Build each service image.
2. Tag each image to your Docker Hub namespace.
3. Push each image.

Implemented helper:

```powershell
.\scripts\deploy\build_and_push_all.ps1 -RegistryUser <your_user> -Tag v1
```

## Pull flow

1. Set `.env.deploy` values (`REGISTRY_USER`, `IMAGE_TAG`, `HF_API_TOKEN`).
2. Pull all service images through compose override.
3. Start all services.

Implemented helper:

```powershell
.\scripts\deploy\pull_and_run_from_registry.ps1 -EnvFile .env.deploy
```

## Important answer to your question

- `docker pull <image>` pulls one image.
- `docker compose pull` pulls all images listed in compose.

So for your whole project, use compose pull with `docker-compose.registry.yml`.
