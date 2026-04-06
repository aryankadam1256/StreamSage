param(
    [Parameter(Mandatory = $false)]
    [string]$EnvFile = ".env.deploy"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $EnvFile)) {
    Write-Error "Env file '$EnvFile' not found. Copy .env.deploy.example to .env.deploy and fill values first."
}

Write-Host "Pulling images from registry..." -ForegroundColor Cyan
docker compose -f docker-compose.yml -f docker-compose.registry.yml --env-file $EnvFile pull

Write-Host "Starting services using pulled images..." -ForegroundColor Cyan
docker compose -f docker-compose.yml -f docker-compose.registry.yml --env-file $EnvFile up -d

Write-Host "Current status:" -ForegroundColor Cyan
docker compose -f docker-compose.yml -f docker-compose.registry.yml --env-file $EnvFile ps
