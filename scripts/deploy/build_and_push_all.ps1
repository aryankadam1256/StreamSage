param(
    [Parameter(Mandatory = $true)]
    [string]$RegistryUser,

    [Parameter(Mandatory = $false)]
    [string]$Tag = "latest"
)

$ErrorActionPreference = "Stop"

$services = @(
    @{ Name = "oracle-rag-service"; LocalImage = "streamsage-oracle-rag-service"; RemoteImage = "$RegistryUser/streamsage-oracle-rag-service:$Tag" },
    @{ Name = "binge-service"; LocalImage = "streamsage-binge-service"; RemoteImage = "$RegistryUser/streamsage-binge-service:$Tag" },
    @{ Name = "sentiment-service"; LocalImage = "streamsage-sentiment-service"; RemoteImage = "$RegistryUser/streamsage-sentiment-service:$Tag" },
    @{ Name = "movie-assistant-service"; LocalImage = "streamsage-movie-assistant-service"; RemoteImage = "$RegistryUser/streamsage-movie-assistant-service:$Tag" },
    @{ Name = "gateway"; LocalImage = "streamsage-gateway"; RemoteImage = "$RegistryUser/streamsage-gateway:$Tag" },
    @{ Name = "frontend"; LocalImage = "streamsage-frontend"; RemoteImage = "$RegistryUser/streamsage-frontend:$Tag" }
)

Write-Host "Building services..." -ForegroundColor Cyan
foreach ($svc in $services) {
    Write-Host "-> Building $($svc.Name)" -ForegroundColor Yellow
    docker compose build $svc.Name
}

Write-Host "Tagging and pushing images..." -ForegroundColor Cyan
foreach ($svc in $services) {
    Write-Host "-> Tag $($svc.LocalImage):latest as $($svc.RemoteImage)" -ForegroundColor Yellow
    docker tag "$($svc.LocalImage):latest" "$($svc.RemoteImage)"

    Write-Host "-> Push $($svc.RemoteImage)" -ForegroundColor Yellow
    docker push "$($svc.RemoteImage)"
}

Write-Host "Done. All service images are pushed." -ForegroundColor Green
