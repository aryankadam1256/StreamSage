param(
    [Parameter(Mandatory = $false)]
    [ValidateSet("up", "down", "restart", "status", "logs", "health", "smoke", "rebuild", "pull-run")]
    [string]$Command = "status",

    [Parameter(Mandatory = $false)]
    [string]$Service = "",

    [Parameter(Mandatory = $false)]
    [string]$EnvFile = ".env.deploy",

    [Parameter(Mandatory = $false)]
    [switch]$Registry,

    [Parameter(Mandatory = $false)]
    [switch]$AllowDegraded
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $repoRoot

function Compose-Args {
    if ($Registry) {
        return @("-f", "docker-compose.yml", "-f", "docker-compose.registry.yml", "--env-file", $EnvFile)
    }
    return @()
}

function Run-Compose {
    param(
        [string[]]$SubArgs
    )

    $baseArgs = Compose-Args
    & docker compose @baseArgs @SubArgs
}

switch ($Command) {
    "up" {
        Write-Host "Starting StreamSage docker stack..." -ForegroundColor Cyan
        Run-Compose -SubArgs @("up", "-d")
        break
    }

    "down" {
        Write-Host "Stopping StreamSage docker stack..." -ForegroundColor Cyan
        Run-Compose -SubArgs @("down")
        break
    }

    "restart" {
        Write-Host "Restarting StreamSage docker stack..." -ForegroundColor Cyan
        Run-Compose -SubArgs @("down")
        Run-Compose -SubArgs @("up", "-d")
        break
    }

    "status" {
        Run-Compose -SubArgs @("ps")
        break
    }

    "logs" {
        if ([string]::IsNullOrWhiteSpace($Service)) {
            Write-Host "Showing tail logs for all services..." -ForegroundColor Cyan
            Run-Compose -SubArgs @("logs", "--tail=120")
        }
        else {
            Write-Host "Showing tail logs for service '$Service'..." -ForegroundColor Cyan
            Run-Compose -SubArgs @("logs", "--tail=160", $Service)
        }
        break
    }

    "health" {
        $verifyScript = Join-Path $PSScriptRoot "verify_stack_health.ps1"
        & $verifyScript -AllowDegraded:$AllowDegraded -ShowComposePs
        break
    }

    "smoke" {
        $smokeScript = Join-Path $PSScriptRoot "run_smoke_tests.ps1"
        & $smokeScript
        break
    }

    "rebuild" {
        if ([string]::IsNullOrWhiteSpace($Service)) {
            throw "For rebuild, pass -Service <service-name>. Example: -Service gateway"
        }

        Write-Host "Rebuilding service '$Service' and restarting it..." -ForegroundColor Cyan
        Run-Compose -SubArgs @("build", $Service)
        Run-Compose -SubArgs @("up", "-d", $Service)
        break
    }

    "pull-run" {
        $pullScript = Join-Path $PSScriptRoot "pull_and_run_from_registry.ps1"
        & $pullScript -EnvFile $EnvFile
        break
    }
}
