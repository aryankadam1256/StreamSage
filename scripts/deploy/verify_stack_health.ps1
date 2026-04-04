param(
    [Parameter(Mandatory = $false)]
    [string]$GatewayBaseUrl = "http://localhost:8000",

    [Parameter(Mandatory = $false)]
    [int]$TimeoutSec = 8,

    [Parameter(Mandatory = $false)]
    [switch]$AllowDegraded,

    [Parameter(Mandatory = $false)]
    [switch]$ShowComposePs
)

$ErrorActionPreference = "Stop"

function Test-JsonEndpoint {
    param(
        [string]$Name,
        [string]$Url
    )

    try {
        $payload = Invoke-RestMethod -Uri $Url -Method Get -TimeoutSec $TimeoutSec
        return @{
            Name = $Name
            Url = $Url
            Ok = $true
            Payload = $payload
            Error = $null
        }
    }
    catch {
        return @{
            Name = $Name
            Url = $Url
            Ok = $false
            Payload = $null
            Error = $_.Exception.Message
        }
    }
}

$checks = @(
    @{ Name = "Gateway"; Url = "$GatewayBaseUrl/health" },
    @{ Name = "Oracle"; Url = "http://localhost:8001/health" },
    @{ Name = "Binge"; Url = "http://localhost:8002/health" },
    @{ Name = "Sentiment"; Url = "http://localhost:8003/health" },
    @{ Name = "Movie Assistant"; Url = "http://localhost:8004/health" }
)

$failed = 0
$warnings = 0

Write-Host "Checking StreamSage service health..." -ForegroundColor Cyan

foreach ($c in $checks) {
    $result = Test-JsonEndpoint -Name $c.Name -Url $c.Url

    if (-not $result.Ok) {
        $failed++
        Write-Host "FAIL  $($result.Name): $($result.Error)" -ForegroundColor Red
        continue
    }

    if ($result.Name -eq "Gateway") {
        $overall = $result.Payload.overall
        if ($overall -eq "healthy") {
            Write-Host "PASS  Gateway overall=healthy" -ForegroundColor Green
        }
        elseif ($overall -eq "degraded") {
            if ($AllowDegraded) {
                $warnings++
                Write-Host "WARN  Gateway overall=degraded (allowed)" -ForegroundColor Yellow
            }
            else {
                $failed++
                Write-Host "FAIL  Gateway overall=degraded (use -AllowDegraded to allow)" -ForegroundColor Red
            }
        }
        else {
            $failed++
            Write-Host "FAIL  Gateway overall=$overall" -ForegroundColor Red
        }

        $svcMap = $result.Payload.services
        if ($svcMap) {
            foreach ($prop in $svcMap.PSObject.Properties) {
                $svcName = $prop.Name
                $svcStatus = $prop.Value.status

                if ($svcStatus -eq "healthy") {
                    Write-Host "PASS  Gateway dependency '$svcName' is healthy" -ForegroundColor Green
                }
                else {
                    $msg = "WARN  Gateway dependency '$svcName' is $svcStatus"
                    if ($AllowDegraded) {
                        $warnings++
                        Write-Host $msg -ForegroundColor Yellow
                    }
                    else {
                        $failed++
                        Write-Host ($msg -replace "WARN", "FAIL") -ForegroundColor Red
                    }
                }
            }
        }

        continue
    }

    $status = $result.Payload.status
    if ($status -eq "healthy") {
        Write-Host "PASS  $($result.Name): healthy" -ForegroundColor Green
    }
    elseif ($status -eq "degraded") {
        if ($AllowDegraded) {
            $warnings++
            Write-Host "WARN  $($result.Name): degraded (allowed)" -ForegroundColor Yellow
        }
        else {
            $failed++
            Write-Host "FAIL  $($result.Name): degraded" -ForegroundColor Red
        }
    }
    else {
        $warnings++
        Write-Host "WARN  $($result.Name): status field missing/unexpected" -ForegroundColor Yellow
    }
}

if ($ShowComposePs) {
    Write-Host "\nContainer status:" -ForegroundColor Cyan
    try {
        docker compose ps
    }
    catch {
        Write-Host "WARN  Could not run 'docker compose ps': $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

Write-Host "\nSummary: $failed failed, $warnings warnings" -ForegroundColor Cyan

if ($failed -gt 0) {
    exit 1
}

exit 0
