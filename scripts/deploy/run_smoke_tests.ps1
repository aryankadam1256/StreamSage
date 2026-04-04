param(
    [Parameter(Mandatory = $false)]
    [string]$GatewayBaseUrl = "http://localhost:8000",

    [Parameter(Mandatory = $false)]
    [string]$DiscoverQuery = "interstellar",

    [Parameter(Mandatory = $false)]
    [string]$OracleMovieId = "interstellar",

    [Parameter(Mandatory = $false)]
    [string]$OracleQuery = "does cooper find brand at the end or not ?",

    [Parameter(Mandatory = $false)]
    [int]$TimeoutSec = 180,

    [Parameter(Mandatory = $false)]
    [switch]$SkipDiscover,

    [Parameter(Mandatory = $false)]
    [switch]$SkipOracle,

    [Parameter(Mandatory = $false)]
    [switch]$SkipOracleStream
)

$ErrorActionPreference = "Stop"
$failed = 0

$discoverEndpoint = "{0}/api/v1/discover" -f $GatewayBaseUrl.TrimEnd('/')
$oracleAskEndpoint = "{0}/api/v1/oracle/ask" -f $GatewayBaseUrl.TrimEnd('/')
$oracleStreamEndpoint = "{0}/api/v1/oracle/ask/stream" -f $GatewayBaseUrl.TrimEnd('/')

function Pass($message) {
    Write-Host "PASS  $message" -ForegroundColor Green
}

function Fail($message) {
    $script:failed++
    Write-Host "FAIL  $message" -ForegroundColor Red
}

function Warn($message) {
    Write-Host "WARN  $message" -ForegroundColor Yellow
}

Write-Host "Running StreamSage smoke tests..." -ForegroundColor Cyan

# Test 1: Gateway health
try {
    $health = Invoke-RestMethod -Uri "$GatewayBaseUrl/health" -Method Get -TimeoutSec 15
    if ($health.overall -eq "healthy" -or $health.overall -eq "degraded") {
        Pass "Gateway /health responded with overall=$($health.overall)"
    }
    else {
        Fail "Gateway /health returned unexpected overall=$($health.overall)"
    }
}
catch {
    Fail "Gateway /health request failed: $($_.Exception.Message)"
}

# Test 2: Movie discover path
if (-not $SkipDiscover) {
    try {
        $discoverBody = @{ query = $DiscoverQuery } | ConvertTo-Json
        $discoverResp = Invoke-RestMethod `
            -Uri $discoverEndpoint `
            -Method Post `
            -ContentType "application/json" `
            -Body $discoverBody `
            -TimeoutSec $TimeoutSec

        if ($null -eq $discoverResp) {
            Fail "Discover response was null"
        }
        elseif ($discoverResp.error) {
            Fail "Discover returned error payload: $($discoverResp.error)"
        }
        else {
            $propCount = @($discoverResp.PSObject.Properties).Count
            if ($propCount -gt 0) {
                Pass "Discover endpoint returned structured JSON ($propCount top-level fields)"
            }
            else {
                Fail "Discover endpoint returned empty JSON object"
            }
        }
    }
    catch {
        Fail "Discover request failed: $($_.Exception.Message)"
    }
}
else {
    Warn "Skipping discover smoke test"
}

# Shared Oracle payload
$oraclePayload = @{
    query = $OracleQuery
    movie_id = $OracleMovieId
    top_k = 5
    conversation_history = @()
    already_watched = $true
}

# Test 3: Oracle sync ask path
if (-not $SkipOracle) {
    try {
        $askBody = $oraclePayload | ConvertTo-Json -Depth 6
        $askResp = Invoke-RestMethod `
            -Uri $oracleAskEndpoint `
            -Method Post `
            -ContentType "application/json" `
            -Body $askBody `
            -TimeoutSec $TimeoutSec

        if (-not $askResp.answer) {
            Fail "Oracle /ask returned no answer"
        }
        else {
            $answerLen = ([string]$askResp.answer).Length
            Pass "Oracle /ask returned answer ($answerLen chars)"
        }

        if ($askResp.sources) {
            $sourceCount = @($askResp.sources).Count
            Pass "Oracle /ask returned $sourceCount source chunks"
        }
        else {
            Warn "Oracle /ask returned no sources"
        }
    }
    catch {
        Fail "Oracle /ask request failed: $($_.Exception.Message)"
    }
}
else {
    Warn "Skipping Oracle sync smoke test"
}

# Test 4: Oracle stream ask path (SSE)
if (-not $SkipOracleStream) {
    try {
        $streamBody = $oraclePayload | ConvertTo-Json -Depth 6
        $streamResp = Invoke-WebRequest `
            -Uri $oracleStreamEndpoint `
            -Method Post `
            -ContentType "application/json" `
            -Body $streamBody `
            -UseBasicParsing `
            -TimeoutSec $TimeoutSec

        $content = [string]$streamResp.Content

        if ([string]::IsNullOrWhiteSpace($content)) {
            Fail "Oracle /ask/stream returned empty body"
        }
        elseif ($content -notmatch '"type"\s*:\s*"sources"') {
            Fail "Oracle /ask/stream missing sources event"
        }
        elseif ($content -notmatch '"type"\s*:\s*"done"') {
            Fail "Oracle /ask/stream missing done event"
        }
        else {
            Pass "Oracle /ask/stream returned valid SSE payload (sources + done present)"
        }
    }
    catch {
        Fail "Oracle /ask/stream request failed: $($_.Exception.Message)"
    }
}
else {
    Warn "Skipping Oracle streaming smoke test"
}

Write-Host "\nSmoke test summary: $failed failed" -ForegroundColor Cyan
if ($failed -gt 0) {
    exit 1
}

exit 0
