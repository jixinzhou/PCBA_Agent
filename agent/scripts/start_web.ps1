param(
    [ValidateRange(1, 65535)]
    [int]$WebPort = 8080,
    [ValidateRange(1, 65535)]
    [int]$AoiPort = 18000,
    [ValidateRange(1, 65535)]
    [int]$ReflowPort = 18001,
    [ValidateRange(1, 65535)]
    [int]$SpiPort = 18002,
    [ValidateRange(1, 65535)]
    [int]$QdrantPort = 16333,
    [string]$PythonPath = "",
    [switch]$NoBrowser,
    [switch]$ExitAfterHealthCheck,
    [switch]$KeepServices
)

$ErrorActionPreference = "Stop"
$workspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$storageRoot = Join-Path $workspaceRoot "agent\storage"
$logRoot = Join-Path $storageRoot "web_logs"
$qdrantContainer = "pcba-qdrant-web"
$serviceProcesses = @{}
$qdrantStarted = $false

function Resolve-PCBAAgentPython {
    param([string]$ExplicitPath)
    if ($ExplicitPath) {
        if (-not (Test-Path -LiteralPath $ExplicitPath)) {
            throw "PCB_Agent Python not found: $ExplicitPath"
        }
        return (Resolve-Path -LiteralPath $ExplicitPath).Path
    }
    $conda = Get-Command conda -ErrorAction SilentlyContinue
    if ($conda) {
        $environmentList = (& $conda.Source env list --json | ConvertFrom-Json).envs
        $environmentPath = $environmentList | Where-Object {
            (Split-Path -Leaf $_) -eq "PCB_Agent"
        } | Select-Object -First 1
        if ($environmentPath) {
            $candidate = Join-Path $environmentPath "python.exe"
            if (Test-Path -LiteralPath $candidate) {
                return $candidate
            }
        }
    }
    $fallback = "D:\conda_envs\PCB_Agent\python.exe"
    if (Test-Path -LiteralPath $fallback) {
        return $fallback
    }
    throw "Unable to locate the PCB_Agent Conda environment. Use -PythonPath."
}

function Wait-HttpReady {
    param(
        [string]$Name,
        [string]$Url,
        [int]$TimeoutSeconds = 120,
        [System.Diagnostics.Process]$Process = $null
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if ($Process -and $Process.HasExited) {
            throw "$Name exited during startup. Inspect $logRoot"
        }
        try {
            $response = Invoke-RestMethod -Uri $Url -TimeoutSec 4
            if ($response.success -and $response.data.status -eq "ready") {
                return
            }
            if ($response.title -eq "qdrant - vector search engine") {
                return
            }
        }
        catch {
            Start-Sleep -Milliseconds 500
        }
    }
    throw "$Name did not become ready within $TimeoutSeconds seconds"
}

function Start-UvicornService {
    param(
        [string]$Name,
        [string]$Module,
        [int]$Port,
        [string]$Python
    )
    if (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue) {
        throw "Port $Port for $Name is already occupied"
    }
    $stdout = Join-Path $logRoot "$Name.stdout.log"
    $stderr = Join-Path $logRoot "$Name.stderr.log"
    return Start-Process `
        -FilePath $Python `
        -ArgumentList @(
            "-m", "uvicorn", $Module,
            "--host", "127.0.0.1", "--port", [string]$Port
        ) `
        -WorkingDirectory $workspaceRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -PassThru
}

function Test-TcpPortInUse {
    param([int]$Port)
    return [bool](
        Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
    )
}

function Find-AvailableWebPort {
    param([int]$RequestedPort)
    if (-not (Test-TcpPortInUse -Port $RequestedPort)) {
        return $RequestedPort
    }

    $searchStart = if ($RequestedPort -eq 8080) { 18080 } else { $RequestedPort + 1 }
    $searchEnd = [Math]::Min($searchStart + 19, 65535)
    for ($candidate = $searchStart; $candidate -le $searchEnd; $candidate++) {
        if (-not (Test-TcpPortInUse -Port $candidate)) {
            Write-Warning "Port $RequestedPort is occupied; using $candidate for PCBA Agent Web."
            return $candidate
        }
    }
    throw "No available Web port found between $searchStart and $searchEnd"
}

$pcbAgentPython = Resolve-PCBAAgentPython -ExplicitPath $PythonPath
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
$WebPort = Find-AvailableWebPort -RequestedPort $WebPort

try {
    if (-not (Test-Path -LiteralPath (Join-Path $workspaceRoot "agent\.env"))) {
        throw "agent/.env is missing. Copy agent/.env.example and configure Qwen first."
    }
    if (-not (Test-Path -LiteralPath (Join-Path $workspaceRoot "kg\.env"))) {
        throw "kg/.env is missing. Copy kg/.env.example and configure Neo4j first."
    }

    docker info | Out-Null
    docker compose `
        --env-file (Join-Path $workspaceRoot "kg\.env") `
        -f (Join-Path $workspaceRoot "kg\docker-compose.neo4j.yml") `
        up -d | Out-Null

    $existingQdrant = docker ps -a `
        --filter "name=^/$qdrantContainer$" `
        --format "{{.Names}}"
    if (-not $existingQdrant) {
        docker run -d `
            --name $qdrantContainer `
            -p "127.0.0.1:$QdrantPort`:6333" `
            -v "pcba_qdrant_v0_1_data:/qdrant/storage" `
            qdrant/qdrant:v1.18.2 | Out-Null
    }
    else {
        docker start $qdrantContainer | Out-Null
    }
    $qdrantStarted = $true

    $qdrantDeadline = (Get-Date).AddSeconds(90)
    $qdrantReady = $false
    while ((Get-Date) -lt $qdrantDeadline) {
        try {
            $response = Invoke-WebRequest `
                -Uri "http://127.0.0.1:$QdrantPort/healthz" `
                -TimeoutSec 3 `
                -UseBasicParsing
            if ($response.StatusCode -eq 200) {
                $qdrantReady = $true
                break
            }
        }
        catch {
            Start-Sleep -Milliseconds 500
        }
    }
    if (-not $qdrantReady) {
        throw "Qdrant did not become ready"
    }

    $sourcePaths = @(
        (Join-Path $workspaceRoot "agent\src"),
        (Join-Path $workspaceRoot "rag\src"),
        (Join-Path $workspaceRoot "kg\src")
    )
    $env:PYTHONPATH = $sourcePaths -join [IO.Path]::PathSeparator
    $env:PCBA_AOI_BASE_URL = "http://127.0.0.1:$AoiPort"
    $env:PCBA_REFLOW_BASE_URL = "http://127.0.0.1:$ReflowPort"
    $env:PCBA_SPI_BASE_URL = "http://127.0.0.1:$SpiPort"
    $env:PCBA_QDRANT_URL = "http://127.0.0.1:$QdrantPort"
    $env:NO_PROXY = "127.0.0.1,localhost"
    $env:no_proxy = "127.0.0.1,localhost"

    $definitions = @(
        @{ Name = "aoi"; Module = "tool.services.aoi.app.main:app"; Port = $AoiPort },
        @{ Name = "reflow"; Module = "tool.services.reflow.app.main:app"; Port = $ReflowPort },
        @{ Name = "spi"; Module = "tool.services.spi.app.main:app"; Port = $SpiPort }
    )
    foreach ($definition in $definitions) {
        $healthUrl = "http://127.0.0.1:$($definition.Port)/health"
        if (Get-NetTCPConnection -State Listen -LocalPort $definition.Port -ErrorAction SilentlyContinue) {
            Wait-HttpReady -Name $definition.Name -Url $healthUrl -TimeoutSeconds 10
            Write-Host "Reusing healthy $($definition.Name) service on port $($definition.Port)"
        }
        else {
            $process = Start-UvicornService `
                -Name $definition.Name `
                -Module $definition.Module `
                -Port $definition.Port `
                -Python $pcbAgentPython
            $serviceProcesses[$definition.Name] = $process
            Wait-HttpReady `
                -Name $definition.Name `
                -Url $healthUrl `
                -Process $process
        }
    }

    $webProcess = Start-UvicornService `
        -Name "agent-web" `
        -Module "pcba_agent.web:app" `
        -Port $WebPort `
        -Python $pcbAgentPython
    $serviceProcesses["agent-web"] = $webProcess
    Wait-HttpReady `
        -Name "agent-web" `
        -Url "http://127.0.0.1:$WebPort/health" `
        -Process $webProcess

    $url = "http://127.0.0.1:$WebPort"
    Write-Host ""
    Write-Host "PCBA Agent Web is ready: $url" -ForegroundColor Green
    Write-Host "Logs: $logRoot"
    if ($ExitAfterHealthCheck) {
        exit 0
    }
    if (-not $NoBrowser) {
        Start-Process $url
    }
    Read-Host "Press Enter to stop the local web services"
}
finally {
    if (-not $KeepServices) {
        foreach ($process in $serviceProcesses.Values) {
            if ($process -and -not $process.HasExited) {
                Stop-Process -Id $process.Id
            }
        }
        if ($qdrantStarted) {
            docker stop $qdrantContainer | Out-Null
        }
    }
}
