param(
    [int]$AoiPort = 18000,
    [int]$ReflowPort = 18001,
    [int]$SpiPort = 18002,
    [int]$QdrantPort = 16333
)

$workspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$pcbAgentPython = "D:\conda_envs\PCB_Agent\python.exe"
$resultRoot = Join-Path $PSScriptRoot "results"
$logRoot = Join-Path $resultRoot "logs"
$qdrantContainer = "pcba-qdrant-t15"
$serviceProcesses = @{}
$qdrantStarted = $false

if (-not (Test-Path -LiteralPath $pcbAgentPython)) {
    throw "PCB_Agent Python not found: $pcbAgentPython"
}

New-Item -ItemType Directory -Path $logRoot -Force | Out-Null

try {
    $existing = docker ps -a --filter "name=^/$qdrantContainer$" --format "{{.Names}}"
    if (-not $existing) {
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
        throw "T15 Qdrant did not become ready"
    }

    $definitions = @(
        @{ Name = "aoi"; Module = "tool.services.aoi.app.main:app"; Port = $AoiPort },
        @{ Name = "reflow"; Module = "tool.services.reflow.app.main:app"; Port = $ReflowPort },
        @{ Name = "spi"; Module = "tool.services.spi.app.main:app"; Port = $SpiPort }
    )
    $occupied = Get-NetTCPConnection `
        -State Listen `
        -LocalPort @($AoiPort, $ReflowPort, $SpiPort) `
        -ErrorAction SilentlyContinue
    if ($occupied) {
        throw "T15 model-service ports are occupied: $($occupied.LocalPort -join ', ')"
    }

    foreach ($definition in $definitions) {
        $stdout = Join-Path $logRoot "$($definition.Name).stdout.log"
        $stderr = Join-Path $logRoot "$($definition.Name).stderr.log"
        $arguments = @(
            "-m", "uvicorn", $definition.Module,
            "--host", "127.0.0.1",
            "--port", [string]$definition.Port
        )
        $process = Start-Process `
            -FilePath $pcbAgentPython `
            -ArgumentList $arguments `
            -WorkingDirectory $workspaceRoot `
            -WindowStyle Hidden `
            -RedirectStandardOutput $stdout `
            -RedirectStandardError $stderr `
            -PassThru
        $serviceProcesses[$definition.Name] = $process
    }

    foreach ($definition in $definitions) {
        $deadline = (Get-Date).AddSeconds(120)
        $ready = $false
        while ((Get-Date) -lt $deadline) {
            $process = $serviceProcesses[$definition.Name]
            if ($process.HasExited) {
                throw "$($definition.Name) exited during startup; inspect $logRoot"
            }
            try {
                $health = Invoke-RestMethod `
                    -Uri "http://127.0.0.1:$($definition.Port)/health" `
                    -TimeoutSec 5
                if ($health.success -and $health.data.status -eq "ready") {
                    $ready = $true
                    break
                }
            }
            catch {
                Start-Sleep -Milliseconds 500
            }
        }
        if (-not $ready) {
            throw "$($definition.Name) did not become ready"
        }
    }

    $env:PCBA_AOI_BASE_URL = "http://127.0.0.1:$AoiPort"
    $env:PCBA_REFLOW_BASE_URL = "http://127.0.0.1:$ReflowPort"
    $env:PCBA_SPI_BASE_URL = "http://127.0.0.1:$SpiPort"
    $env:NO_PROXY = "127.0.0.1,localhost"
    $env:no_proxy = "127.0.0.1,localhost"
    & $pcbAgentPython `
        agent/evaluation/run_t15_evaluation.py `
        --qdrant-url "http://127.0.0.1:$QdrantPort"
    if ($LASTEXITCODE -ne 0) {
        throw "T15 evaluation exited with code $LASTEXITCODE"
    }
}
finally {
    foreach ($process in $serviceProcesses.Values) {
        if (-not $process.HasExited) {
            Stop-Process -Id $process.Id
        }
    }
    if ($qdrantStarted) {
        docker stop $qdrantContainer | Out-Null
    }
}
