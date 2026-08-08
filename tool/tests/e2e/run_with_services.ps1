param(
    [string]$ImagePath = "E:\PCBA智能体\数据资料\AOI图像识别数据\API_test\short (1).png",
    [int]$AoiPort = 18000,
    [int]$ReflowPort = 8001,
    [int]$SpiPort = 8002
)

$workspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$pcbAgentPython = "D:\conda_envs\PCB_Agent\python.exe"
$artifactRoot = Join-Path $PSScriptRoot "artifacts"
$reportPath = Join-Path $artifactRoot "latest_report.json"

if (-not (Test-Path -LiteralPath $pcbAgentPython)) {
    throw "PCB_Agent Python not found: $pcbAgentPython"
}
if (-not (Test-Path -LiteralPath $ImagePath)) {
    throw "AOI test image not found: $ImagePath"
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
    throw "Required test ports are occupied: $($occupied.LocalPort -join ', ')"
}

New-Item -ItemType Directory -Path $artifactRoot -Force | Out-Null
$serviceProcesses = @{}

try {
    foreach ($definition in $definitions) {
        $stdout = Join-Path $artifactRoot "$($definition.Name).stdout.log"
        $stderr = Join-Path $artifactRoot "$($definition.Name).stderr.log"
        $arguments = @(
            "-m", "uvicorn", $definition.Module,
            "--host", "127.0.0.1",
            "--port", [string]$definition.Port
        )
        $serviceProcess = Start-Process `
            -FilePath $pcbAgentPython `
            -ArgumentList $arguments `
            -WorkingDirectory $workspaceRoot `
            -WindowStyle Hidden `
            -RedirectStandardOutput $stdout `
            -RedirectStandardError $stderr `
            -PassThru
        $serviceProcesses[$definition.Name] = $serviceProcess
        Write-Output "STARTED $($definition.Name) PID=$($serviceProcess.Id) PORT=$($definition.Port)"
    }

    foreach ($definition in $definitions) {
        $deadline = (Get-Date).AddSeconds(120)
        $ready = $false
        while ((Get-Date) -lt $deadline) {
            $serviceProcess = $serviceProcesses[$definition.Name]
            if ($serviceProcess.HasExited) {
                throw "$($definition.Name) exited during startup; inspect artifacts logs"
            }
            try {
                $health = Invoke-RestMethod `
                    -Uri "http://127.0.0.1:$($definition.Port)/health" `
                    -TimeoutSec 5
                if ($health.success -and $health.data.status -eq "ready") {
                    $ready = $true
                    Write-Output "READY $($definition.Name)"
                    break
                }
            }
            catch {
                Start-Sleep -Milliseconds 500
            }
        }
        if (-not $ready) {
            throw "$($definition.Name) did not become ready within 120 seconds"
        }
    }

    & $pcbAgentPython `
        -m tool.tests.e2e.run_e2e `
        --image $ImagePath `
        --report $reportPath `
        --aoi-url "http://127.0.0.1:$AoiPort" `
        --reflow-url "http://127.0.0.1:$ReflowPort" `
        --spi-url "http://127.0.0.1:$SpiPort"
    if ($LASTEXITCODE -ne 0) {
        throw "E2E test exited with code $LASTEXITCODE"
    }
}
finally {
    foreach ($serviceProcess in $serviceProcesses.Values) {
        if (-not $serviceProcess.HasExited) {
            Stop-Process -Id $serviceProcess.Id
            Write-Output "STOPPED PID=$($serviceProcess.Id)"
        }
    }
}
