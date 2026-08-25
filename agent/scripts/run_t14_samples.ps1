param(
    [int]$ReflowPort = 8001
)

$workspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$pcbAgentPython = "D:\conda_envs\PCB_Agent\python.exe"
$storageRoot = Join-Path $workspaceRoot "agent\storage"
if (-not (Test-Path -LiteralPath $pcbAgentPython)) {
    throw "PCB_Agent Python not found: $pcbAgentPython"
}
if (Get-NetTCPConnection -State Listen -LocalPort $ReflowPort -ErrorAction SilentlyContinue) {
    throw "Required reflow test port is occupied: $ReflowPort"
}
New-Item -ItemType Directory -Path $storageRoot -Force | Out-Null
$serviceProcess = $null
try {
    $serviceProcess = Start-Process `
        -FilePath $pcbAgentPython `
        -ArgumentList @(
            "-m", "uvicorn", "tool.services.reflow.app.main:app",
            "--host", "127.0.0.1", "--port", [string]$ReflowPort
        ) `
        -WorkingDirectory $workspaceRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $storageRoot "reflow.t14.stdout.log") `
        -RedirectStandardError (Join-Path $storageRoot "reflow.t14.stderr.log") `
        -PassThru

    $deadline = (Get-Date).AddSeconds(120)
    $ready = $false
    while ((Get-Date) -lt $deadline) {
        if ($serviceProcess.HasExited) {
            throw "Reflow service exited during startup"
        }
        try {
            $health = Invoke-RestMethod `
                -Uri "http://127.0.0.1:$ReflowPort/health" `
                -TimeoutSec 3
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
        throw "Reflow service did not become ready within 120 seconds"
    }
    $env:PCBA_REFLOW_BASE_URL = "http://127.0.0.1:$ReflowPort"
    & $pcbAgentPython (Join-Path $workspaceRoot "agent\scripts\run_t14_samples.py")
    if ($LASTEXITCODE -ne 0) {
        throw "T14 sample run failed with code $LASTEXITCODE"
    }
}
finally {
    if ($null -ne $serviceProcess -and -not $serviceProcess.HasExited) {
        Stop-Process -Id $serviceProcess.Id
    }
}
