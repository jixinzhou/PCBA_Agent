param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("aoi", "reflow", "spi")]
    [string]$Service,

    [ValidateRange(1, 65535)]
    [int]$Port = 0
)

$workspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$services = @{
    aoi = @{
        Module = "tool.services.aoi.app.main:app"
        Port = 8000
    }
    reflow = @{
        Module = "tool.services.reflow.app.main:app"
        Port = 8001
    }
    spi = @{
        Module = "tool.services.spi.app.main:app"
        Port = 8002
    }
}

$selected = $services[$Service]
$listenPort = if ($Port -gt 0) { $Port } else { $selected.Port }
Set-Location -LiteralPath $workspaceRoot
conda run --no-capture-output -n PCB_Agent python -m uvicorn `
    $selected.Module `
    --host 0.0.0.0 `
    --port $listenPort
