$ErrorActionPreference = "Stop"

try {
    $workspaceRoot = (Resolve-Path $PSScriptRoot).Path
    $backendPath = Join-Path $workspaceRoot "app\backend"
    $frontendPath = Join-Path $workspaceRoot "app\frontend"
    $frontendModules = Join-Path $frontendPath "node_modules"

    if (-not (Test-Path -LiteralPath $backendPath -PathType Container)) {
        throw "Backend folder not found at $backendPath"
    }

    if (-not (Test-Path -LiteralPath $frontendPath -PathType Container)) {
        throw "Frontend folder not found at $frontendPath"
    }

    Write-Host "Starting Liidar Local Model Studio..."
    Write-Host "Workspace: $workspaceRoot"
    Write-Host ""

    if (-not (Test-Path -LiteralPath $frontendModules -PathType Container)) {
        Write-Host "WARNING: Frontend node_modules was not found."
        Write-Host "Run this before launching the frontend:"
        Write-Host "  cd `"$frontendPath`""
        Write-Host "  npm install"
        Write-Host ""
    }

    $backendPathForCommand = $backendPath -replace "'", "''"
    $frontendPathForCommand = $frontendPath -replace "'", "''"
    $backendCommand = "Set-Location -LiteralPath '$backendPathForCommand'; py -3.12 -m uvicorn local_model_studio.main:app --reload --host 127.0.0.1 --port 8000"
    $frontendCommand = "Set-Location -LiteralPath '$frontendPathForCommand'; npm run dev -- --port 5173"

    Start-Process powershell.exe -ArgumentList @(
        "-NoExit",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        $backendCommand
    )

    Start-Process powershell.exe -ArgumentList @(
        "-NoExit",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        $frontendCommand
    )

    Write-Host "Backend:  http://127.0.0.1:8000"
    Write-Host "Frontend: http://127.0.0.1:5173"
    Write-Host ""
    Write-Host "Two PowerShell windows were opened for logs. Close those windows to stop the servers."
    exit 0
} catch {
    Write-Error "Launch failed: $($_.Exception.Message)"
    exit 1
}
