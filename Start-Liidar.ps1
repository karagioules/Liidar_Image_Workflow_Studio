$ErrorActionPreference = "Stop"

function Test-PythonCanImportUvicorn {
    param(
        [string]$Exe,
        [string[]]$Args = @()
    )

    & $Exe @Args -c "import uvicorn" *> $null
    return ($LASTEXITCODE -eq 0)
}

function Get-BackendPythonCommand {
    param([string]$BackendPath)

    $venvPython = Join-Path $BackendPath ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
        if (Test-PythonCanImportUvicorn -Exe $venvPython) {
            return @{
                Command = "`"$venvPython`""
                Label = $venvPython
            }
        }

        throw "Backend virtual environment exists, but uvicorn is not installed or cannot be imported. Run backend setup first: cd `"$BackendPath`"; .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
    }

    $pyLauncher = Get-Command "py" -ErrorAction SilentlyContinue
    if ($null -ne $pyLauncher -and (Test-PythonCanImportUvicorn -Exe "py" -Args @("-3.12"))) {
        return @{
            Command = "py -3.12"
            Label = "py -3.12"
        }
    }

    throw "Backend Python environment is not ready. Expected $venvPython, or a py -3.12 environment that can import uvicorn. Run backend setup before launching Liidar."
}

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

    $backendPython = Get-BackendPythonCommand -BackendPath $backendPath
    Write-Host "Backend Python: $($backendPython.Label)"
    Write-Host ""

    $backendPathForCommand = $backendPath -replace "'", "''"
    $frontendPathForCommand = $frontendPath -replace "'", "''"
    $backendCommand = "Set-Location -LiteralPath '$backendPathForCommand'; $($backendPython.Command) -m uvicorn local_model_studio.main:app --reload --host 127.0.0.1 --port 8000"
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
