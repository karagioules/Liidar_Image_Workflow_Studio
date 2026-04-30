$ErrorActionPreference = "Stop"

function Write-Info {
    param([string]$Message)
    Write-Host $Message
}

function Resolve-RequiredCommand {
    param(
        [string]$Name,
        [string]$FriendlyName
    )

    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        throw "$FriendlyName was not found on PATH. Install $FriendlyName and rerun this script."
    }

    return $command
}

function Get-PythonCommand {
    $pyLauncher = Get-Command "py" -ErrorAction SilentlyContinue
    if ($null -ne $pyLauncher) {
        $version = & py -3.12 --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            return @{
                Exe = "py"
                Args = @("-3.12")
                Label = "py -3.12 ($version)"
                Path = $pyLauncher.Source
            }
        }
    }

    $python = Get-Command "python" -ErrorAction SilentlyContinue
    if ($null -ne $python) {
        $version = & python -c "import sys; print('%s.%s.%s' % sys.version_info[:3]); raise SystemExit(0 if sys.version_info >= (3, 12) else 1)" 2>&1
        if ($LASTEXITCODE -eq 0) {
            return @{
                Exe = "python"
                Args = @()
                Label = "python ($version)"
                Path = $python.Source
            }
        }

        throw "The python command on PATH is not Python 3.12 or newer. Found: $version. Install Python 3.12, or make py -3.12 available on PATH."
    }

    throw "Python was not found. Install Python 3.12, or make sure py/python is available on PATH."
}

try {
    $workspaceRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
    $comfyUiPath = Join-Path $workspaceRoot "ComfyUI"
    $comfyUiRepo = "https://github.com/comfyanonymous/ComfyUI.git"

    Write-Info "Local Model Studio ComfyUI setup"
    Write-Info "Workspace: $workspaceRoot"
    Write-Info ""

    if (Test-Path -LiteralPath $comfyUiPath -PathType Container) {
        Write-Info "ComfyUI already exists at:"
        Write-Info "  $comfyUiPath"
        Write-Info ""
        Write-Info "No changes were made."
        exit 0
    }

    Resolve-RequiredCommand -Name "git" -FriendlyName "Git" | Out-Null
    $pythonCommand = Get-PythonCommand
    Write-Info "Selected Python: $($pythonCommand.Label)"
    Write-Info "Python path: $($pythonCommand.Path)"
    Write-Info ""

    Write-Info "Cloning ComfyUI into:"
    Write-Info "  $comfyUiPath"
    & git clone $comfyUiRepo $comfyUiPath
    if ($LASTEXITCODE -ne 0) {
        throw "git clone failed with exit code $LASTEXITCODE."
    }

    Write-Info ""
    Write-Info "Creating ComfyUI virtual environment:"
    Write-Info "  $(Join-Path $comfyUiPath ".venv")"
    & $pythonCommand.Exe @($pythonCommand.Args) -m venv (Join-Path $comfyUiPath ".venv")
    if ($LASTEXITCODE -ne 0) {
        throw "Python venv creation failed with exit code $LASTEXITCODE."
    }

    $venvPython = Join-Path $comfyUiPath ".venv\Scripts\python.exe"
    Write-Info "Upgrading pip in the ComfyUI virtual environment..."
    & $venvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        throw "pip upgrade failed with exit code $LASTEXITCODE."
    }

    Write-Info ""
    Write-Info "ComfyUI was cloned and initialized."
    Write-Info ""
    Write-Info "Next steps for AMD/ROCm/PyTorch:"
    Write-Info "  1. Check AMD ROCm and PyTorch support for your exact GPU, driver, and Windows setup."
    Write-Info "  2. From $comfyUiPath, activate .venv and install PyTorch using the official PyTorch selector for your hardware."
    Write-Info "  3. Then install ComfyUI requirements with: python -m pip install -r requirements.txt"
    Write-Info "  4. Do not download checkpoints blindly; choose model files intentionally for disk size, license, and VRAM fit."
    Write-Info ""
    Write-Info "Place model files under:"
    Write-Info "  Checkpoints: $comfyUiPath\models\checkpoints"
    Write-Info "  VAEs:        $comfyUiPath\models\vae"
    Write-Info "  LoRAs:       $comfyUiPath\models\loras"
    Write-Info "  ControlNet:  $comfyUiPath\models\controlnet"
    exit 0
} catch {
    Write-Error "ComfyUI setup failed: $($_.Exception.Message)"
    exit 1
}
