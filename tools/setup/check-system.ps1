$ErrorActionPreference = "Continue"

Write-Host "Local Model Studio system check"
Write-Host ""

$workspaceRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$comfyUiPath = Join-Path $workspaceRoot "ComfyUI"

function Write-WarningLine {
    param([string]$Message)
    Write-Host "WARNING: $Message"
}

$os = $null
try {
    $os = Get-CimInstance Win32_OperatingSystem -ErrorAction Stop
    Write-Host "OS: $($os.Caption) $($os.Version)"
} catch {
    Write-WarningLine "OS check failed: $($_.Exception.Message)"
    Write-Host "OS: Unknown"
}

$cpu = $null
try {
    $cpu = Get-CimInstance Win32_Processor -ErrorAction Stop | Select-Object -First 1
    if ($null -eq $cpu) {
        Write-WarningLine "CPU check returned no processors."
        Write-Host "CPU: Unknown"
    } else {
        Write-Host "CPU: $($cpu.Name)"
    }
} catch {
    Write-WarningLine "CPU check failed: $($_.Exception.Message)"
    Write-Host "CPU: Unknown"
}

try {
    if ($null -eq $os) {
        $os = Get-CimInstance Win32_OperatingSystem -ErrorAction Stop
    }
    $ramGb = [Math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
    Write-Host "RAM: $ramGb GB"
} catch {
    Write-WarningLine "RAM check failed: $($_.Exception.Message)"
    Write-Host "RAM: Unknown"
}

Write-Host "GPU:"

try {
    $gpus = @(Get-CimInstance Win32_VideoController -ErrorAction Stop)
    if ($gpus.Count -eq 0) {
        Write-Host "  No GPUs detected"
    } else {
        foreach ($gpu in $gpus) {
            Write-Host "  $($gpu.Name) (driver $($gpu.DriverVersion))"
        }
    }
} catch {
    Write-WarningLine "GPU check failed: $($_.Exception.Message)"
    Write-Host "  Unknown"
}

$pythonVersion = $null
try {
    $pythonVersion = (& py -3.12 --version 2>&1)
    if ($LASTEXITCODE -ne 0) {
        throw "py -3.12 exited with code $LASTEXITCODE`: $pythonVersion"
    }
} catch {
    try {
        $pythonVersion = (& python --version 2>&1)
        if ($LASTEXITCODE -ne 0) {
            throw "python exited with code $LASTEXITCODE`: $pythonVersion"
        }
    } catch {
        Write-WarningLine "Python check failed: $($_.Exception.Message)"
        $pythonVersion = "Python not found"
    }
}

Write-Host "Python: $pythonVersion"
Write-Host "ComfyUI: $(if (Test-Path -LiteralPath $comfyUiPath -PathType Container) { "Found at $comfyUiPath" } else { "Not found at $comfyUiPath" })"
