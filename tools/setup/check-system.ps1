$ErrorActionPreference = "Continue"

Write-Host "Local Model Studio system check"
Write-Host ""

$os = Get-CimInstance Win32_OperatingSystem
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$ramGb = [Math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
$gpus = @(Get-CimInstance Win32_VideoController)

Write-Host "OS: $($os.Caption) $($os.Version)"
Write-Host "CPU: $($cpu.Name)"
Write-Host "RAM: $ramGb GB"
Write-Host "GPU:"

if ($gpus.Count -eq 0) {
    Write-Host "  No GPUs detected"
} else {
    foreach ($gpu in $gpus) {
        Write-Host "  $($gpu.Name) (driver $($gpu.DriverVersion))"
    }
}

$pythonVersion = $null
try {
    $pythonVersion = (& py -3.12 --version 2>&1)
} catch {
    try {
        $pythonVersion = (& python --version 2>&1)
    } catch {
        $pythonVersion = "Python not found"
    }
}

Write-Host "Python: $pythonVersion"
