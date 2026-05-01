$ErrorActionPreference = "Stop"
$WorkspaceRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$ToolsDir = Join-Path $WorkspaceRoot "tools"
$SdScripts = Join-Path $ToolsDir "sd-scripts"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
  throw "Git is required to install sd-scripts."
}

if (-not (Test-Path -LiteralPath $SdScripts)) {
  git clone https://github.com/kohya-ss/sd-scripts.git $SdScripts
}

Set-Location -LiteralPath $SdScripts
if (-not (Test-Path -LiteralPath ".venv")) {
  python -m venv .venv
}

$Python = Join-Path $SdScripts ".venv\Scripts\python.exe"
& $Python -m pip install --upgrade pip setuptools wheel
if (Test-Path -LiteralPath "requirements_windows_torch2.txt") {
  & $Python -m pip install -r requirements_windows_torch2.txt
} elseif (Test-Path -LiteralPath "requirements.txt") {
  & $Python -m pip install -r requirements.txt
}
& $Python -m pip install accelerate toml safetensors
& $Python -m pip install --pre --force-reinstall torch torchvision torchaudio --index-url https://rocm.nightlies.amd.com/v2/gfx120X-all/
& $Python -c "import torch; print('Trainer torch:', torch.__version__); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available')"

Write-Host "sd-scripts installed at $SdScripts"
Write-Host "Trainer entrypoint: tools\train_global_lora.ps1"
Write-Host "RDNA4 AMD PyTorch was installed for GPU-first training. If training fails, read the log shown in Liidar."
