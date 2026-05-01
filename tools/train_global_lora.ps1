param(
  [Parameter(Mandatory = $true)]
  [string]$ConfigPath
)

$ErrorActionPreference = "Stop"
$WorkspaceRoot = Split-Path -Parent $PSScriptRoot
$Config = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
$SdScripts = Join-Path $WorkspaceRoot "tools\sd-scripts"
$TrainScript = Join-Path $SdScripts "sdxl_train_network.py"

if (-not (Test-Path -LiteralPath $TrainScript)) {
  throw "sd-scripts trainer is missing. Install kohya sd-scripts into $SdScripts, or run tools\setup\install-sd-scripts.ps1 after reviewing it."
}

$OutputDir = [string]$Config.output_dir
if (-not [System.IO.Path]::IsPathRooted($OutputDir)) {
  $OutputDir = Join-Path $WorkspaceRoot $OutputDir
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$DatasetPath = [string]$Config.dataset_path
$BaseModel = [string]$Config.base_model_path
if (-not [System.IO.Path]::IsPathRooted($BaseModel)) {
  $BaseModel = Join-Path $WorkspaceRoot $BaseModel
}

$Python = Join-Path $SdScripts ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
  $Python = "python"
}

& $Python -c "import torch; print('Trainer torch:', torch.__version__); print('GPU available:', torch.cuda.is_available()); print('GPU device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU fallback')"
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

& $Python $TrainScript `
  --pretrained_model_name_or_path="$BaseModel" `
  --train_data_dir="$DatasetPath" `
  --output_dir="$OutputDir" `
  --output_name="$($Config.lora_name)" `
  --save_model_as="safetensors" `
  --network_module="networks.lora" `
  --resolution="$($Config.resolution),$($Config.resolution)" `
  --train_batch_size="$($Config.batch_size)" `
  --max_train_steps="$($Config.max_train_steps)" `
  --learning_rate="$($Config.learning_rate)" `
  --network_dim="$($Config.network_dim)" `
  --network_alpha="$($Config.network_alpha)" `
  --mixed_precision="fp16" `
  --save_precision="fp16" `
  --optimizer_type="AdamW" `
  --cache_latents `
  --gradient_checkpointing `
  --persistent_data_loader_workers `
  --max_data_loader_n_workers=0 `
  --enable_bucket `
  --bucket_no_upscale

if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}
