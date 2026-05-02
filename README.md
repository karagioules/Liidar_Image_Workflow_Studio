# Liidar

Liidar is a local body-sculpture studio for preparing adult character references, training reusable local LoRA packs, and generating finished still images through ComfyUI.

The app is built for a local AMD workflow: lightweight development can happen on any machine, while real generation and training should be tested on the AMD GPU machine with ComfyUI installed.

## Current State

Liidar currently focuses on three simple modules:

- **Characters**: create character blueprints from adult reference images.
- **Generate**: write one image brief, enhance it locally, preview the final recipe, and queue it to ComfyUI.
- **Training**: prepare cropped reference images, name a sculpture pack, and launch high-quality local training.

Runtime status, system metrics, and activity logs live in the app shell instead of a separate tab. The app auto-refreshes backend state in the background.

## What Works Now

- React/Vite frontend at `http://127.0.0.1:5274`
- FastAPI backend at `http://127.0.0.1:8000`
- Runtime and live system monitoring
- Character blueprint creation and local reference analysis
- Single-brief local prompt enhancement
- ComfyUI generation preview, preflight, queueing, and output tracking
- Smart LoRA pack routing from active trained packs
- Merged dataset prep and training workflow
- Local crop preparation for face, chest, butt/hips, genital detail, legs/feet, and full body context
- High-quality training configuration defaults
- Training run launch, polling, cancellation, and log tail support
- Frontend and backend test coverage for the current workflow

## AMD / ComfyUI Requirement

This repo does not include ComfyUI, checkpoints, LoRA weights, model files, generated outputs, datasets, logs, or secrets.

For full generation and training tests, the AMD machine should provide:

- `ComfyUI/` in the project root
- a working ComfyUI Python environment
- SDXL checkpoint files under `ComfyUI/models/checkpoints`
- trained LoRA files under `ComfyUI/models/loras` or registered absolute paths
- AMD GPU drivers and the correct PyTorch/ROCm or DirectML setup for that machine

On non-AMD or non-ComfyUI machines, the app can still be developed and UI-tested, but generation preflight will report missing ComfyUI/checkpoint readiness.

## Start The App

Use the project launcher:

```powershell
.\Start-Liidar.ps1
```

The launcher starts:

- ComfyUI on `http://127.0.0.1:8188`
- backend on `http://127.0.0.1:8000`
- frontend on `http://127.0.0.1:5274`

If ComfyUI is not installed, start the backend and frontend manually for development:

```powershell
cd app\backend
.\.venv\Scripts\python.exe -m uvicorn local_model_studio.main:app --reload --host 127.0.0.1 --port 8000
```

```powershell
cd app\frontend
npm run dev -- --host 127.0.0.1 --port 5274 --strictPort
```

## Verification

Backend:

```powershell
cd app\backend
.\.venv\Scripts\python.exe -m pytest
```

Frontend:

```powershell
cd app\frontend
npm test
npm run build
```

Current verified state:

- backend tests: `122 passed`
- frontend smoke tests: `5 passed`
- frontend production build: passing

## Repository Hygiene

The following are intentionally ignored:

- model weights and checkpoints
- ComfyUI installation
- generated outputs
- datasets and prepared crops
- training runs and logs
- secrets and saved character JSON files
- Python virtual environments
- frontend `node_modules` and build output

This keeps the GitHub repository focused on source code, configuration templates, tests, and documentation.

## Version Tag

The current progress tag documents the simplified studio workflow: merged Training page, one-brief Generate flow, character blueprints, live runtime monitoring, local prompt enhancement, smart pack routing, and verified test/build readiness.
