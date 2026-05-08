# Liidar

Liidar is a local-first AI image workflow studio for organizing repeatable image generation workflows through a local ComfyUI setup. It helps users manage profiles, prompt recipes, local training helpers, generation jobs, output metadata, and runtime checks without relying on hosted token-based services.

The project is MIT-licensed and intended as a general-purpose starting point that others can fork, modify, and adapt for their own lawful local workflows.

## What This Project Is

- A local workflow manager for AI image generation.
- A React/Vite frontend plus FastAPI backend.
- A ComfyUI-oriented job and workflow helper.
- A place to manage profile schemas, prompt recipes, workflow templates, training metadata, and output review.
- A source-code project, not a bundled model, dataset, or media pack.

## What This Project Is Not

- It is not a model-weight repository.
- It is not a dataset repository.
- It is not a generated-image pack.
- It is not a service for impersonating real people or bypassing consent requirements.

## Current State

Liidar currently includes:

- React/Vite frontend at `http://127.0.0.1:5274`.
- FastAPI backend at `http://127.0.0.1:8000`.
- Runtime and live system monitoring.
- Profile creation and local reference analysis.
- Single-brief local prompt enhancement.
- ComfyUI generation preview, preflight, queueing, and output tracking.
- Dataset preparation and local training workflow helpers.
- Training run launch, polling, cancellation, and log tail support.
- Frontend and backend test coverage for the current workflow.

## Runtime Requirements

This repo does not include ComfyUI, checkpoints, adapters, model files, generated outputs, datasets, logs, or secrets.

For full generation and training tests, the local machine should provide:

- `ComfyUI/` in the project root.
- A working ComfyUI Python environment.
- Compatible checkpoint files under the local ComfyUI model folders.
- Optional trained adapter files under the local ComfyUI adapter folders or registered absolute paths.
- GPU drivers and the correct PyTorch backend for the machine.

On machines without ComfyUI or a compatible GPU runtime, the app can still be developed and UI-tested, but generation preflight will report missing runtime or model readiness.

## Start The App

Use the project launcher:

```powershell
.\Start-Liidar.ps1
```

The launcher starts:

- ComfyUI on `http://127.0.0.1:8188`
- Backend on `http://127.0.0.1:8000`
- Frontend on `http://127.0.0.1:5274`

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

## Repository Hygiene

Public releases should not include:

- Generated images or output media.
- Training datasets or prepared local crops.
- Model weights, checkpoints, adapters, or other binary model artifacts.
- Private prompt sets, personal reference files, logs, credentials, or cloud endpoints.
- Temporary local worktree/cache folders.

The repository is intended to contain source code, tests, configuration templates, and documentation only.

## License

MIT. See [LICENSE](LICENSE).
