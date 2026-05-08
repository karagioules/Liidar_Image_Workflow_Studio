# Third-Party Notices

This repository tracks source code, tests, documentation, and dependency manifests. It does not vendor ComfyUI, Python package directories, Node package directories, model weights, generated images, datasets, or binary model artifacts.

## Dependency Manifests

Backend dependencies are declared in:

- `app/backend/pyproject.toml`
- `app/backend/uv.lock`

Frontend dependencies are declared in:

- `app/frontend/package.json`
- `app/frontend/package-lock.json`

Before publishing a formal release, review the licenses of direct and transitive dependencies from those manifests and update this file with any required attribution, notice, source-offer, or redistribution steps.

## Assets Not Bundled

The following are intentionally not bundled in this repository:

- ComfyUI and custom nodes.
- Diffusion checkpoints, adapters, VAEs, text encoders, and upscalers.
- Generated images or videos.
- Training datasets.
- Local run outputs and logs.

Users who install external tools, models, or datasets are responsible for complying with those projects' licenses.
