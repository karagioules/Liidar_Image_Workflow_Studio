# Liidar

Liidar is a local-first AI image workflow studio scaffold. It is intended to help users organize local generation engines, model files, prompt recipes, character/profile metadata, output review, and reproducible generation settings without depending on hosted token-based services.

The repository is currently an early planning/scaffold project. It does not ship model weights, training datasets, generated image sets, or a complete production application.

## What This Project Is

- A starting point for a local AI image generation manager.
- A place to define profile schemas, prompt recipe structures, workflow templates, job metadata, and safety boundaries.
- A public MIT-licensed project that others can fork, modify, and use for their own lawful workflows.

## What This Project Is Not

- It is not a model-weight repository.
- It is not a dataset repository.
- It is not a generated-image dataset, identity pack, or model-weight bundle.
- It is not a service for impersonating real people or bypassing consent requirements.

## Current Repository State

The repo currently contains:

- `app/` placeholder folders for a future frontend/backend.
- `docs/` sanitized planning documents for the local studio architecture and dataset/training safety model.
- `LICENSE` with the MIT license.
- `THIRD_PARTY_NOTICES.md` for current dependency/compliance status.
- `docs/audits/license-policy-audit.md` with the current release-readiness audit.

No generated images, private training data, adapter checkpoints, model weights, or user-specific persona artifacts are intentionally included.

## Safety Boundaries

Liidar is designed around user-controlled local workflows. Public releases should not include:

- Non-consensual likeness cloning workflows.
- Real-person identity datasets.
- Illegal, exploitative, or minor/ambiguous-age content.
- Private model checkpoints, generated images, or personal training data.
- Platform credentials, remote access keys, cloud endpoints, or paid-service tokens.

Users are responsible for complying with the licenses of any models, datasets, custom nodes, plugins, and generation tools they install separately.

## Planned Architecture

The intended app shape is:

- A local UI for profiles, prompt recipes, and generation jobs.
- A backend adapter for local generation engines such as ComfyUI.
- A workflow-template renderer that turns profile + preset data into engine-specific requests.
- An output library that stores prompt, seed, workflow, model, and review metadata.
- Optional dataset/training helpers that enforce consent and identity-safety checks before producing local training configs.

## Suggested Layout

```text
Liidar/
  app/
    backend/
    frontend/
  config/
    profiles/
    presets/
    workflows/
  docs/
  outputs/        # ignored by git
  models/         # ignored by git
  logs/           # ignored by git
```

## License

MIT. See [LICENSE](LICENSE).

## Public Release Checklist

Before publishing a release, verify:

- `git status` contains no private temporary files.
- No generated images or training datasets are tracked.
- No model weights are tracked.
- No cloud credentials, remote access keys, cloud endpoints, or private paths are tracked.
- `README.md`, `LICENSE`, and `THIRD_PARTY_NOTICES.md` are present and consistent.
- Any dependencies added later have their licenses reviewed and notices updated.
