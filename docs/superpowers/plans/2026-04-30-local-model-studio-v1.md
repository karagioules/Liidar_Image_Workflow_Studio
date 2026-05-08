# Local Model Studio V1 Plan

## Goal

Build a local-first image workflow studio that helps users compose repeatable generation jobs, manage safe local configuration, and submit those jobs to a local image backend.

The project is intentionally generic. It does not ship generated images, identity datasets, private prompts, model weights, cloud credentials, or business-specific content.

## Scope

- Local configuration for image backend endpoints.
- Workflow templates for text-to-image and image-to-image jobs.
- Prompt recipe storage without private persona data.
- Job submission and status polling.
- Output indexing for files that are generated locally and ignored by Git.
- Basic health checks for required local services.

## Non-Goals

- Bundling checkpoints, adapters, or generated media.
- Shipping private datasets or reference-image packs.
- Providing hosted inference.
- Encoding a specific character, brand, or creator workflow.

## Proposed Backend Modules

- `app/backend/config.py`: validated local settings.
- `app/backend/workflows.py`: workflow template loading and parameter binding.
- `app/backend/jobs.py`: submit, poll, and cancel local generation jobs.
- `app/backend/storage.py`: local output indexing with ignored output folders.
- `app/backend/policy.py`: repository hygiene checks for public release.

## Proposed Frontend Views

- Settings: local backend URL and model path hints.
- Workflow Builder: prompt, seed, size, steps, sampler, and template controls.
- Queue: active jobs, failures, and retry actions.
- Gallery: local outputs loaded from ignored output folders.
- Audit: quick reminders about files that should stay outside Git.

## Safety And Release Rules

- Keep generated media and datasets outside tracked source control.
- Keep model weights and adapters outside the repository unless their license is explicitly compatible and documented.
- Keep credentials, endpoint addresses, paid-service details, and private workflow notes outside Git.
- Prefer generic examples in docs and tests.
- Add third-party license notices before adding runtime dependencies.

## Acceptance Criteria

- A new user can understand the project from `README.md`.
- The repository can be published without private project artifacts.
- The app can be developed without requiring bundled media or model weights.
- Ignored folders cover outputs, datasets, model files, temporary experiments, credentials, and archives.
- A release audit records license status, third-party dependency status, and repository-hygiene checks.
