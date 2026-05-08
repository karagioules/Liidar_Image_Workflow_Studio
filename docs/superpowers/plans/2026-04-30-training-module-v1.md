# Dataset-Safe Adapter Training Plan

## Goal

Add optional local adapter-training preparation without committing training data, generated samples, model checkpoints, private prompt recipes, or user-specific reference files.

This plan is a scaffold for users who bring their own properly licensed data and train locally. The public repository remains source code, tests, templates, and documentation only.

## Scope

- Dataset folder validation.
- Metadata generation for local training tools.
- Config export for external trainers.
- Run folder creation under ignored paths.
- Post-run registration of local artifacts without tracking artifact files.

## Non-Goals

- Bundling datasets, generated examples, checkpoints, or adapters.
- Hosting training infrastructure.
- Recommending or bundling a specific paid model marketplace.
- Encoding private identity, creator, or business-specific workflows.

## Proposed Modules

- `app/backend/training/datasets.py`: validate folder shape, captions, and missing files.
- `app/backend/training/config.py`: build trainer-neutral config metadata.
- `app/backend/training/runs.py`: create ignored run folders and summarize status.
- `app/backend/training/licenses.py`: record user-supplied license notes for local datasets.

## Dataset Rules

- Training data lives in ignored local folders such as `datasets/`.
- Run output lives in ignored local folders such as `training-runs/`.
- The app stores metadata only when it is generic and safe to publish.
- Users are responsible for confirming rights to any input data or generated outputs they use.
- Real-person identity replication is not a supported public workflow.

## Acceptance Criteria

- The repo contains no media or model artifacts after the training module is added.
- Config files can be generated into ignored run folders.
- The UI makes it clear that datasets and outputs remain local.
- Third-party trainer dependencies are documented before they are introduced.
