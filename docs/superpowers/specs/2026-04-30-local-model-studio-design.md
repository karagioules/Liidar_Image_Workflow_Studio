# Local Model Studio Design

## Goal

Build a local-first application for managing AI image generation workflows. The app should help users organize local generation engines, profiles, prompt recipes, quality presets, outputs, and metadata without depending on hosted token-based services.

The project is a generic scaffold. It must not ship private personas, generated images, model weights, or training datasets.

## Safety And Scope

The app is a workflow manager. Users are responsible for lawful use of their own installed models, datasets, and generation tools.

The app should discourage or block:

- Minor or ambiguous-age content.
- Non-consensual likeness cloning.
- Real-person impersonation.
- Importing identity references without clear rights and consent.
- Committing generated outputs, datasets, credentials, or model weights into the public repo.

## Recommended Architecture

Use a local generation engine adapter pattern. ComfyUI is the first likely target because it already supports local node workflows, model loading, and API-based job execution.

The app layer manages:

- Profiles.
- Prompt recipes.
- Quality presets.
- Workflow templates.
- Job execution.
- Output metadata and review state.

## Main User Experience

The user works from four main areas:

1. **Profiles**
   Create reusable fictional or project-specific generation profiles.

2. **Generate**
   Pick a profile, choose a scene or style preset, select a quality level, and run a local generation job.

3. **Library**
   Review outputs with profile name, workflow, seed, prompt recipe, and favorite/reject status.

4. **Setup**
   Check local runtime, engine path, model folders, workflow templates, and dependency status.

Normal use should avoid exposing low-level engine parameters unless advanced mode is enabled.

## Profile Data

Each profile should store structured data instead of only a plain prompt.

Core fields:

- Display name.
- Description.
- Visual identity summary.
- Style notes.
- Default positive prompt fragments.
- Default negative prompt fragments.
- Reference assets owned or supplied by the user.
- Optional local model or adapter paths.
- Default seed strategy.

## Consistency Strategy

Consistency comes from several layers:

1. Stable profile prompt fragments.
2. Seed management.
3. Reference conditioning when available.
4. Optional user-supplied adapters.
5. Output metadata that records exact settings for repeatability.

## Generation Modes

Initial modes:

- Portrait.
- Full body.
- Product/object.
- Lifestyle scene.
- Studio scene.
- Reference match.

Future modes:

- Batch generation.
- Short video.
- Training helper.
- Upscale and retouch workflows.

## Quality Presets

Expose simple quality choices:

- Fast.
- Balanced.
- High.
- Ultra.

The app translates these choices into engine settings internally.

## Backend Components

The backend should include:

- Local runtime checks.
- Engine installation/path checks.
- Model directory management.
- Workflow template rendering.
- Queue/job runner.
- Metadata writer.
- Error reporting.

## Data Layout

Suggested layout:

```text
Liidar/
  app/
    frontend/
    backend/
  config/
    profiles/
    presets/
    workflows/
  outputs/
    images/
    metadata/
  tools/
  docs/
```

Private runtime folders should be gitignored.

## Verification

Minimum verification:

- App can start locally.
- Profile create/edit/delete works.
- Workflow rendering works from a profile and preset.
- Local engine status is reported clearly.
- Output metadata is saved and readable.
- Gitignored folders are not modified or committed accidentally.
