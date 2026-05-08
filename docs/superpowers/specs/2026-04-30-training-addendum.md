# Training Addendum: Dataset-Safe Local Adapters

## Goal

Liidar may eventually support local training helper workflows for user-supplied datasets. The public repository must not include datasets, generated images, private prompts, model weights, or trained adapters.

Training support should create reproducible configuration files and metadata while keeping user data outside git-tracked paths.

## Dataset Safety Policy

Training imports should default to conservative behavior:

- Require the user to confirm they have rights to use the data.
- Store source data outside the repository by default.
- Deduplicate files by hash.
- Reject unsupported file types.
- Record accepted/rejected counts.
- Avoid preserving real-person identity unless a future contributor implements an explicit consent and rights workflow.
- Never commit imported datasets or generated training outputs.

## Dataset Types

Initial generic dataset types:

- Style.
- Composition.
- Object/product.
- Pose.
- Lighting.
- Texture/material.

Each dataset record stores:

- Name.
- Type.
- Source folder path.
- Accepted file count.
- Rejected file count.
- Duplicate count.
- Caption/tag strategy.
- Created timestamp.

## Training Strategy

V1 should generate configs for external trainers rather than hard-code a full trainer stack.

The app should report trainer readiness:

- Python/runtime available.
- GPU backend available.
- Trainer folder configured.
- Base model path configured.
- Output folder writable.

If a trainer is unavailable, the app should fail clearly before launching work.

## UI Requirements

Add a **Training** area only after generation and metadata workflows are stable:

1. Choose dataset folder.
2. Choose dataset type.
3. Scan dataset.
4. Review accepted/rejected counts.
5. Generate captions/tags.
6. Generate trainer config.
7. Start local training when backend checks pass.
8. Register produced adapter for use in profiles.

## Verification

Training helper verification must include:

- Recursive image scan.
- Hash deduplication.
- Unsupported file handling.
- Captions contain generic tags and no private identity names.
- Generated config includes dataset path, output path, base model path, adapter name, resolution, repeats, batch size, and step settings.
- Missing backend is reported clearly.
