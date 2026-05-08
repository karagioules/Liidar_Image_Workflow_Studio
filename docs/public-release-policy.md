# Public Release Policy

This repository is intended to be safe to publish under MIT. Public commits must not contain private content or data that a downstream user cannot legally reuse.

## Do Not Commit

- Generated image sets.
- Training datasets.
- Real-person identity references.
- Model weights: `*.safetensors`, `*.ckpt`, `*.pt`, `*.pth`, `*.onnx`.
- Private prompts tied to a specific real or fictional persona.
- Cloud endpoints, remote access commands, tokens, keys, or credentials.
- Platform-specific private business notes.
- Temporary experiment folders.

## Allowed

- Generic source code.
- Generic documentation.
- Safety and consent policy documentation.
- Empty placeholder directories with `.gitkeep` files.
- Example config files that use obviously fictional placeholder names and contain no private paths.

## Dataset And Training Rules

Dataset and training helpers must be designed as generic tools. They should not ship datasets or model outputs.

If training support is added, it should:

- Require the user to supply their own lawful data.
- Store imported data outside git-tracked paths by default.
- Keep face/identity cloning disabled unless a future contributor implements a clear consent and rights workflow.
- Record metadata for reproducibility without storing private images in the repository.

## Release Audit

Before public release:

1. Run keyword scans for private names, business-specific terms, cloud endpoints, and model paths.
2. Run file-extension scans for images, videos, model files, archives, and database files.
3. Check `git status --short`.
4. Check `README.md`, `LICENSE`, and `THIRD_PARTY_NOTICES.md`.
5. Update `docs/audits/license-policy-audit.md`.
