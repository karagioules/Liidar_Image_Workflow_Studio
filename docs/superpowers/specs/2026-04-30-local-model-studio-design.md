# Local Model Studio Design

## Goal

Build a Windows application for local, high-quality AI image generation using the user's own computer instead of paid cloud token services. The app should make it easy to create and reuse fictional adult model personas with consistent visual identity, body attributes, styling, and output quality.

The workspace samples under `Models/` are reference material for target quality and organization only. The product must remain flexible and should not be built around those specific sample personas.

## Safety and Scope

The application is for fictional adult personas only.

The app will include guardrails that prevent or discourage:

- Minor or ambiguous-age characters.
- Real-person impersonation or non-consensual likeness cloning.
- Importing a real person's photos as an identity target unless the user explicitly owns the rights and consent for that person.

The app can support mature/adult body and style attributes, but the first implementation will focus on the local generation workflow, character consistency, and ease of use rather than producing any particular content during development.

## Recommended Architecture

Use ComfyUI as the local generation engine and build a simpler app layer on top of it.

Reasons:

- ComfyUI already supports advanced diffusion workflows, LoRA loading, image references, ControlNet-style conditioning, upscaling, and video extensions.
- AMD and ComfyUI now document Windows support paths for RX 9000-series GPUs through ROCm/PyTorch, though it is still described as experimental/beta.
- Hiding ComfyUI behind a curated UI gives the user professional controls without requiring node-graph editing.

The app layer will be a local web UI or desktop shell that manages:

- Character profiles.
- Prompt recipes.
- Quality presets.
- ComfyUI workflow templates.
- Job execution and output library.

## Main User Experience

The user opens the app and works from four main areas:

1. **Characters**
   Create fictional adult personas with locked identity attributes and editable body/style traits.

2. **Generate**
   Pick a character, choose a scene/style preset, select a quality level, and generate images locally.

3. **Library**
   Review outputs with character name, workflow, seed, prompt recipe, and favorite/reject status.

4. **Setup**
   Install/check ComfyUI, AMD runtime, models, custom nodes, and generation templates.

Normal use should not expose ComfyUI parameters such as sampler names, scheduler strings, CFG, denoise, node graphs, or latent dimensions unless an advanced mode is enabled.

## Character Profiles

Each character profile stores structured attributes instead of only a plain prompt.

Core fields:

- Display name.
- Fictional adult age descriptor, fixed to adult-only categories.
- Face identity summary.
- Hair color, hair style, eye color, skin tone.
- Body shape, height, build, waist/hips proportions.
- Chest size and natural shape descriptors.
- Grooming/body-hair descriptors.
- Fashion/style preferences.
- Personality/aesthetic notes.
- Negative prompt notes.
- Reference images, if available.
- LoRA or embedding files, if trained/imported.
- Default seed strategy.

The profile editor should use simple controls and plain labels. It should allow specific adult anatomy/body-trait choices while keeping them stored as structured profile data that can be rendered into prompts.

## Consistency Strategy

Consistency will come from several layers:

1. **Profile prompt locking**
   Stable identity and body descriptors are automatically reused every time a character is selected.

2. **Seed management**
   The app can lock, reuse, or vary seeds depending on whether the user wants repeated likeness or fresh variety.

3. **Reference conditioning**
   Character reference images can be used in workflows that support face/reference guidance.

4. **LoRA support**
   The app will allow later training or importing of character LoRAs for stronger consistency.

5. **Output metadata**
   Every output stores the exact character profile version, scene recipe, workflow, seed, and settings so successful looks can be reproduced.

## Generation Modes

Initial modes:

- **Portrait**
  Best for face consistency and profile/icon work.

- **Full Body**
  Best for body-shape consistency and outfits.

- **Lifestyle Post**
  Candid, social-media-like scenes.

- **Studio**
  Controlled lighting and cleaner product-style outputs.

- **Reference Match**
  Uses a selected reference image to preserve pose, composition, or identity.

Future modes:

- Batch content calendar generation.
- Short video generation.
- LoRA training helper.
- Upscale and retouch workflows.

## Quality Presets

Expose only simple quality choices:

- **Fast**
  Lower steps/resolution for testing ideas.

- **Balanced**
  Default daily-use quality.

- **High**
  Better detail and upscaling, slower.

- **Ultra**
  Highest local quality workflow the hardware can reasonably support.

The app translates these choices into workflow settings internally.

## Backend Components

The backend will include:

- A Windows setup script that installs or checks prerequisites.
- ComfyUI installation management.
- AMD ROCm/PyTorch compatibility checks.
- Model and custom-node directory management.
- Workflow template renderer.
- Queue/job runner that talks to ComfyUI's local API.
- Metadata writer for generated files.

The first build should prefer official ComfyUI/AMD paths before using community workarounds. If ROCm on Windows is unstable on this machine, the app should clearly report that and offer a fallback path rather than silently failing.

## Data Layout

Suggested workspace layout:

```text
Liidar/
  app/
    frontend/
    backend/
  config/
    characters/
    presets/
    workflows/
  outputs/
    images/
    metadata/
  tools/
    setup/
  docs/
    superpowers/
      specs/
```

Existing `Models/` samples remain untouched.

## Error Handling

The app should show simple, useful messages:

- GPU/runtime not detected.
- ComfyUI not installed or not running.
- Model file missing.
- Workflow template failed.
- Out of VRAM or RAM.
- Generation canceled or timed out.

For technical details, the app can keep logs under `logs/`.

## Testing and Verification

Minimum verification for the first implementation:

- Setup script can detect Windows, CPU, RAM, GPU, and driver version.
- App can start locally.
- App can create/edit/delete fictional adult character profiles.
- App can render a workflow from a character profile and quality preset.
- App can submit a job to a local ComfyUI server when available.
- Generated output metadata is saved and readable.
- Existing `Models/` samples are not modified.

## Implementation Order

1. Create project scaffold and profile schema.
2. Build setup/runtime detection script.
3. Build simple local UI for character profiles and generation form.
4. Add ComfyUI workflow rendering and local API client.
5. Add output library and metadata capture.
6. Add installer/checker for ComfyUI and AMD runtime path.
7. Add advanced consistency tools such as reference slots and LoRA mapping.

## V1 Implementation Choices

The first implementation will use these defaults:

- **App shell:** a browser-based local app with a Windows launcher script. This keeps setup simple and avoids desktop packaging work before generation is proven.
- **Generation engine:** ComfyUI managed by the app, using the official AMD/ROCm Windows path first.
- **Model handling:** the app will support importing locally stored checkpoint/model files and will not assume one specific paid or cloud model. V1 will ship workflow templates for an SDXL-class image workflow because it is practical on 16 GB VRAM, with room for FLUX/Z Image-style workflows after the base path is stable.
- **Training:** LoRA training is deferred until basic profile creation, generation, metadata, and output review work reliably.
