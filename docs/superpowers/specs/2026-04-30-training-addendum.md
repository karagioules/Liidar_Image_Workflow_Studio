# Training Addendum: Body and Style LoRAs Without Face Cloning

## Goal

Local Model Studio must let the user provide folders containing many consented adult reference photos for generic body-part, body-shape, pose, style, wardrobe, lighting, or composition learning. The app should use these datasets as generic visual knowledge and must not train or preserve a real person's face identity.

Examples include datasets for natural chest shapes, petite body proportions, grooming styles, posing, lighting, or niche creator-content aesthetics. These datasets should produce reusable body/style LoRAs or training artifacts that can be applied to fictional adult characters.

## Non-Negotiable Face Policy

Training must default to **no face cloning**.

The dataset pipeline will:

- Scan imported folders recursively for supported image files.
- Hash files and deduplicate repeated images.
- Detect likely faces before an image is accepted.
- Reject face-containing images by default.
- Allow optional blur/crop redaction only when the redacted output contains no detectable face.
- Store accepted training images separately from rejected images.
- Store rejection reasons in metadata.
- Generate captions that describe generic traits and never a named real identity.

If face detection is unavailable, the pipeline should fail closed for training imports that are not explicitly marked as body-part crops.

## Dataset Types

Initial dataset types:

- **body_part**
  Cropped or face-free body-part reference data.

- **body_shape**
  Face-free full or partial body silhouettes/proportions.

- **pose**
  Pose/composition references where face identity is irrelevant or absent.

- **style**
  Lighting, wardrobe, camera, or creator-content aesthetic references.

- **fictional_face_identity**
  AI-generated or otherwise owned fictional adult face references for one character. This dataset type is separate from generic body/style training and is allowed to preserve a fictional face identity when the user confirms the images are synthetic, owned, or consented.

Each dataset stores:

- Name.
- Dataset type.
- Source folder path.
- Accepted file count.
- Rejected file count.
- Duplicate count.
- Face-policy mode.
- Captions/tags.
- Source rights label: synthetic, owned, licensed, or consented.
- Created timestamp.

## Fictional Face Identity References

The app should support face references created in Grok, Sozee, or similar generators when the user wants a fictional character to keep the same face.

This is not the same as generic body/style training. Face identity references:

- Belong to one fictional adult character profile.
- Require a source-rights label.
- Are not mixed into generic body/style datasets.
- Can be used by reference-guidance workflows first.
- Can later be used for fictional-character LoRA training if the local trainer backend supports it.

The UI should make this separation obvious:

- **Character Face Pack:** preserves one fictional character's AI-generated face.
- **Body/Style Dataset:** learns generic body, pose, wardrobe, grooming, lighting, or composition traits and rejects faces by default.

The app must not present real-person face cloning as a supported workflow.

## Training Strategy

Training should target reusable local LoRAs rather than persona clones.

V1 will support:

- Dataset scan/import.
- Face-safe accepted dataset folder.
- Caption file generation.
- Training configuration generation for an SDXL LoRA workflow.
- A local trainer adapter that can run a configured trainer command when the trainer is installed.
- Training status metadata and output LoRA registration.

Because AMD Windows LoRA training support is less mature than image generation, the app must clearly show trainer readiness:

- Python/runtime available.
- ROCm/PyTorch available.
- Trainer folder available.
- Base model path configured.
- Output folder writable.

The app may support more than one trainer backend, but V1 should start with an external-trainer adapter that can invoke a local command without hard-coding the entire training ecosystem into the app.

## UI Requirements

Add a **Training** area:

1. Choose dataset folder.
2. Choose dataset type.
3. Choose face policy: Reject faces (default), Blur/crop faces, or Body-part crops only.
4. Scan dataset.
5. Review accepted/rejected counts.
6. Generate captions.
7. Start local training when backend checks pass.
8. Register produced LoRA for use in character profiles and generation presets.

Normal users should see simple controls. Advanced trainer command/config details can be shown in an advanced panel.

## Safety and Consent

The app should label all training imports as requiring consent/rights to use the images. It should not contain any workflow designed to identify, preserve, or replicate a real person's face. Character face identity remains fictional and is controlled through profile prompts, seeds, references that the user has rights to use, and later fictional-character LoRAs if created from synthetic/owned images.

## Verification

Training module verification must include:

- Recursive image scan.
- Hash deduplication.
- Unsupported file ignored.
- Face-containing image rejected when detection is available.
- Captions contain generic body/style tags and no identity names.
- Fictional face identity imports require source-rights metadata and stay associated with one character profile.
- Training config includes dataset path, output path, base model path, LoRA name, resolution, repeats, batch size, and epoch/step settings.
- Trainer status reports missing backend clearly instead of failing silently.
