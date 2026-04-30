# ComfyUI Workflows

Local Model Studio V1 renders a minimal SDXL ComfyUI graph from a `PromptRecipe`.

The V1 workflow is intentionally small:

- `CheckpointLoaderSimple` loads the selected SDXL checkpoint.
- Positive and negative `CLIPTextEncode` nodes receive the recipe prompts.
- `EmptyLatentImage` uses the recipe width and height with `batch_size` 1.
- `KSampler` uses the recipe seed, steps, and CFG with `dpmpp_2m`, `karras`, and full denoise.
- `VAEDecode` and `SaveImage` produce the final image.

This keeps the first ComfyUI adapter predictable for AMD-first local generation and believable-photo prompt recipes. Future workflow templates can add advanced controls such as LoRA loading, ControlNet, IP-Adapter/reference matching, high-resolution refinement, tiled VAE options, and metadata-aware output naming while preserving this minimal SDXL path as the baseline smoke workflow.
