from __future__ import annotations

from typing import Any

from local_model_studio.schemas import PromptRecipe


Workflow = dict[str, dict[str, Any]]


def render_sdxl_workflow(recipe: PromptRecipe, checkpoint_name: str) -> Workflow:
    workflow: Workflow = {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": recipe.seed,
                "steps": recipe.steps,
                "cfg": recipe.cfg,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": 1,
                "model": ["4", 0],
                "positive": ["5", 0],
                "negative": ["6", 0],
                "latent_image": ["7", 0],
            },
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": checkpoint_name,
            },
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": recipe.positive,
                "clip": ["4", 1],
            },
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": recipe.negative,
                "clip": ["4", 1],
            },
        },
        "7": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": recipe.width,
                "height": recipe.height,
                "batch_size": 1,
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["3", 0],
                "vae": ["4", 2],
            },
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "local_model_studio",
                "images": ["8", 0],
            },
        },
    }

    model_source: list[Any] = ["4", 0]
    clip_source: list[Any] = ["4", 1]
    for index, lora_file in enumerate(recipe.lora_files, start=1):
        node_id = str(20 + index)
        workflow[node_id] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": model_source,
                "clip": clip_source,
                "lora_name": lora_file,
                "strength_model": recipe.lora_strength,
                "strength_clip": recipe.lora_strength,
            },
        }
        model_source = [node_id, 0]
        clip_source = [node_id, 1]

    workflow["3"]["inputs"]["model"] = model_source
    workflow["5"]["inputs"]["clip"] = clip_source
    workflow["6"]["inputs"]["clip"] = clip_source
    return workflow
