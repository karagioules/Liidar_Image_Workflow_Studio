from local_model_studio.schemas import PromptRecipe
from local_model_studio.workflow_templates import render_sdxl_workflow


def test_render_sdxl_workflow_contains_prompt_and_seed() -> None:
    recipe = PromptRecipe(
        positive="fictional adult, realistic portrait",
        negative="minor, low detail",
        seed=42,
        width=896,
        height=1344,
        steps=26,
        cfg=5.0,
    )

    workflow = render_sdxl_workflow(recipe, checkpoint_name="example.safetensors")

    assert workflow["3"]["inputs"]["seed"] == 42
    assert workflow["5"]["inputs"]["text"] == "fictional adult, realistic portrait"
    assert workflow["6"]["inputs"]["text"] == "minor, low detail"
    assert workflow["4"]["inputs"]["ckpt_name"] == "example.safetensors"


def test_render_sdxl_workflow_uses_expected_sampler_and_size() -> None:
    recipe = PromptRecipe(
        positive="adult, documentary photo",
        negative="overprocessed",
        seed=99,
        width=1024,
        height=1024,
        steps=30,
        cfg=4.5,
    )

    workflow = render_sdxl_workflow(recipe, checkpoint_name="sdxl.safetensors")

    sampler_inputs = workflow["3"]["inputs"]
    assert sampler_inputs["steps"] == 30
    assert sampler_inputs["cfg"] == 4.5
    assert sampler_inputs["sampler_name"] == "dpmpp_2m"
    assert sampler_inputs["scheduler"] == "karras"
    assert sampler_inputs["denoise"] == 1
    assert workflow["7"]["inputs"]["width"] == 1024
    assert workflow["7"]["inputs"]["height"] == 1024
    assert workflow["7"]["inputs"]["batch_size"] == 1
