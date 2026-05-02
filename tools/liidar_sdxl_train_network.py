from __future__ import annotations

import runpy
from pathlib import Path

import torch


def _patch_directml_rocm_distributed_cleanup() -> None:
    """Make accelerate cleanup tolerant of Windows ROCm torch builds.

    The current Windows ROCm torch preview exposes ``torch.distributed`` but
    does not expose ``is_initialized``. Accelerate calls that function at the
    end of training before sd-scripts writes the final LoRA, so a completed
    training run can crash at the finish line. In a single-GPU local run there
    is no distributed process group to destroy, so returning False is correct.
    """

    distributed = getattr(torch, "distributed", None)
    if distributed is not None and not hasattr(distributed, "is_initialized"):
        distributed.is_initialized = lambda: False  # type: ignore[attr-defined]


if __name__ == "__main__":
    _patch_directml_rocm_distributed_cleanup()
    script = Path(__file__).resolve().parent / "sd-scripts" / "sdxl_train_network.py"
    runpy.run_path(str(script), run_name="__main__")
