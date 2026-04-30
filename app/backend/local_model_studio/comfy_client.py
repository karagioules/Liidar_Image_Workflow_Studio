from __future__ import annotations

from typing import Any

import httpx


class ComfyClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8188", timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def is_available(self) -> bool:
        try:
            response = httpx.get(f"{self.base_url}/system_stats", timeout=self.timeout)
        except httpx.HTTPError:
            return False

        return response.status_code == 200

    def queue_prompt(self, workflow: dict[str, Any]) -> str:
        response = httpx.post(f"{self.base_url}/prompt", json={"prompt": workflow}, timeout=self.timeout)
        response.raise_for_status()

        prompt_id = response.json().get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise RuntimeError("ComfyUI response did not include a valid prompt_id.")

        return prompt_id
