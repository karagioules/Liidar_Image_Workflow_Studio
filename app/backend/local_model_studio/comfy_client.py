from __future__ import annotations

from typing import Any
from urllib.parse import urlencode

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
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = _comfy_error_detail(response)
            raise RuntimeError(f"ComfyUI rejected the workflow: {detail}") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError("ComfyUI returned an invalid response.") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("ComfyUI returned an invalid response.")

        prompt_id = payload.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise RuntimeError("ComfyUI response did not include a valid prompt_id.")

        return prompt_id

    def queue(self) -> dict[str, Any]:
        response = httpx.get(f"{self.base_url}/queue", timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("ComfyUI returned an invalid queue response.")
        return payload

    def history(self, prompt_id: str) -> dict[str, Any]:
        response = httpx.get(f"{self.base_url}/history/{prompt_id}", timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("ComfyUI returned an invalid history response.")
        return payload

    def image_bytes(self, *, filename: str, subfolder: str = "", image_type: str = "output") -> tuple[bytes, str]:
        params = urlencode({"filename": filename, "subfolder": subfolder, "type": image_type})
        response = httpx.get(f"{self.base_url}/view?{params}", timeout=self.timeout)
        response.raise_for_status()
        return response.content, response.headers.get("content-type", "image/png")


def _comfy_error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip() or f"{response.status_code} {response.reason_phrase}"
    if isinstance(payload, dict):
        for key in ("error", "detail", "message"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
            if isinstance(value, dict):
                message = value.get("message") or value.get("details") or value.get("type")
                if isinstance(message, str) and message:
                    return message
        node_errors = payload.get("node_errors")
        if isinstance(node_errors, dict) and node_errors:
            first = next(iter(node_errors.values()))
            if isinstance(first, dict):
                errors = first.get("errors")
                if isinstance(errors, list) and errors:
                    first_error = errors[0]
                    if isinstance(first_error, dict):
                        message = first_error.get("message")
                        if isinstance(message, str) and message:
                            return message
    return f"{response.status_code} {response.reason_phrase}"
