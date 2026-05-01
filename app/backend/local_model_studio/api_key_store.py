from __future__ import annotations

import json
from pathlib import Path

from local_model_studio.schemas import ApiKeyPayload, ApiKeyStatus


DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
RETIRED_MODEL_REPLACEMENTS = {
    "claude-3-haiku-20240307": DEFAULT_ANTHROPIC_MODEL,
    "claude-3-5-haiku-20241022": DEFAULT_ANTHROPIC_MODEL,
}


class ApiKeyStore:
    def __init__(self, root: Path) -> None:
        self.path = root / "config" / "secrets" / "anthropic.json"

    def status(self) -> ApiKeyStatus:
        data = self._read()
        saved = bool(data.get("api_key"))
        stored_model = str(data.get("model") or DEFAULT_ANTHROPIC_MODEL)
        active_model = _active_model(stored_model)
        if saved and active_model != stored_model:
            self._write(str(data["api_key"]), active_model)
        return ApiKeyStatus(
            provider="anthropic",
            saved=saved,
            model=active_model if saved else DEFAULT_ANTHROPIC_MODEL,
        )

    def save(self, payload: ApiKeyPayload) -> ApiKeyStatus:
        self._write(payload.api_key.strip(), _active_model(payload.model.strip() or DEFAULT_ANTHROPIC_MODEL))
        return self.status()

    def delete(self) -> ApiKeyStatus:
        if self.path.exists():
            self.path.unlink()
        return self.status()

    def api_key(self) -> str | None:
        value = self._read().get("api_key")
        return str(value) if value else None

    def model(self) -> str:
        return self.status().model

    def _read(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def _write(self, api_key: str, model: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"api_key": api_key, "model": model}, indent=2),
            encoding="utf-8",
        )


def _active_model(model: str) -> str:
    return RETIRED_MODEL_REPLACEMENTS.get(model, model)
