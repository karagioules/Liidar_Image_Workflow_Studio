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
        return ApiKeyStatus(
            provider="anthropic",
            saved=saved,
            model=_active_model(str(data.get("model") or DEFAULT_ANTHROPIC_MODEL)) if saved else DEFAULT_ANTHROPIC_MODEL,
        )

    def save(self, payload: ApiKeyPayload) -> ApiKeyStatus:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"api_key": payload.api_key.strip(), "model": _active_model(payload.model.strip() or DEFAULT_ANTHROPIC_MODEL)}, indent=2),
            encoding="utf-8",
        )
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


def _active_model(model: str) -> str:
    return RETIRED_MODEL_REPLACEMENTS.get(model, model)
