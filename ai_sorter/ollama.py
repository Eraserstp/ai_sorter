from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise OllamaError(f"Ollama request failed: {exc}") from exc

    def list_models(self) -> list[str]:
        data = self._request("GET", "/api/tags")
        return [model.get("name", "") for model in data.get("models", []) if model.get("name")]

    def generate(self, model: str, prompt: str, images: list[Path] | None = None, json_response: bool = False) -> str:
        if not model:
            raise OllamaError("Model is not selected")
        encoded_images = []
        for image in images or []:
            encoded_images.append(base64.b64encode(image.read_bytes()).decode("ascii"))
        payload: dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
        if encoded_images:
            payload["images"] = encoded_images
        if json_response:
            payload["format"] = "json"
        data = self._request("POST", "/api/generate", payload)
        return str(data.get("response", "")).strip()
