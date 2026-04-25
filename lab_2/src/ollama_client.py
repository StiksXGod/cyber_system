"""HTTP client helpers for working with a local Ollama server."""

from __future__ import annotations

from typing import Any

import requests
from requests import exceptions as requests_exceptions


def build_generate_payload(model: str, prompt: str, max_tokens: int = 128) -> dict[str, Any]:
    """Build a JSON payload for the Ollama generate endpoint."""
    return {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": max_tokens,
        },
    }


def generate_text(
    base_url: str,
    model: str,
    prompt: str,
    timeout: float = 180.0,
    max_tokens: int = 128,
    retries: int = 1,
) -> str:
    """Send one prompt to the local Ollama server and return the generated text."""
    last_error: Exception | None = None
    current_timeout = timeout
    for _ in range(retries + 1):
        try:
            response = requests.post(
                f"{base_url.rstrip('/')}/api/generate",
                json=build_generate_payload(model=model, prompt=prompt, max_tokens=max_tokens),
                timeout=(10.0, current_timeout),
            )
            response.raise_for_status()
            payload = response.json()
            return str(payload.get("response", "")).strip()
        except requests_exceptions.ReadTimeout as error:
            last_error = error
            current_timeout *= 2
    if last_error is not None:
        raise last_error
    raise RuntimeError("Unexpected error while requesting Ollama.")


def check_server(base_url: str, timeout: float = 10.0) -> None:
    """Check that the Ollama HTTP server is reachable."""
    response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
    response.raise_for_status()
