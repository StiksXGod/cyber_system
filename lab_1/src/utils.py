"""Common utility functions for the lab scripts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]


def resolve_from_root(path_like: str | Path) -> Path:
    """Return an absolute path relative to the project root."""
    path = Path(path_like)
    return path if path.is_absolute() else ROOT_DIR / path


def ensure_dir(path_like: str | Path) -> Path:
    """Create a directory if it does not exist and return its absolute path."""
    path = resolve_from_root(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path_like: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from disk."""
    path = resolve_from_root(path_like)
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML file {path} must contain a mapping.")
    return data


def dump_yaml(data: dict[str, Any], path_like: str | Path) -> Path:
    """Write a YAML mapping to disk."""
    path = resolve_from_root(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)
    return path


def load_json(path_like: str | Path) -> dict[str, Any] | None:
    """Load JSON from disk if the file exists."""
    path = resolve_from_root(path_like)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"JSON file {path} must contain an object.")
    return data


def dump_json(data: dict[str, Any], path_like: str | Path) -> Path:
    """Write JSON to disk with stable formatting."""
    path = resolve_from_root(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return path


def filter_none_values(options: dict[str, Any]) -> dict[str, Any]:
    """Drop dictionary items with None values."""
    return {key: value for key, value in options.items() if value is not None}


def to_builtin(value: Any) -> Any:
    """Convert numpy-like values to plain Python objects."""
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def metrics_to_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    """Keep only the most relevant detection metrics."""
    preferred_keys = (
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "fitness",
    )
    summary: dict[str, Any] = {}
    for key in preferred_keys:
        if key in metrics:
            summary[key] = to_builtin(metrics[key])
    return summary


def utc_now_iso() -> str:
    """Return the current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

