"""Utility functions for the second lab scripts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]


def resolve_from_root(path_like: str | Path) -> Path:
    """Return an absolute path relative to the project root."""
    path = Path(path_like)
    return path if path.is_absolute() else ROOT_DIR / path


def load_json_list(path_like: str | Path) -> list[Any]:
    """Load a JSON array from disk."""
    path = resolve_from_root(path_like)
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"JSON file {path} must contain a list.")
    return data


def dump_json(data: Any, path_like: str | Path) -> Path:
    """Write JSON to disk with stable formatting."""
    path = resolve_from_root(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return path


def dump_text(text: str, path_like: str | Path) -> Path:
    """Write UTF-8 text to disk."""
    path = resolve_from_root(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def now_iso() -> str:
    """Return the current local timestamp in ISO format without microseconds."""
    return datetime.now().replace(microsecond=0).isoformat()
