"""Evaluate a trained YOLO11 experiment on the validation split."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from utils import (
    dump_json,
    filter_none_values,
    load_yaml,
    metrics_to_summary,
    resolve_from_root,
    to_builtin,
    utc_now_iso,
)

DEFAULT_CLASS_NAMES = [
    "negative",
    "positive",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the experiment YAML file.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Optional explicit path to model weights. By default best.pt is used.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override, for example cpu, 0 or mps.",
    )
    return parser.parse_args()


def require_yolo():
    """Import Ultralytics lazily and fail with a helpful message if it is missing."""
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as error:
        raise SystemExit("Install dependencies first: pip install -r requirements.txt") from error
    return YOLO


def load_experiment_config(path_like: str | Path) -> dict[str, Any]:
    """Load an experiment configuration and validate required fields."""
    config = load_yaml(path_like)
    required_fields = ("experiment_name", "dataset_yaml", "model")
    missing = [field for field in required_fields if field not in config]
    if missing:
        raise ValueError(f"Missing required fields in config: {', '.join(missing)}")
    return config


def resolve_weights_path(config: dict[str, Any], explicit_weights: str | None) -> Path:
    """Resolve the model weights path."""
    if explicit_weights is not None:
        weights_path = resolve_from_root(explicit_weights)
    else:
        weights_path = (
            resolve_from_root(config.get("project", "runs/lab1"))
            / config["experiment_name"]
            / "weights"
            / "best.pt"
        )
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    return weights_path


def build_val_kwargs(config: dict[str, Any], device_override: str | None) -> dict[str, Any]:
    """Build kwargs passed to Ultralytics validation."""
    project_dir = resolve_from_root(config.get("project", "runs/lab1"))
    val_config = dict(config.get("val", {}))
    kwargs = {
        **val_config,
        "data": str(resolve_from_root(config["dataset_yaml"])),
        "project": str(project_dir),
        "name": f"{config['experiment_name']}_val",
        "exist_ok": True,
        "plots": True,
    }
    if device_override is not None:
        kwargs["device"] = device_override
    return filter_none_values(kwargs)


def extract_dataset_names(config: dict[str, Any]) -> list[str]:
    """Load dataset class names from the YOLO dataset YAML if it exists."""
    dataset_path = resolve_from_root(config["dataset_yaml"])
    if not dataset_path.exists():
        return list(DEFAULT_CLASS_NAMES)

    dataset_config = load_yaml(dataset_path)
    names = dataset_config.get("names", {})
    if isinstance(names, dict):
        ordered_indexes = sorted(int(index) for index in names.keys())
        return [str(names[str(index)] if str(index) in names else names[index]) for index in ordered_indexes]
    if isinstance(names, list):
        return [str(name) for name in names]
    return list(DEFAULT_CLASS_NAMES)


def extract_per_class_metrics(results: Any, dataset_names: list[str]) -> list[dict[str, Any]]:
    """Extract per-class metrics when they are exposed by Ultralytics."""
    box_metrics = getattr(results, "box", None)
    per_class: list[dict[str, Any]] = []

    maps: list[Any] = []
    all_ap: list[Any] = []
    if box_metrics is not None:
        try:
            maps = list(getattr(box_metrics, "maps", []) or [])
        except Exception:
            maps = []
        try:
            all_ap = list(getattr(box_metrics, "all_ap", []) or [])
        except Exception:
            all_ap = []

    for index, class_name in enumerate(dataset_names):
        row = {
            "class_id": index,
            "class_name": class_name,
            "precision": None,
            "recall": None,
            "map50": None,
            "map50_95": None,
        }

        if box_metrics is not None and hasattr(box_metrics, "class_result"):
            try:
                values = list(box_metrics.class_result(index))
                if len(values) >= 4:
                    row["precision"] = to_builtin(values[0])
                    row["recall"] = to_builtin(values[1])
                    row["map50"] = to_builtin(values[2])
                    row["map50_95"] = to_builtin(values[3])
            except Exception:
                pass

        if row["map50_95"] is None and index < len(maps):
            row["map50_95"] = to_builtin(maps[index])

        if row["map50"] is None and index < len(all_ap):
            try:
                class_ap = list(all_ap[index])
                if class_ap:
                    row["map50"] = to_builtin(class_ap[0])
            except Exception:
                pass

        per_class.append(row)

    return per_class


def build_eval_report(
    config: dict[str, Any],
    weights_path: Path,
    eval_dir: Path,
    results: Any,
) -> dict[str, Any]:
    """Build a JSON-friendly evaluation report."""
    metrics = getattr(results, "results_dict", {})
    dataset_names = extract_dataset_names(config)
    return {
        "generated_at": utc_now_iso(),
        "experiment_name": config["experiment_name"],
        "description": config.get("description"),
        "business_context": config.get("business_context"),
        "weights": str(weights_path),
        "eval_dir": str(eval_dir),
        "dataset_names": dataset_names,
        "summary": metrics_to_summary(metrics),
        "raw_metrics": to_builtin(metrics),
        "per_class": extract_per_class_metrics(results, dataset_names),
        "val_config": to_builtin(config.get("val", {})),
    }


def main() -> None:
    """Run evaluation for one configured experiment."""
    args = parse_args()
    config = load_experiment_config(args.config)
    weights_path = resolve_weights_path(config, args.weights)
    YOLO = require_yolo()

    kwargs = build_val_kwargs(config, args.device)
    model = YOLO(str(weights_path))
    results = model.val(**kwargs)

    project_dir = resolve_from_root(config.get("project", "runs/lab1"))
    eval_dir = Path(str(getattr(results, "save_dir", project_dir / f"{config['experiment_name']}_val")))
    report = build_eval_report(config=config, weights_path=weights_path, eval_dir=eval_dir, results=results)
    dump_json(report, eval_dir / "metrics_val.json")
    dump_json(report, project_dir / config["experiment_name"] / "metrics_val.json")
    print(f"evaluation artifacts: {eval_dir}")


if __name__ == "__main__":
    main()
