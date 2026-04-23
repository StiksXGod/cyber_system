"""Train a YOLO11 experiment from a YAML configuration."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from utils import dump_json, filter_none_values, load_yaml, metrics_to_summary, resolve_from_root, to_builtin, utc_now_iso


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the experiment YAML file.",
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
    required_fields = ("experiment_name", "dataset_yaml", "model", "train")
    missing = [field for field in required_fields if field not in config]
    if missing:
        raise ValueError(f"Missing required fields in config: {', '.join(missing)}")
    return config


def build_train_kwargs(config: dict[str, Any], device_override: str | None) -> dict[str, Any]:
    """Build kwargs passed to Ultralytics training."""
    project_dir = resolve_from_root(config.get("project", "runs/lab1"))
    train_config = dict(config["train"])
    kwargs = {
        **train_config,
        "data": str(resolve_from_root(config["dataset_yaml"])),
        "project": str(project_dir),
        "name": config["experiment_name"],
        "exist_ok": True,
        "plots": True,
    }
    if device_override is not None:
        kwargs["device"] = device_override
    return filter_none_values(kwargs)


def build_train_report(config: dict[str, Any], run_dir: Path, results: Any) -> dict[str, Any]:
    """Build a JSON-friendly training report."""
    metrics = getattr(results, "results_dict", {})
    best_weights = run_dir / "weights" / "best.pt"
    last_weights = run_dir / "weights" / "last.pt"
    return {
        "generated_at": utc_now_iso(),
        "experiment_name": config["experiment_name"],
        "description": config.get("description"),
        "business_context": config.get("business_context"),
        "model": config["model"],
        "dataset_yaml": str(resolve_from_root(config["dataset_yaml"])),
        "run_dir": str(run_dir),
        "best_weights": str(best_weights) if best_weights.exists() else None,
        "last_weights": str(last_weights) if last_weights.exists() else None,
        "summary": metrics_to_summary(metrics),
        "raw_metrics": to_builtin(metrics),
        "notes": config.get("notes", []),
        "hypotheses": config.get("hypotheses", []),
        "train_config": to_builtin(config.get("train", {})),
    }


def main() -> None:
    """Run training for one configured experiment."""
    args = parse_args()
    config = load_experiment_config(args.config)
    YOLO = require_yolo()

    project_dir = resolve_from_root(config.get("project", "runs/lab1"))
    run_dir = project_dir / config["experiment_name"]
    kwargs = build_train_kwargs(config, args.device)

    model = YOLO(config["model"])
    results = model.train(**kwargs)

    save_dir = Path(str(getattr(results, "save_dir", run_dir)))
    report = build_train_report(config=config, run_dir=save_dir, results=results)
    dump_json(report, save_dir / "metrics_train.json")
    print(f"training artifacts: {save_dir}")


if __name__ == "__main__":
    main()

