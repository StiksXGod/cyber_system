"""Run inference for one YOLO11 experiment configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils import filter_none_values, load_yaml, resolve_from_root


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for prediction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the experiment YAML file.",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Image, video or directory to process.",
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
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Save YOLO-format predictions alongside rendered images.",
    )
    return parser.parse_args()


def require_yolo():
    """Import Ultralytics lazily and fail with a helpful message if it is missing."""
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as error:
        raise SystemExit("Install dependencies first: pip install -r requirements.txt") from error
    return YOLO


def resolve_weights_path(config: dict[str, object], explicit_weights: str | None) -> Path:
    """Resolve the model weights path."""
    if explicit_weights is not None:
        weights_path = resolve_from_root(explicit_weights)
    else:
        weights_path = (
            resolve_from_root(config.get("project", "runs/lab1"))
            / str(config["experiment_name"])
            / "weights"
            / "best.pt"
        )
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    return weights_path


def resolve_source(source: str) -> str:
    """Resolve the inference source path when it is local."""
    if "://" in source:
        return source
    return str(resolve_from_root(source))


def main() -> None:
    """Run model inference and save rendered predictions."""
    args = parse_args()
    config = load_yaml(args.config)
    weights_path = resolve_weights_path(config, args.weights)
    predict_config = dict(config.get("predict", {}))
    project_dir = resolve_from_root(config.get("project", "runs/lab1"))

    kwargs = {
        **predict_config,
        "source": resolve_source(args.source),
        "project": str(project_dir),
        "name": f"{config['experiment_name']}_predict",
        "exist_ok": True,
        "save": True,
        "save_txt": args.save_txt,
    }
    if args.device is not None:
        kwargs["device"] = args.device

    YOLO = require_yolo()
    model = YOLO(str(weights_path))
    results = model.predict(**filter_none_values(kwargs))
    save_dir = Path(str(results[0].save_dir)) if results else project_dir / f"{config['experiment_name']}_predict"
    print(f"prediction artifacts: {save_dir}")


if __name__ == "__main__":
    main()

