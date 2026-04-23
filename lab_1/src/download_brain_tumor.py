"""Download the Ultralytics brain-tumor dataset and create a local dataset.yaml."""

from __future__ import annotations

import argparse
import shutil
import ssl
import urllib.request
import zipfile
from pathlib import Path

import certifi

from utils import dump_yaml, ensure_dir, resolve_from_root

DATASET_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/brain-tumor.zip"
CLASS_NAMES = ["negative", "positive"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset download."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="data/brain_tumor",
        help="Path where the unpacked dataset will be stored.",
    )
    parser.add_argument(
        "--archive-path",
        default="data/raw/brain-tumor.zip",
        help="Path to the downloaded archive.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download the archive again even if it already exists.",
    )
    return parser.parse_args()


def download_file(url: str, destination: Path) -> None:
    """Download one file from a URL to disk."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, context=ssl_context) as response, destination.open("wb") as file:
        shutil.copyfileobj(response, file)


def find_dataset_root(output_root: Path) -> Path:
    """Locate the extracted dataset root that contains images and labels."""
    candidates = [output_root, output_root / "brain-tumor"]
    for candidate in candidates:
        if (candidate / "images").exists() and (candidate / "labels").exists():
            return candidate
    raise FileNotFoundError(f"Could not find dataset root inside {output_root}")


def build_dataset_yaml(dataset_root: Path) -> dict[str, object]:
    """Build a local YOLO dataset yaml for the downloaded dataset."""
    return {
        "path": str(dataset_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }


def main() -> None:
    """Download, unpack and configure the local brain-tumor dataset."""
    args = parse_args()
    output_root = ensure_dir(args.output_root)
    archive_path = resolve_from_root(args.archive_path)

    if args.force_download or not archive_path.exists():
        print(f"downloading: {DATASET_URL}")
        download_file(DATASET_URL, archive_path)
    else:
        print(f"archive already exists: {archive_path}")

    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(output_root)

    dataset_root = find_dataset_root(output_root)
    dataset_yaml_path = dump_yaml(build_dataset_yaml(dataset_root), dataset_root / "dataset.yaml")
    print(f"dataset root: {dataset_root}")
    print(f"dataset yaml: {dataset_yaml_path}")


if __name__ == "__main__":
    main()
