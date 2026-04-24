"""Download VisDrone archives and prepare a local YOLO-formatted dataset."""

from __future__ import annotations

import argparse
import shutil
import ssl
import urllib.request
import zipfile
from pathlib import Path

import certifi

from prepare_visdrone import build_dataset_yaml, convert_split
from utils import dump_yaml, ensure_dir, resolve_from_root

DATASET_URLS = {
    "VisDrone2019-DET-train.zip": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip",
    "VisDrone2019-DET-val.zip": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset download and preparation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        default="data/raw",
        help="Directory where VisDrone archives and extracted folders are stored.",
    )
    parser.add_argument(
        "--output-root",
        default="data/visdrone_yolo",
        help="Directory where the YOLO-formatted dataset will be created.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download archives again even if they already exist.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images into the YOLO dataset instead of creating symlinks.",
    )
    return parser.parse_args()


def download_file(url: str, destination: Path) -> None:
    """Download one file from a URL to disk."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, context=ssl_context) as response, destination.open("wb") as file:
        shutil.copyfileobj(response, file)


def ensure_archive(archive_path: Path, url: str, force_download: bool) -> None:
    """Download the archive if it is missing or needs to be refreshed."""
    if archive_path.exists() and not force_download:
        print(f"archive already exists: {archive_path}")
        return

    print(f"downloading: {url}")
    download_file(url, archive_path)


def unpack_archive(archive_path: Path, raw_root: Path) -> None:
    """Extract one VisDrone archive into the raw dataset directory."""
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(raw_root)


def main() -> None:
    """Download, unpack and prepare the VisDrone train/val dataset."""
    args = parse_args()
    raw_root = ensure_dir(args.raw_root)
    output_root = ensure_dir(args.output_root)

    for archive_name, url in DATASET_URLS.items():
        archive_path = raw_root / archive_name
        ensure_archive(archive_path=archive_path, url=url, force_download=args.force_download)
        unpack_archive(archive_path=archive_path, raw_root=raw_root)

    total_images = 0
    total_boxes = 0
    for split_name in ("train", "val"):
        image_count, box_count = convert_split(
            raw_root=resolve_from_root(raw_root),
            output_root=output_root,
            split_name=split_name,
            copy_images=args.copy_images,
        )
        total_images += image_count
        total_boxes += box_count
        print(f"{split_name}: images={image_count}, boxes={box_count}")

    dataset_yaml_path = dump_yaml(build_dataset_yaml(output_root), output_root / "dataset.yaml")
    print(f"dataset yaml: {dataset_yaml_path}")
    print(f"total: images={total_images}, boxes={total_boxes}")


if __name__ == "__main__":
    main()
