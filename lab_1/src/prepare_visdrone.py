"""Convert the VisDrone detection dataset to YOLO format."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image

from utils import dump_yaml, ensure_dir, resolve_from_root

CLASS_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]

RAW_SPLITS = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset preparation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        default="data/raw",
        help="Path to the directory with raw VisDrone folders.",
    )
    parser.add_argument(
        "--output-root",
        default="data/visdrone_yolo",
        help="Path where the YOLO-formatted dataset will be created.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images instead of creating symlinks.",
    )
    return parser.parse_args()


def iter_annotation_lines(annotation_path: Path) -> list[str]:
    """Read annotation lines from a VisDrone text file."""
    if not annotation_path.exists():
        return []
    with annotation_path.open("r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def clamp(value: float) -> float:
    """Clamp a floating-point value to the [0, 1] range."""
    return max(0.0, min(1.0, value))


def convert_annotation(annotation_path: Path, image_path: Path) -> list[str]:
    """Convert one VisDrone annotation file to YOLO rows."""
    with Image.open(image_path) as image:
        image_width, image_height = image.size

    rows: list[str] = []
    for raw_line in iter_annotation_lines(annotation_path):
        parts = [item.strip() for item in raw_line.split(",")]
        if len(parts) < 8:
            continue

        x, y, width, height, score, category, _, _ = (int(float(item)) for item in parts[:8])
        if score == 0:
            continue
        if category <= 0 or category > len(CLASS_NAMES):
            continue
        if width <= 0 or height <= 0:
            continue

        x_center = clamp((x + width / 2) / image_width)
        y_center = clamp((y + height / 2) / image_height)
        norm_width = clamp(width / image_width)
        norm_height = clamp(height / image_height)
        class_id = category - 1

        rows.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} "
            f"{norm_width:.6f} {norm_height:.6f}"
        )
    return rows


def link_or_copy_image(source: Path, destination: Path, copy_images: bool) -> None:
    """Place one image in the YOLO dataset using a copy or a symlink."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if copy_images:
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source.resolve())


def convert_split(
    raw_root: Path,
    output_root: Path,
    split_name: str,
    copy_images: bool,
) -> tuple[int, int]:
    """Convert one dataset split and return image and box counts."""
    split_root = raw_root / RAW_SPLITS[split_name]
    images_dir = split_root / "images"
    annotations_dir = split_root / "annotations"
    if not images_dir.exists() or not annotations_dir.exists():
        raise FileNotFoundError(
            f"Expected VisDrone split in {split_root}. "
            "Make sure the raw dataset is unpacked correctly."
        )

    output_images_dir = ensure_dir(output_root / "images" / split_name)
    output_labels_dir = ensure_dir(output_root / "labels" / split_name)

    image_count = 0
    box_count = 0
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue

        label_path = output_labels_dir / f"{image_path.stem}.txt"
        annotation_path = annotations_dir / f"{image_path.stem}.txt"
        yolo_rows = convert_annotation(annotation_path, image_path)
        with label_path.open("w", encoding="utf-8") as file:
            if yolo_rows:
                file.write("\n".join(yolo_rows))
                file.write("\n")

        link_or_copy_image(
            source=image_path,
            destination=output_images_dir / image_path.name,
            copy_images=copy_images,
        )
        image_count += 1
        box_count += len(yolo_rows)

    return image_count, box_count


def build_dataset_yaml(output_root: Path) -> dict[str, object]:
    """Build the Ultralytics dataset configuration for the converted dataset."""
    return {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }


def main() -> None:
    """Convert the raw VisDrone dataset to YOLO format."""
    args = parse_args()
    raw_root = resolve_from_root(args.raw_root)
    output_root = ensure_dir(args.output_root)

    total_images = 0
    total_boxes = 0
    for split_name in ("train", "val"):
        image_count, box_count = convert_split(
            raw_root=raw_root,
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
