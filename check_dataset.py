"""Utilities for validating YOLO dataset structure and labels."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable

import yaml


class YoloValidator:
    """Validate a YOLO dataset directory and its annotations."""

    def __init__(self, dataset_path: Path) -> None:
        self.root = dataset_path
        self.yaml_path = self.root / "data.yaml"
        self.classes_info: Dict[int, str] = {}
        self._errors = 0

    def check_yaml(self) -> bool:
        if not self.yaml_path.exists():
            print(f"❌ Critical error: 'data.yaml' not found in {self.root}")
            self._errors += 1
            return False

        try:
            with self.yaml_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            names = data.get("names")
            if not names:
                print("❌ Error: 'names' missing from data.yaml.")
                self._errors += 1
                return False

            if isinstance(names, list):
                self.classes_info = {i: name for i, name in enumerate(names)}
            elif isinstance(names, dict):
                self.classes_info = {int(k): v for k, v in names.items()}
            else:
                print("❌ Error: 'names' has an invalid format in data.yaml.")
                self._errors += 1
                return False

            for class_id, class_name in self.classes_info.items():
                expected = f"Class {class_id}"
                if class_name != expected:
                    print(
                        "❌ Error: class name mismatch in data.yaml. "
                        f"Expected '{expected}' but found '{class_name}'."
                    )
                    self._errors += 1

            print("✅ 'data.yaml' loaded successfully.")
            return self._errors == 0
        except Exception as exc:  # noqa: BLE001 - Provide user-friendly output
            print(f"❌ Error reading YAML: {exc}")
            self._errors += 1
            return False

    def check_pairs(self, split: str) -> None:
        images_dir = self.root / split / "images"
        labels_dir = self.root / split / "labels"

        if not images_dir.exists():
            if split != "test":
                print(f"⚠️ Warning: '{split}' folder not found.")
            return

        if not labels_dir.exists():
            print(f"⚠️ Warning: labels folder missing for '{split}'.")
            return

        img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        img_files = {p.stem for p in images_dir.iterdir() if p.suffix.lower() in img_exts}
        lbl_files = {p.stem: p for p in labels_dir.iterdir() if p.suffix == ".txt"}

        missing_labels = img_files - lbl_files.keys()
        missing_images = lbl_files.keys() - img_files

        if not missing_labels and not missing_images:
            print(f"✅ Split '{split}': OK ({len(img_files)} pairs).")
        else:
            if missing_labels:
                print(f"❌ Split '{split}': {len(missing_labels)} images without labels.")
                self._errors += len(missing_labels)
            if missing_images:
                print(f"❌ Split '{split}': {len(missing_images)} labels without images.")
                self._errors += len(missing_images)

        self._validate_content(lbl_files, split)

    def _validate_content(self, lbl_files: Dict[str, Path], split: str) -> None:
        errors = 0
        for lbl_path in lbl_files.values():
            try:
                with lbl_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        try:
                            class_id = int(parts[0])
                        except ValueError:
                            errors += 1
                            print(f"❌ Error in {lbl_path.name}: invalid class ID '{parts[0]}'.")
                            continue
                        if class_id not in self.classes_info:
                            print(f"❌ Error in {lbl_path.name}: unknown class {class_id}.")
                            errors += 1
                        if len(parts) < 5:
                            print(
                                f"❌ Error in {lbl_path.name}: incomplete line '{line.strip()}'."
                            )
                            errors += 1
            except OSError as exc:
                errors += 1
                print(f"❌ Error reading {lbl_path.name}: {exc}")

        self._errors += errors
        if errors == 0:
            print(f"   └── '{split}' content validated.")

    def print_classes(self) -> None:
        print("\n" + "=" * 40)
        print("📊 CLASS REPORT")
        print("=" * 40)
        print(f"{'ID':<10} | {'NAME':<20}")
        print("-" * 33)
        for cid in sorted(self.classes_info.keys()):
            print(f"{cid:<10} | {self.classes_info[cid]:<20}")
        print("=" * 40 + "\n")

    def run(self, splits: Iterable[str]) -> bool:
        print(f"🔍 Validating: {self.root}\n")
        if not self.check_yaml():
            return False

        for split in splits:
            self.check_pairs(split)
        self.print_classes()
        return self._errors == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO dataset validator.")
    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to the dataset containing data.yaml and train/val/test folders.",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["train", "val", "test"],
        help="Splits to validate (default: train val test).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validator = YoloValidator(args.dataset)
    success = validator.run(args.splits)
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
