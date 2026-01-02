"""Update YOLO datasets by removing, remapping, and renaming classes."""
from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import matplotlib.pyplot as plt
import yaml


class YoloAutoProcessor:
    """Apply class removal, remapping, and renaming to a YOLO dataset."""

    def __init__(self, input_path: Path, destination_base: Path) -> None:
        self.src = input_path
        dataset_name = self.src.name
        self.dst_base = destination_base
        self.dst = self.dst_base / dataset_name
        self.src_yaml = self.src / "data.yaml"
        self.dst_yaml = self.dst / "data.yaml"
        self.current_names: Dict[int, str] = {}

    def load_original_names(self) -> bool:
        if not self.src_yaml.exists():
            print(f"⛔ ERROR: data.yaml not found in {self.src}")
            return False
        try:
            with self.src_yaml.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            names = data.get("names")
            if isinstance(names, list):
                self.current_names = {i: n for i, n in enumerate(names)}
            elif isinstance(names, dict):
                self.current_names = {int(k): v for k, v in names.items()}
            else:
                print("❌ Error: 'names' has an invalid format in data.yaml.")
                return False
            return True
        except Exception as exc:  # noqa: BLE001 - Provide user-friendly output
            print(f"❌ Error reading original YAML: {exc}")
            return False

    def validate_conflicts(self, ids_to_remove: Iterable[int], id_map: Dict[int, int]) -> bool:
        print("🔍 Checking ID conflicts...")
        final_state: Dict[int, int] = {}
        collision = False
        for original_id in self.current_names.keys():
            if original_id in ids_to_remove:
                continue
            new_id = id_map.get(original_id, original_id)
            if new_id in final_state:
                conflicting = final_state[new_id]
                print(
                    "⛔ CONFLICT: destination ID "
                    f"'{new_id}' would receive data from classes "
                    f"{conflicting} and {original_id}"
                )
                collision = True
            final_state[new_id] = original_id
        if collision:
            print("\n❌ ABORTED: ID conflicts detected.")
            return False
        return True

    def copy_dataset(self) -> bool:
        print(f"📦 Destination: {self.dst}")
        if not self.src.exists():
            print(f"❌ Source does not exist: {self.src}")
            return False
        if not self.dst_base.exists():
            os.makedirs(self.dst_base, exist_ok=True)
        try:
            shutil.copytree(self.src, self.dst, dirs_exist_ok=True)
            print("✅ Copy completed.")
            return True
        except Exception as exc:  # noqa: BLE001 - Provide user-friendly output
            print(f"❌ Error while copying: {exc}")
            return False

    def process_labels(self, remove_ids: Iterable[int], id_map: Dict[int, int]) -> None:
        print("\n🔄 Processing labels...")
        for split in ["train", "valid", "test"]:
            labels_dir = self.dst / split / "labels"
            if not labels_dir.exists():
                continue
            count = 0
            for label_file in labels_dir.glob("*.txt"):
                with label_file.open("r", encoding="utf-8") as f:
                    lines = f.readlines()
                new_lines = []
                changed = False
                for line in lines:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        old_id = int(parts[0])
                    except ValueError:
                        continue
                    if old_id in remove_ids:
                        changed = True
                        continue

                    rest_of_line = parts[1:]

                    if old_id in id_map:
                        new_id = str(id_map[old_id])
                        new_lines.append(f"{new_id} {' '.join(rest_of_line)}\n")
                        changed = True
                    else:
                        new_lines.append(line)
                if changed:
                    with label_file.open("w", encoding="utf-8") as f:
                        f.writelines(new_lines)
                    count += 1
            print(f"   └── {split}: {count} files updated.")

    def update_yaml(
        self,
        remove_ids: Iterable[int],
        id_map: Dict[int, int],
        text_rename: Dict[int, str],
    ) -> Dict[int, str]:
        print("\n📝 Computing new class names for data.yaml...")
        new_names_map: Dict[int, str] = {}
        for old_id, old_name in self.current_names.items():
            if old_id in remove_ids:
                continue
            final_id = id_map.get(old_id, old_id)
            final_name = text_rename.get(old_id, old_name)
            new_names_map[final_id] = final_name

        try:
            with self.dst_yaml.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            data["names"] = new_names_map
            data["nc"] = max(new_names_map.keys()) + 1 if new_names_map else 0
            with self.dst_yaml.open("w", encoding="utf-8") as f:
                yaml.dump(data, f, sort_keys=False, allow_unicode=True)
            return new_names_map
        except Exception as exc:  # noqa: BLE001 - Provide user-friendly output
            print(f"❌ YAML error: {exc}")
            return {}

    def print_comparison_report(
        self,
        remove_ids: Iterable[int],
        id_map: Dict[int, int],
        text_rename: Dict[int, str],
    ) -> None:
        print("\n" + "=" * 75)
        print(f"{'📊 FINAL REPORT':^75}")
        print("=" * 75)
        print(f"{'ORIGINAL':<35} | {'FINAL RESULT':<35}")
        print(f"{'ID':<5} {'NAME':<28} | {'ID':<5} {'NAME':<28}")
        print("-" * 75)
        for old_id in sorted(self.current_names.keys()):
            old_name = self.current_names[old_id]
            if old_id in remove_ids:
                print(f"{old_id:<5} {old_name:<28} | ❌ REMOVED")
            else:
                new_id = id_map.get(old_id, old_id)
                new_name = text_rename.get(old_id, old_name)
                print(f"{old_id:<5} {old_name:<28} | {new_id:<5} {new_name:<28}")
        print("=" * 75 + "\n")

    def visualize_samples(self, class_names_map: Dict[int, str]) -> None:
        print("🎨 Generating verification mosaic...")
        img_dir = self.dst / "train" / "images"
        lbl_dir = self.dst / "train" / "labels"

        if not img_dir.exists():
            img_dir = self.dst / "valid" / "images"
            lbl_dir = self.dst / "valid" / "labels"

        all_images = list(img_dir.glob("*.*"))
        valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}
        all_images = [img for img in all_images if img.suffix.lower() in valid_ext]

        if not all_images:
            print("⚠️ No images to show.")
            return

        samples = random.sample(all_images, min(3, len(all_images)))
        plt.figure(figsize=(15, 5))

        for i, img_path in enumerate(samples):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, _ = img.shape

            label_path = lbl_dir / (img_path.stem + ".txt")
            if label_path.exists():
                with label_path.open("r", encoding="utf-8") as f:
                    lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    try:
                        cls_id = int(parts[0])
                    except ValueError:
                        continue
                    all_coords = list(map(float, parts[1:]))
                    if len(all_coords) >= 4:
                        x_c, y_c, bw, bh = all_coords[:4]
                    else:
                        continue

                    x1, y1 = int((x_c - bw / 2) * w), int((y_c - bh / 2) * h)
                    x2, y2 = int((x_c + bw / 2) * w), int((y_c + bh / 2) * h)

                    color = (0, 255, 0)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    label_text = class_names_map.get(cls_id, str(cls_id))
                    cv2.putText(
                        img,
                        label_text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

            plt.subplot(1, 3, i + 1)
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"{img_path.name}")
        plt.tight_layout()
        plt.show()

    def run(
        self,
        remove: Iterable[int],
        remap: Dict[int, int],
        rename: Dict[int, str],
        visualize: bool = True,
    ) -> None:
        print("🚀 PROCESS START")
        if not self.load_original_names():
            return
        if not self.validate_conflicts(remove, remap):
            return
        if not self.copy_dataset():
            return
        self.process_labels(remove, remap)
        final_names = self.update_yaml(remove, remap, rename)
        self.print_comparison_report(remove, remap, rename)
        if final_names and visualize:
            self.visualize_samples(final_names)
        print("\n✨ DONE ✨")


def parse_remove(value: str | None) -> List[int]:
    if not value:
        return []
    return [int(x) for x in value.split(",") if x.strip()]


def parse_remap(value: str | None) -> Dict[int, int]:
    if not value:
        return {}
    mapping: Dict[int, int] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        source, target = item.split(":")
        mapping[int(source)] = int(target)
    return mapping


def parse_rename(value: str | None) -> Dict[int, str]:
    if not value:
        return {}
    mapping: Dict[int, str] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        source, target = item.split(":", 1)
        mapping[int(source)] = target
    return mapping


def load_config(path: Path) -> Tuple[Path, Path, List[int], Dict[int, int], Dict[int, str]]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    input_path = Path(data["input_path"])
    destination_base = Path(data["destination_base"])
    remove_ids = [int(x) for x in data.get("remove_ids", [])]
    remap_ids = {int(k): int(v) for k, v in (data.get("remap_ids") or {}).items()}
    rename_text = {int(k): str(v) for k, v in (data.get("rename_text") or {}).items()}
    return input_path, destination_base, remove_ids, remap_ids, rename_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update YOLO datasets by removing, remapping, and renaming classes."
    )
    parser.add_argument("--config", type=Path, help="YAML configuration file.")
    parser.add_argument("--input", type=Path, help="Source dataset path.")
    parser.add_argument("--destination", type=Path, help="Destination base folder.")
    parser.add_argument(
        "--remove",
        type=str,
        default=None,
        help="IDs to remove, comma-separated (e.g. 1,3,5).",
    )
    parser.add_argument(
        "--remap",
        type=str,
        default=None,
        help="ID remap pairs (e.g. 1:2,2:1).",
    )
    parser.add_argument(
        "--rename",
        type=str,
        default=None,
        help="Rename IDs (e.g. 0:Bird,1:Dust).",
    )
    parser.add_argument(
        "--no-visual",
        action="store_true",
        help="Disable sample visualization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config:
        input_path, destination_base, remove, remap, rename = load_config(args.config)
    else:
        if not args.input or not args.destination:
            raise SystemExit("You must provide --input and --destination or use --config.")
        input_path = args.input
        destination_base = args.destination
        remove = parse_remove(args.remove)
        remap = parse_remap(args.remap)
        rename = parse_rename(args.rename)

    proc = YoloAutoProcessor(input_path, destination_base)
    proc.run(remove, remap, rename, visualize=not args.no_visual)


if __name__ == "__main__":
    main()
