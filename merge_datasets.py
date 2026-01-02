"""Merge multiple YOLO datasets into a unified dataset with remapped class IDs."""
from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm


class YoloUniversalMerger:
    """Merge multiple YOLO datasets with heterogeneous class IDs."""

    def __init__(
        self,
        root_folder: Path,
        output_folder: Path,
        split_ratio: Tuple[float, float, float] = (0.7, 0.2, 0.1),
        seed: int | None = None,
    ) -> None:
        self.root = root_folder
        self.output = output_folder
        self.split_ratio = split_ratio
        self.seed = seed
        self.global_class_map: Dict[str, int] = {}
        self.dataset_configs: List[Dict[str, object]] = []
        self.all_samples: List[Dict[str, object]] = []

    def analyze_classes(self) -> bool:
        print("🔍 1. Scanning class names across datasets...")
        candidates = list({x.parent for x in self.root.rglob("data.yaml")})
        if not candidates:
            print("❌ No datasets found (missing data.yaml).")
            return False

        unique_names = set()
        for ds_path in candidates:
            yaml_path = ds_path / "data.yaml"
            try:
                with yaml_path.open("r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                names = data.get("names")
                if not names:
                    print(f"⚠️ {ds_path.name}: 'names' is empty. Skipping.")
                    continue
                if isinstance(names, list):
                    local_map = {i: n for i, n in enumerate(names)}
                elif isinstance(names, dict):
                    local_map = {int(k): v for k, v in names.items()}
                else:
                    print(f"⚠️ {ds_path.name}: invalid 'names' format. Skipping.")
                    continue

                unique_names.update(local_map.values())
                self.dataset_configs.append({"path": ds_path, "local_map": local_map})
            except Exception as exc:  # noqa: BLE001 - Provide user-friendly output
                print(f"⚠️ Error reading {ds_path.name}: {exc}")

        sorted_names = sorted(list(unique_names))
        self.global_class_map = {name: idx for idx, name in enumerate(sorted_names)}

        print("\n🌍 UNIFIED CLASS MAP (Global ID):")
        print("-" * 40)
        for name, gid in self.global_class_map.items():
            print(f"   ID {gid}: {name}")
        print("-" * 40)
        return True

    def prepare_file_mapping(self) -> None:
        print("\n🧠 2. Building ID translations per dataset...")
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
        for config in self.dataset_configs:
            ds_path = config["path"]
            local_map = config["local_map"]
            id_translator = {}
            print(f"   📂 Processing '{ds_path.name}':")
            for local_id, name in local_map.items():
                global_id = self.global_class_map[name]
                id_translator[local_id] = global_id
                if local_id != global_id:
                    print(f"      └── Remapping class '{name}': {local_id} -> {global_id}")

            images: List[Path] = []
            for ext in valid_extensions:
                images.extend(list(ds_path.rglob(f"*{ext}")))

            for img_path in images:
                possible_lbl = img_path.with_suffix(".txt")
                if not possible_lbl.exists():
                    parts = list(img_path.parts)
                    if "images" in parts:
                        idx = parts.index("images")
                        parts[idx] = "labels"
                        possible_lbl = Path(*parts).with_suffix(".txt")

                if possible_lbl.exists():
                    self.all_samples.append(
                        {
                            "img": img_path,
                            "lbl": possible_lbl,
                            "translator": id_translator,
                            "prefix": ds_path.name,
                        }
                    )
        print(f"✅ Total samples found: {len(self.all_samples)}")
        if self.seed is not None:
            random.seed(self.seed)
        random.shuffle(self.all_samples)

    def execute_merge(self) -> None:
        if not self.all_samples:
            return
        print("\n🚀 3. Merging & splitting (rewriting labels)...")
        if self.output.exists():
            shutil.rmtree(self.output)

        total = len(self.all_samples)
        idx1 = int(total * self.split_ratio[0])
        idx2 = idx1 + int(total * self.split_ratio[1])

        data_splits = {
            "train": self.all_samples[:idx1],
            "valid": self.all_samples[idx1:idx2],
            "test": self.all_samples[idx2:],
        }

        for split, items in data_splits.items():
            img_dir = self.output / split / "images"
            lbl_dir = self.output / split / "labels"
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            for item in tqdm(items, desc=f"Creating {split}"):
                safe_name = f"{item['prefix']}_{item['img'].name}"
                safe_lbl_name = f"{item['prefix']}_{item['img'].stem}.txt"
                dst_img = img_dir / safe_name
                dst_lbl = lbl_dir / safe_lbl_name

                shutil.copy2(item["img"], dst_img)
                try:
                    with item["lbl"].open("r", encoding="utf-8") as f_in:
                        lines = f_in.readlines()
                    new_lines = []
                    for line in lines:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        try:
                            old_id = int(parts[0])
                        except ValueError:
                            continue
                        if old_id in item["translator"]:
                            new_id = item["translator"][old_id]
                            rest = " ".join(parts[1:])
                            new_lines.append(f"{new_id} {rest}\n")
                    with dst_lbl.open("w", encoding="utf-8") as f_out:
                        f_out.writelines(new_lines)
                except Exception as exc:  # noqa: BLE001 - Provide user-friendly output
                    print(f"Error processing {item['lbl']}: {exc}")

    def create_yaml(self) -> None:
        print("\n📝 4. Creating final data.yaml...")
        final_names = {v: k for k, v in self.global_class_map.items()}
        yaml_content = {
            "path": str(self.output.absolute()),
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": len(final_names),
            "names": final_names,
        }
        with (self.output / "data.yaml").open("w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, sort_keys=False, allow_unicode=True)
        print("✅ data.yaml created.")

    def visual_check(self) -> None:
        print("\n🎨 5. Building random verification mosaic...")
        img_dir = self.output / "train" / "images"
        lbl_dir = self.output / "train" / "labels"

        img_list = list(img_dir.glob("*"))
        valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        img_list = [x for x in img_list if x.suffix.lower() in valid_ext]

        if not img_list:
            print("⚠️ No images to visualize.")
            return

        samples = random.sample(img_list, min(4, len(img_list)))
        id_to_name = {v: k for k, v in self.global_class_map.items()}
        plt.figure(figsize=(15, 10))

        for i, img_p in enumerate(samples):
            img = cv2.imread(str(img_p))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, _ = img.shape
            lbl_p = lbl_dir / (img_p.stem + ".txt")
            found_classes = []

            if lbl_p.exists():
                with lbl_p.open("r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                cls_id = int(parts[0])
                            except ValueError:
                                continue
                            name = id_to_name.get(cls_id, f"ID_{cls_id}")
                            found_classes.append(name)
                            try:
                                cx, cy, bw, bh = map(float, parts[1:5])
                            except ValueError:
                                continue
                            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
                            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
                            color = (0, 255, 0)
                            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(
                                img,
                                name,
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                color,
                                2,
                            )

            if found_classes:
                unique_classes = list(set(found_classes))
                title_text = ", ".join(unique_classes[:3])
                if len(unique_classes) > 3:
                    title_text += "..."
            else:
                title_text = "No objects (background)"

            plt.subplot(2, 2, i + 1)
            plt.imshow(img)
            plt.axis("off")
            plt.title(title_text, fontsize=10)
        plt.tight_layout()
        plt.show()

    def plot_balance(self) -> None:
        print("\n📊 6. Plotting final class balance...")
        counts = Counter()
        for txt_file in self.output.rglob("*.txt"):
            if txt_file.name == "classes.txt":
                continue
            with txt_file.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        cid = int(line.split()[0])
                        counts[cid] += 1
                    except ValueError:
                        continue
        if not counts:
            return
        names, values = [], []
        id_to_name = {v: k for k, v in self.global_class_map.items()}
        for cid in sorted(id_to_name.keys()):
            names.append(id_to_name[cid])
            values.append(counts[cid])
        plt.figure(figsize=(12, 6))
        bars = plt.bar(names, values, color="skyblue")
        plt.title("Final Object Distribution")
        plt.xticks(rotation=45)
        plt.ylabel("Count")
        for bar in bars:
            yval = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                yval,
                int(yval),
                ha="center",
                va="bottom",
            )
        plt.tight_layout()
        plt.show()

    def run(self, visual_check: bool = True, plot_balance: bool = True) -> None:
        print("🛠️  STARTING HETEROGENEOUS DATASET MERGE V4")
        if self.analyze_classes():
            self.prepare_file_mapping()
            self.execute_merge()
            self.create_yaml()
            if visual_check:
                self.visual_check()
            if plot_balance:
                self.plot_balance()
            print("\n✅ PROCESS COMPLETED.")


def parse_ratio(value: str) -> Tuple[float, float, float]:
    parts = [float(x) for x in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Ratio must have 3 values: train,val,test.")
    return parts[0], parts[1], parts[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple YOLO datasets.")
    parser.add_argument("input_root", type=Path, help="Root folder containing datasets.")
    parser.add_argument("output_dir", type=Path, help="Destination for merged dataset.")
    parser.add_argument(
        "--ratio",
        type=parse_ratio,
        default=(0.8, 0.1, 0.1),
        help="Split ratio train,valid,test (e.g. 0.8,0.1,0.1).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducible shuffle.")
    parser.add_argument(
        "--no-visual",
        action="store_true",
        help="Disable verification mosaic.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable class balance plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merger = YoloUniversalMerger(args.input_root, args.output_dir, args.ratio, args.seed)
    merger.run(visual_check=not args.no_visual, plot_balance=not args.no_plot)


if __name__ == "__main__":
    main()
