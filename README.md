# YOLO Dataset Toolkit

Tools to **validate**, **merge**, and **relabel** YOLO-format datasets.

Includes three standalone scripts:

- `check_dataset.py` → validates structure, image/label pairs, and declared classes.
- `merge_datasets.py` → merges multiple YOLO datasets into one with unified class IDs.
- `new_classes.py` → removes, remaps, and renames classes in an existing dataset.

## Requirements

- Python 3.10+
- Dependencies:

```bash
pip install pyyaml opencv-python matplotlib tqdm
```

> `check_dataset.py` only needs `pyyaml`. The other scripts use `opencv`, `matplotlib`, and `tqdm`.

## 1) Validate a dataset

```bash
python check_dataset.py /path/to/my_dataset
```

Optionally specify splits to validate:

```bash
python check_dataset.py /path/to/my_dataset --splits train valid
```

## 2) Merge multiple datasets

The root folder should contain multiple YOLO datasets (each with its own `data.yaml`).

```bash
python merge_datasets.py /path/to/datasets /path/to/output --ratio 0.8,0.1,0.1
```

Useful options:

- `--seed 123` → reproducible shuffle.
- `--no-visual` → disable verification mosaic.
- `--no-plot` → disable class balance chart.

Output: a unified dataset with `data.yaml` and `train/valid/test` folders.

## 3) Modify classes in a dataset

You can use CLI flags or a YAML configuration file.

### Option A: Quick CLI

```bash
python new_classes.py \
  --input /path/to/dataset \
  --destination /path/to/output \
  --remove 3,5 \
  --remap 1:2,2:1 \
  --rename 0:Bird_dropping,1:Dust,2:Mechanical_damage
```

### Option B: YAML config

Create `config.yaml`:

```yaml
input_path: /path/to/dataset
destination_base: /path/to/output
remove_ids: [3]
remap_ids:
  1: 2
  2: 1
rename_text:
  0: Bird_dropping
  1: Dust
  2: Mechanical_damage
```

Then run:

```bash
python new_classes.py --config config.yaml
```

Useful options:

- `--no-visual` → disable sample visualization.

## Expected YOLO layout

```
my_dataset/
  data.yaml
  train/
    images/
    labels/
  valid/
    images/
    labels/
  test/
    images/
    labels/
```

## Notes

- `new_classes.py` works on a **copy** of your dataset, leaving the original untouched.
- `merge_datasets.py` removes the output directory if it exists to avoid accidental mixing.
