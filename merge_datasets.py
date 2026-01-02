import os
import shutil
import yaml
import random
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
from tqdm import tqdm

class YoloUniversalMerger:
    def __init__(self, root_folder: str, output_folder: str, split_ratio: Tuple[float, float, float] = (0.7, 0.2, 0.1)):
        self.root = Path(root_folder)
        self.output = Path(output_folder)
        self.split_ratio = split_ratio
        self.global_class_map = {}
        self.dataset_configs = []
        self.all_samples = []

    def analyze_classes(self):
        print("🔍 1. Analizando clases en todos los datasets...")
        candidates = list(set([x.parent for x in self.root.rglob('data.yaml')]))
        if not candidates:
            print("❌ No se encontraron datasets (falta data.yaml).")
            return False

        unique_names = set()
        for ds_path in candidates:
            yaml_path = ds_path / 'data.yaml'
            try:
                with open(yaml_path, 'r') as f: data = yaml.safe_load(f)
                names = data.get('names')
                local_map = {}
                if isinstance(names, list):
                    local_map = {i: n for i, n in enumerate(names)}
                elif isinstance(names, dict):
                    local_map = {int(k): v for k, v in names.items()}
                
                for name in local_map.values(): unique_names.add(name)
                self.dataset_configs.append({'path': ds_path, 'local_map': local_map})
            except Exception as e: print(f"⚠️ Error leyendo {ds_path.name}: {e}")

        sorted_names = sorted(list(unique_names))
        self.global_class_map = {name: idx for idx, name in enumerate(sorted_names)}
        
        print("\n🌍 MAPA DE CLASES UNIFICADO (Global ID):")
        print("-" * 40)
        for name, gid in self.global_class_map.items(): print(f"   ID {gid}: {name}")
        print("-" * 40)
        return True

    def prepare_file_mapping(self):
        print("\n🧠 2. Calculando traducciones de IDs por dataset...")
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
        for config in self.dataset_configs:
            ds_path = config['path']
            local_map = config['local_map']
            id_translator = {}
            print(f"   📂 Procesando '{ds_path.name}':")
            for local_id, name in local_map.items():
                global_id = self.global_class_map[name]
                id_translator[local_id] = global_id
                if local_id != global_id:
                    print(f"      └── Remapeando clase '{name}': {local_id} -> {global_id}")
            
            images = []
            for ext in valid_extensions: images.extend(list(ds_path.rglob(f'*{ext}')))
            
            for img_path in images:
                possible_lbl = img_path.with_suffix('.txt')
                if not possible_lbl.exists():
                    parts = list(img_path.parts)
                    if 'images' in parts:
                        idx = parts.index('images')
                        parts[idx] = 'labels'
                        possible_lbl = Path(*parts).with_suffix('.txt')
                
                if possible_lbl.exists():
                    self.all_samples.append({
                        'img': img_path, 'lbl': possible_lbl,
                        'translator': id_translator, 'prefix': ds_path.name
                    })
        print(f"✅ Total de muestras encontradas: {len(self.all_samples)}")
        random.shuffle(self.all_samples)

    def execute_merge(self):
        if not self.all_samples: return
        print("\n🚀 3. Ejecutando Fusión y Split (Reescribiendo etiquetas)...")
        if self.output.exists(): shutil.rmtree(self.output)
            
        total = len(self.all_samples)
        idx1 = int(total * self.split_ratio[0])
        idx2 = idx1 + int(total * self.split_ratio[1])
        
        data_splits = {'train': self.all_samples[:idx1], 'valid': self.all_samples[idx1:idx2], 'test':  self.all_samples[idx2:]}

        for split, items in data_splits.items():
            img_dir = self.output / split / 'images'
            lbl_dir = self.output / split / 'labels'
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lbl_dir, exist_ok=True)
            
            for item in tqdm(items, desc=f"Generando {split}"):
                safe_name = f"{item['prefix']}_{item['img'].name}"
                safe_lbl_name = f"{item['prefix']}_{item['img'].stem}.txt"
                dst_img = img_dir / safe_name
                dst_lbl = lbl_dir / safe_lbl_name
                
                shutil.copy2(item['img'], dst_img)
                try:
                    with open(item['lbl'], 'r') as f_in: lines = f_in.readlines()
                    new_lines = []
                    for line in lines:
                        parts = line.strip().split()
                        if not parts: continue
                        try:
                            old_id = int(parts[0])
                            if old_id in item['translator']:
                                new_id = item['translator'][old_id]
                                rest = " ".join(parts[1:])
                                new_lines.append(f"{new_id} {rest}\n")
                        except ValueError: pass
                    with open(dst_lbl, 'w') as f_out: f_out.writelines(new_lines)
                except Exception as e: print(f"Error procesando {item['lbl']}: {e}")

    def create_yaml(self):
        print("\n📝 4. Creando data.yaml final...")
        final_names = {v: k for k, v in self.global_class_map.items()}
        yaml_content = {
            'path': str(self.output.absolute()),
            'train': 'train/images', 'val': 'valid/images', 'test': 'test/images',
            'nc': len(final_names), 'names': final_names
        }
        with open(self.output / 'data.yaml', 'w') as f: yaml.dump(yaml_content, f, sort_keys=False)
        print("✅ data.yaml generado.")

    def visual_check(self):
        print("\n🎨 5. Generando Mosaico de Verificación (Aleatorio)...")
        img_dir = self.output / 'train' / 'images'
        lbl_dir = self.output / 'train' / 'labels'
        
        img_list = list(img_dir.glob('*'))
        valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        img_list = [x for x in img_list if x.suffix.lower() in valid_ext]
        
        if not img_list: 
            print("⚠️ No hay imágenes para visualizar.")
            return
        
        # --- AQUÍ ESTÁ LA SELECCIÓN ALEATORIA ---
        samples = random.sample(img_list, min(4, len(img_list)))
        # ----------------------------------------
        
        id_to_name = {v: k for k, v in self.global_class_map.items()}
        plt.figure(figsize=(15, 10))
        
        for i, img_p in enumerate(samples):
            img = cv2.imread(str(img_p))
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, _ = img.shape
            lbl_p = lbl_dir / (img_p.stem + ".txt")
            found_classes = []
            
            if lbl_p.exists():
                with open(lbl_p, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                cls_id = int(parts[0])
                                name = id_to_name.get(cls_id, f"ID_{cls_id}")
                                found_classes.append(name)
                                cx, cy, bw, bh = map(float, parts[1:5])
                                x1, y1 = int((cx - bw/2)*w), int((cy - bh/2)*h)
                                x2, y2 = int((cx + bw/2)*w), int((cy + bh/2)*h)
                                color = (0, 255, 0)
                                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                                cv2.putText(img, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            except ValueError: continue

            if found_classes:
                unique_classes = list(set(found_classes))
                title_text = ", ".join(unique_classes[:3])
                if len(unique_classes) > 3: title_text += "..."
            else: title_text = "Sin objetos (Background)"

            plt.subplot(2, 2, i+1)
            plt.imshow(img)
            plt.axis('off')
            plt.title(title_text, fontsize=10)
        plt.tight_layout()
        plt.show()

    def plot_balance(self):
        print("\n📊 6. Analizando Balance de Clases Final...")
        counts = Counter()
        for txt_file in self.output.rglob('*.txt'):
            if txt_file.name == 'classes.txt': continue
            with open(txt_file, 'r') as f:
                for line in f:
                    try:
                        cid = int(line.split()[0])
                        counts[cid] += 1
                    except: pass
        if not counts: return
        names, values = [], []
        id_to_name = {v: k for k, v in self.global_class_map.items()}
        for cid in sorted(id_to_name.keys()):
            names.append(id_to_name[cid])
            values.append(counts[cid])
        plt.figure(figsize=(12, 6))
        bars = plt.bar(names, values, color='skyblue')
        plt.title('Distribución Final de Objetos')
        plt.xticks(rotation=45)
        plt.ylabel('Cantidad')
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')
        plt.tight_layout()
        plt.show()

    def run(self):
        print("🛠️  INICIANDO FUSIÓN DE DATASETS HETEROGÉNEOS V4")
        if self.analyze_classes():
            self.prepare_file_mapping()
            self.execute_merge()
            self.create_yaml()
            self.visual_check()
            self.plot_balance()
            print("\n✅ PROCESO COMPLETADO.")

if __name__ == "__main__":
    INPUT_ROOT = r"C:\Users\Usuario\Desktop\Yolo\dataset_modificado"
    OUTPUT_DIR = r"C:\Users\Usuario\Desktop\Yolo\dataset_mergeado"
    RATIO = (0.8, 0.1, 0.1)
    merger = YoloUniversalMerger(INPUT_ROOT, OUTPUT_DIR, RATIO)
    merger.run()