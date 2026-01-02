import os
import yaml
import glob
from pathlib import Path
from typing import Dict, List

class YoloValidator:
    def __init__(self, dataset_path: str):
        self.root = Path(dataset_path)
        self.yaml_path = self.root / 'data.yaml'
        self.classes_info = {}
        
    def check_yaml(self) -> bool:
        if not self.yaml_path.exists():
            print(f"❌ Error Crítico: No se encontró 'data.yaml' en {self.root}")
            return False

        try:
            with open(self.yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                
            names = data.get('names')
            if not names:
                print("❌ Error: 'names' no encontrado en data.yaml.")
                return False
                
            if isinstance(names, list):
                self.classes_info = {i: name for i, name in enumerate(names)}
            elif isinstance(names, dict):
                self.classes_info = {int(k): v for k, v in names.items()}
            
            print("✅ 'data.yaml' cargado correctamente.")
            return True
        except Exception as e:
            print(f"❌ Error leyendo YAML: {e}")
            return False

    def check_pairs(self, split: str) -> None:
        images_dir = self.root / split / 'images'
        labels_dir = self.root / split / 'labels'

        if not images_dir.exists():
            if split != 'test': print(f"⚠️ Aviso: No existe carpeta '{split}'")
            return

        img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
        img_files = {p.stem for p in images_dir.iterdir() if p.suffix.lower() in img_exts}
        lbl_files = {p.stem: p for p in labels_dir.iterdir() if p.suffix == '.txt'}

        missing_labels = img_files - lbl_files.keys()
        missing_images = lbl_files.keys() - img_files

        if not missing_labels and not missing_images:
            print(f"✅ Split '{split}': OK ({len(img_files)} pares).")
        else:
            if missing_labels: print(f"❌ Split '{split}': {len(missing_labels)} imágenes sin label.")
            if missing_images: print(f"❌ Split '{split}': {len(missing_images)} labels sin imagen.")
        
        self._validate_content(lbl_files, split)

    def _validate_content(self, lbl_files, split):
        errors = 0
        for lbl_path in lbl_files.values():
            try:
                with open(lbl_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts and int(parts[0]) not in self.classes_info:
                            print(f"❌ Error en {lbl_path.name}: Clase {parts[0]} desconocida.")
                            errors += 1
            except: pass
        
        if errors == 0: print(f"   └── Contenido '{split}' validado.")

    def print_classes(self):
        print("\n" + "="*40)
        print(f"📊 REPORTE DE CLASES")
        print("="*40)
        print(f"{'ID':<10} | {'NOMBRE':<20}")
        print("-" * 33)
        for cid in sorted(self.classes_info.keys()):
            print(f"{cid:<10} | {self.classes_info[cid]:<20}")
        print("="*40 + "\n")

    def run(self):
        print(f"🔍 Validando: {self.root}\n")
        if self.check_yaml():
            self.check_pairs('train')
            self.check_pairs('valid')
            self.check_pairs('test')
            self.print_classes()

# --- CONFIGURACIÓN ---
if __name__ == "__main__":
    
    # 👇 EDITA ESTA LÍNEA CON TU RUTA REAL 👇
    # Nota: La 'r' al principio es importante para rutas de Windows
    
    RUTA_DEL_DATASET = r"C:\Users\Usuario\Desktop\Yolo\dataset\Solar.v2i.yolov8" 

    # -------------------------------------
    
    if os.path.exists(RUTA_DEL_DATASET):
        validator = YoloValidator(RUTA_DEL_DATASET)
        validator.run()
    else:
        print(f"❌ LA CARPETA NO EXISTE: {RUTA_DEL_DATASET}")
        print("Por favor, edita la variable RUTA_DEL_DATASET al final del código.")