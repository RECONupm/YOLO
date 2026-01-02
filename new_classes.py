import os
import yaml
import shutil
import random
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict

class YoloAutoProcessor:
    def __init__(self, input_path: str, destination_base: str):
        self.src = Path(input_path)
        dataset_name = self.src.name 
        self.dst_base = Path(destination_base)
        self.dst = self.dst_base / dataset_name
        self.src_yaml = self.src / 'data.yaml'
        self.dst_yaml = self.dst / 'data.yaml'
        self.current_names = {} 

    def load_original_names(self) -> bool:
        if not self.src_yaml.exists():
            print(f"⛔ ERROR: No se encuentra data.yaml en {self.src}")
            return False
        try:
            with open(self.src_yaml, 'r') as f:
                data = yaml.safe_load(f)
            names = data.get('names')
            if isinstance(names, list):
                self.current_names = {i: n for i, n in enumerate(names)}
            elif isinstance(names, dict):
                self.current_names = {int(k): v for k, v in names.items()}
            return True
        except Exception as e:
            print(f"❌ Error leyendo YAML original: {e}")
            return False

    def validate_conflicts(self, ids_to_remove, id_map) -> bool:
        print("🔍 Verificando conflictos de IDs...")
        final_state = {}
        collision = False
        for original_id in self.current_names.keys():
            if original_id in ids_to_remove: continue
            new_id = id_map.get(original_id, original_id)
            if new_id in final_state:
                conflicting = final_state[new_id]
                print(f"⛔ CONFLICTO: ID Destino '{new_id}' recibiría datos de clases originales {conflicting} y {original_id}")
                collision = True
            final_state[new_id] = original_id
        if collision:
            print("\n❌ ABORTADO: Conflicto de IDs detectado.")
            return False
        return True

    def copy_dataset(self):
        print(f"📦 Destino: {self.dst}")
        if not self.src.exists():
            print(f"❌ Origen no existe: {self.src}")
            return False
        if not self.dst_base.exists():
            os.makedirs(self.dst_base, exist_ok=True)
        try:
            shutil.copytree(self.src, self.dst, dirs_exist_ok=True)
            print("✅ Copia completada.")
            return True
        except Exception as e:
            print(f"❌ Error al copiar: {e}")
            return False

    def process_labels(self, remove_ids, id_map):
        print("\n🔄 Procesando etiquetas...")
        for split in ['train', 'valid', 'test']:
            labels_dir = self.dst / split / 'labels'
            if not labels_dir.exists(): continue
            count = 0
            for label_file in labels_dir.glob('*.txt'):
                with open(label_file, 'r') as f: lines = f.readlines()
                new_lines = []
                changed = False
                for line in lines:
                    parts = line.strip().split()
                    if not parts: continue
                    try:
                        old_id = int(parts[0])
                        if old_id in remove_ids: changed = True; continue
                        
                        # Mantenemos el resto de la línea intacta (coordenadas, etc.)
                        rest_of_line = parts[1:]
                        
                        if old_id in id_map:
                            new_id = str(id_map[old_id])
                            # Reconstruimos la línea con el nuevo ID y el resto de datos
                            new_lines.append(f"{new_id} {' '.join(rest_of_line)}\n")
                            changed = True
                        else: 
                            new_lines.append(line)
                    except: continue
                if changed:
                    with open(label_file, 'w') as f: f.writelines(new_lines)
                    count += 1
            print(f"   └── {split}: {count} archivos actualizados.")

    def update_yaml(self, remove_ids, id_map, text_rename):
        print("\n📝 Calculando nuevos nombres para data.yaml...")
        new_names_map = {}
        for old_id, old_name in self.current_names.items():
            if old_id in remove_ids: continue
            final_id = id_map.get(old_id, old_id)
            final_name = text_rename.get(old_id, old_name)
            new_names_map[final_id] = final_name

        try:
            with open(self.dst_yaml, 'r') as f: data = yaml.safe_load(f)
            data['names'] = new_names_map
            data['nc'] = max(new_names_map.keys()) + 1 if new_names_map else 0
            with open(self.dst_yaml, 'w') as f: yaml.dump(data, f, sort_keys=False)
            return new_names_map 
        except Exception as e: 
            print(f"❌ Error YAML: {e}")
            return {}

    def print_comparison_report(self, remove_ids, id_map, text_rename):
        print("\n" + "="*75)
        print(f"{'📊 REPORTE FINAL':^75}")
        print("="*75)
        print(f"{'ORIGINAL':<35} | {'RESULTADO FINAL':<35}")
        print(f"{'ID':<5} {'NOMBRE':<28} | {'ID':<5} {'NOMBRE':<28}")
        print("-" * 75)
        for old_id in sorted(self.current_names.keys()):
            old_name = self.current_names[old_id]
            if old_id in remove_ids:
                print(f"{old_id:<5} {old_name:<28} | ❌ BORRADA")
            else:
                new_id = id_map.get(old_id, old_id)
                new_name = text_rename.get(old_id, old_name)
                print(f"{old_id:<5} {old_name:<28} | {new_id:<5} {new_name:<28}")
        print("="*75 + "\n")

    def visualize_samples(self, class_names_map: Dict[int, str]):
        print("🎨 Generando mosaico de verificación...")
        img_dir = self.dst / 'train' / 'images'
        lbl_dir = self.dst / 'train' / 'labels'
        
        if not img_dir.exists():
            img_dir = self.dst / 'valid' / 'images' 
            lbl_dir = self.dst / 'valid' / 'labels'

        all_images = list(img_dir.glob('*.*'))
        valid_ext = {'.jpg', '.jpeg', '.png', '.bmp'}
        all_images = [img for img in all_images if img.suffix.lower() in valid_ext]

        if not all_images:
            print("⚠️ No hay imágenes para mostrar.")
            return

        samples = random.sample(all_images, min(3, len(all_images)))
        plt.figure(figsize=(15, 5))
        
        for i, img_path in enumerate(samples):
            img = cv2.imread(str(img_path))
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, _ = img.shape
            
            label_path = lbl_dir / (img_path.stem + '.txt')
            if label_path.exists():
                with open(label_path, 'r') as f: lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    try:
                        cls_id = int(parts[0])
                        # --- CORRECCIÓN AQUÍ ---
                        # Cogemos todo como float
                        all_coords = list(map(float, parts[1:]))
                        # Solo nos quedamos con los 4 primeros (x,y,w,h)
                        # ignorando columnas extra si existen
                        if len(all_coords) >= 4:
                            x_c, y_c, bw, bh = all_coords[:4]
                        else:
                            continue # Línea corrupta
                        # -----------------------

                        x1, y1 = int((x_c-bw/2)*w), int((y_c-bh/2)*h)
                        x2, y2 = int((x_c+bw/2)*w), int((y_c+bh/2)*h)
                        
                        color = (0, 255, 0)
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        label_text = class_names_map.get(cls_id, str(cls_id))
                        cv2.putText(img, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    except ValueError: continue
            
            plt.subplot(1, 3, i+1)
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"{img_path.name}")
        plt.tight_layout()
        plt.show()

    def run(self, remove, remap, rename):
        print(f"🚀 INICIO DEL PROCESO")
        if not self.load_original_names(): return
        if not self.validate_conflicts(remove, remap): return
        if not self.copy_dataset(): return
        self.process_labels(remove, remap)
        final_names = self.update_yaml(remove, remap, rename)
        self.print_comparison_report(remove, remap, rename)
        if final_names: self.visualize_samples(final_names)
        print("\n✨ FINALIZADO ✨")

# --- ⚙️ CONFIGURACIÓN ⚙️ ---
if __name__ == "__main__":
    
    INPUT_FOLDER = r"C:\Users\Usuario\Desktop\Yolo\dataset\pv.v1i.yolov8"
    DESTINO_GENERAL = r"C:\Users\Usuario\Desktop\Yolo\dataset_modificado"

    # 1. BORRAR CLASES 1,2,3 (bird-drop, ground, etc.)
    BORRAR = [3]   
    
    # 2. MOVER CLASE 1 (ground original) A LA POSICIÓN 0
    CAMBIAR_IDS = {
        1: 2,
        2:1
    }

    # 3. RENOMBRAR LA CLASE 1 ORIGINAL (ground) A "Bird_dropping"
    RENOMBRAR_TEXTO = {
        0:"Bird_dropping",
        1: "Dust",
        2: "Mechanical_damage",
    }
    
    proc = YoloAutoProcessor(INPUT_FOLDER, DESTINO_GENERAL)
    proc.run(BORRAR, CAMBIAR_IDS, RENOMBRAR_TEXTO)