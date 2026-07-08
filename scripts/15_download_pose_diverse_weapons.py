"""
Script 15: Descarga datasets con poses diversas + hard negatives
==================================================================
Datasets HuggingFace validados:

YOLO format (con data.yaml):
  1. PranomVignesh/HandGuns                              (~8,968 imgs, 1 cls 'gun')
     Mirror de Roboflow pistols-p6x3w. Diversidad de poses real.
  2. Simuletic/Weapon_Detection_Dataset_Handgun_vs_BagOfChips (110 imgs)
     handgun vs bagofchips — hard negative explícito.
  3. Simuletic/CCTV_Weapon_Detection_Rifles_vs_Umbrellas (120 imgs)
     rifle vs paraguas CCTV — hard negative + ángulo cenital.

Parquet format (HuggingFace datasets, bbox COCO):
  4. Subh775/WeaponDetection                             (~9,657 imgs)
     Pistol/Rifle/Shotgun/Knife + Person/Hand (hard negs intrínsecos).
  5. harshdadiya-wappnet/phone_detection                 (~605 imgs)
     Sólo móviles → hard negatives puros (descartamos las cajas).

(Opcional) Roboflow Universe si ROBOFLOW_API_KEY está seteada.

Uso:
    python scripts/15_download_pose_diverse_weapons.py
"""

import logging
import os
import shutil
import yaml
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

REPO     = Path(__file__).resolve().parents[1]
HF_CACHE = REPO / "data" / "hf_pose"
RF_CACHE = REPO / "data" / "rf_pose"
HF_CACHE.mkdir(parents=True, exist_ok=True)
RF_CACHE.mkdir(parents=True, exist_ok=True)

CLS_IDX = {"knife": 0, "handgun": 1, "long_gun": 2}

# Mapeo común de nombres → nuestras 3 clases
GENERIC_NAME_MAP = {
    # handgun
    "gun": "handgun", "Gun": "handgun", "GUN": "handgun",
    "handgun": "handgun", "Handgun": "handgun",
    "pistol": "handgun", "Pistol": "handgun", "pistols": "handgun", "Pistols": "handgun",
    "Guns": "handgun", "guns": "handgun",
    "Guns perspective": "handgun",
    # long_gun
    "rifle": "long_gun", "Rifle": "long_gun",
    "shotgun": "long_gun", "Shotgun": "long_gun",
    "long_gun": "long_gun", "Long guns": "long_gun", "larga": "long_gun",
    "Heavy Gun": "long_gun", "heavyweapon": "long_gun",
    "weapon": "long_gun",  # depende del dataset, override por dataset
    # knife
    "knife": "knife", "Knife": "knife", "KNIFE": "knife",
    "Knife_Deploy": "knife", "Knife_Weapon": "knife",
}

# Clases que NO son armas (se descartan → imagen queda como background si era la única)
DROP_NAMES = {
    "person", "Person", "PERSON",
    "Hand", "hand",
    "Aggressor", "Victim", "Blood", "violence", "Stabbing",
    "bagofchips", "BagOfChips",
    "umbrella", "Umbrella",
    "phone", "mobile", "cellphone", "cell phone", "tablet", "smartphone",
    "al",
}

# ────────── HuggingFace YOLO-format ──────────
HF_YOLO_DATASETS = [
    {
        "repo_id":  "PranomVignesh/HandGuns",
        "prefix":   "hfprv_handguns_",
        "yaml_at":  "data.yaml",     # raíz
        "splits":   ["train", "valid", "test"],
        "name_map": {"gun": "handgun"},
        "tag":      "8.9K mirror RF pistols (varied poses)",
        "allow_bg": False,
    },
    {
        "repo_id":  "Simuletic/Weapon_Detection_Dataset_Handgun_vs_BagOfChips",
        "prefix":   "hfsim_handgun_chips_",
        "yaml_at":  "Handgun_vs_BagOfChips/data.yaml",
        "splits":   ["images"],    # un solo dir "images"
        "subdir":   "Handgun_vs_BagOfChips",
        "name_map": {"handgun": "handgun"},
        "tag":      "handgun vs chips (hard neg)",
        "allow_bg": True,
    },
    {
        "repo_id":  "Simuletic/CCTV_Weapon_Detection_Rifles_vs_Umbrellas",
        "prefix":   "hfsim_rifle_umbrella_",
        "yaml_at":  "Simuletic_Weapon_Umbrella_Dataset/data.yaml",
        "splits":   ["images"],
        "subdir":   "Simuletic_Weapon_Umbrella_Dataset",
        "name_map": {"weapon": "long_gun"},
        "tag":      "CCTV rifle vs umbrella (hard neg)",
        "allow_bg": True,
    },
]

# ────────── HuggingFace parquet-format (COCO bbox) ──────────
HF_PARQUET_DATASETS = [
    {
        "repo_id":  "Subh775/WeaponDetection",
        "prefix":   "hfsubh_weapdet_",
        "splits":   ["train", "validation"],   # test lo dejamos
        "name_map": GENERIC_NAME_MAP,
        "tag":      "9.6K weapon detection multicls",
        "allow_bg": False,    # las imágenes ya tienen armas; si sólo trae Person/Hand → background
    },
    {
        "repo_id":  "harshdadiya-wappnet/phone_detection",
        "prefix":   "hfharsh_phone_",
        "splits":   None,    # COCO format custom
        "name_map": {},    # vacío → todas las cajas se descartan → background
        "tag":      "phones (hard negatives)",
        "allow_bg": True,
        "force_background": True,
    },
]

# ────────── Roboflow Universe (opcional) ──────────
RF_DATASETS = [
    {
        "workspace": "joseph-nelson",
        "project":   "pistols",
        "version":   1,
        "prefix":    "rfpose_pistols_",
        "name_map":  {"pistol": "handgun"},
    },
]


def remap_yolo_labels(label_text, name_to_newcls, ds_class_names):
    out = []
    for line in label_text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            old_id = int(parts[0])
        except ValueError:
            continue
        if old_id >= len(ds_class_names):
            continue
        old_name = ds_class_names[old_id]
        if old_name in DROP_NAMES:
            continue
        new_class = name_to_newcls.get(old_name) or name_to_newcls.get(old_name.lower())
        if new_class is None:
            continue
        new_id = CLS_IDX[new_class]
        out.append(f"{new_id} {parts[1]} {parts[2]} {parts[3]} {parts[4]}")
    return out


def get_class_names(data_yaml_path):
    with open(data_yaml_path, encoding="utf-8") as f:
        ds_cfg = yaml.safe_load(f)
    names = ds_cfg.get("names", {})
    if isinstance(names, dict):
        return [names[i] for i in sorted(names.keys())]
    return list(names)


def integrate_yolo_dir(images_dir, labels_dir, ds_class_names, name_map,
                       prefix, img_out, lbl_out, allow_background=True):
    added, with_obj = 0, 0
    if not images_dir.exists() or not labels_dir.exists():
        return 0, 0
    for img in images_dir.iterdir():
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        lbl = labels_dir / (img.stem + ".txt")
        lbl_text = lbl.read_text(encoding="utf-8") if lbl.exists() else ""
        mapped = remap_yolo_labels(lbl_text, name_map, ds_class_names)
        if not mapped and not allow_background:
            continue
        added += 1
        if mapped:
            with_obj += 1
        dst_img = img_out / f"{prefix}{added:06d}{img.suffix.lower()}"
        dst_lbl = lbl_out / f"{prefix}{added:06d}.txt"
        shutil.copy2(img, dst_img)
        dst_lbl.write_text("\n".join(mapped), encoding="utf-8")
    return added, with_obj


def download_hf_yolo(img_out, lbl_out):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        log.error("pip install huggingface_hub")
        return 0, 0
    total_imgs, total_pos = 0, 0
    for ds in HF_YOLO_DATASETS:
        log.info(f"=== HF YOLO: {ds['repo_id']} [{ds['tag']}] ===")
        local_dir = HF_CACHE / ds["repo_id"].replace("/", "_")
        try:
            if not local_dir.exists() or not any(local_dir.iterdir()):
                snapshot_download(
                    repo_id=ds["repo_id"], repo_type="dataset",
                    local_dir=str(local_dir),
                )
            else:
                log.info(f"  cacheado en {local_dir}")
        except Exception as e:
            log.warning(f"  fallo HF: {e}")
            continue

        data_yaml = local_dir / ds["yaml_at"]
        if not data_yaml.exists():
            log.warning(f"  no existe {data_yaml}")
            continue
        ds_class_names = get_class_names(data_yaml)
        log.info(f"  clases ds: {ds_class_names} -> map {ds['name_map']}")

        subroot = local_dir / ds["subdir"] if "subdir" in ds else local_dir
        for split in ds["splits"]:
            # split puede ser "train", "valid", "test", o "images" (un solo dir)
            if split == "images":
                images_dir = subroot / "images"
                labels_dir = subroot / "labels"
            else:
                images_dir = subroot / split / "images"
                labels_dir = subroot / split / "labels"
            added, with_obj = integrate_yolo_dir(
                images_dir, labels_dir, ds_class_names, ds["name_map"],
                f"{ds['prefix']}{split}_", img_out, lbl_out,
                allow_background=ds.get("allow_bg", True),
            )
            if added:
                log.info(f"  {split}: {added} imgs ({with_obj} con armas, {added - with_obj} bg)")
            total_imgs += added
            total_pos  += with_obj
    return total_imgs, total_pos


def coco_bbox_to_yolo(bbox, img_w, img_h):
    """COCO [x_tl, y_tl, w, h] px → YOLO [cx, cy, w, h] normalizado."""
    x, y, w, h = bbox
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    return cx, cy, w / img_w, h / img_h


def download_hf_parquet(img_out, lbl_out):
    try:
        from datasets import load_dataset
    except ImportError:
        log.error("pip install datasets")
        return 0, 0
    total_imgs, total_pos = 0, 0
    for ds in HF_PARQUET_DATASETS:
        log.info(f"=== HF parquet: {ds['repo_id']} [{ds['tag']}] ===")
        try:
            data = load_dataset(ds["repo_id"])
        except Exception as e:
            log.warning(f"  fallo carga: {e}")
            continue

        # Determinar splits a usar
        splits = ds["splits"] if ds["splits"] else list(data.keys())

        # Categorías
        cat_feat = None
        for split in splits:
            if split not in data:
                continue
            try:
                cat_feat = data[split].features["objects"]["category"].feature.names
                break
            except Exception:
                pass
        if cat_feat is None and not ds.get("force_background"):
            log.warning(f"  no se pudo extraer categorías de {ds['repo_id']}")
            continue
        if cat_feat is None:
            cat_feat = []

        log.info(f"  cats dataset: {cat_feat[:20]}{'...' if len(cat_feat) > 20 else ''}")

        for split in splits:
            if split not in data:
                continue
            ds_split = data[split]
            n_added, n_pos = 0, 0
            for i, ex in enumerate(ds_split):
                try:
                    img = ex["image"]
                    img_w = ex.get("width", img.width)
                    img_h = ex.get("height", img.height)
                    bboxes = ex["objects"].get("bbox", [])
                    cats   = ex["objects"].get("category", [])
                except Exception:
                    continue

                yolo_lines = []
                if not ds.get("force_background"):
                    for bb, c in zip(bboxes, cats):
                        if c >= len(cat_feat):
                            continue
                        name = cat_feat[c]
                        if name in DROP_NAMES:
                            continue
                        newcls = ds["name_map"].get(name) or ds["name_map"].get(name.lower())
                        if newcls is None:
                            continue
                        new_id = CLS_IDX[newcls]
                        cx, cy, w, h = coco_bbox_to_yolo(bb, img_w, img_h)
                        if not (0 < w <= 1 and 0 < h <= 1):
                            continue
                        yolo_lines.append(f"{new_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

                if not yolo_lines and not ds.get("allow_bg", False):
                    continue
                n_added += 1
                if yolo_lines:
                    n_pos += 1
                # Guardar
                stem = f"{ds['prefix']}{split}_{i:06d}"
                dst_img = img_out / f"{stem}.jpg"
                dst_lbl = lbl_out / f"{stem}.txt"
                try:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    img.save(dst_img, "JPEG", quality=85)
                    dst_lbl.write_text("\n".join(yolo_lines), encoding="utf-8")
                except Exception:
                    continue
                if n_added % 500 == 0:
                    log.info(f"    {split}: {n_added} guardadas...")
            log.info(f"  {split}: {n_added} imgs ({n_pos} con armas, {n_added - n_pos} bg)")
            total_imgs += n_added
            total_pos  += n_pos
    return total_imgs, total_pos


def download_rf(img_out, lbl_out):
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        log.info("ROBOFLOW_API_KEY no seteada - saltando Roboflow")
        return 0, 0
    try:
        from roboflow import Roboflow
    except ImportError:
        log.error("pip install roboflow")
        return 0, 0
    rf = Roboflow(api_key=api_key)
    total_imgs, total_pos = 0, 0
    for ds in RF_DATASETS:
        log.info(f"=== RF: {ds['workspace']}/{ds['project']} ===")
        target_dir = RF_CACHE / f"{ds['workspace']}_{ds['project']}_v{ds['version']}"
        try:
            project = rf.workspace(ds["workspace"]).project(ds["project"])
            version = project.version(ds["version"])
            if not target_dir.exists():
                version.download("yolov8", location=str(target_dir))
        except Exception as e:
            log.warning(f"  fallo: {e}")
            continue
        data_yaml = target_dir / "data.yaml"
        if not data_yaml.exists():
            continue
        ds_class_names = get_class_names(data_yaml)
        for split in ["train", "valid", "test"]:
            added, with_obj = integrate_yolo_dir(
                target_dir / split / "images",
                target_dir / split / "labels",
                ds_class_names, ds["name_map"],
                f"{ds['prefix']}{split}_", img_out, lbl_out, allow_background=False
            )
            if added:
                log.info(f"  {split}: {added} ({with_obj} con armas)")
            total_imgs += added
            total_pos  += with_obj
    return total_imgs, total_pos


def main():
    processed = REPO / "data" / "processed"
    img_out = processed / "images" / "train"
    lbl_out = processed / "labels" / "train"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    yolo_imgs, yolo_pos = download_hf_yolo(img_out, lbl_out)
    pq_imgs,   pq_pos   = download_hf_parquet(img_out, lbl_out)
    rf_imgs,   rf_pos   = download_rf(img_out, lbl_out)

    total_i = yolo_imgs + pq_imgs + rf_imgs
    total_p = yolo_pos  + pq_pos  + rf_pos
    log.info("================ RESUMEN ================")
    log.info(f"  HF YOLO:    {yolo_imgs} imgs ({yolo_pos} con armas)")
    log.info(f"  HF parquet: {pq_imgs} imgs ({pq_pos} con armas)")
    log.info(f"  Roboflow:   {rf_imgs} imgs ({rf_pos} con armas)")
    log.info(f"  TOTAL:      {total_i} imgs ({total_p} con armas, {total_i - total_p} hard negatives)")


if __name__ == "__main__":
    main()
