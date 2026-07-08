"""
Script 13: Descarga datasets Roboflow y los integra al training set
====================================================================
Descarga datasets públicos de armas desde Roboflow Universe usando
la API key del usuario, los convierte a nuestras 3 clases YOLO
(knife=0, handgun=1, long_gun=2), y los agrega a data/processed/train.

Uso:
    set ROBOFLOW_API_KEY=tu_key
    python scripts/13_download_roboflow_weapons.py
"""

import logging
import os
import shutil
import yaml
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO     = Path(__file__).resolve().parents[1]
RF_CACHE = REPO / "data" / "rf_extra"
RF_CACHE.mkdir(parents=True, exist_ok=True)

# Datasets a descargar: (workspace, project, version, prefix, class_map)
# class_map: nombre original (en data.yaml del dataset) -> nuestra clase YOLO
DATASETS = [
    {
        "workspace":  "roboflow-100",
        "project":    "weapons-cope2",
        "version":    2,
        "prefix":     "rf_w100_",
        "class_map":  {
            "Knife": "knife", "knife": "knife",
            "Pistol": "handgun", "pistol": "handgun", "Handgun": "handgun", "handgun": "handgun",
            "Rifle": "long_gun", "rifle": "long_gun",
            "Shotgun": "long_gun", "shotgun": "long_gun",
            "Gun": "handgun", "gun": "handgun",
        },
    },
    {
        "workspace":  "joseph-nelson",
        "project":    "knives",
        "version":    1,
        "prefix":     "rf_jknife_",
        "class_map":  {"knife": "knife", "Knife": "knife"},
    },
]

CLS_IDX = {"knife": 0, "handgun": 1, "long_gun": 2}


def remap_label_lines(lines: list, dataset_yaml: dict, class_map: dict) -> list:
    """Convierte cls_id originales a nuestros cls_id usando el class_map."""
    names = dataset_yaml.get("names")
    if isinstance(names, list):
        old_id_to_name = {i: n for i, n in enumerate(names)}
    else:
        old_id_to_name = {int(k): v for k, v in names.items()}

    out = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            old_id = int(parts[0])
        except ValueError:
            continue
        old_name = old_id_to_name.get(old_id, "")
        new_class = class_map.get(old_name)
        if new_class is None:
            continue
        new_id = CLS_IDX[new_class]
        out.append(f"{new_id} {parts[1]} {parts[2]} {parts[3]} {parts[4]}")
    return out


def integrate_dataset(rf_dir: Path, prefix: str, class_map: dict, img_out: Path, lbl_out: Path) -> int:
    """Lee data.yaml del dataset descargado, lee splits train/valid/test y copia con remapeo."""
    data_yaml = rf_dir / "data.yaml"
    if not data_yaml.exists():
        logger.warning(f"  No data.yaml en {rf_dir}, saltando")
        return 0
    with open(data_yaml, encoding="utf-8") as f:
        ds_cfg = yaml.safe_load(f)

    added = 0
    for split in ["train", "valid", "test"]:
        split_imgs = rf_dir / split / "images"
        split_lbls = rf_dir / split / "labels"
        if not split_imgs.exists():
            continue
        for img in split_imgs.iterdir():
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            lbl = split_lbls / (img.stem + ".txt")
            if not lbl.exists():
                continue
            lines = lbl.read_text(encoding="utf-8").splitlines()
            mapped = remap_label_lines(lines, ds_cfg, class_map)
            if not mapped:
                continue
            added += 1
            dst_img = img_out / f"{prefix}{added:06d}{img.suffix.lower()}"
            dst_lbl = lbl_out / f"{prefix}{added:06d}.txt"
            shutil.copy2(img, dst_img)
            dst_lbl.write_text("\n".join(mapped), encoding="utf-8")
    return added


def main():
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("Falta ROBOFLOW_API_KEY en env")

    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)

    processed = REPO / "data" / "processed"
    img_out   = processed / "images" / "train"
    lbl_out   = processed / "labels" / "train"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    total_added = 0
    for ds in DATASETS:
        logger.info(f"=== {ds['workspace']}/{ds['project']} v{ds['version']} ===")
        target_dir = RF_CACHE / f"{ds['workspace']}_{ds['project']}_v{ds['version']}"
        try:
            project = rf.workspace(ds["workspace"]).project(ds["project"])
            version = project.version(ds["version"])
            if not target_dir.exists():
                version.download("yolov8", location=str(target_dir))
            else:
                logger.info(f"  Ya descargado en {target_dir}, reutilizando")
        except Exception as e:
            logger.warning(f"  Falló descarga: {e}")
            continue

        added = integrate_dataset(target_dir, ds["prefix"], ds["class_map"], img_out, lbl_out)
        logger.info(f"  Integradas {added} imgs al train")
        total_added += added

    logger.info(f"TOTAL agregado al dataset: {total_added} imágenes")


if __name__ == "__main__":
    main()
