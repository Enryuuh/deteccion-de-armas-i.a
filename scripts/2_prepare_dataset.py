"""
Script 2: Preparación del Dataset → Formato COCO JSON
=======================================================
Convierte las descargas de FiftyOne (Open Images v7) al
formato COCO estándar que acepta el entrenador RT-DETR.

Uso:
    python scripts/2_prepare_dataset.py
"""

import os
import json
import shutil
import logging
import yaml
from pathlib import Path
from datetime import datetime
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Mapa de clases Open Images → clases del proyecto
OI_TO_LABEL = {
    "Knife":    "knife",
    "Firearm":  "firearm",
    "Handgun":  "firearm",
    "Shotgun":  "firearm",
}


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_category_map(cfg: dict) -> dict:
    """Genera mapa nombre→id según config.yaml"""
    return {name: idx + 1 for idx, name in enumerate(cfg["classes"])}


def convert_fiftyone_to_coco(
    raw_split_dir: Path,
    output_json: Path,
    output_images_dir: Path,
    category_map: dict,
) -> dict:
    """
    Lee el directorio de FiftyOne y genera un JSON COCO válido.
    Retorna estadísticas del proceso.
    """
    try:
        import fiftyone as fo
    except ImportError:
        raise ImportError("Instala fiftyone: pip install fiftyone")

    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    coco = {
        "info": {
            "description": "Weapon Detection Dataset — Open Images v7",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "Google Open Images v7",
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [{"id": 1, "name": "CC BY 4.0", "url": "https://creativecommons.org/licenses/by/4.0/"}],
        "categories": [
            {"id": cat_id, "name": name, "supercategory": "weapon"}
            for name, cat_id in category_map.items()
        ],
        "images": [],
        "annotations": [],
    }

    ann_id   = 1
    img_id   = 1
    skipped  = 0
    total    = 0

    # Cargar dataset guardado de FiftyOne
    split_name = raw_split_dir.name  # "train", "validation", "test"
    try:
        dataset_name = f"weapons_{split_name}_2000"
        dataset = fo.load_dataset(dataset_name)
    except Exception:
        logger.warning(f"No se encontró dataset FiftyOne '{split_name}'. Saltando.")
        return {"images": 0, "annotations": 0, "skipped": 0}

    for sample in dataset:
        total += 1
        src_path = Path(sample.filepath)
        if not src_path.exists():
            skipped += 1
            continue

        # Copiar imagen
        dst_filename = f"{img_id:06d}{src_path.suffix}"
        dst_path = output_images_dir / dst_filename
        shutil.copy2(src_path, dst_path)

        try:
            with Image.open(dst_path) as img:
                w, h = img.size
        except Exception:
            skipped += 1
            dst_path.unlink(missing_ok=True)
            continue

        coco["images"].append({
            "id": img_id,
            "file_name": dst_filename,
            "width": w,
            "height": h,
            "license": 1,
        })

        # Procesar anotaciones
        if sample.ground_truth and sample.ground_truth.detections:
            for det in sample.ground_truth.detections:
                oi_label = det.label
                proj_label = OI_TO_LABEL.get(oi_label)
                if proj_label is None or proj_label not in category_map:
                    continue

                # Open Images bbox está en [x_rel, y_rel, w_rel, h_rel]
                bx, by, bw, bh = det.bounding_box
                abs_x = bx * w
                abs_y = by * h
                abs_w = bw * w
                abs_h = bh * h

                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": category_map[proj_label],
                    "bbox": [abs_x, abs_y, abs_w, abs_h],
                    "area": abs_w * abs_h,
                    "iscrowd": 0,
                })
                ann_id += 1

        img_id += 1

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)

    stats = {
        "images": img_id - 1,
        "annotations": ann_id - 1,
        "skipped": skipped,
    }
    return stats


def main():
    cfg = load_config()
    proc_dir  = Path(cfg["dataset"]["processed_dir"])
    raw_dir   = Path(cfg["dataset"]["raw_dir"])
    cat_map   = get_category_map(cfg)

    logger.info("=== Preparación de Dataset — Formato COCO ===")
    logger.info(f"Categorías: {cat_map}")

    split_map = {
        "train":      "train",
        "validation": "val",
        "test":       "test",
    }

    for oi_split, coco_split in split_map.items():
        logger.info(f"--> Procesando split: {oi_split} → {coco_split}")
        stats = convert_fiftyone_to_coco(
            raw_split_dir   = raw_dir / oi_split,
            output_json     = proc_dir / f"annotations_{coco_split}.json",
            output_images_dir = proc_dir / "images" / coco_split,
            category_map    = cat_map,
        )
        logger.info(f"    Imágenes: {stats['images']} | Anotaciones: {stats['annotations']} | Saltadas: {stats['skipped']}")

    logger.info("✅ Dataset preparado en data/processed/")
    logger.info("   Estructura:")
    logger.info("   data/processed/")
    logger.info("   ├── annotations_train.json")
    logger.info("   ├── annotations_val.json")
    logger.info("   ├── annotations_test.json")
    logger.info("   └── images/{train,val,test}/")


if __name__ == "__main__":
    main()
