"""
Script 1: Descarga de Dataset de Armas
========================================
Fuente: Open Images v7 (Google) — CC BY 4.0
Clases: Knife, Firearm, Handgun, Shotgun

Uso:
    python scripts/1_download_dataset.py
"""

import os
import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_open_images(cfg: dict):
    """
    Descarga imágenes de Open Images v7 usando FiftyOne.
    Clases: Knife, Firearm, Handgun, Shotgun
    """
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError:
        logger.error("Instala fiftyone: pip install fiftyone")
        raise

    raw_dir   = Path(cfg["dataset"]["raw_dir"])
    classes   = cfg["dataset"]["open_images_classes"]
    max_samp  = cfg["dataset"]["max_samples_per_class"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "validation", "test"]

    for split in splits:
        logger.info(f"==> Descargando split: {split}  |  clases: {classes}")
        try:
            dataset = foz.load_zoo_dataset(
                "open-images-v7",
                split=split,
                label_types=["detections"],
                classes=classes,
                max_samples=max_samp,
                dataset_dir=str(raw_dir / split),
                dataset_name=f"weapons_{split}_{max_samp}",
            )
            logger.info(f"   Split '{split}': {len(dataset)} imágenes descargadas")
        except Exception as e:
            logger.warning(f"   No se pudo descargar split '{split}': {e}")
            continue

    logger.info("✅ Descarga completada. Revisa data/raw/")


def main():
    cfg = load_config()
    logger.info("=== Descarga de Dataset de Armas — Open Images v7 ===")
    logger.info(f"Clases objetivo: {cfg['dataset']['open_images_classes']}")
    logger.info(f"Máx. muestras por clase: {cfg['dataset']['max_samples_per_class']}")
    download_open_images(cfg)


if __name__ == "__main__":
    main()
